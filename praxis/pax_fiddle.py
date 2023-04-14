# coding=utf-8
# Copyright 2022 The Pax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper functions and types related to Fiddle."""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import functools
from typing import Any, Callable, Collection, Container, Dict, Generic, List, Mapping, Optional, Sequence, Set, Tuple, TypeVar, Union, overload
import weakref

import fiddle as fdl
from fiddle import building
from fiddle import daglish
from fiddle import history
from fiddle import signatures
from fiddle.experimental import auto_config as fdl_auto_config
from fiddle.experimental import dataclasses as fdl_dataclasses
from fiddle.experimental.dataclasses import field as fdl_field
import fiddle.extensions.jax
from flax.linen import module as flax_module
import typing_extensions


# Import standard Fiddle APIs that we don't modify into this namespace.
# (So users can use e.g. `pax_fiddle.set_tags` instead of `fdl.set_tags`.)
add_tag = fdl.add_tag
ArgFactory = fdl.ArgFactory
assign = fdl.assign
Buildable = fdl.Buildable
cast = fdl.cast
clear_tags = fdl.clear_tags
copy_with = fdl.copy_with
deepcopy_with = fdl.deepcopy_with
get_callable = fdl.get_callable
get_tags = fdl.get_tags
NO_VALUE = fdl.NO_VALUE
ordered_arguments = fdl.ordered_arguments
Partial = fdl.Partial
remove_tag = fdl.remove_tag
set_tags = fdl.set_tags
update_callable = fdl.update_callable
materialize_defaults = fdl.materialize_defaults
set_tagged = fdl.set_tagged
Tag = fdl.Tag
TaggedValue = fdl.TaggedValue


fdl_field = fdl_dataclasses.field
TagOrTags = Union[type(fdl.Tag), Collection[type(fdl.Tag)]]
_T = TypeVar('_T')

fiddle.extensions.jax.enable()
history.add_exclude_location('praxis/pax_fiddle.py')


class CloneAndSetMixin:
  """Mixin used to add `clone` and `set` methods, for HParams compatibility."""

  def clone(self):
    return copy.deepcopy(self)

  def set(self, **kwargs):
    # Note: we don't use `fdl.assign` here because this mixin will be used
    # with classes that are not `fdl.Buildable` (e.g.,
    # `BaseLayer.ActivationSharding`).
    for name, value in kwargs.items():
      setattr(self, name, value)
    return self

  # TODO(b/269191093): Remove after updating existing users.
  Set = set  # pylint: disable=invalid-name


class PaxConfig(Generic[_T], fdl.Config[_T], CloneAndSetMixin):
  """Subclasses `fdl.Config` to make it more compatible with HParams."""

  @property
  def cls(self):
    return fdl.get_callable(self)

  def __setattr__(self, name: str, value: Any):
    if name == 'cls':
      fdl.update_callable(self, value)
    else:
      super().__setattr__(name, value)

  def Instantiate(self, **kwargs):
    """Builds `self` with optional argument overrides."""
    return instantiate(self, **kwargs)

  @property
  def mesh_shape(self):
    if self.ici_mesh_shape is not None:
      assert len(self.ici_mesh_shape) == len(self.mesh_axis_names)
    if self.dcn_mesh_shape is None:
      return self.ici_mesh_shape
    else:
      assert len(self.ici_mesh_shape) == len(self.dcn_mesh_shape)
      return [i * d for i, d in zip(self.ici_mesh_shape, self.dcn_mesh_shape)]

  def copy_fields_from(
      self,
      source: PaxConfig,
      missing_fields_in_self: Container[str] = (),
  ) -> None:
    """Copies fields from `source`.

    Corresponds with `BaseHyperparams.copy_fields_from`.

    Args:
      source: `PaxConfig` from which fields will be copied.
      missing_fields_in_self: List of field names in `source` which are allowed
        to be missing in `self`.
    """
    if not isinstance(source, PaxConfig):
      raise TypeError(
          'Can only copy fields to PaxConfig from another '
          f'PaxConfig, but got {type(source)}. '
          '(Copying from HParams not supported yet).'
      )

    # Deepcopy the source, so we don't introduce any unintentional sharing.
    source = source.clone()

    source_fields = {
        field.name: field
        for field in dataclasses.fields(source.__fn_or_cls__)
        if field.init and field.name != 'parent'
    }
    self_fields = {
        field.name: field
        for field in dataclasses.fields(self.__fn_or_cls__)
        if field.init and field.name != 'parent'
    }

    for name in source_fields:
      if name not in self_fields and name not in missing_fields_in_self:
        raise ValueError(f'Copying incompatible HParams: {name!r} not in self')

    for name, self_field in self_fields.items():
      source_field = source_fields.get(name, None)
      # Source doesn't have this field: skip it.
      if source_field is None:
        continue

      elif name in source.__arguments__:
        # Source has an explicit value for this field: copy it.
        setattr(self, name, getattr(source, name))

      elif source_field.default is not dataclasses.MISSING:
        # Source has a default value for this field: copy it.
        setattr(self, name, source_field.default)

      elif source_field.default_factory is not dataclasses.MISSING:
        if source_field.default_factory == self_field.default_factory:
          # Self and source have the same default factory for this field:
          # clear any value from self, so we'll use the default factory.
          self.__arguments__.pop(name, None)
        else:
          # Self and source have different default factories: we can't
          # handle this case.  (Calling the default factory here might
          # introduce unintentional sharing.)
          # TODO(edloper) Consider using fiddle.ArgFactory to handle this
          # case, if it turns out to be important to handle.
          raise ValueError("Can't copy from default_factory "
                           f'{source.__fn_or_cls__.__qualname__}.{name}')

      else:
        raise ValueError("Can't copy from missing required value "
                         f'{source.__fn_or_cls__.__qualname__}.{name}')


Config = PaxConfig  # Alias pax_fiddle.Config -> PaxConfig.


def instantiate(config: fdl.Buildable, **kwargs):
  """Builds `config` with optional argument overrides."""
  if kwargs:
    config = copy.copy(config)
    fdl.assign(config, **kwargs)
  return build(config)


class DoNotBuild(fdl.Tag):
  """Tag specifying that a value should not be built by `fdl.build`.

  This is used for template fields, which should contain `fdl.Buildable` objects
  even after they are built.
  """


def has_do_not_build_tag(field: dataclasses.Field[Any]) -> bool:
  return fdl_dataclasses.field_has_tag(field, DoNotBuild)


def auto_config(
    fn: Optional[Callable[..., Any]] = None,
    **auto_config_kwargs: Any,
) -> Any:
  """Version of Fiddle's auto_config that generates PaxConfig objects."""
  user_exemption_policy = auto_config_kwargs.pop(
      'experimental_exemption_policy',
      fdl_auto_config.auto_config_policy.latest)

  def exemption_policy(fn_or_cls):
    return (fn_or_cls is PaxConfig or
            (getattr(fn_or_cls, '__func__', None) is
             fdl_auto_config.AutoConfig.as_buildable) or
            user_exemption_policy(fn_or_cls))

  auto_config_kwargs['experimental_exemption_policy'] = exemption_policy
  auto_config_kwargs['experimental_allow_control_flow'] = True
  auto_config_kwargs['experimental_config_cls'] = PaxConfig
  auto_config_kwargs['experimental_result_must_contain_buildable'] = False

  def make_auto_config(fn):

    # If `pax_fiddle.auto_config` is applied to a class, then return the result
    # of applying it to the constructor.  This is helpful for the automatic
    # `auto_config` wrapping done by `instance_field` and `template_field`.
    if isinstance(fn, type):
      original_fn = fn

      # Note: We intentionally use a named function here rather than a lambda,
      # since auto_config can handle named functions more efficiently.
      def call_constructor():
        return original_fn()

      fn = call_constructor

    # If `fn` is already an auto-config function, then don't double-wrap it.
    if fdl_auto_config.is_auto_config(fn):
      return fn

    # Wrap `fn` using Fiddle auto_config.
    return fdl_auto_config.auto_config(fn, **auto_config_kwargs)

  return make_auto_config if fn is None else make_auto_config(fn)


def instance_field(
    default_factory: Optional[Callable[..., Any]],
    tags: TagOrTags = (),
) -> Union[dataclasses.Field[Any], Any]:
  """Dataclass field specification for a Fiddle-configurable dataclass field.

  This can be used to specify that a dataclass should have a default value of
  `default_factory`; and that when Fiddle builds a `fdl.Buildable` for the
  dataclass, it should be initialized with `fdl.Config(default_factory)`.

  Example usage:

  >>> class Parent(base_layer.BaseLayer):
  ...   child: Child = instance_field(Child)

  Args:
    default_factory: The dataclass type used by the field.  If `None`, then the
      field defaults to `None`.
    tags: One or more tags to attach to the `fdl.Buildable`'s argument
      corresponding to the field, when building a `fdl.Buildable`.

  Returns:
    A `dataclasses.Field` specification for the field.
  """
  if default_factory is None:
    return fdl_field(default=None, tags=tags)

  # `factory` will return a PaxConfig object in the Fiddle.as_buildable path,
  # but will be `default_factory()` in the Python path.
  factory = auto_config(default_factory)
  return fdl_field(default_factory=factory, tags=tags)


def template_field(
    template: Optional[Callable[..., Any]] = dataclasses.MISSING,
    tags: TagOrTags = (),
) -> Union[dataclasses.Field[Any], Any]:
  """Dataclass field specification for a Fiddle-configurable template field.

  This can be used to specify that a dataclass should have a default value of
  `fdl.Config(template)`; and that when Fiddle builds the dataclass,
  this field should *not* be built, but should be left as a `fdl.Config`.

  Example usage:

  >>> class Parent(base_layer.BaseLayer):
  ...   child_tpl: fdl.Config[Child] = template_field(Child)

  Args:
    template: The template type (or factory function).  If `None`, then the
      field defaults to `None`. If not given, then the field is a required field
      that does not have a default.
    tags: One or more tags to attach to the `fdl.Buildable`'s argument
      corresponding to the field, when building a `fdl.Buildable`.

  Returns:
    A `dataclasses.Field` specification for the field.
  """
  tags = ({tags} if isinstance(tags, type(Tag)) else set(tags)) | {DoNotBuild}

  if template is None or template is dataclasses.MISSING:
    return fdl_field(default=template, tags=tags)

  # `factory` will return a PaxConfig object in both the Fiddle.as_buildable
  # path and the Python path.
  factory = auto_config(template)
  factory = dataclasses.replace(factory, func=factory.as_buildable)
  return fdl_field(default_factory=factory, tags=tags)


# Typing overloads for pax_build


@overload
def build(buildable: fdl.Partial[_T]) -> Callable[..., _T]:
  ...


@overload
def build(buildable: fdl.Partial) -> Callable[..., Any]:
  ...


@overload
def build(buildable: fdl.Config[_T]) -> _T:
  ...


@overload
def build(buildable: Any) -> Any:
  ...


def build(buildable):
  """Specialized version of `fdl.build` with pax-specific behavior.

  1. Any template arguments (i.e., arguments whose annotated type is
     `pax_fiddle.Config` or a container of `pax_fiddle.Config`) are left as-is.
     This ensures that the templates can be used for deferred subtree building.

  2. BaseLayer "inheritable fields" are copied from parent objects to child
     objects, unless either:

     * The field has been overridden in the child object; or
     * The parent and child object are both templates.

  3. BaseLayers are built with an empty flax module stack.  This ensures that
     they don't get auto-parented to the wrong parent.

  Args:
    buildable: A `Buildable` instance to build, or a nested structure of
      `Buildable` objects.

  Returns:
    The built version of `buildable`.
  """
  buildable = wrap_templates(buildable)  # Do not build templates.
  buildable = copy_inheritable_base_layer_params(buildable)
  return build_with_empty_flax_module_stack(buildable)


@dataclasses.dataclass(frozen=True)
class TemplateWrapper:
  """Functor that returns a wrapped `Buildable` when called.

  I.e., `TemplateWrapper(tpl)` is equivalent to `lambda: tpl`, with the added
  benefit that we can use `isinstance` to identify template wrappers.

  Replacing a template `tpl` with `fdl.Config(TemplateWrapper(tpl))` prevents
  `fdl.build` from building the template, making it possible to use the
  template for deferred subtree building.
  """

  template: fdl.Buildable

  def __call__(self):
    return self.template


def wrap_templates(buildable: Any) -> Any:
  """Returns copy of `buildable` with templates wrapped in `TemplateWrapper`s.

  In particular, any template `tpl` in `buildable` will be replaced by
  `fdl.Config(TemplateWrapper(tpl))`.  This ensures that the built value for
  `tpl` will be `TemplateWrapper(tpl)()`, which returns `tpl` -- i.e., the
  template is not built.

  A `fdl.Buildable` is considered a template if it is the value for an argument
  whose type annotation indicates it should contain `Buildable` values (or
  containers of `Buildable` values).

  Only top-level templates are wrapped; i.e., templates nested inside other
  templates are *not* wrapped.

  Args:
    buildable: A `Buildable` instance to transform, or a nested structure of
      `Buildable` objects.
  """
  # As we traverse, keep track of whether we're inside a template field or not.
  in_template_field = False

  def traverse(value, state):
    nonlocal in_template_field

    # Do a normal depth-first traversal for everything but fdl.Buildable.
    if not isinstance(value, fdl.Buildable):
      return state.map_children(value)

    if in_template_field:
      return Config(TemplateWrapper(value))

    template_args = _get_template_arguments(value.__fn_or_cls__)
    fn_or_cls = value.__fn_or_cls__

    new_arguments = {}
    for arg_name, arg_value in value.__arguments__.items():
      if (
          arg_name not in template_args
          and DoNotBuild in value.__argument_tags__.get(arg_name, ())
      ):
        raise ValueError(
            f'{fn_or_cls}.{arg_name} was built using `template_field`, but its '
            'type annotation is not `Config` or `<container>[Config]`.\n'
            'Either change `template_field` to `instance_field`, or update the '
            'type annotation.'
        )
      in_template_field = arg_name in template_args
      new_arguments[arg_name] = state.call(arg_value, daglish.Attr(arg_name))
      in_template_field = False

    return fdl.copy_with(value, **new_arguments)

  return daglish.MemoizedTraversal.run(traverse, buildable)


# Cache dictionary for `_get_template_arguments`.
_template_argument_cache = weakref.WeakKeyDictionary()


def _get_template_arguments(fn_or_cls: Any) -> Set[str]:
  """Returns names of arguments whose type indicates they contain templates.

  I.e., return any arguments whose type is `Buildable` (or a subclass), or
  a container of `Buildable`.  The result may be cached.

  Args:
    fn_or_cls: The callable whose arguments should be checked.
  """
  try:
    return _template_argument_cache[fn_or_cls]
  except TypeError:  # Unhashable value.
    add_to_cache = False
  except KeyError:  # Not cached yet.
    add_to_cache = True
  signature = _get_template_arguments_uncached(fn_or_cls)
  if add_to_cache:
    _template_argument_cache[fn_or_cls] = signature
  return signature


def _get_template_arguments_uncached(fn_or_cls: Any) -> Set[str]:
  """Returns names of arguments whose type indicates they contain templates."""
  result = set()
  annotations = signatures.get_type_hints(fn_or_cls)

  for arg_name, annotation in annotations.items():
    context = f'{fn_or_cls}.{arg_name}'
    arg_is_template = _contains_buildable_type(annotation, context)
    if arg_is_template:
      result.add(arg_name)
      if not _is_supported_buildable_type(annotation):
        raise ValueError(
            f'Unsupported type {annotation!r} in BaseLayer field {context}:'
            ' types that contain pax_fiddle.Config may only use list,'
            ' tuple, dict, Mapping, Sequence, and Union.'
        )

  return result


def copy_inheritable_base_layer_params(buildable: Any) -> Any:
  """Copies inheritable BaseLayer params from parent layers to child layers.

  See `base_layer.BaseLayer.copy_base_hparams` for more information about
  inheritable BaseLayer parameters.

  Args:
    buildable: A `Buildable` instance to transform, or a nested structure of
      `Buildable` objects.

  Returns:
    A copy of `buildable` with inherited fields copied from parent to child.
  """
  # As we do a depth first traversal of `buildable`, keep track of ancestors
  # that have type `fdl.Buildable[BaseLayer]`.
  base_layer_config_ancestors = []

  def traverse(value, state):
    value_is_base_layer_config = False
    if isinstance(value, fdl.Buildable):
      # Copy inheritable BaseLayer fields from parent to child.
      fn_or_cls = value.__fn_or_cls__
      copy_base_hparams = getattr(fn_or_cls, 'copy_base_hparams', None)
      value_is_base_layer_config = copy_base_hparams is not None
      if value_is_base_layer_config and base_layer_config_ancestors:
        value = copy.copy(value)  # copy_base_hparams modifies `value` in place.
        copy_base_hparams(base_layer_config_ancestors[-1], value)

      # Leave wrapped templates as-is (after inheriting fields from parent).
      if isinstance(value, TemplateWrapper):
        return value

    if value_is_base_layer_config:
      base_layer_config_ancestors.append(value)
    result = state.map_children(value)
    if value_is_base_layer_config:
      base_layer_config_ancestors.pop()
    return result

  return daglish.MemoizedTraversal.run(traverse, buildable)


def build_with_empty_flax_module_stack(buildable: Any) -> Any:
  """Build `buildable`, ensuring the flax module stack is always empty.

  This avoids having `nn.Module`s be auto-parented to the current module,
  and is important for directly instantiated *nested* descendants.

  Args:
    buildable: A `Buildable` instance to transform, or a nested structure of
      `Buildable` objects.

  Returns:
    A value built from `buildable`.
  """
  # Clear the flax module stack, to avoid having `nn.Module`s be auto-
  # parented to the current module.  This is important for directly
  # instantiated *nested* descendants.

  def traverse(value, state):
    if isinstance(value, fdl.Buildable):
      arguments = {}
      for arg_name, arg_value in value.__arguments__.items():
        with empty_flax_module_stack():
          arguments[arg_name] = state.call(arg_value, daglish.Attr(arg_name))
      return building.call_buildable(
          value, arguments, current_path=state.current_path)
    else:
      return state.map_children(value)

  result = daglish.MemoizedTraversal.run(traverse, buildable)
  return result


def _is_buildable_type(typ):
  """Returns true if `typ` is a subclass of `Buildable` or `Buildable[...]`."""
  origin = typing_extensions.get_origin(typ)
  if origin is not None:
    typ = origin
  return isinstance(typ, type) and issubclass(typ, fdl.Buildable)


def _contains_buildable_type(typ, context):
  """Returns true if `typ` is a type annotation containing `fdl.Buildable`."""
  if isinstance(typ, str):
    raise TypeError(f'Unable to resolve type annotation {typ!r} for {context}.')
  if _is_buildable_type(typ):
    return True
  origin = typing_extensions.get_origin(typ)
  return origin is not typing_extensions.Literal and any(
      _contains_buildable_type(arg, context)
      for arg in typing_extensions.get_args(typ)
  )


# Origins for type annotations.  Note that these may depend on the Python
# version -- e.g., the orgin for `typing.Sequence[int]` might be
# `typing.Sequence` or `collections.abc.Sequence.`
_SEQUENCE_ORIGINS = {
    typing_extensions.get_origin(typ)
    for typ in [list, tuple, Sequence, List, Tuple]
}
_MAPPING_ORIGINS = {
    typing_extensions.get_origin(typ) for typ in (dict, Dict, Mapping)
}


def _is_supported_buildable_type(typ):
  """Returns true if `typ` can be used for a field containing PaxConfig.

  Fields that contain PaxConfig values (or `fdl.Buildable` values in general)
  are treated specially by `build` -- in particular, their values do not get
  built, but get left as-is.  To help ensure that this doesn't lead to
  unexpected resuts, we only support a limited set of type annotations for
  fields that contain PaxConfig values.  The most commonly used types are
  `PaxConfig[...]`, `Optional[PaxConfig]`, `Sequence[PaxConfig]`, and
  `Mapping[..., PaxConfig]`, but a few other complex types, such as
  `Union[PaxConfig, Sequence[PaxConfig]]` are also supported.

  Args:
    typ: The type annotation that should be checked.
  """
  if _is_buildable_type(typ):
    return True

  origin = typing_extensions.get_origin(typ)
  args = typing_extensions.get_args(typ)
  if origin in _SEQUENCE_ORIGINS:
    return all(
        _is_supported_buildable_type(arg) or arg is Ellipsis for arg in args
    )
  elif origin in _MAPPING_ORIGINS:
    return _is_supported_buildable_type(args[1])
  elif origin is Union:
    return all(_is_supported_buildable_type(arg) or arg is None for arg in args)
  elif origin is Optional:
    return all(_is_supported_buildable_type(arg) for arg in args)
  else:
    return False


@contextlib.contextmanager
def empty_flax_module_stack():
  """Context manager that temporarily clears the flax module stack."""
  module_stack = flax_module._context.module_stack  # pylint: disable=protected-access
  old_modules = list(module_stack)
  try:
    module_stack[:] = [None]  # Reset module stack.
    yield
  finally:
    module_stack[:] = old_modules  # Restore module stack.


_hparams_node_traverser_registry = daglish.NodeTraverserRegistry(
    use_fallback=True)


def _register_traversers_for_subclass(subclass):
  """Registers traversal routines for an HParams subclass."""
  fields = dataclasses.fields(subclass)
  names = tuple(field.name for field in fields)
  path_elements = tuple(daglish.Attr(field.name) for field in fields)

  def _flatten(value):
    return tuple(getattr(value, name) for name in names), ()

  def _unflatten(values, unused_metadata):
    return subclass(**dict(zip(names, values)))

  def _path_elements(unused_value):
    return list(path_elements)

  _hparams_node_traverser_registry.register_node_traverser(
      subclass,
      flatten_fn=_flatten,
      unflatten_fn=_unflatten,
      path_elements_fn=_path_elements,
  )


# APIs from fiddle.daglish, that are HParams-aware.
@dataclasses.dataclass
class BasicTraversal(daglish.BasicTraversal):
  registry: daglish.NodeTraverserRegistry = _hparams_node_traverser_registry


@dataclasses.dataclass
class MemoizedTraversal(daglish.MemoizedTraversal):
  registry: daglish.NodeTraverserRegistry = _hparams_node_traverser_registry


iterate = functools.partial(
    daglish.iterate, registry=_hparams_node_traverser_registry)
