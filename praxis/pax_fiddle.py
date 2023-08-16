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
import types
from typing import Any, Callable, Collection, Container, Dict, Generic, List, Mapping, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, overload
import weakref

import fiddle as fdl
from fiddle import building
from fiddle import daglish
from fiddle import history
from fiddle import signatures
from fiddle.experimental import auto_config as fdl_auto_config
import fiddle.extensions.jax
from flax.linen import module as flax_module
from lingvo.core import nested_map
import typing_extensions


# Import standard Fiddle APIs that we don't modify into this namespace.
# (So users can use e.g. `pax_fiddle.set_tags` instead of `fdl.set_tags`.)
add_tag = fdl.add_tag
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
remove_tag = fdl.remove_tag
set_tags = fdl.set_tags
materialize_defaults = fdl.materialize_defaults
set_tagged = fdl.set_tagged
Tag = fdl.Tag
TaggedValue = fdl.TaggedValue


TagOrTags = Union[type(fdl.Tag), Collection[type(fdl.Tag)]]
_T = TypeVar('_T')
TypeOrCallableProducingT = Union[Callable[..., _T], Type[_T]]

fiddle.extensions.jax.enable()
history.add_exclude_location('praxis/pax_fiddle.py')


_FIDDLE_DATACLASS_METADATA_KEY = object()


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


def field(
    *,
    default_factory: Any = dataclasses.MISSING,
    tags: TagOrTags = (),
    metadata: Optional[Mapping[Any, Any]] = None,
    configurable_factory: bool = False,
    **kwargs,
) -> Union[dataclasses.Field, Any]:  # pylint: disable=g-bare-generic
  """A wrapper around dataclasses.field to add optional Fiddle metadata.

  Args:
    default_factory: This has the same meaning as
      `dataclasses.fields.default_factory`, with the addition that if it's an
      `@auto_config`'d function, then the `as_buildable` will be used to
      initialize this field when creating a `fdl.Buildable` for the enclosing
      type.
    tags: One or more tags to attach to the `fdl.Buildable`'s argument
      corresponding to the field.
    metadata: Any additional metadata to include.
    configurable_factory: If true, then set this field to
      `Config(default_factory)` when creating a `fdl.Buildable` for the
      enclosing type.  For example, if `default_factory` is a dataclass, then
      this will make it possible to configure default values for the fields of
      that dataclass.  This should not be set to True if `default_factory` is an
      `auto_config`'ed function; see above for handling of `auto_config'ed
      `default_factory`.
    **kwargs: All other kwargs are passed to `dataclasses.field`; see the
      documentation on `dataclasses.field` for valid arguments.

  Returns:
    The result of calling dataclasses.field. Note: the return type is marked as
    a union with Any to pass static typechecking at usage sites.
  """
  # TODO(b/272374473): Make a function to return a metadata object to users to
  # enable them to call `dataclasses.field` themselves.
  if isinstance(tags, type(fdl.Tag)):
    tags = (tags,)

  if fdl_auto_config.is_auto_config(default_factory):
    if configurable_factory:
      raise ValueError(
          'configurable_factory should not be used with '
          "auto_config'ed functions."
      )
    buildable_initializer = default_factory.as_buildable
  elif configurable_factory:
    if not (default_factory and signatures.has_signature(default_factory)):
      raise ValueError(
          'configurable_factory requires that default_factory '
          'be set to a function or class with a signature.'
      )
    buildable_initializer = lambda: Config(default_factory)
  else:
    buildable_initializer = None

  metadata: Mapping[Any, Any] = types.MappingProxyType(metadata or {})
  metadata = {
      **metadata,
      _FIDDLE_DATACLASS_METADATA_KEY: FieldMetadata(
          tags=tags, buildable_initializer=buildable_initializer
      ),
  }
  return dataclasses.field(
      default_factory=default_factory, metadata=metadata, **kwargs
  )  # pytype: disable=wrong-keyword-args


# Temporary alias for backwards compatibility:
fdl_field = field


def field_has_tag(
    dc_field: dataclasses.Field,  # pylint: disable=g-bare-generic
    tag: type(fdl.Tag),
) -> bool:
  """Returns True if buildables will attach `tag` to the corresponding arg.

  In particular, `field_has_tag(field(..., tags=tags), tag)` is True if
  `tag in tags`.

  Args:
    dc_field: A dataclass field, describing an argument for a dataclass.
    tag: The tag that should be checked.
  """
  metadata = field_metadata(dc_field)
  return metadata is not None and tag in metadata.tags


def _add_dataclass_tags(buildable, fields):
  """Adds tags to arguments as indicated by dataclass fields.

  If any dataclass field in ``fields`` has metadata indicating that the field
  should be given one or more tags, then add those tags to the argument
  corresponding to the field.

  Args:
    buildable: The buildable that should be updated.
    fields: The dataclass fields for buildable.__fn_or_cls__.
  """
  for dc_field in fields:
    metadata = field_metadata(dc_field)
    if metadata:
      for tag in metadata.tags:
        add_tag(buildable, dc_field.name, tag)


def _expand_dataclass_default_factories(buildable, fields, arguments):
  """Expand default-valued args for dataclass fields with default-factories.

  If an argument has no value supplied when initializing a dataclass, but the
  corresponding field has a default factory, then that factory will be used to
  construct the argument's value. Thus, when creating a ``fdl.Buildable`` for
  the dataclass, it may be possible to fill in the value for the argument with
  ``Config(factory)``, without changing the value that will be built by
  ``buildable`` when calling ``build``.  This is useful because it makes the
  argument "deeply configurable" -- i.e., if the factory has any optional
  arguments, then this makes it possible to configure those objects. And in the
  special case where ``factory`` is an ``@auto_config``'d function, we can make
  the argument even more deeply configurable by inlining the factory.

  However, expanding default-valued args into `Buildable`s should only be
  performed when it can be done safely -- i.e., without changing the value
  that will be built by ``buildable``. In particular, we need to be careful
  not to create any "unintentional sharing," where the value built by the
  default factory is used by multiple instances of the dataclass.

  If we are not able to do the expansion safely, then we raise an exception.
  Note that it would be "safe" to leave the argument empty, in so far as the
  original semantics would be preserved.  But having the argument be
  unexpectedly unconfigurable could lead to difficult-to-diagnose issues.
  E.g., any nested dataclasses with `fdl.Tag`s associated with fields will
  not be accessible.

  One case where it *is* safe to expand default factories is when
  ``type(buildable)`` is ``Config``.  In that case, we know that a
  single dataclass object will be built from `buildable`, so we are guaranteed
  that the value built by the default factory will only be used by that one
  object.

  However, if ``type(buildable)`` is ``Partial``, then the function built
  from ``buildable`` can be used to generate multiple dataclass instances; and
  we need to ensure that the default factory is called for each instance.  For
  this case, we use ``ArgFactory(factory)`` rather than ``Config(factory)`` to
  expand the argument.  This ensures that the factory is called each time the
  partial is called.  We also need to replace any nested ``Config``'s with
  ``ArgFactory``'s, to ensure that the nested values are created each time as
  well.

  Similarly, if ``type(buildable) is ArgFactory``, then the factory function
  built from ``buildable`` can be used to generate multiple dataclass instances,
  so we use ``ArgFactory(factory)`` to expand arguments.

  In the case where ``type(buildable)`` is ``Partial`` or
  ``ArgFactory``, there is one additional corner case to consider, which
  occurs when multiple nested partials makes it impossible for Fiddle to
  describe the correct instance sharing pattern with its current ``Buildable``
  subclasses.  This corner case is demonstrated by the following example:

  ```
  def f(x):
    return x
  def g():
    return object()
  @auto_config.auto_config
  def make_fn():
    return functools.partial(f, x=g())
  @dataclasses.dataclass
  class A:
    fn: Callable[[], object] = field(default_factory=make_fn)
  p = functools.partial(A)
  ```

  Here, if we write ``a1 = p()`` to create an instance of ``A``, then calling
  ``a1.fn()`` multiple times will always return the same object, while another
  instance ``a2 = p()`` will return a different object when calling ``a2.fn()``:

  ```
  a1, a2 = p(), p()              # call the partial function twice.
  assert a1.fn() is a1.fn()      # a1.fn always returns the same object.
  assert a1.fn() is not a2.fn()  # a1 and a2 return different objects.
  ```

  However, if we construct ``Partial(A)``, and try to make ``f`` and ``g``
  deeply configurable, then there's no way to generate the same behavior
  using Fiddle ``Buildable``'s:

  * If we use ``Partial(A, Partial(f, Config(g)))``, then all
    instances of ``A`` generated by ``p`` will return the same instance
    (namely, the instance constructed by ``fdl.build(Config(g))``).
  * If we use ``Partial(A, Partial(f, ArgFactory(g)))``, then every call to
    ``A.fn`` will return a new object.

  Therefore, since is not possible to make the field ``A.fn`` deeply
  configurable while preserving the original semantics, we instead raise
  an exception.  If you believe you have a valid use-case for this, please
  contact the Pax-Fiddle team.

  The precise circumstances that cause this problem are: when we are building
  a ``Partial`` (or ``ArgFactory``), and the default factory expands into an
  expression containing a ``Partial`` (or ``ArgFactory``) that contains a
  ``Config`` -- in that case, the object built for the `Config` should be shared
  for each call to the inner partial; but should *not* be shared for each call
  to the outer partial.

  Args:
    buildable: The buildable that should be updated.
    fields: The dataclass fields for ``buildable.__fn_or_cls__``.
    arguments: The arguments that are being used to construct this
      ``Buildable``. If any argument has no value, and the corresponding field
      has a default factory, then the argument will be expanded into an
      equivalent ``Buildable`` if it's possible to do so without changing the
      semantics of ``fdl.build(buildable)``.
  """

  def convert_to_arg_factory(value, state):
    """Converts `cfg` and any nested `Config` objects to ArgFactory."""
    if not isinstance(value, PaxPartial):  # Don't recurse into partials.
      value = state.map_children(value)
    if isinstance(value, PaxConfig):
      value = cast(PaxArgFactory, value)
    return value

  def contains_partial_that_contains_config(value, state):
    """True if value contains a Partial/ArgFactory that contains a Config."""
    if isinstance(value, (PaxPartial, PaxArgFactory)):
      return any(isinstance(v, PaxConfig) for v, _ in daglish.iterate(value))
    elif state.is_traversable(value):
      return any(state.flattened_map_children(value).values)
    else:
      return False

  for dc_field in fields:
    if dc_field.name in arguments:
      continue  # We have an explicit value for this argument.
    metadata = field_metadata(dc_field)
    if not (metadata and metadata.buildable_initializer):
      continue
    field_config = metadata.buildable_initializer()
    if daglish.MemoizedTraversal.run(
        contains_partial_that_contains_config, field_config
    ):
      cls_name = getattr(
          buildable.__fn_or_cls__, '__qualname__', repr(buildable.__fn_or_cls__)
      )
      raise ValueError(
          f'Unable to safely replace {cls_name}.{dc_field.name} with '
          'a Pax ``Buildable` type, because its default factory contains a '
          '`Partial` that contains a `Config`.  This makes it difficult for '
          'Pax-Fiddle to describe the correct instance-sharing pattern. If you '
          'believe that you have a valid use-case for this, please contact the '
          'Pax-Fiddle team.'
      )
    if isinstance(field_config, PaxConfig) and isinstance(
        buildable, (PaxPartial, PaxArgFactory)
    ):
      field_config = daglish.MemoizedTraversal.run(
          convert_to_arg_factory, field_config
      )
    arguments[dc_field.name] = field_config


def _add_tags_and_defaults_from_dataclass_fields(self):
  if dataclasses.is_dataclass(self.__fn_or_cls__):
    fields = dataclasses.fields(self.__fn_or_cls__)
    _add_dataclass_tags(self, fields)
    arguments = self.__arguments__.copy()
    _expand_dataclass_default_factories(self, fields, arguments)
    fdl.assign(self, **arguments)


class PaxConfig(Generic[_T], fdl.Config[_T], CloneAndSetMixin):
  """Subclasses `fdl.Config` to add Pax-specific functionality."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    _add_tags_and_defaults_from_dataclass_fields(self)

  @property
  def cls(self):
    return fdl.get_callable(self)

  def __setattr__(self, name: str, value: Any):
    if name == 'cls':
      update_callable(self, value)
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
          raise ValueError(
              "Can't copy from default_factory "
              f'{source.__fn_or_cls__.__qualname__}.{name}'
          )

      else:
        raise ValueError(
            "Can't copy from missing required value "
            f'{source.__fn_or_cls__.__qualname__}.{name}'
        )


class PaxPartial(Generic[_T], fdl.Partial[_T], CloneAndSetMixin):
  """Subclasses `fdl.Partial` to add Pax-specific functionality."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    _add_tags_and_defaults_from_dataclass_fields(self)


class PaxArgFactory(Generic[_T], fdl.ArgFactory[_T], CloneAndSetMixin):
  """Subclasses `fdl.ArgFactory` to add Pax-specific functionality."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    _add_tags_and_defaults_from_dataclass_fields(self)


# Aliases to make them usable like regular Fiddle types.
Config = PaxConfig
Partial = PaxPartial
ArgFactory = PaxArgFactory


def instantiate(config: fdl.Buildable, **kwargs):
  """Builds `config` with optional argument overrides."""
  if kwargs:
    config = copy.copy(config)
    fdl.assign(config, **kwargs)
  return build(config)


# TODO(b/249483164): Remove this once all references have been deleted.
class DoNotBuild(fdl.Tag):
  """Deprecated -- do not use this tag."""


# TODO(b/285387519): Add kw_only=True when available.
@dataclasses.dataclass(frozen=True)
class FieldMetadata:
  """Fiddle-specific metadata that can be attached to each dataclasses.Field.

  Attributes:
    tags: A collection of tags to attach to the field.
    buildable_initializer: An optional callable to initialize the field's value
      when creating a `fdl.Buildable` of the enclosing type.
  """

  tags: Collection[type(fdl.Tag)]
  buildable_initializer: Optional[Callable[[], Any]]


def field_metadata(dc_field: dataclasses.Field) -> Optional[FieldMetadata]:  # pylint: disable=g-bare-generic
  """Retrieves the Fiddle-specific metadata (if present) on `field`."""
  return dc_field.metadata.get(_FIDDLE_DATACLASS_METADATA_KEY)


def update_callable(
    buildable: Buildable,
    new_callable: TypeOrCallableProducingT,
    drop_invalid_args: bool = False,
):
  fdl.update_callable(buildable, new_callable, drop_invalid_args)
  if dataclasses.is_dataclass(buildable.__fn_or_cls__):
    fields = dataclasses.fields(buildable.__fn_or_cls__)
    _add_dataclass_tags(buildable, fields)
    _expand_dataclass_default_factories(
        buildable, fields, buildable.__arguments__
    )


def auto_config(
    fn: Optional[Callable[..., Any]] = None,
    **auto_config_kwargs: Any,
) -> Any:
  """Version of Fiddle's auto_config that generates PaxConfig objects."""
  user_exemption_policy = auto_config_kwargs.pop(
      'experimental_exemption_policy', fdl_auto_config.auto_config_policy.latest
  )

  def exemption_policy(fn_or_cls):
    return (
        fn_or_cls is PaxConfig
        or (
            getattr(fn_or_cls, '__func__', None)
            is fdl_auto_config.AutoConfig.as_buildable
        )
        or user_exemption_policy(fn_or_cls)
    )

  auto_config_kwargs['experimental_exemption_policy'] = exemption_policy
  auto_config_kwargs['experimental_allow_control_flow'] = True
  auto_config_kwargs['experimental_config_types'] = fdl_auto_config.ConfigTypes(
      config_cls=PaxConfig,
      partial_cls=PaxPartial,
      arg_factory_cls=PaxArgFactory,
  )  # pytype: disable=wrong-arg-types
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
  dataclass, it should be initialized with `Config(default_factory)`.

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
    return field(default=None, tags=tags)

  # `factory` will return a PaxConfig object in the Fiddle.as_buildable path,
  # but will be `default_factory()` in the Python path.
  factory = auto_config(default_factory)
  return field(default_factory=factory, tags=tags)


def template_field(
    template: Optional[Callable[..., Any]] = dataclasses.MISSING,
    tags: TagOrTags = (),
) -> Union[dataclasses.Field[Any], Any]:
  """Dataclass field specification for a Fiddle-configurable template field.

  This can be used to specify that a dataclass should have a default value of
  `Config(template)`; and that when Fiddle builds the dataclass,
  this field should *not* be built, but should be left as a `Config`.

  Example usage:

  >>> class Parent(base_layer.BaseLayer):
  ...   child_tpl: Config[Child] = template_field(Child)

  Args:
    template: The template type (or factory function).  If `None`, then the
      field defaults to `None`. If not given, then the field is a required field
      that does not have a default.
    tags: One or more tags to attach to the `fdl.Buildable`'s argument
      corresponding to the field, when building a `fdl.Buildable`.

  Returns:
    A `dataclasses.Field` specification for the field.
  """
  if template is None or template is dataclasses.MISSING:
    return field(default=template, tags=tags)

  # `factory` will return a PaxConfig object in both the Fiddle.as_buildable
  # path and the Python path.
  factory = auto_config(template)
  factory = dataclasses.replace(factory, func=factory.as_buildable)
  return field(default_factory=factory, tags=tags)


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

  Replacing a template `tpl` with `Config(TemplateWrapper(tpl))` prevents
  `fdl.build` from building the template, making it possible to use the
  template for deferred subtree building.
  """

  template: fdl.Buildable

  def __call__(self):
    return self.template


def wrap_templates(buildable: Any) -> Any:
  """Returns copy of `buildable` with templates wrapped in `TemplateWrapper`s.

  In particular, any template `tpl` in `buildable` will be replaced by
  `Config(TemplateWrapper(tpl))`.  This ensures that the built value for
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
    new_arguments = {}
    for arg_name, arg_value in value.__arguments__.items():
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
          value, arguments, current_path=state.current_path
      )
    else:
      return state.map_children(value)

  result = daglish.MemoizedTraversal.run(traverse, buildable)
  return result


def _make_nested_maps_serializable():
  """Adds node traverser to make NestedMap serializable."""
  daglish.register_node_traverser(
      nested_map.NestedMap,
      flatten_fn=lambda x: (tuple(x.values()), tuple(x.keys())),
      unflatten_fn=lambda values, keys: nested_map.NestedMap(zip(keys, values)),
      path_elements_fn=lambda x: [daglish.Key(key) for key in x.keys()],
  )


_make_nested_maps_serializable()


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
# version -- e.g., the origin for `typing.Sequence[int]` might be
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
  # `Union[A, B]` and `A | B` rely on slightly distinct implementations.
  elif origin is Union or origin is types.UnionType:
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
    use_fallback=True
)


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
    daglish.iterate, registry=_hparams_node_traverser_registry
)
