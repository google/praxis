# coding=utf-8
# Copyright 2022 Google LLC.
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
from typing import overload, TypeVar, Callable, Any, Union, Optional, Collection, Generic

import fiddle as fdl
from fiddle import building
from fiddle import daglish
from fiddle import history
from fiddle.experimental import auto_config as fdl_auto_config
from fiddle.experimental import dataclasses as fdl_dataclasses
from fiddle.experimental.dataclasses import field as fdl_field
import fiddle.extensions.jax
from flax.linen import module as flax_module

fdl_field = fdl_dataclasses.field
TagOrTags = Union[type(fdl.Tag), Collection[type(fdl.Tag)]]
T = TypeVar('T')

fiddle.extensions.jax.enable()
history.add_exclude_location('praxis/pax_fiddle.py')


class CloneAndSetMixin:
  """Mixin used to add `clone` and `set` methods, for HParams compatibility."""

  def clone(self):
    return copy.deepcopy(self)

  def set(self, **kwargs):
    # Note: we don't use `fdl.assign` here because this mixin will be used
    # with classes that are not `fdl.Buildable` (e.g.,
    # `FiddleBaseLayer.ActivationSharding`).
    for name, value in kwargs.items():
      setattr(self, name, value)
    return self


class PaxConfig(Generic[T], fdl.Config[T], CloneAndSetMixin):
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

  def copy_fields_from(self, source: PaxConfig, missing_fields_in_self=()):
    """Copies fields from `source`.

    Corresponds with `BaseHyperparams.copy_fields_from`.

    Args:
      source: `PaxConfig` from which fields will be copied.
      missing_fields_in_self: List of field names in `source` which are allowed
        to be missing in `self`.
    """
    if not isinstance(source, PaxConfig):
      raise TypeError('Can only copy fields to PaxConfig from another '
                      'PaxConfig.  (Copying from HParams not supported yet).')

    # Deepcopy the source, so we don't introduce any unintentional sharing.
    source = source.clone()

    source_fields = {
        field.name: field
        for field in dataclasses.fields(source.__fn_or_cls__)
        if field.init and field.name not in ('parent', 'name')
    }
    self_fields = {
        field.name: field
        for field in dataclasses.fields(self.__fn_or_cls__)
        if field.init and field.name not in ('parent', 'name')
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


def has_do_not_build_tag(field: dataclasses.Field):  # pylint: disable=g-bare-generic
  return fdl_dataclasses.field_has_tag(field, DoNotBuild)


def _auto_config_exemption_policy(fn_or_cls):
  return (fn_or_cls is PaxConfig or
          fdl_auto_config.auto_config_policy.latest(fn_or_cls))


def auto_config(fn=None, **auto_config_kwargs) -> Any:
  """Version of Fiddle's auto_config that generates PaxConfig objects."""
  auto_config_kwargs['experimental_exemption_policy'] = (
      _auto_config_exemption_policy)
  auto_config_kwargs['experimental_allow_control_flow'] = True
  auto_config_kwargs['experimental_config_cls'] = PaxConfig

  def make_auto_config(fn):

    # If `pax_fiddle.auto_config` is applied to a class, then return the
    # result of applying it to the constructor.  This is helpful for the
    # automatic `auto_config` wrapping done by `sub_field` and `template_field`.
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


def sub_field(
    default_factory: Callable[..., Any],
    tags: Optional[TagOrTags] = tuple(),
) -> Union[dataclasses.Field, Any]:  # pylint: disable=g-bare-generic
  """Dataclass field specification for a Fiddle-configurable dataclass field.

  This can be used to specify that a dataclass should have a default value of
  `default_factory`; and that when Fiddle builds a `fdl.Buildable` for the
  dataclass, it should be initialized with `fdl.Config(default_factory)`.

  Example usage:

  >>> class Parent(base_layer.BaseLayer):
  ...   child: Child = pax_fiddle.sub_field(Child)

  Args:
    default_factory: The dataclass type used by the field.
    tags: One or more tags to attach to the `fdl.Buildable`'s argument
      corresponding to the field, when building a `fdl.Buildable`.

  Returns:
    A `dataclasses.Field` specification for the field.
  """
  # `factory` will return a PaxConfig object in the Fiddle.as_buildable path,
  # but will be `default_factory()` in the Python path.
  factory = auto_config(default_factory)
  return fdl_field(default_factory=factory, tags=tags)


def template_field(
    template: Optional[Callable[..., Any]],
    tags: Optional[TagOrTags] = tuple(),
) -> Union[dataclasses.Field, Any]:  # pylint: disable=g-bare-generic
  """Dataclass field specification for a Fiddle-configurable template field.

  This can be used to specify that a dataclass should have a default value of
  `fdl.Config(template)`; and that when Fiddle builds the dataclass,
  this field should *not* be built, but should be left as a `fdl.Config`.

  Example usage:

  >>> class Parent(base_layer.BaseLayer):
  ...   child_tpl: fdl.Config[Child] = pax_fiddle.template_field(Child)

  Args:
    template: The template type (or factory function).  If `None`, then the
      field defaults to `None`.
    tags: One or more tags to attach to the `fdl.Buildable`'s argument
      corresponding to the field, when building a `fdl.Buildable`.

  Returns:
    A `dataclasses.Field` specification for the field.
  """
  tags = set(tags) | {DoNotBuild}

  if template is None:
    return fdl_field(default=None, tags=tags)

  # `factory` will return a PaxConfig object in both the Fiddle.as_buildable
  # path and the Python path.
  factory = auto_config(template)
  factory = dataclasses.replace(factory, func=factory.as_buildable)
  return fdl_field(default_factory=factory, tags=tags)


# Typing overloads for pax_build
T = TypeVar('T')


@overload
def build(buildable: fdl.Partial[T]) -> Callable[..., T]:
  ...


@overload
def build(buildable: fdl.Partial) -> Callable[..., Any]:
  ...


@overload
def build(buildable: fdl.Config[T]) -> T:
  ...


@overload
def build(buildable: Any) -> Any:
  ...


def build(buildable):
  """Specialized version of `fdl.build` that respects the `DoNotBuild` tag.

  When building `buildable`, if any arguments are tagged with `DoNotBuild`,
  then return them as-is, rather than building them.  This makes it posible
  to keep templates unbuilt, so they can be used for deferred subtree building.

  Args:
    buildable: A `Buildable` instance to build, or a nested structure of
      `Buildable` objects.

  Returns:
    The built version of `buildable`.
  """

  def _build(value, state):
    if isinstance(value, fdl.Buildable):
      arguments = {}
      for key, sub_value in value.__arguments__.items():
        if DoNotBuild in value.__argument_tags__.get(key, ()):
          arguments[key] = sub_value
        else:
          # Clear the flax module stack, to avoid having `nn.Module`s be auto-
          # parented to the current module.  This is important for directly
          # instantiated *nested* descendents.
          with empty_flax_module_stack():
            arguments[key] = state.call(sub_value, daglish.Attr(key))
      return building.call_buildable(
          value, arguments, current_path=state.current_path)
    else:
      return state.map_children(value)

  return _build(buildable, daglish.MemoizedTraversal.begin(_build, buildable))


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
