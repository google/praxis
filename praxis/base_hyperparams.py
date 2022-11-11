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

"""Base hyperparams for Praxis configurable components."""

from __future__ import annotations

from flax.core import frozen_dict
import copy
import dataclasses
import enum
import inspect
import re
import types
import typing
from typing import Any, Callable, Optional, Sequence, Tuple, Type, TypeVar, Union

from absl import logging
import fiddle as fdl
# Internal config_dict import from ml_collections
import numpy as np
from praxis import pax_fiddle
from praxis import py_utils
import tensorflow.compat.v2 as tf

from google.protobuf import message
from google.protobuf import text_format

HParams = py_utils.HParams


def _quote_string(s):
  """Quotes a string with appropriate quotes and escaping.

  This performs lite escaping by choosing enclosing quotation marks that would
  escape the least (either single or double quotes) and escaping those quotes
  and the backslash. Note that this does not escape newlines. If the string
  contains embedded newlines, they will be output verbatim.

  Args:
    s: String to quote.

  Returns:
    Quotes string (possibly multiline).
  """
  single_quote_count = s.count('\'')
  double_quote_count = s.count('"')
  quote_delim = '\'' if single_quote_count <= double_quote_count else '"'
  # Apply escaping to the chosen quote character and the backslash.
  encoded = re.sub(r'([%s\\])' % quote_delim, r'\\\1', s)
  return quote_delim + encoded + quote_delim


def _is_named_tuple(x):
  """Returns whether an object is an instance of a collections.namedtuple.

  Examples::

    _is_named_tuple((42, 'hi')) ==> False
    Foo = collections.namedtuple('Foo', ['a', 'b'])
    _is_named_tuple(Foo(a=42, b='hi')) ==> True

  Args:
    x: The object to check.
  """
  return isinstance(x, tuple) and hasattr(x, '_fields')


class _SortedDict(dict):
  """A dict with a __repr__ that is always sorted by key."""

  def __repr__(self):
    return '{' + ', '.join(
        '%r: %r' % item for item in sorted(self.items())) + '}'


def _is_str_param_pairs(val):
  # Returns true if val is a list or a tuple of (str, HParams) pairs. For
  # example: [('one', HParams), ('two', HParams)].
  return (isinstance(val, (list, tuple)) and all(
      isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], str) and
      isinstance(x[1], HParams) for x in val))


def visit_nested_struct(obj_to_visit: Any,
                        visit_fn: Callable[[str, Any], None],
                        enter_fn: Optional[Callable[[str, Any], bool]] = None,
                        exit_fn: Optional[Callable[[str, Any], None]] = None):
  """Recursively visits objects within a nested pytree structure.

  Visit can traverse HParams, BaseHyperParams, lists, tuples, dataclasses,
  namedtuples, and Fiddle Buildables.

  By default, visit_fn is called on any object we don't know how to
  traverse into, like an integer or a string. enter_fn and exit_fn are
  called on objects we can traverse into, like HParams, lists, tuples,
  dataclasses, and namedtuples. We call enter_fn before traversing the object,
  and exit_fn when we are finished.

  If enter_fn returns false, that means we shouldn't traverse into the
  object; we call visit_fn on it instead and never call exit_fn.

  Keys are of the form::

    key.subkey when traversing HParams or Fiddle Buildable objects
    key[1] when traversing lists/tuples
    key[subkey] when traversing dataclasses or namedtuples

  Lists of (key, value) tuples are treated like namedtuples.

  Args:
    obj_to_visit: The nested structure to visit.
    visit_fn: Called on every object that can't be entered, or when enter_fn
      returns false.
    enter_fn: Called on every enter-able function. If this function returns
      false, we call visit_fn and do not enter the object.
    exit_fn: Called after an enter-able object has been traversed.
  """
  if not enter_fn:
    enter_fn = lambda key, val: True
  if not exit_fn:
    exit_fn = lambda key, val: None

  def _sub_key(key, subkey):
    if key:
      return f'{key}.{subkey}'
    return subkey

  def _visit(key: str, val: Any):
    if isinstance(val, HParams):
      if enter_fn(key, val):
        for k, v in val.IterParams():
          _visit(_sub_key(key, k), v)
        exit_fn(key, val)
      else:
        visit_fn(key, val)
    elif isinstance(val, (dict, frozen_dict.FrozenDict)):
      if enter_fn(key, val):
        for k, v in val.items():
          _visit(_sub_key(key, k), v)
        exit_fn(key, val)
      else:
        visit_fn(key, val)
    # Internal handle of type config_dict.ConfigDict in visit_nested_struct
    elif dataclasses.is_dataclass(val):
      if enter_fn(key, val):
        for f in dataclasses.fields(val):
          _visit(_sub_key(key, f.name), getattr(val, f.name))
        exit_fn(key, val)
      else:
        visit_fn(key, val)
    elif _is_named_tuple(val):
      items = val._asdict().items()
      if enter_fn(key, val):
        for k, v in items:
          _visit(f'{key}[{k}]', v)
        exit_fn(key, val)
      else:
        visit_fn(key, val)
    elif _is_str_param_pairs(val):
      if enter_fn(key, val):
        for subtuple in val:
          _visit(f'{key}[{subtuple[0]}]', subtuple[1])
        exit_fn(key, val)
    elif isinstance(val, list) or isinstance(val, range) or isinstance(
        val, tuple):
      if enter_fn(key, val):
        for i, v in enumerate(val):
          _visit(f'{key}[{i}]', v)
        exit_fn(key, val)
      else:
        visit_fn(key, val)
    elif isinstance(val, fdl.Buildable):
      if enter_fn(key, val):
        cls = fdl.get_callable(val)
        args = fdl.ordered_arguments(val, include_defaults=True)
        # Don't display the default _Sentinel value for linen parent layers.
        if 'parent' in args and type(args['parent']).__name__ == '_Sentinel':
          args.pop('parent')
        # Add cls (except for *_split_dims_mapping -- for compat w/ HParams).
        if not key.endswith('_split_dims_mapping'):
          _visit(f'{key}.cls', cls)
        for param_name, param_val in args.items():
          _visit(f'{key}.{param_name}', param_val)
        exit_fn(key, val)
      else:
        visit_fn(key, val)
    else:
      visit_fn(key, val)

  _visit('', obj_to_visit)


def nested_struct_to_text(obj_to_visit: Any,
                          include_types: bool = False,
                          separator: str = ':'):
  """Encodes nested structure into a simple text format.

  Each param is represented as a single line in the output.  The param
  name and value is separated by a ":".  The nest param name is
  separated by ".".  For values of non-trivial types (types other than
  int, float, bool, str, and a few, etc.), we just print out the name
  of its type.

  Note that strings are enclosed in appropriate single or double quotes
  (whichever would involve the least escaping) and will have some characters
  backslash escaped. String properties can span multiple lines.

  Args:
    obj_to_visit: The nested structure to visit.
    include_types: Should we return types of the values. If True, the types dict
      will be returned as a second val in a return tuple
    separator: Punctuation symbol used to separate param name and value.

  Returns:
    The encoded text or (encoded text, types dict) if include_types is True.
  """

  def get_repr(val: Any):
    """Get the representation of `val`."""
    if isinstance(val, HParams):
      return _SortedDict({k: get_repr(v) for k, v in val.IterParams()})
    if isinstance(val, (dict, frozen_dict.FrozenDict)):
      return _SortedDict({k: get_repr(v) for k, v in val.items()})
    if isinstance(val, np.ndarray):
      return np.array2string(val, separator=', ')
    if isinstance(val, BaseHyperParams):
      return _SortedDict({
          f.name: get_repr(getattr(val, f.name))
          for f in dataclasses.fields(val)
          if hasattr(val, f.name)
      })
    if dataclasses.is_dataclass(val):
      return str(val)
    if _is_named_tuple(val):
      return _SortedDict({k: get_repr(v) for k, v in val._asdict().items()})
    if isinstance(val, (list, tuple)):
      return type(val)([get_repr(v) for v in val])
    if isinstance(val, (int, float, bool, str, enum.Enum)):
      return val
    if isinstance(val, tf.DType):
      return val.name
    # TODO(b/227382805): Add better support for NumPy/JNP dtype?
    if isinstance(val, message.Message):
      proto_str = text_format.MessageToString(val, as_one_line=True)
      return 'proto/%s/%s/%s' % (inspect.getmodule(val).__name__,
                                 type(val).__name__, proto_str)
    if isinstance(val, type):
      return 'type/' + inspect.getmodule(val).__name__ + '/' + val.__name__
    if callable(val) and hasattr(val, '__qualname__'):
      return f'callable/{inspect.getmodule(val).__name__}/{val.__qualname__}'
    if isinstance(val, fdl.Buildable):
      return repr(val)
    return type(val).__name__

  def _enter(key: str, val: Any) -> bool:
    del key
    if isinstance(val, HParams):
      return True
    elif isinstance(val, BaseHyperParams):
      return True
    # Internal handle of type config_dict.ConfigDict in nested_struct_to_text
    elif (isinstance(val, (list, tuple)) and
          all(isinstance(x, HParams) for x in val)):
      return True
    elif (isinstance(val, (list, tuple)) and
          all(isinstance(x, BaseHyperParams) for x in val)):
      return True
    # Internal handle of config_dict.ConfigDict sequence in nested_struct_to_text
    # TODO(jiahuiyu): Create single-direction DebugString for
    # List[(str, HParams)] pattern and remove redundancies.
    elif _is_str_param_pairs(val):
      return True
    elif isinstance(val, (dict, frozen_dict.FrozenDict)):
      return True
    elif isinstance(val, fdl.Buildable):
      return True
    return False

  kv = {}
  visited_types = {}

  def _visit(key: str, p: Any) -> None:
    """Inserts key-value pairs to 'kv'."""
    if isinstance(p, str):
      kv[key] = _quote_string(p)
      visited_types[key] = 'str'
    else:
      kv[key] = str(get_repr(p))
      visited_types[key] = type(p).__name__

  visit_nested_struct(obj_to_visit, _visit, enter_fn=_enter)
  ret = ''
  for (k, v) in sorted(kv.items()):
    ret += k + f' {separator} ' + v + '\n'

  return (ret, visited_types) if include_types else ret


BaseHyperParamsSelf = TypeVar('BaseHyperParamsSelf', bound='BaseHyperParams')


def _get_sub_config(field: dataclasses.Field) -> Optional[fdl.Buildable]:
  """Returns a sub-configuration for a dataclass field."""
  if isinstance(field.default_factory, SubConfigFactory):
    return field.default_factory.get_class().config()
  else:
    return None


def _fill_sub_config_fields(
    cfg: pax_fiddle.Config,
    fields: Tuple[dataclasses.Field, ...],
) -> None:
  """Fills sub-config fields based on their dataclass annotations.

  A common pattern in Pax is to set fields with a `default_factory`, which
  refers to another sub-Params object. Instead of leaving these as unset
  attributes, we call their `.config` method. So if we have parameter classes,

  class FooParams(BaseHyperParams):
    a: int = 1

  class BarParams(BaseHyperParams):
    foo_params: BaseHyperParams = dataclasses.field(default_factory=FooParams)

  then BarParams.config() will return

  pax_fiddle.Config(BarParams, foo_params=pax_fiddle.Config(FooParams))

  Args:
    cfg: Fiddle buildable.
    fields: Dataclass fields. All of these should be valid attributes to set on
      `cfg`.
  """
  for field in fields:
    if field.name not in cfg.__arguments__:
      sub_config = _get_sub_config(field)
      if sub_config is not None:
        setattr(cfg, field.name, sub_config)


class BaseHyperParams:
  """Base class for all HyperParams."""
  # sub-class can explicitly list in the tuple the attribute to override.
  # An exception will be thrown if an attribute is overridden but it is not
  # explicitly listed in the tuple below. This is in order to avoid accidental
  # redefinition of attributes.
  _attribute_overrides: Tuple[str, ...] = ()

  if typing.TYPE_CHECKING:

    def __init__(self, *args, **kwargs):
      # this stub makes sure pytype accepts constructor arguments.
      pass

  @classmethod
  def config(cls: Type[BaseHyperParamsSelf],
             **kwargs: Any) -> pax_fiddle.Config[BaseHyperParamsSelf]:
    cfg = pax_fiddle.Config(cls, **kwargs)
    _fill_sub_config_fields(cfg, dataclasses.fields(cls))
    return cfg

  @classmethod
  def partial(cls: Type[BaseHyperParamsSelf],
              **kwargs: Any) -> fdl.Partial[BaseHyperParamsSelf]:
    return fdl.cast(fdl.Partial, cls.config(**kwargs))

  @classmethod
  def __init_subclass__(cls, **kwargs: Any) -> None:
    """Verifies properties about subclasses.

    Properties verified:
    - Subclasses may only re-declare an attribute if it is also listed in
      `_attribute_overrides` (which is always allowed to be redeclared).

    Args:
      **kwargs: Arguments passed to the corresponding base method.
    """
    super().__init_subclass__(**kwargs)

    attr_override_key = '_attribute_overrides'
    allowed_to_change = getattr(cls, attr_override_key)
    assert isinstance(allowed_to_change, tuple)

    annotations = dict(cls.__dict__.get('__annotations__', {}))
    # Remove attr_override_key from annotations.
    if attr_override_key in annotations:
      del annotations[attr_override_key]

    # In-place transformation to a data-class
    cls.__annotations__ = annotations
    # TODO(b/225403281): Investigate if we can set the dataclasses frozen
    # attribute here.
    dataclasses.dataclass(cls)
    # TODO(b/225403281): reorder so parent fields come after child fields.

    # Sanity check to make sure no field is accidentally overridden.
    parent_dataclass_fields = {}
    for clz in cls.__mro__[1:]:
      pdf = dict(getattr(clz, '__dataclass_fields__', {}))
      for key, value in pdf.items():
        if key not in parent_dataclass_fields:
          parent_dataclass_fields[key] = value

    for key, value in annotations.items():
      if key in parent_dataclass_fields and key not in allowed_to_change:
        raise AttributeError(f'Attribute {key} was overridden while it is not '
                             f'explicitly listed in the _attribute_overrides '
                             f'list.')

  def _check_assignment(self, field: dataclasses.Field[Any],
                        value: Any) -> None:
    """Checks that an assignment to a field is valid.

    Args:
      field: Dataclass field descriptor.
      value: Value that is being set.

    Raises:
      TypeError: If the field is the wrong type.
    """
    # TODO(edloper): Remove the special-case for pax_fiddle.Config once we've
    # moved to the next step of Fiddle migration.
    if (isinstance(value, fdl.Buildable) and
        not isinstance(value, pax_fiddle.Config)):
      logging.warning(
          'It is forbidden (almost always a mistake) to put Fiddle config '
          'objects inside dataclasses. Instead, create a pax_fiddle.Config of '
          'this Params class as well. For example, write '
          'pax_fiddle.Config(ParamsA, a=pax_fiddle.Config(ParamsB)), not '
          'ParamsA(a=pax_fiddle.Config(ParamsB)). However, using '
          'pax_fiddle.Config is ok for template fields.')

    if isinstance(value, BaseParameterizable):
      logging.warning(
          'You seem to be setting a BaseParameterizable to %s.%s; this '
          'is probably a mistake. Maybe you meant to create a '
          '`FooLayer.Params.config()` instead of a `FooLayer.config()`?',
          self.__class__.__qualname__, field.name)

    if (isinstance(value, py_utils.InstantiableParams) and
        field.type != py_utils.InstantiableParams):
      logging.warning(
          "It's not recommended to assign Lingvo params to "
          'sub-configuration of dataclass-based configuration. '
          '(Assigned %s.%s)', self.__class__.__qualname__, field.name)

  # Note: This will go away when we migrate more completely to fiddle and can
  # set frozen=True.
  def __setattr__(self, name: str, value: Any) -> None:
    if hasattr(self, '_internal_frozen') and self._internal_frozen:  # pytype: disable=attribute-error
      raise AttributeError(f'This HParam object is frozen: {self}')
    matching_fields = [
        field for field in dataclasses.fields(self) if field.name == name
    ]
    if not matching_fields:
      qualname = self.__class__.__qualname__
      raise AttributeError(f'Attribute {name} doesn\'t exist on {qualname}.')

    field, = matching_fields
    self._check_assignment(field, value)
    super().__setattr__(name, value)

  def __post_init__(self) -> None:
    # If _internal_frozen is True, this object is frozen.
    object.__setattr__(self, '_internal_frozen', False)
    for field in dataclasses.fields(self):
      name = field.name
      attr_value = getattr(self, name)
      # We make a copy, otherwise, it is possible that all instances points to
      # the class default.
      if isinstance(attr_value, BaseHyperParams):
        setattr(self, name, attr_value.clone())

      self._check_assignment(field, attr_value)

  def freeze(self):
    object.__setattr__(self, '_internal_frozen', True)
    fields = self.__dataclass_fields__  # pytype: disable=attribute-error
    for name in fields:
      attr_value = getattr(self, name)
      # recursively freeze all HyperParams
      if isinstance(attr_value, BaseHyperParams):
        attr_value.freeze()

  def unfreeze(self):
    object.__setattr__(self, '_internal_frozen', False)
    fields = self.__dataclass_fields__  # pytype: disable=attribute-error
    for name in fields:
      attr_value = getattr(self, name)
      # recursively unfreeze all HyperParams
      if isinstance(attr_value, BaseHyperParams):
        attr_value.unfreeze()

  def clone(self) -> 'BaseHyperParams':
    """Performs a deep copy of self and returns a mutable copy."""
    cloned = copy.deepcopy(self)
    # A cloned object is always unfrozen as the reason to make a clone if often
    # to make changes the resulting config.
    cloned.unfreeze()
    return cloned

  def copy_fields_from(
      self,
      source: 'BaseHyperParams',
      missing_fields_in_self: Optional[Sequence[str]] = None) -> None:
    """Copies fields from source.

    Args:
      source: HParams which will be copied.
      missing_fields_in_self: List of field names of source which are allowed to
        be missing in self.
    """
    if not isinstance(source, BaseHyperParams):
      raise TypeError(
          'Can only copy fields to BaseHyperParams from another '
          'BaseHyperParams.  (Copying from Fiddle Config not supported yet).')
    fields = self.__dataclass_fields__  # pytype: disable=attribute-error
    for name in fields:
      # Skip the field in self but not in source.
      if not hasattr(source, name) and hasattr(self, name):
        continue
      if not hasattr(source, name):
        raise ValueError(f'Copying incompatible HParams: {name} not in source')

      # cls is preserved in the destination HParams.
      if name == 'cls':
        continue
      attr_value = getattr(source, name)
      if isinstance(attr_value, BaseHyperParams):
        setattr(self, name, attr_value.clone())
      else:
        setattr(self, name, attr_value)

    for name in source.__dataclass_fields__:
      if not hasattr(self, name):
        if (missing_fields_in_self is None or
            name not in missing_fields_in_self):
          raise ValueError(f'Copying incompatible HParams: {name} not in self')

  def set(self, **kwargs) -> 'BaseHyperParams':
    """Sets dataclasses fields from a kwargs."""
    for key, value in kwargs.items():
      setattr(self, key, value)
    return self

  def to_text(self, include_types: bool = False, separator: str = ':'):
    return nested_struct_to_text(
        self, include_types=include_types, separator=separator)


class InstantiableHyperParams(BaseHyperParams):
  """Parameters from which an object can be instantiated.

  Attributes:
    cls: The class that this HyperParams is associated with.
    name: The name of the object instance that this param configs.
  """
  cls: Type[Any]
  name: str = ''

  def Instantiate(self, **kwargs) -> Any:  # pylint: disable=invalid-name
    assert self.cls is not None
    return self.cls(self, **kwargs)


@dataclasses.dataclass(frozen=True)
class SubConfigFactory:
  """Inspectable sub-configuration factory.

  This is meant to be plugged into `dataclasses.field(default_factory=factory)`.
  In general, please create instances of this through the `sub_config_field`
  method below.
  """
  _get_class: Callable[[], Type[BaseHyperParams]]

  def get_class(self) -> Type[BaseHyperParams]:
    """Returns the sub-configuration type, with some error checking."""
    subclass = self._get_class()
    if not issubclass(subclass, BaseHyperParams):
      raise AssertionError(
          "A SubConfigFactory wasn't configured with a subclass of "
          "BaseHyperParams. This shouldn't happen; if you happen to "
          'create a SubConfigFactory with a `lazy_ref=`, please make '
          'sure that it returns a subclass of BaseHyperParams.')
    return subclass

  def __call__(self):
    return self.get_class()()


class OverrideSubConfigFieldProtocol:
  """Protocol that can be used to override sub_config_field.

  If `x` implements `OverrideSubConfigFieldProtocol`, then `sub_config_field(x)`
  returns `x.__to_sub_config_field__().

  This allows us to override `sub_config_field(SomeLayer.HParams)` to return
  `pax_fiddle.template_field(SomeLayer)` when we migrate `SomeLayer` from a
  (hparams-configured) `BaseLayer` to a (fiddle-configured) `FiddleBaseLayer`.
  In particular, when we migrate `SomeLayer`, `SomeLayer.HParams` will become
  a stub object that implements this protocol.

  TODO(b/249483164): Remove this protocol (and the `sub_config_field` function)
  once all layers have been updated to use `pax_fiddle.template_field` or
  `pax_fiddle.sub_field`.
  """

  def __to_sub_config_field__(self):
    """Returns a dataclass field."""
    raise ValueError(f'Abstract method {type(self)}.__to_sub_config_field__')


def sub_config_field(
    sub_config_cls: Optional[Union[Type[BaseHyperParams],
                                   OverrideSubConfigFieldProtocol]] = None,
    *,
    lazy_ref: Optional[Callable[[], Type[BaseHyperParams]]] = None,
):
  """Returns an instance of the sub-config factory.

  Args:
    sub_config_cls: A reference to a sub-configuration class, e.g.
      `sub_config_cls=Linear.HParams`.  If `None`, then the sub-config defaults
      to `None`.
    lazy_ref: In the rare corner case that a class is not immediately known,
      provide `lazy_ref` as a lambda function which returns `sub_config_cls`.
  """
  if sub_config_cls is not None and lazy_ref is not None:
    raise TypeError('Specify sub_config_cls or lazy_ref, not both.')
  if sub_config_cls is None and lazy_ref is None:
    return pax_fiddle.template_field(None)  # Sets DO_NOT_BUILD tag.
  elif sub_config_cls is not None:
    if hasattr(sub_config_cls, '__to_sub_config_field__'):
      return sub_config_cls.__to_sub_config_field__()
    assert issubclass(sub_config_cls, BaseHyperParams), sub_config_cls
    lazy_ref = lambda: sub_config_cls
  return dataclasses.field(
      default_factory=SubConfigFactory(_get_class=lazy_ref))


BaseParameterizableSelf = TypeVar(
    'BaseParameterizableSelf', bound='BaseParameterizable')


class BaseParameterizable:
  """Base class for parameterizable entities such as tasks or optimizers.

  This class brings common functionalities for parameterizing core components
  such as inputs, learners, optimizers, schedules and tasks.
  """

  class HParams(InstantiableHyperParams):
    """Hyper-parameters for this parameterizable component."""

  # The argument name used in the initializer to pass the configured
  # cls.HParams instance.
  _config_init_name_for_params_object: str = 'hparams'

  # This allows child classes to configure which arguments to `__init__` should
  # not be considered configurable.
  _nonconfigurable_init_args: Tuple[str, ...] = ()

  def __init__(self, hparams: BaseParameterizable.HParams) -> None:
    """Constructor.

    Args:
      hparams: The dataclasses-like instance used to configure this class
        instance.
    """
    # TODO(b/228403686): Remove clone after Fiddle integration.
    self._hparams = hparams.clone()
    self._hparams.freeze()

  @property
  def hparams(self) -> BaseParameterizable.HParams:
    """Returns the hyper-parameters upon which this instance is built."""
    return self._hparams

  @classmethod
  def make(cls: Type[BaseParameterizableSelf],
           **kwargs) -> BaseParameterizableSelf:
    """Creates a `cls` using a `cls.HParams` created from `**kwargs` values.

    This function can be used directly, but is most often used as the callable
    used by Fiddle to configure and instantiate `cls`.

    Note: `__init_subclass__` adds a refined type annotation on every subclass's
    `make` method corresponding to the subclass's `Params` definition.

    Args:
      **kwargs: All values required to instantiate `cls.HParams` and `cls`.

    Returns:
      An instances of `cls`.
    """
    params_params = inspect.signature(cls.HParams).parameters
    missing_param_arg_names = set()
    params_kwargs = {}

    for param in params_params:
      if param in kwargs:
        params_kwargs[param] = kwargs.pop(param)
      elif params_params[param].default is not params_params[param].empty:
        params_kwargs[param] = params_params[param].default
      else:
        missing_param_arg_names.add(param)
    if missing_param_arg_names:
      raise ValueError(
          f'Missing required arguments: {missing_param_arg_names} when '
          f'instantiating {cls}.')
    params_instance = cls.HParams(**params_kwargs)
    kwargs[cls._config_init_name_for_params_object] = params_instance
    return cls(**kwargs)

  @classmethod
  def config(cls: Type[BaseParameterizableSelf],
             **kwargs) -> pax_fiddle.Config[BaseParameterizableSelf]:
    cfg = pax_fiddle.Config(cls.make, **kwargs)
    _fill_sub_config_fields(cfg, dataclasses.fields(cls.HParams))
    return cfg

  @classmethod
  def partial(cls: Type[BaseParameterizableSelf],
              **kwargs) -> fdl.Partial[BaseParameterizableSelf]:
    return fdl.cast(fdl.Partial, cls.config(**kwargs))

  @classmethod
  def __init_subclass__(cls,
                        init_params_arg_name: Optional[str] = None,
                        nonconfigurable_init_arg_names: Optional[Tuple[
                            str, ...]] = None,
                        **kwargs: Any) -> None:
    """Automatically initializes all subclasses as custom dataclasses."""
    super().__init_subclass__(**kwargs)
    _bind_cls_to_nested_params_class(cls)
    _add_precise_signature_to_make(cls, init_params_arg_name,
                                   nonconfigurable_init_arg_names)


def _bind_cls_to_nested_params_class(cls: Type[Any]):
  """Adds a reference from `cls.HParams` to `cls`.

  In order for cls.HParams.Initialize() to work, it needs a reference to cls.
  This function will dynamically create a new dataclass subclass of cls.HParams
  adding a new field "cls" with a default value of `cls`. This new subclass then
  takes over the symbol `cls.HParams`.

  Args:
    cls: The class to initialize.
  """
  # Update the HParams to dynamically bind a few fields.
  fields = [
      ('_attribute_overrides', Tuple[str, ...], ('cls',)),
      ('cls', Type[Any], cls),
  ]
  new_hparams = dataclasses.make_dataclass(
      cls.HParams.__name__, fields=fields, bases=(cls.HParams,))  # pytype: disable=wrong-arg-types
  new_hparams.__doc__ = cls.HParams.__doc__
  # Forward cls.__module__, not cls.HParams.__module__, in case cls doesn't have
  # its own HParams class and its parent's is defined in another module.
  new_hparams.__module__ = cls.__module__
  new_hparams.__qualname__ = f'{cls.__qualname__}.{new_hparams.__name__}'
  cls.HParams = new_hparams


def _add_precise_signature_to_make(
    cls: Type[BaseParameterizableSelf], init_params_arg_name: Optional[str],
    nonconfigurable_init_arg_names: Optional[Tuple[str, ...]]):
  """Gives `cls.make` a helpful type signature.

  `BaseParameterizable.make`'s signature is `**kwargs`. While this is
  supported by Fiddle, it disables eager typo-checking and interactive (colab)
  autocomplete.

  This function gives `cls.make` a more helpful type signature composed of the
  union of `cls.HParams`'s type annotations and `cls.__init__`'s arguments.

  Additionally, this function performs some integrity and consistency checks on
  `cls` to ensure it follows a few rules:

   - `cls.__init__` must contain a `cls._config_init_name_for_params_object`
     parameter.
   - The parameters for `cls.__init__` and `cls.Param.__init__` must not
     overlap.

  Args:
    cls: The class to configure.
    init_params_arg_name: The name used to pass the constructed `cls.HParams`
      instance to `cls.__init__`.
    nonconfigurable_init_arg_names: The names of arguments in `cls.__init__`
      that should not be configurable with a configuration system (e.g. Fiddle).

  Raises:
    TypeError: If expected type invariants are violated.
  """
  params_params = inspect.signature(cls.HParams).parameters
  cls_params = inspect.signature(cls).parameters

  # pylint: disable=protected-access
  if init_params_arg_name is not None:
    assert '__config_init_name_for_params_object' not in cls.__dict__
    cls._config_init_name_for_params_object = init_params_arg_name
  if nonconfigurable_init_arg_names is not None:
    assert '_nonconfigurable_init_args' not in cls.__dict__
    cls._nonconfigurable_init_args = (
        cls._nonconfigurable_init_args + tuple(nonconfigurable_init_arg_names))

  if cls._config_init_name_for_params_object not in cls_params:
    raise TypeError(f'{cls.__qualname__}.__init__ must have a parameter named '
                    f'{cls._config_init_name_for_params_object}; inferred '
                    f'signature: {cls_params} from function defined at: '
                    f'{cls.__init__.__code__.co_filename}:'
                    f'{cls.__init__.__code__.co_firstlineno}.')
  # pylint: enable=protected-access

  # Don't include the 'params' argument or any other hidden arguments.
  # Example use of extra filtered arguments include Flax's auto-added 'name' and
  # 'parent' arguments that should not be set by users.
  filter_arg_names = set(cls._nonconfigurable_init_args)  # pylint: disable=protected-access
  filter_arg_names.add(cls._config_init_name_for_params_object)  # pylint: disable=protected-access
  cls_params_filtered = [
      x for x in cls_params.values() if x.name not in filter_arg_names
  ]
  params_params_filtered = [
      x for x in params_params.values() if x.name != 'cls'
  ]

  overlapping_param_names = (
      set([x.name for x in params_params_filtered])
      & set([x.name for x in cls_params_filtered]))
  if overlapping_param_names:
    raise TypeError(
        f'Duplicated parameter between `{cls}` and `{cls}.HParams`: '
        f'{overlapping_param_names}.')

  # Make a fake signature compsed of every unfiltered parameter from both
  # cls.HParams.__init__ and cls.__init__ (but where every parameter is kw-only)
  manufactured_sig_args = [
      arg.replace(kind=arg.KEYWORD_ONLY)
      for arg in (params_params_filtered + cls_params_filtered)
  ]
  cls_param = inspect.Parameter(
      name='cls', kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)
  manufactured_sig = inspect.Signature(
      [cls_param] + manufactured_sig_args, return_annotation=cls)

  if 'make' not in cls.__dict__:
    # Make a copy of the underlying function to avoid mutating some other class.
    # We use this approach to ensure it's not a closure which makes it
    # pickle-able.
    make_fn = cls.make.__func__  # pytype: disable=attribute-error
    make_copy = types.FunctionType(make_fn.__code__, make_fn.__globals__,
                                   make_fn.__name__, make_fn.__defaults__,
                                   make_fn.__closure__)
    cls.make = classmethod(make_copy)

  cls.make.__func__.__signature__ = manufactured_sig  # pytype: disable=attribute-error

  # Make docs available to fiddleviz.
  # TODO(b/...): Glue the separate docstrings together somehow?
  cls.make.__func__.__doc__ = cls.__doc__  # pytype: disable=attribute-error


def instantiate(config: Union[InstantiableHyperParams, fdl.Buildable],
                **kwargs) -> Any:
  """Converts a config into an instance of the configured type.

  This function is an extra layer of indirection to facilitate migrations
  between configuration systems gracefully.

  Args:
    config: The configuration to instantiate.
    **kwargs: Additional kwargs to pass to instance constructor.

  Returns:
    An instance constructed from the provided configuration object `config`.
  """
  if isinstance(config, fdl.Buildable):
    return pax_fiddle.instantiate(config, **kwargs)
  if isinstance(config, InstantiableHyperParams):
    return config.Instantiate(**kwargs)
  raise ValueError(f'Unknown configuration type: {type(config)}. (Full config: '
                   f'{config})')
