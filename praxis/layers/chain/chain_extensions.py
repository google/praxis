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

"""Extra utilities for chain.py.

Provides utility parametrizations:
- chain()
- copy_n_times()
- apply_padding()
- add_residual()
- dict_to_args()
- log_args()
- full_like()
- scale()
- feed_forward()

See chain_test.py for examples.

Example,

import praxis
chain = praxis.layers.chain

    # A simple mlp stacking a few layers and using `copy_n_times()`.
    act = Config(activations.Swish)
    my_mlp = chain.chain(
        chain.feed_forward(input_dims, hidden_dims, act),
        chain.copy_n_times(
            n_times, se.feed_forward(hidden_dims, hidden_dims, act)
        ),
        chain.feed_forward(hidden_dims, output_dims),
    )
"""
import collections
import copy
from typing import Any, Dict, Optional, Sequence, Tuple, List

from absl import logging
from jax import numpy as jnp
from praxis import base_layer
from praxis import pax_fiddle
from praxis import py_utils
from praxis.layers import activations
from praxis.layers import linears
from praxis.layers import repeats
from praxis.layers.chain import chain as chain_lib

BaseLayer = base_layer.BaseLayer
Config = pax_fiddle.Config
LayerTpl = Config[base_layer.BaseLayer]

JTensor = base_layer.JTensor
Chain = chain_lib.Chain

# Wrappers for creating `Chain` with names assigned.


def chain(*layers_tpl: Any, **kwargs: Any) -> Config[Chain]:
  """Parameterizes Chain passing output from one layer to the next."""
  return Config(
      Chain,
      layers=_name_layers_uniquely(layers_tpl),
      **kwargs_with_name('chain', **kwargs),
  )


def copy_n_times(
    num_times: int, *layers_tpl: Any, **kwargs: Any
) -> Config[Chain]:
  """Parametrizes Chain to copy `layers_tpl` `num_times`.

  WARNING: Prefer using praxis.layers.Repeat if compatible with the usage.
  Repeat results in more efficient XLA compilation.

  Args:
    num_times: The number of times to copy the sequence of layers.
    *layers_tpl: The layers to run.
    **kwargs: Additional kwargs to pass to Chain.

  Returns:
    A Chain configuration.
  """
  return chain(
      *[copy.deepcopy(l) for l in layers_tpl * num_times],
      **kwargs_with_name('chain_copy_n_times', **kwargs),
  )


def repeat(
    x_times: int, *layers_tpl: Any, **kwargs: Any
) -> Config[repeats.Repeat]:
  """Parametrizes a Repeat of the layers `x_times`.

  Args:
    x_times: The number of times to copy the sequence of layers.
    *layers_tpl: The layers to repeat.
    **kwargs: Additional kwargs to pass to Chain.

  Returns:
    A repeats.Repeat with positional_args_as_scan_carry=True.
  """

  return Config(
      repeats.Repeat,
      x_times=x_times,
      sub_tpl=chain(*layers_tpl),
      positional_args_as_scan_carry=True,
      **kwargs_with_name('chain_repeat', **kwargs),
  )


def _name_layers_uniquely(layers_tpl: Sequence[LayerTpl]) -> List[LayerTpl]:
  """Returns layers with unique names assigned."""
  layers_tpl = [l for l in layers_tpl if l]
  if not layers_tpl:
    return []
  name_or_default = lambda layer: layer.name or 'layers'
  count = collections.defaultdict(int)
  for layer in layers_tpl:
    name = name_or_default(layer)
    count[name] += 1
  named_layers = []
  index = collections.defaultdict(int)
  for layer in layers_tpl:
    name = name_or_default(layer)
    suffix = f'{index[name]}' if count[name] > 1 else ''
    index[name] += 1
    named_layers.append(copy.deepcopy(layer).set(name=name + suffix))
  return named_layers


class ApplyPadding(BaseLayer):
  """Adapts py_utils.apply_padding() by default setting axis=0.

  This entails expanding/squeezing `paddings` to the right rank, rather than
  applying regular broadcast semantics.

  That is, `paddings` can be either [B, T] or [B, T, 1] for `inputs` [B, T, D].

  Attributes:
    axis: The argument to pass to py_utils.apply_padding() (default = 0).
  """

  axis: int = 0

  def __call__(self, inputs: JTensor, paddings: JTensor) -> JTensor:
    """Calls py_utils.apply_padding() with axis argument."""
    return py_utils.apply_padding(inputs, paddings, axis=self.axis)


def apply_padding(**kwargs: Any) -> Config[ApplyPadding]:
  """`Config(ApplyPadding)`; applies padding to pads the first argument."""
  return Config(ApplyPadding, **kwargs_with_name('pad', **kwargs))


class AddResidual(chain_lib.Chain):
  """Subclass of `Chain` that adds a residual to the first argument."""

  def __call__(self, *args: Any, **kwargs: Any) -> Any:
    """Calls super().__call__() and adds a residual to the first argument."""
    residual = args[0]
    outputs = list(chain_lib.ensure_tuple(super().__call__(*args, **kwargs)))
    new_outputs = [outputs[0] + residual] + outputs[1:]
    if len(new_outputs) > 1:
      return tuple(new_outputs)
    return new_outputs[0]


def add_residual(*layers: Any, **kwargs: Any) -> Config[AddResidual]:
  """`Config(AddResidual)`; stacks `layers` and adds a residual."""
  return Config(
      AddResidual,
      layers=_name_layers_uniquely(layers),
      **kwargs_with_name('add_residual', **kwargs),
  )


class DictToArgs(BaseLayer):
  """Looks up tensors in a dictionary.

  Attributes:
    keys: Keys to lookup.
  """

  keys: Sequence[str] = ()

  def __call__(self, input_batch: Dict[str, JTensor]) -> Tuple[JTensor, ...]:
    """Looks up tensors."""
    outputs = []
    for k in self.keys:
      if k not in input_batch:
        raise KeyError(f'key=`{k}` not in {input_batch.keys()}')
      outputs.append(input_batch[k])
    if len(outputs) > 1:
      return tuple(outputs)
    return outputs[0]  # pytype: disable=bad-return-type  # jax-ndarray


def dict_to_args(*keys: str, **kwargs: Any) -> Config[DictToArgs]:
  """`Config(DictToArgs)`; looks up tensors by keys in a dictionary."""
  return Config(
      DictToArgs,
      keys=keys,
      **kwargs_with_name('dict_to_args', **kwargs),
  )


class LogArgs(BaseLayer):
  """Logs the arguments (on CPU), useful for debugging.

  Attributes:
    message: The message to log.
    log_values: If tensor values should be logged or only the shape.
  """

  message: Optional[str] = None
  log_values: bool = False

  def __call__(self, *args: Any, **kwargs: Any) -> None:
    """Prints arguments."""
    logging.info(
        '---- %s len(args)=%d len(kwargs)=%d',
        self.message or '',
        len(args),
        len(kwargs),
    )
    for i, a in enumerate(args):
      logging.info('%d: %s', i, self._arg_str(a))
    for k, v in kwargs.items():
      logging.info('%s= %s', k, self._arg_str(v))
    logging.info('----')

  def _arg_str(self, arg: Any) -> str:
    if isinstance(arg, JTensor):
      return (
          f'{arg.shape} values=\n{arg}' if self.log_values else f'{arg.shape}'
      )
    return f'{arg}'


def log_args(
    message: Optional[str] = None, log_values: bool = False, **kwargs: Any
) -> Config[LogArgs]:
  """`Config(LogArgs)`; logs the arguments (for easy debugging)."""
  return Config(
      LogArgs,
      message=message or '',
      log_values=log_values,
      **kwargs_with_name('log_args', **kwargs),
  )


class FullLike(BaseLayer):
  """Simple wrapper around jnp.full_like().

  Attributes:
    fill_value: argument to jnp.full_like().
  """

  fill_value: float = 1.0

  def __call__(self, inputs: JTensor) -> JTensor:
    return jnp.full_like(inputs, self.fill_value)


def full_like(fill_value: float, **kwargs: Any) -> Config[FullLike]:
  """Wraps `jnp.full_like()`."""
  return Config(
      FullLike,
      fill_value=fill_value,
      **kwargs_with_name('full_like', **kwargs),
  )


def feed_forward(
    input_dims: int,
    output_dims: int,
    activation_tpl: Optional[LayerTpl] = None,
    **kwargs: Any,
) -> Config[linears.FeedForward]:
  """Wraps `Config(linears.FeedForward)`."""
  act = activation_tpl or Config(activations.Identity)
  return Config(
      linears.FeedForward,
      input_dims=input_dims,
      output_dims=output_dims,
      activation_tpl=act,
      **kwargs_with_name('feedforward', **kwargs),
  )


def kwargs_with_name(default_name: str, **kwargs: Any) -> Any:
  """Adds name to kwargs if it doesn't already exist."""
  if 'name' in kwargs.keys():
    return kwargs
  else:
    new_kwargs = kwargs.copy()
    new_kwargs['name'] = default_name
    return new_kwargs
