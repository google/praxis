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

"""`Chain` is a utility for chaining layers.

In its most typical use `Chain` the output from the previous layer is given as
input to the next. But `Chain` extends and generalizes combination of layers by
treating `*args` as a stack; popping arguments and pushing outputs as it
iterates through the layers.
"""

import inspect
from typing import Any, Callable, Sequence

from praxis import base_layer


class Chain(base_layer.BaseLayer):
  """A utility layer for chaining a list of layers.

  `Chain` is similar to `Sequential` but extends and generalizes combination of
  layers by treating `*args` as a stack; popping required arguments and pushing
  outputs as it iterates through the layers.

  Specifically when the layers' signatures are different the leading arguments
  are replaced with the outputs from the previous layer:
    outputs = layer(args[:num_args])
    args = outputs + args[len(outputs):]  # replace leading arguments

  That is, `*args` is processed like a stack (leading arguments are poped and
  replaced). This stack logic is implemented in `call_chained_layer()`.

  `**kwargs` are passed straight-through the chain; unmodified. Further kwargs
  are only passed to layers whose `__call__()` specifies the existence of such
  an argument.

  Attributes:
    preserve_adopted_names: Flax module naming property (don't change).
    layers: The list of layers.
  """

  preserve_adopted_names: bool = True
  layers: Sequence[base_layer.BaseLayer] | None = None

  def __call__(self, *args: Any, **kwargs: Any) -> Any:
    """Calls layers one-by-one and returns the last chain output."""
    return self.call_chain(*args, **kwargs)[-1]

  def call_chain(self, *args: Any, **kwargs: Any) -> Sequence[Any]:
    """Calls layers one-by-one and returns a list with each layer's output."""
    return call_chain(self.layers, *args, **kwargs)


def call_chain(
    layers: Sequence[Callable[..., Any]], *args: Any, **kwargs: Any
) -> Sequence[Any]:
  """Calls layers one-by-one and returns a list with each layer's output.

  Wraps `call_chained_layer()` which implements the core logic of chain
  argument passing.

  Args:
    layers: The sequence of `BaseLayer`s.
    *args: The argument stack.
    **kwargs: The optional kwargs.

  Returns:
    A list with the output from each layer.
  """
  outputs = []
  args_stack = args
  for i, l in enumerate([l for l in layers if l is not None]):
    try:
      layer_outs = call_chained_layer(l, *args_stack, **kwargs)
    except Exception as e:
      raise type(e)(
          str(e) + f' Layer index={i} name={_name_attr(l)} args: {args_stack}'
      ) from e

    outputs.append(layer_outs)
    args_stack = ensure_tuple(layer_outs)

  if not outputs:
    return [args if len(args) > 1 else args[0]]

  return outputs


def call_chained_layer(
    layer: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
  """Passes required arguments and matching kwargs to `layer`.

  This is the main function implementing the argument stack; it passes
  required arguments to the layer and all matching kwargs. Other arguments are
  withheld.

  Args:
    layer: The layer to call.
    *args: The arguments to pass to required parameters.
    **kwargs: The kwargs to pass to existing kwargs.

  Returns:
    The new argument stack with leading arguments replaced:
      outs = layer(args[:num_args])
      outputs = outs + args[len(outs):]
  """
  signature = inspect.signature(layer.__call__)
  num_args = len(args)
  count = 0
  is_variadic = False
  for name, p in signature.parameters.items():
    if name in kwargs.keys():
      break
    if p.default != inspect.Signature.empty:
      break
    # Pass everything if variadic.
    if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
      is_variadic = True
      break
    count += 1
  if count > num_args:
    raise ValueError(
        f'Layer name={_name_attr(layer)} has too many args {count} > {num_args}'
        f' signature={signature}'
    )
  matching_kwargs = {
      k: v for k, v in kwargs.items() if k in signature.parameters.keys()
  }

  if is_variadic:
    outs = layer(*args, **kwargs)
  else:
    outs = layer(*args[:count], **matching_kwargs)

  outs = list(ensure_tuple(outs)) if outs is not None else []
  outputs = outs + list(args[len(outs) :])
  return tuple(outputs) if len(outputs) > 1 else outputs[0]


def ensure_tuple(x: Any) -> tuple[Any, ...]:
  """Ensures that `x` is a tuple."""
  return x if isinstance(x, tuple) else (x,)


def _name_attr(layer: Callable[..., Any]) -> str:
  """Returns the `name` attribute if it exists."""
  return layer.name if hasattr(layer, 'name') else ''
