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

"""Pooling layers."""

from typing import Sequence

import jax
from jax import numpy as jnp
from praxis import asserts
from praxis import base_layer
from praxis import py_utils
from praxis import pytypes

NestedMap = py_utils.NestedMap
JTensor = pytypes.JTensor


class Pooling(base_layer.BaseLayer):
  """Pooling layer, which by default performs max pooling.

  Attributes:
    window_shape: Window shape which determines the window sizes over which the
      pooling is computed. It is given as a Sequence of ints of size 2. Elements
      are in the order of height and width, and assumes inputs are in NHWC.
    window_stride: Window stride to use. Must be a pair of ints. The first int
      specifies the stride on the height dimension. The second int specifies the
      stride on the width dimension.
    pooling_type: Pooling type: MAX|AVG.
    padding: Padding type: SAME|VALID.
  """
  window_shape: Sequence[int] = (0, 0)
  window_stride: Sequence[int] = (0, 0)
  pooling_type: str = 'MAX'
  padding: str = 'SAME'

  def setup(self) -> None:
    if len(self.window_shape) != 2:
      raise ValueError('window_shape must be a sequence of length 2.')
    if len(self.window_stride) != 2:
      raise ValueError('window_stride must be a sequence of length 2.')
    if not all([w_shape > 0 for w_shape in self.window_shape]):
      raise ValueError('window_shape entries must be positive integers.')
    if not all([w_stride > 0 for w_stride in self.window_stride]):
      raise ValueError('window_stride entries must be positive integers.')
    if self.pooling_type not in ['MAX', 'AVG']:
      raise ValueError('pooling_type must be one of AVG or MAX.')
    if self.padding not in ['SAME', 'VALID']:
      raise ValueError('padding must be one of SAME or VALID.')

  def __call__(
      self,
      inputs: JTensor,
      paddings: JTensor | None = None,
  ) -> tuple[JTensor, JTensor | None]:
    """Applies pooling to inputs.

    Args:
      inputs: Input sequence of shape [B, H, W, D_in], also known more popularly
        as NHWC format.
      paddings: The paddings tensor. It is expected to be of shape [B, H].
        Defaults to None, which means there are no paddings.

    Returns:
      An (output, paddings) tensor tuple.
    Raises:
      ValueError: If the input dtype is not one of integer or floating point.
    """
    if jnp.issubdtype(inputs.dtype, jnp.inexact):
      dtype_min = -jnp.inf
    elif jnp.issubdtype(inputs.dtype, jnp.integer):
      dtype_min = jnp.iinfo(inputs.dtype).min
    else:
      raise ValueError('Unsupported dtype for inputs.')
    if self.pooling_type == 'MAX':
      init_value = dtype_min
      computation = jax.lax.max
      # If paddings are provided and pooling type is 'MAX', replace the pads
      # with negative infinity.
      if paddings is not None:
        # Fill negative infinity in padded positions.
        min_value = jnp.ones_like(inputs) * dtype_min
        compatible_paddings = paddings[..., jnp.newaxis, jnp.newaxis]
        inputs = jnp.where(compatible_paddings > 0, min_value, inputs)
    else:
      assert self.pooling_type == 'AVG'
      init_value = 0
      computation = jax.lax.add
    # The vars `window_shape` and `window_stride` are given only for [H, W].
    # Make it compatible with inputs of shape [N, H, W, C].
    window_shape = [1, self.window_shape[0], self.window_shape[1], 1]
    window_stride = [1, self.window_stride[0], self.window_stride[1], 1]
    out = jax.lax.reduce_window(
        inputs,
        init_value=init_value,
        computation=computation,
        window_dimensions=window_shape,
        window_strides=window_stride,
        padding=self.padding,
    )
    # If average pooling, rescale outputs by the window size.
    if self.pooling_type == 'AVG':
      ones = jnp.ones((inputs.shape[1], inputs.shape[2]), dtype=inputs.dtype)
      window_sizes = jax.lax.reduce_window(
          ones,
          init_value=0,
          computation=jax.lax.add,
          window_dimensions=self.window_shape,
          window_strides=self.window_stride,
          padding=self.padding,
      )
      out *= jnp.reciprocal(window_sizes[jnp.newaxis, ..., jnp.newaxis])
    if paddings is not None:
      if self.pooling_type == 'AVG':
        # Shape of paddings is [N, H]. Renormalize by count of non-padding items
        # in a window.
        non_padding_items = 1 - paddings
        non_padding_count = jax.lax.reduce_window(
            non_padding_items,
            init_value=0,
            computation=jax.lax.add,
            window_dimensions=(1, self.window_shape[0]),
            window_strides=(1, self.window_stride[0]),
            padding=self.padding,
        )
        non_pad_window_sizes = jax.lax.reduce_window(
            jnp.ones((inputs.shape[1]), dtype=inputs.dtype),
            init_value=0,
            computation=jax.lax.add,
            window_dimensions=(self.window_shape[0],),
            window_strides=(self.window_stride[0],),
            padding=self.padding,
        )
        # Do a safe division, where if denominator is 0, return 0.
        # This is because some `non_padding_window_sizes` may be 0, if an
        # entire window is full of PADs.
        non_padding_count = non_padding_count[..., jnp.newaxis, jnp.newaxis]
        out *= jnp.where(non_padding_count, jnp.reciprocal(non_padding_count),
                         0)
        out *= non_pad_window_sizes[jnp.newaxis, ..., jnp.newaxis, jnp.newaxis]
      # Compute the output paddings.
      if self.window_stride[0] > 1 or self.padding == 'VALID':
        # Output paddings are simply max-pooled since they are 0/1.
        paddings = jax.lax.reduce_window(
            paddings,
            init_value=dtype_min,
            computation=jax.lax.max,
            window_dimensions=(1, self.window_shape[0]),
            window_strides=(1, self.window_stride[0]),
            padding=self.padding,
        )
      # Apply the paddings back to the output.
      # Note that here we shouldn't multiply the output by (1 - paddings)
      # Since the output may contain - np.inf, and -np.inf * 0 = nan.
      non_padding_mask = (1 - paddings[..., jnp.newaxis, jnp.newaxis])
      out = jnp.where(non_padding_mask > 0, out, 0)
    return out, paddings


class GlobalPooling(base_layer.BaseLayer):
  """Performs a simple global pooling over the input with optional paddings.

  Raises:
    ValueError if `pooling_dims` is not a list or if any of their entries is
    negative.

  Attributes:
    pooling_type: Pooling type, can be MAX|AVG.
    pooling_dims: A list of dims to perform pooling over.
    keepdims: If True, keep dimension of inputs after pooling.
  """
  pooling_type: str = 'AVG'
  pooling_dims: Sequence[int] | None = None
  keepdims: bool = False

  def setup(self) -> None:
    if self.pooling_type not in ['MAX', 'AVG']:
      raise ValueError('pooling_type must be one of AVG or MAX.')
    if self.pooling_dims is None:
      raise ValueError('pooling_dims must be set as a list.')
    else:
      if not all([p_dims >= 0 for p_dims in self.pooling_dims]):
        raise ValueError('pooling_dims must be non-negative integers.')

  def __call__(
      self,
      inputs: JTensor,
      epsilon: float = 1e-8,
      compatible_paddings: JTensor | None = None,
  ) -> JTensor:
    """Applies global spatial pooling to inputs.

    Args:
      inputs: An input tensor.
      epsilon: small value to avoid divide by zero error.
      compatible_paddings: optional paddings of inputs with shapes compatible
        with inputs, e.g. compatible_paddings with shape [B, 1] for inputs with
        shape [B, D].

    Returns:
      Output tensor with global pooling applied.
    """

    if compatible_paddings is not None:
      if self.pooling_type == 'MAX':
        if jnp.issubdtype(inputs.dtype, jnp.inexact):
          padded_value = -jnp.inf
        elif jnp.issubdtype(inputs.dtype, jnp.integer):
          padded_value = jnp.iinfo(inputs.dtype).min
        else:
          raise ValueError('Unsupported dtype for inputs.')
      elif self.pooling_type == 'AVG':
        padded_value = jnp.zeros(shape=(), dtype=inputs.dtype)
      padded_value = jnp.ones_like(inputs) * padded_value
      inputs = jnp.where(compatible_paddings > 0, padded_value, inputs)

    if self.pooling_type == 'MAX':
      outputs = jnp.max(inputs, self.pooling_dims, keepdims=self.keepdims)
    elif self.pooling_type == 'AVG':
      if compatible_paddings is not None:
        valid_inputs = (
            jnp.sum(
                1.0 - compatible_paddings,
                self.pooling_dims,
                keepdims=True,
                dtype=inputs.dtype,
            )
            + epsilon
        )
        inputs_sum = jnp.sum(inputs, self.pooling_dims, keepdims=True)
        outputs = jnp.divide(inputs_sum, valid_inputs).astype(inputs.dtype)
        if not self.keepdims:
          outputs = jnp.squeeze(outputs, axis=self.pooling_dims)
      else:
        outputs = jnp.mean(
            inputs, self.pooling_dims, keepdims=self.keepdims
        ).astype(inputs.dtype)
    return outputs


class Pooling1D(base_layer.BaseLayer):
  """Pooling layer that operates over dimension 1 only; assume [B,T,...] inputs.

  Attributes:
    stride: int, to use every k-th token, set the stride to k.
    window: int, Size of the pooling window.
    pooling_type: Pooling type: MAX|AVG.
  """

  stride: int = 1
  window: int = 0
  pooling_type: str = 'AVG'

  def setup(self) -> None:
    if not self.stride > 0:
      raise ValueError('stride must be positive integer.')
    if self.pooling_type not in ['MAX', 'AVG']:
      raise ValueError('pooling_type must be one of AVG or MAX.')

  def __call__(
      self,
      inputs: JTensor,
      paddings: JTensor | None = None,
  ) -> tuple[JTensor, JTensor | None]:
    """Applies pooling to the inputs.

    Args:
      inputs: input tensor of shape [batch, time, dim], where the pooling is
        applied to the time dim.
      paddings: None or padding tensor of shape [batch, time]. If not None, the
        striding will be applied to the time dim.

    Returns:
      An (output, paddings) tensor tuple if paddings is not None, else just
      output, with the pooling/striding applied to the time dimension.
    """
    if jnp.issubdtype(inputs.dtype, jnp.inexact):
      dtype_min = -jnp.inf
    elif jnp.issubdtype(inputs.dtype, jnp.integer):
      dtype_min = jnp.iinfo(inputs.dtype).min
    else:
      raise ValueError('Unsupported dtype for inputs.')

    asserts.ge(self.stride, 1, msg=f'Invaild stride:{self.stride}.')
    if self.window == 0:
      self.window = self.stride

    if self.window == 1 and self.stride == 1:
      return inputs, paddings
    pooled_paddings = None if paddings is None else paddings[:, :: self.stride]
    if self.window == 1:  # stride > 1, strided reducer.
      return inputs[:, :: self.stride], pooled_paddings

    num_trailing_dims = len(inputs.shape) - 2
    trailing_axes = [-1] * num_trailing_dims
    if self.pooling_type == 'MAX':
      init_value = dtype_min
      computation = jax.lax.max
    else:
      assert self.pooling_type == 'AVG'
      init_value = 0
      computation = jax.lax.add
    if paddings is not None:
      compatible_paddings = jnp.expand_dims(paddings, axis=trailing_axes)
      inputs = jnp.where(compatible_paddings > 0, init_value, inputs)

    # seq2seq models should not cut paddings.
    padding = 'SAME'
    # The vars `window` and `stride` are only for [T]; make them compatible
    # with inputs with leading and trailing dimensions [B, T, ...]
    pooled_tensor = jax.lax.reduce_window(
        inputs,
        init_value=init_value,
        computation=computation,
        window_dimensions=[1, self.window] + [1] * num_trailing_dims,
        window_strides=[1, self.stride] + [1] * num_trailing_dims,
        padding=padding,
    )

    if self.pooling_type == 'AVG':
      # Divide by the number of non-padding elements inside each pooling window.
      valid = (
          jnp.ones(inputs.shape[:2], dtype=inputs.dtype)
          if paddings is None
          else 1.0 - paddings
      )
      valid_count = jax.lax.reduce_window(
          valid,
          init_value=0.0,
          computation=jax.lax.add,
          window_dimensions=[1, self.window],
          window_strides=[1, self.stride],
          padding=padding,
      )
      reciprocal = jnp.where(valid_count, jnp.reciprocal(valid_count), 0.0)
      pooled_tensor *= jnp.expand_dims(reciprocal, axis=trailing_axes)
    elif pooled_paddings is not None:
      # In the case of max pooling, set padded values to zero, otherwise padded
      # values will be -jnp.inf, which can cause numerical instability.
      pooled_tensor = jnp.where(
          jnp.expand_dims(1.0 - pooled_paddings, axis=trailing_axes),
          pooled_tensor,
          0.0,
      )

    return pooled_tensor, pooled_paddings
