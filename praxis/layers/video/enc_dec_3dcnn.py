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

"""Encoder and Decoder structures with 3D CNNs."""

import math
from typing import Any, Callable, Sequence

import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import pax_fiddle
from praxis import pytypes
from praxis.layers import activations
from praxis.layers import convolutions
from praxis.layers import linears
from praxis.layers import normalizations


template_field = base_layer.template_field
LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]
JTensor = pytypes.JTensor


def scan_function(
    fn: Callable[..., Any], in_axis: int = 0
) -> Callable[..., Any]:
  """Scans function execution along an axis.

  Given a function with arguments of shape [..., n, ...], where n is the
  in_axis axis, this will return a function that will
  scan over that nth axis, calling the function n times.  This
  wraps the jax.lax.scan operation, with reshaping to maintain the call
  shapes of the input function.

  The function will be called with arguments of shape [..., 1, ...], and the
  results will be concatenated along the nth axis.  The dimension of
  the nth axis size does not need to be the same as the input (e.g. an input
  of [..., 1, ...] could produce and output of [..., 2, ...])  The number of
  dimensions of the output can also change, although currently the
  concatenated output axis is assumed to be same as the input axis.

  All arguments will be scanned along the nth axis.  Only arguments, not kwargs,
  are supported.

  Flax modules can not be scanned with this function.  They require the use of
  the Flax specific nn.scan call, not Jax's lax.scan.

  Args:
    fn: Function to scan.  Assumes data independence with respect to the in_axis
      axis.
    in_axis: Axis to scan over.

  Returns:
    Function with the same signature as fn, which internally scans over the
    in_axis axis.
  """

  def pipe_fn(*args):
    def transpose_and_expand(x):
      if x.ndim <= in_axis:
        raise ValueError(
            f'In_axis {in_axis} should be less than input rank {x.ndim}.'
        )
      if in_axis > 0:
        perm = np.concatenate(
            ([in_axis], np.arange(0, in_axis), np.arange(in_axis + 1, x.ndim))
        )
        x = jnp.transpose(x, perm)
      return jnp.expand_dims(x, axis=in_axis + 1)

    def transpose_and_reshape(x):
      if x.ndim <= in_axis:
        raise ValueError(
            f'In_axis {in_axis} should be less than input rank {x.ndim}.'
        )
      if in_axis > 0:
        perm = np.concatenate(
            (np.arange(1, in_axis + 1), [0], np.arange(in_axis + 1, x.ndim))
        )
        x = jnp.transpose(x, perm)
      s = x.shape
      x = jnp.reshape(
          x, (*s[0:in_axis], s[in_axis] * s[in_axis + 1], *s[in_axis + 2 :])
      )
      return x

    args = jax.tree_util.tree_map(transpose_and_expand, args)

    def fn_with_carry(carry, args):
      del carry
      return (), fn(*args)

    _, results = jax.lax.scan(fn_with_carry, (), args)

    results = jax.tree_util.tree_map(transpose_and_reshape, results)

    return results

  return pipe_fn


def depth_to_space(x, t_stride, filters):
  """depth_to_space."""

  def _depth_to_space(x):
    b, t, h, w, _ = x.shape
    x = x.reshape(b, t, h, w, t_stride, 2, 2, filters)
    x = x.transpose(0, 1, 4, 2, 5, 3, 6, 7)
    x = x.reshape(b, t * t_stride, h * 2, w * 2, filters)
    return x

  return scan_function(_depth_to_space, in_axis=1)(x)


class GroupNormSpatial(normalizations.GroupNorm):
  """GroupNorm with spatial dimensions ignored."""

  num_ignored_dims: int = 1
  epsilon: float = 1e-6
  input_rank: int = 5 - num_ignored_dims  # b, t, h, w, c

  def __call__(
      self, inputs: JTensor, paddings: JTensor | None = None
  ) -> JTensor:
    del paddings
    prefix_shape = inputs.shape[: 1 + self.num_ignored_dims]
    inputs = inputs.reshape(-1, *inputs.shape[2:])
    out = super().__call__(inputs)
    return out.reshape(*prefix_shape, *inputs.shape[1:])


class CondNormLayer(base_layer.BaseLayer):
  """Conditional normalization layer."""

  group_norm_tpl: LayerTpl = template_field(GroupNormSpatial)
  linear_tpl: LayerTpl = template_field(linears.Linear)
  bias_tpl: LayerTpl = template_field(linears.Bias)
  emb_dim: int = 0
  dim: int = 0

  def setup(self):
    group_norm_p = self.group_norm_tpl.clone()
    group_norm_p.name = 'group_norm'
    group_norm_p.dtype = self.dtype
    group_norm_p.fprop_dtype = self.fprop_dtype
    group_norm_p.dim = self.dim
    group_norm_p.use_scale = False
    group_norm_p.use_bias = False
    self.create_child('group_norm', group_norm_p)

    gamma_linear_p = self.linear_tpl.clone()
    gamma_linear_p.name = 'gamma_linear'
    gamma_linear_p.dtype = self.dtype
    gamma_linear_p.fprop_dtype = self.fprop_dtype
    gamma_linear_p.input_dims = self.emb_dim
    gamma_linear_p.output_dims = self.dim
    self.create_child('gamma_linear', gamma_linear_p)

    gamma_bias_p = self.bias_tpl.clone()
    gamma_bias_p.name = 'gamma_bias'
    gamma_bias_p.dtype = self.dtype
    gamma_bias_p.fprop_dtype = self.fprop_dtype
    gamma_bias_p.dims = self.dim
    self.create_child('gamma_bias', gamma_bias_p)

    beta_linear_p = self.linear_tpl.clone()
    beta_linear_p.name = 'beta_linear'
    beta_linear_p.dtype = self.dtype
    beta_linear_p.fprop_dtype = self.fprop_dtype
    beta_linear_p.input_dims = self.emb_dim
    beta_linear_p.output_dims = self.dim
    self.create_child('beta_linear', beta_linear_p)

    beta_bias_p = self.bias_tpl.clone()
    beta_bias_p.name = 'beta_bias'
    beta_bias_p.dtype = self.dtype
    beta_bias_p.fprop_dtype = self.fprop_dtype
    beta_bias_p.dims = self.dim
    self.create_child('beta_bias', beta_bias_p)

  def __call__(self, inputs: JTensor, emb: JTensor) -> JTensor:
    x = self.group_norm(inputs)
    gamma = self.gamma_linear(emb)
    gamma = self.gamma_bias(gamma)
    beta = self.beta_linear(emb)
    beta = self.beta_bias(beta)
    return x * (gamma + 1.0) + beta


class CausalConv(convolutions.Conv3D):
  """Causal 3D convolution."""

  padding: str = 'VALID'
  filter_stride: Sequence[int] = (1, 1, 1)
  bias: bool = True

  def __call__(self, inputs: JTensor) -> JTensor:
    pads = []
    for k in self.filter_shape[:3]:
      pads.append(((k - 1) // 2, k // 2))
    pads[0] = (self.filter_shape[0] - 1, 0)
    pads = [(0, 0)] + pads + [(0, 0)]
    inputs = jnp.pad(inputs, pads, mode='edge')
    return super().__call__(inputs)


class ResBlock(base_layer.BaseLayer):
  """Basic Residual Block."""

  input_dim: int = 0
  output_dim: int = 128
  use_conv_shortcut: bool = False
  norm_layer_tpl: LayerTpl = template_field(GroupNormSpatial)
  conv_layer_tpl: LayerTpl = template_field(CausalConv)
  activation_tpl: pax_fiddle.Config[activations.BaseActivation] = (
      template_field(activations.Swish)
  )

  def setup(self):
    norm_p = self.norm_layer_tpl.clone()
    norm_p.name = 'norm'
    norm_p.dtype = self.dtype
    norm_p.fprop_dtype = self.fprop_dtype
    norm_p.dim = self.input_dim
    self.create_child('norm_0', norm_p)
    norm_p = norm_p.clone()
    norm_p.name = 'norm_1'
    norm_p.dim = self.output_dim
    self.create_child('norm_1', norm_p)

    conv_p = self.conv_layer_tpl.clone()
    conv_p.name = 'conv0'
    conv_p.dtype = self.dtype
    conv_p.fprop_dtype = self.fprop_dtype
    conv_p.bias = False
    conv_p.filter_shape = (3, 3, 3, self.input_dim, self.output_dim)
    self.create_child('conv_0', conv_p)

    conv_p = conv_p.clone()
    conv_p.name = 'conv1'
    conv_p.filter_shape = (3, 3, 3, self.output_dim, self.output_dim)
    self.create_child('conv_1', conv_p)

    if self.input_dim != self.output_dim:
      conv_p = self.conv_layer_tpl.clone()
      conv_p.name = 'conv_shortcut'
      conv_p.dtype = self.dtype
      conv_p.fprop_dtype = self.fprop_dtype
      conv_p.bias = False
      if self.use_conv_shortcut:
        conv_p.filter_shape = (3, 3, 3, self.input_dim, self.output_dim)
      else:
        conv_p.filter_shape = (1, 1, 1, self.input_dim, self.output_dim)
      self.create_child('conv_shortcut', conv_p)

    self.create_child('activation', self.activation_tpl.clone())

  def __call__(self, inputs: JTensor) -> JTensor:
    if self.input_dim != inputs.shape[-1]:
      raise ValueError(
          f'Input dimension {inputs.shape[-1]} does not match {self.input_dim}'
      )
    residual = inputs
    x = self.norm_0(inputs)
    x = self.activation(x)
    x = self.conv_0(x)
    x = self.norm_1(x)
    x = self.activation(x)
    x = self.conv_1(x)
    if self.input_dim != self.output_dim:
      residual = self.conv_shortcut(residual)
    return x + residual


def _conv_dimension_numbers(input_shape):
  """Computes the dimension numbers based on the input shape."""
  ndim = len(input_shape)
  lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
  rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
  out_spec = lhs_spec
  return jax.lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


class BlurPool3D(base_layer.BaseLayer):
  """A layer to do channel-wise blurring + subsampling on 3D inputs.

  Reference:
    Zhang et al. Making Convolutional Networks Shift-Invariant Again.
    https://arxiv.org/pdf/1904.11486.pdf.
  """

  filter_size: int = 3
  strides: tuple[int, int, int] = (2, 2, 2)
  padding: str = 'SAME'

  def setup(self):
    if self.filter_size == 2:
      self.filter = [1.0, 1.0]
    elif self.filter_size == 3:
      self.filter = [1.0, 2.0, 1.0]
    elif self.filter_size == 4:
      self.filter = [1.0, 3.0, 3.0, 1.0]
    elif self.filter_size == 5:
      self.filter = [1.0, 4.0, 6.0, 4.0, 1.0]
    elif self.filter_size == 6:
      self.filter = [1.0, 5.0, 10.0, 10.0, 5.0, 1.0]
    elif self.filter_size == 7:
      self.filter = [1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0]
    else:
      raise ValueError('Only filter_size of 2, 3, 4, 5, 6 or 7 is supported.')

    self.filter = jnp.array(self.filter, dtype=self.dtype)
    self.filter = jnp.einsum(
        'i,j,k->ijk', self.filter, self.filter, self.filter
    )
    precision = 'bfloat16' if self.dtype == jnp.bfloat16 else 'float32'
    with jax.default_matmul_precision(precision):
      self.filter /= jnp.sum(self.filter)
    self.filter = jnp.reshape(
        self.filter,
        [
            self.filter.shape[0],
            self.filter.shape[1],
            self.filter.shape[2],
            1,
            1,
        ],
    )

  def __call__(self, inputs):
    channel_num = inputs.shape[-1]
    dimension_numbers = _conv_dimension_numbers(inputs.shape)
    depthwise_filter = jnp.tile(self.filter, [1, 1, 1, 1, channel_num])
    depthwise_filter = depthwise_filter.astype(self.dtype)
    precision = 'bfloat16' if self.dtype == jnp.bfloat16 else 'float32'
    with jax.default_matmul_precision(precision):
      outputs = jax.lax.conv_general_dilated(
          inputs,
          depthwise_filter,
          self.strides,
          self.padding,
          feature_group_count=channel_num,
          dimension_numbers=dimension_numbers,
      )
    return outputs


class DiscriminatorResBlock(base_layer.BaseLayer):
  """ResBlock for StyleGAN Discriminator."""

  input_dim: int = 0
  output_dim: int = 128
  blur_filter_size: int = 3
  conv_layer_tpl: LayerTpl = template_field(convolutions.Conv3D)
  activation_tpl: pax_fiddle.Config[activations.BaseActivation] = (
      template_field(activations.LeakyReLU)
  )
  blur_pool_tpl: LayerTpl = template_field(BlurPool3D)

  def setup(self):
    activation_p = self.activation_tpl.clone()
    self.create_child('activation', activation_p)

    conv_p = self.conv_layer_tpl.clone()
    conv_p.name = 'conv_first'
    conv_p.dtype = self.dtype
    conv_p.fprop_dtype = self.fprop_dtype
    conv_p.padding = 'SAME'
    conv_p.bias = True
    conv_p.filter_shape = (3, 3, 3, self.input_dim, self.input_dim)
    conv_p.filter_stride = (1, 1, 1)
    self.create_child('conv_first', conv_p)

    conv_p = self.conv_layer_tpl.clone()
    conv_p.name = 'conv_residual'
    conv_p.dtype = self.dtype
    conv_p.fprop_dtype = self.fprop_dtype
    conv_p.padding = 'SAME'
    conv_p.bias = False
    conv_p.filter_shape = (1, 1, 1, self.input_dim, self.output_dim)
    conv_p.filter_stride = (1, 1, 1)
    self.create_child('conv_residual', conv_p)

    conv_p = self.conv_layer_tpl.clone()
    conv_p.name = 'conv_last'
    conv_p.dtype = self.dtype
    conv_p.fprop_dtype = self.fprop_dtype
    conv_p.padding = 'SAME'
    conv_p.bias = True
    conv_p.filter_shape = (3, 3, 3, self.input_dim, self.output_dim)
    conv_p.filter_stride = (1, 1, 1)
    self.create_child('conv_last', conv_p)

    blur_pool_p = self.blur_pool_tpl.clone()
    blur_pool_p.name = 'blur_pool'
    blur_pool_p.dtype = self.dtype
    blur_pool_p.fprop_dtype = self.fprop_dtype
    blur_pool_p.filter_size = self.blur_filter_size
    self.create_child('blur_pool', blur_pool_p)

  def __call__(self, inputs: JTensor) -> JTensor:
    if self.input_dim != inputs.shape[-1]:
      raise ValueError(
          f'Input dimension {inputs.shape[-1]} does not match {self.input_dim}'
      )
    residual = inputs
    x = self.conv_first(inputs)
    x = self.activation(x)
    x = self.blur_pool(x)
    residual = self.blur_pool(residual)
    residual = self.conv_residual(residual)
    x = self.conv_last(x)
    x = self.activation(x)
    return (x + residual) / math.sqrt(2.0)
