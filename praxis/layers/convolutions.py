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

"""Convolutional layers."""

import dataclasses
import math
from typing import Sequence

import jax
from jax import numpy as jnp
from praxis import base_layer
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis.layers import activations
from praxis.layers import linears
from praxis.layers import normalizations
from praxis.layers import stochastics

NestedMap = py_utils.NestedMap
SplitDimsMapping = pytypes.SplitDimsMapping
template_field = base_layer.template_field
LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]
WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams


JTensor = pytypes.JTensor


def _extract_pad_beg_end(size: int) -> tuple[int, int]:
  """Gets the beginning and ending padding for a dimension."""
  pad_total = size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  return pad_beg, pad_end


def _effective_kernel_size(filter_shape: int, dilation_rate: int) -> int:
  """Returns the effective filter size for a convolution."""
  return dilation_rate * (filter_shape - 1) + 1


def causal_padding(filter_shape: int, dilation_rate: int) -> int:
  """Gets the padding for a causal convolution."""
  effective_kernel_size = _effective_kernel_size(filter_shape, dilation_rate)
  return effective_kernel_size - 1


class Conv2D(base_layer.BaseLayer):
  """Conv2D with support of SAME/VALID paddings.

  Attributes:
    filter_shape: Filter shape. Must be a sequence of length 4. Elements are in
      the order of height (time), width (frequency), in_channel, out_channel.
    filter_stride: Filter stride to use. Must be a pair of ints. The first int
      specifies the stride on the height dimension. The second int specifies the
      stride on the width dimension.
    dilations: An optional list of ints. Defaults to (1, 1). 1-D tensor of
      length 2. The dilation factor for each dimension of input. If set to k >
      1, there will be k-1 skipped cells between each filter element on that
      dimension.
    bias: Whether or not to apply a bias before activation.
    bias_init: Bias initializer to use if bias is to be applied.
    kernel_init: Optional kernel initializer to use.
    padding: The type of padding to use. It can be 'SAME' or 'VALID'. Note that
      only 'SAME' padding can be combined with is_causal=True.
    tf_equivalent_padding: Whether to make it equivalent to tf. By default we
      apply extra padding that is different than tf conv when stride > 1. This
      is mainly used for multimodal which leads to better accuracy.
    is_causal: Whether this is a causal convolution. This assumes the first
      dimension of filter is time and if is_causal=True, each position would not
      observe any positions in the right. This is achieved by adding extra
      padding in the left to shift the whole convolution.
    weight_norm_tpl: Template to apply weight normalization to self.theta.w.
  """

  filter_shape: Sequence[int] = (0, 0, 0, 0)
  filter_stride: Sequence[int] = (0, 0)
  dilations: Sequence[int] = (1, 1)
  bias: bool = False
  bias_init: WeightInit = dataclasses.field(
      default_factory=lambda: WeightInit.Constant(0.0)
  )
  kernel_init: WeightInit | None = None
  padding: str = 'SAME'
  tf_equivalent_padding: bool = False
  is_causal: bool = False
  weight_norm_tpl: pax_fiddle.Config[normalizations.BaseNormalization] = (
      template_field(normalizations.IdentityNorm)
  )

  @classmethod
  def HParamsDepthwise(
      cls,
      *,
      kernel_shape: Sequence[int],
      in_channels: int,
      channel_multipliers: int = 1,
      **hparams,
  ):
    """DepthwiseConv2D configuration for Conv2D and its subclasses."""
    if len(kernel_shape) != 2:
      raise ValueError(
          f'kernel_shape must have two elements, got {len(kernel_shape)}'
      )
    if 'filter_shape' in hparams:
      raise ValueError('filter_shape cannot be specified in HParamsDepthwise')
    filter_shape = tuple(kernel_shape) + (1, in_channels * channel_multipliers)
    return pax_fiddle.Config(cls, filter_shape=filter_shape, **hparams)

  def check_dimensions(self) -> None:
    """Check dimensions for conv."""
    if not self.name:
      raise ValueError('self.name needs to be set.')
    if self.padding not in ['SAME', 'VALID']:
      raise ValueError(f'padding should be SAME or VALID, got {self.padding}')
    if len(self.filter_shape) != 4:
      raise ValueError(
          f'filter_shape should have 4 values, got {self.filter_shape}'
      )
    if len(self.filter_stride) != 2:
      raise ValueError(
          f'filter_stride should have 2 values, got {self.filter_stride}'
      )
    if len(self.dilations) != 2:
      raise ValueError(f'dilations should have 2 values, got {self.dilations}')
    if not all(x > 0 for x in self.filter_stride):
      raise ValueError(
          f'filter_stride should be > 0, got f{self.filter_stride}'
      )

    # error if is_causal but not tf_equivalent_padding
    if self.is_causal and (not self.tf_equivalent_padding):
      raise ValueError(
          'Causal convolution is only supported for tf equivalent padding'
      )

    # error if is_causal but padding == 'valid'
    if self.is_causal and self.padding == 'VALID':
      raise NotImplementedError(
          "Causal convolution doesn't support valid padding"
      )

  def setup(self) -> None:
    self.check_dimensions()
    wp = self.weight_split_dims_mapping
    self.create_variable(
        'w',
        WeightHParams(
            shape=self.filter_shape,
            mesh_shape=self.mesh_shape,
            tensor_split_dims_mapping=wp.wt,
            dtype=self.dtype,
            init=self.kernel_init,
        ),
    )
    if self.bias:
      self.create_variable(
          'b',
          WeightHParams(
              shape=[self.filter_shape[-1]],
              init=self.bias_init,
              dtype=self.dtype,
          ),
      )

    wn = self.weight_norm_tpl.clone().set(dim=self.filter_shape[-1])
    self.weight_norm: normalizations.BaseNormalization
    self.create_child('weight_norm', wn)

  def _compute_padding(self, pad_height_zero=False):
    if not self.tf_equivalent_padding:
      if self.padding == 'SAME':
        filter_height = _effective_kernel_size(
            self.filter_shape[0], self.dilations[0]
        )
        filter_width = _effective_kernel_size(
            self.filter_shape[1], self.dilations[1]
        )
        pad_height_beg, pad_height_end = _extract_pad_beg_end(filter_height)
        pad_width_beg, pad_width_end = _extract_pad_beg_end(filter_width)
      else:
        assert self.padding == 'VALID', self.padding
        pad_height_beg = 0
        pad_height_end = 0
        pad_width_beg = 0
        pad_width_end = 0
      padding = [
          (pad_height_beg, pad_height_end),
          (pad_width_beg, pad_width_end),
      ]
    else:
      if not self.is_causal:
        padding = self.padding
      else:
        # Compute padding for causal convolution.
        pad_height_total = causal_padding(
            self.filter_shape[0], self.dilations[0]
        )
        pad_width_total = causal_padding(
            self.filter_shape[1], self.dilations[1]
        )

        # Causal padding for the height dimension.
        pad_height_beg = 0 if pad_height_zero else pad_height_total
        pad_height_end = 0
        # SAME padding for the width dimension. To match TensorFlow SAME padding
        # behavior, put the extra padding at the end when pad amount is odd.
        # https://www.tensorflow.org/api_docs/python/tf/nn#same_padding
        pad_width_beg = pad_width_total // 2
        pad_width_end = pad_width_total - pad_width_beg

        padding = [
            (pad_height_beg, pad_height_end),
            (pad_width_beg, pad_width_end),
        ]
    return padding

  def _shard_bhwc(self, x: JTensor) -> JTensor:
    """Adds sharding annotations to tensors of shape [b, h, w, c]."""
    if len(x.shape) != 4:
      return x
    ap = self.activation_split_dims_mapping
    return base_layer.maybe_shard(x, ap.out, self.mesh_axis_names)

  def __call__(self, inputs: JTensor) -> JTensor:
    """FProp that supports strided, dilated convolution, depthwise convolution.

    Args:
      inputs: Input sequence of shape [B, H, W, D_in], also known more popularly
        as NHWC format.

    Returns:
      Output sequence after applying convolutions of shape [B, H', W', D_out].
      Note that if the padding is SAME and there is no dilation and striding,
      then H' = H and W' = W.
    """
    # Check if the feature_group_count is compatible with the inputs and filter
    # For more information see XLA docs on ConvWithGeneralPadding below
    # https://www.tensorflow.org/xla/operation_semantics#convwithgeneralpadding_convolution
    if inputs.shape[3] % self.filter_shape[2] != 0:
      raise ValueError(
          f'Input features {inputs.shape[3]} must be a'
          f'multiple of filter input dim {self.filter_shape[2]} '
          f'(Input shape: {inputs.shape}, '
          f'filter shape: {self.filter_shape}).'
      )
    # feature group count is D_in // filter input dim
    feature_group_count = inputs.shape[3] // self.filter_shape[2]
    if self.filter_shape[3] % feature_group_count != 0:
      raise ValueError(
          f'Filter output dim {self.filter_shape[3]} must be a '
          f'multiple of feature group count {feature_group_count} '
          f'(Input shape: {inputs.shape}, '
          f'filter shape: {self.filter_shape}).'
      )
    padding = self._compute_padding()
    inputs = self._shard_bhwc(inputs.astype(self.fprop_dtype))

    # The `dimension_numbers=('NHWC', 'HWIO', 'NHWC')` is to be consistent
    # with tf.conv2d, see e.g., see
    # https://github.com/google/jax/blob/main/jax/_src/lax/lax.py#L622
    outputs = jax.lax.conv_general_dilated(
        lhs=inputs,
        rhs=self.weight_norm(self.theta.w),
        window_strides=self.filter_stride,
        padding=padding,
        rhs_dilation=self.dilations,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=feature_group_count,
    )
    outputs = self._shard_bhwc(outputs)
    if self.bias:
      outputs += jnp.reshape(self.theta.b, (1,) * (outputs.ndim - 1) + (-1,))
    return outputs


class Conv3D(base_layer.BaseLayer):
  """Conv3D with support of SAME/VALID paddings.

  Attributes:
    filter_shape: Must be a sequence of length 5. Elements are ordered: time,
      height, width, in_channel, out_channel.
    filter_stride: Filter stride to use. Must be a three ints. The first int
      specifies the stride on the time, then height, then width.
    dilations: An optional list of ints. Defaults to (1, 1, 1). 1-D tensor of
      length 3. The dilation factor for each dimension of input. If set to k >
      1, there will be k-1 skipped cells between each filter element on that
      dimension.
    bias: Whether or not to apply a bias before activation.
    bias_init: Bias initializer to use if bias is to be applied.
    kernel_init: Optional kernel initializer to use.
    padding: The type of padding to use.
    tf_equivalent_padding: Whether to make it equivalent to tf. By default we
      apply extra padding that is different than tf conv when stride > 1. This
      is mainly used for multimodal which leads to better accuracy.
    is_causal: Whether this is a causal convolution. This assumes the first
      dimension of filter is time and if is_causal=True, each position would not
      observe any positions in the right. This is achieved by adding extra
      padding in the left to shift the whole convolution.
  """

  filter_shape: Sequence[int] = (0, 0, 0, 0, 0)
  filter_stride: Sequence[int] = (0, 0, 0)
  dilations: Sequence[int] = (1, 1, 1)
  bias: bool = False
  bias_init: WeightInit = dataclasses.field(
      default_factory=lambda: WeightInit.Constant(0.0)
  )
  kernel_init: WeightInit | None = None
  padding: str = 'SAME'
  tf_equivalent_padding: bool = False
  is_causal: bool = False

  @classmethod
  def HParamsDepthwise(
      cls,
      *,
      kernel_shape: Sequence[int],
      in_channels: int,
      channel_multipliers: int = 1,
      **hparams,
  ):
    """DepthwiseConv3D configuration for Conv3D and its subclasses."""
    if len(kernel_shape) != 3:
      raise ValueError(
          f'kernel_shape must have three elements, got {len(kernel_shape)}'
      )
    if 'filter_shape' in hparams:
      raise ValueError('filter_shape cannot be specified in HParamsDepthwise')
    filter_shape = tuple(kernel_shape) + (1, in_channels * channel_multipliers)
    return pax_fiddle.Config(cls, filter_shape=filter_shape, **hparams)

  def setup(self) -> None:
    if not self.name:
      raise ValueError('self.name needs to be set.')
    if self.padding not in ['SAME', 'VALID']:
      raise ValueError(f'padding should be SAME or VALID, got {self.padding}')
    if len(self.filter_shape) != 5:
      raise ValueError(
          f'filter_shape should have 5 values, got {self.filter_shape}'
      )
    if len(self.filter_stride) != 3:
      raise ValueError(
          f'filter_stride should have 3 values, got {self.filter_stride}'
      )
    if len(self.dilations) != 3:
      raise ValueError(f'dilations should have 3 values, got {self.dilations}')
    if not all(x > 0 for x in self.filter_stride):
      raise ValueError(f'filter_stide should be > 0, got f{self.filter_stride}')

    # error if is_causal but not tf_equivalent_padding
    if self.is_causal and (not self.tf_equivalent_padding):
      raise ValueError(
          'Causal convolution is only supported for tf equivalent padding'
      )

    # error if is_causal but padding == 'valid'
    if self.is_causal and self.padding == 'VALID':
      raise NotImplementedError(
          "Causal convolution doesn't support valid padding"
      )

    wp = self.weight_split_dims_mapping
    self.create_variable(
        'w',
        WeightHParams(
            shape=self.filter_shape,
            mesh_shape=self.mesh_shape,
            tensor_split_dims_mapping=wp.wt,
            init=self.kernel_init,
        ),
    )
    if self.bias:
      self.create_variable(
          'b', WeightHParams(shape=[self.filter_shape[-1]], init=self.bias_init)
      )

  def _compute_padding(self, pad_height_zero=False):
    if not self.tf_equivalent_padding:
      if self.padding == 'SAME':
        filter_time = _effective_kernel_size(
            self.filter_shape[0], self.dilations[0]
        )
        filter_height = _effective_kernel_size(
            self.filter_shape[1], self.dilations[1]
        )
        filter_width = _effective_kernel_size(
            self.filter_shape[2], self.dilations[2]
        )
        pad_time_beg, pad_time_end = _extract_pad_beg_end(filter_time)
        pad_height_beg, pad_height_end = _extract_pad_beg_end(filter_height)
        pad_width_beg, pad_width_end = _extract_pad_beg_end(filter_width)
      else:
        assert self.padding == 'VALID', self.padding
        pad_time_beg = 0
        pad_time_end = 0
        pad_height_beg = 0
        pad_height_end = 0
        pad_width_beg = 0
        pad_width_end = 0
      padding = [
          (pad_time_beg, pad_time_end),
          (pad_height_beg, pad_height_end),
          (pad_width_beg, pad_width_end),
      ]
    else:
      if not self.is_causal:
        padding = self.padding
      else:
        # Compute padding for causal convolution.
        pad_time_total = causal_padding(self.filter_shape[0], self.dilations[0])
        pad_height_total = causal_padding(
            self.filter_shape[1], self.dilations[1]
        )
        pad_width_total = causal_padding(
            self.filter_shape[2], self.dilations[2]
        )

        # Causal padding for the height dimension.
        pad_time_beg = 0 if pad_height_zero else pad_time_total
        pad_time_end = 0
        # SAME padding for the height and width dimensions. To match TensorFlow
        # SAME padding behavior, put the extra padding at the end when pad
        # amount is odd.
        # https://www.tensorflow.org/api_docs/python/tf/nn#same_padding
        pad_height_beg = pad_height_total // 2
        pad_height_end = pad_height_total - pad_height_beg
        pad_width_beg = pad_width_total // 2
        pad_width_end = pad_width_total - pad_width_beg

        padding = [
            (pad_time_beg, pad_time_end),
            (pad_height_beg, pad_height_end),
            (pad_width_beg, pad_width_end),
        ]
    return padding

  def __call__(self, inputs: JTensor) -> JTensor:
    """FProp that supports strided, dilated convolution, depthwise convolution.

    Args:
      inputs: Input sequence of shape [B, T, H, W, D_in].

    Returns:
      Output sequence after applying convolution, shape [B, T', H', W', D_out].
      Note that if the padding is SAME and there is no dilation and striding,
      then H' = H and W' = W.
    """
    # Check if the feature_group_count is compatible with the inputs and filter
    # For more information see XLA docs on ConvWithGeneralPadding below
    # https://www.tensorflow.org/xla/operation_semantics#convwithgeneralpadding_convolution
    if inputs.shape[4] % self.filter_shape[3] != 0:
      raise ValueError(
          f'Input features {inputs.shape[4]} must be a'
          f'multiple of filter input dim {self.filter_shape[3]} '
          f'(Input shape: {inputs.shape}, '
          f'filter shape: {self.filter_shape}).'
      )
    # feature group count is D_in // filter input dim
    feature_group_count = inputs.shape[4] // self.filter_shape[3]
    if self.filter_shape[4] % feature_group_count != 0:
      raise ValueError(
          f'Filter output dim {self.filter_shape[4]} must be a '
          f'multiple of feature group count {feature_group_count} '
          f'(Input shape: {inputs.shape}, '
          f'filter shape: {self.filter_shape}).'
      )
    padding = self._compute_padding()
    inputs = inputs.astype(self.fprop_dtype)

    # The `dimension_numbers=('NTHWC', 'THWIO', 'NTHWC')` is to be consistent
    # with tf.conv3d.
    outputs = jax.lax.conv_general_dilated(
        lhs=inputs,
        rhs=self.theta.w,
        window_strides=self.filter_stride,
        padding=padding,
        rhs_dilation=self.dilations,
        dimension_numbers=('NTHWC', 'THWIO', 'NTHWC'),
        feature_group_count=feature_group_count,
    )
    if self.bias:
      outputs += jnp.reshape(self.theta.b, (1,) * (outputs.ndim - 1) + (-1,))
    return outputs


class ConvBNAct(Conv2D):
  """A block of conv-bn-activation layers used for image encoders.

  By default, we use cross-replica sum on TPUs.

  Attributes:
    batch_norm_tpl: The batchnorm layer template.
    activation_tpl: Activation function to use.
  """

  batch_norm_tpl: LayerTpl | None = template_field(normalizations.BatchNorm)
  activation_tpl: pax_fiddle.Config[activations.BaseActivation] = (
      template_field(activations.ReLU)
  )

  def setup(self) -> None:
    super().setup()

    if self.batch_norm_tpl is not None:
      bn = self.batch_norm_tpl.clone()
      bn.dim = self.filter_shape[3]
      bn.use_moving_avg_in_training = False
      self.create_child('bn', bn)
    self.create_child('activation', self.activation_tpl.clone())

  def __call__(self, inputs: JTensor) -> JTensor:
    """Forward prop which applies conv-bn-activation.

    Args:
      inputs: Input sequence of shape [B, H, W, D_in], also known more popularly
        as NHWC format.

    Returns:
      Output sequence after applying convolutions of shape [B, H', W', D_out].
      Note that if the padding is SAME and there is no dilation and striding,
      then H' = H and W' = W.
    """
    outputs = super().__call__(inputs)
    if self.batch_norm_tpl is not None:
      outputs = self.bn(outputs)
    outputs = self.activation(outputs)
    return outputs


class ConvBNActWithPadding(ConvBNAct):
  """A block of conv-bn-activation layers with padding processing.

  Attributes:
    compat_with_lingvo: If use lingvo-compatible logic.
  """

  compat_with_lingvo: bool = False

  def setup(self) -> None:
    super().setup()
    if self.compat_with_lingvo:
      assert tuple(self.dilations) == (
          1,
          1,
      ), 'compat_with_lingvo supports no dilation.'

  def __call__(
      self, inputs: JTensor, paddings: JTensor
  ) -> tuple[JTensor, JTensor]:  # pytype:disable=signature-mismatch
    """Forward prop which applies conv-bn-activation with time paddings.

    Args:
      inputs: Input sequence of shape [B, H, W, D_in], also known more popularly
        as NHWC format.
      paddings: Input sequence of shape [B, H], where H is the time dimension.

    Returns:
      Output sequence after applying convolutions of shape [B, H', W', D_out].
      Note that if the padding is SAME and there is no dilation and striding,
      then H' = H and W' = W.
      Output padding after applying convolutions.
    """

    # Applying padding.
    inputs = py_utils.apply_padding(inputs, paddings[:, :, None, None])

    outputs = super().__call__(inputs)

    if self.filter_stride[0] == 1 and self.padding == 'SAME':
      return outputs, paddings
    if self.padding == 'SAME':
      input_length = paddings.shape[1]
      stride = self.filter_stride[0]

      if self.compat_with_lingvo:
        out_padding = paddings[:, stride - 1 :: stride]
        out_padding = jnp.pad(
            out_padding,
            [[0, 0], [0, outputs.shape[1] - out_padding.shape[1]]],
            constant_values=1,
        )
      else:
        pad_len = (input_length + stride - 1) // stride * stride - input_length
        out_padding = jax.lax.conv_general_dilated(
            lhs=paddings[:, :, None],
            rhs=self._cast_to_fprop_dtype(jnp.ones([1, 1, 1])),
            window_strides=self.filter_stride[:1],
            padding=[(0, pad_len)],
            rhs_dilation=self.dilations[:1],
            dimension_numbers=('NHC', 'HIO', 'NHC'),
        )
        out_padding = jnp.squeeze(out_padding, axis=-1)
    else:

      def rolling_window(arr: JTensor, window: int, stride: int):
        idx = (
            jnp.arange(0, arr.shape[1] - window + 1, stride)[:, None]
            + jnp.arange(window)[None, :]
        )
        return arr[:, idx]

      window = self.filter_shape[0]
      stride = self.filter_stride[0]
      out_padding = rolling_window(paddings, window, stride)
      out_padding = out_padding.min(axis=-1, keepdims=False)
    outputs = outputs * (
        1.0 - jnp.expand_dims(jnp.expand_dims(out_padding, -1), -1)
    )
    return outputs, out_padding


class BaseDepthwiseConv1D(base_layer.BaseLayer):
  """Base class for Depthwise 1D convolution.

  Attributes:
    filter_shape: Filter shape. Must be a sequence of length 3. Elements are in
      the order of kernel_size, in_channels, channel_multipliers.
    bias:         Whether or not to apply a bias before activation.
    bias_init:    Bias initializer to use if bias is to be applied.
    is_causal:    Whether this is a causal layer.
    use_2d_conv_weight_shape: Whether to use 2d conv's weight shape. This is for
      checkpoint backwards-compatibility.
    rhs_dilation_rate: The dilation rate in atrous convolution.
  """

  filter_shape: Sequence[int] = (0, 0, 0)
  bias: bool = False
  bias_init: WeightInit = dataclasses.field(
      default_factory=lambda: WeightInit.Constant(0.0)
  )
  is_causal: bool = False
  use_2d_conv_weight_shape: bool = False
  rhs_dilation_rate: int = 1

  def __call__(
      self, inputs: JTensor, paddings: JTensor | None = None
  ) -> JTensor:
    """Depthwise convolution.

    Args:
      inputs: Input sequence JTensor of shape [B, T, H].
      paddings: Input paddings JTensor of shape [B, T].

    Returns:
      The depthwise conv output with shape [B, T, H * channel_multipliers]
        where channel_multipliers = filter_shape[-1].
    """
    raise NotImplementedError()


class DepthwiseConv1D(BaseDepthwiseConv1D):
  """Depthwise 1D convolution based on lax implementation."""

  # SPMD partition related params.
  # h - height
  # w - width
  # i - in_channels
  # m - channel_multiplier
  class WeightSharding(base_layer.BaseLayer.WeightSharding):
    """Represents how layer's learned parameters are partitioned across a mesh.

    Attributes:
      him:  Mesh split for weight. If use_2d_conv_weight_shape is set, the
        weight shape is actually him, and w dim is not sharded.
    """

    him: SplitDimsMapping = None

  def _input_channels(self) -> int:
    """The number of in channels (override for special handling)."""
    return self.filter_shape[1]

  def _output_channels(self) -> int:
    """The number of in channels (override for special handling)."""
    return self.filter_shape[1] * self.filter_shape[2]

  def setup(self) -> None:
    wp_him = self.weight_split_dims_mapping.clone().him

    assert len(self.filter_shape) == 3
    assert self.rhs_dilation_rate > 0

    w_shape = [
        self.filter_shape[0],
        1,
        self.filter_shape[1] * self.filter_shape[2],
    ]
    bias_shape = self._output_channels()

    if self.use_2d_conv_weight_shape:
      w_shape = w_shape + [1]
      if wp_him:
        # [h, None, i, m]
        wp_him.insert(1, None)

    self.create_variable(
        'w',
        WeightHParams(
            shape=w_shape,
            mesh_shape=self.mesh_shape,
            tensor_split_dims_mapping=wp_him,
        ),
    )
    if self.bias:
      self.create_variable(
          'b', WeightHParams(shape=[bias_shape], init=self.bias_init)
      )

  def get_w(self) -> JTensor:
    wp_him = self.weight_split_dims_mapping.him
    if self.use_2d_conv_weight_shape:
      w = jnp.squeeze(self.theta.w, -1)
      w = base_layer.maybe_shard(w, wp_him, self.mesh_axis_names)
      return w
    else:
      return self.theta.w

  def __call__(
      self, inputs: JTensor, paddings: JTensor | None = None
  ) -> JTensor:
    """Depthwise convolution layer.

    Args:
      inputs: Input sequence JTensor of shape [B, T, H].
      paddings: Input paddings JTensor of shape [B, T].

    Returns:
      The depthwise conv output with shape [B, T, H].
    """

    # Applying padding.
    if paddings is not None:
      inputs = py_utils.apply_padding(inputs, paddings[:, :, None])

    dn = jax.lax.conv_dimension_numbers(
        inputs.shape, self.get_w().shape, ('NHC', 'HIO', 'NHC')
    )

    if self.is_causal:
      causal_pad_size = causal_padding(
          self.filter_shape[0], self.rhs_dilation_rate
      )
      padding = [(causal_pad_size, 0)]
    else:
      padding = 'SAME'

    if self.fprop_dtype is not None and inputs.dtype != self.fprop_dtype:
      inputs = inputs.astype(self.fprop_dtype)

    out = jax.lax.conv_general_dilated(
        lhs=inputs,
        rhs=self.get_w(),
        window_strides=(1,),
        padding=padding,
        lhs_dilation=(1,),
        rhs_dilation=(self.rhs_dilation_rate,),
        dimension_numbers=dn,
        feature_group_count=self._input_channels(),
    )
    if self.bias:
      out = out + self.theta.b
    return out


class LightConv1D(base_layer.BaseLayer):
  """Lightweight conv layer.

  architecture::

  input-ln()-ff()-glu()-depthwise_conv1d()-norm()-act()-ff()-dropout()-+-output
    |__________________________________________________________________|

  Attributes:
    input_dims:      Input and (in fact,) output dimension.
    kernel_size:     Kernel size of 1d deptwise conv.
    conv_activation_tpl: Activation after normalization.
    dropout_prob:    Dropout probability.
    ln_tpl:          Parameterization of input layer normalization.
    linear_start_tpl:     Parameterization of linear start layer.
    depthwise_conv_tpl:   Parameterization of depthwise conv layer.
    conv_norm_layer_tpl:  Parameterization of normalization layer after conv.
    linear_end_tpl:       Parameterization of linear end layer.
    dropout_tpl:          Parameterization of residual dropout layer.
    is_causal:            Whether this is a causal layer.
    use_2d_conv_norm:     Whether to expand the input to conv_norm to 2d. This
      is for compatibility with old models trained in TF lingvo.
  """

  # TODO(nanxinchen): add causal support
  # TODO(nanxinchen): add SPMD partitioning support

  input_dims: int | None = None
  kernel_size: int | None = None
  conv_activation_tpl: pax_fiddle.Config[activations.BaseActivation] = (
      template_field(activations.Swish)
  )
  dropout_prob: float = 0.0
  ln_tpl: LayerTpl = template_field(normalizations.LayerNorm)

  linear_start_tpl: LayerTpl = template_field(linears.FeedForward)
  depthwise_conv_tpl: LayerTpl = template_field(DepthwiseConv1D)
  conv_norm_layer_tpl: LayerTpl = template_field(normalizations.BatchNorm)
  linear_end_tpl: LayerTpl = template_field(linears.FeedForward)
  dropout_tpl: LayerTpl = template_field(stochastics.Dropout)
  is_causal: bool = False
  use_2d_conv_norm: bool = False

  # SPMD partition related params.
  #
  # d - model_dim
  # f - ff_hidden_dim (here ff_hidden_dim has the same size as model_dim)
  # h - height
  # i - in_channels
  # m - channel_multiplier
  # b - batch_size
  # l - seq_len
  class WeightSharding(base_layer.BaseLayer.WeightSharding):
    """Represents how layer's learned parameters are partitioned across a mesh.

    Attributes:
      df:    Mesh split for lconv linear start weight.
      him:  Mesh split for lconv depthwise conv weight.
    """

    df: SplitDimsMapping = None
    him: SplitDimsMapping = None

  class ActivationSharding(base_layer.BaseLayer.ActivationSharding):
    """Represents how intermediate values should be partitioned across a mesh.

    Attributes:
      blf: Mesh split for lconv linear start act and lconv depthwise conv after
        normalization.
      bld: Mesh split for lconv linear end act.
    """

    blf: SplitDimsMapping = None
    bld: SplitDimsMapping = None

  def _create_conv(self):
    wp = self.weight_split_dims_mapping
    depthwise_conv_p = self.depthwise_conv_tpl.clone().set(
        filter_shape=(self.kernel_size, self.input_dims, 1),
        is_causal=self.is_causal,
    )
    depthwise_conv_p.weight_split_dims_mapping.him = wp.him
    self.create_child('depthwise_conv1d', depthwise_conv_p)

  def _create_conv_norm_layer(self):
    norm_p = self.conv_norm_layer_tpl.clone().set(dim=self.input_dims)
    self.create_child('conv_norm', norm_p)

  def setup(self) -> None:
    wp = self.weight_split_dims_mapping
    ap = self.activation_split_dims_mapping

    ln_p = self.ln_tpl.clone().set(name='ln', dim=self.input_dims)
    self.create_child('ln', ln_p)

    # The activation/gate matrix is a sub-matrix of large matrix so the scale
    # needs to be smaller.
    # More specifically, since input_dim == output_dim, the scale of xavier
    # should be \sqrt(6) / \sqrt(input_dim + input_dim * 2) instead of \sqrt(6)
    # / \sqrt(input_dim + input_dim).

    linear_start_act_p = self.linear_start_tpl.clone().set(
        input_dims=self.input_dims,
        output_dims=self.input_dims,
        activation_tpl=pax_fiddle.Config(activations.Identity),
        params_init=WeightInit.Xavier(math.sqrt(3 / 2)),
    )
    linear_start_act_p.weight_split_dims_mapping.wt = wp.df
    linear_start_act_p.activation_split_dims_mapping.out = ap.blf
    self.create_child('linear_start_act', linear_start_act_p)

    linear_start_gated_p = self.linear_start_tpl.clone().set(
        input_dims=self.input_dims,
        output_dims=self.input_dims,
        activation_tpl=pax_fiddle.Config(activations.Identity),
        params_init=WeightInit.Xavier(math.sqrt(3 / 2)),
    )
    linear_start_gated_p.weight_split_dims_mapping.wt = wp.df
    linear_start_gated_p.activation_split_dims_mapping.out = ap.blf
    self.create_child('linear_start_gated', linear_start_gated_p)

    # TODO(nanxinchen): the end layer doesn't split so it shouldn't use 3/2
    linear_end_p = self.linear_end_tpl.clone().set(
        input_dims=self.input_dims,
        output_dims=self.input_dims,
        activation_tpl=pax_fiddle.Config(activations.Identity),
        params_init=WeightInit.Xavier(math.sqrt(3 / 2)),
    )
    if wp.df:
      linear_end_p.weight_split_dims_mapping.wt = list(reversed(wp.df))
    linear_end_p.activation_split_dims_mapping.out = ap.bld
    self.create_child('linear_end', linear_end_p)

    self._create_conv()
    self._create_conv_norm_layer()
    self.create_child('conv_activation', self.conv_activation_tpl.clone())

    dropout_p = self.dropout_tpl.clone().set(keep_prob=1.0 - self.dropout_prob)
    self.create_child('dropout', dropout_p)

  def _conv_norm(self, inputs: JTensor, paddings: JTensor) -> JTensor:
    if self.use_2d_conv_norm:
      # BTH -> BT1H
      inputs = jnp.expand_dims(inputs, 2)
    inputs = self.conv_norm(inputs, paddings)
    if self.use_2d_conv_norm:
      # BT1H -> BTH
      inputs = jnp.squeeze(inputs, 2)
    return inputs

  def __call__(self, inputs: JTensor, paddings: JTensor) -> JTensor:
    """Lightweight conv layer.

    Args:
      inputs: Input sequence JTensor of shape [B, T, H].
      paddings: Input paddings JTensor of shape [B, T].

    Returns:
      The lconv output with shape [B, T, H].
    """
    ap = self.activation_split_dims_mapping

    unnormalized_inputs = inputs

    inputs = self.ln(inputs, paddings)
    act_inputs = self.linear_start_act(inputs)
    gated_inputs = self.linear_start_gated(inputs)
    inputs = act_inputs * jax.nn.sigmoid(gated_inputs)

    inputs = self.depthwise_conv1d(inputs, paddings)
    inputs = base_layer.maybe_shard(inputs, ap.blf, self.mesh_axis_names)

    inputs = self._conv_norm(inputs, paddings)

    inputs = base_layer.maybe_shard(inputs, ap.blf, self.mesh_axis_names)
    inputs = self.conv_activation(inputs)

    inputs = self.linear_end(inputs)
    inputs = self.dropout(inputs)

    output = inputs + unnormalized_inputs
    return output
