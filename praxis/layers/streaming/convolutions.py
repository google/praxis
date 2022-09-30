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

"""Streaming aware convolutional layers."""

from __future__ import annotations
from typing import Optional, Sequence, Tuple, Union

import jax
from jax import numpy as jnp
from praxis import base_layer
from praxis import py_utils
from praxis import pytypes
from praxis.layers import convolutions
from praxis.layers.streaming import streaming_base

BaseHParams = base_layer.BaseLayer.HParams
NestedMap = py_utils.NestedMap
NestedJTensor = pytypes.NestedJTensor
WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams
sub_config_field = base_layer.sub_config_field

JTensor = pytypes.JTensor


class ConvBNActWithPadding(convolutions.ConvBNActWithPadding,  # pytype: disable=signature-mismatch
                           streaming_base.StreamingBase):
  """A block of conv-bn-activation layers with padding processing."""

  class HParams(convolutions.ConvBNActWithPadding.HParams):
    """Associated hyperparams for this layer class.

    Attributes:
      frequency_dim: Size of frequency dimension. In non streaming mode conv
        processes input data [batch, time, frequency, channels]. Where only
        channels is known ahead of model creation. In streaming mode
        batch, time, frequency, channels have to be known for states creation:
        batch: defined by user.
        time: defined by dilations[0], filter_shape[0], filter_stride[0].
        frequency: is added here, it also can be inferred from the input data.
        channels: defined by filter_shape[2].
    """
    frequency_dim: Optional[int] = None

  def setup(self) -> None:
    super().setup()
    p = self.hparams
    assert p.frequency_dim is not None, ('Streaming aware model needs to know '
                                         'frequency_dim for streaming state '
                                         'initialization.')

  def _apply_conv2d_bn_act(self,
                           inputs: JTensor,
                           streaming: bool = False) -> JTensor:
    """Conv2D with Batch norm and activation.

    It supports convolution with striding, dilation.

    Args:
      inputs: Input sequence of shape [B, H, W, D_in], also known more popularly
        as NHWC format.
      streaming: If True, it will not apply "SAME" padding in time dimension[H].
        In streaming mode time dimmension will be padded with streaming context,
        so there is no need to apply "SAME" padding (will set pad_height_zero).
        But feature dimension [W] still has to use "SAME" padding in both 
        streaming and non streaming modes.

    Returns:
      Output sequence after applying convolutions of shape [B, H', W', D_out].
      Note that if the padding is SAME and there is no dilation and striding,
      then H' = H and W' = W.
    """
    p = self.hparams
    # Check if the feature_group_count is compatible with the inputs and filter
    # For more information see XLA docs on ConvWithGeneralPadding below
    # https://www.tensorflow.org/xla/operation_semantics#convwithgeneralpadding_convolution
    if inputs.shape[3] % p.filter_shape[2] != 0:
      raise ValueError(f'Input features {inputs.shape[3]} must be a'
                       f'multiple of filter input dim {p.filter_shape[2]} '
                       f'(Input shape: {inputs.shape}, '
                       f'filter shape: {p.filter_shape}).')
    # feature group count is D_in // filter input dim
    feature_group_count = inputs.shape[3] // p.filter_shape[2]
    if p.filter_shape[3] % feature_group_count != 0:
      raise ValueError(f'Filter output dim {p.filter_shape[3]} must be a '
                       f'multiple of feature group count {feature_group_count} '
                       f'(Input shape: {inputs.shape}, '
                       f'filter shape: {p.filter_shape}).')
    padding = self._compute_padding(inputs.shape, pad_height_zero=streaming)

    # The `dimension_numbers=('NHWC', 'HWIO', 'NHWC')` is to be consistent
    # with tf.conv2d, see e.g., see
    # https://github.com/google/jax/blob/main/jax/_src/lax/lax.py#L622
    outputs = jax.lax.conv_general_dilated(
        lhs=inputs,
        rhs=self.theta.w,
        window_strides=p.filter_stride,
        padding=padding,
        rhs_dilation=p.dilations,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=feature_group_count)
    if p.bias:
      outputs += jnp.reshape(self.theta.b, (1,) * (outputs.ndim - 1) + (-1,))

    if p.batch_norm_tpl is not None:
      outputs = self.bn(outputs)
    outputs = self.activation(outputs)
    return outputs

  def _apply_padding(self, outputs: JTensor, paddings: JTensor) -> JTensor:
    """Computes strided padding.

    Args:
      outputs: con2d output [B, H', W', D_out]. paddings will be applied on it.
      paddings: Input padding with shape [B, H]

    Returns:
      Padded conv2d output with the same shape.
      Output padding of shape [B, H'].
      Note that if the padding is SAME and there is no dilation and striding,
      then H' = H and W' = W.
    """
    p = self.hparams
    if p.filter_stride[0] == 1 and p.padding == 'SAME':
      return outputs, paddings

    # Recompute paddings due to stride:
    if p.padding == 'SAME':
      input_length = paddings.shape[1]
      stride = p.filter_stride[0]

      if p.compat_with_lingvo:
        out_padding = paddings[:, stride - 1::stride]
        out_padding = jnp.pad(
            out_padding,
            [[0, 0], [0, outputs.shape[1] - out_padding.shape[1]]],
            constant_values=1)
      else:
        pad_len = (input_length + stride - 1) // stride * stride - input_length
        out_padding = jax.lax.conv_general_dilated(
            lhs=paddings[:, :, None],
            rhs=jnp.ones([1, 1, 1]),
            window_strides=p.filter_stride[:1],
            padding=[(0, pad_len)],
            rhs_dilation=p.dilations[:1],
            dimension_numbers=('NHC', 'HIO', 'NHC'))
        out_padding = jnp.squeeze(out_padding, axis=-1)
    else:
      def rolling_window(arr: JTensor, window: int, stride: int):
        idx = jnp.arange(0, arr.shape[1] - window + 1,
                         stride)[:, None] + jnp.arange(window)[None, :]
        return arr[:, idx]

      window = p.filter_shape[0]
      stride = p.filter_stride[0]
      out_padding = rolling_window(paddings, window, stride)
      out_padding = out_padding.min(axis=-1, keepdims=False)

    outputs = outputs * (1.0 -
                         jnp.expand_dims(jnp.expand_dims(out_padding, -1), -1))
    return outputs, out_padding

  def __call__(self, inputs: JTensor,
               paddings: JTensor) -> Tuple[JTensor, JTensor]:
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
    p = self.hparams
    if inputs.shape[2] != p.frequency_dim:
      raise ValueError(f'Input frequency dimension: {inputs.shape[2]}, '
                       f'has to be equal p.frequency_dim: {p.frequency_dim}')

    # Applying pre padding.
    inputs *= (1 - paddings)[:, :, None, None]

    # Apply Conv2D with BN and activation:
    outputs = self._apply_conv2d_bn_act(inputs, streaming=False)

    # Apply post padding.
    return self._apply_padding(outputs, paddings)

  @property
  def context_length(self) -> int:
    p = self.hparams
    return max(
        0, p.dilations[0] * (p.filter_shape[0] - 1) - (p.filter_stride[0] - 1))

  @classmethod
  def get_right_context(cls, hparams: ConvBNActWithPadding.HParams) -> int:
    if hparams.is_causal:
      return 0
    else:
      raise ValueError('Non causal streaming is not supported yet.')

  @classmethod
  def get_stride(cls, hparams: ConvBNActWithPadding.HParams) -> int:
    return hparams.filter_stride[0]

  def init_states(self,
                  batch_size: int,
                  with_paddings=True):
    """Creates streaming states in base_layer.DECODE_CACHE.

    Args:
      batch_size: defines batch size of streaming states.
      with_paddings: if True it will creates streaming states
        for padding processing, else will set it None (it can save some memory).
    """
    p = self.hparams
    if p.is_causal:
      if self.context_length:
        state_shape = [
            batch_size, self.context_length, p.frequency_dim, p.filter_shape[2]
        ]
        context = jnp.zeros(state_shape, dtype=p.dtype)
        self._update_streaming_state('context', context)
      else:
        self._update_streaming_state('context', None)
    else:
      raise ValueError('Non causal streaming is not supported yet.')

  def streaming_step(
      self,
      inputs: NestedJTensor,
  ) -> NestedJTensor:
    """Streaming Conv2D with support of stride and dilation.

    Args:
      inputs: NestedMap with input sequence JTensor of shape [B, H, W, D_in]
        and paddings.

    Returns:
      NestedMap with conv output with shape [B, H', W', D_out] and paddings.
    """
    p = self.hparams
    if inputs.features.shape[1] % self.stride:
      if inputs.features.shape[1] == 1:
        raise ValueError('Time dimension is 1 and it is not aligned with '
                         'stride, check that total stride is used.')
      else:
        raise ValueError(f'Time dimension {inputs.features.shape[1]}, '
                         f'is not aligned with stride {self.stride}')
    if inputs.features.shape[2] != p.frequency_dim:
      raise ValueError(f'Input frequency dimension: {inputs.features.shape[2]},'
                       f'has to be equal p.frequency_dim: {p.frequency_dim}')

    # Apply pre-padding:
    inputs.features *= (1 - inputs.paddings)[:, :, None, None]

    previous_context = self.get_streaming_state('context')
    if previous_context is not None:
      # [B, context_length+H, W, D_in]
      stream_input = jnp.concatenate(
          [previous_context, inputs.features], axis=1)
      # [B, H, W, D_in]
      context = stream_input[:, -self.context_length:,]
      self._update_streaming_state('context', context)
    else:
      stream_input = inputs.features

    # [B, H, W, D_out]
    output_features = self._apply_conv2d_bn_act(
        stream_input, streaming=True)

    if p.filter_stride[0] == 1 and p.padding == 'SAME':
      out_padding = inputs.paddings
    else:
      paddings = inputs.paddings
      if p.compat_with_lingvo:
        out_padding = paddings[:, self.stride - 1::self.stride]
        out_padding = jnp.pad(
            out_padding,
            [[0, 0], [0, output_features.shape[1] - out_padding.shape[1]]],
            constant_values=1)
      else:
        out_padding = jax.lax.conv_general_dilated(
            lhs=paddings[:, :, None],
            rhs=jnp.ones([1, 1, 1]),
            window_strides=p.filter_stride[:1],
            padding=[(0, 0)],
            rhs_dilation=p.dilations[:1],
            dimension_numbers=('NHC', 'HIO', 'NHC'))
        out_padding = jnp.squeeze(out_padding, axis=-1)

    # Apply post-padding:
    output_features = output_features * (
        1.0 - jnp.expand_dims(jnp.expand_dims(out_padding, -1), -1))
    return NestedMap(features=output_features, paddings=out_padding)


class DepthwiseConv1D(convolutions.BaseDepthwiseConv1D,  # pytype: disable=signature-mismatch
                      streaming_base.StreamingBase):
  """Streaming aware DepthwiseConv1D layer."""

  def setup(self) -> None:
    p = self.hparams
    assert len(p.filter_shape) == 3
    assert p.rhs_dilation_rate > 0

    w_shape = [p.filter_shape[0], 1, p.filter_shape[1] * p.filter_shape[2]]
    if p.use_2d_conv_weight_shape:
      w_shape = w_shape + [1]
    self.create_variable('w', WeightHParams(shape=w_shape))
    if p.bias:
      self.create_variable('b', WeightHParams(shape=[p.dim], init=p.bias_init))

  def get_w(self) -> JTensor:
    p = self.hparams
    if p.use_2d_conv_weight_shape:
      return jnp.squeeze(self.theta.w, -1)
    else:
      return self.theta.w

  def __call__(self,
               inputs: JTensor,
               paddings: Optional[JTensor] = None) -> JTensor:
    """Depthwise convolution layer.

    Args:
      inputs: Input sequence JTensor of shape [B, T, H].
      paddings: Input paddings JTensor of shape [B, T].

    Returns:
      The depthwise conv output with shape [B, T, H].
    """
    p = self.hparams

    # Applying pre-padding.
    if paddings is not None:
      inputs = inputs * (1.0 - jnp.expand_dims(paddings, axis=-1))

    if p.is_causal:
      causal_pad_size = self.context_length
      padding = [(causal_pad_size, 0)]
    else:
      padding = 'SAME'

    return self._apply_conv(inputs, padding)

  def _apply_conv(
      self, inputs: JTensor, padding: Union[str, Sequence[Tuple[int, int]]]
  ) -> JTensor:
    """Core part of the layer, applies convolution.

    Args:
      inputs: Input sequence JTensor of shape [B, T, H].
      padding: It can be defined as string (e.g. SAME), or as sequence of int.

    Returns:
      The depthwise conv output with shape [B, T, H].
    """
    p = self.hparams

    dn = jax.lax.conv_dimension_numbers(inputs.shape,
                                        self.get_w().shape,
                                        ('NHC', 'HIO', 'NHC'))

    out = jax.lax.conv_general_dilated(
        lhs=inputs,
        rhs=self.get_w(),
        window_strides=(1,),
        padding=padding,
        lhs_dilation=(1,),
        rhs_dilation=(p.rhs_dilation_rate,),
        dimension_numbers=dn,
        feature_group_count=p.filter_shape[1])
    if p.bias:
      out = out + self.theta.b
    return out

  @property
  def context_length(self) -> int:
    p = self.hparams
    return p.rhs_dilation_rate * (p.filter_shape[0] - 1)

  @classmethod
  def get_right_context(cls, hparams: DepthwiseConv1D.HParams) -> int:
    if hparams.is_causal:
      return 0
    else:
      raise ValueError('Non causal streaming is not supported yet.')

  @classmethod
  def get_stride(cls, hparams: DepthwiseConv1D.HParams) -> int:
    return 1

  def init_states(self,
                  batch_size: int,
                  with_paddings=True):
    """Creates streaming states in base_layer.DECODE_CACHE.

    Args:
      batch_size: defines batch size of streaming states.
      with_paddings: if True it will creates streaming states
        for padding processing, else will set it None (it can save some memory).
    """
    p = self.hparams
    if p.is_causal:
      state_shape = [batch_size, self.context_length, p.filter_shape[1]]
      context = jnp.zeros(state_shape, dtype=p.dtype)
      self._update_streaming_state('context', context)
    else:
      raise ValueError('Non causal streaming is not supported yet.')

  def streaming_step(
      self,
      inputs: NestedJTensor,
  ) -> NestedJTensor:
    """Depthwise convolution layer in streaming mode.

    Args:
      inputs: NestedMap with input sequence JTensor of shape [B, T, H]
        and paddings.

    Returns:
      NestedMap with depthwise conv output with shape [B, T, H] and paddings.
    """
    # Applying pre-padding.
    if inputs.paddings is not None:
      inputs.features = inputs.features * (
          1.0 - jnp.expand_dims(inputs.paddings, axis=-1))

    # Ring buffer logic:
    # [B, context_length+T, H]
    stream_input = jnp.concatenate(
        [self.get_streaming_state('context'), inputs.features], axis=1)
    # [B, context_length, H]
    context = stream_input[:, -self.context_length:, :]
    self._update_streaming_state('context', context)

    # In streaming mode, we apply conv with VALID padding, so that conv op
    # automatically crops the output and make it [B, T, H].
    output_features = self._apply_conv(stream_input, 'VALID')
    # Output padding is the same with the input, in causal models.
    return NestedMap(features=output_features, paddings=inputs.paddings)


class LightConv1D(convolutions.LightConv1D,  # pytype: disable=signature-mismatch
                  streaming_base.StreamingBase):
  """Streaming aware LightConv1D layer."""

  def _create_conv(self):
    p = self.hparams
    if not isinstance(p.depthwise_conv_tpl,
                      convolutions.DepthwiseConv1D.HParams):
      raise ValueError(f'Streaming mode is not implemented for '
                       f'{p.depthwise_conv_tpl}.')

    depthwise_conv_p = p.depthwise_conv_tpl.clone().set(
        filter_shape=(p.kernel_size, p.input_dims, 1), is_causal=p.is_causal)
    depthwise_conv_p_stream = DepthwiseConv1D.HParams()
    depthwise_conv_p_stream.copy_fields_from(depthwise_conv_p)
    self.create_child('depthwise_conv1d', depthwise_conv_p_stream)

  @classmethod
  def get_stride(cls, hparams: LightConv1D.HParams) -> int:
    return 1

  @classmethod
  def get_right_context(cls, hparams: LightConv1D.HParams) -> int:
    if hparams.is_causal:
      return 0
    else:
      raise ValueError('Non causal streaming is not supported yet.')

  def init_states(self, batch_size: int, with_paddings: bool = True):
    """Creates streaming states in base_layer.DECODE_CACHE.

    Args:
      batch_size: defines batch size of streaming states.
      with_paddings: if True it will creates streaming states
        for padding processing, else will set it None (it can save some memory).
    """

    # Initialize states for all streaming aware sub layers:
    self.depthwise_conv1d.init_states(
        batch_size=batch_size, with_paddings=with_paddings)

  def streaming_step(
      self,
      inputs: NestedJTensor,
  ) -> NestedJTensor:
    """LightConv1D layer in streaming mode.

    Args:
      inputs: NestedMap with input sequence JTensor of shape [B, T, H]
        and paddings.

    Returns:
      NestedMap with conv output with shape [B, T, H] and paddings.
    """
    features = inputs.features
    paddings = inputs.paddings
    unnormalized_features = features

    features = self.ln(features)
    act_features = self.linear_start_act(features)
    gated_features = self.linear_start_gated(features)
    features = act_features * jax.nn.sigmoid(gated_features)

    outputs = self.depthwise_conv1d.streaming_step(
        NestedMap(features=features, paddings=paddings))

    features = self._conv_norm(outputs.features, outputs.paddings)
    features = self.conv_activation(features)

    features = self.linear_end(features)
    features = self.dropout(features)

    output = features + unnormalized_features
    return NestedMap(features=output, paddings=outputs.paddings)
