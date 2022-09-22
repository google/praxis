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

    if p.is_causal:
      causal_pad_size = self.left_context
      padding = [(causal_pad_size, 0)]
    else:
      padding = 'SAME'

    return self._apply_conv_paddings(inputs, paddings, padding)

  def _apply_conv_paddings(
      self, inputs: JTensor, paddings: JTensor,
      padding: Union[str, Sequence[Tuple[int, int]]]
  ) -> JTensor:
    """Core part of the layer, applies convolution with paddings.

    Args:
      inputs: Input sequence JTensor of shape [B, T, H].
      paddings: Input paddings JTensor of shape [B, T].
      padding: It can be defined as string (e.g. SAME), or as sequence of int.

    Returns:
      The depthwise conv output with shape [B, T, H].
    """
    p = self.hparams

    # Applying padding.
    if paddings is not None:
      inputs = inputs * (1.0 - jnp.expand_dims(paddings, axis=-1))

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
  def left_context(self) -> int:
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
      state_shape = [batch_size, self.left_context, p.filter_shape[1]]
      context = jnp.zeros(state_shape, dtype=p.dtype)
      self._update_streaming_state('context', context)

      state_shape = [batch_size, self.left_context]
      paddings = jnp.zeros(
          state_shape, dtype=jnp.float32) if with_paddings else None
      self._update_streaming_state('paddings', paddings)
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

    paddings_context = self.get_streaming_state('paddings')
    if paddings_context is not None:
      if inputs.paddings is None:
        raise ValueError('Streaming padding state is defined '
                         'but inputs.paddings is None.')
      # [B, left_context+T]
      stream_paddings = jnp.concatenate([paddings_context, inputs.paddings],
                                        axis=1)
      # [B, left_context, H]
      paddings_context = stream_paddings[:, -self.left_context:,]
      self._update_streaming_state('paddings', paddings_context)
    else:
      stream_paddings = None

    # Ring buffer logic:
    # [B, left_context+T, H]
    stream_input = jnp.concatenate(
        [self.get_streaming_state('context'), inputs.features], axis=1)
    # [B, left_context, H]
    context = stream_input[:, -self.left_context:, :]
    self._update_streaming_state('context', context)

    # In streaming mode, we apply conv with VALID padding, so that conv op
    # automatically crops the output and make it [B, T, H].
    output_features = self._apply_conv_paddings(stream_input, stream_paddings,
                                                'VALID')
    # Output padding is the same with the input, in causal models.
    return NestedMap(features=output_features, paddings=inputs.paddings)


class LightConv1D(convolutions.LightConv1D,  # pytype: disable=signature-mismatch
                  streaming_base.StreamingBase):
  """Streaming aware LightConv1D layer."""

  class HParams(convolutions.LightConv1D.HParams):
    # Replace DepthwiseConv1D by its streaming aware version:
    _attribute_overrides: Tuple[str, ...] = ('depthwise_conv_tpl',)
    depthwise_conv_tpl: BaseHParams = sub_config_field(
        DepthwiseConv1D.HParams)

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
