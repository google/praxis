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

"""Streaming aware Conformer-related layers."""

from typing import Tuple

from jax import numpy as jnp
from praxis import base_layer
from praxis import py_utils
from praxis import pytypes
from praxis.layers import conformers
from praxis.layers.streaming import attentions
from praxis.layers.streaming import convolutions
from praxis.layers.streaming import streaming_base

BaseHParams = base_layer.BaseLayer.HParams
NestedMap = py_utils.NestedMap
NestedJTensor = pytypes.NestedJTensor
sub_config_field = base_layer.sub_config_field


class SelfAttentionWithNormAndResidual(  # pytype: disable=signature-mismatch
    conformers.SelfAttentionWithNormAndResidual,
    streaming_base.StreamingBase):
  """Streaming aware self attention sub-layer used in the Conformer layer.

  Input is first normalized using norm_tpl. Output is processed using
  multi-headed attention. And finally, the output of the attention layer
  is combined with the input by residual connection.

  For the normalization, we can specify pre norm or post norm.
  For the residual connection, we can specify the residual weight.

  Model topology:
     ____________________________
    |                            |
  input -> norm -> self_atten -> + ->

  If self_atten has a delay then residual connection also will need a delay, so
  that signals summed in (+) are aligned in time.
  """

  class HParams(conformers.SelfAttentionWithNormAndResidual.HParams):
    # Replace DotProductAttention by its streaming aware version:
    _attribute_overrides: Tuple[str, ...] = ('self_atten_tpl',)
    self_atten_tpl: BaseHParams = sub_config_field(
        attentions.LocalSelfAttention.HParams)

  def setup(self) -> None:
    super().setup()
    p = self.hparams

    # These optional parameters can be used only with DotProductAttention.
    # In streaming mode we use self attention which has
    # its own left and right parameters. So checking it to avoid a conflict:
    assert p.left_context is None
    assert p.right_context is None
    del p.right_context
    del p.left_context

    assert not issubclass(p.norm_tpl.cls, streaming_base.StreamingBase), (
        'It has to be non streaming aware normalization.')

  @classmethod
  def get_right_context(cls, hparams):
    return hparams.self_atten_tpl.right_context

  @classmethod
  def get_stride(cls, hparams):
    return 1

  def init_states(self,
                  batch_size: int,
                  with_paddings: bool = True):
    """Creates streaming states in base_layer.DECODE_CACHE.

    Args:
      batch_size: defines batch size of streaming states.
      with_paddings: if True it will creates streaming states
        for padding processing, else will set it None (it can save some memory).
    """
    p = self.hparams

    # Initialize states for all streaming aware sub layers:
    self.self_atten.init_states(
        batch_size=batch_size, with_paddings=with_paddings)

    # State for delay
    if self.self_atten.hparams.right_context > 0:
      delay = jnp.zeros([
          batch_size, self.self_atten.hparams.right_context,
          self.self_atten.hparams.input_dim
      ], p.dtype)
      self._update_streaming_state('delay', delay)

  def streaming_step(
      self,
      inputs: NestedJTensor,
  ) -> NestedJTensor:
    """Streaming inference step for local self attention.

    Args:
      inputs: NestedMap with input query_vec, key_vec and value_vec JTensor
        of shape [B, T, H] and paddings [B, T].

    Returns:
      NestedMap with encoded output with shape [B, T, H] and paddings.
    """
    p = self.hparams

    paddings = inputs.paddings
    inputs = inputs.features
    unnormalized_inputs = inputs
    step = inputs.shape[1]

    if p.self_atten_tpl.right_context > 0:
      stream_input = jnp.concatenate(
          [self.get_streaming_state('delay'), inputs], axis=1)
      unnormalized_inputs = stream_input[:, :step]
      stream_state = stream_input[:, -p.self_atten_tpl.right_context:]
      self._update_streaming_state('delay', stream_state)

    if p.pre_layer_norm:
      inputs = self.norm(inputs)

    outputs = self.self_atten.streaming_step(py_utils.NestedMap(
        query_vec=inputs,
        key_vec=inputs,
        value_vec=inputs,
        paddings=paddings))

    result = outputs.encoded
    paddings = outputs.paddings

    result = (
        self.residual_dropout(result) * p.residual_weight +
        unnormalized_inputs * p.input_weight)

    if not p.pre_layer_norm:
      result = self.norm(result)
    return NestedMap(features=result, paddings=paddings)


class Conformer(  # pytype: disable=signature-mismatch
    conformers.Conformer,
    streaming_base.StreamingBase):
  """Streaming aware Conformer layer.

    Canonical version (with default params.)
      x = x + 1/2 * FFN(x)
      x = x + MHSA(x)
      x = x + Lconv(x)
      x = x + 1/2 * FFN(x)
      y = ln(x)

    Residual connections are implemented inside each individual block:
      FFN, MHSA, LConv.
    Optionally one can change the order of MHSA and conv.
  """

  class HParams(conformers.Conformer.HParams):
    _attribute_overrides: Tuple[str, ...] = ('trans_atten_tpl', 'lconv_tpl',)
    # Replace SelfAttentionWithNormAndResidual by its streaming aware version:
    trans_atten_tpl: BaseHParams = sub_config_field(
        SelfAttentionWithNormAndResidual.HParams)

    # Replace LightConv1D by its streaming aware version:
    lconv_tpl: BaseHParams = sub_config_field(
        convolutions.LightConv1D.HParams)

  @classmethod
  def get_right_context(cls, hparams):
    return hparams.trans_atten_tpl.self_atten_tpl.right_context

  @classmethod
  def get_stride(cls, hparams):
    return 1

  def init_states(self,
                  batch_size: int,
                  with_paddings: bool = True):
    """Creates streaming states in base_layer.DECODE_CACHE.

    Args:
      batch_size: defines batch size of streaming states.
      with_paddings: if True it will creates streaming states
        for padding processing, else will set it None (it can save some memory).
    """
    # Initialize states for all streaming aware sub layers:
    self.trans_atten.init_states(
        batch_size=batch_size, with_paddings=with_paddings)
    self.lconv.init_states(batch_size=batch_size, with_paddings=with_paddings)

  def streaming_step(
      self,
      inputs: NestedJTensor,
  ) -> NestedJTensor:
    """Streaming inference step for Conformer.

    Args:
      inputs: NestedMap with input query_vec, key_vec and value_vec JTensor
        of shape [B, T, H] and paddings [B, T].

    Returns:
      NestedMap with encoded output with shape [B, T, H] and paddings.
    """
    p = self.hparams

    features = inputs.features
    paddings = inputs.paddings

    if self.has_fflayer_start:
      features = self.fflayer_start(features, paddings)

    in_nmap = NestedMap(features=features, paddings=paddings)
    if p.layer_order == 'mhsa':
      outputs = self.trans_atten.streaming_step(in_nmap)
    elif p.layer_order == 'conv':
      outputs = self.lconv.streaming_step(in_nmap)
    elif p.layer_order == 'mhsa_before_conv':
      outputs = self.trans_atten.streaming_step(in_nmap)
      outputs = self.lconv.streaming_step(outputs)
    else:
      assert p.layer_order == 'conv_before_mhsa'
      outputs = self.lconv.streaming_step(in_nmap)
      outputs = self.trans_atten.streaming_step(outputs)

    features = outputs.features
    paddings = outputs.paddings

    if self.has_fflayer_end:
      features = self.fflayer_end(features, paddings)
    elif p.fflayer_weight_sharing:
      # With the weight sharing, we apply fflayer_start again
      features = self.fflayer_start(features, paddings)

    if self.has_final_ln:
      features = self.final_ln(features)
    return NestedMap(features=features, paddings=paddings)
