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

"""Conformer-related layers."""

from typing import Optional
from praxis import asserts
from praxis import base_layer
from praxis import py_utils
from praxis.layers import activations
from praxis.layers import attentions
from praxis.layers import convolutions
from praxis.layers import normalizations
from praxis.layers import stochastics
from praxis.layers import transformers

import jax.numpy as jnp

NestedMap = py_utils.NestedMap
JTensor = base_layer.JTensor
sub_config_field = base_layer.sub_config_field

BaseHParams = base_layer.BaseLayer.HParams


class SelfAttentionWithNormAndResidual(base_layer.BaseLayer):
  """Self attention sub-layer used in the Conformer layer.

  Input is first normalized using norm_tpl. Output is processed using
  multi-headed attention. And finally, the output of the attention layer
  is combined with the input by residual connection.

  For the normalization, we can specify pre norm or post norm.
  For the residual connection, we can specify the residual weight.
  """

  class HParams(BaseHParams):
    """Associated hyper-params for this layer class.

    Attributes:
      residual_weight: Weight of the residual connection. Output = fn(x) *
        residual_weight + x * input_weight.
      input_weight: Weight of the input connection. Output = fn(x) *
        residual_weight + x * input_weight.
      self_atten_tpl: Parameterization of the self attention layer.
      norm_tpl: Parameterization of the normalization layer.
      pre_layer_norm: Whether to apply norm before or after the layer.
      residual_dropout_prob: Probability at which we apply dropout to the
        residual layers, such that, residual(x, y) = (x + dropout(y)).
      residual_dropout_tpl: Parameterization of residual dropout layer.
        keep_prop will be reset to (1.0 - residual_dropout_prob).
      left_context: Number of left positions to attend (including current
        position). If set, use a limited attention context from the left.
        Otherwise if it is None, use all the frames in the left.
      right_context: Number of right positions to attend. If set, use a limited
        attention context from the right. Otherwise if it is None, use all the
        frames in the right. For causal, set it to 0.
    """
    residual_weight: float = 1.0
    input_weight: float = 1.0
    self_atten_tpl: BaseHParams = sub_config_field(
        attentions.DotProductAttention.HParams)
    norm_tpl: BaseHParams = sub_config_field(normalizations.LayerNorm.HParams)
    pre_layer_norm: bool = True
    residual_dropout_prob: float = 0.0
    residual_dropout_tpl: BaseHParams = sub_config_field(
        stochastics.Dropout.HParams)
    left_context: Optional[int] = None
    right_context: Optional[int] = None

  def setup(self) -> None:
    p = self.hparams
    asserts.not_none(p.self_atten_tpl)
    self.create_child('self_atten', p.self_atten_tpl)

    self.create_child('norm', p.norm_tpl)

    # Initialize residual dropout.
    params = p.residual_dropout_tpl.clone()
    params.keep_prob = (1.0 - p.residual_dropout_prob)
    self.create_child('residual_dropout', params)

  def __call__(self,
               inputs: JTensor,
               paddings: JTensor,
               atten_mask: Optional[JTensor] = None) -> JTensor:
    p = self.hparams

    unnormalized_inputs = inputs

    if p.pre_layer_norm:
      inputs = self.norm(inputs)

    if p.left_context is not None or p.right_context is not None:
      asserts.none(atten_mask)
      atten_mask = attentions.limited_context_mask_from_padding(
          paddings, p.left_context, p.right_context)
    else:
      if atten_mask is None:
        atten_mask = attentions.convert_paddings_to_mask(paddings, inputs.dtype)
      else:
        padding_mask = attentions.convert_paddings_to_mask(
            paddings, inputs.dtype)
        atten_mask = jnp.minimum(atten_mask, padding_mask)
    result = self.self_atten(
        query_vec=inputs,
        key_vec=inputs,
        value_vec=inputs,
        atten_mask=atten_mask)[0]

    result = (
        self.residual_dropout(result) * p.residual_weight +
        unnormalized_inputs * p.input_weight)
    if not p.pre_layer_norm:
      result = self.norm(result)
    return result


class Conformer(base_layer.BaseLayer):
  """Conformer layer as in https://arxiv.org/abs/2005.08100.

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

  # TODO(nanxinchen): add causal support

  class HParams(BaseHParams):
    """Associated hyper-params for this layer class.

    Attributes:
      input_dims: Input dimension.
      model_dims: Encoder model dimension.
      kernel_size: Conv kernel size.
      ff_activation_tpl: Activation function used in the feedforward network.
      ff_residual_weight: Residual weight used in the fflayer.
      ffn_dim_multiplier: Feed forward hidden dimension will be
        ffn_dim_multiplier * model_dims.
      atten_num_heads: Number of attention heads.
      layer_order: Only mhsa, conv, mhsa_before_conv or conv_before_mhsa are
        supported
      dropout_prob: Dropout prob of inner components.
      conv_residual_dropout: Conv block residual dropout. Will be overwritten by
        p.dropout if it is not None.
      atten_residual_dropout: Attention block residual dropout. Will be
        overwritten by p.dropout if it is not None.
      ffn_residual_dropout: Feed forward block residual dropout. Will be
        overwritten by p.dropout if it is not None.
      atten_dropout: Dropout in Attention layer. Will be overwritten by
        p.dropout if it is not None.
      ffn_relu_dropout: Post activation dropout in Feed-forward layer. Will be
        overwritten by p.dropout if it is not None.
      fflayer_start_tpl: Parameterization for the Feed forward layer at the
        beginning. If set to None, this layer is excluded.
      trans_atten_tpl: Parameterization of self-attention layer.
      lconv_tpl: Parameterization of convolution layer. If set to None, this
        layer is excluded.
      fflayer_end_tpl: Parameterization for Feed forward layer at the end. If
        set to None, this layer is excluded.
      fflayer_weight_sharing: If True, will ignore `fflayer_end_tpl`, and will
        make the fflayer_end layer as a weight-shared copy of the fflayer_start
        layer.
      final_ln_tpl: Parameterization of the final layer norm.
    """
    input_dims: Optional[int] = None
    model_dims: int = 512
    kernel_size: int = 32
    ff_activation_tpl: activations.BaseActivation.HParams = sub_config_field(
        activations.Swish.HParams)
    ff_residual_weight: float = 0.5
    ffn_dim_multiplier: int = 4
    atten_num_heads: int = 8
    layer_order: str = 'mhsa_before_conv'
    dropout_prob: Optional[float] = None
    conv_residual_dropout: Optional[float] = None
    atten_residual_dropout: Optional[float] = None
    ffn_residual_dropout: Optional[float] = None
    atten_dropout: Optional[float] = None
    ffn_relu_dropout: Optional[float] = None
    fflayer_start_tpl: Optional[BaseHParams] = sub_config_field(
        transformers.TransformerFeedForward.HParams)
    trans_atten_tpl: BaseHParams = sub_config_field(
        SelfAttentionWithNormAndResidual.HParams)
    lconv_tpl: Optional[BaseHParams] = sub_config_field(
        convolutions.LightConv1D.HParams)
    fflayer_end_tpl: Optional[BaseHParams] = sub_config_field(
        transformers.TransformerFeedForward.HParams)
    fflayer_weight_sharing: bool = False
    final_ln_tpl: BaseHParams = sub_config_field(
        normalizations.LayerNorm.HParams)

  def setup(self) -> None:
    p = self.hparams
    asserts.in_set(p.layer_order,
                   ['mhsa', 'conv', 'mhsa_before_conv', 'conv_before_mhsa'])

    if p.dropout_prob is not None:
      all_dropouts = [
          p.atten_dropout, p.atten_residual_dropout, p.conv_residual_dropout,
          p.ffn_residual_dropout, p.ffn_relu_dropout
      ]
      for prob in all_dropouts:
        assert prob is None or prob == p.dropout_prob

    def dropout_prob(prob):
      return p.dropout_prob if p.dropout_prob is not None else prob

    if p.fflayer_start_tpl:
      if p.input_dims == p.model_dims:
        fflayer_start_p = p.fflayer_start_tpl.clone().set(
            name='fflayer_start',
            activation_tpl=p.ff_activation_tpl.clone(),
            input_dims=p.input_dims,
            hidden_dims=p.model_dims * p.ffn_dim_multiplier,
            residual_weight=p.ff_residual_weight,
            residual_dropout_prob=dropout_prob(p.ffn_residual_dropout),
            relu_dropout_prob=dropout_prob(p.ffn_relu_dropout),
        )
      else:
        # Need to add another projection layer in fflayer
        fflayer_start_p = p.fflayer_start_tpl.clone().set(
            name='fflayer_start',
            activation_tpl=p.ff_activation_tpl.clone(),
            input_dims=p.input_dims,
            output_dims=p.model_dims,
            hidden_dims=p.model_dims * p.ffn_dim_multiplier,
            residual_weight=p.ff_residual_weight,
            residual_dropout_prob=dropout_prob(p.ffn_residual_dropout),
            relu_dropout_prob=dropout_prob(p.ffn_relu_dropout),
        )
      self.create_child(fflayer_start_p.name, fflayer_start_p)

    if p.fflayer_end_tpl:
      fflayer_end_p = p.fflayer_end_tpl.clone().set(
          name='fflayer_end',
          activation_tpl=p.ff_activation_tpl.clone(),
          input_dims=p.model_dims,
          hidden_dims=p.model_dims * p.ffn_dim_multiplier,
          residual_weight=p.ff_residual_weight,
          residual_dropout_prob=dropout_prob(p.ffn_residual_dropout),
          relu_dropout_prob=dropout_prob(p.ffn_relu_dropout),
      )
      if not p.fflayer_weight_sharing:
        self.create_child(fflayer_end_p.name, fflayer_end_p)
      else:
        asserts.not_none(p.fflayer_start_tpl)

    if 'mhsa' in p.layer_order:
      trans_atten_p = p.trans_atten_tpl.clone().set(
          residual_dropout_prob=dropout_prob(p.atten_residual_dropout),
          self_atten_tpl=p.trans_atten_tpl.self_atten_tpl.clone().set(
              input_dim=p.model_dims,
              hidden_dim=p.model_dims,
              atten_dropout_prob=dropout_prob(p.atten_dropout),
              num_heads=p.atten_num_heads))
      trans_atten_p.norm_tpl = trans_atten_p.norm_tpl.clone().set(
          dim=p.model_dims)
      self.create_child('trans_atten', trans_atten_p)

    if 'conv' in p.layer_order:
      lconv_p = p.lconv_tpl.clone().set(
          input_dims=p.model_dims,
          kernel_size=p.kernel_size,
          dropout_prob=dropout_prob(p.conv_residual_dropout))
      self.create_child('lconv', lconv_p)

    if p.final_ln_tpl:
      ln_p = p.final_ln_tpl.clone().set(name='final_ln', dim=p.model_dims)
      self.create_child('final_ln', ln_p)

  @property
  def has_fflayer_start(self) -> bool:
    return hasattr(self, 'fflayer_start')

  @property
  def has_fflayer_end(self) -> bool:
    return hasattr(self, 'fflayer_end')

  @property
  def has_final_ln(self) -> bool:
    return hasattr(self, 'final_ln')

  def __call__(self,
               inputs: JTensor,
               paddings: JTensor,
               atten_mask: Optional[JTensor] = None) -> JTensor:
    """Conformer layer.

    Args:
      inputs: Input sequence JTensor of shape [B, T, H].
      paddings: Input paddings JTensor of shape [B, T] (only used in FFN layer).

    Returns:
      The conformer output with shape [B, T, D].
    """
    p = self.hparams

    if atten_mask is not None and 'mhsa' not in p.layer_order:
      raise RuntimeError('Attention mask is provided but no attention layer.')

    if self.has_fflayer_start:
      inputs = self.fflayer_start(inputs, paddings)

    if p.layer_order == 'mhsa':
      inputs = self.trans_atten(
          inputs=inputs, paddings=paddings, atten_mask=atten_mask)
    elif p.layer_order == 'conv':
      inputs = self.lconv(inputs, paddings)
    elif p.layer_order == 'mhsa_before_conv':
      inputs = self.trans_atten(
          inputs=inputs, paddings=paddings, atten_mask=atten_mask)
      inputs = self.lconv(inputs, paddings)
    else:
      assert p.layer_order == 'conv_before_mhsa'
      inputs = self.lconv(inputs, paddings)
      inputs = self.trans_atten(
          inputs=inputs, paddings=paddings, atten_mask=atten_mask)

    if self.has_fflayer_end:
      inputs = self.fflayer_end(inputs, paddings)
    elif p.fflayer_weight_sharing:
      # With the weight sharing, we apply fflayer_start again
      inputs = self.fflayer_start(inputs, paddings)

    if self.has_final_ln:
      inputs = self.final_ln(inputs)
    return inputs
