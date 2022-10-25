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

"""Helper function to config GLaM models."""

from praxis import base_layer
from praxis import pax_fiddle
from praxis.layers import activations
from praxis.layers import attentions
from praxis.layers import embedding_softmax
from praxis.layers import normalizations
from praxis.layers import transformer_models
from praxis.layers import transformers

LanguageModelType = transformer_models.LanguageModelType


def GlamStackedTransformerHParams(
    model_dim,
    ff_dim,
    attention_num_heads,
    attention_key_value_dim,
    name='transformer',
    moe=False,
    moe_hidden_dim=None,
    moe_gating_embedding_level='token',
    ffn_activation_cls=activations.GELU,
    use_gated_activation=True,
    mask_self_attention=True,
    cross_attention=False,
    atten_logit_cap=0.0,
    attention_extra_logit=0.0,
    relative_attention_num_buckets=32,
    relative_attention_max_distance=128,
    moe_load_balance_loss_weight=0.01,
    moe_gating_func='top2',
    moe_gating_logit_cap=0.0,
    num_groups=1,
    c_dim=None,
    capacity_factor=0.0,
    e_dim=None,
    combine_qkv=False,
    bidirectional=False) -> transformers.StackedTransformer.HParams:
  """Common setup for GLaM Transformer layers.

  This function setups a transformer block for both MoE and dense GLaM models.
  The MoE block consists of two transformer layer with the feedforward
  sublayer of the first one replaced by a MoE layer. The dense block consists
  of a transformer. The transformer layer used by GLam differs from the
  standard transformer in these configs:
  1) The feedforward sublayer used gated gleu so there are two wi and one wo.
  2) No bias in all projections.
  3) Use no bias RMS norm for the layer norm.
  4) Use relative attention bias

  Args:
    model_dim: model dimension.
    ff_dim: hidden dimension of feed-forward inner layer.
    attention_num_heads: number of attention heads.
    attention_key_value_dim: key value dimension of attention inner layer.
    name: Name of the this layer
    moe: If this is a moe block or not.
    moe_hidden_dim: hidden dimension of MoE layer.
    moe_gating_embedding_level: Specifies the type of MOE gating embedding used.
    ffn_activation_cls: Activation function class used in the ffn layer.
    use_gated_activation: Whether to use gated activation in the ffn layer or
      not.
    mask_self_attention: Use masked self-attention.
    cross_attention: If set, use cross encoder-decoder attention layer.
    atten_logit_cap: Attention logits cap to apply to attention logits.
    attention_extra_logit: Extra logit for attention softmax.
    relative_attention_num_buckets: Relative attention num buckets
    relative_attention_max_distance: Max relative distance.
    moe_load_balance_loss_weight: Weight of load balancing loss in MoE layers.
    moe_gating_func: Gating function choice in the MoE Layer.
    moe_gating_logit_cap:  Cap the absolute values of MoE gating logits by tanh.
      Enabled when a positive value is specified.
    num_groups: Total number of groups for token dispatching in MoE layer.
    c_dim: Expert capacity.
    capacity_factor: This is the ratio between max allowed examples per expert
      over the average number of examples per expert assuming routing is
      completely uniform.
    e_dim: Number of experts.
    combine_qkv: if combined qkv projection layer is used.
    bidirectional: Set to support bidirectional relative attention.

  Returns:
    A Params object to set up a StackedTransformer.
  """
  ffn_activation_tpl = getattr(ffn_activation_cls, 'HParams')()

  p = transformers.StackedTransformer.HParams()
  p.name = name
  p.packed_input = True
  p.moe_layers = [0] if moe else None
  p.num_layers = 2 if moe else 1
  p.model_dims = model_dim
  p.hidden_dims = ff_dim
  p.num_heads = attention_num_heads
  p.dim_per_head = attention_key_value_dim
  p.num_experts = e_dim
  p.num_groups = num_groups
  p.mask_self_attention = mask_self_attention
  p.use_cross_attention = cross_attention
  # Attention setup
  if isinstance(p.transformer_layer_params_tpl, (list, tuple)):
    for pt in p.transformer_layer_params_tpl:
      pt.ln_tpl = normalizations.RmsNorm.HParams()
      pt.ln_tpl.direct_scale = True
  else:
    assert isinstance(p.transformer_layer_params_tpl,
                      (base_layer.BaseLayer.HParams, pax_fiddle.Config))
    p.transformer_layer_params_tpl.ln_tpl = normalizations.RmsNorm.HParams()
    p.transformer_layer_params_tpl.ln_tpl.direct_scale = True
  tr_atten_tpl = p.transformer_layer_params_tpl.tr_atten_tpl  # pytype: disable=attribute-error  # enable-nested-classes
  assert tr_atten_tpl.cls == attentions.DotProductAttention
  tr_atten_tpl.attention_extra_logit = attention_extra_logit
  tr_atten_tpl.use_bias = False
  tr_atten_tpl.atten_logit_cap = atten_logit_cap
  tr_atten_tpl.internal_gshard_gaussian_init = True
  tr_atten_tpl.internal_enable_query_scale = False
  tr_atten_tpl.internal_enable_per_dim_scale = False
  assert tr_atten_tpl.proj_tpl.cls == attentions.AttentionProjection
  tr_atten_tpl.proj_tpl.attention_combine_dims = True
  tr_atten_tpl.relative_bias_tpl = attentions.RelativeBias.HParams(
      relative_attention_num_buckets=relative_attention_num_buckets,
      relative_attention_max_distance=relative_attention_max_distance,
      bidirectional=bidirectional)
  tr_atten_tpl.output_proj_use_nhd_shape = True
  if combine_qkv:
    tr_atten_tpl.combine_qkv = True
    tr_atten_tpl.combined_qkv_proj_tpl.use_bias = False
    tr_atten_tpl.combined_qkv_proj_tpl.attention_combine_dims = True
  # Non-MoE ffn setup
  ff_tpl = p.transformer_layer_params_tpl.tr_fflayer_tpl  # pytype: disable=attribute-error  # enable-nested-classes
  assert ff_tpl.cls == transformers.TransformerFeedForward
  ff_tpl.input_dims = model_dim
  ff_tpl.hidden_dims = ff_dim
  ff_tpl.has_bias = False
  ff_tpl.apply_padding_first = True
  ff_tpl.ln_tpl = normalizations.RmsNorm.HParams()
  ff_tpl.ln_tpl.direct_scale = True
  ff_tpl.add_skip_connection = True
  ff_tpl.activation_tpl = ffn_activation_tpl
  ff_tpl.use_gated_activation = use_gated_activation
  ff_tpl.internal_gshard_variance_scaling_fan_in_init = True
  # MoE ffn setup
  moe_p = p.moe_layer_tpl
  assert moe_p.cls == transformers.TransformerFeedForwardMoe
  moe_p.input_dims = model_dim
  moe_p.hidden_dims = moe_hidden_dim or ff_dim
  moe_p.ln_tpl = normalizations.RmsNorm.HParams()
  moe_p.ln_tpl.direct_scale = True
  moe_p.num_experts = e_dim
  moe_p.num_groups = num_groups
  moe_p.gating_logit_cap = moe_gating_logit_cap
  moe_p.expert_capacity_dim = c_dim
  moe_p.unadjusted_expert_capacity_factor = capacity_factor
  moe_p.internal_gshard_variance_scaling_fan_in_init = True
  moe_p.moe_load_balance_loss_weight = moe_load_balance_loss_weight
  moe_p.gating_func = moe_gating_func
  moe_p.moe_gating_embedding_level = moe_gating_embedding_level
  return p


def GlamUniTransformerLmHParams(
    vocab_size,
    model_dim,
    ff_dim,
    attention_num_heads,
    attention_key_value_dim,
    num_transformer_layers,
    name='transformer',
    moe=False,
    moe_hidden_dim=None,
    moe_gating_embedding_level='token',
    ffn_activation_cls=activations.GELU,
    use_gated_activation=True,
    atten_logit_cap=0.0,
    attention_extra_logit=0.0,
    relative_attention_num_buckets=32,
    relative_attention_max_distance=128,
    moe_gating_func='top2',
    moe_gating_logit_cap=0.0,
    num_groups=1,
    c_dim=None,
    capacity_factor=0.0,
    e_dim=None,
    use_tgt_labels_size_as_loss_denominator=True,
    moe_load_balance_loss_weight=0.01,
    z_loss_weight=1e-4,
    combine_qkv=False,
    bidirectional=False,
    num_pipeline_stages=1,
    num_pipeline_microbatches=1,
    model_type=LanguageModelType.CAUSAL
) -> transformer_models.TransformerLm.HParams:
  """Common setup for GLaM Decoder-only Transformer Model.

  This function sets up configs for both MoE and dense GLaM models.
  The MoE block consists of two transformer layer with the feedforward
  sublayer of the first one replaced by a MoE layer. The dense block consists
  of a transformer. The transformer layer used by GLam differs from the
  standard transformer in these configs:

  1) The feedforward sublayer used gated gleu so there are two wi and one wo.
  2) No bias in all projections and embeddings.
  3) Use no bias RMS norm for the layer norm.
  4) Use relative attention bias.
  5) Add an optional z-loss to stablize final softmax logits.

  Args:
    vocab_size: Size of the vocabulary for LM.
    model_dim: model dimension.
    ff_dim: hidden dimension of feed-forward inner layer.
    attention_num_heads: number of attention heads.
    attention_key_value_dim: key value dimension of attention inner layer.
    num_transformer_layers: Number of transformer layers.
    name: Name of the this layer
    moe: If this is a moe block or not.
    moe_hidden_dim: hidden dimension of MoE layer.
    moe_gating_embedding_level: Specifies the type of MOE gating embedding used.
    ffn_activation_cls: Activation function class used in the ffn layer.
    use_gated_activation: Whether to use gated activation in the ffn layer or
      not.
    atten_logit_cap: Attention logits cap to apply to attention logits.
    attention_extra_logit: Extra logit for attention softmax.
    relative_attention_num_buckets: Relative attention num buckets
    relative_attention_max_distance: Max relative distance.
    moe_gating_func: Gating function choice in the MoE Layer.
    moe_gating_logit_cap:  Cap the absolute values of MoE gating logits by tanh.
      Enabled when a positive value is specified.
    num_groups: Total number of groups for token dispatching in MoE layer.
    c_dim: Expert capacity.
    capacity_factor: This is the ratio between max allowed examples per expert
      over the average number of examples per expert assuming routing is
      completely uniform.
    e_dim: Number of experts.
    use_tgt_labels_size_as_loss_denominator: False to use total number of
      non-padding tokens instead of fixed tgt_labels tensor size.
    moe_load_balance_loss_weight: Weight of the aux loss for MoE layers.
    z_loss_weight: additional loss term to stablize the final softmax logit.
    combine_qkv: if combined qkv projection layer is used.
    bidirectional: Set to support bidirectional relative attention.
    num_pipeline_stages: Number of pipeline stages.
    num_pipeline_microbatches: Number of pipeline microbatches.
    model_type: Type of the Language Model. Either `CAUSAL`, `PREFIX`, or
      `BIDIRECTIONAL`.

  Returns:
    A Params object to set up a StackedTransformer.
  """
  p = transformer_models.TransformerLm.HParams()
  p.name = name
  p.packed_input = True
  p.model_dims = model_dim
  p.vocab_size = vocab_size
  p.position_emb_tpl = None
  p.model_type = model_type

  p.final_ln_tpl = normalizations.RmsNorm.HParams(
      name='rms_norm', dim=model_dim)

  p.softmax_tpl = (
      embedding_softmax.GShardSharedEmbeddingSoftmax.HParams(
          name='emb',
          input_dims=model_dim,
          num_classes=vocab_size,
          z_loss_weight=z_loss_weight))
  p.softmax_tpl.use_tgt_labels_size_as_loss_denominator = (
      use_tgt_labels_size_as_loss_denominator)
  glam_p = GlamStackedTransformerHParams(
      model_dim=model_dim,
      ff_dim=ff_dim,
      attention_num_heads=attention_num_heads,
      attention_key_value_dim=attention_key_value_dim,
      name='decoder_block',
      moe=moe,
      moe_hidden_dim=moe_hidden_dim,
      ffn_activation_cls=ffn_activation_cls,
      use_gated_activation=use_gated_activation,
      mask_self_attention=True,
      cross_attention=False,
      atten_logit_cap=atten_logit_cap,
      attention_extra_logit=attention_extra_logit,
      relative_attention_num_buckets=relative_attention_num_buckets,
      relative_attention_max_distance=relative_attention_max_distance,
      moe_load_balance_loss_weight=moe_load_balance_loss_weight,
      moe_gating_func=moe_gating_func,
      moe_gating_logit_cap=moe_gating_logit_cap,
      num_groups=num_groups,
      c_dim=c_dim,
      capacity_factor=capacity_factor,
      e_dim=e_dim,
      combine_qkv=combine_qkv,
      bidirectional=bidirectional,
      moe_gating_embedding_level=moe_gating_embedding_level,
  )

  num_blocks = num_transformer_layers // 2 if moe else num_transformer_layers

  if num_pipeline_stages == 1:
    p.stacked_transformer_tpl = (
        transformers.StackedTransformerRepeated.HParams(
            name='decoder',
            unroll_in_decode=True,
            block=glam_p,
            x_times=num_blocks))
  else:
    assert num_blocks % num_pipeline_stages == 0
    glam_p.num_layers = num_transformer_layers // num_pipeline_stages
    glam_p.moe_layers = list(range(0, glam_p.num_layers, 2))
    p.stacked_transformer_tpl = transformers.PipelinedTransformer.HParams(
        pipeline_stage=glam_p,
        num_pipeline_stages=num_pipeline_stages,
        num_pipeline_microbatches=num_pipeline_microbatches,
        stream_io=True)
  return p
