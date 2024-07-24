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

"""Helper function to config Grok models."""

import fiddle as fdl
from praxis import pax_fiddle
from praxis.layers import activations
from praxis.layers import attentions
from praxis.layers import checkpoint_policy
from praxis.layers import embedding_softmax
from praxis.layers import multi_query_attention
from praxis.layers import normalizations
from praxis.layers import transformer_models
from praxis.layers import transformers
from praxis.layers.injection import fp8_nvidia_gpu as fp8_ops

AutodiffCheckpointType = checkpoint_policy.AutodiffCheckpointType
LanguageModelType = transformer_models.LanguageModelType


def GrokStackedTransformerHParams(
    model_dim,
    ff_dim,
    attention_num_heads,
    attention_num_groups,
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
    bidirectional=False,
    use_fp8=False,
) -> pax_fiddle.Config[transformers.StackedTransformer]:
  """Common setup for Grok-1 Transformer layers.

  This function setups a transformer block for both MoE and dense Grok-1 models.
  The MoE block consists of two transformer layer with the feedforward
  sublayer of the first one replaced by a MoE layer. The dense block consists
  of a transformer.

  Args:
    model_dim: model dimension.
    ff_dim: hidden dimension of feed-forward inner layer.
    attention_num_heads: number of attention heads.
    attention_num_groups: number of attention kv_heads.
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
  ffn_activation_tpl = pax_fiddle.Config(ffn_activation_cls)

  p = pax_fiddle.Config(transformers.StackedTransformer)

  p.name = name
  p.packed_input = True
  p.moe_layers = [0] if moe else None
  p.num_layers = 1
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
      pt.ln_tpl = pax_fiddle.Config(normalizations.RmsNorm)
      pt.ln_tpl.direct_scale = True
  else:
    assert isinstance(p.transformer_layer_params_tpl, pax_fiddle.Config)
    p.transformer_layer_params_tpl.ln_tpl = pax_fiddle.Config(
        normalizations.RmsNorm
    )
    p.transformer_layer_params_tpl.ln_tpl.direct_scale = True
  tr_atten_tpl = p.transformer_layer_params_tpl.tr_atten_tpl  # pytype: disable=attribute-error  # enable-nested-classes
  assert fdl.get_callable(tr_atten_tpl) == attentions.DotProductAttention
  tr_atten_tpl.attention_extra_logit = attention_extra_logit
  tr_atten_tpl.use_bias = True
  tr_atten_tpl.atten_logit_cap = atten_logit_cap
  tr_atten_tpl.internal_gshard_gaussian_init = True
  tr_atten_tpl.internal_enable_query_scale = False
  tr_atten_tpl.internal_enable_per_dim_scale = False

  assert tr_atten_tpl.proj_tpl.cls == attentions.AttentionProjection
  tr_atten_tpl.proj_tpl.attention_combine_dims = True
  tr_atten_tpl.output_proj_use_nhd_shape = True
  if combine_qkv:
    tr_atten_tpl.combine_qkv = True
    tr_atten_tpl.combined_qkv_proj_tpl.use_bias = False
    tr_atten_tpl.combined_qkv_proj_tpl.attention_combine_dims = True

  # MoE ffn setup
  moe_p = p.moe_layer_tpl
  assert fdl.get_callable(moe_p) == transformers.TransformerFeedForwardMoe
  moe_p.input_dims = model_dim
  moe_p.hidden_dims = moe_hidden_dim or ff_dim
  moe_p.ln_tpl = pax_fiddle.Config(normalizations.RmsNorm)
  moe_p.ln_tpl.direct_scale = True
  moe_p.add_skip_connection = True
  moe_p.activation_tpl = ffn_activation_tpl
  moe_p.use_gated_activation = use_gated_activation
  moe_p.num_experts = e_dim
  moe_p.num_groups = num_groups
  moe_p.gating_logit_cap = moe_gating_logit_cap
  moe_p.expert_capacity_dim = c_dim
  moe_p.unadjusted_expert_capacity_factor = capacity_factor
  moe_p.internal_gshard_variance_scaling_fan_in_init = True
  moe_p.moe_load_balance_loss_weight = moe_load_balance_loss_weight
  moe_p.gating_func = moe_gating_func
  moe_p.moe_gating_embedding_level = moe_gating_embedding_level
  p.transformer_layer_params_tpl.tr_atten_tpl = pax_fiddle.Config(
      multi_query_attention.MultiQueryDotProductAttention,
      num_kv_heads=attention_num_groups,
  )
  tr_atten_tpl = p.transformer_layer_params_tpl.tr_atten_tpl
  tr_atten_tpl.combine_qkv = False
  tr_atten_tpl.proj_tpl.use_bias = True

  if use_fp8:
    p.transformer_layer_params_tpl.tr_atten_tpl.proj_tpl.einsum_tpl = (
        pax_fiddle.Config(fp8_ops.Fp8EinsumOp)
    )
    if combine_qkv:
      p.transformer_layer_params_tpl.tr_atten_tpl.combined_qkv_proj_tpl.einsum_tpl = pax_fiddle.Config(
          fp8_ops.Fp8EinsumOp
      )
    p.transformer_layer_params_tpl.tr_fflayer_tpl.fflayer_tpl.linear_tpl.einsum_tpl = pax_fiddle.Config(
        fp8_ops.Fp8EinsumOp
    )
    p.moe_layer_tpl.einsum_tpl = pax_fiddle.Config(fp8_ops.Fp8EinsumOp)
    p.moe_layer_tpl.einsum_gated_tpl = pax_fiddle.Config(
        fp8_ops.Fp8EinsumGatedOp
    )
  return p


def GrokUniTransformerLmHParams(
    vocab_size,
    model_dim,
    ff_dim,
    attention_num_heads,
    attention_num_groups,
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
    model_type=LanguageModelType.CAUSAL,
    checkpoint_policy=AutodiffCheckpointType.SAVE_NOTHING,
    use_fp8=False,
) -> pax_fiddle.Config[transformer_models.TransformerLm]:
  """Common setup for Grok-1 Decoder-only Transformer Model.

  This function sets up configs for both MoE and dense Grok-1 models.

  1) The feedforward sublayer used gated gleu so there are two wi and one wo.
  2) Use bias in all projections and embeddings.
  3) Use no bias RMS norm for the layer norm.
  4) No relative attention bias.
  5) Add an optional z-loss to stablize final softmax logits.

  Args:
    vocab_size: Size of the vocabulary for LM.
    model_dim: model dimension.
    ff_dim: hidden dimension of feed-forward inner layer.
    attention_num_heads: number of attention heads.
    attention_num_groups: number of attention kv_heads.
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
  p = pax_fiddle.Config(transformer_models.TransformerLm)
  p.name = name
  p.packed_input = True
  p.model_dims = model_dim
  p.vocab_size = vocab_size
  p.position_emb_tpl = None
  p.model_type = model_type

  p.final_ln_tpl = pax_fiddle.Config(
      normalizations.RmsNorm, name='rms_norm', dim=model_dim
  )

  p.softmax_tpl = pax_fiddle.Config(
      embedding_softmax.GShardSharedEmbeddingSoftmax,
      name='emb',
      input_dims=model_dim,
      num_classes=vocab_size,
      z_loss_weight=z_loss_weight,
  )
  p.softmax_tpl.use_tgt_labels_size_as_loss_denominator = (
      use_tgt_labels_size_as_loss_denominator
  )
  grok_p = GrokStackedTransformerHParams(
      model_dim=model_dim,
      ff_dim=ff_dim,
      attention_num_heads=attention_num_heads,
      attention_num_groups=attention_num_groups,
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
      use_fp8=use_fp8,
  )
  num_blocks = num_transformer_layers

  if num_pipeline_stages == 1:
    p.stacked_transformer_tpl = pax_fiddle.Config(
        transformers.StackedTransformerRepeated,
        name='decoder',
        unroll_in_decode=True,
        block=grok_p,
        x_times=num_blocks,
        checkpoint_policy=checkpoint_policy,
    )
  else:
    assert num_blocks % num_pipeline_stages == 0
    grok_p.num_layers = num_transformer_layers // num_pipeline_stages
    grok_p.moe_layers = list(range(0, grok_p.num_layers))
    p.stacked_transformer_tpl = pax_fiddle.Config(
        transformers.PipelinedTransformer,
        pipeline_stage=grok_p,
        num_pipeline_stages=num_pipeline_stages,
        num_pipeline_microbatches=num_pipeline_microbatches,
        stream_io=True,
        checkpoint_policy=checkpoint_policy,
    )
  return p
