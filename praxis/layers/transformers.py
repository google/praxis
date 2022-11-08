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

"""Transformer-related layers."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

from absl import logging
import jax
from jax import numpy as jnp
from jax.ad_checkpoint import checkpoint_name
import numpy as np
from praxis import base_layer
from praxis import gshard_utils
from praxis import py_utils
from praxis import pytypes
from praxis.layers import activations as activations_lib
from praxis.layers import attentions
from praxis.layers import checkpoint_policy
from praxis.layers import linears
from praxis.layers import normalizations
from praxis.layers import pipeline
from praxis.layers import repeats
from praxis.layers import stats
from praxis.layers import stochastics

NestedMap = py_utils.NestedMap
WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams
sub_config_field = base_layer.sub_config_field

JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor

SplitDimsMapping = pytypes.SplitDimsMapping
BaseHParams = base_layer.BaseLayer.HParams
BaseWtShardingHParams = base_layer.BaseLayer.WeightShardingHParams
BaseActShardingHParams = base_layer.BaseLayer.ActivationShardingHParams
AutodiffCheckpointType = checkpoint_policy.AutodiffCheckpointType


def compute_attention_masks_for_fprop(
    inputs: JTensor,
    paddings: Optional[JTensor] = None,
    causal_attention: Optional[bool] = False,
    segment_mask: Optional[JTensor] = None,
    cross_inputs: Optional[JTensor] = None,
    cross_paddings: Optional[JTensor] = None,
    cross_segment_mask: Optional[JTensor] = None,
    fold_padding_with_segment_mask: Optional[bool] = False,
) -> Tuple[JTensor, Union[JTensor, None]]:
  """Computes attention mask from paddings, segment masks etc for fprop.

  Args:
    inputs: Input sequence JTensor of shape [B, T, H].
    paddings: Input paddings JTensor of shape [B, T] (optional). Note that one
      of paddings or segment_mask must be provided.
    causal_attention: Boolean to apply causal masking (optional).
    segment_mask: Segment mask JTensor for packed input of shape [B, 1, T, T]
      ready to add to logits (optional).
    cross_inputs: Output JTensor of the encoder, to be used for cross attention,
      of shape [B, S, H].
    cross_paddings: Paddings JTensor for cross atention of shape [B, S].
    cross_segment_mask: Segment mask JTensor for encoder-decoder in packed input
      case of shape [B, 1, T, S].
    fold_padding_with_segment_mask: If True then segment mask is supposed to
      include the padding mask as well, i.e. treating PADs as one sequence and
      non-PADs as another.

  Returns:
    attention_mask: Attention mask JTensor ready to add to logits for self
      attention of shape [1|B, 1, 1|T, T].
    cross_attention_mask: Attention mask ready to add to logits for cross
      attention of shape [1|B, 1, 1|T, S]. This will be None if cross_inputs
      are None.
  """
  if fold_padding_with_segment_mask:
    # In this case a separate padding mask is not needed, it is assumed
    # folded with segment mask.
    assert segment_mask is not None
    attention_mask = segment_mask
  else:
    # Paddings must be provided to create the attention mask
    assert paddings is not None
    # Get paddings mask to [B, 1, 1, T]
    attention_mask = attentions.convert_paddings_to_mask(paddings, inputs.dtype)

    # Additional segment_mask may also be provided in this case
    if segment_mask is not None:
      attention_mask = jnp.minimum(attention_mask, segment_mask)

  # Causal mask of shape [1, 1, T, T]
  if causal_attention:
    causal_mask = attentions.causal_mask(inputs)
    attention_mask = jnp.minimum(attention_mask, causal_mask)

  # Compute cross attention mask if applicable
  cross_attention_mask = None
  if cross_inputs is not None:
    assert cross_paddings is not None

    # Compute paddings
    cross_attention_mask = attentions.convert_paddings_to_mask(
        cross_paddings, dtype=cross_inputs.dtype)

    # Packed inputs
    if cross_segment_mask is not None:
      cross_attention_mask = jnp.minimum(cross_attention_mask,
                                         cross_segment_mask)
  return attention_mask, cross_attention_mask


def compute_attention_masks_for_extend_step(
    time_step: JTensor,
    seq_len: int,
    segment_mask: Optional[JTensor] = None,
    cross_paddings: Optional[JTensor] = None,
    cross_segment_mask: Optional[JTensor] = None
) -> Tuple[JTensor, Union[JTensor, None]]:
  """Computes attention mask from paddings, segment masks etc for extend_step.

  Args:
    time_step: Time step for which to generate causal mask.
    seq_len: Sequence length for generating causal mask.
    segment_mask: if not None, per step segment mask JTensor for this time step,
      of shape [B, 1, T].
    cross_paddings: Source paddings JTensor of shape [B, S].
    cross_segment_mask: if not None, cross_segment_mask JTensor for this time
      step, of shape [B, 1, S].

  Returns:
    attention_mask: Attention mask JTensor ready to add to logits for self
      attention of shape [1|B, 1, 1, T].
    cross_attention_mask: Attention mask JTensor ready to add to logits for
      cross attention of shape [1|B, 1, 1, S]. This will be None if
      cross_paddings are None.
  """
  # Create a broadcast friendly version of time step of shape [1, 1]
  batch_time_step = jnp.asarray(time_step, dtype=jnp.uint32)
  batch_time_step = jnp.reshape(batch_time_step, [1, 1])

  # Create causal padding by masking out any index > time_step.
  # [1, T], 0 for non-pad and 1 for pad.
  causal_padding = jnp.greater(
      jnp.expand_dims(jnp.arange(seq_len), 0), batch_time_step)

  # Create attention mask from padding of shape [1|B, 1, T]
  attention_mask = jnp.squeeze(
      attentions.convert_paddings_to_mask(causal_padding), axis=1)

  # Include segment mask, has shape [B, 1, T]
  if segment_mask is not None:
    attention_mask = jnp.minimum(attention_mask, segment_mask)

  # Compute cross attention mask if applicable
  cross_attention_mask = None
  if cross_paddings is not None:
    # Compute paddings mask [B, 1, 1, S]
    cross_attention_mask = attentions.convert_paddings_to_mask(
        cross_paddings, dtype=attention_mask.dtype)

    # Cross segment mask may be overloaded
    if cross_segment_mask is not None:
      # [B, 1, S] -> [B, 1, 1, S]
      cross_segment_mask = jnp.expand_dims(cross_segment_mask, axis=1)
      cross_attention_mask = jnp.minimum(cross_attention_mask,
                                         cross_segment_mask)
  return attention_mask, cross_attention_mask


def _get_sentence_embeddings(inputs: JTensor, segment_ids: JTensor) -> JTensor:
  """Returns the average sentence embedding to gate by.

  Args:
    inputs: Input sequence JTensor of the shape [S, D]
    segment_ids:  A JTensor of shape [S] specifying the segment each token
      belongs to.

  Returns:
    sentence_embeddings: A JTensor of the shape [S, D] that is an average of the
    input tensor per segment. The sentence embeddings will contain the same
    averaged vector for a particular segment_id.
  """
  # Max segments is required for segment_sum to be statically compiled.
  # max_segments = S
  max_segments = inputs.shape[0]
  # Compute the sum within segments(specified by segment_ids) of the input.
  # segment_sum shape: [max_segments, D]
  segment_sum = jax.ops.segment_sum(
      inputs, segment_ids, num_segments=max_segments)

  # Zero out the segment_sum belonging to segment_ids=0, which corresponds to
  # padding.
  segment_sum = segment_sum.at[0].set(0.0)

  # Compute the number of elements per segment. This is used to calculate the
  # mean.
  # num_elements_per_segment shape: [max_segments, D]
  num_elements_per_segment = jax.ops.segment_sum(
      jnp.ones_like(segment_ids), segment_ids, num_segments=max_segments)

  # segment_mean shape: [max_segments, D]
  segment_mean = segment_sum / jnp.maximum(
      num_elements_per_segment[:, jnp.newaxis], 1)
  # Sentence embedding contains the average of the input tensor per segment.
  # The sentence embeddings will contain the same averaged vector for a
  # particular segment_id.
  # sentence_embedding shape: [S, D]
  sentence_embeddings = segment_mean[segment_ids]
  return sentence_embeddings


class TransformerFeedForward(base_layer.BaseLayer):
  """Transformer feedforward layer with residual connection and dropout."""

  class HParams(BaseHParams):
    """Associated hyper-params for this layer class.

    Attributes:
      input_dims: Depth of the input.
      output_dims: Depth of the output. If unset or output_dims == input_dims,
        there is no residual projection layer. Otherwise, add a residual
        projection layer followed by batch normalization.
      hidden_dims: Hidden dimension of FFN.
      has_bias: Adds bias weights to Feedforward or not.
      apply_padding_first: Apply padding to inputs before everything else or
        not. For example, it is better to apply padding before batch norm.
      activation_tpl: Activation function to use.
      use_gated_activation: Boolean indicating whether to use a gated activation
        function for the second feedforward layer or not.
      fflayer_tpl: Parameterization of the feedforward layer.
      ln_tpl: Parameterization of the layer normalization layer. Other options
        include RmsNorm as well.
      residual_dropout_prob: Residual dropout.
      relu_dropout_tpl: Parameterization of the relu dropout layer. keep_prop
        will be reset to (1.0 - relu_dropout_prob).
      relu_dropout_prob: FFN dropout.
      residual_dropout_tpl: Parameterization of the residual dropout params
        template. keep_prop will be reset to (1.0 - residual_dropout_prob).
      add_skip_connection: Whether to add residual connection.
      residual_weight: Weight of the residual connection. Output = fn(x) *
        residual_weight + x.
      residual_droppath_prob: Probability at which we drop the entire residual
        path.
      norm_policy: Policy for applying normaliztion wrt. transformations.
        Options are: (1) "pre", applied before transformation. (2)
        "primer_hybrid", applied before and after transformation. (3) "post",
        applied after transformation.
      internal_gshard_variance_scaling_fan_in_init: Feedforward weight init
        follows uniform distribution withbound = 1.0 / sqrt(3 / dim_0).
    """
    input_dims: int = 0
    output_dims: int = 0
    hidden_dims: int = 0
    has_bias: bool = True
    apply_padding_first: bool = False
    activation_tpl: activations_lib.BaseActivation.HParams = sub_config_field(
        activations_lib.ReLU.HParams)
    use_gated_activation: bool = False
    fflayer_tpl: BaseHParams = sub_config_field(linears.FeedForward.HParams)
    ln_tpl: BaseHParams = sub_config_field(normalizations.LayerNorm.HParams)
    residual_dropout_prob: float = 0.0
    relu_dropout_tpl: BaseHParams = sub_config_field(
        stochastics.Dropout.HParams)
    relu_dropout_prob: float = 0.0
    residual_dropout_tpl: BaseHParams = sub_config_field(
        stochastics.Dropout.HParams)
    add_skip_connection: bool = True
    residual_weight: float = 1.0
    residual_droppath_prob: float = 0.0
    norm_policy: str = 'pre'
    internal_gshard_variance_scaling_fan_in_init: bool = False

  class WeightShardingHParams(BaseWtShardingHParams):
    """Represents how layer's learned parameters are partitioned across a mesh.

    Attributes:
      ffn0: Weight-split dims mapping for the first ffw network.
      ffn1: Weight-split dims mapping for the second ffw network.
    """
    ffn0: SplitDimsMapping = None
    ffn1: SplitDimsMapping = None

  class ActivationShardingHParams(BaseActShardingHParams):
    """Represents how intermediate values should be partitioned across a mesh.

    Attributes:
      ffn0: Activation-split dims mapping for the first ffw network.
      ffn1: Activation-split dims mapping for the second ffw network.
    """
    ffn0: SplitDimsMapping = None
    ffn1: SplitDimsMapping = None

  def setup(self) -> None:
    p = self.hparams

    output_dims = p.output_dims
    if output_dims == 0:
      # Make it compatible with previous implementation
      output_dims = p.input_dims
    else:
      assert output_dims == p.input_dims, (p.input_dims, output_dims)

    wp = p.weight_split_dims_mapping
    ap = p.activation_split_dims_mapping
    # Create Layer Norm
    if p.norm_policy == 'primer_hybrid':
      ln_p = p.ln_tpl.clone()
      ln_p.dim = p.input_dims
      self.create_child('pre_layer_norm', ln_p)
      self.create_child('post_layer_norm', ln_p)
    elif p.norm_policy == 'pre' or p.norm_policy == 'post':
      ln_p = p.ln_tpl.clone()
      ln_p.name = 'fflayer_ln'
      ln_p.dim = p.input_dims
      self.create_child('layer_norm', ln_p)
    else:
      raise ValueError('Unrecognized norm_policy: %s' % p.norm_policy)

    self._is_ffn1_gated = p.use_gated_activation
    if self._is_ffn1_gated:
      activation = activations_lib.Identity.HParams()
      gate_activation = p.activation_tpl.clone()
    else:
      activation = p.activation_tpl.clone()
      gate_activation = None

    # Create the first Feedforward layer mapping to hidden dims
    ffn1_p = p.fflayer_tpl.clone()
    ffn1_p.name = 'ffn_layer1'
    ffn1_p.input_dims = p.input_dims
    ffn1_p.has_bias = p.has_bias
    ffn1_p.activation_tpl = activation
    ffn1_p.output_dims = p.hidden_dims
    ffn1_p.weight_split_dims_mapping.wt = wp.ffn0
    ffn1_p.activation_split_dims_mapping.out = ap.ffn0
    if p.internal_gshard_variance_scaling_fan_in_init:
      scale = (1. / p.input_dims)**0.5 * (3.0**0.5)
      ffn1_p.linear_tpl.params_init = WeightInit.Uniform(scale)
    self.create_child('ffn_layer1', ffn1_p)

    if self._is_ffn1_gated:
      # This is a gated ffw network, corresponding to gshard_builder's wi0
      gate_p = p.fflayer_tpl.clone()
      gate_p.name = 'ffn_layer1_gate'
      gate_p.input_dims = p.input_dims
      gate_p.has_bias = p.has_bias
      gate_p.activation_tpl = gate_activation
      gate_p.output_dims = p.hidden_dims
      gate_p.weight_split_dims_mapping.wt = wp.ffn0
      gate_p.activation_split_dims_mapping.out = ap.ffn0
      if p.internal_gshard_variance_scaling_fan_in_init:
        scale = (1. / p.input_dims)**0.5 * (3.0**0.5)
        gate_p.linear_tpl.params_init = WeightInit.Uniform(scale)
      self.create_child('ffn_layer1_gate', gate_p)

    # Create RELU dropout layer
    relu_dropout_p = p.relu_dropout_tpl.clone()
    relu_dropout_p.keep_prob = 1.0 - p.relu_dropout_prob
    self.create_child('relu_dropout', relu_dropout_p)

    # Create the second Feedforward layer mapping to input dims
    ffn2_p = p.fflayer_tpl.clone()
    ffn2_p.name = 'ffn_layer2'
    ffn2_p.input_dims = p.hidden_dims
    ffn2_p.has_bias = p.has_bias
    ffn2_p.activation_tpl = activations_lib.Identity.HParams()
    ffn2_p.output_dims = output_dims
    ffn2_p.weight_split_dims_mapping.wt = wp.ffn1
    ffn2_p.activation_split_dims_mapping.out = ap.ffn1
    if p.internal_gshard_variance_scaling_fan_in_init:
      scale = (1. / p.hidden_dims)**0.5 * (3.0**0.5)
      ffn2_p.linear_tpl.params_init = WeightInit.Uniform(scale)
    self.create_child('ffn_layer2', ffn2_p)

    # Create residual dropout layer
    residual_dropout_p = p.residual_dropout_tpl.clone()
    residual_dropout_p.keep_prob = 1.0 - p.residual_dropout_prob
    self.create_child('residual_dropout', residual_dropout_p)

    if p.residual_droppath_prob > 0:
      assert p.add_skip_connection
      droppath_p = stochastics.StochasticResidual.HParams(
          name='residual_droppath',
          survival_prob=1.0 - p.residual_droppath_prob)
      self.create_child('residual_droppath', droppath_p)

  def __call__(self,
               inputs: JTensor,
               paddings: Optional[JTensor] = None,
               segment_ids: Optional[JTensor] = None) -> JTensor:
    p = self.hparams
    # Expand paddings to last dim if not None to have shape [batch, time, 1]
    if paddings is not None:
      paddings = jnp.expand_dims(paddings, axis=-1)

    if p.apply_padding_first and paddings is not None:
      inputs *= (1.0 - paddings)

    if p.norm_policy == 'primer_hybrid':
      inputs_normalized = self.pre_layer_norm(inputs)
    elif p.norm_policy == 'pre':
      inputs_normalized = self.layer_norm(inputs)
    else:
      inputs_normalized = inputs

    # Apply first FFN layer
    if self._is_ffn1_gated:
      # theta.ffn_layer1_gate corresponds to gshard_builder's wi0
      gate_value = self.ffn_layer1_gate(inputs_normalized)
      # theta.ffn_layer1 corresponds to gshard_builder's wi1
      projected_inputs = gate_value * self.ffn_layer1(inputs_normalized)
    else:
      projected_inputs = self.ffn_layer1(inputs_normalized)
      projected_inputs = checkpoint_name(projected_inputs, 'ffn1')

    # Apply paddings if not None
    if not p.apply_padding_first and paddings is not None:
      projected_inputs *= (1.0 - paddings)

    # Apply RELU dropout
    projected_inputs = self.relu_dropout(projected_inputs)

    # Apply second FFN layer
    projected_inputs = self.ffn_layer2(projected_inputs)
    projected_inputs = checkpoint_name(projected_inputs, 'ffn2')

    # Apply paddings if not None
    if not p.apply_padding_first and paddings is not None:
      projected_inputs *= (1.0 - paddings)

    # Apply Primer normalization before dropout.
    if p.norm_policy == 'primer_hybrid':
      projected_inputs = self.post_layer_norm(projected_inputs)
    elif p.norm_policy == 'post':
      projected_inputs = self.layer_norm(projected_inputs)

    # Apply residual dropout
    projected_inputs = self.residual_dropout(projected_inputs)

    # Apply skip connection
    if p.add_skip_connection:
      if p.residual_droppath_prob:
        projected_inputs = self.residual_droppath(inputs, projected_inputs)
      else:
        projected_inputs = inputs + projected_inputs * p.residual_weight

    return projected_inputs


class TransformerFeedForwardMoe(base_layer.BaseLayer):
  """A sharded MoE Layer.

  This is a drop-in replacement of the transformer feedforward layer. It is a
  composite of the following sub-layers.

  ln_inputs = ln(inputs)
  moe_output = moe(ln_inputs)
  drop_output = dropout(moe_output)
  output = inputs + drop_output
  """

  class HParams(BaseHParams):
    """Associated hyper-params for this layer class.

    Attributes:
      input_dims: Dimension of the layer input.
      hidden_dims: Dimension of the hidden layer.
      apply_padding_first: Apply padding to inputs before everything else or
        not. For example, it is better to apply padding before batch norm.
      ln_tpl: Parameterization of the layer normalization layer.
      activation_tpl: Activation function to use.
      relu_dropout_tpl: Parameterization of the relu dropout layer. keep_prop
        will be reset to (1.0 - relu_dropout_prob).
      relu_dropout_prob: Probability at which we apply dropout to the hidden
        layer of feedforward network..
      residual_dropout_tpl: Parameterization of the residual dropout params
        template. keep_prop will be reset to (1.0 - residual_dropout_prob).
      residual_dropout_prob: Probability at which we apply dropout to the
        residual layers, such that, residual(x, y) = (x + dropout(y)).
      add_skip_connection: If True, add skip_connection from input to output.
      residual_weight: Weight applied on residual connection. Final output is
        residual_weight * residual_fn(x) + x. Only in effect when
        add_skip_connection is True.
      norm_policy: Policy for applying normaliztion wrt. transformations.
        Options are: (1) "pre", applied before transformation. (2)
        "primer_hybrid", applied before and after transformation. (3) "post",
        applied after transformation.
      residual_droppath_prob: Probability at which we drop the entire residual
        path.
      gating_func: Gating function type--can be one of the following options:
        'top2', based on the GShard paper: https://arxiv.org/abs/2006.16668,
        'expert_choice', based on https://arxiv.org/abs/2202.09368,
        'dense_top2': experimental gating function for decodiing. Similar to
        'top2' gating, but no capacity constrainst for each expert.
      num_experts: Total number of experts in this layer.
      num_groups: Total number of groups for dispatching. num_groups typically
        should be the same as num devices.
      min_group_size: If not None, num_groups will be adjusted so that there
        will be at least min_group_size tokens in each group.
      expert_capacity_dim: Internal. Exact expert capacity. Setting non-zero
        unadjusted_expert_capacity_factor is a preferred way.
      unadjusted_expert_capacity_factor: Expert capacity factor. This is the
        ratio between global batch size and total capacity across all experts
        and all routing groups. If the global batch size is G*S (num_groups*
        group_size) or B*L(batch*length) and the total expert capacity across
        all routing groups is E*G*C (num_experts*num_groups*expert_capacity),
        then unadjusted_expert_capacity_factor == (E*G*C)/(G*S)
        unadjusted_expert_capacity_factor is set to 2 by default for top-2
        routing.
      expert_weight_shards: Shard each expert params into this many number of
        shards to reduce the size of individual weight params.
      second_expert_policy: How to pick second expert: all, sampling or random.
      internal_gshard_variance_scaling_fan_in_init: Internal. Do not use. To
        study MoE layer init.
      moe_load_balance_loss_weight: Weight for the load balancing loss of the
        MoE layer.
      gating_logit_cap:  Cap the absolute values of MoE gating logits by tanh.
        Enabled when a positive value is specified.
      moe_gating_embedding_level: Specifies the type of MOE gating embedding
      used.
      See Section 3.1 https://arxiv.org/pdf/2110.03742.pdf.
      Options are:
        (1) "token" -> The gating function uses tokens to route.
        (2) "sentence" -> The gating function uses the sentence embeddings,
            calculated by taking the average token representation, to route.
    """
    input_dims: int = 0
    hidden_dims: int = 0
    apply_padding_first: bool = False
    ln_tpl: BaseHParams = sub_config_field(normalizations.LayerNorm.HParams)
    activation_tpl: activations_lib.BaseActivation.HParams = sub_config_field(
        activations_lib.ReLU.HParams)
    relu_dropout_tpl: BaseHParams = sub_config_field(
        stochastics.Dropout.HParams)
    relu_dropout_prob: float = 0.0
    residual_dropout_tpl: BaseHParams = sub_config_field(
        stochastics.Dropout.HParams)
    residual_dropout_prob: float = 0.0
    add_skip_connection: bool = True
    residual_weight: float = 1.0
    norm_policy: str = 'pre'
    residual_droppath_prob: float = 0.0
    gating_func: str = 'top2'
    num_experts: int = 0
    num_groups: int = 0
    min_group_size: Optional[int] = None
    expert_capacity_dim: int = 0
    unadjusted_expert_capacity_factor: float = 2.0
    expert_weight_shards: int = 1
    second_expert_policy: str = 'all'
    internal_gshard_variance_scaling_fan_in_init: bool = True
    moe_load_balance_loss_weight: float = 1.0
    gating_logit_cap: float = 0.0
    moe_gating_embedding_level: str = 'token'

  # SPMD partition related params.
  # M - model_dim, for both inputs and outputs
  # E - experts dim
  # G - groups dim
  # C - experts capacity dim
  # H - hidden dim
  # S - sequence dim

  class WeightShardingHParams(BaseWtShardingHParams):
    """Represents how layer's learned parameters are partitioned across a mesh.

    Attributes:
      me: Sharding for the gating network weight, of shape [input_dims,
        num_experts].
      emh: Sharding of the first projection matrix that maps from input to
        hidden dim, of shape [num_experts, input_dims, hidden_dims].
      ehm: Sharding of the second projection matrix that maps from hidden to
        output dim, of shape [num_experts, hidden_dims, output_dims].
    """
    me: SplitDimsMapping = None
    emh: SplitDimsMapping = None
    ehm: SplitDimsMapping = None

  class ActivationShardingHParams(BaseActShardingHParams):
    """Represents how intermediate values should be partitioned across a mesh.

    Attributes:
      gsm: Sharding of the gsm tensors.
      gs: Sharding of the gs tensors.
      gsec: Sharding of the gsec tensors.
      egcm: Sharding of the egcm tensors.
      gecs: Sharding of the gecs tensors.
      gec: Sharding of the gec tensors.
      egch: Sharding of the egch tensors.
      gecm: Sharding of the gecm tensors.
    """
    gsm: SplitDimsMapping = None
    gs: SplitDimsMapping = None
    gsec: SplitDimsMapping = None
    gecs: SplitDimsMapping = None
    gec: SplitDimsMapping = None
    egcm: SplitDimsMapping = None
    egch: SplitDimsMapping = None
    gecm: SplitDimsMapping = None

  def setup(self) -> None:
    p = self.hparams
    assert p.name
    assert p.input_dims
    assert p.hidden_dims

    assert (p.unadjusted_expert_capacity_factor or p.expert_capacity_dim)
    assert p.num_experts > 0
    assert p.num_groups > 0
    assert p.expert_weight_shards == 1, (
        f'[Deprecated] Should be removed {p.expert_weight_shards} != 1')

    if p.norm_policy == 'primer_hybrid':
      params = p.ln_tpl.clone()
      params.dim = p.input_dims
      self.create_child('pre_layer_norm', params)
      self.create_child('post_layer_norm', params)
    elif p.norm_policy == 'pre' or p.norm_policy == 'post':
      params = p.ln_tpl.clone()
      params.name = 'layer_norm'
      params.dim = p.input_dims
      self.create_child('layer_norm', params)
    else:
      raise ValueError('Unrecognized norm_policy: %s' % p.norm_policy)

    dropout_tpl = p.residual_dropout_tpl.clone()
    dropout_tpl.keep_prob = (1.0 - p.residual_dropout_prob)
    self.create_child('residual_dropout', dropout_tpl)

    dropout_tpl = p.relu_dropout_tpl.clone()
    dropout_tpl.keep_prob = (1.0 - p.relu_dropout_prob)
    self.create_child('relu_dropout', dropout_tpl)

    if p.residual_droppath_prob > 0:
      assert p.add_skip_connection
      droppath_p = stochastics.StochasticResidual.HParams(
          name='residual_droppath',
          survival_prob=1.0 - p.residual_droppath_prob)
      self.create_child('residual_droppath', droppath_p)

    self.create_child('activation', p.activation_tpl.clone())

    # Assume output_dims == input_dims
    output_dims = p.input_dims

    # First create the gating network.
    wp = p.weight_split_dims_mapping
    gate_init = None  # default xavier init
    if p.internal_gshard_variance_scaling_fan_in_init:
      # TODO(lepikhin): this init is related with Adafactor settings, study
      stddev = (1.0 / p.input_dims)**0.5
      gate_scale = stddev * 3.0**0.5
      gate_init = WeightInit.Uniform(gate_scale)
    gate_pc = WeightHParams(
        shape=[p.input_dims, p.num_experts],
        init=gate_init,
        mesh_shape=p.mesh_shape,
        tensor_split_dims_mapping=wp.me)
    logging.debug('moe gate WeightHParams %s', gate_pc)
    self.create_variable('gate', gate_pc)

    # Next create the expert network.
    # Params initialization follows gshard_builder.py.
    # emh tensor typically mesh-shard on first dim and last dim. Hence, here we
    # split the tensor manually into multiple tensors on the second dim.
    emh_shape = [
        p.num_experts, p.input_dims // p.expert_weight_shards, p.hidden_dims
    ]
    wi_init = None
    if p.internal_gshard_variance_scaling_fan_in_init:
      stddev = (1.0 / p.input_dims)**0.5
      wi_init_scale = stddev * 3.0**0.5
      wi_init = WeightInit.Uniform(wi_init_scale)
    wi_pc = WeightHParams(
        shape=emh_shape,
        init=wi_init,
        mesh_shape=p.mesh_shape,
        tensor_split_dims_mapping=wp.emh)
    logging.debug('moe wi WeightHParams %s', wi_pc)
    for ii in range(p.expert_weight_shards):
      self.create_variable('wi_%d' % ii, wi_pc)

    # EHM Tensor (output transformation after RELU)
    # ehm tensor typically shard on the first dim and the second dim. Here we
    # manually split the tensor on the last dim into multiple tensors.
    ehm_shape = [
        p.num_experts, p.hidden_dims, output_dims // p.expert_weight_shards
    ]
    wo_init = None
    if p.internal_gshard_variance_scaling_fan_in_init:
      wi_init = None
      stddev = (1.0 / p.hidden_dims)**0.5
      wo_init_scale = stddev * 3.0**0.5
      wo_init = WeightInit.Uniform(wo_init_scale)
    wo_pc = WeightHParams(
        shape=ehm_shape,
        init=wo_init,
        mesh_shape=p.mesh_shape,
        tensor_split_dims_mapping=wp.ehm)
    logging.debug('moe wo WeightHParams %s', wo_pc)
    for ii in range(p.expert_weight_shards):
      self.create_variable('wo_%d' % ii, wo_pc)

  def _split(self, t_in, sharding):
    return base_layer.maybe_shard(t_in, sharding, self.hparams.mesh_axis_names)

  def _get_weights(self):
    """Get the expert weights."""
    theta_wis = []
    theta_wos = []
    for ii in range(self.hparams.expert_weight_shards):
      theta_wis.append(getattr(self.theta, f'wi_{ii}'))
      theta_wos.append(getattr(self.theta, f'wo_{ii}'))

    # Concatenate theta_wis and theta_wos
    # since each sub-theta_wi has shape
    # (p.num_experts, p.input_dims // p.expert_weight_shards, p.hidden_dims)
    # and each sub-theta_wo has shape
    # (p.num_experts, p.hidden_dims, output_dims // p.expert_weight_shards)
    if len(theta_wis) == 1:
      theta_wi = theta_wis[0]
    else:
      # new shape: (p.num_experts, p.input_dims, p.hidden_dims)
      theta_wi = jnp.concatenate(theta_wis, 1)

    if len(theta_wos) == 1:
      theta_wo = theta_wos[0]
    else:
      # new shape: (p.num_experts, p.hidden_dims, output_dims)
      theta_wo = jnp.concatenate(theta_wos, 2)
    return theta_wi, theta_wo

  def _combine_top2_expert_outputs(self, inputs, paddings, segment_ids):
    """Combine outputs from top 2 experts directly."""
    p = self.hparams
    fprop_dtype = self.fprop_dtype
    ap = self.hparams.activation_split_dims_mapping

    theta_wi, theta_wo = self._get_weights()
    if p.moe_gating_embedding_level == 'sentence':
      if segment_ids is None and paddings is None:
        sentence_embeddings = jnp.tile(
            jnp.average(inputs, axis=1, keepdims=True), [1, inputs.shape[1], 1])
      else:
        if segment_ids is None:
          segment_ids = jnp.asarray(1 - paddings, jnp.int32)
        sentence_embeddings_fn = jax.vmap(_get_sentence_embeddings, (0, 0), 0)
        sentence_embeddings = sentence_embeddings_fn(inputs, segment_ids)
      logits = jnp.einsum('gsm,me->gse', sentence_embeddings, self.theta.gate)
    else:
      logits = jnp.einsum('gsm,me->gse', inputs, self.theta.gate)
    # gsm -> gse tensor
    raw_gates = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
    if raw_gates.dtype != fprop_dtype:
      raw_gates = raw_gates.astype(fprop_dtype)

    # When picking top-1 or top-2
    raw_gates_max = jnp.max(raw_gates, axis=-1, keepdims=True)
    raw_gates_thr = jnp.max(
        raw_gates * jnp.less(raw_gates, raw_gates_max).astype(fprop_dtype),
        axis=-1,
        keepdims=True)

    gates = raw_gates * jnp.greater_equal(raw_gates,
                                          raw_gates_thr).astype(fprop_dtype)

    gates_denom = jnp.sum(gates, axis=-1, keepdims=True) + 1e-12
    gates /= gates_denom

    hidden = jnp.einsum('gsm,emh->egsh', inputs, theta_wi)
    # Activation function.
    hidden = self.activation(hidden)
    # Dropout.
    hidden = self.relu_dropout(hidden)
    # Output.
    expert_output = jnp.einsum('egsh,ehm->egsm', hidden, theta_wo)
    combined_output = jnp.einsum('egsm,gse->gsm', expert_output, gates)
    combined_output = self._split(combined_output, ap.gsm)
    aux_loss = jnp.array(0.0)
    return combined_output, aux_loss

  def _dispatch_and_combine_expert_outputs(self, inputs, paddings, segment_ids):
    """Combine expert outputs using GShard-style dispatch-combine tensors."""
    p = self.hparams
    fprop_dtype = self.fprop_dtype
    ap = p.activation_split_dims_mapping
    output_dims = p.input_dims
    assert p.gating_func != 'dense_top2'

    theta_wi, theta_wo = self._get_weights()

    token_shape = inputs.shape[:-1]
    num_tokens = np.prod(token_shape)
    m_dim = inputs.shape[-1]
    if paddings is not None:
      assert paddings.shape == token_shape

    num_groups = p.num_groups
    assert num_groups
    if (p.min_group_size is not None and
        num_tokens / num_groups < p.min_group_size):
      num_groups = (num_tokens + p.min_group_size - 1) // p.min_group_size
      logging.info('num_groups adjusted to %s.', num_groups)
    if num_tokens % num_groups != 0:
      raise ValueError(f'The value of num_groups {num_groups} does not '
                       f'evenly divide the value of num_tokens {num_tokens}')
    g_len = num_tokens // num_groups

    reshaped_inputs = inputs.reshape([num_groups, g_len, m_dim])
    reshaped_inputs = self._split(reshaped_inputs, ap.gsm)
    if paddings is not None:
      reshaped_paddings = paddings.reshape([num_groups, g_len])
      reshaped_paddings = self._split(reshaped_paddings, ap.gs)
      reshaped_paddings = reshaped_paddings.astype(fprop_dtype)
    else:
      reshaped_paddings = None
    if p.moe_gating_embedding_level == 'sentence':
      if segment_ids is None and paddings is None:
        sentence_embeddings = jnp.tile(
            jnp.average(inputs, axis=1, keepdims=True), [1, inputs.shape[1], 1])
      else:
        if segment_ids is None:
          segment_ids = jnp.asarray(1 - paddings, jnp.int32)
        # It is important that the segment_ids have shape [B, S, D] and are not
        # reshaped to [G, B*S / G, D] when calculating the sentence embeddings.
        # The vmap below assumes that it is operating on a batch.
        sentence_embeddings_fn = jax.vmap(_get_sentence_embeddings, (0, 0), 0)
        sentence_embeddings = sentence_embeddings_fn(inputs, segment_ids)
      # Reshape the sentence embeddings, so that it corresponds to sharding
      # groups.
      reshaped_sentence_embeddings = sentence_embeddings.reshape(
          [num_groups, g_len, m_dim])
      reshaped_sentence_embeddings = self._split(reshaped_sentence_embeddings,
                                                 ap.gsm)
      logits = jnp.einsum('gsm,me->gse', reshaped_sentence_embeddings,
                          self.theta.gate)
    else:
      logits = jnp.einsum('gsm,me->gse', reshaped_inputs, self.theta.gate)

    # Here and below, we assume num devices equals num groups.
    # TODO(yonghui): Expose some of the options below through params.
    # NOTE(yonghui): The following code might break during beam search decode
    # due to much smaller group size.
    # TODO(yonghui): Avoid explicitly casting everything to fp32 once
    # top2_gating_on_logits is stable in low-precision mode.
    # TODO(lepikhin): Validate stability. mask_dtype=np.int32 and
    # logits.astype(np.float32) should generally be sufficient.
    if p.second_expert_policy == 'all':
      prng_key = None
    else:
      prng_key = self.next_prng_key()
    gating = gshard_utils.compute_gating(
        paddings=reshaped_paddings,
        logits=logits.astype(jnp.float32),
        experts_dim=p.num_experts,
        expert_capacity_dim=p.expert_capacity_dim,
        fprop_dtype=fprop_dtype,
        gating_func=p.gating_func,
        prng_key=prng_key,
        second_expert_policy=p.second_expert_policy,
        second_expert_threshold=0.0,
        legacy_mtf_behavior=True,
        capacity_factor=p.unadjusted_expert_capacity_factor,
        mask_dtype=jnp.int32,
        gating_logit_cap=p.gating_logit_cap)

    if p.gating_func == 'top2':
      aux_loss, combine_tensor, dispatch_tensor, summary = gating
      over_capacity_1_ratio, over_capacity_2_ratio = summary
      self.add_summary('over_capacity_1_ratio', over_capacity_1_ratio)
      self.add_summary('over_capacity_2_ratio', over_capacity_2_ratio)
    else:
      aux_loss, combine_tensor, dispatch_tensor = gating
    if fprop_dtype != np.float32:
      combine_tensor = combine_tensor.astype(fprop_dtype)
      dispatch_tensor = dispatch_tensor.astype(fprop_dtype)

    # both tensors have shape [g, s, e, c]
    if p.gating_func in ['top2', 'expert_choice_v2']:
      combine_tensor = self._split(combine_tensor, ap.gsec)
      dispatch_tensor = self._split(dispatch_tensor, ap.gsec)
      expert_inputs = jnp.einsum('gsec,gsm->egcm', dispatch_tensor,
                                 reshaped_inputs)
    elif p.gating_func == 'expert_choice':
      combine_tensor = self._split(combine_tensor, ap.gec)
      dispatch_tensor = self._split(dispatch_tensor, ap.gecs)
      expert_inputs = jnp.einsum('gecs,gsm->egcm', dispatch_tensor,
                                 reshaped_inputs)
    else:
      raise ValueError('Unsupported gating function: %s ' % p.gating_func)
    expert_inputs = self._split(expert_inputs, ap.egcm)

    hidden = jnp.einsum('egcm,emh->egch', expert_inputs, theta_wi)
    hidden = self._split(hidden, ap.egch)

    if p.gating_func in ['top2', 'expert_choice_v2']:
      threshold = 0
      activation_class_name = p.activation_tpl.cls.__name__
      if isinstance(p.activation_tpl.cls, activations_lib.GELU):
        logging.info('Setting dead neuron count threshold=-3.0 '
                     'for approximate GeLU activation')
        threshold = -3.0

      nonpadding_indicator = jnp.einsum('gsec->ec', dispatch_tensor)
      nonpadding_indicator = nonpadding_indicator[:, jnp.newaxis, :,
                                                  jnp.newaxis]
      padding_indicator = 1 - nonpadding_indicator
      hidden_minus_ten_padding_indicator = hidden - 10 * padding_indicator
      # EG, taking max over G and C dim
      max_hidden = jnp.max(
          jnp.max(hidden_minus_ten_padding_indicator, axis=1), axis=1)
      dead_neuron_indicator = jnp.less(max_hidden, threshold).astype(jnp.int32)
      dead_neuron_count = jnp.count_nonzero(dead_neuron_indicator)
      self.add_summary('dead_%s_count' % activation_class_name,
                       dead_neuron_count)

    # Activation function.
    hidden = self.activation(hidden)
    # Dropout.
    hidden = self.relu_dropout(hidden)
    # Output.
    expert_output = jnp.einsum('egch,ehm->egcm', hidden, theta_wo)
    expert_output = self._split(expert_output, ap.egcm)
    # Now transpose and reshard.
    transposed_expert_output = jnp.einsum('egcm->gecm', expert_output)
    transposed_expert_output = self._split(transposed_expert_output, ap.gecm)
    if p.gating_func in ['top2', 'expert_choice_v2']:
      combined_output = jnp.einsum('gecm,gsec->gsm', transposed_expert_output,
                                   combine_tensor)
    elif p.gating_func == 'expert_choice':
      combined_output = jnp.einsum('gecm,gecs,gec->gsm',
                                   transposed_expert_output, dispatch_tensor,
                                   combine_tensor)
    else:
      raise ValueError('Unsupported gating function: %s ' % p.gating_func)
    combined_output = self._split(combined_output, ap.gsm)

    combined_output = combined_output.reshape(token_shape + (output_dims,))
    return combined_output, aux_loss

  def __call__(self,
               inputs: JTensor,
               paddings: JTensor = None,
               segment_ids: JTensor = None) -> JTensor:
    """Layer-norm, route, feed-forward, combine, residual.

    Args:
      inputs: [batch, seq_len, model].
      paddings: [batch, seq_len], optional when called by extend_step.
      segment_ids: [batch, seq_len] Optional. Segment_ids is used when
        moe_gating_embedding_level == 'sentence'.

    Returns:
      Tensor of the same shape as inputs.
    """
    p = self.hparams
    # Assume output_dims == input_dims
    fprop_dtype = self.fprop_dtype

    # Consistent with gshard implementation.
    if p.apply_padding_first and paddings is not None:
      inputs *= (1.0 - jnp.expand_dims(paddings, axis=-1))

    # TODO(zhangqiaorjc): Handle input of shape [batch, seq_len, g, model/g]?
    if p.norm_policy == 'primer_hybrid':
      inputs_normalized = self.pre_layer_norm(inputs)
    elif p.norm_policy == 'pre':
      inputs_normalized = self.layer_norm(inputs)
    else:
      inputs_normalized = inputs

    assert len(inputs_normalized.shape) in [2, 3]

    if p.gating_func == 'dense_top2':
      combined_output, aux_loss = self._combine_top2_expert_outputs(
          inputs_normalized, paddings, segment_ids)
    else:
      combined_output, aux_loss = self._dispatch_and_combine_expert_outputs(
          inputs_normalized, paddings, segment_ids)

    # Apply padding.
    if paddings is not None:
      combined_output *= (1.0 -
                          jnp.expand_dims(paddings, -1)).astype(fprop_dtype)
    # Primer normalization before dropout.
    if p.norm_policy == 'primer_hybrid':
      combined_output = self.post_layer_norm(combined_output)
    elif p.norm_policy == 'post':
      combined_output = self.layer_norm(combined_output)
    # Residual dropout.
    after_residual = self.residual_dropout(combined_output)
    if p.add_skip_connection:
      if p.residual_droppath_prob:
        out = self.residual_droppath(inputs, after_residual)
      else:
        out = inputs + after_residual * p.residual_weight

    # Add loss to a global collection. We don't return the loss to the caller
    # to avoid the change of the api here.
    assert p.moe_load_balance_loss_weight, (
        'p.moe_load_balance_loss_weight > 0 when there is an aux '
        'load balancing loss in MoE layers.')
    aux_loss = aux_loss.astype(fprop_dtype)
    aux_loss *= p.moe_load_balance_loss_weight
    self.add_summary('aux_moe_load_balance_loss', aux_loss)
    self.add_aux_loss('aux_moe_load_balance_loss', aux_loss)

    return out


class Transformer(base_layer.BaseLayer):
  """Transformer layer with multi-headed attention."""

  class HParams(BaseHParams):
    """Associated hyper-params for this layer class.

    Attributes:
      input_dims: Dimension of the transformer block input.
      hidden_dims: Hidden dimension of FFN layer.
      num_heads: Number of heads in self-attention.
      dim_per_head: Dimension of each attention head. If None then dim_per_head
        == hidden_dim // num_heads.
      dropout_tpl: Residual dropout params template. keep_prop will be reset to
        (1.0 - residual_dropout_prob).
      atten_dropout_prob: Probability at which we apply dropout to the attention
        weights.
      residual_dropout_prob: Probability at which we apply dropout to the
        residual layers, such that, residual(x, y) = (x + dropout(y)).
      relu_dropout_prob: Probability at which we apply dropout to the FFN
        layers.
      residual_droppath_prob: Probability at which we drop the entire residual
        path.
      mask_self_attention: If True, use causal mask.
      cross_attention: If True, perform cross encoder-decoder attention.
      allow_skip_cross_attention: If True, allow skipping cross attention during
        forward pass and decoding. This allows to skip cross attention when
        cross inputs are not available. For example, if we want to train the
        model with paired data and unimodal data. For paired data, we need cross
        attention but for unimodal data, we don't have cross inputs.
      cross_atten_tpl: Optional cross attention params template that can be set
        when cross attention is enabled. If cross-attention is enabled and this
        is set to None, then cross-attention params will be inherited from
        tr_atten_tpl.
      ln_tpl: Parameterization of the layer normalization layer.
      norm_policy: Policy for applying normaliztion wrt. transformations.
        Options are: (1) "pre", applied before transformation. (2)
        "primer_hybrid", applied before and after transformation. (3) "post",
        applied after transformation.
      tr_atten_tpl: Parameterization of the DotProductAttention layer.
      packed_input: If True, each training example may pack multiple sequences.
      tr_fflayer_tpl: Parameterization of the transformer Feed-Forward Layer.
      ngrammer_tpl: Params for the Ngrammer layer. This param must correspond to
        the VQNgrammer layer. If this is None, then there is no NGrammer layer
        present in this layer.
    """
    input_dims: int = 0
    hidden_dims: int = 0
    num_heads: Optional[int] = None
    dim_per_head: Optional[int] = None
    dropout_tpl: BaseHParams = sub_config_field(stochastics.Dropout.HParams)
    atten_dropout_prob: float = 0.0
    residual_dropout_prob: float = 0.0
    relu_dropout_prob: float = 0.0
    residual_droppath_prob: float = 0.0
    mask_self_attention: bool = False
    use_cross_attention: bool = False
    allow_skip_cross_attention: bool = False
    cross_atten_tpl: Optional[BaseHParams] = base_layer.sub_config_field(None)
    ln_tpl: BaseHParams = sub_config_field(normalizations.LayerNorm.HParams)
    norm_policy: str = 'pre'
    tr_atten_tpl: BaseHParams = sub_config_field(
        attentions.DotProductAttention.HParams)
    packed_input: bool = False
    tr_fflayer_tpl: BaseHParams = sub_config_field(
        TransformerFeedForward.HParams)
    ngrammer_tpl: Optional[BaseHParams] = base_layer.sub_config_field(None)

  def setup(self) -> None:
    p = self.hparams

    # Initialize Layer Norm
    if p.norm_policy == 'primer_hybrid':
      params = p.ln_tpl.clone()
      params.dim = p.input_dims
      self.create_child('pre_layer_norm', params)
      self.create_child('post_layer_norm', params)
    elif p.norm_policy == 'pre' or p.norm_policy == 'post':
      params = p.ln_tpl.clone()
      params.name = 'layer_norm'
      params.dim = p.input_dims
      self.create_child('layer_norm', params)
    else:
      raise ValueError('Unrecognized norm_policy: %s' % p.norm_policy)

    # Initialize multi-headed self-attention
    params = p.tr_atten_tpl.clone()
    params.name = 'multihead_self_atten'
    params.input_dim = p.input_dims
    params.hidden_dim = p.input_dims
    params.num_heads = p.num_heads
    params.dim_per_head = p.dim_per_head
    params.atten_dropout_prob = p.atten_dropout_prob
    if p.ngrammer_tpl:
      params.ngrammer_tpl = p.ngrammer_tpl
    self.create_child('self_attention', params)

    # Initialize residual dropout.
    params = p.dropout_tpl.clone()
    params.keep_prob = (1.0 - p.residual_dropout_prob)
    self.create_child('residual_dropout', params)

    # Initialize multi-headed cross-attention and layer norm.
    if p.use_cross_attention:
      if p.norm_policy in ('pre', 'post'):
        params = p.ln_tpl.clone()
        params.name = 'cross_layer_norm'
        params.dim = p.input_dims
        self.create_child('cross_layer_norm', params)
      elif p.norm_policy == 'primer_hybrid':
        params = p.ln_tpl.clone()
        params.dim = p.input_dims
        self.create_child('pre_cross_layer_norm', params)
        self.create_child('post_cross_layer_norm', params)
      else:
        raise ValueError(f'Unrecognized norm_policy: {p.norm_policy}')

      if p.cross_atten_tpl is not None:
        params = p.cross_atten_tpl.clone()
      else:
        params = p.tr_atten_tpl.clone()
      params.name = 'multihead_cross_atten'
      params.input_dim = p.input_dims
      params.hidden_dim = p.input_dims
      params.num_heads = p.num_heads
      params.dim_per_head = p.dim_per_head
      params.atten_dropout_prob = p.atten_dropout_prob
      # Note that cross attention should not use any position embeddings.
      if params.use_rotary_position_emb:
        raise ValueError('Rotary position embedding should not be enabled for '
                         'cross attention.')
      # Note that cross attention should not use depth-wise convolution.
      if params.dconv_qkv:
        raise ValueError('Depth-wise convolution should not be enabled for '
                         'cross attention.')
      self.create_child('cross_attention', params)

    # Initialize residual droppath
    if p.residual_droppath_prob > 0:
      droppath_p = stochastics.StochasticResidual.HParams(
          name='residual_droppath',
          survival_prob=1.0 - p.residual_droppath_prob)
      self.create_child('residual_droppath', droppath_p)

    # Initialize feed-forward layer
    if p.tr_fflayer_tpl:
      params = p.tr_fflayer_tpl.clone()
      params.name = 'tr_fflayer'
      params.input_dims = p.input_dims
      params.hidden_dims = p.hidden_dims
      params.relu_dropout_prob = p.relu_dropout_prob
      params.residual_dropout_prob = p.residual_dropout_prob
      params.residual_droppath_prob = p.residual_droppath_prob
      params.norm_policy = p.norm_policy
      self.create_child('ff_layer', params)

  def init_states(self, target_batch_size: int, target_max_length: int) -> None:
    """Initialize the cache for the Transformer layer.

    Args:
      target_batch_size: Batch size for the target.
      target_max_length: The length to decode the target.

    Returns:
      None.
    """
    raise NotImplementedError(type(self))

  def __call__(
      self,
      inputs: JTensor,
      paddings: JTensor,
      attention_mask: JTensor,
      cross_inputs: Optional[JTensor] = None,
      cross_attention_mask: Optional[JTensor] = None,
      segment_pos: Optional[JTensor] = None,
      segment_ids: Optional[JTensor] = None) -> Tuple[JTensor, JTensor]:
    """Transformer decoder layer.

    Args:
      inputs: Input sequence JTensor of shape [B, T, H].
      paddings: Input paddings JTensor of shape [B, T] (only used in FFN layer).
      attention_mask: Self attention mask ready to add to the logits. It can be
        of shape [1|B, 1, 1|T, T] which is broadcast compatible with the self
        attention matrix of shape [B, N, T, T]. This is assumed to have combined
        paddings, causal masking as well as segment maskings.
      cross_inputs: Output of the encoder, to be used for cross attention, of
        shape [B, S, H].
      cross_attention_mask: Cross attention mask ready to add to the logits. It
        can be of shape [1|B, 1, 1|T, S] which is broadcast compatible with the
        cross attention matrix of shape [B, N, T, T]. This is assumed to have
        combined paddings as well as segment maskings.
      segment_pos: A JTensor of shape [B, T]. The position of each token in a
        segment.
      segment_ids: A JTensor of shape [B, T] specifying which segment each token
        belongs to.

    Returns:
      The fflayer output with shape [B, T, D].
      atten_probs: A NestedMap with keys `self_atten` <float>[B, N, T, T].
    """
    # Layer normalize input
    p = self.hparams

    inputs_stats = stats.compute_stats(inputs, jnp.expand_dims(paddings, -1))
    self.add_summary('xformer_input_mean', inputs_stats.mean_v, verbosity=3)
    self.add_summary('xformer_input_std', inputs_stats.std_v, verbosity=3)
    self.add_summary('xformer_input_abs_max', inputs_stats.max_v, verbosity=3)

    if p.norm_policy == 'primer_hybrid':
      inputs_normalized = self.pre_layer_norm(inputs)
    elif p.norm_policy == 'pre':
      inputs_normalized = self.layer_norm(inputs)
    else:
      inputs_normalized = inputs

    # Compute self-attention, key/value vectors are the input itself
    atten_output, self_atten_probs = self.self_attention(
        inputs_normalized,
        inputs_normalized,
        inputs_normalized,
        atten_mask=attention_mask,
        query_segment_pos=segment_pos,
        key_segment_pos=segment_pos)
    atten_probs = NestedMap(self_atten=self_atten_probs)
    if p.norm_policy == 'primer_hybrid':
      atten_output = self.post_layer_norm(atten_output)
    elif p.norm_policy == 'post':
      atten_output = self.layer_norm(atten_output)

    # Residual dropout and connection
    atten_output = self.residual_dropout(atten_output)

    # Apply skip connection
    if p.residual_droppath_prob > 0.0:
      atten_output = self.residual_droppath(inputs, atten_output)
    else:
      atten_output += inputs

    # Apply cross attention if applicable
    if p.use_cross_attention and (not p.allow_skip_cross_attention or
                                  cross_inputs is not None):
      assert cross_inputs is not None
      assert cross_attention_mask is not None
      if p.norm_policy == 'pre':
        atten_output_normalized = self.cross_layer_norm(atten_output)
      elif p.norm_policy == 'primer_hybrid':
        atten_output_normalized = self.pre_cross_layer_norm(atten_output)
      elif p.norm_policy == 'post':
        atten_output_normalized = atten_output

      cross_atten_output, cross_atten_probs = self.cross_attention(
          atten_output_normalized,
          cross_inputs,
          cross_inputs,
          atten_mask=cross_attention_mask)
      atten_probs.cross_atten = cross_atten_probs

      if p.norm_policy == 'post':
        cross_atten_output = self.cross_layer_norm(cross_atten_output)
      elif p.norm_policy == 'primer_hybrid':
        cross_atten_output = self.post_cross_layer_norm(cross_atten_output)

      # Residual dropout and connection
      cross_atten_output = self.residual_dropout(cross_atten_output)

      if p.residual_droppath_prob > 0.0:
        atten_output = self.residual_droppath(atten_output, cross_atten_output)
      else:
        atten_output += cross_atten_output

    # Apply FFN layer
    output = self.ff_layer(atten_output, paddings=paddings)
    return output, atten_probs

  def extend_step(self,
                  inputs: JTensor,
                  *,
                  time_step: JTensor,
                  segment_pos: Optional[JTensor] = None,
                  attention_mask: JTensor,
                  cross_attention_mask: Optional[JTensor] = None) -> JTensor:
    # pyformat:disabled
    """Transformer decoder layer, autoregressive cached decoding.

    For cross attention, the key/value cache may have a smaller batch size b
    than inputs batch size B. In this case, we require B % b == 0, and this
    corresponds to multi-sample decoding for each input in b, and cross-
    attention states will be repeated by (B // b) times. Each consecutive
    (B // b) chunk in B correspond to multiple samples for the same cross
    # inputs.

    When `inputs` has shape [B, D], it will do extend_step on one token per
    batch in regular autoregressive decoding.

    When `inputs` has shape [B, L, D], it will do extend_step on L tokens per
    batch. This is used to do suffix scoring after autoregressive decoding.

    Args:
      inputs:         [B, D] or [B, L, D], target sequence at index time_step.
      time_step:      a 0-based scalar, the current decode step.
      segment_pos:    [B] or [B, L], the current position in the same segment.
        If unspecified, time_step will be used.

      attention_mask: [B, 1|L, T], per step attention mask for this time step.
        This combines causal mask with any segment mask if applicable.
      cross_attention_mask: [b|B, 1, 1 S], optional, cross_segment_mask for
        this time step. This combines padding mask with any segment mask if
        applicable.

    Returns:
      output: [B, D] or [B, L, D].
    """
    # pyformat:enabled
    p = self.hparams
    # Layer normalize input
    if p.norm_policy == 'primer_hybrid':
      inputs_normalized = self.pre_layer_norm(inputs)
    elif p.norm_policy == 'pre':
      inputs_normalized = self.layer_norm(inputs)

    # Self-attention layer.
    atten_output = self.self_attention.extend_step(
        inputs_normalized,
        atten_mask=attention_mask,
        time_step=time_step,
        segment_pos=segment_pos)
    if p.norm_policy == 'primer_hybrid':
      atten_output = self.post_layer_norm(atten_output)
    elif p.norm_policy == 'post':
      atten_output = self.layer_norm(atten_output)

    # Residual dropout and connection
    atten_output = self.residual_dropout(atten_output)
    atten_output += inputs

    # Apply cross attention if applicable
    if p.use_cross_attention and (not p.allow_skip_cross_attention or
                                  cross_attention_mask is not None):
      assert cross_attention_mask is not None
      if p.norm_policy == 'pre':
        atten_output_normalized = self.cross_layer_norm(atten_output)
      elif p.norm_policy == 'primer_hybrid':
        atten_output_normalized = self.pre_cross_layer_norm(atten_output)
      elif p.norm_policy == 'post':
        atten_output_normalized = atten_output

      cross_atten_output = self.cross_attention.extend_step(
          atten_output_normalized,
          atten_mask=jnp.squeeze(cross_attention_mask, 2),
          time_step=time_step,
          segment_pos=segment_pos,
          is_cross_attention=True)

      if p.norm_policy == 'post':
        cross_atten_output = self.cross_layer_norm(cross_atten_output)
      elif p.norm_policy == 'primer_hybrid':
        cross_atten_output = self.post_cross_layer_norm(cross_atten_output)

      # Residual dropout and connection
      cross_atten_output = self.residual_dropout(cross_atten_output)
      atten_output += cross_atten_output

    # Apply FFN layer
    output = self.ff_layer(atten_output)
    return output

  def transform_decode_state(
      self, transform_fn: base_layer.DecodeStateTransformFn) -> None:
    """Transforms all decode state variables based on transform_fn."""
    self.self_attention.transform_decode_state(transform_fn)

  def lazy_broadcast_prefix(self, num_suffix_samples: int,
                            suffix_length: int) -> None:
    """Performs lazy prefix broadcast on the decoding states.

    Current decoding states will be moved to PREFIX_DECODE_CACHE. New decoding
    state will be created for the suffixes with multiple samples sharing
    previous prefixes.

    Args:
      num_suffix_samples: Number of samples that will share the same previous
        decoding state.
      suffix_length: The length of the new suffix samples.
    """
    self.self_attention.lazy_broadcast_prefix(num_suffix_samples, suffix_length)

  def right_align_decode_state_with_prefix(
      self, max_prefix_size: int,
      right_align_fn: base_layer.DecodeStateTransformFn) -> None:
    """Right aligns decode state with prefix decode states."""
    self.self_attention.right_align_decode_state_with_prefix(
        max_prefix_size, right_align_fn)


class StackedTransformer(base_layer.BaseLayer):
  """A stack of Transformer layers."""

  class HParams(BaseHParams):
    """Associated hyper-params for this layer class.

    Attributes:
      use_cross_attention: If True, introduces cross encoder-decoder attention
        layer.
      mask_self_attention: Use masked self-attention.
      num_layers: Number of layers in this stack.
      model_dims: Model dimension in Transformer layers.
      hidden_dims: The hidden layer dimension of FFN in Transformer layers.
      num_heads: Number of attention heads.
      dim_per_head: Dimension of each attention head. If None then dim_per_head
        == hidden_dim // num_heads.
      dropout_prob: Apply dropout at this prob at various places.
      residual_droppath_prob: Probability at which we drop the entire residual
        path.
      input_dropout_prob: Dropout probability applied to the input before any
        processing happens.
      gating_func: Gating function type--can be one of the following options:
        'top2', based on the GShard paper: https://arxiv.org/abs/2006.16668,
        'expert_choice', based on https://arxiv.org/abs/2202.09368,
        'dense_top2': experimental gating function for decodiing. Similar to
        'top2' gating, but no capacity constrainst for each expert.
      unadjusted_expert_capacity_factor: Unadjusted expert capacity_factor. This
        is the ratio between global batch size and total capacity across all
        experts and all routing groups.
      transformer_layer_params_tpl: A template of Transformer.params, can be a
        list of params of length equal to the num_layers or a factor of
        num_layers. For a factor, the params are tiled as [a, a, ..., b,
        b,...,].
      packed_input: If True, each training example may pack multiple sequences.
      fold_padding_with_segment_mask: If True then segment mask is supposed to
        include the padding mask as well, i.e. treating PADs as one sequence and
        non-PADs as another.
      moe_layer_tpl: Template configuration for the moe feedforward layer.
      num_experts: Total number of experts.
      num_groups: Number of groups for dispathcing.
      min_group_size: If not None, num_groups will be adjusted so that there
        will be at least min_group_size tokens in each group.
      moe_layers: List of MoE layer indices, e.g. [0, 2, 4].
      ngrammer_tpls: Sequence of params for the Ngrammer layer. This param is
        shared between the Ngrammer layer as well as the VQNgrammer layer. The
        length of the sequence must match the number of attention layers. If an
        entry in the sequence is None, then there is no NGrammer layer present
        in that corresponding layer.
    """
    use_cross_attention: bool = False
    mask_self_attention: bool = False
    num_layers: int = 0
    model_dims: int = 0
    hidden_dims: int = 0
    num_heads: int = 0
    dim_per_head: Optional[int] = None
    dropout_prob: float = 0.0
    atten_dropout_prob: Optional[float] = None
    residual_dropout_prob: Optional[float] = None
    relu_dropout_prob: Optional[float] = None
    residual_droppath_prob: float = 0.0
    input_dropout_prob: float = 0.0
    gating_func: str = 'top2'
    unadjusted_expert_capacity_factor: float = 2.0
    transformer_layer_params_tpl: Union[
        BaseHParams,
        Sequence[BaseHParams]] = sub_config_field(Transformer.HParams)
    packed_input: bool = False
    fold_padding_with_segment_mask: bool = False
    moe_layer_tpl: BaseHParams = sub_config_field(
        TransformerFeedForwardMoe.HParams)
    num_experts: int = 0
    num_groups: int = 1
    min_group_size: Optional[int] = None
    moe_layers: Optional[Sequence[int]] = ()
    ngrammer_tpls: Optional[Sequence[BaseHParams]] = sub_config_field(None)

  def setup(self) -> None:
    p = self.hparams

    assert p.num_layers > 0
    assert p.model_dims > 0
    assert p.hidden_dims > 0
    assert p.num_heads > 0
    assert 0.0 <= p.dropout_prob < 1.0
    assert 0.0 <= p.input_dropout_prob < 1.0

    def _layer_params(i):
      """Construct i-th layer params."""
      if isinstance(p.transformer_layer_params_tpl, (list, tuple)):
        factor = p.num_layers // len(p.transformer_layer_params_tpl)
        ii = i // factor
        p_i = p.transformer_layer_params_tpl[ii].clone()
      else:
        p_i = p.transformer_layer_params_tpl.clone()
      p_i.name = f'layer_{i}'
      p_i.use_cross_attention = p.use_cross_attention
      p_i.mask_self_attention = p.mask_self_attention
      p_i.num_heads = p.num_heads
      p_i.dim_per_head = p.dim_per_head
      p_i.input_dims = p.model_dims
      p_i.packed_input = p.packed_input
      p_i.atten_dropout_prob = p.atten_dropout_prob or p.dropout_prob
      p_i.residual_dropout_prob = p.residual_dropout_prob or p.dropout_prob
      p_i.relu_dropout_prob = p.relu_dropout_prob or p.dropout_prob
      p_i.hidden_dims = p.hidden_dims

      if p.residual_droppath_prob > 0.0:
        p_i.residual_droppath_prob = (
            p.residual_droppath_prob * i / max(1, p.num_layers))

      if p.moe_layers and i in p.moe_layers:
        assert p.num_experts > 0
        moe_p = p.moe_layer_tpl.clone()
        moe_p.num_experts = p.num_experts
        moe_p.num_groups = p.num_groups
        moe_p.min_group_size = p.min_group_size
        moe_p.gating_func = p.gating_func
        if moe_p.hidden_dims:
          # MoE hidden_dims could be different from FFN hidden_dims
          p_i.hidden_dims = moe_p.hidden_dims
        p_i.tr_fflayer_tpl = moe_p

      if p.ngrammer_tpls is not None:
        if p.ngrammer_tpls[i] is not None:
          p_i.ngrammer_tpl = p.ngrammer_tpls[i]
      return p_i

    if isinstance(p.transformer_layer_params_tpl, (list, tuple)):
      if p.num_layers % len(p.transformer_layer_params_tpl):
        raise ValueError('num_layers should be divisible by '
                         'transformer_layer_params_tpl')

    layer_params = [_layer_params(i) for i in range(p.num_layers)]
    self.create_children('x_layers', layer_params)

    if p.input_dropout_prob > 0.0:
      self.create_child(
          'input_dropout',
          stochastics.Dropout.HParams(keep_prob=1.0 - p.input_dropout_prob))

  def init_states(self, *args: Any, **kwargs: Any) -> None:
    """Initialize the cache for the StackedTransformer layer.

    Args:
      *args: Other arguments.
      **kwargs: Other keyword arguments.

    Returns:
      None.
    """
    raise NotImplementedError(type(self))

  def __call__(self,
               inputs: JTensor,
               paddings: JTensor,
               segment_mask: Optional[JTensor] = None,
               cross_inputs: Optional[JTensor] = None,
               cross_paddings: Optional[JTensor] = None,
               cross_segment_mask: Optional[JTensor] = None,
               segment_pos: Optional[JTensor] = None) -> JTensor:
    """Stacked Transformer layer.

    Args:
      inputs: Input sequence of shape [B, T, H].
      paddings: Input paddings of shape [B, T].
      segment_mask: Segment mask for packed input of shape [B, 1, T, T] ready to
        add to logits.
      cross_inputs: Output of the encoder, to be used for cross attention, of
        shape [B, S, H].
      cross_paddings: Paddings for cross atention of shape [B, S].
      cross_segment_mask: Segment mask for encoder-decoder in packed input case
        of shape [B, 1, T, S].
      segment_pos: Segment pos for packed input of shape [B, T].

    Returns:
      Output vector with shape [B, T, D].
    """
    p = self.hparams
    x_out = inputs
    if p.packed_input:
      assert segment_mask is not None

    if p.use_cross_attention:
      assert cross_inputs is not None
      assert cross_paddings is not None
      if p.packed_input:
        assert cross_segment_mask is not None

    attention_mask, cross_attention_mask = compute_attention_masks_for_fprop(
        inputs,
        paddings,
        p.mask_self_attention,
        segment_mask,
        cross_inputs,
        cross_paddings,
        cross_segment_mask,
        fold_padding_with_segment_mask=p.fold_padding_with_segment_mask)

    if p.input_dropout_prob > 0.0:
      x_out = self.input_dropout(x_out)

    for i in range(p.num_layers):
      x_in = x_out
      x_out, _ = self.x_layers[i](
          x_in,
          paddings,
          attention_mask,
          cross_inputs,
          cross_attention_mask,
          segment_pos=segment_pos)
    return x_out

  def extend_step(self,
                  inputs: JTensor,
                  *,
                  time_step: JTensor,
                  segment_pos: Optional[JTensor] = None,
                  atten_mask: Optional[JTensor] = None,
                  cross_paddings: Optional[JTensor] = None,
                  cross_segment_mask: Optional[JTensor] = None) -> JTensor:
    """Transformer stacked decoder layers, autoregressive cached decoding.

    When `inputs` has shape [B, L, D], it will do extend_step on N tokenks per
    batch. This is used to do suffix scoring after autoregressive decoding.

    When `inputs` has shape [B, D], it will do extend_step on one token per
    batch in regular autoregressive decoding.

    Args:
      inputs:         [B, D] or [B, L, D], target sequence at index time_step.
      time_step:      a 0-based scalar, the current decode step.
      segment_pos:    [B] or [B, L], the current position in the same segment.
        If unspecified, time_step will be used.
      atten_mask:     [B, 1, L, S], optional. If None, a causal mask on a
        contiguous sequence is used by default. This is unsupported for
        cross-attention.
      cross_paddings: [B|b, S], optional 0/1 JTensor.
      cross_segment_mask: [B|b, 1, S], optional.

    Returns:
      decoder_output: [B, D], the last decoder layer output.
    """
    p = self.hparams

    max_t = self.x_layers[0].self_attention.decoding_state_sequence_length()

    if p.use_cross_attention:
      assert cross_paddings is not None

    if atten_mask is None:
      if segment_pos is None:
        segment_mask = None
      else:
        assert segment_pos.ndim == 1
        # Calculate the segment mask for this step. We assume the segment is
        # contiguous.
        segment_pos_2d = jnp.expand_dims(segment_pos, 1)
        # [B, T]
        src_positions = jnp.arange(max_t)[
            jnp.newaxis, :] - time_step + segment_pos_2d
        # [B, T]
        src_segment_ids = jnp.where(src_positions < 0, 0, 1)
        # [B, 1, 1, T]
        segment_mask = attentions.segment_mask(
            jnp.ones_like(segment_pos_2d), src_segment_ids, inputs.dtype)
        # [B, 1, T]
        segment_mask = jnp.squeeze(segment_mask, 1)

      attention_mask, cross_attention_mask = (
          compute_attention_masks_for_extend_step(time_step, max_t,
                                                  segment_mask, cross_paddings,
                                                  cross_segment_mask))
    else:
      if p.use_cross_attention:
        raise NotImplementedError('cross attention does not support customized '
                                  'attention_mask passed in yet.')

      attention_mask = atten_mask
      cross_attention_mask = None

    decoder_input = inputs
    for layer in self.x_layers:
      decoder_output = layer.extend_step(
          decoder_input,
          time_step=time_step,
          attention_mask=attention_mask,
          segment_pos=segment_pos,
          cross_attention_mask=cross_attention_mask)
      decoder_input = decoder_output
    return decoder_output

  def transform_decode_state(
      self, transform_fn: base_layer.DecodeStateTransformFn) -> None:
    """Transforms all decode state variables based on transform_fn."""
    for layer in self.x_layers:
      layer.transform_decode_state(transform_fn)

  def lazy_broadcast_prefix(self, num_suffix_samples: int,
                            suffix_length: int) -> None:
    """Performs lazy prefix broadcast on the decoding states.

    Current decoding states will be moved to PREFIX_DECODE_CACHE. New decoding
    state will be created for the suffixes with multiple samples sharing
    previous prefixes.

    Args:
      num_suffix_samples: Number of samples that will share the same previous
        decoding state.
      suffix_length: The length of the new suffix samples.
    """
    for layer in self.x_layers:
      layer.lazy_broadcast_prefix(num_suffix_samples, suffix_length)

  def right_align_decode_state_with_prefix(
      self, max_prefix_size: int,
      right_align_fn: base_layer.DecodeStateTransformFn) -> None:
    """Right aligns decode state with prefix decode states."""
    for layer in self.x_layers:
      layer.right_align_decode_state_with_prefix(max_prefix_size,
                                                 right_align_fn)


class StackedTransformerRepeated(base_layer.BaseLayer):
  """A StackedTransformer implemented using the generic Repeat."""

  class HParams(BaseHParams):
    """Associated hyper-params for this layer class.

    Attributes:
      block: The params of a block. A block can be a single transformer
        layer,multiple layers, or a dense layer followed by a sparse layer,
        a.k.a. MOE block.
      x_times: Num times to repeat a block.
      checkpoint_policy: How to checkpoint residuals for BProp: save nothing,
        dot only or dot with no batch dimensions.
      unroll_in_decode: Whether to unroll the layers during extend_step.
    """
    block: BaseHParams = sub_config_field(StackedTransformer.HParams)
    x_times: int = 0
    checkpoint_policy: repeats.AutodiffCheckpointType = repeats.AutodiffCheckpointType.SAVE_NOTHING
    unroll_in_decode: bool = True
    repeat_layer_name: str = 'repeat'
    sublayer_name: str = 'sub'

  class WeightShardingHParams(BaseWtShardingHParams):
    """Represents how layer's learned parameters are partitioned across a mesh.

    Attributes:
      stages: How the list of blocks should be sharded.
    """
    block: SplitDimsMapping = None

  def setup(self) -> None:
    p = self.hparams
    wp = p.weight_split_dims_mapping

    repeat_l_params = repeats.Repeat.HParams(
        sub_tpl=p.block,
        x_times=p.x_times,
        checkpoint_policy=p.checkpoint_policy,
        unpack_summaries=True,
        unroll_in_decode=p.unroll_in_decode,
        sublayer_name=p.sublayer_name)
    repeat_l_params.weight_split_dims_mapping.sub = wp.block

    self.create_child(self.hparams.repeat_layer_name, repeat_l_params)

  @property
  def repeat_layer(self) -> repeats.Repeat:
    return getattr(self, self.hparams.repeat_layer_name)

  def __call__(self,
               inputs: JTensor,
               paddings: JTensor,
               segment_mask: Optional[JTensor] = None,
               cross_inputs: Optional[JTensor] = None,
               cross_paddings: Optional[JTensor] = None,
               cross_segment_mask: Optional[JTensor] = None,
               segment_pos: Optional[JTensor] = None) -> JTensor:
    """Stacked Transformer layer.

    Args:
      inputs: Input sequence of shape [B, T, H].
      paddings: Input paddings of shape [B, T].
      segment_mask: Segment mask for packed input of shape [B, 1, T, T] ready to
        add to logits.
      cross_inputs: Output of the encoder, to be used for cross attention, of
        shape [B, S, H].
      cross_paddings: Paddings for cross atention of shape [B, S].
      cross_segment_mask: Segment mask for encoder-decoder in packed input case
        of shape [B, 1, T, S].
      segment_pos: Segment position of shape [B, T].

    Returns:
      Output vector with shape [B, T, D].
    """

    # TODO(zhangqiaorjc): Use positional args until nn.scan supports kwargs.
    out = self.repeat_layer(inputs, paddings, segment_mask, cross_inputs,
                            cross_paddings, cross_segment_mask, segment_pos)

    return out

  def init_states(self, *args: Any, **kwargs: Any) -> None:
    """Initialize the cache for the StackedTransformerRepeated layer.

    Args:
      *args: Other arguments.
      **kwargs: Other keyword arguments.
    Return: None.
    """
    raise NotImplementedError(type(self))

  def extend_step(self,
                  inputs: JTensor,
                  *,
                  time_step: JTensor,
                  segment_pos: Optional[JTensor] = None,
                  atten_mask: Optional[JTensor] = None,
                  cross_paddings: Optional[JTensor] = None,
                  cross_segment_mask: Optional[JTensor] = None) -> JTensor:
    """Transformer stacked decoder layers, autoregressive cached decoding.

    When `inputs` has shape [B, L, D], it will do extend_step on N tokenks per
    batch. This is used to do suffix scoring after autoregressive decoding.

    When `inputs` has shape [B, D], it will do extend_step on one token per
    batch in regular autoregressive decoding.

    Args:
      inputs: Target sequence of shape [B, D] corresponding to target sequence
        at index time_step.
      time_step: A scalar, the current decode step, 0-based.
      segment_pos: An optional JTensor of shape [B]. Current position in the
        same segment. If unspecified, time_step will be used.
      atten_mask: An optional JTensor of shape [B, 1, L, S] for attention mask
        between inputs and the whole sequence. If it is None, it will be
        computed as a causal mask on a contiguous sequence. This passed in
        atten_mask is unsupported with cross-attention.
      cross_paddings: Source paddings - [b|B, S].
      cross_segment_mask: if not None, cross_segment_mask for this time step, of
        shape [b|B, 1, S].

    Returns:
      decoder_output: The last decoder layer output of shape [B, D].
    """

    return self.repeat_layer.extend_step(
        inputs,
        time_step=time_step,
        segment_pos=segment_pos,
        atten_mask=atten_mask,
        cross_paddings=cross_paddings,
        cross_segment_mask=cross_segment_mask)

  def transform_decode_state(
      self, transform_fn: base_layer.DecodeStateTransformFn) -> None:
    """Transforms all decode state variables based on transform_fn."""
    self.repeat_layer.transform_decode_state(transform_fn)

  def lazy_broadcast_prefix(self, num_suffix_samples: int,
                            suffix_length: int) -> None:
    """Performs lazy prefix broadcast on the decoding states.

    Current decoding states will be moved to PREFIX_DECODE_CACHE. New decoding
    state will be created for the suffixes with multiple samples sharing
    previous prefixes.

    Args:
      num_suffix_samples: Number of samples that will share the same previous
        decoding state.
      suffix_length: The length of the new suffix samples.
    """
    self.repeat_layer.lazy_broadcast_prefix(num_suffix_samples, suffix_length)

  def right_align_decode_state_with_prefix(
      self, max_prefix_size: int,
      right_align_fn: base_layer.DecodeStateTransformFn) -> None:
    """Right aligns decode state with prefix decode states."""
    self.repeat_layer.right_align_decode_state_with_prefix(
        max_prefix_size, right_align_fn)


class PipelinedTransformer(base_layer.BaseLayer):
  """A pipelined Transformer."""

  class HParams(BaseHParams):
    """Associated hyper-params for this layer class.

    Attributes:
      pipeline_stage: The layer params of each stage.
      circular_repeat: Number of round-robin layers in each stage for the
        circular pipeline schedule. If 1, this will be basic GPipe schedule.
      num_pipeline_stages: Number of pipeline stages.
      num_pipeline_microbatches: Number of pipeline microbatches.
      pipeline_microbatch_size: Size of each pipeline microbatch.
      stream_io: Whether to enable input/output streaming across stages. This is
        typically useful for DCN.
      pipeline_broadcast_inputs: If true, broadcast inputs (shared between all
        stages instead of being computed by the previous stage) will be passed
        stage-by-stage instead of being replicated.
    """
    pipeline_stage: BaseHParams = sub_config_field(StackedTransformer.HParams)
    circular_repeat: int = 1
    num_pipeline_stages: Optional[int] = None
    num_pipeline_microbatches: Optional[int] = None
    pipeline_microbatch_size: Optional[int] = None
    stream_io: bool = False
    pipeline_broadcast_inputs: bool = False
    checkpoint_policy: AutodiffCheckpointType = AutodiffCheckpointType.SAVE_ITERATION_INPUT

  class WeightShardingHParams(BaseWtShardingHParams):
    """Represents how layer's learned parameters are partitioned across a mesh.

    Attributes:
      stages: How the num_stages dimension should be sharded.
    """
    stages: SplitDimsMapping = (None,)

  class ActivationShardingHParams(BaseActShardingHParams):
    """Represents how intermediate values should be partitioned across a mesh.

    Attributes:
      final_out: How the final output should be sharded.
    """
    final_out: SplitDimsMapping = None

  def setup(self) -> None:
    p = self.hparams
    assert p.num_pipeline_stages > 0

    stage_params = p.pipeline_stage.clone()
    if p.circular_repeat == 1:
      pipeline_params = pipeline.LayerwiseShardablePipelined.HParams(
          name=p.name,
          num_stages=p.num_pipeline_stages,
          single_stage_body=stage_params,
          num_microbatches=p.num_pipeline_microbatches,
          microbatch_size=p.pipeline_microbatch_size,
          unpack_summaries=True,
          stream_io=p.stream_io,
          pipeline_broadcast_inputs=p.pipeline_broadcast_inputs,
          checkpoint_policy=p.checkpoint_policy)
    else:
      pipeline_params = pipeline.CircularLayerwiseShardablePipelined.HParams(
          name=p.name,
          num_stages=p.num_pipeline_stages,
          circular_repeat=p.circular_repeat,
          single_stage_body=stage_params,
          num_microbatches=p.num_pipeline_microbatches,
          microbatch_size=p.pipeline_microbatch_size,
          unpack_summaries=True,
          stream_io=p.stream_io,
          pipeline_broadcast_inputs=p.pipeline_broadcast_inputs,
          checkpoint_policy=p.checkpoint_policy)

    pipeline_params.weight_split_dims_mapping.stages = (
        p.weight_split_dims_mapping.stages)
    self.create_child('pipeline', pipeline_params)

  def __call__(self,
               inputs: JTensor,
               paddings: JTensor,
               segment_mask: Optional[JTensor] = None,
               cross_inputs: Optional[JTensor] = None,
               cross_paddings: Optional[JTensor] = None,
               cross_segment_mask: Optional[JTensor] = None,
               segment_pos: Optional[JTensor] = None) -> JTensor:
    """Pipelined Transformer layer.

    Args:
      inputs: Input sequence of shape [B, T, H].
      paddings: Input paddings of shape [B, T].
      segment_mask: Segment mask for packed input of shape [B, 1, T, T] ready to
        add to logits.
      cross_inputs: Output of the encoder, to be used for cross attention, of
        shape [B, S, H].
      cross_paddings: Paddings for cross atention of shape [B, S].
      cross_segment_mask: Segment mask for encoder-decoder in packed input case
        of shape [B, 1, T, S].
      segment_pos: Segment position of shape [B, T].

    Returns:
      Output vector with shape [B, T, D].
    """
    p = self.hparams
    if p.pipeline_stage.cls == StackedTransformer:
      xformer_layer_p = p.pipeline_stage.transformer_layer_params_tpl
    else:
      assert p.pipeline_stage.cls == StackedTransformerRepeated
      xformer_layer_p = p.pipeline_stage.block.transformer_layer_params_tpl
    bld_mapping = xformer_layer_p.tr_atten_tpl.activation_split_dims_mapping.bld
    if not p.stream_io:
      # Annotate the inputs before the pipeline to prevent unexpected
      # propagation from earlier layers.
      inputs = base_layer.maybe_shard(inputs, bld_mapping, p.mesh_axis_names)
      if bld_mapping is not None:
        # Annotate other broadcast inputs.
        paddings = base_layer.maybe_shard(paddings, bld_mapping[:-1],
                                          p.mesh_axis_names)

        # For cross inputs, we only specify the batch dim sharding.
        def _shard_batch_dim_only(x):
          return base_layer.maybe_shard(
              x, [bld_mapping[0]] + [-1] * (x.ndim - 1),
              p.mesh_axis_names,
              unconstrained_dims=range(1, x.ndim))

        if segment_mask is not None:
          segment_mask = _shard_batch_dim_only(segment_mask)
        if cross_inputs is not None:
          cross_inputs = _shard_batch_dim_only(cross_inputs)
        if cross_paddings is not None:
          cross_paddings = _shard_batch_dim_only(cross_paddings)
        if cross_segment_mask is not None:
          cross_segment_mask = _shard_batch_dim_only(cross_segment_mask)

        if segment_pos is not None:
          segment_pos = base_layer.maybe_shard(segment_pos, bld_mapping[:-1],
                                               p.mesh_axis_names)
    outputs = self.pipeline(
        inputs,
        paddings,
        segment_mask=segment_mask,
        cross_inputs=cross_inputs,
        cross_paddings=cross_paddings,
        cross_segment_mask=cross_segment_mask,
        segment_pos=segment_pos)

    # For non-streaming cases, we need to annotate the `outputs` twice below
    # because the first one makes sure output has the same sharding as input so
    # that the pipeine body is sharded properly.
    # The second is to switch to other shardings for later layers;
    # e.g., repurpose pipeline stages cores to data parallelism for softmax.

    if not p.stream_io:
      # Annotate the output to match input sharding.
      outputs = base_layer.maybe_shard(outputs, bld_mapping, p.mesh_axis_names)
    # Re-annotate the final output.
    outputs = base_layer.maybe_shard(outputs,
                                     p.activation_split_dims_mapping.final_out,
                                     p.mesh_axis_names)
    return outputs

  def init_states(self, *args, **kwargs) -> NestedMap:
    raise NotImplementedError(type(self))

  def extend_step(
      self,
      inputs: JTensor,
      *,
      time_step: JTensor,
      segment_pos: Optional[JTensor] = None,
      cross_paddings: Optional[JTensor] = None,
      cross_segment_mask: Optional[JTensor] = None
  ) -> Tuple[JTensor, NestedMap]:
    raise NotImplementedError(type(self))


class PipelineCompatibleStackedTransformerRepeated(StackedTransformerRepeated):
  """Repeated transformer for inference compatible with pipeline."""

  def setup(self) -> None:
    p = self.hparams.clone()
    p.repeat_layer_name = 'pipeline'
    p.sublayer_name = 'body'
    super().setup()
