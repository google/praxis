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

"""SSM plus Transformer layers. See https://arxiv.org/pdf/2212.10544.pdf."""

from __future__ import annotations

from typing import Any, Optional, Tuple

from absl import logging
from jax import numpy as jnp
from praxis import base_layer
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis.layers import activations as activations_lib
from praxis.layers import checkpoint_policy
from praxis.layers import linears
from praxis.layers import ssm
from praxis.layers import stats
from praxis.layers import stochastics
from praxis.layers import transformers

NestedMap = py_utils.NestedMap
WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams
sub_config_field = base_layer.sub_config_field
template_field = base_layer.template_field
LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]

JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor

SplitDimsMapping = pytypes.SplitDimsMapping
AutodiffCheckpointType = checkpoint_policy.AutodiffCheckpointType


class SSMTransformer(transformers.Transformer):
  """Transformer layer using SSM instead of self-attention.

  Attributes:
    ssm_tpl: Parameterization of the SSM Layer.
    ssm_nheads: Number of heads (polynomials) per channel for the SSM model.
    ssm_dim: Input dimension/channel size.
    ssm_l_max: Longest seq length.
    ssm_hippo_type: Which type of hippo to use.
    ssm_step_size: The step size for SSM discretization.
    decode_num_samples: How many decoding samples for each example
  """
  ssm_tpl: LayerTpl = template_field(ssm.SSM)
  ssm_nheads: int = 0
  ssm_dim: int = 0
  ssm_l_max: int = 0
  ssm_hippo_type: str = 'ss4d-1d-legs'
  ssm_step_size: float = 0.1
  decode_num_samples: int = 4

  def setup(self) -> None:

    # Initialize Layer Norm
    if self.norm_policy == 'primer_hybrid':
      params = self.ln_tpl.clone()
      params.dim = self.input_dims
      self.create_child('pre_layer_norm', params)
      self.create_child('post_layer_norm', params)
    elif self.norm_policy == 'pre' or self.norm_policy == 'post':
      params = self.ln_tpl.clone()
      params.name = 'layer_norm'
      params.dim = self.input_dims
      self.create_child('layer_norm', params)
    else:
      raise ValueError('Unrecognized norm_policy: %s' % self.norm_policy)

    # Initialize residual dropout.
    params = self.dropout_tpl.clone()
    params.keep_prob = 1.0 - self.residual_dropout_prob
    self.create_child('residual_dropout', params)

    # Initialize multi-headed cross-attention and layer norm.
    if self.use_cross_attention:
      if self.norm_policy in ('pre', 'post'):
        params = self.ln_tpl.clone()
        params.name = 'cross_layer_norm'
        params.dim = self.input_dims
        self.create_child('cross_layer_norm', params)
      elif self.norm_policy == 'primer_hybrid':
        params = self.ln_tpl.clone()
        params.dim = self.input_dims
        self.create_child('pre_cross_layer_norm', params)
        self.create_child('post_cross_layer_norm', params)
      else:
        raise ValueError(f'Unrecognized norm_policy: {self.norm_policy}')

      if self.cross_atten_tpl is not None:
        params = self.cross_atten_tpl.clone()
      else:
        params = self.tr_atten_tpl.clone()
      params.name = 'multihead_cross_atten'
      params.input_dim = self.input_dims
      params.hidden_dim = self.input_dims
      params.num_heads = self.num_heads
      params.dim_per_head = self.dim_per_head
      params.atten_dropout_prob = self.atten_dropout_prob
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
    if self.residual_droppath_prob > 0:
      droppath_p = pax_fiddle.Config(
          stochastics.StochasticResidual,
          name='residual_droppath',
          survival_prob=1.0 - self.residual_droppath_prob,
      )
      self.create_child('residual_droppath', droppath_p)

    # Initialize feed-forward layer
    if self.tr_fflayer_tpl:
      params = self.tr_fflayer_tpl.clone()
      params.name = 'tr_fflayer'
      params.input_dims = self.input_dims
      params.hidden_dims = self.hidden_dims
      params.relu_dropout_prob = self.relu_dropout_prob
      params.residual_dropout_prob = self.residual_dropout_prob
      params.residual_droppath_prob = self.residual_droppath_prob
      params.norm_policy = self.norm_policy
      self.create_child('ff_layer', params)

    # Initialize ssm layer
    params = self.ssm_tpl.clone()
    params.name = 'ssm'
    params.nheads = self.ssm_nheads
    assert self.ssm_dim == self.input_dims, (self.ssm_dim, self.input_dims)
    params.dim = self.ssm_dim
    params.l_max = self.ssm_l_max
    params.decode_num_samples = self.decode_num_samples
    params.hippo_type = self.ssm_hippo_type
    params.step_size = self.ssm_step_size
    self.create_child('ssm', params)

  def decoding_sequence_length(self) -> int:
    """Get the decoding sequence length."""
    return self.ssm_l_max

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
    inputs_stats = stats.compute_stats(inputs, jnp.expand_dims(paddings, -1))
    self.add_summary('xformer_input_mean', inputs_stats.mean_v, verbosity=3)
    self.add_summary('xformer_input_std', inputs_stats.std_v, verbosity=3)
    self.add_summary('xformer_input_abs_max', inputs_stats.max_v, verbosity=3)

    if self.norm_policy == 'primer_hybrid':
      inputs_normalized = self.pre_layer_norm(inputs)
    elif self.norm_policy == 'pre':
      inputs_normalized = self.layer_norm(inputs)
    else:
      inputs_normalized = inputs

    atten_output = self.ssm(inputs_normalized)

    if self.norm_policy == 'primer_hybrid':
      atten_output = self.post_layer_norm(atten_output)
    elif self.norm_policy == 'post':
      atten_output = self.layer_norm(atten_output)

    # Residual dropout and connection
    atten_output = self.residual_dropout(atten_output)

    # Apply skip connection
    if self.residual_droppath_prob > 0.0:
      atten_output = self.residual_droppath(inputs, atten_output)
    else:
      atten_output += inputs

    # Apply cross attention if applicable
    if self.use_cross_attention and (
        not self.allow_skip_cross_attention or cross_inputs is not None
    ):
      assert cross_inputs is not None
      assert cross_attention_mask is not None
      if self.norm_policy == 'pre':
        atten_output_normalized = self.cross_layer_norm(atten_output)
      elif self.norm_policy == 'primer_hybrid':
        atten_output_normalized = self.pre_cross_layer_norm(atten_output)
      elif self.norm_policy == 'post':
        atten_output_normalized = atten_output

      cross_atten_output, _ = self.cross_attention(
          atten_output_normalized,
          cross_inputs,
          cross_inputs,
          atten_mask=cross_attention_mask)

      if self.norm_policy == 'post':
        cross_atten_output = self.cross_layer_norm(cross_atten_output)
      elif self.norm_policy == 'primer_hybrid':
        cross_atten_output = self.post_cross_layer_norm(cross_atten_output)

      # Residual dropout and connection
      cross_atten_output = self.residual_dropout(cross_atten_output)

      if self.residual_droppath_prob > 0.0:
        atten_output = self.residual_droppath(atten_output, cross_atten_output)
      else:
        atten_output += cross_atten_output

    # Apply FFN layer
    output = self.ff_layer(atten_output, paddings=paddings)
    return output, None  # pytype: disable=bad-return-type  # jax-ndarray

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

      attention_mask: [B, 1, L, S] if extends multiple steps (i.e. `inputs` is
        of shape [B, L, D]) or [B, 1, T] if extends one step (i.e. `inputs` is
        of shape [B, D]), optional attention mask for this time step. This
        combines causal mask with any segment mask if applicable.
      cross_attention_mask: [b|B, 1, 1 S], optional, cross_segment_mask for
        this time step. This combines padding mask with any segment mask if
        applicable.

    Returns:
      output: [B, D] or [B, L, D].
    """
    # Layer normalize input
    if self.norm_policy == 'primer_hybrid':
      inputs_normalized = self.pre_layer_norm(inputs)
    elif self.norm_policy == 'pre':
      inputs_normalized = self.layer_norm(inputs)

    assert inputs_normalized.ndim == 2 or inputs_normalized.ndim == 3
    if inputs_normalized.ndim == 2:  # [B, D]
      atten_output = self.ssm.extend_step(
          inputs_normalized)
    elif inputs_normalized.ndim == 3:  # [B, L, D]
      atten_output = self.ssm(inputs_normalized)

    if self.norm_policy == 'primer_hybrid':
      atten_output = self.post_layer_norm(atten_output)
    elif self.norm_policy == 'post':
      atten_output = self.layer_norm(atten_output)

    # Residual dropout and connection
    atten_output = self.residual_dropout(atten_output)
    atten_output += inputs

    # Apply cross attention if applicable
    if self.use_cross_attention and (
        not self.allow_skip_cross_attention or cross_attention_mask is not None
    ):
      assert cross_attention_mask is not None
      if self.norm_policy == 'pre':
        atten_output_normalized = self.cross_layer_norm(atten_output)
      elif self.norm_policy == 'primer_hybrid':
        atten_output_normalized = self.pre_cross_layer_norm(atten_output)
      elif self.norm_policy == 'post':
        atten_output_normalized = atten_output

      cross_atten_output = self.cross_attention.extend_step(
          atten_output_normalized,
          atten_mask=jnp.squeeze(cross_attention_mask, 2),
          time_step=time_step,
          segment_pos=segment_pos,
          is_cross_attention=True)

      if self.norm_policy == 'post':
        cross_atten_output = self.cross_layer_norm(cross_atten_output)
      elif self.norm_policy == 'primer_hybrid':
        cross_atten_output = self.post_cross_layer_norm(cross_atten_output)

      # Residual dropout and connection
      cross_atten_output = self.residual_dropout(cross_atten_output)
      atten_output += cross_atten_output

    # Apply FFN layer
    output = self.ff_layer(atten_output)
    return output

  def transform_decode_state(
      self, transform_fn: base_layer.DecodeStateTransformFn) -> None:
    pass


class SSMGated(SSMTransformer):
  """Gated State Space Model, https://arxiv.org/pdf/2212.10544.pdf.

  Attributes:
    gss_fflayer_tpl: Params for the Gated SSM feedforward layers.
  """
  gss_fflayer_tpl: LayerTpl = template_field(linears.FeedForward)

  def setup(self) -> None:

    # Initialize Layer Norm
    if self.norm_policy == 'primer_hybrid':
      params = self.ln_tpl.clone()
      params.dim = self.input_dims
      self.create_child('pre_layer_norm', params)
      self.create_child('post_layer_norm', params)
    elif self.norm_policy == 'pre' or self.norm_policy == 'post':
      params = self.ln_tpl.clone()
      params.name = 'layer_norm'
      params.dim = self.input_dims
      self.create_child('layer_norm', params)
    else:
      raise ValueError('Unrecognized norm_policy: %s' % self.norm_policy)

    # Initialize residual dropout.
    params = self.dropout_tpl.clone()
    params.keep_prob = 1.0 - self.residual_dropout_prob
    self.create_child('residual_dropout', params)

    # Initialize multi-headed cross-attention and layer norm.
    if self.use_cross_attention:
      if self.norm_policy in ('pre', 'post'):
        params = self.ln_tpl.clone()
        params.name = 'cross_layer_norm'
        params.dim = self.input_dims
        self.create_child('cross_layer_norm', params)
      elif self.norm_policy == 'primer_hybrid':
        params = self.ln_tpl.clone()
        params.dim = self.input_dims
        self.create_child('pre_cross_layer_norm', params)
        self.create_child('post_cross_layer_norm', params)
      else:
        raise ValueError(f'Unrecognized norm_policy: {self.norm_policy}')

      if self.cross_atten_tpl is not None:
        params = self.cross_atten_tpl.clone()
      else:
        params = self.tr_atten_tpl.clone()
      params.name = 'multihead_cross_atten'
      params.input_dim = self.input_dims
      params.hidden_dim = self.input_dims
      params.num_heads = self.num_heads
      params.dim_per_head = self.dim_per_head
      params.atten_dropout_prob = self.atten_dropout_prob
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
    if self.residual_droppath_prob > 0:
      droppath_p = pax_fiddle.Config(
          stochastics.StochasticResidual,
          name='residual_droppath',
          survival_prob=1.0 - self.residual_droppath_prob,
      )
      self.create_child('residual_droppath', droppath_p)

    # Initialize ssm and gss layer
    params = self.ssm_tpl.clone()
    params.name = 'ssm'

    params.nheads = self.ssm_nheads
    params.dim = self.ssm_dim
    params.l_max = self.ssm_l_max
    params.decode_num_samples = self.decode_num_samples
    params.hippo_type = self.ssm_hippo_type
    params.step_size = self.ssm_step_size
    self.create_child('ssm', params)

    gss_ffn_v = self.gss_fflayer_tpl.clone()
    gss_ffn_v.name = 'gss_ffn_v'
    gss_ffn_v.input_dims = self.input_dims
    gss_ffn_v.output_dims = self.input_dims * 3
    gss_ffn_v.activation_tpl = pax_fiddle.Config(activations_lib.GELU)
    self.create_child('gss_ffn_v', gss_ffn_v)

    gss_ffn_u = self.gss_fflayer_tpl.clone()
    gss_ffn_u.name = 'gss_ffn_u'
    gss_ffn_u.input_dims = self.input_dims
    gss_ffn_u.output_dims = self.ssm_dim
    gss_ffn_u.activation_tpl = pax_fiddle.Config(activations_lib.GELU)
    self.create_child('gss_ffn_u', gss_ffn_u)

    gss_ffn_uc = self.gss_fflayer_tpl.clone()
    gss_ffn_uc.name = 'gss_ffn_uc'
    gss_ffn_uc.input_dims = self.ssm_dim
    gss_ffn_uc.output_dims = self.input_dims
    gss_ffn_uc.activation_tpl = pax_fiddle.Config(activations_lib.Identity)
    self.create_child('gss_ffn_uc', gss_ffn_uc)

    gss_ffn_uco = self.gss_fflayer_tpl.clone()
    gss_ffn_uco.name = 'gss_ffn_uco'
    gss_ffn_uco.input_dims = self.input_dims
    gss_ffn_uco.output_dims = self.input_dims * 3
    gss_ffn_uco.activation_tpl = pax_fiddle.Config(activations_lib.GELU)
    self.create_child('gss_ffn_uco', gss_ffn_uco)

    gss_ffn_o = self.gss_fflayer_tpl.clone()
    gss_ffn_o.name = 'gss_ffn_o'
    gss_ffn_o.input_dims = self.input_dims * 3
    gss_ffn_o.output_dims = self.input_dims
    gss_ffn_o.activation_tpl = pax_fiddle.Config(activations_lib.Identity)
    self.create_child('gss_ffn_o', gss_ffn_o)


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
    inputs_stats = stats.compute_stats(inputs, jnp.expand_dims(paddings, -1))
    self.add_summary('xformer_input_mean', inputs_stats.mean_v, verbosity=3)
    self.add_summary('xformer_input_std', inputs_stats.std_v, verbosity=3)
    self.add_summary('xformer_input_abs_max', inputs_stats.max_v, verbosity=3)

    if self.norm_policy == 'primer_hybrid':
      inputs_normalized = self.pre_layer_norm(inputs)
    elif self.norm_policy == 'pre':
      inputs_normalized = self.layer_norm(inputs)
    else:
      inputs_normalized = inputs

    v = self.gss_ffn_v(inputs_normalized)
    u = self.gss_ffn_u(inputs_normalized)
    y = self.ssm(u)
    atten_output = self.gss_ffn_uc(y)

    if self.norm_policy == 'primer_hybrid':
      atten_output = self.post_layer_norm(atten_output)
    elif self.norm_policy == 'post':
      atten_output = self.layer_norm(atten_output)

    # Residual dropout and connection
    atten_output = self.residual_dropout(atten_output)

    # Apply skip connection
    if self.residual_droppath_prob > 0.0:
      atten_output = self.residual_droppath(inputs, atten_output)
    else:
      atten_output += inputs

    # Apply cross attention if applicable
    if self.use_cross_attention and (
        not self.allow_skip_cross_attention or cross_inputs is not None
    ):
      assert cross_inputs is not None
      assert cross_attention_mask is not None
      if self.norm_policy == 'pre':
        atten_output_normalized = self.cross_layer_norm(atten_output)
      elif self.norm_policy == 'primer_hybrid':
        atten_output_normalized = self.pre_cross_layer_norm(atten_output)
      elif self.norm_policy == 'post':
        atten_output_normalized = atten_output

      cross_atten_output, _ = self.cross_attention(
          atten_output_normalized,
          cross_inputs,
          cross_inputs,
          atten_mask=cross_attention_mask)

      if self.norm_policy == 'post':
        cross_atten_output = self.cross_layer_norm(cross_atten_output)
      elif self.norm_policy == 'primer_hybrid':
        cross_atten_output = self.post_cross_layer_norm(cross_atten_output)

      # Residual dropout and connection
      cross_atten_output = self.residual_dropout(cross_atten_output)

      if self.residual_droppath_prob > 0.0:
        atten_output = self.residual_droppath(atten_output, cross_atten_output)
      else:
        atten_output += cross_atten_output

    # Apply FFN layer
    atten_output = self.gss_ffn_uco(atten_output)
    output = self.gss_ffn_o(atten_output * v)

    return output, None  # pytype: disable=bad-return-type  # jax-ndarray

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

      attention_mask: [B, 1, L, S] if extends multiple steps (i.e. `inputs` is
        of shape [B, L, D]) or [B, 1, T] if extends one step (i.e. `inputs` is
        of shape [B, D]), optional attention mask for this time step. This
        combines causal mask with any segment mask if applicable.
      cross_attention_mask: [b|B, 1, 1 S], optional, cross_segment_mask for
        this time step. This combines padding mask with any segment mask if
        applicable.

    Returns:
      output: [B, D] or [B, L, D].
    """
    # Layer normalize input
    if self.norm_policy == 'primer_hybrid':
      inputs_normalized = self.pre_layer_norm(inputs)
    elif self.norm_policy == 'pre':
      inputs_normalized = self.layer_norm(inputs)

    assert inputs_normalized.ndim == 2 or inputs_normalized.ndim == 3
    v = self.gss_ffn_v(inputs_normalized)
    u = self.gss_ffn_u(inputs_normalized)
    if u.ndim == 2:  # [B, D]
      y = self.ssm.extend_step(u)
    elif u.ndim == 3:  # [B, L, D]
      y = self.ssm(u)
    atten_output = self.gss_ffn_uc(y)

    if self.norm_policy == 'primer_hybrid':
      atten_output = self.post_layer_norm(atten_output)
    elif self.norm_policy == 'post':
      atten_output = self.layer_norm(atten_output)

    # Residual dropout and connection
    atten_output = self.residual_dropout(atten_output)
    atten_output += inputs

    # Apply cross attention if applicable
    if self.use_cross_attention and (
        not self.allow_skip_cross_attention or cross_attention_mask is not None
    ):
      assert cross_attention_mask is not None
      if self.norm_policy == 'pre':
        atten_output_normalized = self.cross_layer_norm(atten_output)
      elif self.norm_policy == 'primer_hybrid':
        atten_output_normalized = self.pre_cross_layer_norm(atten_output)
      elif self.norm_policy == 'post':
        atten_output_normalized = atten_output

      cross_atten_output = self.cross_attention.extend_step(
          atten_output_normalized,
          atten_mask=jnp.squeeze(cross_attention_mask, 2),
          time_step=time_step,
          segment_pos=segment_pos,
          is_cross_attention=True)

      if self.norm_policy == 'post':
        cross_atten_output = self.cross_layer_norm(cross_atten_output)
      elif self.norm_policy == 'primer_hybrid':
        cross_atten_output = self.post_cross_layer_norm(cross_atten_output)

      # Residual dropout and connection
      cross_atten_output = self.residual_dropout(cross_atten_output)
      atten_output += cross_atten_output

    # Apply FFN layer
    atten_output = self.gss_ffn_uco(atten_output)
    output = self.gss_ffn_o(atten_output * v)
    return output

  def transform_decode_state(
      self, transform_fn: base_layer.DecodeStateTransformFn) -> None:
    pass
