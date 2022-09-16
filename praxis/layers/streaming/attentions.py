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

"""Streaming aware attention layers."""

import jax
from jax import numpy as jnp
from praxis import base_layer
from praxis import py_utils
from praxis import pytypes
from praxis.layers import attentions
from praxis.layers.streaming import streaming_base

NestedMap = py_utils.NestedMap
NestedJTensor = pytypes.NestedJTensor
WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams
BaseHParams = base_layer.BaseLayer.HParams

JTensor = pytypes.JTensor


class LocalSelfAttention(attentions.LocalSelfAttention,  # pytype: disable=signature-mismatch
                         streaming_base.StreamingBase):
  """Streaming aware LocalSelfAttention layer."""

  QUERY_STRIDE = 1  # Keeping it for a case if we add it as a parameter.

  @classmethod
  def get_right_context(cls, hparams):
    return hparams.right_context

  @classmethod
  def get_stride(cls, hparams):
    return 1

  def _zero_state_dynamic_length(self, batch_size, with_paddings):
    """Creates initial state given the batch size.

    It will create states:
      - key:   [B, context_len, N, H].
      - value: [B, context_len, N, H].
      - masks: [B, context_len]. A Tensor where Falses are masked out positions.
      - query: (only if p.right_context > 0) [B, p.right_context, N, H].
      - out_masks : (only if p.right_context> 0): [B, p.right_context].
      where:
        context_len = p.left_context - 1 + p.right_context;
        B: batch_size;
        N: num_heads;
        H: per_head_dim.

    Args:
      batch_size: the batch size.
      with_paddings: if False paddings will not be computed.
    """
    p = self.hparams

    context_len = p.left_context - 1 + p.right_context
    per_head_dim = p.hidden_dim // p.num_heads

    key_state = jnp.zeros([batch_size, context_len, p.num_heads, per_head_dim],
                          p.dtype)
    self._update_streaming_state('key', key_state)

    value_state = jnp.zeros_like(key_state, p.dtype)
    self._update_streaming_state('value', value_state)

    # At the beginning, all positions are masked out.
    masks = jnp.zeros([batch_size, context_len], jnp.bool_)
    self._update_streaming_state('masks', masks)

    if p.right_context > 0:
      query_right = p.right_context // self.QUERY_STRIDE
      query = jnp.zeros(
          [batch_size, query_right, p.num_heads, per_head_dim], p.dtype)
      self._update_streaming_state('query', query)

      out_masks = jnp.zeros([batch_size, query_right],
                            jnp.bool_) if with_paddings else None
      self._update_streaming_state('out_masks', out_masks)

      # This is used only if the caller of the layer uses skip_connection in
      # the layer's client code.
      skip_conn_input = jnp.zeros(
          [batch_size, query_right, p.hidden_dim], p.dtype)
      self._update_streaming_state('skip_conn_input', skip_conn_input)

  def init_states(self,
                  batch_size: int,
                  with_paddings: bool = True):
    """Creates streaming states in base_layer.DECODE_CACHE.

    Args:
      batch_size: defines batch size of streaming states.
      with_paddings: if True it will creates streaming states
        for padding processing, else will set it None (it can save some memory).
    """
    self._zero_state_dynamic_length(batch_size, with_paddings)

  def _stream_atten_logits(self, query: JTensor, key: JTensor) -> JTensor:
    """Computes the dot products of a set of queries and a set of keys."""
    # [B, Q, N, T]
    return jnp.einsum('BQNH,BTNH->BQNT', query, key)

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
    assert not p.dconv_qkv
    assert not p.use_rotary_position_emb
    assert not p.relative_bias_tpl

    query_vec = inputs.query_vec
    key_vec = inputs.key_vec
    value_vec = inputs.value_vec

    # TODO(rybakov) Instead of paddings switch to atten_mask?
    query_paddings = inputs.paddings
    key_paddings = inputs.paddings

    if p.combine_qkv:
      # Only supports self attention.
      assert query_vec is key_vec
      assert query_vec is value_vec
      # Project inputs to key, value and query using a combined weight for
      # faster performance on TPU.
      query_proj, key_proj, value_proj = self.combined_qkv(query_vec)
    else:
      # Project inputs to key, value and query, respectively has shape
      # [B, S, N, H], [B, S, N, H], and [B, T, N, H].
      query_proj = self.query(query_vec)
      key_proj = self.key(key_vec)
      value_proj = self.value(value_vec)

    query = query_proj
    key = key_proj
    value = value_proj

    b, s, n, h = key.shape
    base_layer.assert_has_shape(value, [b, s, n, h])
    base_layer.assert_has_shape(query, [b, -1, n, h])

    query = self._scale_query(query)
    query_proj = query

    k = s
    q = (k + self.QUERY_STRIDE - 1) // self.QUERY_STRIDE
    h = p.hidden_dim // p.num_heads
    context_len = p.left_context - 1 + p.right_context

    input_masks = jnp.logical_not(query_paddings.astype(jnp.bool_))
    if p.right_context == 0:
      # [B, Q, N, H]
      query = query_proj
      out_paddings = query_paddings
    else:
      state_query = self.get_streaming_state('query')
      # [B, QR + Q, N, H]
      concat_query = jnp.concatenate([state_query, query_proj], axis=1)
      # [B, Q, N, H]
      query = concat_query[:, :q]

      state_out_masks = self.get_streaming_state('out_masks')
      if state_out_masks is not None:
        concat_out_masks = jnp.concatenate([state_out_masks, input_masks],
                                           axis=1)
        out_masks = concat_out_masks[:, :q]
        out_paddings = jnp.logical_not(out_masks).astype(query_paddings.dtype)
      else:
        concat_out_masks = None
        out_paddings = None

    state_key = self.get_streaming_state('key')
    # key, value, mask.
    # [B, T, N, H].
    key = jnp.concatenate(
        [state_key, self.key(key_vec)],
        axis=1)

    state_value = self.get_streaming_state('value')
    # [B, T, N, H]
    value = jnp.concatenate(
        [state_value, self.value(value_vec)],
        axis=1)

    state_masks = self.get_streaming_state('masks')
    # [B, T]
    key_masks = jnp.logical_not(jnp.bool_(key_paddings))
    state_masks = jnp.concatenate([state_masks, key_masks], axis=1)

    # [B, Q, N, T]
    logits = self._stream_atten_logits(query, key)
    logits = self._cap_logits(logits)

    # Generate local atten mask.
    # [Q, 1]
    # Assuming the current query index starts from 0
    query_right = p.right_context // self.QUERY_STRIDE
    query_indices = jnp.expand_dims(
        jnp.arange(-query_right, -query_right + q) * self.QUERY_STRIDE, -1)
    # [1, T]
    target_indices = jnp.expand_dims(jnp.arange(-context_len, k), 0)
    # 1s are masked positions.
    # [Q, T]
    distance = query_indices - target_indices
    effective_right_context = p.right_context + self.QUERY_STRIDE - 1
    local_atten_per_step_masks = jnp.logical_and(
        distance <= p.left_context - 1,
        distance >= -effective_right_context)
    # [1, Q, T]
    local_atten_per_step_masks = jnp.expand_dims(local_atten_per_step_masks, 0)
    # [B, 1, T]
    expanded_state_masks = jnp.expand_dims(state_masks, 1)

    # [B, Q, T]
    final_masks = jnp.logical_and(expanded_state_masks,
                                  local_atten_per_step_masks)
    # [B, Q, 1, T]
    final_masks = jnp.expand_dims(final_masks, axis=2)

    # [B, Q, N, T]
    logits = py_utils.apply_padding(
        logits, jnp.logical_not(final_masks),
        py_utils.get_large_negative_number(logits.dtype))

    # [B, Q, N, T]
    if p.attention_extra_logit is None:
      probs = jax.nn.softmax(logits, axis=-1).astype(logits.dtype)
    else:
      probs = jnp.exp(self._log_softmax_with_extra_logit(logits)).astype(
          logits.dtype)

    probs = self.atten_dropout(probs)

    # [B, Q, N, H]
    output = jnp.einsum('BQNT,BTNH->BQNH', probs, value)

    # Post projection.
    # [B, Q, D]
    output = self.post(output)

    self._update_streaming_state('key', key[:, k:, :, :])
    self._update_streaming_state('value', value[:, k:, :, :])
    self._update_streaming_state('masks', state_masks[:, k:])

    if p.right_context > 0:
      self._update_streaming_state('query', concat_query[:, q:])
      if concat_out_masks is not None:
        self._update_streaming_state('out_masks', concat_out_masks[:, q:])

    return NestedMap(encoded=output, paddings=out_paddings)
