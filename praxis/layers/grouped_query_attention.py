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

"""Attention layer that supports different number of q and kv heads."""

from typing import Sequence, Tuple

from flax import linen as nn
import jax
from jax import numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from praxis import base_layer
from praxis import pax_fiddle as fdl
from praxis import py_utils
from praxis import pytypes
from praxis.layers import einsum
from praxis.layers import sharding as shd

_SplitDimsMapping = pytypes.SplitDimsMapping
JTensor = pytypes.JTensor


# return Bool[ArrayT, '1 1 T S']
def causal_mask(seqlen: int) -> JTensor:
  mask = jnp.arange(seqlen)[:, None] < jnp.arange(seqlen)[None, :]
  return mask[None, None, :, :]


# segment_ids: Int[ArrayT, 'B T'],
# src_segment_ids: Int[ArrayT, 'B S']
# return Bool[ArrayT, 'B 1 T S']
def segment_mask(
    segment_ids: JTensor,
    src_segment_ids: JTensor | None = None,
) -> JTensor:
  if src_segment_ids is None:
    src_segment_ids = segment_ids
  # [B, T, S]
  mask = segment_ids[..., None] != src_segment_ids[..., None, :]
  # [B, 1, T, S]
  return mask[:, None, :, :]


# segment_ids: Int[ArrayT, 'B T']
# return Bool[ArrayT, 'B 1 T T']
def attention_mask(segment_ids: JTensor) -> JTensor:
  seq_length = segment_ids.shape[1]
  mask = jnp.logical_or(causal_mask(seq_length), segment_mask(segment_ids))
  return mask


# paddings: Float[ArrayT, 'B T']
# return Float[ArrayT, 'B 1 1 T']
def padding_mask(paddings: JTensor) -> JTensor:
  # Assume paddings with 1.0 for padded out positions.
  return paddings[:, None, None, :]


# logits: Float[ArrayT, 'B K G T S']
# atten_mask: Bool[ArrayT, 'B #K #G T S']
# return Float[ArrayT, 'B N T S']
def apply_mask(logits: JTensor, atten_mask: JTensor) -> JTensor:
  return jnp.where(
      atten_mask, py_utils.get_large_negative_number(logits.dtype), logits
  )


class GroupedQueryAttention(base_layer.BaseLayer):
  """Dot-product attention sharing keys and values across heads.

  This implementation heavily uses einsum to be efficient on TPUs.  We use the
  following conventions to denote array axes.

    B = batch size
    S = length of the key/value (source)
    T = length of the query (target)
    D = model dimension
    N = number of attention heads
    H = dimensions of each attention head.
    K = number of K/V heads

  When K > 1, we require N % K == 0 and every (N // K) q heads share the same
  KV heads.

  Attributes:
    input_dim: An integer as number of input nodes.
    hidden_dim: Unused legacy field. Set "input_dim" instead.
    num_heads: Number of attention heads.
    num_kv_heads: Number of kv heads. num_heads % num_kv_heads = 0.
    dim_per_head: Dimension of each attention head.
    atten_dropout_prob: Probability at which we apply dropout to the attention
      weights.
    atten_temp: Temperature for attention logit
    atten_logit_cap: Cap the absolute values of logits by tanh. Enabled when a
      positive value is specified. May not be supported by a subclass.
    rope: If set, the Rope function to apply.
    rope_min_max_timescales: If provided, these are the min/max timescales to
      add rotary position embedding (ROPE) to the queries and keys before
      computing self attention scores.
  """

  input_dim: int = 0
  hidden_dim: int = 0
  num_heads: int = 1
  num_kv_heads: int = 1
  dim_per_head: int = 0
  atten_dropout_prob: float = 0.0
  atten_temp: float = 1.0
  use_bias: bool = True
  atten_logit_cap: float = 0.0
  rope_min_max_timescales: tuple[int, int] | None = None

  class WeightSharding(base_layer.BaseLayer.WeightSharding):
    """Weight sharding.

    Attributes:
      dnh: q and post weights.
      dkh: k and v weights. If None, dnh is used.
    """

    dnh: _SplitDimsMapping = None
    dkh: _SplitDimsMapping = None

  class ActivationSharding(base_layer.BaseLayer.ActivationSharding):
    """Activation sharding.

    Attributes:
      btd: input/output.
      btnh: q projection.
      bskh: k/v projection.
    """

    btd: _SplitDimsMapping = None
    btnh: _SplitDimsMapping = None
    bskh: _SplitDimsMapping = None

  def setup(self) -> None:
    d = self.input_dim
    n = self.num_heads
    k = self.num_kv_heads
    h = self.dim_per_head
    assert d, f'input_dim is {d}'
    assert h, f'dim_per_head is {h}'
    assert n % k == 0, f'num_heads {n} % num_kv_heads {k} != 0'

    def project_input(num_heads, name, multihead_w_sharding):
      single_head_w_sharding = shd.derive(multihead_w_sharding, 'dnh->dh')
      return einsum.Einsum(
          name=name,
          eqn='...D,DH->...H' if num_heads == 1 else '...D,DNH->...NH',
          w_shape=(d, h) if num_heads == 1 else (d, num_heads, h),
          use_bias=self.use_bias,
          weight_split_dims_mapping=fdl.Config(
              einsum.Einsum.WeightSharding,
              wt=single_head_w_sharding
              if num_heads == 1
              else multihead_w_sharding,
          ),
      )

    ws = self.weight_split_dims_mapping
    self.query = project_input(n, 'query', ws.dnh)
    self.key = project_input(k, 'key', ws.dkh or ws.dnh)
    self.value = project_input(k, 'value', ws.dkh or ws.dnh)
    self.post = einsum.Einsum(
        name='post',
        eqn='...NH,DNH->...D',
        w_shape=(d, n, h),
        use_bias=self.use_bias,
        weight_split_dims_mapping=fdl.Config(
            einsum.Einsum.WeightSharding, wt=ws.dnh
        ),
    )

  def _atten_context(
      self,
      query: JTensor,
      key: JTensor,
      value: JTensor,
      atten_mask: JTensor,
  ) -> Tuple[JTensor, JTensor]:
    """Computes atten context."""
    b, t, n, h = query.shape
    query = query * (self.dim_per_head**-0.5) / self.atten_temp
    s, nk = key.shape[1:3]
    grouped_query = jnp.reshape(query, (b, t, nk, n // nk, h))
    logits = jnp.einsum('BTKGH,BSKH->BKGTS', grouped_query, key)
    logits = checkpoint_name(logits, 'logits')

    if self.atten_logit_cap and self.atten_logit_cap > 0.0:
      cap = jnp.array(self.atten_logit_cap, dtype=logits.dtype)
      # Since this caps the negative side as well, we must defer the
      # pad-with-very-negative-logits logic after this.
      logits = cap * jnp.tanh(logits / cap)

    # Attention softmax is always carried out in fp32.
    logits = logits.astype(jnp.float32)
    mask_heads = atten_mask.shape[1]
    if mask_heads == 1:
      atten_mask = atten_mask[:, :, None, :, :]
    else:
      atten_mask = atten_mask.reshape((b, nk, n // nk, t, s))
    padded_logits = apply_mask(logits, atten_mask)
    probs = jax.nn.softmax(padded_logits, axis=-1).astype(key.dtype)
    # Apply attention dropout.
    if self.atten_dropout_prob > 0.0 and not self.do_eval:
      probs = apply_dropout(
          probs,
          self.next_prng_key(),
          self.atten_dropout_prob,
      )
    # Compute the attention context.
    encoded = jnp.einsum('BKGTS,BSKH->BTKGH', probs, value)
    encoded = jnp.reshape(encoded, (b, t, n, h))
    encoded = checkpoint_name(encoded, 'context')
    return encoded, probs

  def _maybe_rope(self, x: JTensor, pos: JTensor) -> JTensor:
    if self.rope_min_max_timescales is None:
      return x
    min_ts, max_ts = self.rope_min_max_timescales
    return apply_rope(x, min_ts, max_ts, pos)

  def _qkv(
      self,
      q_in: JTensor,
      k_in: JTensor,
      v_in: JTensor,
      unshard_kv_d: bool,
  ) -> tuple[
      JTensor,
      JTensor,
      JTensor,
  ]:
    """QKV projection and normalization."""
    sh = self.activation_split_dims_mapping
    q_in = shd.shard(q_in, sh.btd, 'btd->b[t]d')
    # unshard_kv_d: work around b/305051548 when there is mesh transpose and
    # major/minor model axes.
    kv_in_eqn = 'btd->b[t]x' if unshard_kv_d else 'btd->b[t]d'
    k_in = shd.shard(k_in, sh.btd, kv_in_eqn)
    v_in = shd.shard(v_in, sh.btd, kv_in_eqn)
    q = checkpoint_name(self.query(q_in), 'query_proj')
    k = checkpoint_name(self.key(k_in), 'key_proj')
    v = checkpoint_name(self.value(v_in), 'value_proj')
    # Work around b/305051548
    q = shd.shard(q, sh.btd, 'btd->b[t]??')
    if self.num_kv_heads == 1:
      k = shd.shard(k, sh.btd, 'btd->b[t]?')
      v = shd.shard(v, sh.btd, 'btd->b[t]?')
      k = jnp.expand_dims(k, axis=-2)
      v = jnp.expand_dims(v, axis=-2)
    else:
      k = shd.shard(k, sh.btd, 'btd->b[t]??')
      v = shd.shard(v, sh.btd, 'btd->b[t]??')
    q = shd.shard(q, sh.btnh, 'btnh->b[t]nh')
    eqn = 'bskh->b[s]1h' if self.num_kv_heads == 1 else 'bskh->b[s]kh'
    k = shd.shard(k, sh.bskh, eqn)
    v = shd.shard(v, sh.bskh, eqn)
    return q, k, v

  def __call__(
      self,
      query_vec: JTensor,
      key_vec: JTensor,
      value_vec: JTensor,
      atten_mask: JTensor,
      query_segment_pos: JTensor,
      key_segment_pos: JTensor,
  ) -> Tuple[JTensor, JTensor]:
    """Computes the value vector given the current query output.

    Args:
      query_vec: jax.Array of shape [B, T, D].
      key_vec: jax.Array of shape [B, S, D].
      value_vec: jax.Array of shape [B, S, D].
      atten_mask: jax.Array of shape [B, 1, 1/T, S] which is a mask that is
        applied to prevent attention between unwanted pairs. This is a boolean
        mask where True means unwanted/masked out. Note that the first and third
        dimension allow size 1 if the mask is shared by every item in the batch
        or every token in the target sequence.
      query_segment_pos: jax.Array of shape [B, T]
      key_segment_pos: jax.Array of shape [B, S]

    Returns:
      encoded: jax.Array of shape [B, T, D].
    """
    q, k, v = self._qkv(query_vec, key_vec, value_vec, unshard_kv_d=True)
    # Apply rotary position embeddings.
    q = self._maybe_rope(q, query_segment_pos)
    k = self._maybe_rope(k, key_segment_pos)

    self.update_decode_state('key_state', k)
    self.update_decode_state('value_state', v)
    # Replicate sequence dim for k and v.
    k = shd.shard_one_dim(k, None, dim=1)
    v = shd.shard_one_dim(v, None, dim=1)
    atten_mask = shd.shard_one_dim(atten_mask, None, dim=3)
    encoded, atten_probs = self._atten_context(q, k, v, atten_mask)
    sh = self.activation_split_dims_mapping
    encoded = shd.shard(encoded, sh.btnh)
    # Post projection
    encoded = self.post(encoded)
    encoded = shd.shard(encoded, sh.btd)
    return checkpoint_name(encoded, 'out_proj'), atten_probs

  @nn.nowrap
  def extend_decode_state(
      self, name: str, value: jax.Array, time_step: JTensor
  ) -> jax.Array:
    """Extends decode state at time_step.

    The decode state is batch major with shape [B, T, K, H].
    Args:
      name: Variable name in decoder cache.
      value: Value to extend at time step.
      time_step: A scalar. Time step to update the state.

    Returns:
      Updated decode cache state of that variable.
    """
    extend_value = jnp.expand_dims(value, axis=1)
    indices = [0] * extend_value.ndim
    indices[1] = time_step.astype(jnp.int32)
    state = self.get_decode_state(name)
    assert state is not None
    new_state = jax.lax.dynamic_update_slice(
        state, extend_value.astype(state.dtype), indices
    )
    eqn = 'bskh->bs1h' if self.num_kv_heads == 1 else None
    new_state = shd.shard(
        new_state, self.activation_split_dims_mapping.bskh, eqn
    )
    self.update_decode_state(name, new_state)
    return new_state

  def extend_step(
      self,
      query_vec: JTensor,
      *,
      atten_mask: JTensor,
      time_step: JTensor,
      segment_pos: JTensor | None,
      is_cross_attention: bool = False,
  ) -> JTensor:
    """Computes the value vector given the query of the current step.

    This function is used by autoregressive decoding.

    Args:
      query_vec: jax.Array of shape [B, D].
      atten_mask: jax.Array of shape [B/1, 1, S].
      time_step: A scalar jax.Array. Current time-step, 0-based, physical offset
        in the kv cache.
      segment_pos: An optional jax.Array of shape [B]. Current position in the
        same segment. If unspecified, time_step will be used.
      is_cross_attention: Whether this is cross-attention, where kv cache should
        not be updated.

    Returns:
      encoded: Output jax.Array of shape [B, D] at `time_step`.
    """
    time_step = jnp.array(time_step)
    assert time_step.ndim == 0
    q, k, v = self._qkv(query_vec, query_vec, query_vec, unshard_kv_d=False)

    if segment_pos is None:
      position = jnp.broadcast_to(time_step, [query_vec.shape[0]])[:, None]
    else:
      position = segment_pos[:, None]
    q = self._maybe_rope(q, position)
    k = self._maybe_rope(k, position)

    if not is_cross_attention:
      self.extend_decode_state('key_state', k, time_step)
      self.extend_decode_state('value_state', v, time_step)
    encoded, atten_probs = self._atten_context(
        jnp.expand_dims(q, axis=1),
        self.get_decode_state('key_state'),
        self.get_decode_state('value_state'),
        jnp.expand_dims(atten_mask, axis=2),
    )
    del atten_probs
    encoded = shd.shard(encoded, self.activation_split_dims_mapping.btnh)
    # Work around b/305051548
    encoded = shd.shard(
        encoded, self.activation_split_dims_mapping.btd, 'btd->bt??'
    )
    # Post projection.
    encoded = self.post(encoded)
    encoded = shd.shard(encoded, self.activation_split_dims_mapping.btd)
    return jnp.squeeze(encoded, axis=1)

  def decoding_state_sequence_length(self) -> int:
    """Returns the length of full decoding sequences."""
    return self.get_decode_state('key_state').shape[1]

  def transform_decode_state(
      self, transform_fn: base_layer.DecodeStateTransformFn
  ) -> None:
    """Transforms all decode state variables based on transform_fn."""
    batch_dim = 0
    time_dim = 1
    for name, state in self.variables[base_layer.DECODE_CACHE].items():
      if not isinstance(state, jax.Array):
        continue
      new_state = transform_fn(state, batch_dim, time_dim)
      eqn = 'bskh->bs1h' if self.num_kv_heads == 1 else None
      new_state = shd.shard(
          new_state, self.activation_split_dims_mapping.bskh, eqn
      )
      self.update_decode_state(name, new_state)


def apply_rope(
    inputs: JTensor,
    min_timescale: int,
    max_timescale: int,
    position: JTensor | None = None,
) -> JTensor:
  """Applies rotary position embedding for a given 1-d sequence.

  Args:
    inputs: The input sequence on which to apply the Rotary position embedding.
      The shape should be [B, S, ...].
    min_timescale: Start of the geometric index. Determines the periodicity of
      the added signal.
    max_timescale: End of the geometric index. Determines the frequency of the
      added signal.
    position: Optional position jax.Array which denotes the position of each
      token in the sequence. This only needs to be supplied when the sequence is
      packed. It is of shape [B, S].

  Returns:
    A jax.Array that includes the inputs together with the rotary position
    embedding incorporated in it.
  """
  embedding_dim = inputs.shape[-1]
  half_embedding_dim = embedding_dim // 2
  fraction = 2 * jnp.arange(0, half_embedding_dim) / embedding_dim
  timescale = min_timescale * (max_timescale / min_timescale) ** fraction
  timescale = jnp.expand_dims(timescale, axis=tuple(range(inputs.ndim - 1)))
  position = jnp.expand_dims(position, axis=tuple(range(2, inputs.ndim)))
  sinusoid_inp = position / timescale
  sin = jnp.sin(sinusoid_inp)
  cos = jnp.cos(sinusoid_inp)
  first_half, second_half = jnp.split(inputs, 2, axis=-1)
  first_part = first_half * cos - second_half * sin
  second_part = second_half * cos + first_half * sin
  first_part = first_part.astype(inputs.dtype)
  second_part = second_part.astype(inputs.dtype)
  return jnp.concatenate([first_part, second_part], axis=-1)


def apply_dropout(
    x: JTensor,
    prng_key: JTensor,
    dropout_prob: float,
    broadcast_dims: Sequence[int] = (),
) -> JTensor:
  """Applies dropout to x.

  Args:
    x: The input tensor.
    prng_key: PRNG key for randomness.
    dropout_prob: The dropout probability.
    broadcast_dims: Dimension indices in x where the dropout mask is broadcast.

  Returns:
    The result tensor.
  """
  if dropout_prob == 0.0:
    return x
  mask_shape = [1 if d in broadcast_dims else x.shape[d] for d in range(x.ndim)]
  keep = jax.random.bernoulli(prng_key, 1.0 - dropout_prob, shape=mask_shape)
  return x * keep / (1.0 - dropout_prob)
