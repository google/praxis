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

"""Util functions for decoder."""

from typing import Callable, Tuple

import jax
from jax import numpy as jnp
from praxis import base_layer

JTensor = base_layer.JTensor
ExtendStepFn = Callable[[base_layer.BaseLayer, JTensor, JTensor], JTensor]
FPropFn = Callable[[base_layer.BaseLayer, JTensor, JTensor], None]
TransformStateFn = Callable[
    [base_layer.BaseLayer, base_layer.DecodeStateTransformFn], None]
# lazy_broadcast_prefix_fn(model, num_suffix_samples, suffix_length)
LazyBroadcastPrefixFn = Callable[[base_layer.BaseLayer, int, int], None]


def length_norm(t, length_norm_alpha) -> jnp.ndarray:
  """Length norm for beam search."""
  return jax.lax.pow((t.astype(jnp.float32) + 5.) / 5., length_norm_alpha)


def gather_output_id(long_output_ids: jnp.ndarray,
                     topk_indices: jnp.ndarray) -> jnp.ndarray:
  """Gather output ids from long topk ouput ids.

  Args:
    long_output_ids: The first TopK indices from [batch_size, beam_size
      *beam_size].
    topk_indices: The topk indices from the second Topk of shape [batch_size,
      beam_size].

  Returns:
    Final output id of shape [batch_size, beam_size].
  """
  one_hot = jax.nn.one_hot(
      topk_indices, long_output_ids.shape[-1], dtype=jnp.int32)
  output_id = jnp.einsum('bt,bkt->bk', long_output_ids.astype(jnp.int32),
                         one_hot)
  return output_id.astype(jnp.int32)


def two_stage_topk(
    logits: jnp.ndarray, hyp_scores: jnp.ndarray,
    eos_id: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Two stage TopK to choose TopK values and indices from each beam.

  Args:
    logits: The logits of [batch_size, beam_size, vocab_size or beam_size *
      vocab_size].
    hyp_scores: The topK scores of [batch_size, beam_size].
    eos_id: EOS id.

  Returns:
    topk_value, topk_indices of shape [batch_size, beam_size * beam_size], they
      are the topk of `logits` on the last dimenension and merged on the last
      two dimensions.
    final_topk_value, final_topk_indices of shape [batch_size, beam_size],
      they are the topk value and inidices of `topk_value`.
  """
  vocab_size = logits.shape[-1]
  batch_size = hyp_scores.shape[0]
  beam_size = hyp_scores.shape[1]
  logits_reshape = jnp.reshape(
      logits, newshape=(batch_size * beam_size, vocab_size))
  topk_value, topk_indices = jax.lax.top_k(logits_reshape, beam_size)
  topk_value = jnp.reshape(
      topk_value, newshape=(batch_size, beam_size, beam_size))
  topk_indices = jnp.reshape(
      topk_indices, newshape=(batch_size, beam_size, beam_size))
  topk_value += jnp.expand_dims(hyp_scores, -1)
  topk_value -= 1e9 * jnp.equal(topk_indices, eos_id).astype(topk_value.dtype)

  topk_value = jnp.reshape(
      topk_value, newshape=(batch_size, beam_size * beam_size))
  topk_indices = jnp.reshape(
      topk_indices, newshape=(batch_size, beam_size * beam_size))

  final_topk_value, final_topk_indices = jax.lax.top_k(topk_value, beam_size)
  return topk_value, topk_indices, final_topk_value, final_topk_indices


def pad_state_fn(state_padding_size: int) -> base_layer.DecodeStateTransformFn:
  """A function used to pad attention states after prefix fprop."""

  def _pad_state_fn(x, batch_dim, time_dim):
    del batch_dim
    if time_dim < 0:
      return x
    paddings = [
        [0, state_padding_size if i == time_dim else 0] for i in range(x.ndim)
    ]
    return jnp.pad(x, paddings)

  return _pad_state_fn


def slice_state_fn(slice_start: int,
                   slice_limit: int) -> base_layer.DecodeStateTransformFn:
  """A function used to slice attention states on the time dimension."""

  def _slice_state_fn(x, batch_dim, time_dim):
    del batch_dim
    if time_dim < 0:
      return x
    # Allow jax/numpy convention on negative indices.
    start = slice_start
    if start < 0:
      start += x.shape[time_dim]
    limit = slice_limit
    if limit < 0:
      limit += x.shape[time_dim]
    return jax.lax.slice_in_dim(x, start, limit, axis=time_dim)

  return _slice_state_fn


def batch_broadcast_state_fn(
    multiplier: int) -> base_layer.DecodeStateTransformFn:
  """A function used to broadcast attention states on the batch dimension."""

  def _broadcast_state_fn(x, batch_dim, time_dim):
    del time_dim
    if batch_dim < 0:
      return x
    return jnp.repeat(x, multiplier, axis=batch_dim)

  return _broadcast_state_fn


def right_align_tensors(x: JTensor,
                        lengths: JTensor,
                        align_dim: int = 1) -> JTensor:
  """Aligns a tensor with padding to the right.

  x could have the following left align format:
  |---max_length---|
  [P, P, 0, 0, 0, 0]
  where lengths = 2, there are 4 paddings.

  After right align, x will have the following format:
  |---max_length---|
  [0, 0, 0, 0, P, P]

  Args:
    x: Tensors to right align with paddings on the right.
    lengths: JTensor [batch_size] with dtype jnp.int32, length of x without
      padding.
    align_dim: Dim to align, align_dim < len(x.shape).

  Returns:
    Right align x with shape [batch_size, seq_len].
  """
  if align_dim >= len(x.shape):
    raise ValueError(f'The align_dim: {align_dim} should be smaller than '
                     f'x.rank: {len(x.shape)}.')
  seq_len = x.shape[align_dim]

  def _align_one(x: JTensor, length: JTensor) -> JTensor:
    """Aligns a single tensor to the right, moving paddings to the left."""
    # Pad the tensor one the first dimension.
    paddings = [[0, 0]] * len(x.shape)
    paddings[0] = [seq_len, 0]
    padded = jnp.pad(x, paddings)

    # Slice out the right align tensor.
    start_indices = [0] * len(x.shape)
    start_indices[0] = length
    sizes = list(x.shape)
    sizes[0] = seq_len
    return jax.lax.dynamic_slice(padded, start_indices, sizes)

  return jax.vmap(_align_one)(x, lengths)


def right_align_state_fn(
    seq_lengths: JTensor) -> base_layer.DecodeStateTransformFn:
  """Returns a function that is used to right align attention states."""

  def _right_align_state_fn(x, batch_dim, time_dim):
    del batch_dim
    if time_dim < 0:
      return x
    return right_align_tensors(x, seq_lengths, time_dim)

  return _right_align_state_fn
