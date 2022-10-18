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

from typing import Callable, List, Tuple

import jax
from jax import numpy as jnp
from praxis import base_layer
from praxis import pytypes

# TODO(b/249483164): Rename BaseLayerApi->BaseLayer after Fiddle migration.
JTensor = pytypes.JTensor
ExtendStepFn = Callable[[base_layer.BaseLayerApi, JTensor, JTensor], JTensor]
FPropFn = Callable[[base_layer.BaseLayerApi, JTensor, JTensor], None]
TransformStateFn = Callable[
    [base_layer.BaseLayerApi, base_layer.DecodeStateTransformFn], None]
# lazy_broadcast_prefix_fn(model, num_suffix_samples, suffix_length)
LazyBroadcastPrefixFn = Callable[[base_layer.BaseLayerApi, int, int], None]


def length_norm(t, length_norm_alpha) -> jnp.ndarray:
  """Length norm for beam search."""
  return jax.lax.pow((t.astype(jnp.float32) + 5.) / 5., length_norm_alpha)


def gather_output_id(long_output_ids: jnp.ndarray,
                     topk_indices: jnp.ndarray) -> jnp.ndarray:
  """Gather output ids from long topk output ids.

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


def gather_logprobs(logprobs: jnp.ndarray,
                    hyp_ids: jnp.ndarray,
                    ids: jnp.ndarray) -> jnp.ndarray:
  """Gather logprobs from output ids.

  Args:
    logprobs: The log probability of shape [batch_size, beam_size, vocab_size].
    hyp_ids: The hyp ids of shape [batch_size, beam_size]
    ids: The output ids of shape [batch_size, beam_size].

  Returns:
    Final output log prob of shape [batch_size, beam_size].
  """
  new_logprobs = jax.vmap(
      lambda s, i: jnp.take(s, i, axis=0), in_axes=0, out_axes=0)(
          logprobs, hyp_ids)
  one_hot = jax.nn.one_hot(
      ids, logprobs.shape[-1], dtype=logprobs.dtype)
  output_logprobs = jnp.einsum('bkv,bkv->bk', new_logprobs, one_hot)
  return output_logprobs


def two_stage_topk(
    logits: jnp.ndarray, hyp_scores: jnp.ndarray, terminal_ids: List[int]
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Two stage TopK to choose TopK values and indices from each beam.

  Args:
    logits: The logits of [batch_size, beam_size, vocab_size or beam_size *
      vocab_size].
    hyp_scores: The topK scores of [batch_size, beam_size].
    terminal_ids: terminal ids. In most cases this is simply eos_id.

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
  for terminal_id in terminal_ids:
    topk_value -= 1e9 * jnp.equal(topk_indices, terminal_id).astype(
        topk_value.dtype)

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


def left_align_tensor(x: JTensor,
                      prefix_lengths: JTensor,
                      max_prefix_len: int,
                      pad_value: float = 0.0) -> JTensor:
  """Changes middle aligned sequence to be left aligned.

  x has the following middle aligned format:
  |-max_prefix_len--|
  [0, 0, 0, 0, P, P, X, X, X, 0, 0, 0]
  where prefix_lengths = 2, max_prefix_len = 6, there are 4 paddings in the
  prefix.

  After left aligned, x will have the following format:
  |-max_prefix_len--|
  [P, P, X, X, X, 0, 0, 0, 0, 0, 0, 0]

  Args:
    x: Tensor of shape [batch_size, seq_len].
    prefix_lengths: prefix lengths of shape [batch_size].
    max_prefix_len: max prefix lengths.
    pad_value: Value for padding.

  Returns:
    Left aligned tensor with shape [batch_size, seqlen].
  """
  if len(x.shape) != 2:
    raise ValueError(
        f'Argument `x` needs to be 2-index, but has shape: {x.shape}')

  seqlen = x.shape[1]

  def _align_one(x: JTensor, prefix_length: JTensor) -> JTensor:
    """Aligns one middle align tensor to be left align."""
    padded = jnp.pad(
        x, [[0, max_prefix_len]],
        mode='constant',
        constant_values=x.dtype.type(pad_value))
    return jax.lax.dynamic_slice(padded, [max_prefix_len - prefix_length],
                                 [seqlen])

  return jax.vmap(_align_one)(x, prefix_lengths)


def concat_suffix_and_left_align(decoded_tensors: JTensor,
                                 suffix_tensors: JTensor,
                                 decode_end_indices: JTensor,
                                 prefix_lengths: JTensor, max_prefix_len: int,
                                 num_samples: int, num_suffix: int,
                                 pad_value: float):
  """Concatenates suffix tensor to decoded tensor and then left aligns.


  When batch_size = 1, num_samples = 1, num_suffix = 1 if decoded_tensors has
  the following middle align format:
  |-max_prefix_len--|
  [0, 0, 0, 0, P, P, X, X, X, 0, 0, 0]
  where prefix_lengths = 2, max_prefix_len = 6, end_decode_indices = 8, and
  suffix_id = [S, S]

  After concat and left aligned, return tensor will have the following format:
  |-max_prefix_len--|
  [P, P, X, X, X, S, S, 0, 0, 0, 0, 0]

  Args:
    decoded_tensors: JTensor generated from decoding with shape
      [batch_size*num_samples, seq_len].
    suffix_tensors: Suffix JTensor to append, has shape
      [batch_size * num_samples * num_suffix, suffix_len].
    decode_end_indices: Indices after last decode token position in decoded
      tensor, JTensor of shape [batch_size * num_samples],
    prefix_lengths: Prefix lengths of shape [batch_size * num_samples].
    max_prefix_len: Max prefix length.
    num_samples: Number of samples.
    num_suffix: Number of suffixes.
    pad_value: Value for padding.

  Returns:
    The left aligned tensor has suffix_tensor concatenated to the back of
    decoded_tensor.
  """
  suffix_batch_size, suffix_length = suffix_tensors.shape

  # [batch_size * num_samples * num_suffix, seq_len]
  broadcast_decoded_tensors = jnp.repeat(
      decoded_tensors, repeats=num_suffix, axis=0)

  # [batch_size * num_samples * num_suffix, seq_len + suffix_len]
  padded_decoded_tensors = jnp.pad(
      broadcast_decoded_tensors, [[0, 0], [0, suffix_length]],
      mode='constant',
      constant_values=decoded_tensors.dtype.type(pad_value))

  def _update_one(x: JTensor, suffix: JTensor, start: JTensor) -> JTensor:
    """Concats suffix to x at start index."""
    return jax.lax.dynamic_update_slice(x, suffix, [start])

  # Concat suffix tensors to the back of decoded tensors.
  concat_tensors = jax.vmap(_update_one)(padded_decoded_tensors,
                                         suffix_tensors,
                                         jnp.repeat(
                                             decode_end_indices,
                                             repeats=num_suffix,
                                             axis=0))
  # Left align concatenated tensors.
  left_align_tensors = left_align_tensor(
      concat_tensors,
      jnp.repeat(prefix_lengths, repeats=num_suffix, axis=0),
      max_prefix_len,
      pad_value=pad_value)
  # Reshape to [batch_size, num_samples, num_suffix, seq_len + suffix_len]
  reshaped_tensors = jnp.reshape(left_align_tensors, [
      suffix_batch_size //
      (num_samples * num_suffix), num_samples, num_suffix, -1
  ])

  # Output has [batch_size, num_suffix, num_samples, seq_len + suffix_len]
  return jnp.transpose(reshaped_tensors, (0, 2, 1, 3))
