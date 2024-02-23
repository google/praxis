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

"""Util functions for decoder."""

import dataclasses
import functools
import inspect
from typing import Callable, Sequence, cast

from flax import core as flax_core
import jax
from jax import lax
from jax import numpy as jnp
from praxis import base_layer
from praxis import py_utils
from praxis import pytypes


# TODO(b/249483164): Rename BaseLayerApi->BaseLayer after Fiddle migration.
JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor
NestedMap = pytypes.NestedMap
ExtendStepFn = Callable[[base_layer.BaseLayerApi, JTensor, JTensor], JTensor]
ExpandedExtendStepFn = Callable[
    [
        base_layer.BaseLayerApi,  # model
        JTensor,  # int[batch_size] or int[batch_size * beam_size] extend_ids
        JTensor,  # int[batch_size] or int[batch_size * beam_size] segment_pos
        NestedMap,  # decode_loop_state
    ],
    JTensor,  # logits
]
FPropFn = Callable[[base_layer.BaseLayerApi, JTensor, JTensor], None]
TransformStateFn = Callable[
    [base_layer.BaseLayerApi, base_layer.DecodeStateTransformFn], None
]
# lazy_broadcast_prefix_fn(model, num_suffix_samples, suffix_length)
LazyBroadcastPrefixFn = Callable[[base_layer.BaseLayerApi, int, int], None]
ProcessResultFn = Callable[[base_layer.BaseLayerApi, NestedMap], NestedMap]
# Dummy prng key to avoid deterministic random seed from sample decode input.
DUMMY_PRNG_KEY = 15753


@dataclasses.dataclass(frozen=True)
class StreamingResultCallback:
  """Callback to be invoked for every N steps of decoding with its results."""

  # Callable to be called once every `interval_steps` decoding steps. Accepts
  # decoding results as the argument.
  callback_fn: Callable[[NestedJTensor], None]

  # Number of steps between decoding callback invocations.
  interval_steps: int = 1

  # Optional callable to be called at the beginning of decoding. Accepts batch
  # size (`batch` or `[batch, num_samples]`) as the argument.
  init_fn: Callable[[int | Sequence[int]], None] | None = None

  # Optional callable to be called at the end of decoding.
  done_fn: Callable[[], None] | None = None


@dataclasses.dataclass
class ControlledDecodingHParams:
  # Number of steps between applying blockwise controlled decoding.
  # Set to 0 to disable.
  interval: int = 0

  # Number of samples within each block for controlled decoding reranking.
  # Only used if interval is > 0.
  block_num_samples: int = 0


def length_norm(t, length_norm_alpha) -> jnp.ndarray:
  """Length norm for beam search."""
  return jax.lax.pow((t.astype(jnp.float32) + 5.0) / 5.0, length_norm_alpha)


def gather_output_id(
    long_output_ids: jnp.ndarray, topk_indices: jnp.ndarray
) -> jnp.ndarray:
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
      topk_indices, long_output_ids.shape[-1], dtype=jnp.int32
  )
  output_id = jnp.einsum(
      'bt,bkt->bk', long_output_ids.astype(jnp.int32), one_hot
  )
  return output_id.astype(jnp.int32)


def gather_logprobs(
    logprobs: jnp.ndarray, hyp_ids: jnp.ndarray, ids: jnp.ndarray
) -> jnp.ndarray:
  """Gather logprobs from output ids.

  Args:
    logprobs: The log probability of shape [batch_size, beam_size, vocab_size].
    hyp_ids: The hyp ids of shape [batch_size, beam_size]
    ids: The output ids of shape [batch_size, beam_size].

  Returns:
    Final output log prob of shape [batch_size, beam_size].
  """
  new_logprobs = jax.vmap(
      lambda s, i: jnp.take(s, i, axis=0), in_axes=0, out_axes=0
  )(logprobs, hyp_ids)
  one_hot = jax.nn.one_hot(ids, logprobs.shape[-1], dtype=logprobs.dtype)
  output_logprobs = jnp.einsum('bkv,bkv->bk', new_logprobs, one_hot)
  return output_logprobs


def two_stage_topk(
    logits: jnp.ndarray,
    hyp_scores: jnp.ndarray,
    terminal_ids: list[int],
    tokens_per_beam: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Two stage TopK to choose TopK values and indices from each beam.

  Args:
    logits: The logits of [batch_size, beam_size, vocab_size or beam_size *
      vocab_size].
    hyp_scores: The topK scores of [batch_size, beam_size].
    terminal_ids: terminal ids. In most cases this is simply eos_id.
    tokens_per_beam: Number of tokens to explore per beam, defaulting to
      beam_size if None.

  Returns:
    topk_value, topk_indices of shape [batch_size, beam_size * tokens_per_beam],
      they are the topk of `logits` on the last dimenension and merged on the
      last two dimensions.
    final_topk_value, final_topk_indices of shape [batch_size, beam_size],
      they are the topk value and inidices of `topk_value`.
  """
  vocab_size = logits.shape[-1]
  batch_size = hyp_scores.shape[0]
  beam_size = hyp_scores.shape[1]
  tokens_per_beam = beam_size if tokens_per_beam is None else tokens_per_beam
  logits_reshape = jnp.reshape(
      logits, newshape=(batch_size * beam_size, vocab_size)
  )
  topk_value, topk_indices = jax.lax.top_k(logits_reshape, tokens_per_beam)
  topk_value = jnp.reshape(
      topk_value, newshape=(batch_size, beam_size, tokens_per_beam)
  )
  topk_indices = jnp.reshape(
      topk_indices, newshape=(batch_size, beam_size, tokens_per_beam)
  )
  topk_value += jnp.expand_dims(hyp_scores, -1)
  for terminal_id in terminal_ids:
    topk_value -= 1e9 * jnp.equal(topk_indices, terminal_id).astype(
        topk_value.dtype
    )

  topk_value = jnp.reshape(
      topk_value, newshape=(batch_size, beam_size * tokens_per_beam)
  )
  topk_indices = jnp.reshape(
      topk_indices, newshape=(batch_size, beam_size * tokens_per_beam)
  )

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


def slice_state_fn(
    slice_start: int, slice_limit: int
) -> base_layer.DecodeStateTransformFn:
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
    multiplier: int,
) -> base_layer.DecodeStateTransformFn:
  """A function used to broadcast attention states on the batch dimension."""

  def _broadcast_state_fn(x, batch_dim, time_dim):
    del time_dim
    if batch_dim < 0:
      return x
    return jnp.repeat(x, multiplier, axis=batch_dim)

  return _broadcast_state_fn


def right_align_tensors(
    x: JTensor, lengths: JTensor, align_dim: int = 1
) -> JTensor:
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
    raise ValueError(
        f'The align_dim: {align_dim} should be smaller than '
        f'x.rank: {len(x.shape)}.'
    )
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
    seq_lengths: JTensor,
) -> base_layer.DecodeStateTransformFn:
  """Returns a function that is used to right align attention states."""

  def _right_align_state_fn(x, batch_dim, time_dim):
    del batch_dim
    if time_dim < 0:
      return x
    return right_align_tensors(x, seq_lengths, time_dim)

  return _right_align_state_fn


def left_align_tensor(
    x: JTensor,
    prefix_lengths: JTensor,
    max_prefix_len: int,
    pad_value: float = 0.0,
) -> JTensor:
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
        f'Argument `x` needs to be 2-index, but has shape: {x.shape}'
    )

  seqlen = x.shape[1]

  def _align_one(x: JTensor, prefix_length: JTensor) -> JTensor:
    """Aligns one middle align tensor to be left align."""
    padded = jnp.pad(
        x,
        [[0, max_prefix_len]],
        mode='constant',
        constant_values=x.dtype.type(pad_value),
    )
    return jax.lax.dynamic_slice(
        padded, [max_prefix_len - prefix_length], [seqlen]
    )

  return jax.vmap(_align_one)(x, prefix_lengths)


def left_align_kv_cache(
    x: JTensor,
    prefix_lengths: JTensor,
    max_prefix_len: int,
    pad_value: float = 0.0,
    batch_size: int = 1,
) -> JTensor:
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
    x: Tensor of shape [batch_size, seq_len, num_heads, head_dim].
    prefix_lengths: prefix lengths of shape [batch_size].
    max_prefix_len: max prefix lengths.
    pad_value: Value for padding.
    batch_size: x.shape[0] in int.

  Returns:
    Left aligned tensor with shape [batch_size, seqlen, num_heads, head_dim].
  """
  rank = len(x.shape)
  slice_sizes = x.shape[1:]
  if rank not in [3, 4]:
    raise ValueError(
        f'Argument `x` needs to be 3 or 4-index, but has shape: {x.shape}'
    )
  pad_width = [[0, max_prefix_len], [0, 0]]
  if rank == 4:
    pad_width = pad_width + [[0, 0]]
  for i in range(batch_size):
    start_indices = [max_prefix_len - prefix_lengths[i], 0]
    if rank == 4:
      start_indices = start_indices + [0]
    padded = jnp.pad(
        x[i],
        pad_width,
        mode='constant',
        constant_values=x.dtype.type(pad_value),
    )
    padded = jax.lax.dynamic_slice(padded, start_indices, slice_sizes)
    x = x.at[i].set(padded)
  return x


def concat_suffix_and_left_align(
    decoded_tensors: JTensor,
    suffix_tensors: JTensor,
    decode_end_indices: JTensor,
    prefix_lengths: JTensor,
    max_prefix_len: int,
    num_samples: int,
    num_suffix: int,
    pad_value: float,
):
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
    suffix_tensors: Suffix JTensor to append, has shape [batch_size *
      num_samples * num_suffix, suffix_len].
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
      decoded_tensors, repeats=num_suffix, axis=0
  )

  # [batch_size * num_samples * num_suffix, seq_len + suffix_len]
  padded_decoded_tensors = jnp.pad(
      broadcast_decoded_tensors,
      [[0, 0], [0, suffix_length]],
      mode='constant',
      constant_values=decoded_tensors.dtype.type(pad_value),
  )

  def _update_one(x: JTensor, suffix: JTensor, start: JTensor) -> JTensor:
    """Concats suffix to x at start index."""
    return jax.lax.dynamic_update_slice(x, suffix, [start])

  # Concat suffix tensors to the back of decoded tensors.
  concat_tensors = jax.vmap(_update_one)(
      padded_decoded_tensors,
      suffix_tensors,
      jnp.repeat(decode_end_indices, repeats=num_suffix, axis=0),
  )
  # Left align concatenated tensors.
  left_align_tensors = left_align_tensor(
      concat_tensors,
      jnp.repeat(prefix_lengths, repeats=num_suffix, axis=0),
      max_prefix_len,
      pad_value=pad_value,
  )
  # Reshape to [batch_size, num_samples, num_suffix, seq_len + suffix_len]
  reshaped_tensors = jnp.reshape(
      left_align_tensors,
      [
          suffix_batch_size // (num_samples * num_suffix),
          num_samples,
          num_suffix,
          -1,
      ],
  )

  # Output has [batch_size, num_suffix, num_samples, seq_len + suffix_len]
  return jnp.transpose(reshaped_tensors, (0, 2, 1, 3))


def maybe_reshard_mdl_for_decode(
    model: base_layer.BaseLayer,
    mesh_transpose: dict[str, str] | None,
    model_var_pspecs: base_layer.NestedPartitionSpec | None,
    transform_decode_state_fn: TransformStateFn,
) -> base_layer.BaseLayer:
  """Reshards model variables if mesh_transpose is given."""
  if mesh_transpose is None or model.is_initializing():
    return model
  lm_vars = flax_core.unfreeze(model.variables)  # pytype: disable=wrong-arg-types
  assert model_var_pspecs is not None

  def _annotate():
    for col in (base_layer.PARAMS, base_layer.NON_TRAINABLE):
      if col not in model_var_pspecs:
        continue
      assert col in lm_vars, lm_vars.keys()

      def _shard(x, s):
        return base_layer.maybe_shard(x, list(s), model.mesh_axis_names)

      lm_vars[col] = jax.tree_util.tree_map(
          _shard,
          lm_vars[col],
          model_var_pspecs[col],
      )

  # Original sharding.
  _annotate()
  # Transposed sharding.
  with maybe_decode_mesh_transpose(model, mesh_transpose):
    _annotate()

  resharded_model = model.bind(
      lm_vars,
      rngs={base_layer.RANDOM: model.next_prng_key()},
      mutable=True,
  )

  def _identity_fn(x, batch_dim, time_dim):
    del batch_dim, time_dim
    return x

  # Trigger sharding annotation on decode cache. The transform function is an
  # identity, but we expect individual layers to annotate the new state in
  # transform_decode_state_fn.
  transform_decode_state_fn(resharded_model, _identity_fn)
  with maybe_decode_mesh_transpose(model, mesh_transpose):
    transform_decode_state_fn(resharded_model, _identity_fn)
  return resharded_model


def maybe_decode_mesh_transpose(
    model: base_layer.BaseLayer, mesh_transpose: dict[str, str] | None
) -> base_layer.JaxContext:
  """Creates a new JaxContext with mesh_transpose."""
  if base_layer.JaxContext.has_context():
    new_context_params = base_layer.cur_jax_context().hparams.clone()
  else:
    new_context_params = base_layer.JaxContext.HParams()
  if mesh_transpose is not None and not model.is_initializing():
    new_context_params.mesh_axes_transpose = mesh_transpose
  return base_layer.JaxContext.new_context(hparams=new_context_params)


def find_first_new_stop_seq_match(
    first_new_decode_idx: JTensor,
    num_new_tokens: int,
    stop_sequences: JTensor,
    sequences: JTensor,
):
  """Finds index of first stop sequence match in newly decoded tokens.

  *The returned index is relative to the first new decode index.*

  Example:
    first_new_decode_idx = [2, 0, 2]
    num_new_tokens = 2
    stop_sequences = [[[3, 4, 5], [0, 2, 6]],
                      [[3, 4, 5], [0, 2, 6]]]
    sequences = [[3, 4, 5, 0],
                 [3, 4, 5, 0],
                 [4, 2, 6, 7]]

    * batch element 0: we look for a match starting from index 2. The
        subsequence ending at index 2 is [3, 4, 5], which matches stop sequence
        0. Relative to the first new decode index (2), the match index is 0.
    * batch element 1: we look for a match starting from index 0. The
        subsequence ending at index 0 is [3], doesn't match any stop sequence.
        At position 1 the subsequence is [3, 4] -> no match. The number of new
        tokens is 2, so we stop with no match and return 2.
    * batch element 2: we look for a match starting from index 2. The
        subsequence ending at index 2 is [4, 2, 6], which matches stop sequence
        1. Relative to the first new decode index (2), the match index is 0.

  Args:
    first_new_decode_idx: [B], idx of first new decode
    num_new_tokens: scalar, number of new decoded tokens
    stop_sequences: [B, num_stop_sequences, max_stop_seq_len]
    sequences: [B, max_seq_len]

  Returns:
    Tensor of shape [B]. Indices of the first stop sequence match for each batch
    element, *relative to their respective first new decode indices*.
  """

  # [B, num_new_tokens]
  col_idxs = (
      first_new_decode_idx[:, jnp.newaxis]
      + jnp.arange(num_new_tokens)[jnp.newaxis, :]
  )

  # First find all idx, stop_seq, seq triples that match.
  # [B, num_new_tokens, num_stop_sequences]
  matches = end_with_any_sequence_any_position(
      stop_sequences, sequences, col_idxs
  )

  # For any idx and sequence, we only care if _any_ stop sequence matched.
  # [B, num_new_tokens]
  any_eos_match = jax.numpy.any(matches, axis=-1, keepdims=False)

  # Some funky way to find the index of the first true value along the axis.
  # We add first new decode idx because the indices we tried for each sequence
  # don't start at 0.
  # If the value is equal to the number of sequences, no match was found.
  # [B]
  first_stop_seq_hit = jnp.sum(
      jax.lax.cummin(1 - any_eos_match.astype(jnp.int32), axis=1), axis=-1
  )

  return first_stop_seq_hit


def _end_with_sequence_single(
    stop_sequence: JTensor,
    output_ids: JTensor,
    decode_step: int | JTensor,
) -> JTensor:
  """Check if the output_ids end with given sequence.

  The stop_sequence tensor is a 1D tensor. If you want to use the vmapped
  version of this function, and the original stop_sequences is
  [[2], [3, 4], [5, 5, 5]], it should be padded to
  [[0, 0, 2], [0, 3, 4], [5, 5, 5]] before passed to this function.

  The comparison is performed by matching the tokens of output_ids ending at
  index 'decode_step' with the tokens in `stop_sequence`.

  Args:
    stop_sequence: Given end of sequences of shape [eos_len].
    output_ids: Generated output ids of shape [seq_len].
    decode_step: Current decode step as an int or a 0D tensor.

  Returns:
    A JTensor of rank 0 which indicates if the output_ids ended with
    stop_sequences.
  """

  eos_len = stop_sequence.shape[0]
  padded_output_ids = jnp.pad(output_ids, [eos_len, 0])
  # Slice start index = decode_step + eos_len - eos_len + 1.
  sliced_output_ids = jax.lax.dynamic_slice(
      padded_output_ids, [decode_step + 1], [eos_len]
  )

  # stop_sequences are padded from the left with 0s.
  ignore_tokens = jnp.equal(stop_sequence, 0)

  tokens_equal = jnp.logical_or(
      jnp.equal(sliced_output_ids, stop_sequence), ignore_tokens
  )
  return jnp.all(tokens_equal, axis=-1)


def end_with_sequences(
    stop_sequences: JTensor, output_ids: JTensor, decode_step: int | JTensor
) -> JTensor:
  """Applies _end_with_sequence_single to a batch of (stop_seq, sequence) pairs."""
  return jax.vmap(_end_with_sequence_single, in_axes=(0, 0, None))(
      stop_sequences, output_ids, decode_step
  )


@functools.partial(jax.vmap, in_axes=(0, 0, 0))  # map over batch
@functools.partial(jax.vmap, in_axes=(None, None, 0))  # map over decode step
@functools.partial(jax.vmap, in_axes=(0, None, None))  # map over stop sequence
def end_with_any_sequence_any_position(stop_sequence, output_ids, decode_step):
  """Checks for matches of stop_sequence in output_ids at at decode_step.

  The un-vmapped function will return true if stop_sequence matches
  output_ids[decode_step-len(stop_seq):decode_step].

  The vmapped function allows you to pass multiple stop sequences, check
  multiple column indices, and do this for all items in a batch.

  Args:
    stop_sequence: stop sequences to look for of shape [B, num_stop_seqs,
      max_stop_seq_len].
    output_ids: sequences, of shape [B, seq_len].
    decode_step: step to check for matches, shape [B, num_positions_to_check].

  Returns:
    Tensor of shape [B, num_positions_to_check, num_stop_seqs] denoting whether
    there was a match for each position to check in the sequence with each
    stop sequence.
  """

  return _end_with_sequence_single(stop_sequence, output_ids, decode_step)


def has_any_eos(arr: JTensor, eos_ids: int | Sequence[int]):
  """Check if the given array contains any of the eos_ids."""
  eos = jnp.array(eos_ids, dtype=jnp.int32).reshape([1] * arr.ndim + [-1])
  return jnp.any(jnp.equal(arr[..., jnp.newaxis], eos), axis=-1)


def coerce_to_expanded_extend_step_fn(
    extend_step_fn: ExtendStepFn | ExpandedExtendStepFn,
) -> ExpandedExtendStepFn:
  """Wraps or casts the `extend_step_fn` into an `ExpandedExtendStepFn`."""
  if len(inspect.signature(extend_step_fn).parameters) == 4:
    return cast(ExpandedExtendStepFn, extend_step_fn)

  extend_step_fn = cast(ExtendStepFn, extend_step_fn)

  def _expanded_extend_step_fn(
      model: base_layer.BaseLayerApi,
      extend_ids: JTensor,
      segment_pos: JTensor,
      decode_loop_state: NestedMap,
  ) -> JTensor:
    del decode_loop_state
    return extend_step_fn(model, extend_ids, segment_pos)

  return _expanded_extend_step_fn


def collect_results_to_optimize_eos(
    result: NestedMap, decode_length_shift: int = 0
) -> NestedMap:
  """Collects decoding results when optimize_eos=True."""
  new_result = result.DeepCopy()
  cumulative_logprobs = jnp.cumsum(new_result.logprobs, -1)
  valid_logprobs = jnp.where(
      jnp.arange(new_result.logprobs.shape[1]) <= result.start_step,
      jnp.zeros_like(new_result.logprobs),
      cumulative_logprobs,
  )
  eos_logprobs = jnp.where(
      jnp.arange(new_result.eos_logprobs.shape[1]) <= result.start_step,
      jnp.ones_like(new_result.eos_logprobs)
      * py_utils.get_large_negative_number(jnp.float32),
      new_result.eos_logprobs,
  )
  end_logprobs = (
      jnp.pad(valid_logprobs, [[0, 0], [1, 0]])[:, :-1] + eos_logprobs
  )
  best_pos = jnp.argmax(end_logprobs, -1)
  batch_dim = jnp.arange(new_result.output_ids.shape[0])
  new_result.output_ids = new_result.output_ids.at[batch_dim, best_pos].set(
      new_result.eos_ids[batch_dim, best_pos]
  )
  new_result.output_ids = jnp.where(
      jnp.arange(new_result.output_ids.shape[1])[jnp.newaxis, :]
      > best_pos[:, jnp.newaxis],
      jnp.zeros_like(new_result.output_ids),
      new_result.output_ids,
  )
  new_result.logprobs = new_result.logprobs.at[batch_dim, best_pos].set(
      new_result.eos_logprobs[batch_dim, best_pos]
  )
  new_result.logprobs = jnp.where(
      jnp.arange(new_result.logprobs.shape[1])[jnp.newaxis, :]
      > best_pos[:, jnp.newaxis],
      jnp.ones_like(new_result.logprobs),
      new_result.logprobs,
  )
  new_result.decode_lengths = best_pos + 1 - decode_length_shift
  new_result.done = jnp.ones_like(new_result.done)
  new_result.has_eos = jnp.ones_like(new_result.has_eos)
  del new_result.eos_logprobs
  del new_result.eos_ids
  return new_result
