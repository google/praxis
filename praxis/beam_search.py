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

"""Vanilla Beam search algorithm."""

from typing import Callable, Sequence

from flax import linen as nn
import jax
from jax import numpy as jnp
from praxis import base_layer
from praxis import decoder_hparams
from praxis import decoder_utils
from praxis import py_utils

NestedMap = py_utils.NestedMap
JTensor = base_layer.JTensor
BeamSearchHParams = decoder_hparams.BeamSearchHParams
GlobalBeam = tuple[
    JTensor,  # int[batch_size, beam_size, seq_len] Decoded IDs including </s>
    JTensor,  # int[batch_size, beam_size] Complete sequence lengths
    JTensor,  # float[batch_size, beam_size] Complete sequence scores
    JTensor,  # float[batch_size, beam_size, seq_len] Per-step log-probs
]
ComputeLogprobsFn = Callable[
    [
        JTensor,  # float[batch_size * beam_size, vocab_size] logits
        base_layer.BaseLayerApi,  # model
        JTensor,  # int[batch_size * beam_size] extend_ids
        JTensor,  # int[batch_size * beam_size] segment_pos
        NestedMap,  # decode_loop_state
    ],
    JTensor,  # float[batch_size * beam_size, vocab_size] logprobs
]


def update_global_beam(
    end_hyps: GlobalBeam, cur_hyps: GlobalBeam
) -> GlobalBeam:
  """Updates top-k beams of ended sequences.

  Args:
    end_hyps: Tuple of best completed sequences; see comments on `GlobalBeam`.
    cur_hyps: Tuple of new completed sequences to maximize in.

  Returns:
    Updated `end_hyps`.
  """
  (end_ids, end_lengths, end_scores, end_logprobs) = end_hyps
  (cur_ids, cur_lengths, cur_scores, cur_logprobs) = cur_hyps
  beam_dim = 1
  k = end_ids.shape[beam_dim]
  m = cur_ids.shape[beam_dim]
  ids = jnp.concatenate([end_ids, cur_ids], beam_dim)
  lengths = jnp.concatenate([end_lengths, cur_lengths], beam_dim)
  scores = jnp.concatenate([end_scores, cur_scores], beam_dim)
  logprobs = jnp.concatenate([end_logprobs, cur_logprobs], beam_dim)
  end_scores, indices = jax.lax.top_k(scores, k)
  one_hot = jax.nn.one_hot(indices, k + m, dtype=jnp.int32)
  end_ids = jnp.einsum("bkt,bjk->bjt", ids, one_hot)
  end_lengths = jnp.einsum("bk,bjk->bj", lengths, one_hot)
  end_logprobs = jnp.einsum("bkt,bjk->bjt", logprobs, one_hot)
  return (end_ids, end_lengths, end_scores, end_logprobs)


def shuffle_state(x: JTensor, hyp_id: JTensor, use_one_hot_matmul=False):
  """Shuffle cache state at beam dimension.

  Args:
    x: The decode state with shape [batch_size, ...].
    hyp_id: The desired beam ids with shape [batch_size, beam_size].
    use_one_hot_matmul: Use one hot matmul to shuffle decode state or not.

  Returns:
    A reshuffle of x on beam dimension.
  """
  # No need to shuffle R1 tensor such as time_step.
  if not isinstance(x, JTensor) or len(x.shape) < 2:
    return x

  if use_one_hot_matmul:
    one_hot_matrix = jax.nn.one_hot(hyp_id, hyp_id.shape[1], dtype=x.dtype)
    return jnp.einsum("bk...,bjk->bj...", x, one_hot_matrix)
  return jax.vmap(
      lambda s, i: jnp.take(s, i, axis=0), in_axes=0, out_axes=0)(x, hyp_id)


def broadcast_beam_dim(x: JTensor, beam_dim: int, beam_size: int) -> JTensor:
  """Broadcasts the tensor beam_size times at beam dimension.

  Args:
    x: The input tensor of shape [batch_size, ...].
    beam_dim: Beam dimension to insert.
    beam_size: Beam size in beam search.

  Returns:
    A tensor with `beam_size` copies of `x` along the newly-inserted `beam_dim`
    axis.  Its shape depends on `beam_dim`; for example,
      if beam_dim=0: [beam_size, batch_size, ...]
      if beam_dim=1: [batch_size, beam_size, ...]
  """
  return jnp.repeat(
      jnp.expand_dims(x, beam_dim), repeats=beam_size, axis=beam_dim)


def default_compute_logprobs_fn(
    logits: JTensor,
    model: base_layer.BaseLayerApi,
    extend_ids: JTensor,
    segment_pos: JTensor,
    decode_loop_state: NestedMap,
) -> JTensor:
  """Default `ComputeLogprobsFn` that calls log_softmax()."""
  del model, extend_ids, segment_pos, decode_loop_state
  return jax.nn.log_softmax(logits)


# TODO(b/249483164): Rename BaseLayerApi->BaseLayer after Fiddle migration.
def beam_search(
    model: base_layer.BaseLayerApi,
    extend_step_fn: decoder_utils.ExtendStepFn
    | decoder_utils.ExpandedExtendStepFn,
    fprop_fn: decoder_utils.FPropFn,
    transform_state_fn: decoder_utils.TransformStateFn,
    prefix_ids: JTensor,
    prefix_paddings: JTensor,
    beam_search_hparams: BeamSearchHParams,
    compute_logprobs_fn: ComputeLogprobsFn = default_compute_logprobs_fn,
    decode_loop_mesh_axes_transpose: dict[str, str] | None = None,
    model_var_pspecs: base_layer.NestedPartitionSpec | None = None,
    process_result_fn: decoder_utils.ProcessResultFn | None = None,
    lazy_broadcast_prefix_fn: decoder_utils.LazyBroadcastPrefixFn | None = None,
) -> NestedMap:
  """Vanilla beam search decode the input batch.

  Args:
    model: The model object.
    extend_step_fn: A function that takes in the decoded sequence at the current
      time step (with shape [B] or [B, P] where B corresponds to the batch size
      and P corresponds to a possible prefix) and returns `JTensor` corresponds
      to the logits of the next step.  The following signatures are allowed:
      extend_step_fn(model, extend_ids, segment_pos)
      extend_step_fn(model, extend_ids, segment_pos, decode_loop_state)
    fprop_fn: A function that takes in the prefix information and initialize the
      decode cache states.
    transform_state_fn: A function that transforms the decode state.
    prefix_ids: The token ids that correspond to the prefix sequence, with shape
      [batch_size, prefix_sequence_length].
    prefix_paddings: The token paddings that correspond to the prefix sequence,
      with shape [batch_size, prefix_sequence_length].
    beam_search_hparams: Beam search hyper parameters.
    compute_logprobs_fn: Computes log-probabilities from logits.
    decode_loop_mesh_axes_transpose: Optional mesh transpose for decoding loop.
    model_var_pspecs: needed if decode_loop_mesh_axes_transpose is provided.
    process_result_fn: Optional function that further processes the results,
      such as performing suffix scoring.

  Returns:
    A NestedMap with `.decode_lengths` (vector of ints indicating the lengths
    of non-padding tokens in `.output_ids`, which includes the prefix),
    `.output_ids` (matrix of int ids with the
    decoded output), `.scores` (Scores of decoding sequence), `.log_probs`
    (matrix of floats indicating log probabilities for each output),
    `prefix_lengths` (vector of ints for prefix length),
    `prefix_ids` (matrix of ints for prefix ids).
  """

  # Init decode state using fprop_fn, state seq size is max_prefix_len.
  fprop_fn(model, prefix_ids, prefix_paddings)
  model = decoder_utils.maybe_reshard_mdl_for_decode(
      model,
      decode_loop_mesh_axes_transpose,
      model_var_pspecs,
      transform_state_fn,
  )
  with decoder_utils.maybe_decode_mesh_transpose(
      model, decode_loop_mesh_axes_transpose
  ):
    result = beam_search_after_prefix_fprop(
        model,
        extend_step_fn,
        transform_state_fn,
        prefix_ids,
        prefix_paddings,
        beam_search_hparams,
        compute_logprobs_fn,
        lazy_broadcast_prefix_fn,
    )
    if process_result_fn is not None:
      result = process_result_fn(model, result)
    return result


# TODO(b/249483164): Rename BaseLayerApi->BaseLayer after Fiddle migration.
def beam_search_after_prefix_fprop(
    model: base_layer.BaseLayerApi,
    extend_step_fn: decoder_utils.ExtendStepFn
    | decoder_utils.ExpandedExtendStepFn,
    transform_state_fn: decoder_utils.TransformStateFn,
    prefix_ids: JTensor,
    prefix_paddings: JTensor,
    beam_search_hparams: BeamSearchHParams,
    compute_logprobs_fn: ComputeLogprobsFn = default_compute_logprobs_fn,
    lazy_broadcast_prefix_fn: decoder_utils.LazyBroadcastPrefixFn | None = None,
) -> NestedMap:
  """Same as beam_search but this is after prefix fprop."""
  # TODO(b/229679837): Move right align prefix ids and paddings logic inside
  # the beam_search function.

  # max_decode_steps doesn't count the prefix part.
  assert beam_search_hparams.max_decode_steps is not None
  max_decode_steps = beam_search_hparams.max_decode_steps
  max_decode_steps = (
      [max_decode_steps]
      if isinstance(max_decode_steps, int)
      else max_decode_steps
  )
  max_decode_steps = sorted(max_decode_steps)
  beam_dim = 1
  beam_size = beam_search_hparams.beam_size
  batch_size = prefix_ids.shape[0]
  max_prefix_len = prefix_ids.shape[1]
  terminal_ids = (
      beam_search_hparams.eos_id
      if isinstance(beam_search_hparams.eos_id, Sequence)
      else [beam_search_hparams.eos_id]
  )
  seq_len = max(max_decode_steps) + max_prefix_len
  if lazy_broadcast_prefix_fn is not None:
    # We need to exclude the last token from prefix, and instead move it to
    # the multi-sample suffix. This is because the last token only as an Input
    # ID, but not an output ID (label), and we need to start decoding from it.
    transform_state_fn(model, decoder_utils.slice_state_fn(0, -1))
    first_decode_steps = min(max_decode_steps)
    # max_decode_steps + 1 to include last token from prefix.
    lazy_broadcast_prefix_fn(model, beam_size, first_decode_steps + 1)

  else:
    # Pad max_decode_steps to the state.
    transform_state_fn(model, decoder_utils.pad_state_fn(min(max_decode_steps)))

    # Broadcast cache states before the while loop.
    def _broadcast_state_fn(x, batch_dim, time_dim):
      del time_dim
      return jnp.repeat(x, repeats=beam_size, axis=batch_dim)

    transform_state_fn(model, _broadcast_state_fn)

  # Set up init loop variables.
  val = NestedMap()
  loop_state_shape = (batch_size, beam_size)
  val.output_ids = jnp.zeros(
      shape=(batch_size, beam_size, seq_len), dtype=jnp.int32)
  val.end_scores_norm = jnp.zeros(shape=loop_state_shape, dtype=jnp.float32)
  val.end_scores_norm -= 1e9
  val.hyp_scores = jnp.zeros(shape=loop_state_shape, dtype=jnp.float32)
  # Penalize all hyps except the first
  val.hyp_scores -= jnp.arange(beam_size, dtype=jnp.float32) * 1e9
  val.hyp_ids = jnp.zeros(shape=loop_state_shape, dtype=jnp.int32)
  val.logprobs = jnp.ones(
      shape=(batch_size, beam_size, seq_len), dtype=jnp.float32)
  val.end_logprobs = jnp.zeros(
      shape=(batch_size, beam_size, seq_len), dtype=jnp.float32
  )
  # Whether the hyp has terminated and should stop.
  val.done = jnp.zeros(shape=loop_state_shape, dtype=jnp.bool_)

  # Gets prefix_lengths from prefix_paddings.
  prefix_lengths = jnp.sum(1 - prefix_paddings.astype(jnp.int32), axis=1)
  # [batch, beam]
  prefix_lengths = broadcast_beam_dim(
      prefix_lengths, beam_dim=beam_dim, beam_size=beam_size)
  # Update output_ids with prefix_ids.
  # [batch, beam, prefix_seq_len]
  prefix_ids = broadcast_beam_dim(
      prefix_ids, beam_dim=beam_dim, beam_size=beam_size)
  val.output_ids = jax.lax.dynamic_update_slice(val.output_ids, prefix_ids,
                                                [0] * val.output_ids.ndim)
  val.end_ids = val.output_ids
  # Update loop init states with prefix.
  val.start_step = max_prefix_len - 1
  val.step = val.start_step
  val.segment_pos = jnp.reshape(prefix_lengths - 1, (batch_size * beam_size,))
  val.end_decode_lengths = jnp.ones_like(prefix_lengths) * seq_len

  # Get an `ExpandedExtendStepFn`, regardless of which variant was passed in.
  expanded_extend_step_fn = decoder_utils.coerce_to_expanded_extend_step_fn(
      extend_step_fn
  )

  def get_cond_func(stop_decode_steps, early_exit):
    """Get condition function for given stop decode steps."""

    def cond_func(model, val):
      """Whether the while loop should continue."""
      del model
      # We continue the decoding search iff both:
      #   (1) We have yet to exceed the max steps.
      #   (2) At least one row in the batch has not terminated.
      max_steps = max_prefix_len + stop_decode_steps - 1
      length_ok = val.step < min(seq_len - 1, max_steps)
      if not early_exit:
        return length_ok
      all_hyps_done = jnp.all(val.done)
      return jnp.logical_and(length_ok, jnp.logical_not(all_hyps_done))

    return cond_func

  def loop_body(model, val):
    """From ids at `step`, update output ids at `step + 1`."""
    step = val.step
    extend_ids = jnp.reshape(val.output_ids[:, :, step], (-1,))
    logits = expanded_extend_step_fn(model, extend_ids, val.segment_pos, val)
    logprobs = compute_logprobs_fn(
        logits.astype(jnp.float32), model, extend_ids, val.segment_pos, val
    )
    logprobs = jnp.reshape(logprobs, (batch_size, beam_size, -1))

    # Before reaching the minimum number of decode steps required (default: 0),
    # prevent EOS from getting selected by setting its logprobs to a very
    # negative value.
    def prevent_terminal_ids(x):
      for terminal_id in terminal_ids:
        x = x.at[:, :, terminal_id].set(-1e20)
      return x

    if beam_search_hparams.min_decode_steps > 0:
      logprobs = jax.lax.cond(
          step >= max_prefix_len - 1 + beam_search_hparams.min_decode_steps,
          lambda x: x,
          prevent_terminal_ids,
          logprobs,
      )

    # Select the best ids with terminal tokens.
    eos_scores = jnp.ones_like(val.hyp_scores) * -1e9
    new_end_ids = val.output_ids
    new_end_logprobs = val.logprobs
    for terminal_id in terminal_ids:
      eos_scores_candidate = val.hyp_scores + logprobs[:, :, terminal_id]
      new_best = eos_scores_candidate > eos_scores
      eos_scores = jnp.maximum(eos_scores, eos_scores_candidate)
      new_end_ids = new_end_ids.at[:, :, step + 1].set(
          jnp.where(
              new_best,
              terminal_id * jnp.ones_like(prefix_lengths),
              new_end_ids[:, :, step + 1],
          )
      )
      new_end_logprobs = new_end_logprobs.at[:, :, step + 1].set(
          jnp.where(
              new_best,
              logprobs[:, :, terminal_id],
              new_end_logprobs[:, :, step + 1],
          )
      )
    decode_length = step + 2
    new_decode_lengths = decode_length - max_prefix_len + prefix_lengths
    eos_scores_norm = eos_scores / decoder_utils.length_norm(
        decode_length - max_prefix_len, beam_search_hparams.length_norm_alpha)
    (val.end_ids, val.end_decode_lengths, val.end_scores_norm,
     val.end_logprobs) = update_global_beam(
        (val.end_ids, val.end_decode_lengths, val.end_scores_norm,
         val.end_logprobs),
        (new_end_ids, new_decode_lengths, eos_scores_norm, new_end_logprobs))

    # early_exit doesn't explore non-EOS hyps.
    topk_terminal_ids = [] if beam_search_hparams.early_exit else terminal_ids
    # Choose the topk indices.
    tokens_per_beam = (
        beam_size
        if beam_search_hparams.tokens_per_beam is None
        else beam_search_hparams.tokens_per_beam
    )
    _, topk_indices, final_topk_value, final_topk_indices = (
        decoder_utils.two_stage_topk(
            logprobs, val.hyp_scores, topk_terminal_ids, tokens_per_beam
        )
    )
    # update scores with or without EOS depending on early_exit.
    val.hyp_scores = final_topk_value
    hyp_id = final_topk_indices // tokens_per_beam
    val.hyp_ids = hyp_id

    use_one_hot_matmul = beam_search_hparams.use_matmul_beam_shuffle
    # Shuffle at beam dimension for the cache states using hyp_id.
    def _shuffle_state_fn(x, batch_dim, time_dim):
      del time_dim
      if lazy_broadcast_prefix_fn is not None:
        new_state = shuffle_state(x, hyp_id, use_one_hot_matmul)
        return new_state
      else:
        x_shape = x.shape
        new_shape = list(x_shape)
        new_shape.insert(batch_dim + 1, beam_size)
        new_shape[batch_dim] = x.shape[batch_dim] // beam_size
        new_state = shuffle_state(
            jnp.reshape(x, new_shape), hyp_id, use_one_hot_matmul
        )
        return jnp.reshape(new_state, x_shape)

    transform_state_fn(model, _shuffle_state_fn)

    # Gather output ids
    # new_ids [batch_size, beam_size]
    new_ids = decoder_utils.gather_output_id(topk_indices, final_topk_indices)
    new_logprobs = decoder_utils.gather_logprobs(logprobs, hyp_id, new_ids)

    # Shuffle output ids at beam dimension using hyp_id.
    val.output_ids = shuffle_state(val.output_ids, hyp_id, use_one_hot_matmul)
    val.logprobs = shuffle_state(val.logprobs, hyp_id, use_one_hot_matmul)
    val.done = shuffle_state(val.done, hyp_id, use_one_hot_matmul)
    # Update output_ids.
    val.output_ids = val.output_ids.at[:, :, step + 1].set(new_ids)
    val.logprobs = val.logprobs.at[:, :, step + 1].set(new_logprobs)
    for terminal_id in terminal_ids:
      has_eos = decoder_utils.has_any_eos(new_ids, terminal_id)
      val.done = jnp.logical_or(val.done, has_eos)
    val.step += 1
    val.segment_pos += 1
    return val

  # Beam search loop. Cache state is broacasted before the while loop.
  result = val
  for i in range(len(max_decode_steps)):
    if i > 0:
      pad_size = max_decode_steps[i] - max_decode_steps[i - 1]
      transform_state_fn(model, decoder_utils.pad_state_fn(pad_size))
    result = nn.while_loop(
        get_cond_func(
            max_decode_steps[i],
            beam_search_hparams.early_exit,
        ),
        loop_body,
        model,
        result,
        split_rngs={base_layer.RANDOM: True},
        carry_variables=[base_layer.DECODE_CACHE],
    )

  prefix_ids = decoder_utils.left_align_tensor(prefix_ids[:, 0, :],
                                               prefix_lengths[:, 0],
                                               max_prefix_len)
  prefix_ids = jnp.expand_dims(prefix_ids, 1)

  result.output_ids = result.end_ids
  result.output_ids = decoder_utils.left_align_tensor(
      jnp.reshape(result.output_ids, (batch_size * beam_size, -1)),
      jnp.reshape(prefix_lengths, (-1)), max_prefix_len)
  result.output_ids = jnp.reshape(result.output_ids,
                                  (batch_size, beam_size, -1))
  result.scores = result.end_scores_norm
  result.logprobs = result.end_logprobs
  result.logprobs = decoder_utils.left_align_tensor(
      jnp.reshape(result.logprobs, (batch_size * beam_size, -1)),
      jnp.reshape(prefix_lengths, (-1)), max_prefix_len)
  result.logprobs = jnp.reshape(result.logprobs, (batch_size, beam_size, -1))
  result.decode_lengths = result.end_decode_lengths
  result.original_lengths = prefix_lengths
  result.prefix_lengths = prefix_lengths
  result.prefix_ids = prefix_ids
  del (
      result.end_ids,
      result.end_decode_lengths,
      result.end_scores_norm,
      result.hyp_scores,
      result.step,
      result.start_step,
  )
  return result
