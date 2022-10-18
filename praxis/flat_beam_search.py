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

"""Flat beam search algorithm."""

from typing import Any, Optional, Tuple

from flax import linen as nn
import jax
from jax import numpy as jnp
from praxis import base_layer
from praxis import decoder_utils
from praxis import py_utils

NestedMap = py_utils.NestedMap
JTensor = base_layer.JTensor


def init_loop_var(target_prefix_ids: JTensor, beam_size: int, seq_len: int,
                  fprop_dtype: Any) -> NestedMap:
  """Init loop variable for beam search.

  Args:
    target_prefix_ids: The prefix token ids of shape [batch_size,
      target_sequence_length], that correspond to the target sequence.
    beam_size: Beam size of beam search.
    seq_len: The output sequence length to decode to.
    fprop_dtype: fprop data type.

  Returns:
    A NestedMap with
      - output_ids: Matrix of int ids with the decoded output, shape
        [seq_length, batch_size, beam_size].
      - hyp_scores: Best scores without EOS, shape [batch_size, beam_size].
      - beam_mask: Beam mask without EOS of shape
        [batch_size, beam_size, beam_size * seq_len].
      - end_scores: Best scores with EOS, shape [batch_size, beam_size].
      - end_mask: Beam mask with EOS of shape
        [batch_size, beam_size, beam_size * seq_len].
      - end_scores_norm: Best scores with EOS and length norm, shape
        [batch_size, beam_size].
  """
  batch_size = target_prefix_ids.shape[0]
  # The last elements of target_prefix_ids becomes the the first input
  # for the decoding loop.
  prefix_last = target_prefix_ids[:, -1]
  buf_size = beam_size * seq_len
  new_ids = jnp.zeros(
      shape=(batch_size, beam_size), dtype=jnp.int32) + jnp.expand_dims(
          prefix_last, -1)
  output_ids = jnp.zeros(
      shape=(seq_len, batch_size, beam_size), dtype=jnp.int32)
  output_ids = output_ids.at[0].set(new_ids)

  val = NestedMap()
  val.step = 0
  val.output_ids = output_ids
  val.beam_mask = jnp.zeros(
      shape=(batch_size, beam_size, buf_size), dtype=fprop_dtype)
  val.beam_mask += jax.nn.one_hot(jnp.arange(beam_size), buf_size)
  val.hyp_scores = jnp.zeros(shape=(batch_size, beam_size), dtype=fprop_dtype)
  # penalize all hyps except the first
  val.hyp_scores -= jnp.arange(beam_size, dtype=fprop_dtype) * 1e9

  val.end_mask = jnp.zeros(
      shape=(batch_size, beam_size, buf_size), dtype=fprop_dtype)
  val.end_scores = jnp.zeros(shape=(batch_size, beam_size), dtype=fprop_dtype)
  val.end_scores -= 1e9
  val.end_scores_norm = val.end_scores

  # Shape [batch_size, beam_size], whether each row has terminated and
  # should stop.
  val.done = jnp.zeros(shape=(batch_size, beam_size), dtype=jnp.bool_)
  return val


def update_beam_mask(beam_mask: jnp.ndarray, hyp_id: jnp.ndarray,
                     time_step: Optional[int]) -> jnp.ndarray:
  """Update beam search output id mask.

  Args:
    beam_mask: The output ids mask of shape [batch_size, beam_size, beam_size *
      seq_len].
    hyp_id: The topK beam ids.
    time_step: current time step.

  Returns:
    Updated beam_mask.
  """
  # 1st step: reorder, the second part of the beam_mask is an identify matrix.
  # beam_mask = [A I O]
  # einsum(beam_mask, one_hot) = [einsum(A, one_hot) one_hot O]
  buf_size = beam_mask.shape[-1]
  beam_size = beam_mask.shape[1]
  beam_mask = jnp.einsum('bkt,bjk->bjt', beam_mask,
                         jax.nn.one_hot(hyp_id, beam_size))

  # 2nd step: add identity matrix to the new step.
  if time_step is not None:
    beam_mask += jax.nn.one_hot(
        jnp.arange(beam_size) + time_step * beam_size, buf_size)
  return beam_mask


def get_final_output_ids(beam_mask: jnp.ndarray,
                         output_ids: jnp.ndarray) -> jnp.ndarray:
  """Gather final output ids beam search masks and output_ids.

  If beam mask is
  [0, 0, 1, 0, 0, 0, 0, 1]
  [0, 0, 1, 0, 0, 0, 1, 0]
  [1, 0, 0, 0, 1, 0, 0, 0]
  [0, 0, 1, 0, 0, 1, 0, 0]

  output_ids are
  [0, 5]
  [1, 6]
  [2, 7]
  [3, 8]

  final_output_ids will be
  [2, 8]
  [2, 7]
  [0, 5]
  [2, 6]

  Args:
    beam_mask: The first TopK indices from [batch_size, beam_size, beam_size *
      seq_len].
    output_ids: The output_ids of shape [batch_size, beam_size, seq_len].

  Returns:
    Final output ids of shape [batch_size, beam_size, seq_len].
  """
  batch_size = output_ids.shape[0]
  beam_size = output_ids.shape[1]
  seq_len = output_ids.shape[2]

  beam_mask_reshape = jnp.reshape(beam_mask,
                                  (batch_size, beam_size, seq_len, beam_size))
  return jnp.einsum('bkl, bnlk->bnl', output_ids,
                    beam_mask_reshape).astype(jnp.int32)


def update_topk_scores_with_eos(
    end_hyps: Tuple[jnp.ndarray, jnp.ndarray,
                    jnp.ndarray], cur_hyps: Tuple[jnp.ndarray, jnp.ndarray,
                                                  jnp.ndarray]
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Updates topk scores with eos.

  Args:
    end_hyps: Tuple (mask, score, sore_norms) with eos.
    cur_hyps: Tuple (mask, score, score_norms) without eos.

  Returns:
    Updated end_hyps..
  """
  (end_mask, end_scores, end_scores_norm) = end_hyps
  (cur_mask, cur_scores, cur_scores_norm) = cur_hyps
  k = end_mask.shape[1]
  m = cur_mask.shape[1]
  mask = jnp.concatenate([end_mask, cur_mask], 1)
  scores = jnp.concatenate([end_scores, cur_scores], 1)
  scores_norm = jnp.concatenate([end_scores_norm, cur_scores_norm], 1)
  end_scores_norm, indices = jax.lax.top_k(scores_norm, k)
  one_hot = jax.nn.one_hot(indices, k + m)
  end_mask = jnp.einsum('bkt,bjk->bjt', mask, one_hot)
  end_scores = jnp.einsum('bk,bjk->bj', scores, one_hot)
  return (end_mask, end_scores, end_scores_norm)


def post_process(result: NestedMap, eos_id: int,
                 max_decoding_steps: int) -> NestedMap:
  """Post process of beam search result after the loop.

  Args:
    result: NestedMap for loop vars.
    eos_id: Id for EOS.
    max_decoding_steps: max decoding steps.

  Returns:
    Final processed result of beam search.
  """
  result.output_ids = jnp.transpose(result.output_ids, (1, 2, 0))
  result.output_ids = get_final_output_ids(result.end_mask, result.output_ids)
  result.decode_lengths = jnp.sum(result.end_mask, -1)
  result.output_ids += eos_id * jax.nn.one_hot(
      result.decode_lengths, max_decoding_steps, dtype=jnp.int32)
  result.decode_lengths += 1
  result.decode_lengths = jnp.minimum(result.decode_lengths,
                                      max_decoding_steps).astype(jnp.int32)
  result.scores = result.end_scores_norm

  del result.step, result.done, result.end_scores_norm
  return result


# TODO(b/249483164): Rename BaseLayerApi->BaseLayer after Fiddle migration.
def flat_beam_search(model: base_layer.BaseLayerApi,
                     extend_step_fn: decoder_utils.ExtendStepFn,
                     target_prefix_ids: JTensor,
                     target_prefix_paddings: JTensor,
                     seq_len: int,
                     beam_size: int,
                     fprop_dtype: Any,
                     max_decode_steps: Optional[int] = None,
                     eos_id: Optional[int] = None,
                     length_norm_alpha: float = 0.8) -> NestedMap:
  """Beam search decode the input batch.

  Args:
    model: The model object.
    extend_step_fn: A function that takes in `states` and the decoded sequence
      at the current time step (with shape [B] or [B, P] where B corresponds to
      the batch size and P corresponds to a possible prefix) and returns a tuple
      of (`NestedMap`, `JTensor`), where the first `NestedMap` corresponds to
      the `new_states` and the second `JTensor` corresponds to the logits of the
      next step.
    target_prefix_ids: The token ids that correspond to the target sequence,
      with shape [batch_size, prefix_sequence_length].
    target_prefix_paddings: The paddings corresponding to the target sequence,
      with a 1 denoting padding token and 0 denoting non-padding tokens.
    seq_len: The output sequence length to decode to.
    beam_size: Beam size of beam search.
    fprop_dtype: fprop data type.
    max_decode_steps: Python int or None, the max decode step to run after the
      prefix (if any). Since the prefixes might be of unequal lengths, this
      value is not equivalent with `seq_len` above. When None, decode steps is
      only limited by `seq_len` above.
    eos_id: Optional EOS id which to terminate the decoding early.
    length_norm_alpha: alpha parameter of length norm.

  Returns:
    A NestedMap with `.decode_lengths` (vector of ints indicating the lengths
    of non-padding tokens in `.output_ids`, which includes the prefix)`,
    `.output_ids` (matrix of int ids with the
    decoded output), `.scores` (Scores of decoding sequence).
  """

  # TODO(wangtao): Add decoding logic with prefix target_ids
  # TODO(wangtao): Add decoding logic with prefix target_paddings
  # TODO(wangtao): Add decoding logic with prefix_lengths
  # TODO(wangtao): Add prefix loop
  # TODO(wangtao): Update state in prefix loop
  del target_prefix_paddings
  if seq_len <= 0:
    raise ValueError('The sequence length for decoding must be > 0, '
                     f'current value = {seq_len}.')
  max_decode_steps = max_decode_steps or seq_len
  val = init_loop_var(target_prefix_ids, beam_size, seq_len, fprop_dtype)

  def cond_func(model, val):
    """Whether the while loop should continue."""
    del model
    length_ok = val.step < seq_len - 1
    all_rows_done = jnp.all(val.done)
    return jnp.logical_and(length_ok, jnp.logical_not(all_rows_done))

  def loop_body(model, val):
    """From ids at `step`, update output ids at `step + 1`."""
    step = val.step
    logits = extend_step_fn(model, val.output_ids[step], None)
    eos_scores = val.hyp_scores + logits[:, eos_id]
    eos_scores_norm = eos_scores / decoder_utils.length_norm(
        step + 1, length_norm_alpha)
    updated_topk_scores = update_topk_scores_with_eos(
        (val.end_mask, val.end_scores, val.end_scores_norm),
        (val.beam_mask, val.hyp_scores, eos_scores_norm))
    val.end_mask, val.end_scores, val.end_scores_norm = updated_topk_scores

    _, topk_indices, final_topk_value, final_topk_indices = (
        decoder_utils.two_stage_topk(logits, val.hyp_scores, [eos_id]))
    # update scores
    val.hyp_scores = final_topk_value
    hyp_id = final_topk_indices // beam_size

    # update beam_mask
    val.beam_mask = update_beam_mask(val.beam_mask, hyp_id, step + 1)

    # Gather output ids
    new_ids = decoder_utils.gather_output_id(topk_indices, final_topk_indices)

    # TODO(wangtao): add logic for prefix to make sure the tokens are
    # right-aligned so that convolutions are allowed.
    prev_done = val.done
    new_ids = jnp.where(prev_done, jnp.zeros_like(new_ids), new_ids)
    if eos_id is not None:
      val.done = jnp.logical_or(prev_done, jnp.equal(new_ids, eos_id))
    val.output_ids = val.output_ids.at[step + 1].set(new_ids)
    val.step += 1
    return val

  # Beam search loop.
  result = nn.while_loop(
      cond_func,
      loop_body,
      model,
      val,
      carry_variables=[base_layer.DECODE_CACHE])

  # Get final output ids from flat beam search.
  return post_process(result, eos_id, max_decode_steps)
