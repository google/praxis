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

"""Objective function for Connectionist Temporal Classification (CTC)."""

import dataclasses
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
from praxis import pytypes

JTensor = pytypes.JTensor


def _shift(values: np.ndarray, shift_amounts: np.ndarray, pad_value):
  """Shift values along the last axis.

  For the input:
    [[1, 2, 3, 4, 5]], shift_amounts[2], pad_value=0
  the result is:
    [[0, 0, 1, 2, 3]]

  This function is like roll(), except that we pad values instead of wrapping
  them around, and shift_amounts can be different for every row.

  Args:
    values: A >=2 dimensional array.
    shift_amounts: 1-dimensional array of per-row shift amounts.
    pad_value: Value to store in empty spaces created by the shift operation.

  Returns:
    See description above.
  """
  shift_amounts = shift_amounts.astype(jnp.int32)[..., jnp.newaxis]
  col_dim = values.shape[-1]
  rowranges = jnp.tile(jnp.arange(col_dim), values.shape[:-1] + (1,))
  column_indexes = rowranges - shift_amounts
  invalid_column_indexes = jnp.logical_or(
      column_indexes >= col_dim, column_indexes < 0
  )
  clamped_column_indexes = jnp.where(
      invalid_column_indexes, jnp.zeros_like(column_indexes), column_indexes
  )
  shifted_values = jnp.take_along_axis(values, clamped_column_indexes, axis=-1)
  shifted_values = jnp.where(
      invalid_column_indexes,
      jnp.ones_like(shifted_values) * pad_value,
      shifted_values,
  )
  return shifted_values


@dataclasses.dataclass
class CtcAlignments:
  """A set of arrays returned by ctc_loss_with_alignments.

  alignment: (B, T)-array of integers describing the best label alignment.
    0 means 'before the first label', >1 is means the 1-based label index.
  logalpha_phi: (T, B, N+1)-array of logprobs. In the alpha direction (left to
    right), this is the probability that we started the frame at a state,
    and remained there. State 0 is before the first label, state N is after
    the last label.
  logalpha_emit: (T, B, N)-array of logprobs. In the alpha direction (left to
    right), this is the probability that we emitted the Nth label at frame T.
  logbeta_phi: (T, B, N+1)-array of logprobs. This is like logalpha_phi but
    in the backward direction, starting from the end of the sequence and
    going toward the start.
  logbeta_emit: (T, B, N)-array of logprobs. Like logalpha_emit, but in the
    backward direction, starting from the end of the sequence and
    going toward the start.
  state_logprobs: (T, B, N+1)-array of logprobs of being at a particular
    label state, using 1-based label indexes. alignment is the argmax of this.
  """

  alignment: np.ndarray
  logalpha_phi: np.ndarray
  logalpha_emit: np.ndarray
  logbeta_phi: np.ndarray
  logbeta_emit: np.ndarray
  state_logprobs: np.ndarray


def ctc_loss_with_alignments(
    logits: Union[np.ndarray, JTensor],
    logitpaddings: Union[np.ndarray, JTensor],
    labels: Union[np.ndarray, JTensor],
    labelpaddings: Union[np.ndarray, JTensor],
    blank_id: int = 0,
    logepsilon: float = -1e5,
) -> tuple[np.ndarray, CtcAlignments]:
  """Forward and backward computation of CTC loss.

  This the same as ctc_loss above, but the loss is computed backward as well
  to produce sequence alignments.

  Args:
    logits: (B, T, K)-Array containing log-probabilities of the target class.
    logitpaddings: (B, T)-array. Padding array for `logprobs`.
    labels: (B, N)-array containing reference labels.
    labelpaddings: (B, N)-array. Paddings for `labels`. Currently `labels` must
      be right-padded, i.e. each row of labelspaddings must be repetition of
      zeroes, followed by repetition of ones. On the other hand, `logprobs` can
      have padded values at any position.
    blank_id: Id for blank token.
    logepsilon: log(epsilon), where epsilon is a tiny number. Used to
      approximate log(0).

  Returns:
    A pair of `(per_seq_loss, aux)`.
    per_seq_loss:
      (B,)-array containing loss values for each sequence in the batch.
    aux: Dictionary containing interim variables used for computing losses:
      alignment: (B, T)-array of integers describing the best label alignment.
        0 means 'before the first label', >1 is means the 1-based label index.
      state_logprobs: (T, B, N+1)-array of logprobs of being at a particular
        label state, using 1-based label indexes (same as alignment).
  """
  # Compute the probability of being at a particular state (label) starting
  # from the beginning of the sequence.
  # logalpha_phi is [T, B, N+1]
  # logalpha_emit is [T, B, N]
  per_seq_loss, logalpha_phi, logalpha_emit = optax.ctc_loss_with_forward_probs(
      logits,
      logitpaddings,
      labels,
      labelpaddings,
      blank_id,
      log_epsilon=logepsilon,
  )

  # Now we compute the same computation, but backwards; starting from the
  # end of the logits and end of the word, and working back toward the start.

  # ctc_loss works fine with logits padding at the start of the sequence, so
  # we just flip both on the time axis.
  # [B, T, K]
  reverse_logits = jnp.flip(logits, axis=1)
  reverse_logitpaddings = jnp.flip(logitpaddings, axis=1)

  # ctc_loss requires that the labels padding be at the end of the sequence,
  # so we flip the time axis but then shift the label sequence left.
  # In pseudocode, if the label sequence was 'hello    ',
  # we flip it to '    olleh', then move it left to become 'olleh    '.
  reverse_labels = jnp.flip(labels, axis=1)
  sum_labelpaddings = jnp.sum(labelpaddings, axis=1)
  reverse_labels = _shift(reverse_labels, -sum_labelpaddings, 0)  # pytype: disable=wrong-arg-types  # jnp-type

  # _, [T, B, N+1], [T, B, N]
  _, logbeta_reverse_phi, logbeta_reverse_emit = (
      optax.ctc_loss_with_forward_probs(
          reverse_logits,
          reverse_logitpaddings,
          reverse_labels,
          labelpaddings,
          blank_id,
          log_epsilon=logepsilon,
      )
  )

  # These results are backward twice: the time is backward and the label
  # sequence is backward. So we flip the time and label axes, then shift
  # them left (see the hello example, above).
  logbeta_phi = jnp.flip(jnp.flip(logbeta_reverse_phi, axis=0), axis=2)
  logbeta_phi = _shift(logbeta_phi, -sum_labelpaddings, logepsilon)  # pytype: disable=wrong-arg-types  # jnp-type
  logbeta_emit = jnp.flip(jnp.flip(logbeta_reverse_emit, axis=0), axis=2)
  logbeta_emit = _shift(logbeta_emit, -sum_labelpaddings, logepsilon)  # pytype: disable=wrong-arg-types  # jnp-type

  # The emit tensor is length N, one per label.
  # phi is length N+1; the 0 element is the probability of being in the
  # before-label state.
  # So the probability of being in state i is emit[i-1] + phi[i].
  def _state_logprob(emit, phi):
    return jnp.logaddexp(
        jnp.pad(emit, [[0, 0], [0, 0], [1, 0]], constant_values=logepsilon), phi
    )

  # [T, B, N]
  state_logprobs = (
      # Probability of being in state N at time T measured from sequence start.
      _state_logprob(logalpha_emit, logalpha_phi)
      +
      # Probability of being in state N at time T measured from sequence end.
      _state_logprob(logbeta_emit, logbeta_phi)
  )

  # If we take the argmax of state_logprobs, we usually get a valid alignment,
  # but sometimes it's not correct: sometimes labels are missing or appear
  # out of order. Therefore, we apply the argmax left-to-right to ensure
  # that all labels are selected.
  batch_size, logits_arraylen, _ = logits.shape
  _, labels_arraylen = labels.shape
  logits_lengths = logits_arraylen - jnp.sum(logitpaddings, axis=-1)
  logits_lengths = logits_lengths[:, np.newaxis]
  label_lengths = labels_arraylen - jnp.sum(labelpaddings, axis=-1)
  label_lengths = label_lengths[:, np.newaxis]

  # min_label is the minimum possible label index at each point in the
  # alignment, given that we have to output all of the labels.
  # Example: seq_len = 5, num_labels = 2:
  # min_label = [0, 0, 0, 1, 2]
  # min_label.shape = (batch, seq_len)
  logits_arange = jnp.tile(jnp.arange(0, logits_arraylen), (batch_size, 1))
  num_blanks = logits_lengths - label_lengths
  min_label = logits_arange - num_blanks
  min_label = jnp.where(min_label < 0, jnp.zeros_like(min_label), min_label)
  min_label = jnp.where(
      min_label > label_lengths,
      jnp.ones_like(min_label) * label_lengths,
      min_label,
  )

  def _pick_best_label(last_label, xs):
    logits_slice, min_label_slice = xs
    # (batch_size, num_classes), contents are the index of the class.
    classes_arange = jnp.tile(
        jnp.arange(0, labels_arraylen + 1), (batch_size, 1)
    )
    # The minimum label index we can use at this step. Alignments are
    # monotonic, so it has to be at least as large as the previous frame.
    # It also has to be as large as min_label_slice to ensure we don't run
    # out of time to emit all of the labels.
    local_min_label = jnp.where(
        last_label > min_label_slice, last_label, min_label_slice
    )
    masked_logits = jnp.where(
        classes_arange < local_min_label[:, np.newaxis],
        jnp.ones_like(logits_slice) * logepsilon,
        logits_slice,
    )
    # We can't skip labels, so the alignment index can't be more than 1
    # greater than the previous label output.
    masked_logits = jnp.where(
        classes_arange > (last_label[:, np.newaxis] + 1),
        jnp.ones_like(masked_logits) * logepsilon,
        masked_logits,
    )
    # label.shape = (batch_size)
    label = jnp.argmax(masked_logits, axis=-1)
    return label, label

  # state_logprobs.shape = (frames, batch, classes)
  # min_label.shape = (batch, frames)
  xs = (state_logprobs, jnp.transpose(min_label, (1, 0)))
  _, alignment = jax.lax.scan(
      _pick_best_label, jnp.zeros((batch_size), jnp.int32), xs
  )
  # (frames, batch) -> (batch, frames)
  alignment = jnp.transpose(alignment, (1, 0))

  return (
      per_seq_loss,
      CtcAlignments(  # type: ignore  # jnp-types
          alignment=alignment,
          logalpha_phi=logalpha_phi,
          logalpha_emit=logalpha_emit,
          logbeta_phi=logbeta_phi,
          logbeta_emit=logbeta_emit,
          state_logprobs=state_logprobs,
      ),
  )


def is_valid_ctc_seq(logitpaddings, labels, labelpaddings):
  """Returns for per example sequence if it passes validity check.

  Note that the above `ctc_loss_with_alignments` returns logeps
  (usually a very large number) if the input length is smaller than
  the label length plus number of consectutive duplications.
  However, in that case, we should ignore the loss.

  A validity check is passed if for an example when :
    input.length >= labels.length + num(consecutive dup label tokens)

  Args:
    logitpaddings:  [b, t], 0/1 JTensor.
    labels:         [b, t], int32 JTensor.
    labelpaddings:  [b, t], 0/1 JTensor.

  Returns:
    A shape [b] float tensor indicating if each (input, label) pair is valid,
      with a value of 1.0 indicating valid and 0.0 otherwise.
  """
  # [b]
  label_lengths = jnp.sum(1.0 - labelpaddings, axis=-1)
  # [b]
  input_lengths = jnp.sum(1.0 - logitpaddings, axis=-1)
  # [b, t-1]
  dups = (1.0 - labelpaddings[:, 1:]) * (labels[:, :-1] == labels[:, 1:])
  # [b]
  num_consecutive_dups = jnp.sum(dups, axis=-1)
  # [b]
  is_valid = (label_lengths + num_consecutive_dups) <= input_lengths
  is_valid = is_valid.astype(jnp.float32)
  return is_valid


def frame_alignment_to_label_alignment(
    frame_alignment: Union[np.ndarray, JTensor],
    frame_paddings: Union[np.ndarray, JTensor],
) -> tuple[JTensor, JTensor]:
  """Converts frame alignment to label alignment.

  Args:
    frame_alignment: (B, T)-array of integers describing the best label
      alignment to label, i.e., frame_alignment[b, t] is the index of label that
      aligned to b-th sequence, t-th frame;
    frame_paddings: (B, T)-array. Padding array for each frame

  Returns:
    label_alignment: (B, T)-array, label_alignment[b, n] means the starting
      frame index of the n-th label in the b-th sequence;
    label_lengths: (B,)-array. label lengths of each sequence; this is only
      used to verify `frame_alignment` is a valid input.

  Note: frame_alignment describes how the frame index is mapped to the label
    index; while label_alignment describes how the label index is mapped to the
    corresponding starting frames. The latter is more widely used to convert
    the alignment to timestamp of each output label.

    An example is:

      frame_alignment = [
          [0, 0, 0, 0, 1, 1, 2, 2, 3, 0, 0],
          [1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 0]
      ]
      frame_paddings = [
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
      ]

      This means that there are 9 and 10 frames in 2 sequences respectively.
      The `frame_alignment` provides the best alignment paths against 3 labels.
      For the first sequence, the 0st, 1st and 2nd label starts from 4-th,
      6-th and 8-th frame respectively; for the second sequence, the 0th, 1st,
      and 2nd label starts from 0-th, 3-rd and 6-th frame respectively, so the
      `label_alignment` =

      [
          [4, 6, 8, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 3, 6, 0, 0, 0, 0, 0, 0, 0, 0]
      ]
      and `label_length` = [3, 3]

      Note that `label_alignment` is padded to length T.

      It is user's responsibility to make sure the input consistency, i.e.,
      `jnp.max(frame_alignment, axis=-1) == label_lengths`.
  """
  # 0. get shape
  batch_size, max_num_frames = frame_alignment.shape
  # 1. right shift `frame_alignment`
  frame_alignment_rshift = jnp.concatenate(
      [
          jnp.full([batch_size, 1], fill_value=-1, dtype=jnp.int32),
          frame_alignment[:, :-1],
      ],
      axis=-1,
  )
  # 2. get `segment_boundary_mask`, segment_boundary_mask[b, t] means t-th
  # frame's label is different from its previous label or it is the sequence
  # start frame.
  segment_boundary_mask = jnp.not_equal(
      frame_alignment_rshift, frame_alignment
  ).astype(jnp.int32)
  segment_boundary_mask = segment_boundary_mask * (
      1 - frame_paddings.astype(jnp.int32)
  )
  # 3. also note that frame_alignment may contain 0s, which means those frames
  # are aligned to blank symbols before the 0-th label starts -- we don't need
  # them.
  normal_label_emitted = jnp.not_equal(frame_alignment, 0).astype(jnp.int32) * (
      1 - frame_paddings.astype(jnp.int32)
  )
  # now the `segment_boundary_mask`=1 means the corresponding frame has a label
  # emitted (the first frame in the segment)
  segment_boundary_mask = segment_boundary_mask * normal_label_emitted
  label_lengths = jnp.sum(segment_boundary_mask, axis=-1)

  label_alignment = jnp.tile(
      jnp.arange(max_num_frames, dtype=jnp.int32)[jnp.newaxis, :],
      jnp.array([batch_size, 1]),
  )
  label_alignment = jnp.where(
      segment_boundary_mask, label_alignment, max_num_frames + 1
  )
  label_alignment = jnp.sort(label_alignment, axis=-1)
  label_alignment = jnp.where(
      label_alignment == max_num_frames + 1, 0, label_alignment
  )

  return label_alignment, label_lengths


def forced_alignment(
    logits: Union[np.ndarray, JTensor],
    logitpaddings: Union[np.ndarray, JTensor],
    labels: Union[np.ndarray, JTensor],
    labelpaddings: Union[np.ndarray, JTensor],
    blank_id: int = 0,
) -> JTensor:
  """Given a pair of logits and labels sequence, calculate the best alignment.

  Args:
    logits: (B, T, K)-Array containing log-probabilities of the target class.
    logitpaddings: (B, T)-array. Padding array for `logprobs`.
    labels: (B, N)-array containing reference labels.
    labelpaddings: (B, N)-array. Paddings for `labels`. Currently `labels` must
      be right-padded, i.e. each row of labelspaddings must be repetition of
      zeroes, followed by repetition of ones. On the other hand, `logprobs` can
      have padded values at any position.
    blank_id: Id for blank token.

  Returns:
    label_alignment: (B, N)-array, label_alignment[b, n] means the starting
      frame index of the n-th label in the b-th sequence; Also note that due
      to the limit of CTC loss, it is possible that some sequence cannot find
      alignment, in that case, label_alignment[b, :] = -1.
  """
  max_label_lengths = labels.shape[-1]
  max_input_lengths = logitpaddings.shape[-1]
  _, aux = ctc_loss_with_alignments(
      logits, logitpaddings, labels, labelpaddings, blank_id
  )
  frame_alignment = aux.alignment
  label_alignment, _ = frame_alignment_to_label_alignment(
      frame_alignment=frame_alignment, frame_paddings=logitpaddings
  )
  # label_alignment now is a [B, T]-shaped tensor
  is_valid_seq = is_valid_ctc_seq(
      logitpaddings=logitpaddings, labels=labels, labelpaddings=labelpaddings
  )
  # is_valid_seq is now a [B,]-shaped tensor, broadcast to [B, T]
  is_valid = jnp.tile(is_valid_seq[:, jnp.newaxis], [1, max_input_lengths])
  label_alignment = jnp.where(is_valid, label_alignment, -1)
  # it is possible max_label_lengths > max_input_lengths, but it looks like jax
  # can hanle this situation.
  label_alignment = label_alignment[:, :max_label_lengths]
  return label_alignment
