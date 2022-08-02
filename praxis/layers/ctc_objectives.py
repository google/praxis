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

"""Objective function for Connectionist Temporal Classification (CTC)."""

import dataclasses
from typing import Tuple

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
  clamped_column_indexes = jnp.where(
      jnp.logical_or(column_indexes >= col_dim, column_indexes <= 0),
      jnp.zeros_like(column_indexes), column_indexes)
  shifted_values = jnp.take_along_axis(values, clamped_column_indexes, axis=-1)
  shifted_values = jnp.where(column_indexes >= col_dim,
                             jnp.ones_like(shifted_values) * pad_value,
                             shifted_values)
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
    logits: np.ndarray,
    logitpaddings: np.ndarray,
    labels: np.ndarray,
    labelpaddings: np.ndarray,
    blank_id: int = 0,
    logepsilon: float = -1e5) -> Tuple[np.ndarray, CtcAlignments]:
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
      logits, logitpaddings, labels, labelpaddings, blank_id)

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
  reverse_labels = _shift(reverse_labels, -sum_labelpaddings, 0)

  # _, [T, B, N+1], [T, B, N]
  _, logbeta_reverse_phi, logbeta_reverse_emit = optax.ctc_loss_with_forward_probs(
      reverse_logits, reverse_logitpaddings, reverse_labels, labelpaddings,
      blank_id)

  # These results are backward twice: the time is backward and the label
  # sequence is backward. So we flip the time and label axes, then shift
  # them left (see the hello example, above).
  logbeta_phi = jnp.flip(jnp.flip(logbeta_reverse_phi, axis=0), axis=2)
  logbeta_phi = _shift(logbeta_phi, -sum_labelpaddings, logepsilon)
  logbeta_emit = jnp.flip(jnp.flip(logbeta_reverse_emit, axis=0), axis=2)
  logbeta_emit = _shift(logbeta_emit, -sum_labelpaddings, logepsilon)

  # The emit tensor is length N, one per label.
  # phi is length N+1; the 0 element is the probability of being in the
  # before-label state.
  # So the probability of being in state i is emit[i-1] + phi[i].
  def _state_logprob(emit, phi):
    return jnp.logaddexp(
        jnp.pad(emit, [[0, 0], [0, 0], [1, 0]], constant_values=logepsilon),
        phi)

  # [T, B, N]
  state_logprobs = (
      # Probability of being in state N at time T measured from sequence start.
      _state_logprob(logalpha_emit, logalpha_phi) +
      # Probability of being in state N at time T measured from sequence end.
      _state_logprob(logbeta_emit, logbeta_phi))

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
  min_label = jnp.where(min_label > label_lengths,
                        jnp.ones_like(min_label) * label_lengths, min_label)

  def _pick_best_label(last_label, xs):
    logits_slice, min_label_slice = xs
    # (batch_size, num_classes), contents are the index of the class.
    classes_arange = jnp.tile(
        jnp.arange(0, labels_arraylen + 1), (batch_size, 1))
    # The minimum label index we can use at this step. Alignments are
    # monotonic, so it has to be at least as large as the previous frame.
    # It also has to be as large as min_label_slice to ensure we don't run
    # out of time to emit all of the labels.
    local_min_label = jnp.where(last_label > min_label_slice, last_label,
                                min_label_slice)
    masked_logits = jnp.where(classes_arange < local_min_label[:, np.newaxis],
                              jnp.ones_like(logits_slice) * logepsilon,
                              logits_slice)
    # We can't skip labels, so the alignment index can't be more than 1
    # greater than the previous label output.
    masked_logits = jnp.where(classes_arange > (last_label[:, np.newaxis] + 1),
                              jnp.ones_like(masked_logits) * logepsilon,
                              masked_logits)
    # label.shape = (batch_size)
    label = jnp.argmax(masked_logits, axis=-1)
    return label, label

  # state_logprobs.shape = (frames, batch, classes)
  # min_label.shape = (batch, frames)
  xs = (state_logprobs, jnp.transpose(min_label, (1, 0)))
  _, alignment = jax.lax.scan(_pick_best_label,
                              jnp.zeros((batch_size), jnp.int32), xs)
  # (frames, batch) -> (batch, frames)
  alignment = jnp.transpose(alignment, (1, 0))

  return per_seq_loss, CtcAlignments(
      alignment=alignment,
      logalpha_phi=logalpha_phi,
      logalpha_emit=logalpha_emit,
      logbeta_phi=logbeta_phi,
      logbeta_emit=logbeta_emit,
      state_logprobs=state_logprobs)
