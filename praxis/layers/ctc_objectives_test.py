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

"""Tests for ctc_objectives."""

from absl.testing import absltest
import jax
from jax.experimental import jax2tf
import jax.numpy as jnp
import numpy as np
import optax
from praxis import pytypes
from praxis import test_utils
from praxis.layers import ctc_objectives
import tensorflow as tf

JTensor = pytypes.JTensor


def tf_ctc_loss(logits: np.ndarray,
                logits_paddings: np.ndarray,
                labels: np.ndarray,
                labels_paddings: np.ndarray,
                blank_id: int = 0):
  assert blank_id == 0

  def tf_ctc_loss_wrapper(logits, logits_paddings, labels, labels_paddings):
    labels = tf.cast(labels, tf.int32)
    logit_length = tf.cast(
        tf.reduce_sum(1.0 - logits_paddings, axis=-1), tf.int32)
    label_length = tf.cast(
        tf.reduce_sum(1.0 - labels_paddings, axis=-1), tf.int32)
    return tf.nn.ctc_loss(
        labels=labels,
        logits=logits,
        label_length=label_length,
        logit_length=logit_length,
        logits_time_major=False)

  return jax2tf.call_tf(tf_ctc_loss_wrapper)(logits, logits_paddings, labels,
                                             labels_paddings)


def average_ctc_loss(logprobs: JTensor, logprob_paddings: JTensor,
                     labels: JTensor, label_paddings: JTensor) -> JTensor:
  return jnp.average(
      optax.ctc_loss(logprobs, logprob_paddings, labels, label_paddings))


def lengths_to_paddings(lengths: JTensor, maxlength: int) -> JTensor:
  indices = jnp.arange(maxlength).reshape((1,) * lengths.ndim + (maxlength,))
  lengths = jnp.expand_dims(lengths, axis=-1)
  elem_valid = indices < lengths
  return np.logical_not(elem_valid).astype(np.float32)


class CtcTest(test_utils.TestCase):
  """Tests for the CTC loss function.

  CTC loss is no longer a part of Pax, still this test is remaining to ensure
  the upstream is compatible with our usage.
  """

  def setUp(self):
    super().setUp()
    np.random.seed(1234)

  def test_against_tf_ctc_loss(self):
    batchsize = 8
    timesteps = 150
    labelsteps = 25
    nclasses = 400
    logits = np.random.randn(batchsize, timesteps, nclasses)
    logprobs = jax.nn.log_softmax(logits)
    logprob_paddings = np.zeros((batchsize, timesteps))
    labels = np.random.randint(
        1, nclasses, size=(batchsize, labelsteps)).astype(np.int32)
    label_paddings = np.zeros((batchsize, labelsteps))

    inputs = [logprobs, logprob_paddings, labels, label_paddings]

    jax_per_seq = optax.ctc_loss(*inputs)
    tf_per_seq = tf_ctc_loss(*inputs)
    self.assertAllClose(jax_per_seq.squeeze(), tf_per_seq.squeeze())

    average_tf_ctc_loss = lambda *args: jnp.average(tf_ctc_loss(*args))
    jax_dloss = jax.grad(average_ctc_loss)
    tf_dloss = jax.grad(average_tf_ctc_loss)

    jax_dlogits = jax_dloss(*inputs)
    tf_dlogits = tf_dloss(*inputs)
    # Relative error check is disabled as numerical errors explodes when a
    # probability computed from the input logits is close to zero.
    self.assertAllClose(jax_dlogits, tf_dlogits, rtol=0.0, atol=1e-4)

  def test_against_tf_ctc_loss_with_paddings(self):
    batchsize = 8
    timesteps = 150
    labelsteps = 25
    nclasses = 400

    logits = np.random.randn(batchsize, timesteps, nclasses)
    logprobs = jax.nn.log_softmax(logits)
    logprob_lens = np.random.randint(25, timesteps - 3, size=(batchsize,))
    logprob_paddings = lengths_to_paddings(logprob_lens, timesteps)

    labels = np.random.randint(
        1, nclasses, size=(batchsize, labelsteps)).astype(np.int32)
    label_lens = np.random.randint(10, labelsteps, size=(batchsize,))
    label_paddings = lengths_to_paddings(label_lens, labelsteps)

    inputs = [logprobs, logprob_paddings, labels, label_paddings]

    jax_per_seq = optax.ctc_loss(*inputs)
    tf_per_seq = tf_ctc_loss(*inputs)
    self.assertAllClose(jax_per_seq.squeeze(), tf_per_seq.squeeze())


class CtcAlignmentsTest(test_utils.TestCase):

  def test_ctc_loss_with_alignments(self):
    label_sequence = [
        0, 0, 0, 0, 5, 0, 0, 7, 0, 8, 0, 0, 9, 9, 0, 14, 0, 0, 12, 0, 0, 0
    ]
    expected_aligment = [
        0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 6,
        6
    ]
    num_classes = 15
    sequence_len = len(label_sequence) + 3

    logits = np.ones((1, sequence_len, num_classes)) * -1e3
    logitpaddings = np.ones((1, sequence_len), dtype=float)
    for i in range(len(label_sequence)):
      logits[0, i, label_sequence[i]] = 0
      logitpaddings[0, i] = 0.0
    labels = np.array([[5, 7, 8, 9, 14, 12, 0]],
                      dtype=np.intp)
    labelpaddings = np.array([[0, 0, 0, 0, 0, 0, 1]],
                             dtype=float)

    # Double everything to test with batch > 1.
    logits = np.tile(logits, [2, 1, 1])
    logitpaddings = np.tile(logitpaddings, [2, 1])
    labels = np.tile(labels, [2, 1])
    labelpaddings = np.tile(labelpaddings, [2, 1])

    per_seq_loss, aux = ctc_objectives.ctc_loss_with_alignments(
        logits, logitpaddings, labels, labelpaddings)
    self.assertAllClose([expected_aligment, expected_aligment], aux.alignment)
    self.assertAllClose(per_seq_loss, [0.0, 0.0])

  def test_ctc_loss_alignment_validity(self):
    # Test that the alignments generated contain all of the labels in the
    # proper order.
    num_classes = 7
    sequence_len = 14

    for i in range(10):
      logits = np.random.random_sample((1, sequence_len, num_classes))
      logitpaddings = np.zeros((1, sequence_len), dtype=float)
      labels = np.array([[2, 5, 3, 1, 4, 0]], dtype=np.intp)
      labelpaddings = np.array([[0, 0, 0, 0, 0, 1]], dtype=float)
      _, aux = ctc_objectives.ctc_loss_with_alignments(logits, logitpaddings,
                                                       labels, labelpaddings)

      last_output = 0
      for output in aux.alignment[i].tolist():
        # Alignments are monotonically increasing.
        self.assertLessEqual(last_output, output)
        # You can't skip tokens in the alignment.
        self.assertLessEqual(output, last_output + 1)
        last_output = output

if __name__ == '__main__':
  absltest.main()
