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

"""Unit tests for sample_decode."""

from absl.testing import absltest
import jax
from jax import numpy as jnp
import numpy as np
from praxis import py_utils
from praxis import sample_decode
from praxis import test_utils

NestedMap = py_utils.NestedMap


class SampleDecodeHelperTest(test_utils.TestCase):

  def test_split_batch_dim(self):
    x = jnp.array([[1, 2], [1, 2], [3, 4], [3, 4]], dtype=np.int32)
    self.assertArraysEqual(
        sample_decode.split_batch_dim(x, batch_dim=0, num_samples=2),
        np.array([[[1, 2], [1, 2]], [[3, 4], [3, 4]]], dtype=np.int32))

  def test_sample_from_topk_with_gumbel_noise(self):
    logits = jnp.array(
        [[0, 0, 1, 1, 0], [1, 0, 0, 0, 1], [0, 1, 1, 0, 0], [1, 1, 0, 0, 0]],
        dtype=jnp.float32)
    noise = jnp.array([[0.5, 0], [-0.5, 0], [-0.5, -1], [1, 0.5]],
                      dtype=jnp.float32)
    # logits + noise =
    # [[0, 0, 1.5, 1, 0], # argmax: 2
    #  [0.5, 0, 0, 0, 1]  # argmax: 4
    #  [0, 0.5, 0, 0, 0], # argmax: 1
    #  [2, 1.5, 0, 0, 0]] # argmax: 0
    new_ids = sample_decode.sample_from_topk_with_gumbel_noise(
        logits, noise, temperature=1.0, topk=2)
    self.assertArraysEqual(new_ids, np.array([2, 4, 1, 0], dtype=np.int32))

  def test_sample_from_topk(self):
    logits = jnp.array(
        [
            [0, 0, 1, 0, 0],  # argmax: 2
            [1, 0, 0, 0, 0],  # argmax: 0
            [0, 1, 0, 0, 0],  # argmax: 1
            [1, 0, 0, 0, 0],  # argmax: 0
        ],
        dtype=jnp.float32)
    new_ids = sample_decode.sample_from_topk(
        logits,
        jax.random.PRNGKey(seed=123),
        temperature=1.0,
        topk=2)
    # gumbel noise is relatively smaller compared to the logits value.
    self.assertArraysEqual(new_ids, np.array([2, 0, 1, 0], dtype=np.int32))

  def test_sample_from_topk_distribution(self):
    logits = jnp.array([
        [0, 0.25, 0.2, 0.15, 0.4],
    ], dtype=jnp.float32)
    count = [0] * 5

    for i in range(100):
      new_ids = sample_decode.sample_from_topk(
          logits,
          jax.random.PRNGKey(seed=i),
          temperature=1.0,
          topk=4)
      count[new_ids[0]] += 1

    # Top4 value won't choose token 0.
    self.assertEqual(count[0], 0)
    # Token #1, #2, #3 should be chosen more than 10%.
    self.assertGreaterEqual(count[1], 10)
    self.assertGreaterEqual(count[2], 10)
    self.assertGreaterEqual(count[3], 10)
    # Token #4 should be chosen more than 25%.
    self.assertGreaterEqual(count[4], 25)

  def test_reorder_with_indices(self):
    indices = jnp.array([[0, 2, 1], [2, 0, 1]], dtype=jnp.int32)
    x = jnp.array(
        [[[0, 0, 0], [1, 1, 1], [2, 2, 2]], [[0, 0, 0], [1, 1, 1], [2, 2, 2]]],
        dtype=jnp.int32)

    reordered_x = sample_decode.reorder_with_indices(x, indices)

    self.assertArraysEqual(
        reordered_x,
        jnp.array([[[0, 0, 0], [2, 2, 2], [1, 1, 1]],
                   [[2, 2, 2], [0, 0, 0], [1, 1, 1]]],
                  dtype=jnp.int32))

  def test_sort_samples_by_scores(self):
    logprobs = jnp.array(
        [
            [
                [0.1, 0.1, 1.0],  #  sum is 0.2
                [0.3, 0.3, 0.3],  #  sum is 0.9
                [0.2, 0.2, 1.0]  #  sum is 0.4
            ],
            [
                [0.9, 0.9, 0.9],  # sum is 2.7
                [0.1, 1.0, 1.0],  # sum is 0.1
                [0.2, 0.2, 1.0]  # sum is 0.4
            ]
        ],
        dtype=jnp.float32)
    x = jnp.array(
        [[[0, 0, 0], [1, 1, 1], [2, 2, 2]], [[0, 0, 0], [1, 1, 1], [2, 2, 2]]],
        dtype=jnp.int32)
    result = NestedMap()
    result.logprobs = logprobs
    result.x = x

    result = sample_decode.sort_samples_by_scores(result)

    # Verify values in result are sorted and ordered at dimension 1.
    self.assertArraysEqual(
        result.x,
        jnp.array([[[1, 1, 1], [2, 2, 2], [0, 0, 0]],
                   [[0, 0, 0], [2, 2, 2], [1, 1, 1]]],
                  dtype=jnp.int32))
    self.assertArraysEqual(
        result.logprobs,
        jnp.array(
            [
                [
                    [0.3, 0.3, 0.3],  #  sum is 0.9
                    [0.2, 0.2, 1.0],  #  sum is 0.4
                    [0.1, 0.1, 1.0],  #  sum is 0.2
                ],
                [
                    [0.9, 0.9, 0.9],  # sum is 2.7
                    [0.2, 0.2, 1.0],  # sum is 0.4
                    [0.1, 1.0, 1.0],  # sum is 0.1
                ]
            ],
            dtype=jnp.float32))

  def test_right_align_prefix_ids(self):
    prefix_ids = jnp.array([[1, 2, 0], [1, 0, 0], [0, 1, 2]], dtype=jnp.int32)
    prefix_lengths = jnp.array([2, 1, 3], dtype=jnp.int32)

    (right_align_prefix_ids,
     right_align_prefix_paddings) = sample_decode.right_align_prefix_ids(
         prefix_ids, prefix_lengths, jnp.int32)

    self.assertArraysEqual(
        right_align_prefix_ids,
        jnp.array([[0, 1, 2], [0, 0, 1], [0, 1, 2]], dtype=jnp.int32))

    self.assertArraysEqual(
        right_align_prefix_paddings,
        jnp.array([[1, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=jnp.int32))

  def test_right_align_segment_position(self):
    lengths = jnp.array([5, 4, 6], dtype=jnp.int32)

    right_align_segment_pos = sample_decode.right_align_segment_position(
        lengths, max_length=6)

    self.assertArraysEqual(
        right_align_segment_pos,
        jnp.array([[0, 0, 1, 2, 3, 4], [0, 0, 0, 1, 2, 3], [0, 1, 2, 3, 4, 5]],
                  dtype=jnp.int32))


if __name__ == '__main__':
  absltest.main()
