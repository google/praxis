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

"""Unit tests for decoder_utils."""

from absl.testing import absltest
from absl.testing import parameterized
from jax import numpy as jnp
import numpy as np
from praxis import decoder_utils
from praxis import test_utils


class DecoderUtilsTest(test_utils.TestCase):

  @parameterized.parameters(
      ([4], [[-1.0e+09, 4, 3, 2, 25, 4, 3, 2, 0, 0, 0, 0, -1.0e+09, 4, 3, 2]
            ], [[25, 4, 4, 4]], [[4, 1, 5, 13]]),
      ([1, 4], [[
          -1.0e+09, 4, -1.0e+09, 2, 25, -1.0e+09, 3, 2, 0, -1.0e+09, 0, 0,
          -1.0e+09, 4, 3, -1.0e+09
      ]], [[25, 4, 4, 3]], [[4, 1, 13, 6]]),
  )
  def test_two_stage_topk(self, terminal_ids, topk_value_target,
                          target_final_topk_value, target_final_topk_indices):
    hyp_scores = np.zeros((1, 4))
    logits = [
        [2, 3, 4, 1, 15],  # Top-4 id: 4, 0, 1, 2
        [25, 4, 3, 2, 1],  # Top-4 id: 0, 1, 2, 3
        [0, 0, 0, 0, 0],  # Top-4 id: 0, 1, 2, 3
        [1, 2, 3, 4, 5],  # Top-4 id: 4, 3, 2, 1
    ]
    topk_value, topk_indices, final_topk_value, final_topk_indices = (
        decoder_utils.two_stage_topk(
            np.array(logits, dtype=np.float32), hyp_scores, terminal_ids))

    # Compares 1st topk
    self.assertArraysEqual(topk_value,
                           np.array(topk_value_target, dtype=np.float32))
    self.assertArraysEqual(
        topk_indices,
        np.array([[4, 2, 1, 0, 0, 1, 2, 3, 0, 1, 2, 3, 4, 3, 2, 1]],
                 dtype=np.int32))

    # Compares 2nd topk
    self.assertArraysEqual(final_topk_value,
                           np.array(target_final_topk_value, dtype=np.float32))
    self.assertArraysEqual(final_topk_indices,
                           np.array(target_final_topk_indices, dtype=np.int32))

  def test_gather_output_id(self):
    long_output_ids = jnp.array(
        [[4, 2, 1, 0, 0, 1, 2, 3, 0, 1, 2, 3, 4, 3, 2, 1]], dtype=np.int32)
    topk_indices = jnp.array([[4, 1, 5, 13]])
    output_ids = decoder_utils.gather_output_id(long_output_ids, topk_indices)
    self.assertArraysEqual(output_ids, np.array([[0, 2, 1, 3]], dtype=np.int32))

  def test_gather_logprobs(self):
    logprobs = jnp.array(
        [[[1, 2, 4, 0], [2, 1, 3, 0], [3, 2, 3, 4]],
         [[4, 5, 2, 1], [1, 3, 4, 4], [4, 3, 3, 2]],],
        dtype=np.float32)
    hyp_ids = jnp.array([[1, 0, 1], [2, 1, 0]], dtype=np.int32)
    ids = jnp.array([[2, 3, 1], [1, 1, 2]], dtype=np.int32)
    output_logprobs = decoder_utils.gather_logprobs(logprobs, hyp_ids, ids)
    self.assertArraysEqual(output_logprobs, np.array(
        [[3, 0, 1], [3, 3, 2]], dtype=np.float32))

  def test_right_align_tensors(self):
    long_output_ids = jnp.array(
        [[5, 3, 2, 5, 0, 0], [0, 6, 7, 0, 0, 0], [1, 3, 9, 5, 6, 2]],
        dtype=np.int32)
    seq_lengths = jnp.array([4, 3, 6])
    output_ids = decoder_utils.right_align_tensors(long_output_ids, seq_lengths)
    self.assertArraysEqual(
        output_ids,
        np.array([[0, 0, 5, 3, 2, 5], [0, 0, 0, 0, 6, 7], [1, 3, 9, 5, 6, 2]],
                 dtype=np.int32))

  def test_right_align_states(self):
    decode_cache = jnp.array(
        [[[[0.1, 0.1], [0.2, 0.3]], [[0.2, 0.3], [0.4, 0.5]],
          [[0, 0.1], [0.1, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]]],
        dtype=np.float32)
    seq_lengths = jnp.array([3])
    right_align_decode_cache = decoder_utils.right_align_state_fn(seq_lengths)(
        decode_cache, batch_dim=0, time_dim=1)
    self.assertArraysEqual(
        right_align_decode_cache,
        np.array([[[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0.1, 0.1], [0.2, 0.3]],
                   [[0.2, 0.3], [0.4, 0.5]], [[0, 0.1], [0.1, 0]]]],
                 dtype=np.float32))

  def test_left_align_tensor(self):
    output_ids = jnp.array([[0, 1, 2, 3], [0, 1, 0, 0], [1, 2, 3, 4]],
                           dtype=jnp.int32)
    prefix_lengths = jnp.array([1, 1, 2], dtype=jnp.int32)
    max_prefix_len = 2

    left_align_prefix_ids = decoder_utils.left_align_tensor(
        output_ids, prefix_lengths, max_prefix_len)

    self.assertArraysEqual(
        left_align_prefix_ids,
        jnp.array([[1, 2, 3, 0], [1, 0, 0, 0], [1, 2, 3, 4]], dtype=jnp.int32))

  def test_concat_suffix_and_left_align_tensor(self):
    num_samples = 3
    num_suffix = 2
    output_ids = jnp.array([[0, 1, 2, 3], [0, 1, 3, 0], [1, 2, 3, 4]],
                           dtype=jnp.int32)
    prefix_lengths = jnp.array([1, 1, 2], dtype=jnp.int32)
    decode_end_indices = jnp.array([4, 3, 4], dtype=jnp.int32)
    max_prefix_len = 2

    suffix_ids = jnp.array([[5, 2], [7, 3], [5, 2], [7, 3], [5, 2], [7, 3]],
                           dtype=jnp.int32)

    left_align_ids = decoder_utils.concat_suffix_and_left_align(
        output_ids,
        suffix_ids,
        decode_end_indices,
        prefix_lengths,
        max_prefix_len,
        num_samples,
        num_suffix,
        pad_value=0)

    self.assertArraysEqual(
        left_align_ids,
        jnp.array(
            [[[[1, 2, 3, 5, 2, 0], [1, 3, 5, 2, 0, 0], [1, 2, 3, 4, 5, 2]],
              [[1, 2, 3, 7, 3, 0], [1, 3, 7, 3, 0, 0], [1, 2, 3, 4, 7, 3]]]],
            dtype=jnp.int32))


if __name__ == '__main__':
  absltest.main()
