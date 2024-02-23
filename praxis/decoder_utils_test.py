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

"""Unit tests for decoder_utils."""

from absl.testing import absltest
from absl.testing import parameterized
from jax import numpy as jnp
import numpy as np
from praxis import decoder_utils
from praxis import pytypes
from praxis import test_utils


class DecoderUtilsTest(test_utils.TestCase):

  # pyformat:disable
  @parameterized.parameters(
      ([4],
       [
           [-1.0e+09, 4, 3, 2, 25, 4, 3, 2, 0, 0, 0, 0, -1.0e+09, 4, 3, 2]
       ],
       [
           [25, 4, 4, 4]
       ],
       [
           [4, 1, 5, 13]
       ]),
      ([1, 4],
       [
           [
               -1.0e+09, 4, -1.0e+09, 2, 25, -1.0e+09, 3, 2, 0, -1.0e+09, 0, 0,
               -1.0e+09, 4, 3, -1.0e+09
           ]
       ],
       [
           [25, 4, 4, 3]
       ],
       [
           [4, 1, 13, 6]
       ]),
  )
  # pyformat:enable
  def test_two_stage_topk(self, terminal_ids, target_topk_value,
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
                           np.array(target_topk_value, dtype=np.float32))
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

  def test_end_with_sequences(self):
    end_sequences = jnp.array(
        [[1, 5, 2], [0, 0, 3], [0, 5, 2]], dtype=jnp.int32
    )
    output_ids = jnp.array(
        [[0, 1, 3, 1, 5, 2], [1, 2, 3, 4, 5, 0], [7, 8, 9, 5, 2, 3]],
        dtype=jnp.int32,
    )

    # True for the 1st element in the batch.
    result = decoder_utils.end_with_sequences(
        end_sequences,
        output_ids,
        decode_step=5,
    )
    self.assertArraysEqual(result, jnp.array([1, 0, 0], dtype=jnp.bool_))

    # True for the 2nd element in the batch.
    result = decoder_utils.end_with_sequences(
        end_sequences, output_ids, decode_step=jnp.array(2, dtype=jnp.int32)
    )
    self.assertArraysEqual(result, jnp.array([0, 1, 0], dtype=jnp.bool_))

    # True for the 3rd element in the batch.
    result = decoder_utils.end_with_sequences(
        end_sequences, output_ids, decode_step=4
    )
    self.assertArraysEqual(result, jnp.array([0, 0, 1], dtype=jnp.bool_))

  def test_has_any_eos(self):
    test_arr = jnp.array([[0, 1, 2], [10, 11, 12], [20, 21, 22]])
    has_eos = decoder_utils.has_any_eos(test_arr, 1)
    self.assertArraysEqual(
        has_eos, jnp.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=jnp.bool_)
    )
    has_eos = decoder_utils.has_any_eos(test_arr, [1])
    self.assertArraysEqual(
        has_eos, jnp.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=jnp.bool_)
    )
    has_eos = decoder_utils.has_any_eos(test_arr, [10, 21])
    self.assertArraysEqual(
        has_eos, jnp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=jnp.bool_)
    )

  def test_coerce_to_expanded_extend_step_fn_wrap(self):

    def _extend_step_fn(model, ids, pos):
      return ids.astype(jnp.float32) / 2

    expanded_extend_step_fn = decoder_utils.coerce_to_expanded_extend_step_fn(
        _extend_step_fn
    )

    self.assertArraysEqual(
        expanded_extend_step_fn(
            None, jnp.array([[1, 2]]), jnp.array([3, 4]), pytypes.NestedMap()
        ),
        jnp.array([[0.5, 1.0]]),
    )

  def test_coerce_to_expanded_extend_step_fn_cast(self):

    def _extend_step_fn(model, ids, pos, state):
      return ids.astype(jnp.float32) / 2

    expanded_extend_step_fn = decoder_utils.coerce_to_expanded_extend_step_fn(
        _extend_step_fn
    )

    self.assertIs(expanded_extend_step_fn, _extend_step_fn)  # merely casted
    self.assertArraysEqual(
        expanded_extend_step_fn(
            None, jnp.array([[1, 2]]), jnp.array([3, 4]), pytypes.NestedMap()
        ),
        jnp.array([[0.5, 1.0]]),
    )

  def test_collect_results_to_optimize_eos(self):
    result = pytypes.NestedMap(
        logprobs=jnp.array(
            [[-0.2, -0.3, -0.4, -0.1], [-0.2, 0.1, -0.3, -0.2]]),
        eos_logprobs=jnp.array(
            [[-10, -10, -0.1, -0.1], [-10, -10, -1, -0.1]]),
        output_ids=jnp.array([[4, 3, 3, 0], [3, 1, 4, 5]]),
        eos_ids=jnp.array([[1, 1, 1, 2], [2, 2, 1, 2]]),
        done=jnp.array([False, False]),
        has_eos=jnp.array([False, False]),
        start_step=0
    )
    new_result = decoder_utils.collect_results_to_optimize_eos(result)
    self.assertArraysEqual(
        new_result.logprobs,
        jnp.array([[-0.2, -0.3, -0.1, 1], [-0.2, 0.1, -0.3, -0.1]]))
    self.assertArraysEqual(
        new_result.output_ids,
        jnp.array([[4, 3, 1, 0], [3, 1, 4, 2]]))
    self.assertArraysEqual(
        new_result.done,
        jnp.array([True, True]))
    self.assertArraysEqual(
        new_result.has_eos,
        jnp.array([True, True]))

  def _tile_to_bsize(self, arr, bsize) -> jnp.ndarray:
    return jnp.tile(arr[jnp.newaxis, ...], (bsize, 1, 1))

  def test_end_with_any_sequence_any_position(self):
    sequences = np.asarray([[2, 3, 4, 5], [0, 2, 6, 8]])

    col_idxs = np.asarray([[2, 3], [2, 3]])
    # Repeat stop seqs.
    stop_seqs = self._tile_to_bsize(
        np.asarray([[3, 4, 5], [0, 2, 6]]), sequences.shape[0]
    )
    out = decoder_utils.end_with_any_sequence_any_position(
        stop_seqs, sequences, col_idxs
    )
    # TODO(pcyc) add example where sequences is shorter than stop seqs
    self.assertArraysEqual(
        out,
        np.array(
            [[[False, False], [True, False]], [[False, True], [False, False]]]
        ),
    )

  def test_find_first_stop_seq_match(self):

    # We only look for stop sequence matches at these indices or after!
    first_new_decode_idx = np.asarray([
        2,
        0,
        2,
        2,
    ])
    sequences = np.asarray([
        [2, 3, 4, 5],  # Will match first eos seq.
        [2, 3, 4, 5],  # Nearly match first eos seq, but index too small.
        [6, 7, 8, 9],  # No match.
        [4, 2, 6, 7],  # Match second eos sequence, which starts with padding.
    ])

    # 0 corresponds to padding. Stop sequences are left padded.
    stop_sequences = self._tile_to_bsize(
        np.asarray([
            [3, 4, 5],
            [0, 2, 6],
        ]),
        sequences.shape[0],
    )

    self.assertArraysEqual(
        decoder_utils.find_first_new_stop_seq_match(
            first_new_decode_idx=first_new_decode_idx,
            num_new_tokens=2,
            stop_sequences=stop_sequences,
            sequences=sequences,
        ),
        # First sequence matches first stop sequence at index 3, which is
        # 1 token past the first new token.
        # Second sequence doesn't match any stop sequence.
        # Third sequence doesn't match any stop sequence.
        # Fourth sequence matches second stop sequence at index 2, which is 0
        # tokens past the first new token.
        np.array([1, 2, 2, 0]),
    )

  def test_left_align_kv_cache(self):
    num_cache_slots = 3
    sql_len = 6
    num_kv_heads = 3
    head_dims = 2

    kv_state_shape = (
        num_cache_slots,
        sql_len,
        num_kv_heads,
        head_dims,
    )
    x = jnp.ones(kv_state_shape)
    prefix_lengths = jnp.array([1, 3, 2], dtype=jnp.int32)
    # max step is 4 which means the longest seq length is 5 at this moment.
    # Given the sql_len is 6, the slots[:,-1,:,:] should be padded with zero.
    max_step = 4
    left_align_steps_arr = jnp.ones_like(prefix_lengths) * max_step
    got = decoder_utils.left_align_kv_cache(
        x, left_align_steps_arr, sql_len - 1, pad_value=0, batch_size=3
    )
    self.assertArraysEqual(
        got[:, -1, :, :], jnp.zeros((num_cache_slots, num_kv_heads, head_dims))
    )
    self.assertArraysEqual(
        got[:, 0:-1, :, :],
        jnp.ones((num_cache_slots, sql_len - 1, num_kv_heads, head_dims)),
    )


if __name__ == '__main__':
  absltest.main()
