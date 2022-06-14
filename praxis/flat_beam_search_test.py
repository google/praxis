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

"""Unit tests for flat_beam_search."""

from absl.testing import absltest
from jax import numpy as jnp
import numpy as np
from praxis import flat_beam_search
from praxis import test_utils


class FlatBeamSearchHelperTest(test_utils.TestCase):

  def test_update_mask_without_time_step(self):
    beam_mask = jnp.array(
        [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]],
        dtype=jnp.float32)
    hyp_id = jnp.array([[0, 3, 0, 1]], jnp.float32)
    update_beam_mask = flat_beam_search.update_beam_mask(
        beam_mask, hyp_id, time_step=None)
    self.assertArraysEqual(
        update_beam_mask,
        np.array([[[1, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]],
                 dtype=np.float32))

  def test_update_mask_without_step2(self):
    beam_mask = np.array([[[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]]],
                         dtype=jnp.float32)
    hyp_id = np.array([[3, 3, 0, 1]], jnp.float32)
    update_beam_mask = flat_beam_search.update_beam_mask(
        beam_mask, hyp_id, time_step=None)
    self.assertArraysEqual(
        update_beam_mask,
        np.array([[[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                   [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]]],
                 dtype=np.float32))

  def test_update_mask_with_step(self):
    beam_mask = np.array([[[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]]],
                         dtype=jnp.float32)
    hyp_id = np.array([[3, 3, 0, 1]], jnp.float32)
    update_beam_mask = flat_beam_search.update_beam_mask(
        beam_mask, hyp_id, time_step=2)
    self.assertArraysEqual(
        update_beam_mask,
        np.array([[[0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                   [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1]]],
                 dtype=np.float32))

  def test_get_final_output_ids(self):
    beam_mask = jnp.array(
        [[[0, 0, 1, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0, 1, 0],
          [1, 0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 1, 0, 0]]],
        dtype=jnp.int32)
    output_ids = jnp.array([[[0, 5], [1, 6], [2, 7], [3, 8]]], dtype=np.int32)
    final_output_ids = flat_beam_search.get_final_output_ids(
        beam_mask, output_ids)

    self.assertArraysEqual(
        final_output_ids,
        np.array([[[2, 8], [2, 7], [0, 5], [2, 6]]], dtype=np.int32))

  def test_update_topk_scores_with_eos(self):
    end_mask = jnp.array([[[0, 1], [0, 1]]], dtype=np.float32)
    cur_mask = jnp.array([[[1, 0], [0, 1]]], dtype=np.float32)

    end_scores = jnp.array([[0, 1]], dtype=np.float32)
    cur_scores = jnp.array([[2, 3]], dtype=np.float32)

    end_scores_norm = jnp.array([[2, 0]], dtype=np.float32)
    cur_scores_norm = jnp.array([[3, 1]], dtype=np.float32)

    (output_mask, output_scores,
     output_scores_norm) = flat_beam_search.update_topk_scores_with_eos(
         (end_mask, end_scores, end_scores_norm),
         (cur_mask, cur_scores, cur_scores_norm))

    self.assertArraysEqual(output_mask,
                           np.array([[[1, 0], [0, 1]]], dtype=np.float32))
    self.assertArraysEqual(output_scores, np.array([[2, 0]], dtype=np.float32))
    self.assertArraysEqual(output_scores_norm,
                           np.array([[3, 2]], dtype=np.float32))


if __name__ == '__main__':
  absltest.main()
