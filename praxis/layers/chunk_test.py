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

"""Tests for Praxis Chunk functions."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from praxis import test_utils
from praxis.layers import chunk


class ChunkTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  @parameterized.parameters(
      ([3, 4, 5], 0, 2, [2, 2, 4, 5]),
      ([3, 5], 1, 2, [3, 3, 2]),
      ([3, 4, 2, 8], 1, 3, [2, 3, 3, 2, 8]),
      ([3, 4, 2, 8], 3, 4, [2, 3, 4, 2, 4]),
  )
  def test_chunk(self, in_shape, axis, chunk_size, chunk_shape):
    x = np.random.normal(1.0, 0.5, in_shape)
    chunk_x = chunk.chunk(x, chunk_size=chunk_size, axis=axis)
    self.assertArraysEqual(chunk_x.shape, chunk_shape)

    out_x = chunk.unchunk(chunk_x, axis=axis, seqlen=x.shape[axis])
    self.assertAllClose(x, out_x)


if __name__ == '__main__':
  absltest.main()
