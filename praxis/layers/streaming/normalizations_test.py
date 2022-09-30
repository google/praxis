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

"""Tests for Praxis streaming normalization layers."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from praxis import base_layer
from praxis.layers import streaming
from praxis.layers import normalizations
from praxis.layers.streaming import test_utils

instantiate = base_layer.instantiate


class StreamingNormalizationTest(test_utils.TestCase):

  @parameterized.named_parameters(
      ('Basic',),
      ('Group4Dim2', 1, 4, 2),
      ('Group1', 1, 1),
      ('Stride2', 2),
      ('Stride2Group1', 2, 1),
      ('Stride4', 4),
  )
  def test_stream(self, stride=1, num_groups=2, dim=4):
    seqlen = 16
    batch_size, input_dim = 2, dim
    inputs = np.random.normal(
        0.1, 0.5, [batch_size, seqlen, 1, input_dim]).astype(np.float32)
    paddings = np.random.randint(0, 2, [batch_size, seqlen]).astype('float32')

    p_non_stream = normalizations.GroupNorm.HParams(
        name='non_stream_gn',
        dim=input_dim,
        num_groups=num_groups,
        cumulative=True,
        input_rank=4)

    # Streaming aware layer
    p_stream = streaming.GroupNorm.HParams(name='stream_gn')
    p_stream.copy_fields_from(p_non_stream)

    self.assertEqual(p_stream.cls.get_stride(p_stream), 1)
    self.assertEqual(p_stream.cls.get_right_context(p_stream), 0)
    self._compare_stream_non_stream(
        inputs, paddings, p_non_stream, p_stream, stride)


if __name__ == '__main__':
  absltest.main()
