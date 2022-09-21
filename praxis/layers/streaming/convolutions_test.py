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

"""Tests for Praxis streaming convolutional layers."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from praxis import base_layer
from praxis.layers import convolutions
from praxis.layers import streaming
from praxis.layers.streaming import test_utils

instantiate = base_layer.instantiate


class StreamingConvolutionsTest(test_utils.StreamingTest):

  def setUp(self):
    super().setUp()

    # Input data.
    self.inputs = np.random.normal(size=[1, 8, 4]).astype(np.float32)
    self.paddings = np.array([[1, 1, 1, 0, 0, 0, 1, 1]]).astype(np.float32)

  @parameterized.parameters((2, 3, True), (2, 3, False), (4, 3, True),
                            (4, 4, True))
  def test_depthwise_conv1D_streaming(self, step, kernel_size, with_paddings):
    # Original layer.
    p_non_stream = convolutions.DepthwiseConv1D.HParams(
        name='dw_1D',
        # (kernel_size, in_channels, channel_multipliers)
        filter_shape=(kernel_size, 4, 1),
        is_causal=True,
        rhs_dilation_rate=2
    )

    # Streaming aware layer
    p_stream = streaming.DepthwiseConv1D.HParams(name='dw_1D_stream')
    p_stream.copy_fields_from(p_non_stream)

    # Striding is always 1 for now:
    self.assertEqual(p_stream.cls.get_stride(p_stream), 1)
    # Causal model has no delay:
    self.assertEqual(p_stream.cls.get_right_context(p_stream), 0)
    self._compare_stream_non_stream(
        self.inputs,
        self.paddings if with_paddings else None,
        p_non_stream,
        p_stream,
        step,
        expand_padding_rank=1)

  @parameterized.parameters((2, True), (2, False), (4, True))
  def test_light_conv1D_streaming(self, step, with_paddings):
    # Streaming aware layer. It can run in both streaming and non streaming.
    p_stream = streaming.LightConv1D.HParams(
        name='conv1D',
        input_dims=self.inputs.shape[-1],
        kernel_size=3,
        is_causal=True)

    # Striding is always 1.
    self.assertEqual(p_stream.cls.get_stride(p_stream), 1)
    # Causal model has no delay:
    self.assertEqual(p_stream.cls.get_right_context(p_stream), 0)
    self._compare_stream_non_stream(
        self.inputs,
        self.paddings if with_paddings else None,
        p_stream,
        p_stream,
        step,
        expand_padding_rank=1)


if __name__ == '__main__':
  absltest.main()
