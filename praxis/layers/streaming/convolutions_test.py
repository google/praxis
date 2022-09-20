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
import jax
import numpy as np
from praxis import base_layer
from praxis import py_utils
from praxis import test_utils
from praxis.layers import convolutions
from praxis.layers import streaming
from praxis.layers.streaming import operations

instantiate = base_layer.instantiate


class StreamingConvolutionsTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

    # Input data.
    self.inputs = np.random.normal(size=[1, 8, 4]).astype(np.float32)
    self.paddings = np.array([[1, 1, 1, 0, 0, 0, 1, 1]]).astype(np.float32)

  def _compare_stream_non_stream(self,
                                 p_non_stream,
                                 p_stream,
                                 with_paddings,
                                 step,
                                 do_eval=True):
    paddings = self.paddings if with_paddings else None
    context_p = base_layer.JaxContext.HParams(do_eval=do_eval)
    with base_layer.JaxContext.new_context(hparams=context_p):
      layer_non_stream = instantiate(p_non_stream)
      prng_key = jax.random.PRNGKey(seed=123)
      prng_key, init_key = jax.random.split(prng_key)
      initial_vars = layer_non_stream.init(init_key, self.inputs, paddings)
      base_output_non_stream = layer_non_stream.apply(initial_vars, self.inputs,
                                                      paddings)

      # Run streaming aware layer in non streaming mode
      layer_stream = instantiate(p_stream)
      output_non_stream = layer_stream.apply(initial_vars, self.inputs,
                                             paddings)

    # Run streaming aware layer in streaming mode
    output_names = ['features', 'paddings']
    in_nmap = py_utils.NestedMap(features=self.inputs, paddings=paddings)
    output_stream = operations.run_streaming(layer_stream, initial_vars,
                                             in_nmap, output_names, step)

    self.assertAllClose(output_non_stream, output_stream.features)
    self.assertAllClose(base_output_non_stream, output_stream.features)

    if with_paddings:
      # Streaming paddings is the same with the input paddings in causal model:
      self.assertAllClose(paddings, output_stream.paddings)

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
    self._compare_stream_non_stream(p_non_stream, p_stream, with_paddings, step)

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
    self._compare_stream_non_stream(p_stream, p_stream, with_paddings, step)


if __name__ == '__main__':
  absltest.main()
