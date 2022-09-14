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

  @parameterized.parameters((2, 3, True), (2, 3, False), (4, 3, True),
                            (4, 4, True))
  def test_depthwise_conv1D_streaming(self, step, kernel_size, with_paddings):

    # Input data.
    inputs = np.random.normal(size=[1, 8, 4]).astype(np.float32)
    paddings = None
    if with_paddings:
      paddings = np.array([[1, 1, 1, 0, 0, 0, 1, 1]]).astype(np.float32)

    context_p = base_layer.JaxContext.HParams(do_eval=True)
    with base_layer.JaxContext.new_context(hparams=context_p):
      # Original layer.
      base_layer_p = convolutions.DepthwiseConv1D.HParams(
          name='dw_1D',
          # (kernel_size, in_channels, channel_multipliers)
          filter_shape=(kernel_size, 4, 1),
          is_causal=True,
          rhs_dilation_rate=2
      )
      layer = instantiate(base_layer_p)
      prng_key = jax.random.PRNGKey(seed=123)
      prng_key, init_key = jax.random.split(prng_key)
      initial_vars = layer.init(init_key, inputs)
      base_output_non_stream = layer.apply(initial_vars, inputs, paddings)

      # Streaming aware layer
      p = streaming.DepthwiseConv1D.HParams(name='dw_1D_stream')
      p.copy_fields_from(base_layer_p)
      layer = instantiate(p)
      output_non_stream = layer.apply(initial_vars, inputs, paddings)

    output_names = ['features', 'paddings']
    in_nmap = py_utils.NestedMap(features=inputs, paddings=paddings)
    output_stream = operations.run_streaming(layer, initial_vars, in_nmap,
                                             output_names, step)

    # Striding is always 1 for now:
    self.assertEqual(p.cls.get_stride(p), 1)

    # Causal model has no delay:
    self.assertEqual(p.cls.get_right_context(p), 0)

    self.assertAllClose(output_non_stream, output_stream.features)
    self.assertAllClose(base_output_non_stream, output_stream.features)

    if with_paddings:
      # Streaming paddings is the same with the input paddings in causal model:
      self.assertAllClose(paddings, output_stream.paddings)


if __name__ == '__main__':
  absltest.main()
