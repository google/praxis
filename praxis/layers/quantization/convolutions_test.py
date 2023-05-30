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

"""Tests for Quantized Praxis convolutional layers."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import pax_fiddle
from praxis import test_utils
from praxis.layers import convolutions
from praxis.layers.quantization import convolutions as qconvolutions
from praxis.layers.quantization import operations
from praxis.layers.quantization import quantization_hparams

instantiate = base_layer.instantiate
QuantizationParams = quantization_hparams.QuantizationParams
QuantizationMode = quantization_hparams.QuantizationMode

PARAMS = base_layer.PARAMS
NON_TRAINABLE = base_layer.NON_TRAINABLE


class ConvolutionsTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  @parameterized.parameters(
      ((5, 4, 24, 36), (1, 1), (1, 1), False, [2, 16, 36, 72]),
      ((2, 4, 16, 8), (2, 2), (1, 1), False, [2, 16, 32, 128]),
      ((4, 8, 16, 32), (1, 1), (1, 1), False, [2, 16, 32, 64]),
      ((2, 8, 16, 32), (1, 1), (2, 2), False, [2, 16, 32, 64]),
      ((2, 8, 16, 32), (2, 2), (2, 2), False, [2, 16, 32, 64]),
      ((2, 8, 16, 32), (1, 1), (2, 1), False, [2, 16, 32, 64]),
      ((2, 8, 16, 32), (2, 2), (2, 1), False, [2, 16, 32, 64]),
      ((2, 8, 16, 32), (1, 1), (2, 2), True, [2, 16, 32, 64]),
      ((2, 8, 16, 32), (2, 2), (2, 2), True, [2, 16, 32, 64]),
  )
  def test_conv2d_layer_same_padding(
      self,
      filter_shape,
      filter_stride,
      dilations,
      tf_equivalent_padding,
      input_shape,
  ):
    npy_inputs = np.random.normal(0.0, 0.5, input_shape).astype('float32')
    inputs = jnp.asarray(npy_inputs)
    prng_key = jax.random.PRNGKey(seed=123)

    # Float layer.
    p = pax_fiddle.Config(
        convolutions.Conv2D,
        name='jax_conv2d',
        filter_shape=filter_shape,
        filter_stride=filter_stride,
        dilations=dilations,
        tf_equivalent_padding=tf_equivalent_padding,
        padding='SAME',
    )
    conv_layer = instantiate(p)
    initial_vars = conv_layer.init(prng_key, inputs)
    outputs = conv_layer.apply(initial_vars, inputs)

    # Quantized layer.
    qp = pax_fiddle.Config(
        qconvolutions.Conv2D,
        name='jax_conv2_q',
        filter_shape=filter_shape,
        filter_stride=filter_stride,
        dilations=dilations,
        tf_equivalent_padding=tf_equivalent_padding,
        padding='SAME',
        quantization=QuantizationParams(mode=QuantizationMode.INFERENCE)
    )
    qconv_layer = instantiate(qp)
    qinitial_vars = qconv_layer.init(prng_key, inputs)
    # TODO(jianlijianli): Use quantize_weight() once it's implemented.
    qweight, qscale, _ = operations.reduce_precision(
        initial_vars['params']['w'], [0, 1, 2]
    )
    qinitial_vars['params']['w'] = qweight
    qinitial_vars['params']['w_quantized_scale'] = jnp.squeeze(qscale)
    qoutput = qconv_layer.apply(qinitial_vars, inputs)

    self.assertAllClose(qoutput, outputs, rtol=1e-02, atol=1e-02)


if __name__ == '__main__':
  absltest.main()
