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

"""Tests for Praxis quantized multi-query attention layers."""

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from praxis import base_layer
from praxis import pax_fiddle
from praxis import test_utils
from praxis.layers import multi_query_attention as mqa_f
from praxis.layers.quantization import multi_query_attention as mqa_q
from praxis.layers.quantization import quantization_hparams
from praxis.layers.quantization import utils

instantiate = base_layer.instantiate
QuantizationHParams = quantization_hparams.QuantizationHParams
QuantizationMode = quantization_hparams.QuantizationMode
QuantizationType = quantization_hparams.QuantizationType
WeightQuantizationParams = quantization_hparams.WeightQuantizationParams


class MultiQueryAttentionTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  @parameterized.named_parameters(
      dict(testcase_name='PTQ', quantization_type=QuantizationType.PTQ),
      dict(testcase_name='AQT', quantization_type=QuantizationType.AQT),
  )
  def test_one_headed_projection_shape(self, quantization_type):
    test_layer_p = pax_fiddle.Config(
        mqa_q.OneHeadedAttentionProjection,
        name='mh',
        input_dim=16,
        output_dim=5,
        quantization=QuantizationHParams(
            quantization_type=quantization_type,
            mode=QuantizationMode.TRAINING,
            weight_params=quantization_hparams.WeightQuantizationParams(),
        ),
    )
    layer = instantiate(test_layer_p)

    inputs = np.random.normal(1.5, 2.0, [5, 16]).astype(np.float32)

    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)
    initial_vars = layer.init(init_key, inputs)
    logging.info('initial_vars: %s', initial_vars)

    jax_out = layer.apply(initial_vars, inputs)
    self.assertSequenceEqual(jax_out.shape, [5, 5])

  @parameterized.product(
      quantization_type=[QuantizationType.PTQ, QuantizationType.AQT],
      use_symmetric=[True, False],
      precision=[8, 4],
  )
  def test_one_headed_projection_quantized(
      self, quantization_type, use_symmetric, precision
  ):
    input_dim = 16
    output_dim = 3
    batch = 5
    p_f = pax_fiddle.Config(
        mqa_f.OneHeadedAttentionProjection,
        name='ohp_f',
        input_dim=input_dim,
        output_dim=output_dim,
    )
    p_q = pax_fiddle.Config(
        mqa_q.OneHeadedAttentionProjection,
        name='ohp_q',
        input_dim=input_dim,
        output_dim=output_dim,
        quantization=QuantizationHParams(
            quantization_type=QuantizationType.PTQ,
            mode=QuantizationMode.INFERENCE,
            weight_params=WeightQuantizationParams(
                use_symmetric=use_symmetric, precision=precision
            ),
        ),
    )

    inputs = np.random.normal(1.5, 2.0, [batch, input_dim]).astype(np.float32)
    quantized_weight_range = (-8, 7) if precision == 4 else (-128, 127)
    quantized_weight = np.random.randint(
        *quantized_weight_range, (input_dim, output_dim), dtype=np.int8
    )
    w_scale = np.array([0.5, 2.0, 3.3], dtype=np.float32)
    if use_symmetric:
      weight_rescaled = quantized_weight * w_scale
    else:
      w_zp = np.array([-10.0, 10.0, -2.5], dtype=np.float32)
      weight_rescaled = quantized_weight * w_scale - w_zp

    ohp_f = instantiate(p_f)
    ohp_q = instantiate(p_q)

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars_f = ohp_f.init(prng_key, inputs)
      initial_vars_q = ohp_q.init(prng_key, inputs)
      initial_vars_f['params']['w'] = weight_rescaled
      initial_vars_q['params']['w'] = (
          utils.pack_4bit(quantized_weight, pack_dim=0)
          if precision == 4
          else quantized_weight
      )
      initial_vars_q['params']['w_quantized_scale'] = w_scale
      if not use_symmetric:
        initial_vars_q['params']['w_quantized_zp'] = w_zp
      outputs_f = ohp_f.apply(initial_vars_f, inputs)
      outputs_q = ohp_q.apply(initial_vars_q, inputs)
      self.assertAllClose(outputs_f, outputs_q)


if __name__ == '__main__':
  absltest.main()
