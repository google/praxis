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

"""Tests for Praxis quantized multi-query attention layers."""

from absl import logging
from absl.testing import absltest
import jax
import numpy as np
from praxis import base_layer
from praxis import pax_fiddle
from praxis import test_utils
from praxis.layers import multi_query_attention as mqa_f
from praxis.layers.quantization import multi_query_attention as mqa_q
from praxis.layers.quantization import quantization_hparams

instantiate = base_layer.instantiate
QuantizationHParams = quantization_hparams.QuantizationHParams
QuantizationMode = quantization_hparams.QuantizationMode
QuantizationType = quantization_hparams.QuantizationType


class MultiQueryAttentionTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  def test_one_headed_projection_shape(self):
    test_layer_p = pax_fiddle.Config(
        mqa_q.OneHeadedAttentionProjection,
        name='mh',
        input_dim=16,
        output_dim=5,
    )
    layer = instantiate(test_layer_p)

    inputs = np.random.normal(1.5, 2.0, [5, 16]).astype(np.float32)

    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)
    initial_vars = layer.init(init_key, inputs)
    logging.info('initial_vars: %s', initial_vars)

    jax_out = layer.apply(initial_vars, inputs)
    self.assertSequenceEqual(jax_out.shape, [5, 5])


def test_one_headed_projection_quantized(self):
  p_f = pax_fiddle.Config(mqa_f.OneHeadedAttentionProjection, name='ohp_f')
  p_q = pax_fiddle.Config(
      mqa_q.OneHeadedAttentionProjection,
      name='ohp_q',
      quantization=QuantizationHParams(
          quantization_type=QuantizationType.PTQ,
          mode=QuantizationMode.INFERENCE,
      ),
  )

  inputs = np.random.normal(1.5, 2.0, [5, 16]).astype(np.float32)
  quantized_weight = np.random.randint(
      jax.random.PRNGKey(0), (4, 2), minval=-128, maxval=127, dtype=np.int8
  )
  w_scale = np.array([0.5, 2.0], dtype=np.float32)
  weight_rescaled = quantized_weight * w_scale

  ohp_f = instantiate(p_f)
  ohp_q = instantiate(p_q)

  prng_key = jax.random.PRNGKey(seed=123)
  initial_vars_f = ohp_f.init(prng_key, inputs)
  initial_vars_q = ohp_q.init(prng_key, inputs)
  initial_vars_f['params']['w'] = weight_rescaled
  initial_vars_q['params']['w'] = quantized_weight
  initial_vars_q['params']['w_quantized_scale'] = w_scale
  outputs_f = ohp_f.apply(initial_vars_f, inputs)
  outputs_q = ohp_q.apply(initial_vars_q, inputs)
  self.assertAllClose(outputs_f, outputs_q)


if __name__ == '__main__':
  absltest.main()
