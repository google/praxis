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

"""Tests for TensorQuantizer in Quantization-aware Training."""

from typing import Tuple

from absl.testing import absltest
import jax
from jax import numpy as jnp
from praxis import pax_fiddle
from praxis import pytypes
from praxis import test_utils
from praxis.layers.quantization import aqt

JTensor = pytypes.JTensor


class AqtTest(test_utils.TestCase):

  def get_quantized_and_scale(self, p_quant, sample) -> Tuple[JTensor, JTensor]:
    quant = p_quant.Instantiate()
    state = quant.init(jax.random.PRNGKey(0))
    scale = quant.apply(
        state, sample, [0, 1], jnp.float32, method=quant.get_quant_scale
    )
    qx = quant.apply(state, sample / scale, jnp.float32, method=quant.to_quant)
    qx = qx * scale

    return qx, scale

  def test_single_quant_example(self):
    """Compares quantization to hand-computed example."""
    x = jnp.array([
        [0.99, 1.01, 1.99, 2.01],  #
        [2.99, 3.01, 3.99, 4.01],  #
        [4.99, 5.01, 5.99, 7.0],  #
        [-0.99, -1.01, -1.99, -2.01],  #
        [-2.99, -3.01, -3.99, -4.01],  #
        [-4.99, -5.01, -5.99, -7.0],  #
    ], dtype=jnp.float32)
    expected_output = jnp.array([
        [0.00, 2.00, 2.00, 2.00],  #
        [2.00, 4.00, 4.00, 4.00],  #
        [4.00, 6.00, 6.00, 6.00],  #
        [-0.00, -2.00, -2.00, -2.00],  #
        [-2.00, -4.00, -4.00, -4.00],  #
        [-4.00, -6.00, -6.00, -6.00],  #
    ], dtype=jnp.float32)

    p_quant = pax_fiddle.Config(aqt.TensorQuantizer, name='tq', precision=3)

    qx, scale = self.get_quantized_and_scale(p_quant, x)

    self.assertEqual(scale, jnp.full((1, 1), 2.0, dtype=jnp.float32))
    self.assertArraysEqual(qx, expected_output)

  def test_none_prec_not_quantize(self):
    x = jax.random.uniform(
        jax.random.PRNGKey(0), shape=(4, 5), dtype=jnp.float32)
    p_quant = pax_fiddle.Config(aqt.TensorQuantizer, name='tq', precision=None)
    qx, scale = self.get_quantized_and_scale(p_quant, x)

    self.assertEqual(scale, jnp.full((1, 1), 1.0, dtype=jnp.float32))
    self.assertArraysEqual(qx, x)

  def test_quant_noise_for_different_scaling_granularities(self):
    """Ensures per_example scaling produces smaller noises than per_tensor."""
    x = jax.random.normal(
        jax.random.PRNGKey(0), shape=(128, 512), dtype=jnp.float32
    )
    y = jax.random.normal(
        jax.random.PRNGKey(0), shape=(512, 256), dtype=jnp.float32
    )
    p_quant = pax_fiddle.Config(aqt.TensorQuantizer, name='tq', precision=8)
    quant = p_quant.Instantiate()
    state = quant.init(jax.random.PRNGKey(0))

    per_example_scale = quant.apply(
        state, x, 1, jnp.float32, method=quant.get_quant_scale
    )
    per_tensor_scale = quant.apply(
        state, x, None, jnp.float32, method=quant.get_quant_scale
    )

    per_example_qx = quant.apply(
        state, x * per_example_scale, jnp.float32, method=quant.to_quant
    )
    per_tensor_qx = quant.apply(
        state, x * per_tensor_scale, jnp.float32, method=quant.to_quant
    )

    float_result = jax.lax.dot(x, y)

    per_example_result = jax.lax.dot(per_example_qx, y)
    per_example_result = per_example_result / per_example_scale

    per_tensor_result = jax.lax.dot(per_tensor_qx, y)
    per_tensor_result = per_tensor_result / per_tensor_scale

    per_example_error = jnp.sum((float_result - per_example_result)**2)
    per_tensor_error = jnp.sum((float_result - per_tensor_result)**2)

    self.assertLessEqual(per_example_error, per_tensor_error)

  def test_get_quant_scale_zeros(self):
    p_quant = pax_fiddle.Config(aqt.TensorQuantizer, name='tq', precision=8)
    quant = p_quant.Instantiate()
    state = quant.init(jax.random.PRNGKey(0))
    scale = quant.apply(
        state, jnp.zeros((1, 4)), 1, jnp.float32, method=quant.get_quant_scale)
    self.assertFalse(jnp.isinf(scale).any())
    self.assertEqual(scale, jnp.zeros_like(scale))


if __name__ == '__main__':
  absltest.main()
