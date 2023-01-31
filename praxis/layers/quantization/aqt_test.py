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
from absl.testing import parameterized
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

    self.assertAllClose(scale, jnp.full((1, 1), 2.0, dtype=jnp.float32))
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

  @parameterized.named_parameters(
      dict(testcase_name='scale_gradient', stop_scale_gradient=False),
      dict(testcase_name='stop_scale_gradient', stop_scale_gradient=True),
  )
  def test_zeros_quant_rescaling(self, stop_scale_gradient):
    p_quant = pax_fiddle.Config(
        aqt.TensorQuantizer,
        name='tq',
        precision=8,
        stop_scale_gradient=stop_scale_gradient,
    )
    quant = p_quant.Instantiate()
    state = quant.init(jax.random.PRNGKey(0))
    x = jnp.zeros((1, 4))
    scale = quant.apply(
        state,
        x,
        contract_dims=1,
        dtype=jnp.float32,
        method=quant.get_quant_scale,
    )
    x_scaled = x * scale
    self.assertArraysEqual(x_scaled, jnp.zeros_like(x_scaled))
    x_rescaled = x_scaled / scale
    self.assertArraysEqual(x_rescaled, jnp.zeros_like(x_rescaled))

  def test_clipping_optimization(self):
    p_quant = pax_fiddle.Config(
        aqt.TensorQuantizer,
        name='tq',
        precision=4,
    )
    p_quant_opt = pax_fiddle.Config(
        aqt.TensorQuantizer,
        name='tq',
        precision=4,
        min_clipping=0.4,
        num_optimize_clipping=12,
    )
    quant = p_quant.Instantiate()
    quant_opt = p_quant_opt.Instantiate()
    state = quant.init(jax.random.PRNGKey(0))
    batch_size = 3
    feature_dim1 = 2
    feature_dim2 = 256
    input_shape = [batch_size, feature_dim1, feature_dim2]
    x = jax.random.normal(jax.random.PRNGKey(12), input_shape)

    # Compute dequantization error with the standard quantizer:
    scale = quant.apply(
        state,
        x,
        contract_dims=-1,
        dtype=jnp.float32,
        method=quant.get_quant_scale,
    )
    x_q = quant.apply(state, x / scale, jnp.int8, method=quant.to_quant)
    x_q_deq = jnp.multiply(scale, x_q)
    sum_error = jnp.sum(jnp.abs(jnp.subtract(x, x_q_deq)))

    # Compute dequantization error with the clipping optimization:
    scale_opt = quant_opt.apply(
        state,
        x,
        contract_dims=-1,
        dtype=jnp.float32,
        method=quant.get_quant_scale,
    )
    x_q_opt = quant_opt.apply(
        state, x / scale_opt, jnp.int8, method=quant_opt.to_quant
    )
    x_q_deq_opt = jnp.multiply(scale_opt, x_q_opt)
    sum_error_opt = jnp.sum(jnp.abs(jnp.subtract(x, x_q_deq_opt)))

    # Validate that x is quantized
    self.assertEqual(7, jnp.max(x_q))
    self.assertEqual(7, jnp.max(x_q_opt))
    self.assertEqual(7, -jnp.min(x_q))
    self.assertEqual(7, -jnp.min(x_q_opt))
    sum_x_q = jnp.sum(jnp.abs(x_q))
    sum_x_q_opt = jnp.sum(jnp.abs(x_q_opt))
    self.assertNotEqual(sum_x_q, sum_x_q_opt)

    # Validated that quantization with optimization has lower error.
    # With feature_dim2 we observe that difference between sum_error_opt and
    # sum_error belongs to range: 10...30, so selected 20 as middle point.
    self.assertLess(sum_error_opt, sum_error-20)


if __name__ == '__main__':
  absltest.main()
