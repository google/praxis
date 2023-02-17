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

  def get_quantize_dequantized_and_scale(
      self, p_quant, sample, axis=None
  ) -> Tuple[JTensor, JTensor]:
    # Computes quantized-dequantized and scale of input sample.

    quant = p_quant.Instantiate()
    state = quant.init(jax.random.PRNGKey(0))

    # Quantize.
    q_x, q_scale, zp = quant.apply(
        state, sample, axis, False, jnp.float32, method=quant.quantize
    )

    # Dequantize.
    if zp is None:
      deq_q_x = q_x * q_scale
    else:
      zp = jnp.expand_dims(zp, axis=axis)
      deq_q_x = q_x * q_scale - zp

    return q_x, deq_q_x, q_scale

  @parameterized.named_parameters(
      dict(testcase_name='add_eps_to_scale', add_scale_eps=True),
      dict(testcase_name='replace_zero_scale_by_one', add_scale_eps=False),
  )
  def test_single_symmetric_quant_example(self, add_scale_eps=False):
    """Compares quantization to hand-computed example."""
    # representable values: -6, -4, -2, 0, 2, 4, 6
    x = jnp.array(
        [
            [0.99, 1.01, 1.99, 2.01],  #
            [2.99, 3.01, 3.99, 4.01],  #
            [4.99, 5.01, 5.99, 7.0],  #
            [-0.99, -1.01, -1.99, -2.01],  #
            [-2.99, -3.01, -3.99, -4.01],  #
            [-4.99, -5.01, -5.99, -7.0],  #
        ],
        dtype=jnp.float32,
    )
    expected_q_deq = jnp.array(
        [
            [0.00, 2.00, 2.00, 2.00],  #
            [2.00, 4.00, 4.00, 4.00],  #
            [4.00, 6.00, 6.00, 6.00],  #
            [-0.00, -2.00, -2.00, -2.00],  #
            [-2.00, -4.00, -4.00, -4.00],  #
            [-4.00, -6.00, -6.00, -6.00],  #
        ],
        dtype=jnp.float32,
    )
    expected_q = jnp.array(
        [
            [0, 1, 1, 1],  #
            [1, 2, 2, 2],  #
            [2, 3, 3, 3],  #
            [0, -1, -1, -1],  #
            [-1, -2, -2, -2],  #
            [-2, -3, -3, -3],  #
        ],
        dtype=jnp.float32,
    )

    p_quant = pax_fiddle.Config(
        aqt.TensorQuantizer, name='tq', precision=3, add_scale_eps=add_scale_eps
    )

    q_x, q_deq_x, scale = self.get_quantize_dequantized_and_scale(p_quant, x)

    # Validate quantized range for 3 bits precision:
    self.assertLessEqual(-4, jnp.min(q_x))
    self.assertGreaterEqual(3, jnp.max(q_x))

    self.assertAllClose(scale, jnp.full((1, 1), 2, dtype=jnp.float32))
    self.assertArraysEqual(q_x, expected_q)
    self.assertAllClose(q_deq_x, expected_q_deq, atol=1e-6)

  def test_none_prec_not_quantize(self):
    x = jax.random.uniform(
        jax.random.PRNGKey(0), shape=(4, 5), dtype=jnp.float32
    )
    p_quant = pax_fiddle.Config(aqt.TensorQuantizer, name='tq', precision=None)
    _, q_deq_x, scale = self.get_quantize_dequantized_and_scale(p_quant, x)

    self.assertEqual(scale, jnp.full((1, 1), 1.0, dtype=jnp.float32))
    self.assertArraysEqual(q_deq_x, x)

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
        state, x, 1, method=quant.get_quant_scale
    )
    per_tensor_scale = quant.apply(
        state, x, None, method=quant.get_quant_scale
    )

    per_example_qx = quant.apply(
        state, x * per_example_scale, method=quant.to_quant
    )
    per_tensor_qx = quant.apply(
        state, x * per_tensor_scale, method=quant.to_quant
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
        method=quant.get_quant_scale,
    )
    x_scaled = x * scale
    self.assertArraysEqual(x_scaled, jnp.zeros_like(x_scaled))
    x_rescaled = x_scaled / scale
    self.assertArraysEqual(x_rescaled, jnp.zeros_like(x_rescaled))

  def test_clipping_optimization(self):
    p_quant = pax_fiddle.Config(
        aqt.TensorQuantizer,
        name='quant',
        precision=4,
    )
    p_quant_opt = pax_fiddle.Config(
        aqt.TensorQuantizer,
        name='quant_opt',
        precision=4,
        min_clipping=0.8,
        num_optimize_clipping=8,
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
        method=quant.get_quant_scale,
    )
    self.assertEqual(scale.shape, (batch_size, feature_dim1, 1))

    x_q = quant.apply(state, x / scale, method=quant.to_quant)
    x_q_deq = jnp.multiply(scale, x_q)
    sum_error = jnp.sum(jnp.abs(jnp.subtract(x, x_q_deq)))

    # Compute dequantization error with the clipping optimization:
    scale_opt = quant_opt.apply(
        state,
        x,
        contract_dims=-1,
        method=quant.get_quant_scale,
    )
    x_q_opt = quant_opt.apply(
        state, x / scale_opt, method=quant_opt.to_quant
    )
    x_q_deq_opt = jnp.multiply(scale_opt, x_q_opt)
    sum_error_opt = jnp.sum(jnp.abs(jnp.subtract(x, x_q_deq_opt)))

    # Validate that x is quantized
    self.assertEqual(7, jnp.max(x_q))
    self.assertEqual(-7, jnp.min(x_q))

    self.assertEqual(7, jnp.max(x_q_opt))
    self.assertEqual(-8, jnp.min(x_q_opt))
    sum_x_q = jnp.sum(jnp.abs(x_q))
    sum_x_q_opt = jnp.sum(jnp.abs(x_q_opt))
    self.assertNotEqual(sum_x_q, sum_x_q_opt)

    # Validated that quantization with optimization has lower error.
    # With feature_dim2 we observe that difference between sum_error_opt and
    # sum_error belongs to range: 10...30, so selected 20 as middle point.
    self.assertLess(sum_error_opt, sum_error-20)

  @parameterized.named_parameters(
      dict(testcase_name='1bit', precision=1),
      dict(testcase_name='2bit', precision=2),
      dict(testcase_name='4bit', precision=4),
      dict(testcase_name='8bit', precision=8))
  def test_clip_to_unsigned_int(self, precision):
    """Checks if an input gets clipped to [0, 2**precision-1] when unsigned_int=True."""
    p_quant = pax_fiddle.Config(
        aqt.TensorQuantizer,
        name='tq',
        precision=precision,
        unsigned_int_bounds=True,
    )

    x = jax.random.uniform(
        jax.random.PRNGKey(0), shape=(1024, 1), minval=0, maxval=1.0
    )
    x *= 2.0**8
    quant = p_quant.Instantiate()
    state = quant.init(jax.random.PRNGKey(0))
    scale = quant.apply(
        state, x, [0, 1], method=quant.get_quant_scale
    )
    ix = quant.apply(state, x / scale, method=quant.to_quant)

    self.assertGreaterEqual(jnp.min(ix), 0.0)
    self.assertLessEqual(jnp.max(ix), jnp.float32(2**precision - 1))
    self.assertArraysEqual(ix, jnp.round(ix))

  def test_single_quant_with_unsigned_int_bound(self):
    p_quant = pax_fiddle.Config(
        aqt.TensorQuantizer,
        name='tq',
        precision=3,
        unsigned_int_bounds=True
    )

    x = jnp.array([
        [0.99, 1.01, 2.99, 3.01],  #
        [4.99, 5.01, 6.99, 7.01],  #
        [8.99, 9.01, 10.99, 11.00],  #
        [12.99, 13.01, 13.99, 14.00],  #
        [-0.99, -1.01, -2.99, -3.01],  #
        [-4.99, -5.01, -6.99, -7.01],  #
        [-8.99, -9.01, -10.99, -11.00],  #
        [-12.99, -13.01, -13.99, -14.00],  #
    ], dtype=jnp.float32)

    expected_q_deq_x = jnp.array([
        [0.00, 2.00, 2.00, 4.00],  #
        [4.00, 6.00, 6.00, 8.00],  #
        [8.00, 10.00, 10.00, 12.00],  #
        [12.00, 14.00, 14.00, 14.00],  #
        [-0.00, -0.00, -0.00, -0.00],  #
        [-0.00, -0.00, -0.00, -0.00],  #
        [-0.00, -0.00, -0.00, -0.00],  #
        [-0.00, -0.00, -0.00, -0.00],  #
    ], dtype=jnp.float32)

    expected_q_x = jnp.array([
        [0., 1., 1., 2.],  #
        [2., 3., 3., 4.],  #
        [4., 5., 5., 6.],  #
        [6., 7., 7., 7.],  #
        [0., 0., 0., 0.],  #
        [0., 0., 0., 0.],  #
        [0., 0., 0., 0.],  #
        [0., 0., 0., 0.],  #
    ], dtype=jnp.float32)

    q_x, q_deq_x, _ = self.get_quantize_dequantized_and_scale(p_quant, x)

    self.assertArraysEqual(q_deq_x, expected_q_deq_x)
    self.assertArraysEqual(q_x, expected_q_x)

  def test_quantize_asymmetric(self):
    p_quant = pax_fiddle.Config(
        aqt.TensorQuantizer,
        name='tq',
        precision=8,
        add_scale_eps=False,
        use_symmetric=False,
    )
    x = jnp.array([[1.2, 3.1, 5.5, 2.9], [0.2, -1.5, 3.3, 4.0]])
    quant = p_quant.Instantiate()
    state = quant.init(jax.random.PRNGKey(0))
    qx, scale, zp = quant.apply(
        state, x, [1], True, jnp.float32, method=quant.quantize
    )
    expected_qx = jnp.array(
        [[-128, -15, 127, -27], [-49, -128, 95, 127]], dtype=jnp.float32
    )
    self.assertTrue((qx == expected_qx).all())
    self.assertAllClose(
        scale, jnp.array([0.016797, 0.021484], dtype=jnp.float32)
    )
    self.assertAllClose(
        zp, jnp.array([-3.358399, -1.260742], dtype=jnp.float32)
    )

  @parameterized.named_parameters(
      dict(testcase_name='2bit', precision=2),
      dict(testcase_name='4bit', precision=4),
      dict(testcase_name='8bit', precision=8),
  )
  def test_asymmetric_quant_error_smaller_than_symmetric(self, precision):
    p_quant_asymmetric = pax_fiddle.Config(
        aqt.TensorQuantizer,
        name='asymmetric',
        precision=precision,
        add_scale_eps=False,
        use_symmetric=False,
    )
    p_quant_symmetric = pax_fiddle.Config(
        aqt.TensorQuantizer,
        name='symmetric',
        precision=precision,
        add_scale_eps=False,
        use_symmetric=True,
    )

    x = jnp.array([[1.2, 3.1, 5.5, 2.9], [0.2, -1.5, 3.3, 4.0]])

    _, x_dequant_asymmetric, _ = self.get_quantize_dequantized_and_scale(
        p_quant_asymmetric, x, axis=[1]
    )
    quant_error_asymmetric = jnp.sum(jnp.abs(x_dequant_asymmetric - x))

    _, x_dequant_symmetric, _ = self.get_quantize_dequantized_and_scale(
        p_quant_symmetric, x, axis=[1]
    )
    quant_error_symmetric = jnp.sum(jnp.abs(x_dequant_symmetric - x))

    self.assertLessEqual(quant_error_asymmetric, quant_error_symmetric)

  @parameterized.named_parameters(
      dict(testcase_name='2bit', precision=2),
      dict(testcase_name='4bit', precision=4),
      dict(testcase_name='8bit', precision=8),
  )
  def test_asymmetric_quant_error_less_than_unsigned(self, precision):
    contract_dims = [1]
    p_quant_asymmetric = pax_fiddle.Config(
        aqt.TensorQuantizer,
        name='asymmetric',
        precision=precision,
        add_scale_eps=False,
        use_symmetric=False,
    )
    p_quant_unsigned = pax_fiddle.Config(
        aqt.TensorQuantizer,
        name='symmetric',
        precision=precision,
        add_scale_eps=False,
        unsigned_int_bounds=True,
    )

    x = jax.random.uniform(
        jax.random.PRNGKey(0), shape=(4, 128), minval=0.1, maxval=1
    )

    x_quant_asym, x_dequant_asymmetric, q_scale_asym = (
        self.get_quantize_dequantized_and_scale(
            p_quant_asymmetric, x, axis=contract_dims
        )
    )
    self.assertEqual((4, 1), q_scale_asym.shape)
    quant_error_asymmetric = jnp.sum(jnp.abs(x_dequant_asymmetric - x))

    self.assertEqual(2 ** (precision - 1) - 1, jnp.max(x_quant_asym))
    self.assertEqual(-2 ** (precision - 1), jnp.min(x_quant_asym))

    x_quant_unsigned, x_dequant_unsigned, _ = (
        self.get_quantize_dequantized_and_scale(
            p_quant_unsigned, x, axis=contract_dims
        )
    )
    self.assertEqual(2**precision - 1, jnp.max(x_quant_unsigned))
    quant_error_unsigned = jnp.sum(jnp.abs(x_dequant_unsigned - x))

    self.assertLessEqual(quant_error_asymmetric, quant_error_unsigned)

  @parameterized.named_parameters(
      dict(testcase_name='2bit', precision=2),
      dict(testcase_name='4bit', precision=4),
  )
  def test_asymmetric_quant_error_less_than_symmetric(self, precision):
    contract_dims = [1]
    p_quant_asymmetric = pax_fiddle.Config(
        aqt.TensorQuantizer,
        name='asymmetric',
        precision=precision,
        add_scale_eps=False,
        use_symmetric=False,
    )
    p_quant_symmetric = pax_fiddle.Config(
        aqt.TensorQuantizer,
        name='symmetric',
        precision=precision,
        add_scale_eps=False,
        unsigned_int_bounds=False,
        use_symmetric=True,
    )

    x = jax.random.uniform(
        jax.random.PRNGKey(0), shape=(2, 128), minval=-1, maxval=1
    )

    x_quant_asym, x_dequant_asymmetric, q_scale_asym = (
        self.get_quantize_dequantized_and_scale(
            p_quant_asymmetric, x, axis=contract_dims
        )
    )
    self.assertEqual((2, 1), q_scale_asym.shape)
    quant_error_asymmetric = jnp.sum(jnp.abs(x_dequant_asymmetric - x))

    self.assertEqual(2 ** (precision - 1) - 1, jnp.max(x_quant_asym))
    self.assertEqual(-2 ** (precision - 1), jnp.min(x_quant_asym))

    x_quant_symmetric, x_dequant_symmetric, _ = (
        self.get_quantize_dequantized_and_scale(
            p_quant_symmetric, x, axis=contract_dims
        )
    )
    self.assertEqual(2 ** (precision - 1) - 1, jnp.max(x_quant_symmetric))
    self.assertLessEqual(-(2 ** (precision - 1)), jnp.min(x_quant_symmetric))
    quant_error_symmetric = jnp.sum(jnp.abs(x_dequant_symmetric - x))

    self.assertLessEqual(quant_error_asymmetric, quant_error_symmetric)

  def test_clip_bound(self):
    precision = 8
    p_quant = pax_fiddle.Config(
        aqt.TensorQuantizer,
        name='signed',
        precision=precision,
        add_scale_eps=False,
        use_symmetric=True,
    )
    quant = p_quant.Instantiate()
    state = quant.init(jax.random.PRNGKey(0))
    bounds = quant.apply(
        state, method=quant._get_clip_bound
    )
    self.assertEqual(bounds, (-128.5, 127.5))

    p_quant = pax_fiddle.Config(
        aqt.TensorQuantizer,
        name='unsigned',
        precision=precision,
        add_scale_eps=False,
        use_symmetric=True,
        unsigned_int_bounds=True,
    )
    quant = p_quant.Instantiate()
    state = quant.init(jax.random.PRNGKey(0))
    bounds = quant.apply(
        state, method=quant._get_clip_bound
    )
    self.assertEqual(bounds, (0, 255))


if __name__ == '__main__':
  absltest.main()
