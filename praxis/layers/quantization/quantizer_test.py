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

"""Tests for TensorQuantizer in Quantization-aware Training."""

from typing import Tuple

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import pax_fiddle
from praxis import pytypes
from praxis import test_utils
from praxis.layers.quantization import quantizer

JTensor = pytypes.JTensor


class QuantizerTest(test_utils.TestCase):

  def get_quantize_dequantized_and_scale(
      self, p_quant, sample, axis=None
  ) -> Tuple[JTensor, JTensor]:
    # Computes quantized-dequantized and scale of input sample.

    quant = p_quant.Instantiate()
    state = quant.init(jax.random.PRNGKey(0))

    # Quantize.
    q_x, q_scale, zp_time_scale = quant.apply(
        state, sample, axis, False, jnp.float32, method=quant.quantize
    )

    # Dequantize.
    deq_q_x = quant.apply(
        state, q_x, q_scale, axis, zp_time_scale, method=quant.dequantize
    )

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
            [0.0, 1.0, 2.0, 2.0],  #
            [2.0, 3.0, 3.0, 4],  #
            [4.0, 5.0, 5.0, 7.0],  #
            [-0.0, -1.0, -1.0, -2.0],  #
            [-2.0, -3.0, -3.0, -4.0],  #
            [-4.0, -5.0, -5.0, -7.0],  #
        ],
        dtype=jnp.float32,
    )
    expected_q_deq = jnp.array(
        [
            [0.0, 0.0, 2.3333335, 2.3333335],  #
            [2.3333335, 2.3333335, 2.3333335, 4.666667],  #
            [4.666667, 4.666667, 4.666667, 7.0000005],  #
            [0.0, 0.0, 0.0, -2.3333335],  #
            [-2.3333335, -2.3333335, -2.3333335, -4.666667],  #
            [-4.666667, -4.666667, -4.666667, -7.0000005],  #
        ],
        dtype=jnp.float32,
    )
    expected_q = jnp.array(
        [
            [0, 0, 1, 1],  #
            [1, 1, 1, 2],  #
            [2, 2, 2, 3],  #
            [0, 0, 0, -1],  #
            [-1, -1, -1, -2],  #
            [-2, -2, -2, -3],  #
        ],
        dtype=jnp.float32,
    )

    p_quant = pax_fiddle.Config(
        quantizer.TensorQuantizer, name='tq', precision=3, add_scale_eps=add_scale_eps
    )

    q_x, q_deq_x, scale = self.get_quantize_dequantized_and_scale(p_quant, x)

    # Validate quantized range for 3 bits precision:
    self.assertLessEqual(-4, jnp.min(q_x))
    self.assertGreaterEqual(3, jnp.max(q_x))

    self.assertAllClose(scale, jnp.full((1, 1), 2.333333, dtype=jnp.float32))
    self.assertArraysEqual(q_x, expected_q)
    self.assertAllClose(q_deq_x, expected_q_deq, atol=1e-6)

  def test_none_prec_not_quantize(self):
    x = jax.random.uniform(
        jax.random.PRNGKey(0), shape=(4, 5), dtype=jnp.float32
    )
    p_quant = pax_fiddle.Config(quantizer.TensorQuantizer, name='tq', precision=None)
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
    p_quant = pax_fiddle.Config(quantizer.TensorQuantizer, name='tq', precision=8)
    quant = p_quant.Instantiate()
    state = quant.init(jax.random.PRNGKey(0))

    per_example_scale, _ = quant.apply(
        state, x, 1, method=quant.get_quant_scale
    )
    per_tensor_scale, _ = quant.apply(
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
        quantizer.TensorQuantizer,
        name='tq',
        precision=8,
        stop_scale_gradient=stop_scale_gradient,
    )
    quant = p_quant.Instantiate()
    state = quant.init(jax.random.PRNGKey(0))
    x = jnp.zeros((1, 4))
    scale, _ = quant.apply(
        state,
        x,
        contract_dims=1,
        method=quant.get_quant_scale,
    )
    x_scaled = x * scale
    self.assertArraysEqual(x_scaled, jnp.zeros_like(x_scaled))
    x_rescaled = x_scaled / scale
    self.assertArraysEqual(x_rescaled, jnp.zeros_like(x_rescaled))

  def _quant_with_dequant_error(self, quant, state, x, contract_dims):
    # Compute dequantization error with the symmetric quantizer:
    q_x, q_s, zp_time_scale = quant.apply(
        state,
        x,
        contract_dims,
        squeeze_scale=False,
        method=quant.quantize,
    )
    q_deq_x = quant.apply(
        state,
        q_x,
        q_s,
        contract_dims,
        zp_time_scale,
        method=quant.dequantize,
    )
    sum_error = jnp.sum(jnp.abs(jnp.subtract(x, q_deq_x)))
    return q_x, sum_error

  def test_clipping_optimization(self):
    sym_p_quant = pax_fiddle.Config(
        quantizer.TensorQuantizer,
        name='sym_quant',
        precision=4,
        use_symmetric=True,
    )
    asym_p_quant = pax_fiddle.Config(
        quantizer.TensorQuantizer,
        name='asym_quant',
        precision=4,
        use_symmetric=False,
    )
    sym_p_quant_opt = pax_fiddle.Config(
        quantizer.TensorQuantizer,
        name='sym_quant_opt',
        precision=4,
        min_clipping=0.8,
        num_optimize_clipping=8,
        use_symmetric=True,
    )
    asym_p_quant_opt = pax_fiddle.Config(
        quantizer.TensorQuantizer,
        name='asym_quant_opt',
        precision=4,
        min_clipping=0.8,
        num_optimize_clipping=8,
        use_symmetric=False,
    )
    sym_quant = sym_p_quant.Instantiate()
    asym_quant = asym_p_quant.Instantiate()
    sym_quant_opt = sym_p_quant_opt.Instantiate()
    asym_quant_opt = asym_p_quant_opt.Instantiate()

    state = sym_quant.init(jax.random.PRNGKey(0))
    batch_size = 3
    feature_dim1 = 2
    feature_dim2 = 256
    input_shape = [batch_size, feature_dim1, feature_dim2]
    x = jax.random.normal(jax.random.PRNGKey(12), input_shape)
    contract_dims = -1

    sym_q, sym_error = self._quant_with_dequant_error(
        sym_quant, state, x, contract_dims
    )
    asym_q, asym_error = self._quant_with_dequant_error(
        asym_quant, state, x, contract_dims
    )
    sym_opt_q, sym_opt_error = self._quant_with_dequant_error(
        sym_quant_opt, state, x, contract_dims
    )
    asym_opt_q, asym_opt_error = self._quant_with_dequant_error(
        asym_quant_opt, state, x, contract_dims
    )

    # Validate that x is quantized
    self.assertEqual(7, jnp.max(sym_q))
    self.assertEqual(-7, jnp.min(sym_q))
    self.assertEqual(7, jnp.max(asym_q))
    self.assertEqual(-8, jnp.min(asym_q))
    self.assertEqual(7, jnp.max(sym_opt_q))
    self.assertEqual(-8, jnp.min(sym_opt_q))
    self.assertEqual(7, jnp.max(asym_opt_q))
    self.assertEqual(-8, jnp.min(asym_opt_q))

    delta = 2  # Set experimentally.
    # They must be in sorted order as below:
    self.assertLess(asym_error, sym_error - delta)
    self.assertLess(sym_opt_error, asym_error - delta)
    self.assertLess(asym_opt_error, sym_opt_error - delta)

  @parameterized.named_parameters(
      dict(testcase_name='1bit', precision=1),
      dict(testcase_name='2bit', precision=2),
      dict(testcase_name='4bit', precision=4),
      dict(testcase_name='8bit', precision=8))
  def test_clip_to_unsigned_int(self, precision):
    """Checks if an input gets clipped to [0, 2**precision-1] when unsigned_int=True."""
    p_quant = pax_fiddle.Config(
        quantizer.TensorQuantizer,
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
    scale, _ = quant.apply(
        state, x, [0, 1], method=quant.get_quant_scale
    )
    ix = quant.apply(state, x / scale, method=quant.to_quant)

    self.assertGreaterEqual(jnp.min(ix), 0.0)
    self.assertLessEqual(jnp.max(ix), jnp.float32(2**precision - 1))
    self.assertArraysEqual(ix, jnp.round(ix))

  def test_single_quant_with_unsigned_int_bound(self):
    p_quant = pax_fiddle.Config(
        quantizer.TensorQuantizer,
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
        quantizer.TensorQuantizer,
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
        scale, jnp.array([0.016863, 0.021569], dtype=jnp.float32)
    )
    self.assertAllClose(
        zp, jnp.array([-3.358431, -1.260784], dtype=jnp.float32)
    )

  @parameterized.named_parameters(
      dict(testcase_name='2bit', precision=2),
      dict(testcase_name='4bit', precision=4),
      dict(testcase_name='8bit', precision=8),
  )
  def test_asymmetric_quant_error_smaller_than_symmetric(self, precision):
    p_quant_asymmetric = pax_fiddle.Config(
        quantizer.TensorQuantizer,
        name='asymmetric',
        precision=precision,
        add_scale_eps=False,
        use_symmetric=False,
    )
    p_quant_symmetric = pax_fiddle.Config(
        quantizer.TensorQuantizer,
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
        quantizer.TensorQuantizer,
        name='asymmetric',
        precision=precision,
        add_scale_eps=False,
        use_symmetric=False,
    )
    p_quant_unsigned = pax_fiddle.Config(
        quantizer.TensorQuantizer,
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

  def test_clipped_asymmetric_quant_error_less_than_non_clipped(self):
    precision = 4
    contract_dims = [1]
    p_clip = pax_fiddle.Config(
        quantizer.TensorQuantizer,
        name='clipped',
        precision=precision,
        add_scale_eps=False,
        use_symmetric=False,
        clipping_coeff=0.8,
    )
    p_no_clip = pax_fiddle.Config(
        quantizer.TensorQuantizer,
        name='not_clipped',
        precision=precision,
        add_scale_eps=False,
        use_symmetric=False,
        clipping_coeff=None,
    )

    x = jax.random.normal(
        jax.random.PRNGKey(0), shape=(4, 128), dtype=jnp.float32
    )
    x_quant_clip, x_dequant_clip, q_scale_clip = (
        self.get_quantize_dequantized_and_scale(
            p_clip, x, axis=contract_dims
        )
    )
    self.assertEqual((4, 1), q_scale_clip.shape)
    quant_error_clip = jnp.sum(jnp.abs(x_dequant_clip - x))

    self.assertEqual(2 ** (precision - 1) - 1, jnp.max(x_quant_clip))
    self.assertEqual(-2 ** (precision - 1), jnp.min(x_quant_clip))

    x_quant_no_clip, x_dequant_no_clip, _ = (
        self.get_quantize_dequantized_and_scale(
            p_no_clip, x, axis=contract_dims
        )
    )
    self.assertEqual(2 ** (precision - 1) - 1, jnp.max(x_quant_no_clip))
    self.assertEqual(-2 ** (precision - 1), jnp.min(x_quant_no_clip))
    quant_error_no_clip = jnp.sum(jnp.abs(x_dequant_no_clip - x))

    # Note that if x has uniform distribution then below will be false.
    self.assertLessEqual(quant_error_clip, quant_error_no_clip)

  @parameterized.named_parameters(
      dict(testcase_name='2bit', precision=2),
      dict(testcase_name='4bit', precision=4),
  )
  def test_asymmetric_quant_error_less_than_symmetric(self, precision):
    contract_dims = [1]
    p_quant_asymmetric = pax_fiddle.Config(
        quantizer.TensorQuantizer,
        name='asymmetric',
        precision=precision,
        add_scale_eps=False,
        use_symmetric=False,
    )
    p_quant_symmetric = pax_fiddle.Config(
        quantizer.TensorQuantizer,
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

  def test_aux_quantization_loss(self):
    p_quant = pax_fiddle.Config(
        quantizer.TensorQuantizer,
        name='quant',
        precision=4,
        quant_loss_weight=1,
    )

    axis = None
    x = jax.random.uniform(
        jax.random.PRNGKey(0), shape=(4, 5), dtype=jnp.float32
    )

    context_p = base_layer.JaxContext.HParams(do_eval=True)
    with base_layer.JaxContext.new_context(hparams=context_p):

      quant = p_quant.Instantiate()
      state = quant.init(jax.random.PRNGKey(0))
      _, states = quant.apply(
          state,
          x,
          axis,
          False,
          quantized_dtype=jnp.int8,
          method=quant.quantize,
          mutable=True,
      )
      self.assertNotIn(base_layer.AUX_LOSS, states)

    context_p = base_layer.JaxContext.HParams(do_eval=False)
    with base_layer.JaxContext.new_context(hparams=context_p):
      quant = p_quant.Instantiate()
      state = quant.init(jax.random.PRNGKey(0))
      _, states = quant.apply(
          state,
          x,
          axis,
          False,
          quantized_dtype=None,
          method=quant.quantize,
          mutable=True,
      )
      self.assertIn(base_layer.AUX_LOSS, states)

      _, states = quant.apply(
          state,
          x,
          axis,
          False,
          quantized_dtype=jnp.int8,
          method=quant.quantize,
          mutable=True,
      )
      self.assertNotIn(base_layer.AUX_LOSS, states)

  def test_clipping_per_channel(self):
    np.random.seed(0)
    x = np.random.uniform(size=[2, 512])

    # The first feature will be scaled to 1.0 without clipping, while the second
    # feature will be clipped to mitigate the effect of the outlier.
    x[0, :] = 1.0
    x[1, -1] = 2.0

    p_quant = pax_fiddle.Config(
        quantizer.TensorQuantizer,
        name='quant',
        precision=2,
        min_clipping=0.8,
        num_optimize_clipping=12,
        use_symmetric=True,
        add_scale_eps=False,
        optimize_clipping_per_channel=True,
    )

    axis = [1]
    quant = p_quant.Instantiate()
    state = quant.init(jax.random.PRNGKey(0))
    _, q_s, _ = quant.apply(
        state,
        x,
        axis,
        False,
        quantized_dtype=jnp.int8,
        method=quant.quantize,
    )

    # Test that the first feature isn't clipped.
    self.assertAllClose(q_s[0, 0], 1.0)
    # Test that the second feature is clipped and will have a smaller max value
    # than the outlier we added = 2.0.
    self.assertAllClose(q_s[1, 0], 1.6)

  @parameterized.named_parameters(
      dict(testcase_name='2bit', precision=2),
      dict(testcase_name='4bit', precision=4),
  )
  def test_sub_channel_quant_error_less_than_standard(self, precision):
    contract_dims = [1]
    p_quant = pax_fiddle.Config(
        quantizer.TensorQuantizer,
        name='asymmetric',
        precision=precision,
        add_scale_eps=False,
        use_symmetric=False,
    )
    p_quant_sub = pax_fiddle.Config(
        quantizer.TensorQuantizer,
        name='asymmetric_sub_channel',
        precision=precision,
        add_scale_eps=False,
        use_symmetric=False,
        sub_channels=8,
    )

    x = jax.random.uniform(
        jax.random.PRNGKey(0), shape=(2, 128), minval=-1, maxval=1
    )

    x_quant, x_dequant, q_scale = (
        self.get_quantize_dequantized_and_scale(
            p_quant, x, axis=contract_dims
        )
    )
    self.assertEqual((2, 1), q_scale.shape)
    quant_error = jnp.sum(jnp.abs(x_dequant - x))

    self.assertEqual(2 ** (precision - 1) - 1, jnp.max(x_quant))
    self.assertEqual(-2 ** (precision - 1), jnp.min(x_quant))

    x_quant_sub, x_dequant_sub, _ = (
        self.get_quantize_dequantized_and_scale(
            p_quant_sub, x, axis=contract_dims
        )
    )
    self.assertEqual(2 ** (precision - 1) - 1, jnp.max(x_quant_sub))
    self.assertLessEqual(-(2 ** (precision - 1)), jnp.min(x_quant_sub))
    quant_error_sub = jnp.sum(jnp.abs(x_dequant_sub - x))

    self.assertLessEqual(quant_error, quant_error_sub)


if __name__ == '__main__':
  absltest.main()
