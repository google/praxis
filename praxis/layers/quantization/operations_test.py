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

"""Tests for quantized operations."""

from typing import Any, Dict, Sequence

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import pax_fiddle
from praxis import test_utils
from praxis.layers.quantization import aqt
from praxis.layers.quantization import operations


class QuantizationUtilsTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  @parameterized.named_parameters(
      ('regular_eqn', 'ab,bc->ac'),
      ('eqn_with_dot', '...y,yz->...z'),
  )
  def test_quantized_einsum(self, eqn):
    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 1.0, 2.0]], dtype=jnp.bfloat16)
    w = jnp.array([[1, 2, 1], [2, 1, 2], [1, 3, 1]], dtype=jnp.int8)
    s = jnp.array([0.1, 0.2, 0.3], dtype=jnp.bfloat16)

    ret = operations.einsum(eqn, x, w, s)
    expected = jnp.array([[0.800781, 2.60938, 2.40625], [0.800781, 3, 2.40625]],
                         dtype=jnp.bfloat16)
    self.assertArraysEqual(ret, expected)

  def test_quantized_einsum_with_expand_dim(self):
    # pylint: disable=invalid-name
    A, B, D, K, N, H = 6, 4, 5, 3, 7, 2
    # pylint: enable=invalid-name
    x = jnp.ones([A, B, D], dtype=jnp.bfloat16)
    w = jnp.ones([K, D, N, H], dtype=jnp.int8)
    s = jnp.ones([K, N, H], dtype=jnp.bfloat16)

    ret = operations.einsum('ABD,KDNH->KABNH', x, w, s)
    expected = jnp.ones([K, A, B, N, H], dtype=jnp.bfloat16) * D
    self.assertArraysEqual(ret, expected)


class ReducePrecisionEinsumTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  @parameterized.named_parameters(
      ('eqn1', 'ab,bc->ac', (4, 3), (3,), ()),
      ('eqn2', '...y,yz->...z', (6, 5), (5,), ()),
      ('eqn3', 'ABD,KDNH->KABNH', (2, 3, 4, 5), (2, 4, 5), (1)),
  )
  def test_reduce_einsum_weight_precision(self, eqn, w_shape,
                                          expected_scale_shape, expand_dims):

    weight = np.random.normal(1.5, 2.0, w_shape).astype(np.float32)
    reduced_weight, scale = operations.reduce_einsum_weight_precision(
        eqn, weight)
    self.assertEqual(scale.shape, expected_scale_shape)
    if expand_dims:
      scale = jnp.expand_dims(scale, expand_dims)
    self.assertAllClose(
        weight,
        jnp.multiply(reduced_weight, scale).astype(jnp.float32),
        rtol=0.02,
        atol=0.02)
    weight_nudged = operations.fakequant_einsum(eqn, weight)
    self.assertAllClose(weight, weight_nudged, rtol=0.02, atol=0.02)

  def test_percentile(self):
    weight = np.random.normal(-2.0, 2.0, (4, 3)).astype(np.float32)
    reduced_weight, scale = operations.reduce_einsum_weight_precision(
        'ab,bc->ac', weight, percentile=0.9
    )
    # Large value discrepancy is expected since we use 0.9 percentile.
    # This just makes sure the percentile logic is correct. In practice, 0.9 is
    # small for any real use case.
    tol = 0.5
    self.assertEqual(scale.shape, (3,))
    self.assertAllClose(
        weight,
        jnp.multiply(reduced_weight, scale).astype(jnp.float32),
        rtol=tol,
        atol=tol,
    )

  def test_reduce_activation_precision(self):
    act = np.random.normal(-1.0, 1.0, [10, 100]).astype(np.float32)
    act_nudged = operations.fakequant_activation(act)
    self.assertAllClose(act, act_nudged, rtol=0.02, atol=0.02)


class DotGeneral(base_layer.BaseLayer):
  lhs_prec: int = 8
  rhs_prec: int = 8

  def setup(self):
    self.create_child(
        'lhs_quantizer',
        pax_fiddle.Config(
            aqt.TensorQuantizer, name='lhs_quantizer', precision=self.lhs_prec
        ),
    )
    self.create_child(
        'rhs_quantizer',
        pax_fiddle.Config(
            aqt.TensorQuantizer, name='rhs_quantizer', precision=self.rhs_prec
        ),
    )

  def __call__(self, lhs, rhs, is_eval=False):
    if not is_eval:
      self.lhs_quantizer.update(lhs)
      self.rhs_quantizer.update(rhs)

    return lambda lhs, rhs, dimension_numbers: operations.dot_general(  # pylint: disable=g-long-lambda
        lhs, rhs, self.lhs_quantizer, self.rhs_quantizer, dimension_numbers,
        is_eval)


def _generate_dimension_numbers() -> Sequence[Dict[str, Any]]:
  """Generates arbitrary dimension numbers for a tensor of shape (2, 2, 2)."""
  keys = ['testcase_name', 'dimension_numbers']
  # ((lhs_contracting_dims, rhs_contracting_dims), (lhs_batch_dims,
  # rhs_batch_dims))
  cases = [
      ('batch_matmul', (((2,), (1,)), ((0,), (0,)))),
      ('one_cont_two_batch_dims', (((2,), (2,)), ((0, 1,), (0, 1,)))),
      ('two_cont_one_batch_dims', (((1, 2), (1, 2)), ((0,), (0,)))),
      ('one_contracting_dims', (((2,), (1,)), ((), ()))),
      ('two_contracting_dims', (((1, 2), (1, 2)), ((), ()))),
  ]
  return [dict(zip(keys, vals)) for vals in cases]


class AqtDotGeneralTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(0)

  def get_dot_general_module(self, lhs, rhs, lhs_prec, rhs_prec):
    p_dot_general = pax_fiddle.Config(
        DotGeneral, name='dot_general', lhs_prec=lhs_prec, rhs_prec=rhs_prec
    )
    module = base_layer.instantiate(p_dot_general)
    state = module.init(jax.random.PRNGKey(0), lhs, rhs)
    return module.apply(state, lhs, rhs, mutable=['non_trainable'])

  def basic_quant_example(self):
    lhs = np.array(
        [
            [-7.0, 4.01, 4.01],  #
            [-7.0, 0.01, -4.01],
        ],)
    qlhs = np.array(
        [
            [-6, 4, 4],  #
            [-6, 0, -4]
        ],)

    # Representable values: -1, 0, 1
    rhs = np.array(
        [
            [-1.5, 0.99],  #
            [-0.99, 0],
            [-0.01, 1.5]
        ],)
    qrhs = np.array(
        [
            [-1, 1],  #
            [-1, 0],
            [0, 1]
        ],)

    return lhs, qlhs, rhs, qrhs

  def test_basic_dot_general(self):
    lhs, qlhs, rhs, qrhs = self.basic_quant_example()

    dot_general, _ = self.get_dot_general_module(lhs, rhs, 3, 2)
    dimension_numbers = (((1,), (0,)), ((), ()))
    actual_ret = dot_general(lhs, rhs, dimension_numbers)
    expected_ret = jax.lax.dot_general(qlhs, qrhs,
                                       dimension_numbers).astype(jnp.float32)
    self.assertArraysEqual(actual_ret, expected_ret)

  @parameterized.named_parameters(_generate_dimension_numbers())
  def test_dot_general_none(self, dimension_numbers):
    """Ensures no quantization gives aqt_dot_general=lax.dot_general."""
    lhs = np.random.uniform(-1.0, 1.0, size=(2, 2, 2)).astype(np.float32)
    rhs = np.random.uniform(-1.0, 1.0, size=(2, 2, 2)).astype(np.float32)

    dot_general, _ = self.get_dot_general_module(lhs, rhs, None, None)
    actual_ret = dot_general(lhs, rhs, dimension_numbers)
    expected_ret = jax.lax.dot_general(lhs, rhs, dimension_numbers)
    self.assertArraysEqual(actual_ret, expected_ret)

  def test_scaling_stability(self):
    lhs = np.random.uniform(-1.0, 1.0, size=(2, 3)).astype(np.float32)
    # Small values cause NaN gradients if the gradients for scaling and
    # rescaling are unnecessarily propagated.
    rhs = np.random.normal(0, 1e-23, size=(3, 4)).astype(np.float32)

    dot_general, _ = self.get_dot_general_module(lhs, rhs, None, 4)
    matmul_dimension_numbers = [[[1], [0]], [[], []]]

    @jax.grad
    def grad_qmatmul(rhs, lhs):
      return jnp.sum(dot_general(lhs, rhs, matmul_dimension_numbers))

    gradients = grad_qmatmul(rhs, lhs)
    self.assertFalse(np.isnan(gradients).any())

  def test_dot_general_high_bitwidth(self):
    lhs = np.random.normal(size=(2, 3)).astype(np.float32)
    rhs = np.random.normal(size=(3, 4)).astype(np.float32)

    precision = 23
    matmul_dimension_numbers = [[[1], [0]], [[], []]]
    dot_general, _ = self.get_dot_general_module(lhs, rhs, precision, precision)
    actual_ret = dot_general(lhs, rhs, matmul_dimension_numbers)
    expected_ret = jax.lax.dot_general(lhs, rhs, matmul_dimension_numbers)
    self.assertAllClose(actual_ret, expected_ret)


if __name__ == '__main__':
  absltest.main()
