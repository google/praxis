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

"""Tests for quantized operations."""

from typing import Dict, Sequence

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import pax_fiddle
from praxis import test_utils
from praxis.layers.quantization import operations
from praxis.layers.quantization import quantizer


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

  @parameterized.parameters(jnp.int8, jnp.uint8)
  def test_int_einsum(self, dtype):
    # pylint: disable=invalid-name
    A, D, H = 6, 5, 2

    x = jnp.ones([A, D], dtype=dtype)
    w = jnp.ones([D, H], dtype=dtype)
    s = jnp.ones([H], dtype=dtype)

    ret = operations.einsum('AD,DH->AH', x, w, s)
    expected = jnp.ones([A, H], dtype=jnp.int32) * D
    self.assertArraysEqual(ret, expected)

  @parameterized.named_parameters(
      ('eqn_with_dot', '...y,yz->...z'),
  )
  def test_quantized_einsum_with_zp(self, eqn):
    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 1.0, 2.0]], dtype=jnp.bfloat16)
    w = jnp.array([[1, 2, 1], [2, 1, 2], [1, 3, 1]], dtype=jnp.int8)
    s = jnp.array([0.1, 0.2, 0.3], dtype=jnp.bfloat16)
    zp = jnp.array([-0.5, 3.2, 2.7])

    ret = operations.einsum(eqn, x, w, s, zp)
    expected = jnp.array(
        [[3.800781, -16.590626, -13.793751], [4.300781, -19.4, -16.49375]],
        dtype=jnp.float32,
    )
    self.assertAllClose(ret, expected, rtol=0.02, atol=0.02)

  def test_min_max(self):
    self.assertEqual(operations.get_min_max(8), (-128, 127))
    self.assertEqual(operations.get_min_max(8, True), (0, 255))


class ReducePrecisionEinsumTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(1234567)

  @parameterized.named_parameters(
      ('eqn1', 'ab,bc->ac', (4, 3), (3,), ()),
      ('eqn2', '...y,yz->...z', (6, 5), (5,), ()),
      ('eqn3', 'ABD,KDNH->KABNH', (2, 3, 4, 5), (2, 4, 5), (1)),
  )
  def test_reduce_einsum_weight_precision(self, eqn, w_shape,
                                          expected_scale_shape, expand_dims):

    weight = np.random.normal(1.5, 2.0, w_shape).astype(np.float32)
    reduced_weight, scale, _ = operations.reduce_einsum_weight_precision(
        eqn, weight)
    self.assertEqual(scale.shape, expected_scale_shape)
    if expand_dims:
      scale = jnp.expand_dims(scale, expand_dims)
    self.assertAllClose(
        weight,
        jnp.multiply(reduced_weight, scale).astype(jnp.float32),
        rtol=0.02,
        atol=0.02,
    )
    for use_symmetric in [True, False]:
      weight_nudged = operations.fakequant_einsum(
          eqn, weight, use_symmetric=use_symmetric
      )
      self.assertAllClose(weight, weight_nudged, rtol=0.02, atol=0.02)

  def test_percentile(self):
    weight = np.random.normal(-2.0, 2.0, (4, 3)).astype(np.float32)
    reduced_weight, scale, _ = operations.reduce_einsum_weight_precision(
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


def _generate_einsum_eqn() -> Sequence[Dict[str, str]]:
  """Generates arbitrary dimension numbers for a tensor of shape (2, 2, 2)."""
  keys = ['testcase_name', 'eqn']
  # ((lhs_contracting_dims, rhs_contracting_dims), (lhs_batch_dims,
  # rhs_batch_dims))
  cases = [
      ('batch_matmul', 'abc,acd->abd'),
      ('one_cont_two_batch_dims', 'abc,abc->ab'),
      ('two_cont_one_batch_dims', 'abc,abc->a'),
      ('one_contracting_dims', 'abc,dce->abde'),
      ('two_contracting_dims', 'abc,dbc->ad'),
  ]
  return [dict(zip(keys, vals)) for vals in cases]


class AqtEinsum(base_layer.BaseLayer):
  lhs_prec = None
  rhs_prec: int = 8
  add_scale_eps: bool = False
  use_symmetric: bool = True

  def setup(self):
    self.create_child(
        'lhs_quantizer',
        pax_fiddle.Config(
            quantizer.TensorQuantizer,
            name='lhs_quantizer',
            precision=self.lhs_prec,
        ),
    )
    self.create_child(
        'rhs_quantizer',
        pax_fiddle.Config(
            quantizer.TensorQuantizer,
            name='rhs_quantizer',
            precision=self.rhs_prec,
            add_scale_eps=self.add_scale_eps,
            use_symmetric=self.use_symmetric,
        ),
    )

  def __call__(self):
    def aqt_einsum(eqn, lhs, rhs):
      return operations.aqt_einsum(
          eqn,
          lhs,
          rhs,
          lhs_quantizer=self.lhs_quantizer,
          rhs_quantizer=self.rhs_quantizer,
      )

    return aqt_einsum


class AqtEinsumTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(0)

  def get_aqt_einsum_module(
      self, rhs_prec, add_scale_eps=False, use_symmetric=True
  ):
    p_aqt_einsum = pax_fiddle.Config(
        AqtEinsum,
        name='aqt_einsum',
        rhs_prec=rhs_prec,
        add_scale_eps=add_scale_eps,
        use_symmetric=use_symmetric,
    )
    module = base_layer.instantiate(p_aqt_einsum)
    state = module.init(jax.random.PRNGKey(0))
    return module.apply(state, mutable=['non_trainable'])

  def basic_quant_example(self):
    lhs = np.array(
        [
            [-7.0, 4.01, 4.01],  #
            [-7.0, 0.01, -4.01],
        ],)
    rhs = np.array(
        [
            [-1.5, 0.99],  #
            [-0.99, 0],
            [-0.01, 1.5]
        ],)
    q_deq_rhs = np.array(
        [
            [-1.5, 1.5],  #
            [-1.5, 0],
            [0, 1.5]
        ],)

    return lhs, rhs, q_deq_rhs

  def test_basic_aqt_einsum(self):
    lhs, rhs, q_deq_rhs = self.basic_quant_example()

    aqt_einsum, _ = self.get_aqt_einsum_module(rhs_prec=2)
    eqn = 'xy,yz->xz'
    actual_ret = aqt_einsum(eqn, lhs, rhs)
    expected_ret = jnp.einsum(eqn, lhs, q_deq_rhs)
    self.assertArraysEqual(actual_ret, expected_ret)

  @parameterized.named_parameters(_generate_einsum_eqn())
  def test_aqt_einsum_noquant(self, eqn):
    lhs = np.random.uniform(-1.0, 1.0, size=(2, 2, 2)).astype(np.float32)
    rhs = np.random.uniform(-1.0, 1.0, size=(2, 2, 2)).astype(np.float32)

    aqt_einsum, _ = self.get_aqt_einsum_module(rhs_prec=None)
    actual_ret = aqt_einsum(eqn, lhs, rhs)
    expected_ret = jnp.einsum(eqn, lhs, rhs)
    self.assertArraysEqual(actual_ret, expected_ret)


class SubChannelTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(0)

  def test_last_dim_subchannel_reshape(self):
    inputs = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    inputs_shape = inputs.shape

    # Get the new tensor size with specified number of subchannels.
    new_inputs_shape = operations.compute_shape_with_subchannels(
        sub_channels=4, inputs_shape=inputs_shape, contract_dims=[1]
    )
    self.assertArraysEqual(new_inputs_shape, [4, 2])

    # Split tensor into subchannels.
    sub_channel = jnp.reshape(inputs, new_inputs_shape)
    expected_subchannel = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])
    self.assertArraysEqual(sub_channel, expected_subchannel)

    # Here quantization could be applied.

    # Reshape it back to original order.
    outputs = jnp.reshape(sub_channel, inputs_shape)
    self.assertArraysEqual(outputs, inputs)

  def test_middle_dim_subchannel_reshape(self):
    inputs = np.array([[
        [0],  #
        [1],  #
        [2],  #
        [3],  #
        [4],  #
        [5],  #
        [6],  #
        [7]]])
    inputs_shape = inputs.shape

    # Get the new tensor size with specified number of subchannels.
    new_inputs_shape = operations.compute_shape_with_subchannels(
        sub_channels=4, inputs_shape=inputs_shape, contract_dims=[1],
        min_sub_channel_size=1
    )
    self.assertArraysEqual(new_inputs_shape, [4, 2, 1])

    # Split tensor into subchannels.
    sub_channel = jnp.reshape(inputs, new_inputs_shape)
    expected_subchannel = np.array([
        [[0],  #
         [1]],  #
        [[2],  #
         [3]],  #
        [[4],  #
         [5]],  #
        [[6],  #
         [7]]])
    self.assertArraysEqual(sub_channel, expected_subchannel)

    # Reshape it back to original order.
    outputs = jnp.reshape(sub_channel, inputs_shape)
    self.assertArraysEqual(outputs, inputs)

  @parameterized.parameters(2, 4, 8)
  def test_subchannel_shape(self, sub_channels):
    feature0, feature1, feature2, feature3 = (3, 3, 4, 16)
    new_shape = operations.compute_shape_with_subchannels(
        sub_channels=sub_channels,
        inputs_shape=[feature0, feature1, feature2, feature3],
        contract_dims=[2, 3],
        min_sub_channel_size=1,
    )

    expected_shape = [
        feature0 * sub_channels,
        feature1,
        feature2,
        feature3 // sub_channels,
    ]
    self.assertArraysEqual(new_shape, expected_shape)


if __name__ == '__main__':
  absltest.main()
