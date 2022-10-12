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

from absl.testing import absltest
from absl.testing import parameterized
from jax import numpy as jnp
import numpy as np
from praxis import test_utils
from praxis.layers.quantization import operations
from typing import List



class QuantizationUtilsTest(test_utils.TestCase):

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


if __name__ == '__main__':
  absltest.main()
