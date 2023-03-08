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

"""Tests for utilities."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from praxis import test_utils
from praxis.layers.quantization import utils


class UtilsTest(test_utils.TestCase):

  @parameterized.parameters(
      dict(eqn='ANH,NHD->AD', lhs_shape=(2, 3, 4), rhs_shape=(3, 4, 2)),
      dict(eqn='ANH,DNH->AD', lhs_shape=(2, 3, 4), rhs_shape=(2, 3, 4)),
      dict(eqn='AD,DNH->ANH', lhs_shape=(2, 3), rhs_shape=(3, 4, 2)),
      dict(eqn='AD,KDNH->KANH', lhs_shape=(2, 3), rhs_shape=(2, 3, 4, 2)),
      dict(
          eqn='BTNH,BSNH->BTNS',
          lhs_shape=(1, 2, 3, 4),
          rhs_shape=(1, 5, 3, 4),
      ),
      dict(
          eqn='BTNH,HNBS->STNB',
          lhs_shape=(1, 2, 3, 4),
          rhs_shape=(4, 3, 1, 5),
      ),
  )
  def test_einsum_equation_conversion(self, eqn, lhs_shape, rhs_shape):
    """Validate that lax.dot_general produces the same output as jnp.einsum."""
    lhs = jnp.arange(np.prod(lhs_shape)).reshape(lhs_shape)
    rhs = jnp.arange(np.prod(rhs_shape)).reshape(rhs_shape)

    einsum_result = jnp.einsum(eqn, lhs, rhs)
    dimension_numbers, perm = utils.einsum_eqn_to_dimension_numbers(eqn)
    dot_general_result = jax.lax.dot_general(
        lhs, rhs, dimension_numbers=dimension_numbers)
    if perm is not None:
      dot_general_result = jax.lax.transpose(dot_general_result, perm)
    self.assertAllClose(einsum_result, dot_general_result)

  @parameterized.parameters(
      dict(eqn='...AB,BC->...AC', error=NotImplementedError, regex=r'\.\.\.'),
      dict(eqn='AB->BA', error=ValueError, regex=r'arguments'),
      dict(eqn='ABB,BC->AC', error=ValueError, regex=r'Repeated'),
      dict(eqn='ABC,AB->AB', error=ValueError, regex=r'Contraction.*C'),
  )
  def test_unsupported_einsum_equations(self, eqn, error, regex):
    with self.assertRaisesRegex(error, regex):
      utils.einsum_eqn_to_dimension_numbers(eqn)

  @parameterized.parameters(
      dict(vals=[range(-8, 8)], shape=(8, 2), dtype=jnp.int8, pack_dim=0),
      dict(
          vals=[range(-8, 8)] * 2, shape=(8, 2, 2), dtype=jnp.int8, pack_dim=0
      ),
      dict(
          vals=[range(-8, 8)] * 2, shape=(2, 8, 2), dtype=jnp.int8, pack_dim=1
      ),
      dict(
          vals=[range(7, -9, -1)] * 2,
          shape=(2, 8, 2),
          dtype=jnp.int8,
          pack_dim=1,
      ),
      dict(
          vals=[range(0, 16)] * 2, shape=(8, 2, 2), dtype=jnp.uint8, pack_dim=0
      ),
      dict(
          vals=[range(0, 16)] * 2, shape=(2, 8, 2), dtype=jnp.uint8, pack_dim=1
      ),
  )
  def test_pack_4bit_unpack_4bit(self, vals, shape, dtype, pack_dim):
    x = jnp.array(vals).reshape(shape).astype(dtype)
    packed = utils.pack_4bit(x, pack_dim)
    expected_packed_shape = list(x.shape)
    expected_packed_shape[pack_dim] //= 8
    self.assertSequenceEqual(packed.shape, expected_packed_shape)

    unpacked = utils.unpack_4bit(packed, pack_dim, x.dtype)
    self.assertArraysEqual(unpacked, x.astype(jnp.int32))

  def test_get_packed_shape(self):
    self.assertSequenceEqual(utils.get_packed_shape((4, 8, 3), 1, 8), (4, 1, 3))
    self.assertRaises(ValueError, utils.get_packed_shape, (4, 7, 3), 1, 8)


if __name__ == '__main__':
  absltest.main()
