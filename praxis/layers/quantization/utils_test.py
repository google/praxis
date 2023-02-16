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


if __name__ == '__main__':
  absltest.main()
