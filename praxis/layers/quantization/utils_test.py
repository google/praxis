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
from praxis import test_utils
from praxis.layers.quantization import utils


class UtilsTest(test_utils.TestCase):

  @parameterized.parameters(
      dict(eqn='ANH,NHD->AD', lhs_shape=(2, 3, 4), rhs_shape=(3, 4, 2)),
      dict(eqn='ANH,DNH->AD', lhs_shape=(2, 3, 4), rhs_shape=(2, 3, 4)),
      dict(eqn='AD,DNH->ANH', lhs_shape=(2, 3), rhs_shape=(3, 4, 2)),
      dict(eqn='AD,KDNH->KANH', lhs_shape=(2, 3), rhs_shape=(2, 3, 4, 2)),
  )
  def test_einsum_equation_conversion(self, eqn, lhs_shape, rhs_shape):
    """Given an einsum equations, ensures lax.dot_general with its converted dimension numbers produces almost the same output as jnp.einsum."""
    key = jax.random.PRNGKey(seed=123)
    lhs = jax.random.uniform(key, shape=lhs_shape)
    rhs = jax.random.uniform(key, shape=rhs_shape)

    einsum_result = jnp.einsum(eqn, lhs, rhs)
    dimension_numbers, perm = utils.convert_einsum_eqn_to_dimension_numbers(
        eqn)
    dot_general_result = jax.lax.dot_general(
        lhs, rhs, dimension_numbers=dimension_numbers)
    if perm is not None:
      dot_general_result = jax.lax.transpose(dot_general_result, perm)
    self.assertAllClose(einsum_result, dot_general_result)


if __name__ == '__main__':
  absltest.main()
