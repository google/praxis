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

"""Tests for quantization optimizations."""

from absl.testing import absltest
from jax import numpy as jnp
import numpy as np
from praxis import test_utils
from praxis.layers.quantization import optimization


class OptimizationTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  def test_optimization_int8(self):
    t = jnp.array([[1.0, 2.0, 3.0], [4.0, 1.0, 2.0]])
    bound = jnp.array([[3.0], [4.0]])
    ret = optimization.get_best_bound(t, bound, -128.0, 127.0)
    expected = jnp.array([[3.0], [4.0]])
    self.assertArraysEqual(ret, expected)

  def test_optimization_int4(self):
    t = jnp.array([[1.0, 2.0, 3.0], [4.0, 1.0, 2.0]])
    bound = jnp.array([[3.0], [4.0]])
    ret = optimization.get_best_bound(t, bound, -8.0, 7.0)
    expected = jnp.array([[3.0], [4.0]])
    self.assertArraysEqual(ret, expected)


if __name__ == '__main__':
  absltest.main()
