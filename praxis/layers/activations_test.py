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

"""Tests for activations."""

from absl.testing import absltest
import jax
from jax import numpy as jnp
from praxis import base_layer
from praxis import test_utils
from praxis.layers import activations


class ActivationsTest(test_utils.TestCase):

  def _run(self, p, inputs, outputs):
    self.assertEqual(outputs, base_layer.instantiate(p.set(name='n'))(inputs))

  def test_Exp(self):
    inputs = jnp.array([1.])
    self._run(activations.Exp.HParams(), inputs, jnp.exp(inputs))

  def test_Softplus(self):
    inputs = jnp.array([1.])
    self._run(activations.Softplus.HParams(), inputs, jax.nn.softplus(inputs))


if __name__ == '__main__':
  absltest.main()
