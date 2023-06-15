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

"""Tests for Praxis Einsum layers."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from praxis import base_layer
from praxis import pax_fiddle
from praxis import test_utils
from praxis.layers import einsum

instantiate = base_layer.instantiate


class EinsumTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  @parameterized.parameters(
      ('...d,df->...f', [3, 4, 5], [5, 8], False),
      ('...d,dnh->...nh', [3, 5], [5, 2, 8], True),
      ('blnh,dnh->bld', [3, 4, 2, 8], [5, 2, 8], False),
      ('...nh,hdn->...d', [3, 4, 2, 8], [8, 5, 2], True),
  )
  def test_einsum(self, eqn, in_shape, w_shape, use_bias):
    p = pax_fiddle.Config(
        einsum.Einsum,
        name='einsum',
        eqn=eqn,
        w_shape=w_shape,
        use_bias=use_bias,
    )
    layer = instantiate(p)
    inputs = np.random.normal(1.0, 0.5, in_shape).astype(
        'float32'
    )
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = layer.init(prng_key, inputs)
    if use_bias:
      # Make sure the bias is non-zero.
      initial_vars['params']['b'] = np.random.normal(
          1.0, 0.5, initial_vars['params']['b'].shape
      ).astype('float32')
    outputs = layer.apply(initial_vars, inputs)
    np_outputs = np.einsum(eqn, inputs, initial_vars['params']['w'])
    if use_bias:
      np_outputs += initial_vars['params']['b']
    self.assertAllClose(outputs, np_outputs, atol=1e-6)


if __name__ == '__main__':
  absltest.main()
