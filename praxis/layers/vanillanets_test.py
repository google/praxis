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

"""Tests for vanillanets."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from praxis import base_layer
from praxis.layers import poolings
from praxis.layers import vanillanets

instantiate = base_layer.instantiate


class VanillanetsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  @parameterized.parameters(
      (3, 1, 'RELU', [2, 16, 36, 72], 72),
      (4, 2, 'TANH', [4, 12, 12, 16], 32),
      (4, 2, 'RELU', [4, 12, 12, 16], 8),
      (5, 1, 'NONE', [4, 12, 12, 16], 8),
  )
  def test_vanilla_block(self, kernel_size, stride, activation, input_shape,
                         output_dim):
    input_dim = input_shape[-1]
    p = vanillanets.VanillaBlock.HParams(
        name='vanilla_block',
        input_dim=input_dim,
        output_dim=output_dim,
        kernel_size=kernel_size,
        stride=stride)
    resnet_layer = instantiate(p)
    npy_inputs = np.random.normal(1.0, 0.5, input_shape).astype('float32')
    inputs = jnp.asarray(npy_inputs)

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = resnet_layer.init(prng_key, inputs)
      output = resnet_layer.apply(initial_vars, inputs)

  @parameterized.parameters(
      ([1, 4, 4, 3], None),
      ([1, 4, 4, 3], [1, 2]),
  )
  def test_vanilla_net(self, input_shape, spatial_pooling_dims):
    p = vanillanets.VanillaNet.HParamsVanillaNet5().set(
        name='vanillanet', output_spatial_pooling_params=spatial_pooling_dims)
    if spatial_pooling_dims is not None:
      p.output_spatial_pooling_params = poolings.GlobalPooling.HParams(
          pooling_dims=spatial_pooling_dims)
    vanillanet_layer = instantiate(p)
    npy_inputs = np.random.normal(1.0, 0.5, input_shape).astype('float32')
    inputs = jnp.asarray(npy_inputs)

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = vanillanet_layer.init(prng_key, inputs)
      output = vanillanet_layer.apply(initial_vars, inputs)


if __name__ == '__main__':
  absltest.main()
