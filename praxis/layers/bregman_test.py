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

"""Tests for Praxis Bregman PCA layer."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import py_utils
from praxis import test_utils
from praxis.layers import bregman

instantiate = base_layer.instantiate
to_np = test_utils.to_np
NON_TRAINABLE = base_layer.NON_TRAINABLE


class BregmanTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  @parameterized.parameters(
      ('IDENTITY', 0.99, 0.0, 0.0, True),
      ('IDENTITY', 1.0, 0.01, 0.0, True),
      ('IDENTITY', 1.0, 0.01, 0.01, False),
      ('IDENTITY', 0.99, 0.01, 0.01, True),
      ('LEAKY_RELU', 0.99, 0.0, 0.0, False),
      ('LEAKY_RELU', 1.0, 0.01, 0.0, True),
      ('LEAKY_RELU', 1.0, 0.01, 0.01, False),
      ('LEAKY_RELU', 0.99, 0.01, 0.01, True),
      ('SOFTMAX', 0.99, 0.0, 0.0, True),
      ('SOFTMAX', 1.0, 0.01, 0.0, True),
      ('SOFTMAX', 1.0, 0.01, 0.01, False),
      ('SOFTMAX', 0.99, 0.01, 0.01, False),
  )
  def test_bregman_layer(self, activation, mean_beta, coefficients_lr,
                         components_lr, constant_lr_schedule):
    """Tests layer construction and the expected outputs."""
    activation_type = getattr(bregman.ActivationType, activation)
    p = bregman.BregmanPCA.HParams(
        name='bregman_pca',
        num_components=3,
        input_dims=[8, 10],
        activation_type=activation_type,
        negative_slope=0.1,
        mean_beta=mean_beta,
        coefficients_lr=coefficients_lr,
        coefficients_beta=0.9,
        coefficients_steps=20,
        components_lr=components_lr,
        components_beta=0.9,
        start_step=0,
        end_step=1,
        constant_lr_schedule=constant_lr_schedule)
    layer = instantiate(p)
    if activation == 'SOFTMAX':
      npy_input = np.random.random([16] + p.input_dims).astype('float32')
      npy_input = npy_input / np.sum(npy_input, axis=-1, keepdims=True)
    else:
      npy_input = np.random.normal(1.0, 0.5,
                                   [16] + p.input_dims).astype('float32')
    inputs = jnp.asarray(npy_input)
    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = layer.init(prng_key, inputs)

    @jax.jit
    def comp(theta, inputs):
      with base_layer.JaxContext.new_context():
        return layer.apply(theta, inputs, mutable=[NON_TRAINABLE])

    (outputs, coefficients), updated_vars = comp(initial_vars, inputs)
    self.assertAllClose(outputs, inputs, atol=1e-5)

    with base_layer.JaxContext.new_context():
      layer = layer.bind(initial_vars, mutable=[NON_TRAINABLE])
    initial_vars = py_utils.NestedMap.FromNestedDict(
        initial_vars['non_trainable'])
    init_components = initial_vars.components
    init_mean = initial_vars.mean
    mean = updated_vars[NON_TRAINABLE]['mean']
    components = updated_vars[NON_TRAINABLE]['components']
    init_loss = layer.bregman_loss_fn(
        jnp.zeros_like(coefficients), init_components, init_mean, inputs)
    final_loss = layer.bregman_loss_fn(coefficients, components, mean, inputs)
    self.assertLess(final_loss, init_loss)

    representations = layer.reconstruct(coefficients)
    self.assertEqual(representations.shape, inputs.shape)

  def test_pca_convergence(self):
    """Tests whether the gradients are zero at the solution."""
    p = bregman.BregmanPCA.HParams(
        name='bregman_pca',
        num_components=3,
        input_dims=[3],
        activation_type=bregman.ActivationType.IDENTITY,
        start_step=0,
        end_step=1)
    layer = instantiate(p)
    npy_input = np.random.normal(1.0, 0.5,
                                 [16] + p.input_dims).astype('float32')
    inputs = jnp.asarray(npy_input)

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = layer.init(prng_key, inputs)
      layer = layer.bind(initial_vars, mutable=[NON_TRAINABLE])
    mean = jnp.zeros((1, 3))
    components = jnp.eye(3)
    coefficients_grad = layer.coefficients_grad_fn(inputs, components, mean,
                                                   inputs)
    components_grad = layer.components_grad_fn(inputs, components, mean, inputs)
    self.assertAllClose(
        coefficients_grad, jnp.zeros_like(coefficients_grad), atol=1e-5)
    self.assertAllClose(
        components_grad, jnp.zeros_like(components_grad), atol=1e-5)


if __name__ == '__main__':
  absltest.main()
