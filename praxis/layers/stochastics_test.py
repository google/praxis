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

"""Tests for Praxis stochastic layers."""

from absl import logging
from absl.testing import absltest
import jax
from jax import numpy as jnp
from praxis import base_layer
from praxis import test_utils
from praxis.layers import stochastics

instantiate = base_layer.instantiate


class StochaticsTest(test_utils.TestCase):

  def test_dropout_layer01(self):
    test_layer_p = stochastics.Dropout.HParams(name='dropout', keep_prob=0.8)
    layer = instantiate(test_layer_p)

    inputs = jnp.ones([10, 1000], dtype=jnp.bfloat16)

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=12346)
      prng_key, init_key = jax.random.split(prng_key)
      prng_key, dropout_k1, dropout_k2 = jax.random.split(prng_key, 3)
      initial_vars = layer.init({
          'random': dropout_k1,
          'params': init_key
      }, inputs)
      logging.info('initial_vars: %s', initial_vars)
      output1 = layer.apply(initial_vars, inputs, rngs={'random': dropout_k1})
      output2 = layer.apply(initial_vars, inputs, rngs={'random': dropout_k2})

    out1_sum = jnp.sum(output1)
    out2_sum = jnp.sum(output2)
    out1_nonzero = jnp.sum(output1 > 0.0)
    out2_nonzero = jnp.sum(output2 > 0.0)

    logging.info('out1_sum: %s', out1_sum)
    logging.info('out2_sum: %s', out2_sum)
    logging.info('out1_nonzero: %s', out1_nonzero)
    logging.info('out2_nonzero: %s', out2_nonzero)

    self.assertEqual(9984.0, out1_sum)
    self.assertEqual(9920.0, out2_sum)
    self.assertEqual(8000.0, out1_nonzero)
    self.assertEqual(7952.0, out2_nonzero)

  def test_dropout_layer_02(self):
    test_layer_p = stochastics.Dropout.HParams(
        name='dropout',
        keep_prob=0.8,
        noise_shape=[10, 6, 8],
        noise_shape_broadcast_dims=[2])
    layer = instantiate(test_layer_p)

    inputs = jnp.ones([2, 10, 6, 8], dtype=jnp.bfloat16)

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=12346)
      prng_key, init_key = jax.random.split(prng_key)
      prng_key, compute_key = jax.random.split(prng_key)
      initial_vars = layer.init({
          'random': compute_key,
          'params': init_key
      }, inputs)
      logging.info('initial_vars: %s', initial_vars)
      output1 = layer.apply(initial_vars, inputs, rngs={'random': compute_key})

    out1_sum = jnp.sum(output1)
    out1_nonzero = jnp.sum(output1 > 0.0)

    logging.info('out1_sum: %s', out1_sum)
    logging.info('out1_nonzero: %s', out1_nonzero)

    self.assertEqual(980, out1_sum)
    self.assertEqual(784, out1_nonzero)

  def test_dropout_layer_03(self):
    test_layer_p = stochastics.Dropout.HParams(
        name='dropout', keep_prob=0.8, noise_shape_broadcast_dims=[0, 3])
    layer = instantiate(test_layer_p)

    inputs = jnp.ones([2, 10, 6, 8], dtype=jnp.bfloat16)

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=12346)
      prng_key, init_key = jax.random.split(prng_key)
      prng_key, compute_key = jax.random.split(prng_key)
      initial_vars = layer.init({
          'random': compute_key,
          'params': init_key
      }, inputs)
      logging.info('initial_vars: %s', initial_vars)

      output1 = layer.apply(initial_vars, inputs, rngs={'random': compute_key})

    out1_sum = jnp.sum(output1)
    out1_nonzero = jnp.sum(output1 > 0.0)

    logging.info('out1_sum: %s', out1_sum)
    logging.info('out1_nonzero: %s', out1_nonzero)

    self.assertEqual(980, out1_sum)
    self.assertEqual(784, out1_nonzero)


if __name__ == '__main__':
  absltest.main()
