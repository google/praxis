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

"""Tests for adapters."""

from absl.testing import absltest
import jax
from jax import numpy as jnp
from lingvo.core import cluster_factory
from lingvo.core import layers
import numpy as np
from praxis import base_layer
from praxis import py_utils
from praxis import test_utils
from praxis.layers import adapters
from praxis.layers import normalizations
import tensorflow.compat.v2 as tf

instantiate = base_layer.instantiate


class AdaptersTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)
    tf.random.set_seed(123)

  def test_residual_adapter_tf_equivalent(self):
    input_dims = 10
    bottleneck_dim = 2
    num_tasks = 8
    seq_len = 5
    batch_size = 2

    layer = adapters.MultitaskResidualAdapter.HParams(
        name='adapter',
        input_dims=input_dims,
        bottleneck_dims=bottleneck_dim,
        num_tasks=num_tasks)
    layer = instantiate(layer)

    npy_inputs = np.random.normal(
        1.0, 0.5, [batch_size, seq_len, input_dims]).astype('float32')
    inputs = jnp.asarray(npy_inputs)
    tasks = np.random.randint(
        0, num_tasks, size=[
            batch_size,
        ])

    context_p = base_layer.JaxContext.HParams(do_eval=True)

    with base_layer.JaxContext.new_context(hparams=context_p):
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = layer.init(prng_key, inputs, tasks=tasks)
      output = layer.apply(initial_vars, inputs, tasks=tasks)

    tf_p = layers.MultitaskAdapterEinsumLayer.Params().Set(
        name='tf_adapter',
        input_dim=input_dims,
        bottleneck_dim=bottleneck_dim,
        num_tasks=num_tasks)

    tf_adapter = tf_p.Instantiate()

    theta = initial_vars['params'].copy()
    theta['layer_norm'] = theta['norm']
    del theta['norm']
    theta = jax.tree_map(np.array, theta)
    theta = py_utils.NestedMap.FromNestedDict(theta)

    with cluster_factory.SetEval(True):
      tf_output = tf_adapter.FProp(theta, inputs, tf.convert_to_tensor(tasks))
    np_output = test_utils.to_np(output)
    tf_np_output = test_utils.to_np(tf_output)
    self.assertAllClose(tf_np_output, np_output, atol=1e-5)

  def test_residual_adapter_run01(self):
    """This tests batch normalization with paddings with per-frame task id."""
    input_dims = 10
    bottleneck_dim = 2
    num_tasks = 8
    seq_len = 5
    batch_size = 2

    layer = adapters.MultitaskResidualAdapter.HParams(
        name='adapter',
        input_dims=input_dims,
        bottleneck_dims=bottleneck_dim,
        num_tasks=num_tasks,
        norm_tpl=normalizations.BatchNorm.HParams())
    layer = instantiate(layer)

    npy_inputs = np.random.normal(
        1.0, 0.5, [batch_size, seq_len, input_dims]).astype('float32')
    inputs = jnp.asarray(npy_inputs)
    tasks = np.random.randint(0, num_tasks, size=[batch_size, seq_len])
    paddings = np.random.randint(0, 2, size=[batch_size, seq_len])

    context_p = base_layer.JaxContext.HParams(do_eval=True)

    with base_layer.JaxContext.new_context(hparams=context_p):
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = layer.init(
          prng_key, inputs, paddings=paddings, tasks=tasks)
      output = layer.apply(initial_vars, inputs, paddings=paddings, tasks=tasks)
    self.assertEqual(output.shape, inputs.shape)

  def test_residual_adapter_run02(self):
    """This tests batch normalization with paddings without task id."""
    input_dims = 10
    bottleneck_dim = 2
    num_tasks = 1
    seq_len = 5
    batch_size = 2

    layer = adapters.MultitaskResidualAdapter.HParams(
        name='adapter',
        input_dims=input_dims,
        bottleneck_dims=bottleneck_dim,
        num_tasks=num_tasks,
        norm_tpl=normalizations.BatchNorm.HParams())
    layer = instantiate(layer)

    npy_inputs = np.random.normal(
        1.0, 0.5, [batch_size, seq_len, input_dims]).astype('float32')
    inputs = jnp.asarray(npy_inputs)
    paddings = np.random.randint(0, 2, size=[batch_size, seq_len])

    context_p = base_layer.JaxContext.HParams(do_eval=True)

    with base_layer.JaxContext.new_context(hparams=context_p):
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = layer.init(prng_key, inputs, paddings=paddings)
      output = layer.apply(initial_vars, inputs, paddings=paddings)
    self.assertEqual(output.shape, inputs.shape)


if __name__ == '__main__':
  absltest.main()
