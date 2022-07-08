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

"""Tests for Praxis pooling layers."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from lingvo.core import layers as lingvo_layers
import numpy as np
from praxis import base_layer
from praxis import test_utils
from praxis.layers import poolings
import tensorflow.compat.v2 as tf

instantiate = base_layer.instantiate
to_np = test_utils.to_np


class PoolingsTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)
    tf.random.set_seed(123)

  @parameterized.parameters(
      ((3, 3), (2, 2), 'SAME', 'MAX', [2, 4, 4, 3], False),
      ((4, 4), (1, 1), 'SAME', 'AVG', [8, 16, 32, 64], True),
      ((4, 4), (1, 1), 'SAME', 'AVG', [8, 2, 4, 64], False),
      ((8, 8), (2, 2), 'SAME', 'AVG', [8, 16, 32, 128], False),
      ((4, 4), (1, 1), 'VALID', 'AVG', [8, 16, 16, 64], True),
      ((5, 5), (2, 2), 'SAME', 'MAX', [8, 16, 32, 64], True),
      ((2, 2), (1, 1), 'VALID', 'MAX', [8, 16, 16, 64], True),
      ((2, 2), (2, 2), 'SAME', 'AVG', [4, 16, 32, 64], False),
      ((4, 4), (2, 2), 'VALID', 'MAX', [8, 16, 16, 64], False),
  )
  def test_pooling_layer(self, window_shape, window_stride, padding,
                         pooling_type, input_shape, int_inputs):
    p = poolings.Pooling.HParams(
        name='jax_pooling',
        window_shape=window_shape,
        window_stride=window_stride,
        pooling_type=pooling_type,
        padding=padding)
    pooling_layer = instantiate(p)
    if int_inputs:
      npy_inputs = np.random.randint(0, 100, input_shape).astype('int32')
    else:
      npy_inputs = np.random.normal(1.0, 0.5, input_shape).astype('float32')
    inputs = jnp.asarray(npy_inputs)
    paddings = None
    tf_paddings = None
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = pooling_layer.init(prng_key, inputs, paddings)
    output, _ = pooling_layer.apply(initial_vars, inputs, paddings)
    # Test whether tf Pooling layer returns the same output.
    # Modify initial_vars to use TF compatible params.
    tf_initial_vars = initial_vars
    tf_p = lingvo_layers.PoolingLayer.Params().Set(
        name='tf_pooling',
        window_shape=window_shape,
        window_stride=window_stride,
        pooling_type=pooling_type,
        padding_algorithm=padding)
    tf_pooling_layer = tf_p.Instantiate()
    tf_input = tf.constant(npy_inputs, dtype=tf.float32)
    tf_output = tf_pooling_layer.FProp(tf_initial_vars, tf_input, tf_paddings)
    # Check the actual output.
    np_output = to_np(output)
    tf_np_output = to_np(tf_output)
    self.assertAllClose(tf_np_output, np_output)

  @parameterized.parameters(
      ((3, 3), (1, 1), 'SAME', 'MAX', [2, 4, 4, 3], False, True),
      ((3, 3), (2, 2), 'SAME', 'MAX', [2, 4, 4, 3], False, True),
      ((4, 4), (1, 1), 'SAME', 'AVG', [8, 16, 32, 64], True, False),
      ((4, 4), (1, 1), 'SAME', 'AVG', [8, 2, 4, 64], False, True),
      ((8, 8), (2, 2), 'SAME', 'AVG', [8, 16, 32, 128], False, False),
      ((4, 4), (2, 2), 'VALID', 'AVG', [8, 16, 16, 64], True, False),
      ((5, 5), (2, 2), 'SAME', 'MAX', [8, 16, 32, 64], True, False),
      ((2, 2), (2, 2), 'VALID', 'MAX', [8, 16, 16, 64], True, False),
      ((2, 2), (2, 2), 'SAME', 'AVG', [4, 16, 32, 64], False, False),
      ((4, 4), (2, 2), 'VALID', 'MAX', [8, 16, 16, 64], False, True),
  )
  def test_pooling_layer_with_paddings(self, window_shape, window_stride,
                                       padding, pooling_type, input_shape,
                                       int_inputs, paddings_all_ones):
    p = poolings.Pooling.HParams(
        name='jax_pooling',
        window_shape=window_shape,
        window_stride=window_stride,
        pooling_type=pooling_type,
        padding=padding)
    pooling_layer = instantiate(p)
    if int_inputs:
      npy_inputs = np.random.randint(0, 100, input_shape).astype('int32')
    else:
      npy_inputs = np.random.normal(1.0, 0.5, input_shape).astype('float32')
    inputs = jnp.asarray(npy_inputs)
    paddings = None
    tf_paddings = None
    if paddings_all_ones:
      npy_paddings = np.ones([input_shape[0],
                              input_shape[1]]).astype(npy_inputs.dtype)
    else:
      npy_paddings = np.random.randint(
          0, 2, [input_shape[0], input_shape[1]]).astype(npy_inputs.dtype)
    paddings = jnp.asarray(npy_paddings)
    tf_paddings = tf.constant(npy_paddings, dtype=tf.float32)
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = pooling_layer.init(prng_key, inputs, paddings)
    output, out_paddings = pooling_layer.apply(initial_vars, inputs, paddings)
    # Test whether tf Pooling layer returns the same output.
    # Modify initial_vars to use TF compatible params.
    tf_initial_vars = initial_vars
    tf_p = lingvo_layers.PoolingLayer.Params().Set(
        name='tf_pooling',
        window_shape=window_shape,
        window_stride=window_stride,
        pooling_type=pooling_type,
        padding_algorithm=padding)
    tf_pooling_layer = tf_p.Instantiate()
    tf_input = tf.constant(npy_inputs, dtype=tf.float32)
    tf_output = tf_pooling_layer.FProp(tf_initial_vars, tf_input, tf_paddings)
    # Check the actual output.
    np_output = to_np(output)
    tf_np_output = to_np(tf_output[0])
    np_paddings = to_np(out_paddings)
    tf_np_paddings = to_np(tf_output[1])
    # Check the paddings.
    self.assertAllClose(np_paddings, tf_np_paddings)
    self.assertAllClose(tf_np_output, np_output)

  @parameterized.parameters(
      ('MAX', True, False, [5., 3.]),
      ('MAX', True, True, [5, 3]),
      ('MAX', False, False, [5., 5.]),
      ('MAX', False, True, [5, 5]),
      ('AVG', True, False, [4., 2.]),
      ('AVG', True, True, [4, 2]),
      ('AVG', False, False, [3., 3.]),
      ('AVG', False, True, [3, 3]),
  )
  def test_global_pooling_layer(self, pooling_type, apply_padding, int_inputs,
                                ground_truth):
    p = poolings.GlobalPooling.HParams(
        name='global_pooling',
        pooling_type=pooling_type,
        pooling_dims=[1],
        keepdims=False)
    pooling_layer = instantiate(p)
    if int_inputs:
      inputs = jnp.asarray([[1, 2, 3, 4, 5],
                            [5, 4, 3, 2, 1]])
    else:
      inputs = jnp.asarray([[1.0, 2.0, 3.0, 4.0, 5.0],
                            [5.0, 4.0, 3.0, 2.0, 1.0]])
    paddings = None
    if apply_padding:
      paddings = jnp.asarray([[1.0, 1.0, 0.0, 0.0, 0.0],
                              [1.0, 1.0, 0.0, 0.0, 0.0]])

    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = pooling_layer.init(
        prng_key, inputs, compatible_paddings=paddings)
    output = pooling_layer.apply(
        initial_vars, inputs, compatible_paddings=paddings)

    self.assertAllClose(output, ground_truth)

if __name__ == '__main__':
  absltest.main()
