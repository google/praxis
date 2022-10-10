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

"""Tests for Praxis convolutional layers."""

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from lingvo.core import cluster_factory
from lingvo.core import conformer_layer
from lingvo.core import conv_layers_with_time_padding as clwp
import numpy as np
from praxis import base_layer
from praxis import py_utils
from praxis import test_utils
from praxis.layers import convolutions
import tensorflow.compat.v2 as tf

to_np = test_utils.to_np
instantiate = base_layer.instantiate

PARAMS = base_layer.PARAMS
NON_TRAINABLE = base_layer.NON_TRAINABLE


class ConvolutionsTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  @parameterized.parameters(
      ((5, 4, 24, 36), (1, 1), [2, 16, 36, 72]),
      ((2, 4, 16, 8), (2, 2), [2, 16, 32, 128]),
      ((4, 8, 16, 32), (1, 1), [2, 16, 32, 64]),
  )
  def test_conv2d_layer_same_padding(self, filter_shape, filter_stride,
                                     input_shape):
    p = convolutions.Conv2D.HParams(
        name='jax_conv2d',
        filter_shape=filter_shape,
        filter_stride=filter_stride,
        dilations=(1, 1),
        padding='SAME')
    conv_layer = instantiate(p)
    npy_inputs = np.random.normal(1.0, 0.5, input_shape).astype('float32')
    inputs = jnp.asarray(npy_inputs)

    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = conv_layer.init(prng_key, inputs)

    output = conv_layer.apply(initial_vars, inputs)
    # Test whether output has same shape as input in height and width
    for i in [1, 2]:
      self.assertEqual(output.shape[i], inputs.shape[i] // filter_stride[i - 1])

  def test_causal_conv2d_layer(self):
    p = convolutions.Conv2D.HParams(
        name='jax_conv2d',
        filter_shape=[3, 3, 1, 1],
        filter_stride=[2, 2],
        dilations=[2, 2],
        padding='SAME',
        is_causal=True,
        tf_equivalent_padding=True)
    conv_layer = instantiate(p)
    npy_inputs = np.arange(25).reshape((1, 5, 5, 1)).astype('float32')
    inputs = jnp.asarray(npy_inputs)

    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = conv_layer.init(prng_key, inputs)
    initial_vars = jax.tree_map(jnp.ones_like, initial_vars)

    output = conv_layer.apply(initial_vars, inputs)
    np_output = np.array([[2., 6., 6.], [24., 42., 32.], [66., 108., 78.]])
    self.assertAllClose(to_np(output[0, :, :, 0]), np_output)

  @parameterized.parameters(
      (2, 10, 3, 10, 1, True),
      (3, 12, 5, 11, 1, False),
      (5, 7, 2, 8, 1, True),
      (7, 8, 4, 5, 1, False),
      (2, 10, 3, 10, 2, True),
      (3, 12, 5, 11, 2, False),
  )
  def test_depthwise_conv1d_layer(self, batch_size, seq_len, kernel_size,
                                  input_dims, rhs_dilation_rate, bias):
    p = convolutions.DepthwiseConv1D.HParams(
        name='jax_depthwise_conv1d',
        filter_shape=(kernel_size, input_dims, 1),
        rhs_dilation_rate=rhs_dilation_rate,
        bias=bias)
    depthwiseconv1d = instantiate(p)
    npy_inputs = np.random.normal(
        1.0, 0.5, [batch_size, seq_len, input_dims]).astype('float32')
    inputs = jnp.asarray(npy_inputs)
    npy_paddings = np.random.randint(0, 2,
                                     [batch_size, seq_len]).astype('float32')
    paddings = jnp.asarray(npy_paddings)

    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = depthwiseconv1d.init(prng_key, inputs, paddings)
    output = depthwiseconv1d.apply(initial_vars, inputs, paddings)

    # Test whether tf DepthwiseConv layer returns the same output
    # Modify initial_vars to use TF compatible params
    tf_initial_vars = py_utils.NestedMap.FromNestedDict(initial_vars[PARAMS])
    tf_initial_vars.w = tf.expand_dims(tf_initial_vars.w, -1)

    tf_l = clwp.DepthwiseConv2DLayer.Params().Set(
        name='tf_depthwise_conv_layer',
        filter_shape=(kernel_size, 1, input_dims, 1),
        dilation_rate=(rhs_dilation_rate, 1),
        bias=bias)
    tf_depthwiseconv1d = tf_l.Instantiate()
    tf_output = tf_depthwiseconv1d.FProp(
        tf_initial_vars,
        tf.constant(jnp.expand_dims(inputs, axis=2), dtype=tf.float32),
        tf.constant(paddings, dtype=tf.float32))
    np_output = to_np(output)

    # TF implementation returns both output and padding
    tf_np_output = to_np(tf_output[0])[:, :, 0, :]
    self.assertAllClose(tf_np_output, np_output)

  @parameterized.parameters(
      (2, 10, 3, 10, 0.0, True),
      (3, 12, 5, 11, 0.1, False),
      (5, 7, 2, 8, 0.25, True),
      (7, 8, 4, 5, 0.5, False),
  )
  def test_light_conv1d_layer(self, batch_size, seq_len, kernel_size,
                              input_dims, dropout_prob, is_causal):
    p = convolutions.LightConv1D.HParams(
        name='jax_light_conv1d_layer',
        input_dims=input_dims,
        kernel_size=kernel_size,
        dropout_prob=dropout_prob,
        is_causal=is_causal)
    lconv = instantiate(p)
    npy_inputs = np.random.normal(
        1.0, 0.5, [batch_size, seq_len, input_dims]).astype('float32')
    inputs = jnp.asarray(npy_inputs)
    npy_paddings = np.random.randint(0, 2,
                                     [batch_size, seq_len]).astype('float32')
    paddings = jnp.asarray(npy_paddings)

    context_p = base_layer.JaxContext.HParams(do_eval=True)
    with base_layer.JaxContext.new_context(hparams=context_p):
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = lconv.init(prng_key, inputs, paddings)
      output = lconv.apply(initial_vars, inputs, paddings)
    # Test whether tf LConvLayer layer returns the same output
    # Modify initial_vars to use TF compatible params
    tf_initial_vars = py_utils.NestedMap.FromNestedDict(initial_vars[PARAMS])
    tf_initial_non_trainable_vars = py_utils.NestedMap.FromNestedDict(
        initial_vars[NON_TRAINABLE])
    tf_initial_vars = test_utils.replace_jax_light_conv_vars_to_tf(
        tf_initial_vars, tf_initial_non_trainable_vars)
    tf_l = conformer_layer.LConvLayer.Params().Set(
        name='tf_light_conv1d_block',
        input_dim=input_dims,
        kernel_size=kernel_size,
        dropout_prob=dropout_prob,
        is_causal=is_causal)
    tf_lconv = tf_l.Instantiate()
    with cluster_factory.SetEval(True):
      tf_output = tf_lconv.FProp(
          tf_initial_vars,
          tf.constant(inputs, dtype=tf.float32),
          paddings=tf.constant(npy_paddings, dtype=tf.float32))
    np_output = to_np(output)

    # TF implementation returns both output and padding
    tf_np_output = to_np(tf_output[0])
    self.assertAllClose(tf_np_output, np_output)

  @parameterized.parameters(
      (2, 3, 'SAME'),
      (3, 5, 'SAME'),
      (5, 3, 'SAME'),
      (7, 3, 'SAME'),
      (1, 3, 'SAME'),
      (5, 3, 'SAME'),
      (2, 3, 'VALID'),
      (3, 5, 'VALID'),
      (5, 3, 'VALID'),
      (7, 3, 'VALID'),
      (1, 3, 'VALID'),
      (5, 3, 'VALID'),
  )
  def test_conv_bnact_withpadding(self, stride, kernel_size, padding):
    p = convolutions.ConvBNActWithPadding.HParams(
        name='jax_withpadding_convolution',
        filter_shape=(kernel_size, kernel_size, 1, 1),
        filter_stride=(stride, stride),
        bias=False,
        batch_norm_tpl=None,
        padding=padding)
    layer = instantiate(p)
    prng_key = jax.random.PRNGKey(seed=123)

    batch_size = 10
    max_t = 20

    def get_padding_from_length(length):
      idx = np.tile(np.arange(max_t), [batch_size, 1])
      return (idx < np.expand_dims(length, -1)).astype('float32')

    length = np.random.randint(max_t // 2, max_t, [
        batch_size,
    ])
    logging.info('length:%s', str(length))
    npy_features = np.random.uniform(size=[batch_size, max_t, 80, 1])
    npy_paddings = get_padding_from_length(length)

    features = jnp.asarray(npy_features)
    paddings = jnp.asarray(npy_paddings)

    theta = layer.init(
        prng_key, features, paddings)
    _, output = layer.apply(
        theta, features, paddings)
    if padding == 'SAME':
      expect_output = get_padding_from_length((length + stride - 1) // stride)
    elif padding == 'VALID':
      expect_output = get_padding_from_length(
          (length - kernel_size + 1 + stride - 1) // stride)
    self.assertAllClose(expect_output[:, :output.shape[1]], to_np(output))

  @parameterized.parameters(1, 2)
  def test_dilated_conv(self, rhs_dilation_rate):
    test_layer_p = convolutions.DepthwiseConv1D.HParams(
        name='dw_1D',
        filter_shape=(3, 4, 1),  # kernel_size, in_channels, channel_multipliers
        is_causal=True,
        rhs_dilation_rate=rhs_dilation_rate
    )
    layer = instantiate(test_layer_p)
    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)

    inputs = np.random.normal(size=[1, 8, 4]).astype(np.float32)
    initial_vars = layer.init(init_key, inputs)
    jax_out = layer.apply(initial_vars, inputs, paddings=None)

    self.assertArraysEqual(inputs.shape, jax_out.shape)


if __name__ == '__main__':
  absltest.main()
