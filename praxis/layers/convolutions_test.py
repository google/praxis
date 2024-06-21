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
from praxis import pax_fiddle
from praxis import py_utils
from praxis import test_utils
from praxis.layers import activations
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
      ((5, 4, 24, 36), (1, 1), (1, 1), False, [2, 16, 36, 72]),
      ((2, 4, 16, 8), (2, 2), (1, 1), False, [2, 16, 32, 128]),
      ((4, 8, 16, 32), (1, 1), (1, 1), False, [2, 16, 32, 64]),
      ((2, 8, 16, 32), (1, 1), (2, 2), False, [2, 16, 32, 64]),
      ((2, 8, 16, 32), (2, 2), (2, 2), False, [2, 16, 32, 64]),
      ((2, 8, 16, 32), (1, 1), (2, 1), False, [2, 16, 32, 64]),
      ((2, 8, 16, 32), (2, 2), (2, 1), False, [2, 16, 32, 64]),
      ((2, 8, 16, 32), (1, 1), (2, 2), True, [2, 16, 32, 64]),
      ((2, 8, 16, 32), (2, 2), (2, 2), True, [2, 16, 32, 64]),
  )
  def test_conv2d_layer_same_padding(
      self,
      filter_shape,
      filter_stride,
      dilations,
      tf_equivalent_padding,
      input_shape,
  ):
    p = pax_fiddle.Config(
        convolutions.Conv2D,
        name='jax_conv2d',
        filter_shape=filter_shape,
        filter_stride=filter_stride,
        dilations=dilations,
        tf_equivalent_padding=tf_equivalent_padding,
        padding='SAME',
    )
    conv_layer = instantiate(p)
    npy_inputs = np.random.normal(1.0, 0.5, input_shape).astype('float32')
    inputs = jnp.asarray(npy_inputs)

    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = conv_layer.init(prng_key, inputs)

    output = conv_layer.apply(initial_vars, inputs)
    # Test whether output has same shape as input in height and width
    for i in [1, 2]:
      self.assertEqual(output.shape[i], inputs.shape[i] // filter_stride[i - 1])

  @parameterized.parameters(
      (2, 3, 2),
      (2, 4, 2),
      (3, 3, 2),
      (3, 4, 2),
  )
  def test_conv_2d_for_causal_1d_conv(
      self,
      stride_time_dim,
      filter_time_dim,
      dilation_time_dim,
  ):
    filter_shape = (filter_time_dim, 1, 1, 1)
    p = pax_fiddle.Config(
        convolutions.Conv2D,
        name='jax_conv2d',
        filter_shape=filter_shape,
        filter_stride=(stride_time_dim, 1),
        dilations=(dilation_time_dim, 1),
        is_causal=True,
        tf_equivalent_padding=True,
        padding='SAME',
    )
    conv_layer = instantiate(p)

    length = 7
    inputs = jnp.arange(length * stride_time_dim)[
        jnp.newaxis, :, jnp.newaxis, jnp.newaxis
    ].astype(jnp.float32)
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = conv_layer.init(prng_key, inputs)
    initial_vars['params']['w'] = jnp.ones(filter_shape)

    output = conv_layer.apply(initial_vars, inputs)

    # With padding == 'SAME' and stride > 1 output length should be
    # (length*stride) / stride = length.
    self.assertEqual(output.shape, (1, length, 1, 1))

  def test_causal_conv2d_layer(self):
    p = pax_fiddle.Config(
        convolutions.Conv2D,
        name='jax_conv2d',
        filter_shape=[3, 3, 1, 1],
        filter_stride=[2, 2],
        dilations=[2, 2],
        padding='SAME',
        is_causal=True,
        tf_equivalent_padding=True,
    )
    conv_layer = instantiate(p)
    npy_inputs = np.arange(25).reshape((1, 5, 5, 1)).astype('float32')
    inputs = jnp.asarray(npy_inputs)

    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = conv_layer.init(prng_key, inputs)
    initial_vars = jax.tree.map(jnp.ones_like, initial_vars)

    # Test odd length sequence.
    output = conv_layer.apply(initial_vars, inputs)
    np_output = np.array(
        [[2.0, 6.0, 6.0], [24.0, 42.0, 32.0], [66.0, 108.0, 78.0]]
    )
    self.assertAllClose(to_np(output[0, :, :, 0]), np_output)

    # Test even length sequence.
    npy_inputs = np.arange(36).reshape((1, 6, 6, 1)).astype('float32')
    inputs = jnp.asarray(npy_inputs)

    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = conv_layer.init(prng_key, inputs)
    initial_vars = jax.tree.map(jnp.ones_like, initial_vars)

    output = conv_layer.apply(initial_vars, inputs)
    np_output = np.array(
        [[2.0, 6.0, 6.0], [28.0, 48.0, 36.0], [78.0, 126.0, 90.0]]
    )
    self.assertAllClose(to_np(output[0, :, :, 0]), np_output)

    # Add some more causal tests using nan as indicator
    # Case 1:
    p = pax_fiddle.Config(
        convolutions.Conv2D,
        name='conv2d_nan',
        filter_shape=[3, 1, 1, 1],
        filter_stride=[2, 1],
        padding='SAME',
        is_causal=True,
        tf_equivalent_padding=True,
    )
    conv_layer = instantiate(p)
    conv_filter = {'params': {'w': jnp.ones([3, 1, 1, 1])}}
    conv_layer = conv_layer.bind(conv_filter)
    x = jnp.arange(7)[jnp.newaxis, :, jnp.newaxis, jnp.newaxis].astype(
        jnp.float32
    )
    # x: [0, nan, 2, 3, 4, 5, 6]
    x = x.at[:, 1, :, :].set(jnp.nan)
    # expected y: [0, nan, 9, 15]
    # 0 <-- x x 0
    # nan <-- 0 nan 2
    # 9 <-- 2 3 4
    # 15 <-- 4 5 6
    y = conv_layer(x)

    assert y[0, 0, 0, 0] == 0
    assert jnp.isnan(y[0, 1, 0, 0])
    assert y[0, 2, 0, 0] == 9
    assert y[0, 3, 0, 0] == 15

    # Case 2:
    p = pax_fiddle.Config(
        convolutions.Conv2D,
        name='conv2d_nan',
        filter_shape=[2, 1, 1, 1],
        filter_stride=[1, 1],
        padding='SAME',
        is_causal=True,
        tf_equivalent_padding=True,
    )
    conv_layer = instantiate(p)
    conv_filter = {'params': {'w': jnp.ones([2, 1, 1, 1])}}
    conv_layer = conv_layer.bind(conv_filter)
    x = jnp.arange(5)[jnp.newaxis, :, jnp.newaxis, jnp.newaxis].astype(
        jnp.float32
    )
    # x: [0, nan, 2, 3, nan]
    x = x.at[:, 1, :, :].set(jnp.nan)
    x = x.at[:, -1, :, :].set(jnp.nan)
    # expected y: [0, nan, nan, 5, nan]
    y = conv_layer(x)

    print('#bai#:', x[0, :, 0, 0])
    print('#bai#:', y[0, :, 0, 0])
    assert y[0, 0, 0, 0] == 0
    assert jnp.isnan(y[0, 1, 0, 0])
    assert jnp.isnan(y[0, 2, 0, 0])
    assert y[0, 3, 0, 0] == 5
    assert jnp.isnan(y[0, 4, 0, 0])

  @parameterized.parameters(
      ((2, 5, 4, 24, 36), (1, 1, 1), [2, 4, 16, 36, 72], jnp.float32),
      ((2, 2, 4, 16, 8), (2, 2, 2), [2, 8, 16, 32, 128], jnp.float32),
      ((2, 4, 8, 16, 32), (1, 1, 1), [2, 8, 16, 32, 64], jnp.bfloat16),
  )
  def test_conv3d_layer_same_padding(
      self,
      filter_shape,
      filter_stride,
      input_shape,
      fprop_dtype,
  ):
    p = pax_fiddle.Config(
        convolutions.Conv3D,
        name='jax_conv3d',
        filter_shape=filter_shape,
        filter_stride=filter_stride,
        dilations=(1, 1, 1),
        padding='SAME',
        fprop_dtype=fprop_dtype,
    )
    conv_layer = instantiate(p)
    npy_inputs = np.random.normal(1.0, 0.5, input_shape).astype('float32')
    inputs = jnp.asarray(npy_inputs)

    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = conv_layer.init(prng_key, inputs)

    output = conv_layer.apply(initial_vars, inputs)
    # Test whether output has same shape as input in time, height and width
    for i in [1, 2, 3]:
      self.assertEqual(output.shape[i], inputs.shape[i] // filter_stride[i - 1])
    self.assertEqual(output.dtype, fprop_dtype)

  @parameterized.parameters(
      (2, 10, 3, 10, 1, True),
      (3, 12, 5, 11, 1, False),
      (5, 7, 2, 8, 1, True),
      (7, 8, 4, 5, 1, False),
      (2, 10, 3, 10, 2, True),
      (3, 12, 5, 11, 2, False),
  )
  def test_depthwise_conv1d_layer(
      self,
      batch_size,
      seq_len,
      kernel_size,
      input_dims,
      rhs_dilation_rate,
      bias,
  ):
    p = pax_fiddle.Config(
        convolutions.DepthwiseConv1D,
        name='jax_depthwise_conv1d',
        filter_shape=(kernel_size, input_dims, 1),
        rhs_dilation_rate=rhs_dilation_rate,
        bias=bias,
    )
    depthwiseconv1d = instantiate(p)
    npy_inputs = np.random.normal(
        1.0, 0.5, [batch_size, seq_len, input_dims]
    ).astype('float32')
    npy_inputs[:, -1, :] = np.nan
    inputs = jnp.asarray(npy_inputs)
    npy_paddings = np.random.randint(0, 2, [batch_size, seq_len]).astype(
        'float32'
    )
    npy_paddings[0, -1] = 1.0
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
        bias=bias,
    )
    tf_depthwiseconv1d = tf_l.Instantiate()
    tf_output = tf_depthwiseconv1d.FProp(
        tf_initial_vars,
        tf.constant(jnp.expand_dims(inputs, axis=2), dtype=tf.float32),
        tf.constant(paddings, dtype=tf.float32),
    )
    np_output = to_np(output)

    # TF implementation returns both output and padding
    tf_np_output = to_np(tf_output[0])[:, :, 0, :]
    self.assertAllClose(tf_np_output, np_output)

  @parameterized.product(
      batch_size=[2, 7],
      seq_len=[7, 12],
      kernel_size=[2, 3, 5],
      input_dims=[5, 11],
      dilations=[(1, 1), (1, 2), (2, 1)],
      channel_multipliers=[1, 3],
  )
  def test_depthwise_conv2d_layer(
      self,
      batch_size,
      seq_len,
      kernel_size,
      input_dims,
      dilations,
      channel_multipliers,
  ):
    # Create fake inputs.
    npy_inputs = np.random.normal(
        1.0, 0.5, [batch_size, seq_len, seq_len, input_dims]
    ).astype('float32')
    inputs = jnp.asarray(npy_inputs)
    npy_paddings = np.random.randint(0, 2, [batch_size, seq_len]).astype(
        'float32'
    )
    paddings = jnp.asarray(npy_paddings)

    # Use ConvBNActWithPadding to test paddings.
    kernel_shape = (kernel_size, kernel_size)
    p = convolutions.ConvBNActWithPadding.HParamsDepthwise(
        name='jax_depthwise_conv2d',
        kernel_shape=kernel_shape,
        in_channels=input_dims,
        channel_multipliers=channel_multipliers,
        filter_stride=(1, 1),
        dilations=dilations,
        bias=True,
        tf_equivalent_padding=True,
        batch_norm_tpl=None,
        activation_tpl=pax_fiddle.Config(activations.Identity),
    )
    depthwiseconv2d = instantiate(p)
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = depthwiseconv2d.init(prng_key, inputs, paddings)
    output, out_paddings = depthwiseconv2d.apply(initial_vars, inputs, paddings)

    # Test whether tf DepthwiseConv layer returns the same output
    # Modify initial_vars to use TF compatible params
    tf_initial_vars = py_utils.NestedMap.FromNestedDict(initial_vars[PARAMS])
    tf_filter_shape = kernel_shape + (input_dims, channel_multipliers)
    tf_initial_vars.w = tf.reshape(tf_initial_vars.w, tf_filter_shape)

    tf_l = clwp.DepthwiseConv2DLayer.Params().Set(
        name='tf_depthwise_conv_layer',
        filter_shape=tf_filter_shape,
        filter_stride=(1, 1),
        dilation_rate=dilations,
        bias=True,
    )
    tf_depthwiseconv2d = tf_l.Instantiate()
    tf_output, tf_out_paddings = tf_depthwiseconv2d.FProp(
        tf_initial_vars, tf.constant(inputs), tf.constant(paddings)
    )

    self.assertAllClose(to_np(tf_out_paddings), to_np(out_paddings))
    self.assertAllClose(to_np(tf_output), to_np(output))

  @parameterized.product(
      batch_size=[2, 7],
      seq_len=[7, 12],
      kernel_size=[2, 3, 5],
      input_dims=[5, 11],
      channel_multipliers=[1, 3],
  )
  def test_depthwise_conv3d_layer(
      self, batch_size, seq_len, kernel_size, input_dims, channel_multipliers
  ):
    # Create fake inputs.
    npy_inputs = np.random.normal(
        1.0, 0.5, [batch_size, seq_len, seq_len, seq_len, input_dims]
    ).astype('float32')
    inputs = jnp.asarray(npy_inputs)
    kernel_shape = (kernel_size, kernel_size, kernel_size)
    p = convolutions.Conv3D.HParamsDepthwise(
        name='jax_depthwise_conv3d',
        kernel_shape=kernel_shape,
        in_channels=input_dims,
        channel_multipliers=channel_multipliers,
        filter_stride=(1, 1, 1),
        bias=True,
        tf_equivalent_padding=True,
    )
    depthwiseconv3d = instantiate(p)
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = depthwiseconv3d.init(prng_key, inputs)
    output = depthwiseconv3d.apply(initial_vars, inputs)
    # check time, height, width shape.
    for i in [1, 2, 3]:
      self.assertEqual(output.shape[i], inputs.shape[i])
    # Check channels shape.
    self.assertEqual(output.shape[-1], input_dims * channel_multipliers)

  @parameterized.parameters(
      (2, 10, 3, 10, 0.0, True),
      (3, 12, 5, 11, 0.1, False),
      (5, 7, 2, 8, 0.25, True),
      (7, 8, 4, 5, 0.5, False),
  )
  def test_light_conv1d_layer(
      self,
      batch_size,
      seq_len,
      kernel_size,
      input_dims,
      dropout_prob,
      is_causal,
  ):
    p = pax_fiddle.Config(
        convolutions.LightConv1D,
        name='jax_light_conv1d_layer',
        input_dims=input_dims,
        kernel_size=kernel_size,
        dropout_prob=dropout_prob,
        is_causal=is_causal,
    )
    lconv = instantiate(p)
    npy_inputs = np.random.normal(
        1.0, 0.5, [batch_size, seq_len, input_dims]
    ).astype('float32')
    inputs = jnp.asarray(npy_inputs)
    npy_paddings = np.random.randint(0, 2, [batch_size, seq_len]).astype(
        'float32'
    )
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
        initial_vars[NON_TRAINABLE]
    )
    tf_initial_vars = test_utils.replace_jax_light_conv_vars_to_tf(
        tf_initial_vars, tf_initial_non_trainable_vars
    )
    tf_l = conformer_layer.LConvLayer.Params().Set(
        name='tf_light_conv1d_block',
        input_dim=input_dims,
        kernel_size=kernel_size,
        dropout_prob=dropout_prob,
        is_causal=is_causal,
    )
    tf_lconv = tf_l.Instantiate()
    with cluster_factory.SetEval(True):
      tf_output = tf_lconv.FProp(
          tf_initial_vars,
          tf.constant(inputs, dtype=tf.float32),
          paddings=tf.constant(npy_paddings, dtype=tf.float32),
      )
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
    p = pax_fiddle.Config(
        convolutions.ConvBNActWithPadding,
        name='jax_withpadding_convolution',
        filter_shape=(kernel_size, kernel_size, 1, 1),
        filter_stride=(stride, stride),
        bias=False,
        batch_norm_tpl=None,
        padding=padding,
    )
    layer = instantiate(p)
    prng_key = jax.random.PRNGKey(seed=123)

    batch_size = 10
    max_t = 20

    def get_padding_from_length(length):
      idx = np.tile(np.arange(max_t), [batch_size, 1])
      return (idx < np.expand_dims(length, -1)).astype('float32')

    length = np.random.randint(
        max_t // 2,
        max_t,
        [
            batch_size,
        ],
    )
    logging.info('length:%s', str(length))
    npy_features = np.random.uniform(size=[batch_size, max_t, 80, 1])
    npy_paddings = get_padding_from_length(length)
    npy_features[0, -1, :] = np.nan
    npy_paddings[0, -1] = 1.0

    features = jnp.asarray(npy_features)
    paddings = jnp.asarray(npy_paddings)

    theta = layer.init(prng_key, features, paddings)
    output, output_padding = layer.apply(theta, features, paddings)
    if padding == 'SAME':
      expect_output_padding = get_padding_from_length(
          (length + stride - 1) // stride
      )
      # get_padding_from_length() doesn't consider paddings, here we fix it for
      # stride 1. But this may break for other inputs.
      if stride == 1:
        expect_output_padding[0, -1] = 1.0
    elif padding == 'VALID':
      expect_output_padding = get_padding_from_length(
          (length - kernel_size + 1 + stride - 1) // stride
      )
    self.assertAllClose(
        expect_output_padding[:, : output_padding.shape[1]],
        to_np(output_padding),
    )

    # Validate that batch and time dimension are the same.
    self.assertArraysEqual(output.shape[0:2], output_padding.shape)

  @parameterized.parameters(1, 2)
  def test_dilated_conv(self, rhs_dilation_rate):
    test_layer_p = pax_fiddle.Config(
        convolutions.DepthwiseConv1D,
        name='dw_1D',
        filter_shape=(3, 4, 1),  # kernel_size, in_channels, channel_multipliers
        is_causal=True,
        rhs_dilation_rate=rhs_dilation_rate,
    )
    layer = instantiate(test_layer_p)
    prng_key = jax.random.PRNGKey(seed=123)
    _, init_key = jax.random.split(prng_key)

    inputs = np.random.normal(size=[1, 8, 4]).astype(np.float32)
    initial_vars = layer.init(init_key, inputs)
    jax_out = layer.apply(initial_vars, inputs, paddings=None)

    self.assertArraysEqual(inputs.shape, jax_out.shape)


if __name__ == '__main__':
  absltest.main()
