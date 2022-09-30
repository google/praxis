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

"""Tests for Praxis streaming convolutional layers."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from praxis import base_layer
from praxis.layers import convolutions
from praxis.layers import streaming
from praxis.layers.streaming import test_utils

instantiate = base_layer.instantiate


class StreamingConvolutionsTest(test_utils.TestCase):

  def _get_padding_from_length(self, max_time_size, batch_size):
    length = np.random.randint(max_time_size // 4, max_time_size // 2, [
        batch_size,
    ])
    idx = np.tile(np.arange(max_time_size), [batch_size, 1])
    return (idx < np.expand_dims(length, -1)).astype('float32')

  @parameterized.parameters(
      (1, 3, 'SAME', 2, True, True),
      (2, 3, 'SAME', 1, False, True),
      (3, 5, 'SAME', 1, True, True),
      (5, 3, 'SAME', 1, False, True),
      (7, 3, 'SAME', 1, False, True),
      (2, 3, 'VALID', 2, False, False),
      (3, 5, 'VALID', 1, False, False),
      (5, 3, 'VALID', 1, False, False),
      (7, 3, 'VALID', 1, True, False),
      (1, 3, 'VALID', 2, True, False),
      (5, 3, 'VALID', 1, True, False),
  )
  def test_conv_bnact_withpadding(self, stride, kernel_size, padding,
                                  batch_size, compat_with_lingvo, is_causal):
    # Compare original ConvBNActWithPadding non streaming version with
    # ConvBNActWithPadding streaming aware in non streaming mode.
    feature_dim = stride * 3
    max_time_size = 5 * stride
    features = np.random.uniform(
        size=[batch_size, max_time_size, feature_dim, 1])
    paddings = self._get_padding_from_length(max_time_size, batch_size)
    features = np.asarray(features)
    paddings = np.asarray(paddings)

    p_non_stream = convolutions.ConvBNActWithPadding.HParams(
        name='conv_bn_act_padding',
        is_causal=is_causal,
        tf_equivalent_padding=True,
        compat_with_lingvo=compat_with_lingvo,
        filter_shape=(kernel_size, kernel_size, 1, 1),
        filter_stride=(stride, stride),
        bias=False,
        batch_norm_tpl=None,
        padding=padding)
    layer = instantiate(p_non_stream)

    # Streaming aware layer
    p_stream = streaming.ConvBNActWithPadding.HParams(
        name='conv_bn_act_padding_stream')
    p_stream.copy_fields_from(p_non_stream)
    # Streaming aware layer needs explicit frequency_dim (for states creation).
    p_stream.frequency_dim = feature_dim

    layer_stream = instantiate(p_stream)

    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = layer.init(prng_key, features, paddings)

    output, output_padding = layer.apply(initial_vars, features, paddings)
    output_stream, output_padding_stream = layer_stream.apply(
        initial_vars, features, paddings)

    self.assertAllClose(output, output_stream)
    self.assertAllClose(output_padding, output_padding_stream)

  @parameterized.parameters(
      (1, 3, 'SAME', 2, True),
      (2, 3, 'SAME', 1, False),
      (3, 5, 'SAME', 1, True),
      (5, 3, 'SAME', 1, False),
  )
  def test_conv_bnact_withpadding_streaming(self, stride, kernel_size, padding,
                                            batch_size, compat_with_lingvo):
    # Compare streaming aware ConvBNActWithPadding
    # in streaming and non streaming modes.
    feature_dim = stride * 3
    # Streaming aware layer
    p_stream = streaming.ConvBNActWithPadding.HParams(
        name='conv_bn_act_padding_stream',
        is_causal=True,
        tf_equivalent_padding=True,
        compat_with_lingvo=compat_with_lingvo,
        filter_shape=(kernel_size, kernel_size, 1, 1),
        filter_stride=(stride, stride),
        bias=False,
        batch_norm_tpl=None,
        frequency_dim=feature_dim,
        padding=padding)

    max_time_size = 8 * stride
    features = np.random.uniform(
        size=[batch_size, max_time_size, feature_dim, 1])
    paddings = self._get_padding_from_length(max_time_size, batch_size)
    features = np.asarray(features)
    paddings = np.asarray(paddings)

    self.assertEqual(p_stream.cls.get_stride(p_stream), stride)
    # Causal model has no delay:
    self.assertEqual(p_stream.cls.get_right_context(p_stream), 0)
    self._compare_stream_non_stream(
        features,
        paddings,
        p_stream,
        p_stream,
        stride)

  @parameterized.parameters((2, 3, True), (2, 3, False), (4, 3, True),
                            (4, 4, True))
  def test_depthwise_conv1D_streaming(self, step, kernel_size, with_paddings):

    # Input data.
    inputs = np.random.normal(size=[1, 8, 4]).astype(np.float32)
    paddings = np.array([[1, 1, 1, 0, 0, 0, 1, 1]]).astype(np.float32)

    # Original layer.
    p_non_stream = convolutions.DepthwiseConv1D.HParams(
        name='dw_1D',
        # (kernel_size, in_channels, channel_multipliers)
        filter_shape=(kernel_size, 4, 1),
        is_causal=True,
        rhs_dilation_rate=2
    )

    # Streaming aware layer
    p_stream = streaming.DepthwiseConv1D.HParams(name='dw_1D_stream')
    p_stream.copy_fields_from(p_non_stream)

    # Striding is always 1 for now:
    self.assertEqual(p_stream.cls.get_stride(p_stream), 1)
    # Causal model has no delay:
    self.assertEqual(p_stream.cls.get_right_context(p_stream), 0)
    self._compare_stream_non_stream(
        inputs,
        paddings if with_paddings else None,
        p_non_stream,
        p_stream,
        step)
    self._compare_stream_non_stream(
        inputs,
        paddings if with_paddings else None,
        p_stream,
        p_stream,
        step)

  @parameterized.parameters((2, True), (2, False), (4, True))
  def test_light_conv1D_streaming(self, step, with_paddings):

    # Input data.
    inputs = np.random.normal(size=[1, 8, 4]).astype(np.float32)
    paddings = np.array([[1, 1, 1, 0, 0, 0, 1, 1]]).astype(np.float32)

    p_non_stream = convolutions.LightConv1D.HParams(
        name='non_stream_conv1D',
        input_dims=inputs.shape[-1],
        kernel_size=3,
        is_causal=True)

    # Streaming aware layer.
    p_stream = streaming.LightConv1D.HParams(name='stream_conv1D')
    p_stream.copy_fields_from(p_non_stream)

    # Striding is always 1.
    self.assertEqual(p_stream.cls.get_stride(p_stream), 1)
    # Causal model has no delay:
    self.assertEqual(p_stream.cls.get_right_context(p_stream), 0)
    self._compare_stream_non_stream(
        inputs,
        paddings if with_paddings else None,
        p_non_stream,
        p_stream,
        step)

  def test_light_conv1D_assert_on_wrong_layer(self):
    # Input data.
    inputs = np.random.normal(size=[1, 8, 4]).astype(np.float32)
    paddings = np.array([[1, 1, 1, 0, 0, 0, 1, 1]]).astype(np.float32)

    # Assign depthwise_conv_tpl with wrong Dummy layer:
    p_non_stream = convolutions.LightConv1D.HParams(
        name='non_stream_conv1D',
        input_dims=inputs.shape[-1],
        kernel_size=3,
        is_causal=True,
        depthwise_conv_tpl=convolutions.LightConv1D.HParams(name='dummy'))

    p_stream = streaming.LightConv1D.HParams(name='stream_conv1D')
    p_stream.copy_fields_from(p_non_stream)

    context_p = base_layer.JaxContext.HParams(do_eval=True)
    with base_layer.JaxContext.new_context(hparams=context_p):
      layer_stream = instantiate(p_stream)
      prng_key = jax.random.PRNGKey(seed=123)
      prng_key, init_key = jax.random.split(prng_key)
      with self.assertRaises(ValueError):
        _ = layer_stream.init(init_key, inputs, paddings)

if __name__ == '__main__':
  absltest.main()
