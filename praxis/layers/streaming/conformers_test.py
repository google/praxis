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

"""Tests for Praxis streaming conformers layers."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from praxis import base_layer
from praxis.layers import attentions
from praxis.layers import conformers
from praxis.layers import streaming
from praxis.layers.streaming import test_utils

instantiate = base_layer.instantiate


class StreamingConformersTest(test_utils.TestCase):

  def _generate_data(self, seq_len, batch_size, input_dims):
    def get_padding_from_length(length):
      idx = np.tile(np.arange(seq_len), [batch_size, 1])
      return (idx >= np.expand_dims(length, -1)).astype('float32')
    inputs = np.random.normal(
        1.0, 0.5, [batch_size, seq_len, input_dims]).astype('float32')
    length = np.random.randint(seq_len // 2, seq_len, (batch_size,))
    paddings = get_padding_from_length(length)
    return np.asarray(inputs), np.asarray(paddings)

  @parameterized.parameters([(4, 2, 1, 1), (6, 3, 2, 2)])
  def test_streaming_self_attention_with_norm_residual(self, block_size,
                                                       left_context,
                                                       right_context, step):
    batch_size, sequence_length = 1, 16
    mdl_dim = 4
    inputs, paddings = self._generate_data(sequence_length, batch_size, mdl_dim)
    hidden_dim, num_heads = 4, 4

    # Non streaming layer
    p_non_stream = conformers.SelfAttentionWithNormAndResidual.HParams(
        name='self_atten_non_stream',
        left_context=left_context,
        right_context=right_context)
    p_non_stream.self_atten_tpl.set(
        input_dim=mdl_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        block_size=block_size)
    p_non_stream.norm_tpl.set(dim=mdl_dim)

    # Streaming aware layer
    p_stream = streaming.SelfAttentionWithNormAndResidual.HParams(
        name='self_atten_norm_res_stream')
    p_stream.copy_fields_from(p_non_stream)

    self.assertEqual(p_stream.cls.get_stride(p_stream), 1)
    self.assertEqual(p_stream.cls.get_right_context(p_stream), right_context)
    self._compare_stream_non_stream(
        inputs,
        paddings,
        p_non_stream,
        p_stream,
        step)

  @parameterized.parameters(
      (2, 10, 3, 8, 8, 4, 0.0, 6, 3, 2, 2),
      (3, 12, 5, 16, 16, 2, 0.1, 8, 2, 2, 1),
      (5, 7, 2, 8, 8, 8, 0.25, 4, 2, 1, 1),
      (7, 8, 4, 16, 16, 4, 0.5, None, 3, 3, 2),
  )
  def test_streaming_conformer(self, batch_size, seq_len, kernel_size,
                               input_dims, model_dims, atten_num_heads,
                               dropout_prob, block_size, left_context,
                               right_context, step):
    inputs, paddings = self._generate_data(seq_len, batch_size, input_dims)
    # Non streaming layer
    p_non_stream = conformers.Conformer.HParams(
        name='conformer_non_stream',
        input_dims=input_dims,
        conv_residual_dropout=0.0,
        atten_residual_dropout=dropout_prob,
        ffn_residual_dropout=dropout_prob,
        atten_dropout=dropout_prob,
        ffn_relu_dropout=dropout_prob,
        kernel_size=kernel_size,
        model_dims=model_dims,
        atten_num_heads=atten_num_heads)
    p_non_stream.trans_atten_tpl.set(
        left_context=left_context,
        right_context=right_context)
    p_non_stream.trans_atten_tpl.self_atten_tpl.set(
        block_size=block_size)
    p_non_stream.lconv_tpl.set(is_causal=True)

    # Streaming aware layer
    p_stream = streaming.Conformer.HParams(
        name='conformer_stream')
    p_stream.copy_fields_from(p_non_stream)

    self.assertEqual(p_stream.cls.get_stride(p_stream), 1)
    self.assertEqual(p_stream.cls.get_right_context(p_stream), right_context)
    self._compare_stream_non_stream(inputs, paddings, p_non_stream, p_stream,
                                    step)

  def _non_stream_model_params(self, input_dims):
    return conformers.Conformer.HParams(
        name='conformer_non_stream',
        input_dims=input_dims,
        conv_residual_dropout=0.0,
        atten_residual_dropout=0.0,
        ffn_residual_dropout=0.0,
        atten_dropout=0.0,
        ffn_relu_dropout=0.0,
        kernel_size=3,
        model_dims=8,
        atten_num_heads=4)

  def _test_streaming_with_wrong_parameter(self, p_non_stream, seq_len,
                                           batch_size, input_dims):
    inputs, paddings = self._generate_data(seq_len, batch_size, input_dims)
    p_stream = streaming.Conformer.HParams(name='conformer_stream')
    p_stream.copy_fields_from(p_non_stream)
    context_p = base_layer.JaxContext.HParams(do_eval=True)
    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)
    context_p = base_layer.JaxContext.HParams(do_eval=True)
    with base_layer.JaxContext.new_context(hparams=context_p):
      layer_stream = instantiate(p_stream)
      with self.assertRaises(ValueError):
        _ = layer_stream.init(init_key, inputs, paddings)

  def test_conformer_assert_on_wrong_params(self, batch_size=1,
                                            seq_len=10, input_dims=8):
    p_non_stream = self._non_stream_model_params(input_dims)
    # Set different context in both trans_atten_tpl and self_atten_tpl,
    p_non_stream.trans_atten_tpl.set(left_context=3, right_context=2)
    p_non_stream.trans_atten_tpl.self_atten_tpl = (
        attentions.LocalSelfAttention.HParams(
            left_context=1, right_context=1))
    self._test_streaming_with_wrong_parameter(p_non_stream, seq_len, batch_size,
                                              input_dims)

    p_non_stream = self._non_stream_model_params(input_dims)
    # Set unsupported self_atten_tpl yet.
    p_non_stream.trans_atten_tpl.self_atten_tpl = (
        attentions.LocalSelfAttentionXL.HParams(
            left_context=1, right_context=1))
    self._test_streaming_with_wrong_parameter(p_non_stream, seq_len, batch_size,
                                              input_dims)


if __name__ == '__main__':
  absltest.main()
