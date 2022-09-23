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
import numpy as np
from praxis import base_layer
from praxis.layers import conformers
from praxis.layers import streaming
from praxis.layers.streaming import test_utils

instantiate = base_layer.instantiate


class StreamingConformersTest(test_utils.StreamingTest):

  @parameterized.parameters([(4, 2, 1, 1), (6, 3, 2, 2)])
  def test_streaming_self_attention_with_norm_residual(self, block_size,
                                                       left_context,
                                                       right_context, step):
    batch_size = 1
    sequence_length = 16
    mdl_dim = 4

    inputs = np.random.normal(
        size=[batch_size, sequence_length, mdl_dim]).astype(
            np.float32)

    paddings = range(sequence_length)[-batch_size:]
    paddings = [[0] * l + [1] * (sequence_length - l) for l in paddings]
    paddings = np.array(paddings)

    hidden_dim = 4
    num_heads = 4

    # Non streaming layer
    p_non_stream = conformers.SelfAttentionWithNormAndResidual.HParams(
        name='self_atten_non_stream')
    p_non_stream.set(left_context=left_context, right_context=right_context,)
    p_non_stream.self_atten_tpl.set(
        input_dim=mdl_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads)
    p_non_stream.norm_tpl.set(dim=mdl_dim)

    # Streaming aware layer
    p_stream = streaming.SelfAttentionWithNormAndResidual.HParams(
        name='self_atten_norm_res_stream')
    p_stream.self_atten_tpl.set(
        input_dim=mdl_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        block_size=block_size,
        left_context=left_context,
        right_context=right_context)
    p_stream.norm_tpl.set(dim=mdl_dim)

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
    inputs = np.random.normal(
        1.0, 0.5, [batch_size, seq_len, input_dims]).astype('float32')
    inputs = np.asarray(inputs)

    def get_padding_from_length(length):
      idx = np.tile(np.arange(seq_len), [batch_size, 1])
      return (idx >= np.expand_dims(length, -1)).astype('float32')

    length = np.random.randint(seq_len // 2, seq_len, (batch_size,))
    paddings = get_padding_from_length(length)
    paddings = np.asarray(paddings)

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

    # Setting left and right context in SelfAttentionWithNormAndResidual,
    # so that it becomes local self attention.
    # Note: it uses non sreaming DotProductAttention
    # to emulate LocalSelfAttention we set additional
    # left_context and right_context, so that it will create attention mask
    # for DotProductAttention.
    p_non_stream.trans_atten_tpl.set(
        left_context=left_context,
        right_context=right_context)
    # Conv has to be causal.
    p_non_stream.lconv_tpl.set(is_causal=True)

    # Streaming aware layer
    p_stream = streaming.Conformer.HParams(
        name='conformer_stream',
        input_dims=input_dims,
        conv_residual_dropout=0.0,
        atten_residual_dropout=dropout_prob,
        ffn_residual_dropout=dropout_prob,
        atten_dropout=dropout_prob,
        ffn_relu_dropout=dropout_prob,
        kernel_size=kernel_size,
        model_dims=model_dims,
        atten_num_heads=atten_num_heads)
    p_stream.trans_atten_tpl.self_atten_tpl.set(
        left_context=left_context,
        right_context=right_context,
        block_size=block_size)
    p_stream.lconv_tpl.set(is_causal=True)

    self.assertEqual(p_stream.cls.get_stride(p_stream), 1)
    self.assertEqual(p_stream.cls.get_right_context(p_stream), right_context)
    self._compare_stream_non_stream(inputs, paddings, p_non_stream, p_stream,
                                    step)


if __name__ == '__main__':
  absltest.main()
