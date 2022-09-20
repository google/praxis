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
import jax.numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import py_utils
from praxis import test_utils
from praxis.layers import conformers
from praxis.layers import streaming
from praxis.layers.streaming import operations

instantiate = base_layer.instantiate


class StreamingConformersTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  def _append_dims(self, x, ndim):
    if ndim == 1:
      return jnp.expand_dims(x, -1)
    else:
      return jnp.reshape(x, jnp.concat([x.shape, [1] * ndim], axis=0))

  def _compare_stream_non_stream(self,
                                 inputs,
                                 paddings,
                                 p_non_stream,
                                 p_stream,
                                 step,
                                 is_eval=True):
    context_p = base_layer.JaxContext.HParams(do_eval=True)
    with base_layer.JaxContext.new_context(hparams=context_p):
      layer_non_stream = instantiate(p_non_stream)
      prng_key = jax.random.PRNGKey(seed=123)
      prng_key, init_key = jax.random.split(prng_key)
      initial_vars = layer_non_stream.init(init_key, inputs, paddings)
      output_non_stream = layer_non_stream.apply(
          initial_vars, inputs, paddings)

      layer_stream = instantiate(p_stream)
      in_nmap = py_utils.NestedMap(features=inputs, paddings=paddings)
      output_names = ['features', 'paddings']
      output_stream = operations.run_streaming(layer_stream, initial_vars,
                                               in_nmap, output_names, step)

    right_context = p_stream.cls.get_right_context(p_stream)

    # Remove delay elements from streaming output.
    output_stream.features = output_stream.features[:, right_context:,]
    output_stream.paddings = output_stream.paddings[:, right_context:,]

    # Size of time dim after removing streaming delay elements.
    time_size = output_stream.features.shape[1]
    non_stream_paddings = paddings[:, :time_size,]
    mask = self._append_dims(1. - non_stream_paddings,
                             output_non_stream.ndim - 2)

    # Last elements in non streaming outputs have to be removed due to delay.
    self.assertAllClose(non_stream_paddings, output_stream.paddings)
    self.assertAllClose(output_non_stream[:, :time_size,] * mask,
                        output_stream.features * mask)

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
    self._compare_stream_non_stream(inputs, paddings, p_non_stream, p_stream,
                                    step)

  @parameterized.parameters(
      (2, 10, 3, 8, 8, 4, 0.0, 6, 3, 2, 2),
      (3, 12, 5, 16, 16, 2, 0.1, 8, 2, 2, 1),
      (5, 7, 2, 8, 8, 8, 0.25, 4, 2, 1, 1),
      (7, 8, 4, 16, 16, 4, 0.5, None, 3, 3, 2),
  )
  def test_streaming_conformer(self, batch_size, seq_len, kernel_size, input_dims,
                               model_dims, atten_num_heads, dropout_prob,
                               block_size, left_context, right_context, step):
    inputs = np.random.normal(
        1.0, 0.5, [batch_size, seq_len, input_dims]).astype('float32')
    inputs = jnp.asarray(inputs)

    def get_padding_from_length(length):
      idx = np.tile(np.arange(seq_len), [batch_size, 1])
      return (idx >= np.expand_dims(length, -1)).astype('float32')

    length = np.random.randint(seq_len // 2, seq_len, (batch_size,))
    paddings = get_padding_from_length(length)
    paddings = jnp.asarray(paddings)

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
        right_context=right_context)
    p_stream.lconv_tpl.set(is_causal=True)

    self.assertEqual(p_stream.cls.get_stride(p_stream), 1)
    self.assertEqual(p_stream.cls.get_right_context(p_stream), right_context)
    self._compare_stream_non_stream(inputs, paddings, p_non_stream, p_stream,
                                    step)


if __name__ == '__main__':
  absltest.main()
