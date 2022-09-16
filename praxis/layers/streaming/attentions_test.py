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

"""Tests for Praxis streaming attention layers."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import py_utils
from praxis import test_utils
from praxis.layers import attentions
from praxis.layers import streaming
from praxis.layers.streaming import operations

instantiate = base_layer.instantiate


class StreamingAttentionsTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  @parameterized.parameters([(4, 2, 1, 1, True), (4, 2, 1, 1, False),
                             (8, 3, 5, 2, True), (8, 3, 5, 1, False),
                             (5, 4, 0, 1, False), (5, 4, 0, 1, True)])
  def test_streaming_local_self_attention(self, block_size, left_context,
                                          right_context, step, is_full):

    mdl_dim = 16
    hidden_dim = 32
    num_heads = 4

    batch_size = 1
    sequence_length = 16

    query_vec = np.random.normal(
        size=[batch_size, sequence_length, mdl_dim]).astype(np.float32)
    key_vec = np.random.normal(
        size=[batch_size, sequence_length, mdl_dim]).astype(np.float32)
    value_vec = np.random.normal(
        size=[batch_size, sequence_length, mdl_dim]).astype(np.float32)

    paddings = range(sequence_length)[-batch_size:]
    paddings = [[0] * l + [1] * (sequence_length - l) for l in paddings]
    paddings = np.array(paddings)
    atten_mask = attentions.convert_paddings_to_mask(paddings, np.float32)
    if is_full:
      atten_mask = jnp.tile(atten_mask, [1, 1, sequence_length, 1])

    base_layer_p = attentions.LocalSelfAttention.HParams(
        name='self_atten',
        input_dim=mdl_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        block_size=block_size,
        left_context=left_context,
        right_context=right_context,
    )
    layer = instantiate(base_layer_p)

    context_p = base_layer.JaxContext.HParams(do_eval=True)
    with base_layer.JaxContext.new_context(hparams=context_p):
      prng_key = jax.random.PRNGKey(seed=123)
      prng_key, init_key = jax.random.split(prng_key)
      initial_vars = layer.init(init_key, query_vec, key_vec, value_vec,
                                atten_mask)
      encoded_base_non_stream, atten_base_prob_non_stream = layer.apply(
          initial_vars, query_vec, key_vec, value_vec, atten_mask)

    # Streaming aware layer
    with base_layer.JaxContext.new_context(hparams=context_p):
      p = streaming.LocalSelfAttention.HParams(name='self_atten_stream')
      p.copy_fields_from(base_layer_p)
      layer = instantiate(p)
      encoded_non_stream, atten_prob_non_stream = layer.apply(
          initial_vars, query_vec, key_vec, value_vec, atten_mask)

    in_nmap = py_utils.NestedMap(
        query_vec=query_vec,
        key_vec=key_vec,
        value_vec=value_vec,
        paddings=paddings)
    output_names = ['encoded', 'paddings']
    output_stream = operations.run_streaming(layer, initial_vars, in_nmap,
                                             output_names, step)

    self.assertEqual(p.cls.get_stride(p), 1)
    self.assertEqual(p.cls.get_right_context(p), right_context)

    self.assertAllClose(encoded_base_non_stream, encoded_non_stream)
    self.assertAllClose(atten_base_prob_non_stream, atten_prob_non_stream)

    # Remove delay elements from streaming output.
    output_stream.encoded = output_stream.encoded[:, p.right_context:,]
    output_stream.paddings = output_stream.paddings[:, p.right_context:,]

    # Size of time dim after removing streaming delay elements.
    time_size = output_stream.encoded.shape[1]
    # Last elements in non streaming outputs have to be removed due to delay.
    self.assertAllClose(paddings[:, :time_size,], output_stream.paddings)
    self.assertAllClose(encoded_base_non_stream[:, :time_size,],
                        output_stream.encoded)

if __name__ == '__main__':
  absltest.main()
