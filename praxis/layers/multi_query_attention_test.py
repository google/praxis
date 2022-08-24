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

"""Tests for Praxis attention layers."""

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import py_utils
from praxis import test_utils
from praxis.layers import attentions
from praxis.layers import multi_query_attention
import tensorflow.compat.v2 as tf

instantiate = base_layer.instantiate


class MultiQueryAttentionTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)
    tf.random.set_seed(123)

  def test_one_headed_projection_shape(self):
    test_layer_p = multi_query_attention.OneHeadedAttentionProjection.HParams(
        name='mh', input_dim=16, output_dim=5)
    layer = instantiate(test_layer_p)

    inputs = np.random.normal(1.5, 2.0, [5, 16]).astype(np.float32)

    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)
    initial_vars = layer.init(init_key, inputs)
    logging.info('initial_vars: %s', initial_vars)

    jax_out = layer.apply(initial_vars, inputs)
    self.assertSequenceEqual(jax_out.shape, [5, 5])

  @parameterized.parameters([False, True])
  def test_multi_query_attention_shape(self, use_rotary_position_emb):
    test_layer_p = multi_query_attention.MultiQueryDotProductAttention.HParams(
        name='mqa', input_dim=16, hidden_dim=60, num_heads=10,
        use_rotary_position_emb=use_rotary_position_emb)
    layer = instantiate(test_layer_p)
    inputs = np.random.normal(1.5, 2.0, [5, 12, 16]).astype(np.float32)
    atten_mask = jnp.zeros([1, 1, 1, 12])
    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)

    with base_layer.JaxContext.new_context():
      initial_vars = layer.init(init_key, inputs, inputs, inputs, atten_mask)
      logging.info('initial_vars: %s', initial_vars)
      encoded, attens = layer.apply(initial_vars, inputs, inputs, inputs,
                                    atten_mask)
    self.assertSequenceEqual(encoded.shape, [5, 12, 16])
    self.assertSequenceEqual(attens.shape, [5, 10, 12, 12])

  @parameterized.parameters([False, True])
  def test_multi_query_attention_decoding_shape(self, use_rotary_position_emb):
    test_layer_p = multi_query_attention.MultiQueryDotProductAttention.HParams(
        name='mqa', input_dim=16, hidden_dim=60, num_heads=10,
        use_rotary_position_emb=use_rotary_position_emb)
    layer = instantiate(test_layer_p)
    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)

    query_vec = jnp.zeros([5, 10, 16])

    with base_layer.JaxContext.new_context():
      initial_vars = layer.init(init_key, query_vec, query_vec, query_vec,
                                attentions.causal_mask(query_vec))
      logging.info('initial_vars: %s', initial_vars)
      _, attention_states = layer.apply(
          initial_vars,
          query_vec,
          query_vec,
          query_vec,
          attentions.causal_mask(query_vec),
          mutable=[base_layer.DECODE_CACHE])
      updated_vars = py_utils.MergeDictsWithValueCheck(attention_states,
                                                       initial_vars)
      encoded = layer.apply(
          updated_vars,
          method=layer.extend_step,
          query_vec=query_vec[:, 0, :],
          atten_mask=attentions.causal_mask(query_vec)[:, 0, :],
          time_step=1,
          segment_pos=None)
    self.assertSequenceEqual(encoded.shape, [5, 16])

  def test_multi_query_attention_consistent(self):
    test_layer_p = multi_query_attention.MultiQueryDotProductAttention.HParams(
        name='mqa', input_dim=16, hidden_dim=50, num_heads=10)
    layer = instantiate(test_layer_p)

    inputs = np.random.normal(1.5, 2.0, [5, 2, 16]).astype(np.float32)
    mask = attentions.causal_mask(inputs)

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      prng_key, init_key = jax.random.split(prng_key)
      initial_vars = layer.init(init_key, inputs, inputs, inputs, mask)
      logging.info('initial_vars: %s', initial_vars)
      encoded, _ = layer.apply(initial_vars, inputs, inputs, inputs, mask)

    with base_layer.JaxContext.new_context():
      zero_vec = jnp.zeros_like(inputs)
      _, attention_states = layer.apply(
          initial_vars,
          zero_vec,
          zero_vec,
          zero_vec,
          attentions.causal_mask(zero_vec),
          method=layer.__call__,
          mutable=[base_layer.DECODE_CACHE])
      updated_vars = py_utils.MergeDictsWithValueCheck(attention_states,
                                                       initial_vars)
      output = jnp.zeros_like(encoded)
      for t in range(inputs.shape[1]):
        e, a = layer.apply(
            updated_vars,
            method=layer.extend_step,
            query_vec=inputs[:, t, :],
            atten_mask=mask[:, :, t],
            time_step=t,
            segment_pos=None,
            mutable=[base_layer.DECODE_CACHE])
        updated_vars = py_utils.MergeDictsWithValueCheck(a, initial_vars)
        output = output.at[:, t, :].set(e)
    self.assertAllClose(encoded, output)

  # TODO(apassos) test the SPMD codepath for deriving sharding annotations.

if __name__ == '__main__':
  absltest.main()
