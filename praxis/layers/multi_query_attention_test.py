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

"""Tests for Praxis attention layers."""
import itertools
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import pax_fiddle
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
    test_layer_p = pax_fiddle.Config(
        multi_query_attention.OneHeadedAttentionProjection,
        name='mh',
        input_dim=16,
        output_dim=5,
    )
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
    test_layer_p = pax_fiddle.Config(
        multi_query_attention.MultiQueryDotProductAttention,
        name='mqa',
        input_dim=16,
        hidden_dim=60,
        num_heads=10,
        use_rotary_position_emb=use_rotary_position_emb,
    )
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

  @parameterized.parameters(*list(itertools.product([True, False], repeat=2)))
  def test_multi_query_attention_decoding_shape(self,
                                                use_rotary_position_emb,
                                                lpb):
    if lpb:
      mqa = multi_query_attention.MultiQueryDotProductAttentionLPB
    else:
      mqa = multi_query_attention.MultiQueryDotProductAttention
    test_layer_p = pax_fiddle.Config(
        mqa,
        name='mqa',
        input_dim=16,
        hidden_dim=60,
        num_heads=10,
        use_rotary_position_emb=use_rotary_position_emb,
    )
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
      updated_vars = py_utils.merge_dict(attention_states, initial_vars)
      atten_mask = attentions.causal_mask(query_vec)[:, :, 0, :]
      encoded = layer.apply(
          updated_vars,
          method=layer.extend_step,
          query_vec=query_vec[:, 0, :],
          atten_mask=atten_mask,
          time_step=1,
          segment_pos=None)
    self.assertSequenceEqual(encoded.shape, [5, 16])

  @parameterized.parameters([True, False])
  def test_mqa_extend_n_steps_with_lazy_broadcast_state(
      self, use_rotary_position_emb):
    mdl_dim = 4
    hidden_dim = 8
    num_heads = 2
    mqa_lpb = multi_query_attention.MultiQueryDotProductAttentionLPB
    test_layer_p = mqa_lpb.config(
        name='mq',
        input_dim=mdl_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dim_per_head=4 if use_rotary_position_emb else None,
        atten_logit_cap=20.0,
        use_rotary_position_emb=use_rotary_position_emb)
    layer = instantiate(test_layer_p)
    target_batch_size = 3
    source_max_length = 8
    prefix_len = 4
    num_samples = 2
    query_vec = np.random.normal(
        size=[target_batch_size, source_max_length, mdl_dim]).astype(np.float32)
    prefix = query_vec[:, 0:prefix_len, :]
    key_vec = query_vec
    value_vec = query_vec
    atten_mask = attentions.causal_mask(query_vec)
    prefix_atten_mask = attentions.causal_mask(prefix)

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      prng_key, init_key = jax.random.split(prng_key)
      initial_vars = layer.init(init_key, query_vec, key_vec, value_vec,
                                atten_mask)
      logging.info('initial_vars: %s', initial_vars)
      fprop_out, attention_states = layer.apply(
          initial_vars,
          query_vec,
          key_vec,
          value_vec,
          atten_mask,
          method=layer.__call__)
      # Updates decode states in fprop.
      _, attention_states = layer.apply(
          initial_vars,
          prefix,
          prefix,
          prefix,
          prefix_atten_mask,
          method=layer.__call__,
          mutable=[base_layer.DECODE_CACHE])
      updated_vars = py_utils.merge_dict(attention_states, initial_vars)

      # First lazy broadcast.
      _, attention_states = layer.apply(
          updated_vars,
          num_suffix_samples=num_samples,
          suffix_length=source_max_length - prefix_len,
          method=layer.lazy_broadcast_prefix,
          mutable=[base_layer.DECODE_CACHE, base_layer.PREFIX_DECODE_CACHE])
      updated_vars = py_utils.merge_dict(attention_states, initial_vars)

      def _broadcast_sample(x, num_samples):
        return jnp.repeat(x, num_samples, axis=0)

      suffix_segment_pos = jnp.stack(
          [jnp.arange(prefix_len, source_max_length)] * target_batch_size)

      # Call extend step start from prefix_len.
      encoded, _ = layer.apply(
          updated_vars,
          query_vec=_broadcast_sample(query_vec[:, prefix_len:, :],
                                      num_samples),
          atten_mask=atten_mask[:, :, prefix_len:, :],
          time_step=prefix_len,
          segment_pos=_broadcast_sample(suffix_segment_pos, num_samples),
          method=layer.extend_step,
          mutable=[base_layer.DECODE_CACHE])
      self.assertAllClose(
          encoded,
          _broadcast_sample(fprop_out, num_samples)[:, prefix_len:, :])

  @parameterized.parameters([True, False])
  def test_mqa_with_lazy_broadcast_state(self, use_rotary_position_emb):
    mdl_dim = 4
    hidden_dim = 8
    num_heads = 2
    mqa_lpb = multi_query_attention.MultiQueryDotProductAttentionLPB
    test_layer_p = mqa_lpb.config(
        name='mq',
        input_dim=mdl_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dim_per_head=4 if use_rotary_position_emb else None,
        atten_logit_cap=20.0)
    layer = instantiate(test_layer_p)
    target_batch_size = 3
    source_max_length = 8
    suffix_1_len = 2
    suffix_2_len = 2
    prefix_len = 4
    query_vec = np.random.normal(
        size=[target_batch_size, source_max_length, mdl_dim]).astype(np.float32)
    prefix = query_vec[:, 0:prefix_len, :]
    key_vec = query_vec
    value_vec = query_vec
    atten_mask = attentions.causal_mask(query_vec)
    prefix_atten_mask = attentions.causal_mask(prefix)

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      prng_key, init_key = jax.random.split(prng_key)
      initial_vars = layer.init(init_key, query_vec, key_vec, value_vec,
                                atten_mask)
      logging.info('initial_vars: %s', initial_vars)
      fprop_out, attention_states = layer.apply(
          initial_vars,
          query_vec,
          key_vec,
          value_vec,
          atten_mask,
          method=layer.__call__)
      # Updates decode states in fprop.
      _, attention_states = layer.apply(
          initial_vars,
          prefix,
          prefix,
          prefix,
          prefix_atten_mask,
          method=layer.__call__,
          mutable=[base_layer.DECODE_CACHE])
      updated_vars = py_utils.merge_dict(attention_states, initial_vars)

      # First lazy broadcast.
      _, attention_states = layer.apply(
          updated_vars,
          num_suffix_samples=2,
          suffix_length=suffix_1_len,
          method=layer.lazy_broadcast_prefix,
          mutable=[base_layer.DECODE_CACHE, base_layer.PREFIX_DECODE_CACHE])
      updated_vars = py_utils.merge_dict(attention_states, initial_vars)

      def _broadcast_sample(x, num_samples):
        return jnp.repeat(x, num_samples, axis=0)

      # Call extend step start from prefix_len.
      for t in range(prefix_len, prefix_len + suffix_1_len):
        encoded, attention_states = layer.apply(
            updated_vars,
            query_vec=_broadcast_sample(query_vec[:, t, :], 2),
            atten_mask=atten_mask[:, :, t, :],
            time_step=t,
            segment_pos=None,
            method=layer.extend_step,
            mutable=[base_layer.DECODE_CACHE])
        del updated_vars[base_layer.DECODE_CACHE]
        updated_vars = py_utils.merge_dict(attention_states, updated_vars)
        encoded = jnp.reshape(encoded, (-1, 2) + encoded.shape[1:])
        # First sample.
        self.assertAllClose(fprop_out[:, t, :], encoded[:, 0])
        # Second sample.
        self.assertAllClose(fprop_out[:, t, :], encoded[:, 1])

      # Second lazy broadcast.
      _, attention_states = layer.apply(
          updated_vars,
          num_suffix_samples=3,
          suffix_length=suffix_2_len,
          method=layer.lazy_broadcast_prefix,
          mutable=[base_layer.DECODE_CACHE, base_layer.PREFIX_DECODE_CACHE])
      updated_vars = py_utils.merge_dict(attention_states, initial_vars)

      # Call extend step start from prefix_len + suffix_1_len.
      for t in range(prefix_len + suffix_1_len,
                     prefix_len + suffix_1_len + suffix_2_len):
        encoded, attention_states = layer.apply(
            updated_vars,
            query_vec=_broadcast_sample(query_vec[:, t, :], 6),
            atten_mask=atten_mask[:, :, t, :],
            time_step=t,
            segment_pos=None,
            method=layer.extend_step,
            mutable=[base_layer.DECODE_CACHE])
        del updated_vars[base_layer.DECODE_CACHE]
        updated_vars = py_utils.merge_dict(attention_states, updated_vars)
        encoded = jnp.reshape(encoded, (-1, 6) + encoded.shape[1:])
        for sample_id in range(6):
          self.assertAllClose(fprop_out[:, t, :], encoded[:, sample_id])

  @parameterized.parameters([True, False])
  def test_multi_query_attention_consistent(self, lpb):
    if lpb:
      mqa = multi_query_attention.MultiQueryDotProductAttentionLPB
    else:
      mqa = multi_query_attention.MultiQueryDotProductAttention
    test_layer_p = pax_fiddle.Config(
        mqa,
        name='mqa',
        input_dim=16,
        hidden_dim=50,
        num_heads=10,
    )
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
      updated_vars = py_utils.merge_dict(attention_states, initial_vars)
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
        updated_vars = py_utils.merge_dict(a, initial_vars)
        output = output.at[:, t, :].set(e)
    self.assertAllClose(encoded, output)

  # TODO(apassos) test the SPMD codepath for deriving sharding annotations.

if __name__ == '__main__':
  absltest.main()
