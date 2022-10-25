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

import itertools

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from lingvo.core import batch_major_attention
import numpy as np
from praxis import base_layer
from praxis import decoder_utils
from praxis import py_utils
from praxis import test_utils
from praxis.layers import attentions
import tensorflow.compat.v2 as tf

instantiate = base_layer.instantiate


def var_stats(x):
  return np.mean(x), np.std(x)


def assert_var_stats_close(map01, map02, test_case):

  map01_items = map01.FlattenItems()
  map02_items = map02.FlattenItems()

  def have_similar_stats(x, y):
    mean1, std1 = var_stats(test_utils.to_np(x))
    mean2, std2 = var_stats(test_utils.to_np(y))
    delta_mean = np.abs(mean1 - mean2)
    delta_std = np.abs(std1 - std2)
    logging.info('mean1: %s, mean2: %s', mean1, mean2)
    logging.info('std1: %s, std2: %s', std1, std2)
    test_case.assertLess(delta_mean, 0.0002)
    test_case.assertLess(delta_std, 0.0002)

  for x, y in zip(map01_items, map02_items):
    assert x[0] == y[0]
    have_similar_stats(x[1], y[1])


class BlockUtilsTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('_single_block', 7),
      ('_one_frame_block', 1),
      ('_two_frame_blocks', 2),
  )
  def test_convert_to_block(self, block_size):
    x = np.random.random([2, 6, 2, 3, 4])
    x_blocks = attentions.convert_to_block(x, block_size)
    # Check shape.
    batch_size = x.shape[0]
    other_dims = x.shape[2:]
    num_blocks = int(np.ceil(x.shape[1] / float(block_size)))
    expected_shape = (batch_size, num_blocks, block_size) + other_dims
    self.assertEqual(expected_shape, x_blocks.shape)

    # Check values.
    x_recover = x_blocks.reshape((x_blocks.shape[0], -1) + x_blocks.shape[3:])
    x_recover = x_recover[:, :x.shape[1], ...]
    self.assertAllClose(x, x_recover)

  @parameterized.named_parameters(
      ('_single_block', 7, 2, 1),
      ('_single_frame_context', 1, 1, 0),
      ('_other_case_1', 3, 4, 1),
      ('_other_case_2', 4, 2, 4),
  )
  def test_extract_block_context(self, block_size, left_context, right_context):
    x = np.random.random([2, 6, 2, 3, 4])
    x_context = attentions.extract_block_context(x, block_size, left_context,
                                                 right_context)
    # Check shape.
    batch_size = x.shape[0]
    other_dims = x.shape[2:]
    num_blocks = int(np.ceil(x.shape[1] / float(block_size)))
    context_size = block_size + left_context - 1 + right_context
    expected_shape = (batch_size, num_blocks, context_size) + other_dims
    self.assertEqual(expected_shape, x_context.shape)

    # Check values block by block.
    for block_idx in range(num_blocks):
      context_start = block_idx * block_size - left_context + 1
      context_end = (block_idx + 1) * block_size + right_context
      slice_start = max(0, context_start)
      slice_end = min(x.shape[1], context_end)
      expected_val = x[:, slice_start:slice_end, ...]
      actual_val = x_context[:, block_idx, ...]
      # remove paddings
      front_padding = slice_start - context_start
      back_padding = context_end - slice_end
      actual_val = actual_val[:, front_padding:context_size - back_padding, ...]
      self.assertAllClose(expected_val, actual_val)

  def _get_reference_causal_padding(self, seq_len, block_size, left_context,
                                    right_context):
    num_blocks = int(np.ceil(seq_len / float(block_size)))
    context_size = block_size + left_context - 1 + right_context
    padding = np.ones((num_blocks, block_size, context_size))

    for i in range(num_blocks):
      for j in range(block_size):
        actual_src_pos = j + (i * block_size)
        if actual_src_pos < seq_len:
          for k in range(context_size):
            actual_tgt_pos = k + i * block_size - (left_context - 1)
            if 0 <= actual_tgt_pos and actual_tgt_pos < seq_len:
              diff = actual_src_pos - actual_tgt_pos
              if -right_context <= diff and diff < left_context:
                padding[i, j, k] = 0

    return padding

  @parameterized.named_parameters(
      ('_single_block', 6, 9, 2, 1),
      ('_single_frame_block', 6, 1, 2, 1),
      ('_single_frame_context', 6, 1, 1, 0),
      ('_other_case_1', 6, 3, 4, 1),
      ('_other_case_2', 6, 4, 2, 4),
  )
  def test_make_local_mask(self, seq_len, block_size, left_context,
                           right_context):
    mask = attentions._make_local_mask(seq_len, block_size, left_context,
                                       right_context)
    padding = 1.0 - mask

    ref_padding = self._get_reference_causal_padding(seq_len, block_size,
                                                     left_context,
                                                     right_context)
    self.assertAllClose(ref_padding, padding)



class AttentionsTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)
    tf.random.set_seed(123)

  def test_per_dim_scale(self):
    test_layer_p = attentions.PerDimScale.HParams(name='scale', dim=4)
    layer = instantiate(test_layer_p)
    inputs = np.random.normal(1.5, 2.0, [5, 4]).astype(np.float32)

    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)
    initial_vars = layer.init(init_key, inputs)
    initial_vars['params']['per_dim_scale'] = jnp.array([-0.5, 0.5, 1.0, 0.0],
                                                        dtype=jnp.float32)
    logging.info('initial_vars: %s', initial_vars)

    jax_out = layer.apply(initial_vars, inputs)

    logging.info('jax_output: %s', jax_out)

    # Now run TF based computation.
    tf_initial_vars = py_utils.NestedMap.FromNestedDict(initial_vars['params'])
    tf_layer_p = batch_major_attention.PerDimScaleLayer.Params().Set(
        name='scale', dim=4)
    tf_layer = tf_layer_p.Instantiate()
    tf_output1 = tf_layer.FProp(tf_layer.theta, inputs)
    logging.info('tf_output1: %s', tf_output1)
    tf_output2 = tf_layer.FProp(tf_initial_vars, inputs)
    logging.info('tf_output2: %s', tf_output2)
    self.assertAllClose(test_utils.to_np(jax_out), test_utils.to_np(tf_output2))

  def test_mhd_projection_01(self):
    test_layer_p = attentions.AttentionProjection.HParams(
        name='mh',
        input_dim=16,
        num_heads=2,
        dim_per_head=5,
        is_output_projection=False)
    layer = instantiate(test_layer_p)
    inputs = np.random.normal(1.5, 2.0, [5, 16]).astype(np.float32)

    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)
    initial_vars = layer.init(init_key, inputs)
    logging.info('initial_vars: %s', initial_vars)

    jax_out = layer.apply(initial_vars, inputs)

    logging.info('jax_output: %s', jax_out)

    # Now run TF based computation.
    tf_initial_vars = py_utils.NestedMap.FromNestedDict(initial_vars['params'])
    tf_layer_p = batch_major_attention.MultiHeadedProjectionLayer.Params().Set(
        name='mh',
        input_dim=16,
        num_heads=2,
        dim_per_head=5,
        is_output_projection=False)
    tf_layer = tf_layer_p.Instantiate()
    tf_output1 = tf_layer.FProp(tf_layer.theta, inputs)
    logging.info('tf_output1: %s', tf_output1)
    tf_output2 = tf_layer.FProp(tf_initial_vars, inputs)
    logging.info('tf_output2: %s', tf_output2)
    self.assertGreater(
        np.sum(
            np.abs(test_utils.to_np(tf_output1) -
                   test_utils.to_np(tf_output2))), 0.1)
    self.assertAllClose(test_utils.to_np(jax_out), test_utils.to_np(tf_output2))

  @parameterized.parameters([False, True])
  def test_mhd_projection_02(self, use_nhd_shape):
    test_layer_p = attentions.AttentionProjection.HParams(
        name='mh',
        input_dim=16,
        num_heads=2,
        dim_per_head=5,
        is_output_projection=True,
        use_nhd_shape=use_nhd_shape,
    )
    layer = instantiate(test_layer_p)
    inputs = np.random.normal(1.5, 2.0, [5, 2, 5]).astype(np.float32)

    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)

    with base_layer.JaxContext.new_context():
      initial_vars = layer.init(init_key, inputs)
      logging.info('initial_vars: %s', initial_vars)
      jax_out = layer.apply(initial_vars, inputs)

    logging.info('jax_output: %s', jax_out)

    if use_nhd_shape:
      initial_vars['params']['w'] = jnp.einsum('ABC->CAB',
                                               initial_vars['params']['w'])

    # Now run TF based computation.
    tf_initial_vars = py_utils.NestedMap.FromNestedDict(initial_vars['params'])
    tf_layer_p = batch_major_attention.MultiHeadedProjectionLayer.Params().Set(
        name='mh',
        input_dim=16,
        num_heads=2,
        dim_per_head=5,
        is_output_projection=True)
    tf_layer = tf_layer_p.Instantiate()
    tf_output1 = tf_layer.FProp(tf_layer.theta, inputs)
    logging.info('tf_output1: %s', tf_output1)
    tf_output2 = tf_layer.FProp(tf_initial_vars, inputs)
    logging.info('tf_output2: %s', tf_output2)
    self.assertGreater(
        np.sum(
            np.abs(test_utils.to_np(tf_output1) -
                   test_utils.to_np(tf_output2))), 0.1)
    self.assertAllClose(test_utils.to_np(jax_out), test_utils.to_np(tf_output2))

  def test_mhd_projection_var_stats(self):
    test_layer_p = attentions.AttentionProjection.HParams(
        name='mh',
        input_dim=256,
        num_heads=16,
        dim_per_head=16,
        is_output_projection=True)
    layer = instantiate(test_layer_p)
    inputs = np.random.normal(1.5, 2.0, [2, 16, 16]).astype(np.float32)

    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)
    initial_vars = layer.init(init_key, inputs)

    # Now run TF based computation.
    tf_layer_p = batch_major_attention.MultiHeadedProjectionLayer.Params().Set(
        name='mh',
        input_dim=256,
        num_heads=16,
        dim_per_head=16,
        is_output_projection=True)
    tf_layer = tf_layer_p.Instantiate()

    tf_initial_vars = jax.tree_util.tree_map(lambda x: x.numpy(),
                                             tf_layer.theta)
    initial_vars = py_utils.NestedMap.FromNestedDict(initial_vars['params'])
    assert_var_stats_close(initial_vars, tf_initial_vars, self)

  def test_mask(self):
    a = np.random.randint(0, 6, size=[2, 50])
    jax_mask = attentions.causal_segment_mask(a, jnp.float32)
    tf_mask = batch_major_attention.CausalSegmentMask(a, tf.float32)
    self.assertAllClose(test_utils.to_np(jax_mask), test_utils.to_np(tf_mask))

  def test_causal_attention_mask(self):
    a = np.zeros([2, 5])
    causal_mask = np.zeros([2, 5])
    # Only enable causal mask for the last 2 token positions.
    causal_mask[:, 3:] = 1
    jax_mask = attentions.causal_segment_mask(a, jnp.float32, causal_mask)
    mask = np.array(jax_mask)
    mask[mask < 0] = -1
    mask = mask.astype(np.int32)
    self.assertAllClose(
        mask, [[[[0, 0, 0, -1, -1], [0, 0, 0, -1, -1], [0, 0, 0, -1, -1],
                 [0, 0, 0, 0, -1], [0, 0, 0, 0, 0]]],
               [[[0, 0, 0, -1, -1], [0, 0, 0, -1, -1], [0, 0, 0, -1, -1],
                 [0, 0, 0, 0, -1], [0, 0, 0, 0, 0]]]])

  @parameterized.parameters([(False, True, 3, True, True),
                             (True, True, 3, True, True),
                             (False, True, 3, True, False),
                             (True, True, 3, True, False),
                             (False, True, 4, False, False),
                             (True, True, 4, True, False),
                             (False, False, 1, False, False),
                             (True, False, 1, True, False),
                             (False, False, 1, True, False),
                             (True, False, 1, True, False)])
  def test_mha_01(self, combine_qkv, dconv_qkv, dconv_kernel_size,
                  use_rotary_position_emb, simulate_packed):
    mdl_dim = 16
    hidden_dim = 32
    num_heads = 4
    test_layer_p = attentions.DotProductAttention.HParams(
        name='mh',
        input_dim=mdl_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dim_per_head=16 if use_rotary_position_emb else None,
        atten_logit_cap=20.0,
        combine_qkv=combine_qkv,
        dconv_qkv=dconv_qkv,
        dconv_kernel_size=dconv_kernel_size,
        use_rotary_position_emb=use_rotary_position_emb)
    layer = instantiate(test_layer_p)
    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)
    target_batch_size = 3
    source_max_length = 16
    target_max_length = 16
    query_vec = np.random.normal(
        size=[target_batch_size, source_max_length, mdl_dim]).astype(np.float32)
    key_vec = query_vec
    value_vec = query_vec
    fake_query_vec = jnp.zeros_like(query_vec)
    atten_mask = attentions.causal_mask(query_vec)
    segment_pos = np.tile(np.arange(source_max_length), (target_batch_size, 1))

    starting_index = 0
    if simulate_packed:
      starting_index = dconv_kernel_size
      atten_mask = atten_mask.at[:, :, :, :starting_index].set(-2.3819763e+38)
      segment_pos = jnp.maximum(segment_pos - starting_index, 0)

    with base_layer.JaxContext.new_context():
      initial_vars = layer.init(
          init_key,
          fake_query_vec,
          fake_query_vec,
          fake_query_vec,
          atten_mask,
          query_segment_pos=segment_pos,
          key_segment_pos=segment_pos)
      logging.info('initial_vars: %s', initial_vars)
      _, attention_states = layer.apply(
          initial_vars,
          fake_query_vec,
          fake_query_vec,
          fake_query_vec,
          atten_mask,
          query_segment_pos=segment_pos,
          key_segment_pos=segment_pos,
          method=layer.__call__,
          mutable=[base_layer.DECODE_CACHE])
      fprop_out, _ = layer.apply(
          initial_vars,
          query_vec,
          key_vec,
          value_vec,
          atten_mask,
          query_segment_pos=segment_pos,
          key_segment_pos=segment_pos,
          method=layer.__call__)

      decoder_output = jnp.zeros(
          shape=[target_max_length, target_batch_size, mdl_dim])

      updated_vars = py_utils.MergeDictsWithValueCheck(attention_states,
                                                       initial_vars)
      for t in range(starting_index, target_max_length):
        encoded, attention_states = layer.apply(
            updated_vars,
            query_vec=query_vec[:, t, :],
            atten_mask=atten_mask[:, :, t, :],
            time_step=t,
            segment_pos=None,
            method=layer.extend_step,
            mutable=[base_layer.DECODE_CACHE])
        updated_vars = py_utils.MergeDictsWithValueCheck(
            attention_states, initial_vars)
        decoder_output = decoder_output.at[t].set(encoded)

    decoder_output = decoder_output[starting_index:]
    decoder_out_transposed = jnp.transpose(decoder_output, [1, 0, 2])
    fprop_out = fprop_out[:, starting_index:]

    logging.info('fprop_out: %s', fprop_out)
    logging.info('decoder_out: %s', decoder_output)
    self.assertAllClose(fprop_out, decoder_out_transposed)

  @parameterized.product(
      enable_query_scale=[True, False],
      enable_per_dim_scale=[True, False],
  )
  def test_mha_02(self, enable_query_scale, enable_per_dim_scale):
    mdl_dim = 16
    hidden_dim = 32
    num_heads = 4
    test_layer_p = attentions.DotProductAttention.HParams(
        name='mh',
        input_dim=mdl_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        atten_logit_cap=20.0,
        internal_enable_query_scale=enable_query_scale,
        internal_enable_per_dim_scale=enable_per_dim_scale,
    )
    layer = instantiate(test_layer_p)

    target_batch_size = 3
    source_max_length = 8
    target_max_length = 8

    query_vec = np.random.normal(
        size=[target_batch_size, source_max_length, mdl_dim]).astype(np.float32)
    key_vec = np.random.normal(
        size=[target_batch_size, source_max_length, mdl_dim]).astype(np.float32)
    value_vec = np.random.normal(
        size=[target_batch_size, source_max_length, mdl_dim]).astype(np.float32)
    segment_ids = np.random.randint(
        0, 2, size=[target_batch_size, target_max_length]).astype(np.int32)
    atten_mask = attentions.causal_segment_mask(segment_ids, np.float32)

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      prng_key, init_key = jax.random.split(prng_key)
      initial_vars = layer.init(init_key, query_vec, key_vec, value_vec,
                                atten_mask)

      jax_fprop_out, jax_atten_prob = layer.apply(initial_vars, query_vec,
                                                  key_vec, value_vec,
                                                  atten_mask)

    tf_initial_vars = py_utils.NestedMap.FromNestedDict(initial_vars['params'])
    tf_initial_vars.atten_dropout = None
    tf_layer_p = batch_major_attention.MultiHeadedAttention.Params().Set(
        name='mh',
        input_dim=mdl_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        atten_logit_cap=20.0,
        enable_query_scale=enable_query_scale,
        enable_per_dim_scale=enable_per_dim_scale,
        packed_input=True)
    tf_layer = tf_layer_p.Instantiate()
    tf_out, tf_atten_prob = tf_layer.FProp(
        tf_initial_vars,
        query_vec,
        key_vec,
        value_vec,
        paddings=tf.zeros([target_batch_size, source_max_length]),
        segment_mask=atten_mask)

    logging.info('jax_layer_out: %s', jax_fprop_out)
    logging.info('jax_atten_probs: %s', jax_atten_prob)
    logging.info('tf_layer_out: %s', tf_out)
    logging.info('tf_atten_probs: %s', tf_atten_prob)
    self.assertAllClose(
        test_utils.to_np(jax_fprop_out), test_utils.to_np(tf_out))
    self.assertAllClose(
        test_utils.to_np(jax_atten_prob), test_utils.to_np(tf_atten_prob))

  @parameterized.product(
      rel_pos_emb_dim=[10, 16],)
  def test_attention_xl(self, rel_pos_emb_dim):
    mdl_dim = 16
    hidden_dim = 32
    num_heads = 4
    test_layer_p = attentions.DotProductAttentionXL.HParams(
        name='mh',
        input_dim=mdl_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        rel_pos_emb_dim=rel_pos_emb_dim,
    )
    layer = instantiate(test_layer_p)

    target_batch_size = 3
    source_max_length = 8

    query_vec = np.random.normal(
        size=[target_batch_size, source_max_length, mdl_dim]).astype(np.float32)
    key_vec = np.random.normal(
        size=[target_batch_size, source_max_length, mdl_dim]).astype(np.float32)
    value_vec = np.random.normal(
        size=[target_batch_size, source_max_length, mdl_dim]).astype(np.float32)
    paddings = np.zeros([target_batch_size, source_max_length], dtype=np.int32)
    atten_mask = attentions.convert_paddings_to_mask(paddings, np.float32)

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      prng_key, init_key = jax.random.split(prng_key)
      initial_vars = layer.init(init_key, query_vec, key_vec, value_vec,
                                atten_mask)
      prng_key, init_key = jax.random.split(prng_key)
      initial_vars['params']['u'] = jax.random.uniform(
          shape=initial_vars['params']['u'].shape, key=init_key)
      prng_key, init_key = jax.random.split(prng_key)
      initial_vars['params']['v'] = jax.random.uniform(
          shape=initial_vars['params']['v'].shape, key=init_key)
      initial_vars['params']['pos_emb'] = {}

      jax_fprop_out, jax_atten_prob = layer.apply(initial_vars, query_vec,
                                                  key_vec, value_vec,
                                                  atten_mask)

    tf_initial_vars = py_utils.NestedMap.FromNestedDict(initial_vars['params'])
    tf_initial_vars.atten_dropout = None
    tf_layer_p = batch_major_attention.MultiHeadedAttentionXL.Params().Set(
        name='mh',
        input_dim=mdl_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        rel_pos_emb_dim=rel_pos_emb_dim)
    tf_layer = tf_layer_p.Instantiate()
    tf_out, tf_atten_prob = tf_layer.FProp(
        tf_initial_vars, query_vec, key_vec, value_vec, paddings=paddings)

    logging.info('jax_layer_out: %s', jax_fprop_out)
    logging.info('jax_atten_probs: %s', jax_atten_prob)
    logging.info('tf_layer_out: %s', tf_out)
    logging.info('tf_atten_probs: %s', tf_atten_prob)
    self.assertAllClose(
        test_utils.to_np(jax_fprop_out), test_utils.to_np(tf_out))
    self.assertAllClose(
        test_utils.to_np(jax_atten_prob), test_utils.to_np(tf_atten_prob))

  def text_attention_xl_step(self):
    mdl_dim = 16
    hidden_dim = 32
    num_heads = 4
    test_layer_p = DotProductAttentionXL.HParams(
        name='mh',
        input_dim=mdl_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        rel_pos_emb_dim=10,
        atten_logit_cap=20.0)
    layer = test_layer_p.Instantiate()
    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)
    target_batch_size = 3
    source_max_length = 8
    target_max_length = 8
    query_vec = np.random.normal(
        size=[target_batch_size, source_max_length, mdl_dim]).astype(np.float32)
    key_vec = query_vec
    value_vec = query_vec
    fake_query_vec = jnp.zeros_like(query_vec)
    atten_mask = attentions.causal_mask(query_vec)
    segment_pos = np.tile(np.arange(source_max_length), (target_batch_size, 1))

    starting_index = 0

    with base_layer.JaxContext.new_context():
      initial_vars = layer.init(
          init_key,
          fake_query_vec,
          fake_query_vec,
          fake_query_vec,
          atten_mask,
          query_segment_pos=segment_pos,
          key_segment_pos=segment_pos)
      logging.info('initial_vars: %s', initial_vars)
      _, attention_states = layer.apply(
          initial_vars,
          fake_query_vec,
          fake_query_vec,
          fake_query_vec,
          atten_mask,
          query_segment_pos=segment_pos,
          key_segment_pos=segment_pos,
          method=layer.__call__,
          mutable=[base_layer.DECODE_CACHE])
      fprop_out, _ = layer.apply(
          initial_vars,
          query_vec,
          key_vec,
          value_vec,
          atten_mask,
          query_segment_pos=segment_pos,
          key_segment_pos=segment_pos,
          method=layer.__call__)

      decoder_output = jnp.zeros(
          shape=[target_max_length, target_batch_size, mdl_dim])

      updated_vars = py_utils.MergeDictsWithValueCheck(attention_states,
                                                       initial_vars)
      for t in range(starting_index, target_max_length):
        encoded, attention_states = layer.apply(
            updated_vars,
            query_vec=query_vec[:, t, :],
            atten_mask=atten_mask[:, :, t, :],
            time_step=t,
            segment_pos=None,
            method=layer.extend_step,
            mutable=[base_layer.DECODE_CACHE])
        updated_vars = py_utils.MergeDictsWithValueCheck(
            attention_states, initial_vars)
        decoder_output = decoder_output.at[t].set(encoded)

    decoder_output = decoder_output[starting_index:]
    decoder_out_transposed = jnp.transpose(decoder_output, [1, 0, 2])
    fprop_out = fprop_out[:, starting_index:]

    logging.info('fprop_out: %s', fprop_out)
    logging.info('decoder_out: %s', decoder_output)
    self.assertAllClose(fprop_out, decoder_out_transposed)

  @parameterized.product(
      rel_pos_emb_dim=[10, 16],
      left_context=[1, 2],
      right_context=[0, 2],
  )
  def test_local_attention_xl(self, rel_pos_emb_dim, left_context,
                              right_context):
    mdl_dim = 16
    hidden_dim = 32
    num_heads = 4
    block_size = 4
    test_layer_p = attentions.LocalSelfAttentionXL.HParams(
        name='mh',
        input_dim=mdl_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        rel_pos_emb_dim=rel_pos_emb_dim,
        left_context=left_context,
        right_context=right_context,
        block_size=block_size,
    )
    layer = instantiate(test_layer_p)

    target_batch_size = 3
    source_max_length = 8

    query_vec = np.random.normal(
        size=[target_batch_size, source_max_length, mdl_dim]).astype(np.float32)
    key_vec = np.random.normal(
        size=[target_batch_size, source_max_length, mdl_dim]).astype(np.float32)
    value_vec = np.random.normal(
        size=[target_batch_size, source_max_length, mdl_dim]).astype(np.float32)
    paddings = np.zeros([target_batch_size, source_max_length],
                        dtype=np.float32)
    atten_mask = attentions.convert_paddings_to_mask(paddings, np.float32)

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      prng_key, init_key = jax.random.split(prng_key)
      initial_vars = layer.init(init_key, query_vec, key_vec, value_vec,
                                atten_mask)

      prng_key, init_key = jax.random.split(prng_key)
      initial_vars['params']['u'] = jax.random.uniform(
          shape=initial_vars['params']['u'].shape, key=init_key)
      prng_key, init_key = jax.random.split(prng_key)
      initial_vars['params']['v'] = jax.random.uniform(
          shape=initial_vars['params']['v'].shape, key=init_key)
      initial_vars['params']['pos_emb'] = {}
      jax_fprop_out, jax_atten_prob = layer.apply(initial_vars, query_vec,
                                                  key_vec, value_vec,
                                                  atten_mask)

    tf_initial_vars = py_utils.NestedMap.FromNestedDict(initial_vars['params'])
    tf_initial_vars.atten_dropout = None
    tf_layer_p = batch_major_attention.LocalSelfAttentionXL.Params().Set(
        name='mh',
        input_dim=mdl_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        rel_pos_emb_dim=rel_pos_emb_dim,
        block_size=block_size,
        left_context=left_context,
        right_context=right_context)
    tf_layer = tf_layer_p.Instantiate()
    tf_out, tf_atten_prob = tf_layer.FProp(
        tf_initial_vars, query_vec, key_vec, value_vec, paddings=paddings)

    logging.info('jax_layer_out: %s', jax_fprop_out)
    logging.info('jax_atten_probs: %s', jax_atten_prob)
    logging.info('tf_layer_out: %s', tf_out)
    logging.info('tf_atten_probs: %s', tf_atten_prob)
    self.assertAllClose(
        test_utils.to_np(jax_fprop_out), test_utils.to_np(tf_out))
    self.assertAllClose(
        test_utils.to_np(jax_atten_prob), test_utils.to_np(tf_atten_prob))

  @parameterized.parameters([(4, 2, 1, True), (4, 2, 1, False), (8, 3, 5, True),
                             (8, 3, 5, False), (5, 4, 0, False),
                             (5, 4, 0, True)])
  def test_local_attention(self, block_size, left_context, right_context,
                           is_full):
    mdl_dim = 16
    hidden_dim = 32
    num_heads = 4
    test_layer_p = attentions.LocalSelfAttention.HParams(
        name='mh',
        input_dim=mdl_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        block_size=block_size,
        left_context=left_context,
        right_context=right_context,
    )
    layer = instantiate(test_layer_p)

    target_batch_size = 3
    source_max_length = 16

    query_vec = np.random.normal(
        size=[target_batch_size, source_max_length, mdl_dim]).astype(np.float32)
    key_vec = np.random.normal(
        size=[target_batch_size, source_max_length, mdl_dim]).astype(np.float32)
    value_vec = np.random.normal(
        size=[target_batch_size, source_max_length, mdl_dim]).astype(np.float32)
    paddings = range(source_max_length)[-target_batch_size:]
    paddings = [[0] * l + [1] * (source_max_length - l) for l in paddings]
    paddings = np.array(paddings)
    atten_mask = attentions.convert_paddings_to_mask(paddings, np.float32)
    if is_full:
      atten_mask = jnp.tile(atten_mask, [1, 1, source_max_length, 1])

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      prng_key, init_key = jax.random.split(prng_key)
      initial_vars = layer.init(init_key, query_vec, key_vec, value_vec,
                                atten_mask)
      jax_fprop_out, jax_atten_prob = layer.apply(initial_vars, query_vec,
                                                  key_vec, value_vec,
                                                  atten_mask)

    tf_initial_vars = py_utils.NestedMap.FromNestedDict(initial_vars['params'])
    tf_initial_vars.atten_dropout = None
    tf_layer_p = batch_major_attention.LocalSelfAttention.Params().Set(
        name='mh',
        input_dim=mdl_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        block_size=block_size,
        left_context=left_context,
        right_context=right_context)
    tf_layer = tf_layer_p.Instantiate()
    tf_out, tf_atten_prob = tf_layer.FProp(
        tf_initial_vars, query_vec, key_vec, value_vec, paddings=paddings * 1.0)

    logging.info('jax_layer_out: %s', jax_fprop_out)
    logging.info('jax_atten_probs: %s', jax_atten_prob)
    logging.info('tf_layer_out: %s', tf_out)
    logging.info('tf_atten_probs: %s', tf_atten_prob)
    self.assertAllClose(
        test_utils.to_np(jax_fprop_out), test_utils.to_np(tf_out))
    self.assertAllClose(
        test_utils.to_np(jax_atten_prob), test_utils.to_np(tf_atten_prob))

  @parameterized.parameters(
      ([1, 2, 3, 4, 5], 1, 0, [0, 1, 2, 3, 4]),
      ([1, 2, 3, 4, 5], -1, 0, [2, 3, 4, 5, 0]),
      ([1, 2, 3, 4, 5], 2, 0, [0, 0, 1, 2, 3]),
      ([1, 2, 3, 4, 5], -2, 0, [3, 4, 5, 0, 0]),
      ([[1, 2, 3, 4], [6, 7, 8, 9]], 1, 0, [[0, 0, 0, 0], [1, 2, 3, 4]]),
      ([[1, 2, 3, 4], [6, 7, 8, 9]], -1, 0, [[6, 7, 8, 9], [0, 0, 0, 0]]),
      ([[1, 2, 3, 4], [6, 7, 8, 9]], 1, 1, [[0, 1, 2, 3], [0, 6, 7, 8]]),
      ([[1, 2, 3, 4], [6, 7, 8, 9]], -1, 1, [[2, 3, 4, 0], [7, 8, 9, 0]]),
      ([1], 1, 0, [0]),
  )
  def test_shift1d(self, inputs, offset, axis, outputs):
    inputs = np.asarray(inputs)
    shift_outputs = attentions.shift_1d(inputs, offset, axis)
    self.assertArraysEqual(shift_outputs, np.asarray(outputs))

  @parameterized.parameters(
      ([8, 16, 32], 3, 1, 32),
      ([8, 8, 4, 34], 2, 0, [4, 34]),
      ([2, 32, 8, 16, 128], 3, 1, [8, 16, 128]),
  )
  def test_causal_depthwise_conv1d(self, shape, kernel_size, axis, hidden_dims):
    inputs = np.random.normal(1.5, 2.0, shape).astype(np.float32)
    p = attentions.CausalDepthwiseConv1D.HParams(
        name='causal_dconv', kernel_size=kernel_size, hidden_dims=hidden_dims)
    causal_dconv_layer = instantiate(p)
    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)
    initial_vars = causal_dconv_layer.init(init_key, inputs, axis)
    if isinstance(hidden_dims, (list, tuple)):
      kernel_shape = hidden_dims
    else:
      kernel_shape = [hidden_dims]
    for k in range(kernel_size):
      initial_vars['params'][f'dconv_{k}'] = jnp.ones(kernel_shape)
    jax_dconv_out = causal_dconv_layer.apply(initial_vars, inputs, axis)
    jax_np_out = test_utils.to_np(jax_dconv_out)
    outputs = inputs
    for _ in range(1, kernel_size):
      inputs = attentions.shift_1d(inputs, offset=1, axis=axis)
      outputs += inputs
    self.assertArraysEqual(jax_np_out, outputs)

  @parameterized.parameters(
      ([8, 16, 32], 3, 1, 32),
      ([8, 8, 4, 34], 2, 0, [4, 34]),
      ([2, 32, 8, 16, 128], 3, 1, [8, 16, 128]),
  )
  def test_causal_depthwise_conv1d_extend_step(self, shape, kernel_size, axis,
                                               hidden_dims):
    inputs = np.random.normal(1.5, 2.0, shape).astype(np.float32)
    p = attentions.CausalDepthwiseConv1D.HParams(
        name='causal_dconv', kernel_size=kernel_size, hidden_dims=hidden_dims)
    causal_dconv_layer = instantiate(p)
    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)
    initial_vars = causal_dconv_layer.init(init_key, inputs, axis)
    jax_dconv_out = causal_dconv_layer.apply(
        initial_vars, inputs, axis, method=causal_dconv_layer.__call__)
    jax_np_out = test_utils.to_np(jax_dconv_out)
    jax_extend_step_out = jnp.zeros_like(jax_dconv_out)
    for i in range(shape[1]):
      jax_extend_step_out = causal_dconv_layer.apply(
          initial_vars,
          inputs,
          method=causal_dconv_layer.extend_step,
          axis=axis,
          step=i,
          segment_pos=None)
      jax_np_extend_step_out = test_utils.to_np(jax_extend_step_out)
      jax_extend_step_out_tensor = causal_dconv_layer.apply(
          initial_vars,
          inputs,
          method=causal_dconv_layer.extend_step,
          axis=axis,
          step=jnp.array(i),
          segment_pos=None)
      jax_np_extend_step_out_tensor = test_utils.to_np(
          jax_extend_step_out_tensor)
      jax_fprop_slice = jax.lax.dynamic_slice_in_dim(
          jax_np_out, start_index=i, slice_size=1, axis=axis)
      jax_fprop_slice = jnp.squeeze(jax_fprop_slice, axis)
      self.assertArraysEqual(jax_fprop_slice, jax_np_extend_step_out)
      self.assertArraysEqual(jax_fprop_slice, jax_np_extend_step_out_tensor)

  @parameterized.parameters([(32, 128), (2, 8), (8, 32)])
  def test_attention_with_relative_bias(self, num_buckets, max_distance):
    mdl_dim = 16
    hidden_dim = 32
    num_heads = 4
    test_layer_p = attentions.DotProductAttention.HParams(
        name='relative_attn',
        input_dim=mdl_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        relative_bias_tpl=attentions.RelativeBias.HParams(
            relative_attention_num_buckets=num_buckets,
            relative_attention_max_distance=max_distance))
    layer = instantiate(test_layer_p)
    target_batch_size = 3
    source_max_length = 16
    query_vec = np.random.normal(
        size=[target_batch_size, source_max_length, mdl_dim]).astype(np.float32)
    key_vec = query_vec
    value_vec = query_vec
    segment_pos = np.random.randint(
        0, source_max_length,
        [target_batch_size, source_max_length]).astype('int32')
    atten_mask = attentions.causal_mask(query_vec)

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      prng_key, init_key = jax.random.split(prng_key)
      initial_vars = layer.init(init_key, query_vec, key_vec, value_vec,
                                atten_mask, segment_pos)
      atten_output, _ = layer.apply(initial_vars, query_vec, key_vec, value_vec,
                                    atten_mask, segment_pos)

    self.assertEqual(atten_output.shape,
                     (target_batch_size, source_max_length, mdl_dim))

  def test_relative_bias_bidirectional_bucket(self):
    num_buckets = 8
    max_distance = 8
    bias_p = attentions.RelativeBias.HParams(
        relative_attention_num_buckets=num_buckets,
        relative_attention_max_distance=max_distance,
        bidirectional=True)
    bias_p.name = 'bias_layer'
    bias_layer = bias_p.Instantiate()
    relative_position = np.asarray([5, 2, 0, 12, -3, -1, -15, 1])
    buckets = bias_layer._relative_position_bucket(relative_position)
    buckets = np.asarray(buckets, np.int32)

    self.assertAllClose(buckets, [7, 6, 0, 7, 2, 1, 3, 5])

  def test_relative_bias_unidirectional_bucket(self):
    num_buckets = 4
    max_distance = 8
    bias_p = attentions.RelativeBias.HParams(
        relative_attention_num_buckets=num_buckets,
        relative_attention_max_distance=max_distance,
        bidirectional=False)
    bias_p.name = 'bias_layer'
    bias_layer = bias_p.Instantiate()
    relative_position = np.asarray([5, 2, 0, 12, -3, -1, -15])
    buckets = bias_layer._relative_position_bucket(relative_position)
    buckets = np.asarray(buckets, np.int32)

    self.assertAllClose(buckets, [0, 0, 0, 0, 2, 1, 3])

  @parameterized.parameters([(32, 128), (2, 8), (8, 32)])
  def test_attention_with_relative_bias_extend_step(self, num_buckets,
                                                    max_distance):
    mdl_dim = 16
    hidden_dim = 32
    num_heads = 4
    test_layer_p = attentions.DotProductAttention.HParams(
        name='relative_attn',
        input_dim=mdl_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        relative_bias_tpl=attentions.RelativeBias.HParams(
            relative_attention_num_buckets=num_buckets,
            relative_attention_max_distance=max_distance))
    layer = instantiate(test_layer_p)
    target_batch_size = 2
    source_max_length = 8
    inputs = np.random.normal(
        size=[target_batch_size, source_max_length, mdl_dim]).astype(np.float32)
    fake_input = jnp.zeros_like(inputs)
    atten_mask = attentions.causal_mask(inputs)
    segment_pos = np.random.randint(
        0, source_max_length,
        [target_batch_size, source_max_length]).astype('int32')

    time_step = 2

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      prng_key, init_key = jax.random.split(prng_key)
      initial_vars = layer.init(init_key, fake_input, fake_input, fake_input,
                                jnp.ones_like(atten_mask), segment_pos)
      _, attention_states = layer.apply(
          initial_vars,
          fake_input,
          fake_input,
          fake_input,
          jnp.ones_like(atten_mask),
          segment_pos,
          method=layer.__call__,
          mutable=[base_layer.DECODE_CACHE])
      initial_vars = py_utils.MergeDictsWithValueCheck(attention_states,
                                                       initial_vars)
      atten_output, _ = layer.apply(
          initial_vars,
          inputs[:, time_step, :],
          atten_mask=atten_mask[:, :, time_step, :],
          time_step=time_step,
          segment_pos=None,
          method=layer.extend_step,
          mutable=[base_layer.DECODE_CACHE])

    self.assertEqual(atten_output.shape, (target_batch_size, mdl_dim))

  @parameterized.parameters([(32, 128), (2, 8), (8, 32)])
  def test_relative_bias_layer(self, num_buckets, max_distance):
    num_heads = 4
    test_layer_p = attentions.RelativeBias.HParams(
        name='relative_bias',
        relative_attention_num_buckets=num_buckets,
        relative_attention_max_distance=max_distance,
        num_heads=num_heads)
    test_layer_p.use_length_as_position = False
    layer_raw = instantiate(test_layer_p)
    test_layer_p.use_length_as_position = True
    layer_len = instantiate(test_layer_p)
    target_batch_size = 3
    source_max_length = 8
    segment_ids = np.array([
        [0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 1, 1],
    ],
                           dtype=jnp.int32)
    segment_pos = np.array([
        [0, 1, 2, 3, 4, 0, 1, 2],
        [0, 1, 2, 3, 0, 1, 2, 3],
        [0, 1, 2, 0, 1, 2, 3, 4],
    ],
                           dtype=jnp.int32)
    segment_mask = attentions.segment_mask(segment_ids) == 0
    logging.info('segment mask: %s', segment_mask)

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      prng_key, init_key = jax.random.split(prng_key)
      initial_vars = layer_raw.init(init_key, segment_pos, segment_pos)
      rb_raw = layer_raw.apply(initial_vars, segment_pos, segment_pos)
      rb_len = layer_len.apply(initial_vars, segment_pos, segment_pos)

    # Test shape.
    self.assertEqual(
        rb_raw.shape,
        (target_batch_size, num_heads, source_max_length, source_max_length))
    self.assertEqual(rb_len.shape,
                     (1, num_heads, source_max_length, source_max_length))
    # Test value.
    self.assertAllClose(
        segment_mask * rb_raw,
        segment_mask * np.tile(rb_len, [target_batch_size, 1, 1, 1]))

  @parameterized.parameters(
      (5, 15, None, 0),
      (4, 10, None, 3),
      (8, 8, 3, None),
      (9, 6, 3, 3),
  )
  def test_limited_context_mask(self, batch_size, max_length,
                                left_context, right_context):

    def get_padding_from_length(length):
      idx = np.tile(np.arange(max_length), [batch_size, 1])
      return (idx >= np.expand_dims(length, -1)).astype('float32')

    length = np.random.randint(max_length // 2, max_length, [
        batch_size,
    ])
    padding = jnp.asarray(get_padding_from_length(length))

    mask = attentions.limited_context_mask(left_context, right_context,
                                           padding.shape[1], np.float32)

    # Merge the above mask with paddings:
    padding_mask = attentions.convert_paddings_to_mask(padding)
    rev_padding_mask = jnp.transpose(padding_mask, (0, 1, 3, 2))
    result = jnp.minimum(jnp.minimum(mask, padding_mask), rev_padding_mask)

    expect = np.zeros((batch_size, 1, max_length, max_length))
    for b in range(batch_size):
      for t1 in range(max_length):
        if t1 >= length[b]:
          continue
        start_p, end_p = 0, length[b]
        if left_context is not None:
          start_p = max(0, t1 - left_context + 1)
        if right_context is not None:
          end_p = min(length[b], t1 + right_context + 1)
        expect[b, 0, t1, start_p:end_p] = 1.0
    self.assertAllClose(
        test_utils.to_np(result),
        (1.0 - expect) * py_utils.get_large_negative_number(jnp.float32))

  def test_combine_qkv_with_attention_combine_dims(self):
    input_dim = 64
    dim_per_head = 8
    num_heads = 8
    # Reference combine qkv projection layer.
    ref_proj_p = attentions.CombinedQKVProjectionLayer.HParams(
        name='ref',
        input_dim=input_dim,
        dim_per_head=dim_per_head,
        num_heads=num_heads)
    proj = instantiate(ref_proj_p)

    # Combine attention dim combine qkv projection layer.
    combine_proj_p = attentions.CombinedQKVProjectionLayer.HParams(
        name='ref',
        input_dim=input_dim,
        dim_per_head=dim_per_head,
        num_heads=num_heads,
        attention_combine_dims=True)
    combine_proj = instantiate(combine_proj_p)

    batch_size = 3
    inputs = np.random.normal(size=[batch_size, input_dim]).astype(np.float32)

    with base_layer.JaxContext.new_context():
      # Set up initial vars for combine attention dim projection.
      prng_key = jax.random.PRNGKey(seed=123)
      prng_key, init_key = jax.random.split(prng_key)
      initial_vars = proj.init(init_key, inputs)

      combine_initial_vars = combine_proj.init(init_key, inputs)
      combine_initial_vars['params']['w'] = np.reshape(
          initial_vars['params']['w'], (3, input_dim, num_heads * dim_per_head))
      combine_initial_vars['params']['b'] = np.reshape(
          initial_vars['params']['b'], (3, num_heads * dim_per_head))

      q_proj_ref, k_proj_ref, v_proj_ref = proj.apply(initial_vars, inputs)
      q_proj_combine, k_proj_combine, v_proj_combine = combine_proj.apply(
          combine_initial_vars, inputs)

    self.assertAllClose(q_proj_ref, q_proj_combine)
    self.assertAllClose(k_proj_ref, k_proj_combine)
    self.assertAllClose(v_proj_ref, v_proj_combine)

  @parameterized.parameters([(False, True, 3, True), (True, True, 3, True),
                             (False, True, 4, False), (True, True, 4, True),
                             (False, False, 1, False), (True, False, 1, True),
                             (False, False, 1, True), (True, False, 1, True)])
  def test_mha_with_fprop_update_state(self, combine_qkv, dconv_qkv,
                                       dconv_kernel_size,
                                       use_rotary_position_emb):
    mdl_dim = 16
    hidden_dim = 32
    num_heads = 4
    test_layer_p = attentions.DotProductAttention.config(
        name='mh',
        input_dim=mdl_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dim_per_head=16 if use_rotary_position_emb else None,
        atten_logit_cap=20.0,
        combine_qkv=combine_qkv,
        dconv_qkv=dconv_qkv,
        dconv_kernel_size=dconv_kernel_size,
        use_rotary_position_emb=use_rotary_position_emb)
    layer = instantiate(test_layer_p)
    target_batch_size = 3
    source_max_length = 16
    target_max_length = 16
    prefix_len = 8
    query_vec = np.random.normal(
        size=[target_batch_size, source_max_length, mdl_dim]).astype(np.float32)
    prefix = jnp.zeros_like(query_vec)
    prefix = jax.lax.dynamic_update_slice(prefix, query_vec[:, 0:prefix_len, :],
                                          [0, 0, 0])
    key_vec = query_vec
    value_vec = query_vec
    atten_mask = attentions.causal_mask(query_vec)

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      prng_key, init_key = jax.random.split(prng_key)
      initial_vars = layer.init(init_key, query_vec, key_vec, value_vec,
                                atten_mask)
      logging.info('initial_vars: %s', initial_vars)
      fprop_out, _ = layer.apply(
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
          atten_mask,
          method=layer.__call__,
          mutable=[base_layer.DECODE_CACHE])

      updated_vars = py_utils.MergeDictsWithValueCheck(attention_states,
                                                       initial_vars)
      # Call extend step start from prefix_len.
      for t in range(prefix_len, target_max_length):
        encoded, attention_states = layer.apply(
            updated_vars,
            query_vec=query_vec[:, t, :],
            atten_mask=atten_mask[:, :, t, :],
            time_step=t,
            segment_pos=None,
            method=layer.extend_step,
            mutable=[base_layer.DECODE_CACHE])
        updated_vars = py_utils.MergeDictsWithValueCheck(
            attention_states, initial_vars)
        logging.info('encoded: %s', encoded)
        logging.info('fprop_out[:, t, :]: %s', fprop_out[:, t, :])
        self.assertAllClose(fprop_out[:, t, :], encoded)

  @parameterized.parameters([(False, True, 3, True), (True, True, 3, True),
                             (False, True, 4, False), (True, True, 4, True),
                             (False, False, 1, False), (True, False, 1, True),
                             (False, False, 1, True), (True, False, 1, True)])
  def test_mha_with_lazy_broadcast_state(self, combine_qkv, dconv_qkv,
                                         dconv_kernel_size,
                                         use_rotary_position_emb):
    mdl_dim = 4
    hidden_dim = 8
    num_heads = 2
    test_layer_p = attentions.DotProductAttentionWithLPB.config(
        name='mh',
        input_dim=mdl_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dim_per_head=4 if use_rotary_position_emb else None,
        atten_logit_cap=20.0,
        combine_qkv=combine_qkv,
        dconv_qkv=dconv_qkv,
        dconv_kernel_size=dconv_kernel_size,
        use_rotary_position_emb=use_rotary_position_emb)
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
      updated_vars = py_utils.MergeDictsWithValueCheck(attention_states,
                                                       initial_vars)

      # First lazy broadcast.
      _, attention_states = layer.apply(
          updated_vars,
          num_suffix_samples=2,
          suffix_length=suffix_1_len,
          method=layer.lazy_broadcast_prefix,
          mutable=[base_layer.DECODE_CACHE, base_layer.PREFIX_DECODE_CACHE])
      updated_vars = py_utils.MergeDictsWithValueCheck(attention_states,
                                                       initial_vars)

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
        updated_vars = py_utils.MergeDictsWithValueCheck(
            attention_states, updated_vars)
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
      updated_vars = py_utils.MergeDictsWithValueCheck(attention_states,
                                                       initial_vars)

      # Call extend step start from prefix_len + suffix_1_len.
      for t in range(prefix_len + suffix_1_len, prefix_len + suffix_2_len):
        encoded, attention_states = layer.apply(
            updated_vars,
            query_vec=_broadcast_sample(query_vec[:, t, :], 6),
            atten_mask=atten_mask[:, :, t, :],
            time_step=t,
            segment_pos=None,
            method=layer.extend_step,
            mutable=[base_layer.DECODE_CACHE])
        del updated_vars[base_layer.DECODE_CACHE]
        updated_vars = py_utils.MergeDictsWithValueCheck(
            attention_states, updated_vars)
        encoded = jnp.reshape(encoded, (-1, 6) + encoded.shape[1:])
        for sample_id in range(6):
          self.assertAllClose(fprop_out[:, t, :], encoded[:, sample_id])

  @parameterized.parameters(*list(itertools.product([True, False], repeat=2)))
  def test_mha_extend_n_steps_with_lazy_broadcast_state(
      self, combine_qkv, use_rotary_position_emb):
    mdl_dim = 4
    hidden_dim = 8
    num_heads = 2

    test_layer_p = attentions.DotProductAttentionWithLPB.config(
        name='mh',
        input_dim=mdl_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dim_per_head=4 if use_rotary_position_emb else None,
        atten_logit_cap=20.0,
        combine_qkv=combine_qkv,
        dconv_qkv=False,
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
      updated_vars = py_utils.MergeDictsWithValueCheck(attention_states,
                                                       initial_vars)

      # First lazy broadcast.
      _, attention_states = layer.apply(
          updated_vars,
          num_suffix_samples=num_samples,
          suffix_length=source_max_length - prefix_len,
          method=layer.lazy_broadcast_prefix,
          mutable=[base_layer.DECODE_CACHE, base_layer.PREFIX_DECODE_CACHE])
      updated_vars = py_utils.MergeDictsWithValueCheck(attention_states,
                                                       initial_vars)

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

  def test_right_align_decode_state(self):
    mdl_dim = 4
    hidden_dim = 8
    num_heads = 2
    test_layer_p = attentions.DotProductAttentionWithLPB.config(
        name='mh',
        input_dim=mdl_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dim_per_head=None)
    layer = instantiate(test_layer_p)
    target_batch_size = 3
    source_max_length = 8
    prefix_len = 4
    query_vec = np.random.normal(
        size=[target_batch_size, source_max_length, mdl_dim]).astype(np.float32)
    prefix = query_vec[:, 0:prefix_len, :]
    key_vec = query_vec
    value_vec = query_vec
    atten_mask = attentions.causal_mask(query_vec)
    prefix_atten_mask = attentions.causal_mask(prefix)
    suffix_len = 8

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      prng_key, init_key = jax.random.split(prng_key)
      initial_vars = layer.init(init_key, query_vec, key_vec, value_vec,
                                atten_mask)
      # Updates decode states in fprop.
      _, attention_states = layer.apply(
          initial_vars,
          prefix,
          prefix,
          prefix,
          prefix_atten_mask,
          method=layer.__call__,
          mutable=[base_layer.DECODE_CACHE])
      logging.info('attention_states: %s', attention_states)

      updated_vars = py_utils.MergeDictsWithValueCheck(attention_states,
                                                       initial_vars)

      def _set_state(value):
        """Sets decode state as the value."""

        def set_state_fn(x, batch_dim, time_dim):
          del x, batch_dim, time_dim
          return value

        return set_state_fn

      _, attention_states = layer.apply(
          updated_vars,
          _set_state(jnp.array([[0, 0, 0, 1], [0, 0, 1, 1]], dtype=jnp.int32)),
          method=layer.transform_decode_state,
          mutable=[base_layer.DECODE_CACHE])

      updated_vars = py_utils.MergeDictsWithValueCheck(attention_states,
                                                       initial_vars)

      # Second lazy broadcast.
      _, attention_states = layer.apply(
          updated_vars,
          num_suffix_samples=2,
          suffix_length=suffix_len,
          method=layer.lazy_broadcast_prefix,
          mutable=[base_layer.DECODE_CACHE, base_layer.PREFIX_DECODE_CACHE])
      updated_vars = py_utils.MergeDictsWithValueCheck(attention_states,
                                                       initial_vars)

      _, attention_states = layer.apply(
          updated_vars,
          _set_state(jnp.ones([2, 2, 8], dtype=jnp.int32) * 2),
          method=layer.transform_decode_state,
          mutable=[base_layer.DECODE_CACHE, base_layer.PREFIX_DECODE_CACHE])

      logging.info('attention_states: %s', attention_states)
      updated_vars = py_utils.MergeDictsWithValueCheck(attention_states,
                                                       initial_vars)

      max_prefix_len = 4
      decode_lengths = np.array([2, 3, 4, 5])
      prefix_lengths = np.array([1, 1, 2, 2])

      # Before transform, prefix_decode_state has:
      # [[0, 0, 0, 1],
      #  [0, 0, 1, 1]]
      # decode state has:
      # [[[2, 2, 2, 2, 2, 2, 2, 2],
      #   [2, 2, 2, 2, 2, 2, 2, 2]],
      #  [[2, 2, 2, 2, 2, 2, 2, 2],
      #   [2, 2, 2, 2, 2, 2, 2, 2]]]
      _, attention_states = layer.apply(
          updated_vars,
          max_prefix_len,
          decoder_utils.right_align_state_fn(max_prefix_len + decode_lengths -
                                             prefix_lengths),
          method=layer.right_align_decode_state_with_prefix,
          mutable=[base_layer.DECODE_CACHE, base_layer.PREFIX_DECODE_CACHE])

      logging.info('attention_states: %s', attention_states)
      self.assertArraysEqual(
          attention_states[base_layer.DECODE_CACHE]['key_state'],
          np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2]],
                    [[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2],
                     [0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2]]], jnp.int32))
      self.assertArraysEqual(
          attention_states[base_layer.PREFIX_DECODE_CACHE]['key_state_0_pfx'],
          np.array([[0], [0]], jnp.int32))

  def test_no_attention_decode_state(self):
    mdl_dim = 16
    hidden_dim = 32
    num_heads = 4
    test_layer_p = attentions.DotProductAttention.HParams(
        name='mh',
        input_dim=mdl_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dim_per_head=16,
        decode_cache=False)
    layer = instantiate(test_layer_p)
    target_batch_size = 3
    source_max_length = 16
    query_vec = np.random.normal(
        size=[target_batch_size, source_max_length, mdl_dim]).astype(np.float32)
    fake_query_vec = jnp.zeros_like(query_vec)
    atten_mask = attentions.causal_mask(query_vec)
    segment_pos = np.tile(np.arange(source_max_length), (target_batch_size, 1))

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      prng_key, init_key = jax.random.split(prng_key)
      initial_vars = layer.init(
          init_key,
          fake_query_vec,
          fake_query_vec,
          fake_query_vec,
          atten_mask,
          query_segment_pos=segment_pos,
          key_segment_pos=segment_pos)
      logging.info('initial_vars: %s', initial_vars)
      _, attention_states = layer.apply(
          initial_vars,
          fake_query_vec,
          fake_query_vec,
          fake_query_vec,
          atten_mask,
          query_segment_pos=segment_pos,
          key_segment_pos=segment_pos,
          method=layer.__call__,
          mutable=[base_layer.DECODE_CACHE])
    logging.info('attention_states: %s', attention_states)
    # Makes sure there is no decoder state.
    self.assertEqual(attention_states, {})


if __name__ == '__main__':
  absltest.main()
