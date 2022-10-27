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

"""Tests for conformers."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from lingvo.core import cluster_factory
from lingvo.core import conformer_layer
import numpy as np
from praxis import base_layer
from praxis import py_utils
from praxis import test_utils
from praxis.layers import attentions
from praxis.layers import conformers
import tensorflow as tf

instantiate = base_layer.instantiate
to_np = test_utils.to_np
NestedMap = py_utils.NestedMap


class ConformerTest(test_utils.TestCase):

  @parameterized.parameters(
      (2, 10, 3, 8, 8, 4, 0.0),
      (3, 12, 5, 16, 16, 2, 0.1),
      (5, 7, 2, 8, 8, 8, 0.25),
      (7, 8, 4, 16, 16, 4, 0.5),
  )
  def test_conformer_layer(self, batch_size, seq_len, kernel_size, input_dims,
                           model_dims, atten_num_heads, dropout_prob):
    # Lingvo TF layers only use dropout on FF and Attention layers
    p = conformers.Conformer.HParams(
        name='jax_conformer_layer',
        input_dims=input_dims,
        conv_residual_dropout=0.0,
        atten_residual_dropout=dropout_prob,
        ffn_residual_dropout=dropout_prob,
        atten_dropout=dropout_prob,
        ffn_relu_dropout=dropout_prob,
        kernel_size=kernel_size,
        model_dims=model_dims,
        atten_num_heads=atten_num_heads)
    conformer = instantiate(p)
    npy_inputs = np.random.normal(
        1.0, 0.5, [batch_size, seq_len, input_dims]).astype('float32')
    inputs = jnp.asarray(npy_inputs)

    def get_padding_from_length(length):
      idx = np.tile(np.arange(seq_len), [batch_size, 1])
      return (idx >= np.expand_dims(length, -1)).astype('float32')

    length = np.random.randint(seq_len // 2, seq_len, (batch_size,))
    npy_paddings = get_padding_from_length(length)
    paddings = jnp.asarray(npy_paddings)

    context_p = base_layer.JaxContext.HParams(do_eval=True)

    with base_layer.JaxContext.new_context(hparams=context_p):
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = conformer.init(prng_key, inputs, paddings)
      output = conformer.apply(initial_vars, inputs, paddings)
    # Test whether tf Conformer layer returns the same output
    # Modify initial_vars to use TF compatible params
    initial_vars_trainable = py_utils.NestedMap.FromNestedDict(
        initial_vars['params'])
    initial_vars_non_trainable = py_utils.NestedMap.FromNestedDict(
        initial_vars['non_trainable'])
    tf_initial_vars = test_utils.replace_jax_conformer_layer_vars_to_tf(
        initial_vars_trainable, initial_vars_non_trainable)

    tf_p = conformer_layer.ConformerLayer.CommonParams(
        input_dim=input_dims,
        dropout_prob=dropout_prob,
        atten_num_heads=atten_num_heads,
        kernel_size=kernel_size,
        fflayer_hidden_dim=model_dims * p.ffn_dim_multiplier,
        use_relative_atten=False,
        fflayer_residual_weight=0.5).Set(name='tf_conformer')
    tf_p.trans_atten_tpl = tf_p.trans_atten_tpl.Set(hidden_dim=model_dims)

    tf_conformer = tf_p.Instantiate()
    with cluster_factory.SetEval(True):
      tf_output = tf_conformer.FProp(
          tf_initial_vars,
          py_utils.NestedMap(
              features=tf.constant(inputs, dtype=tf.float32),
              paddings=tf.constant(npy_paddings, dtype=tf.float32)))
    np_output = to_np(output)
    tf_np_output = to_np(tf_output.features)
    self.assertAllClose(tf_np_output, np_output, atol=1e-5)

  @parameterized.parameters([
      (10, None, None),
      (16, 2, None),
      (16, None, 2),
      (10, 0, None),
      (10, None, 0),
      (10, 2, 2),
      (10, 2, 0),
  ])
  def test_local_global_attention_xl(self, rel_pos_emb_dim, left_context,
                                     right_context):
    mdl_dim = 16
    hidden_dim = 32
    num_heads = 4

    # Layer which can do both local emulated and global self attention.
    local_layer_p = conformers.DotProductAttentionWithContextXL.HParams(
        name='local_mh',
        input_dim=mdl_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        rel_pos_emb_dim=rel_pos_emb_dim,
        left_context=left_context,
        right_context=right_context
    )

    # Layer which can do only global attention.
    global_layer_p = attentions.DotProductAttentionXL.HParams(
        name='global_mh',
        input_dim=mdl_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        rel_pos_emb_dim=rel_pos_emb_dim
    )

    # Prepare input data.
    target_batch_size = 3
    source_max_length = 8
    query_vec = np.random.normal(
        size=[target_batch_size, source_max_length, mdl_dim]).astype(np.float32)
    key_vec = np.random.normal(
        size=[target_batch_size, source_max_length, mdl_dim]).astype(np.float32)
    value_vec = np.random.normal(
        size=[target_batch_size, source_max_length, mdl_dim]).astype(np.float32)
    def get_padding_from_length(length):
      idx = np.tile(np.arange(source_max_length), [target_batch_size, 1])
      return np.asarray(idx >= np.expand_dims(length, -1)).astype('float32')
    length = np.random.randint(source_max_length // 2, source_max_length,
                               (target_batch_size,))
    paddings = get_padding_from_length(length)

    # Convert paddings to atten_mask:
    atten_mask_padding = attentions.convert_paddings_to_mask(
        paddings, np.float32)
    rev_padding_mask = jnp.transpose(atten_mask_padding, (0, 1, 3, 2))
    atten_mask_padding = jnp.minimum(atten_mask_padding, rev_padding_mask)

    # Emulate local attention for DotProductAttentionXL.
    context_mask = attentions.limited_context_mask(left_context, right_context,
                                                   paddings.shape[1])
    local_atten_mask_padding = jnp.minimum(atten_mask_padding, context_mask)

    # Run DotProductAttentionWithContextXL which will compute its own local mask
    # vs DotProductAttentionXL which uses local_atten_mask_padding.
    global_layer = instantiate(global_layer_p)
    local_layer = instantiate(local_layer_p)
    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      prng_key, init_key = jax.random.split(prng_key)
      initial_vars = global_layer.init(init_key, query_vec, key_vec, value_vec,
                                       atten_mask_padding)
      out_local = local_layer.apply(
          initial_vars,
          query_vec=query_vec,
          key_vec=key_vec,
          value_vec=value_vec,
          atten_mask=atten_mask_padding)[0]
      out_global = global_layer.apply(
          initial_vars,
          query_vec=query_vec,
          key_vec=key_vec,
          value_vec=value_vec,
          atten_mask=local_atten_mask_padding)[0]
      self.assertAllClose(out_local, out_global, atol=1e-5)

  @parameterized.parameters([(10, 1, 1), (10, 2, 0)])
  def test_local_attention_xl(self, rel_pos_emb_dim, left_context,
                              right_context):
    mdl_dim = 16
    hidden_dim = 32
    num_heads = 4

    # Layer which can do only local self attention.
    local_layer_p = attentions.LocalSelfAttentionXL.HParams(
        name='local_mh',
        input_dim=mdl_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        rel_pos_emb_dim=rel_pos_emb_dim,
        left_context=left_context,
        right_context=right_context,
    )

    # Layer which can do only global attention.
    global_layer_p = attentions.DotProductAttentionXL.HParams(
        name='global_mh',
        input_dim=mdl_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        rel_pos_emb_dim=rel_pos_emb_dim
    )

    # Prepare input data.
    target_batch_size = 3
    source_max_length = 8
    query_vec = np.random.normal(
        size=[target_batch_size, source_max_length, mdl_dim]).astype(np.float32)
    key_vec = np.random.normal(
        size=[target_batch_size, source_max_length, mdl_dim]).astype(np.float32)
    value_vec = np.random.normal(
        size=[target_batch_size, source_max_length, mdl_dim]).astype(np.float32)
    def get_padding_from_length(length):
      idx = np.tile(np.arange(source_max_length), [target_batch_size, 1])
      return np.asarray(idx >= np.expand_dims(length, -1)).astype('float32')
    length = np.random.randint(source_max_length // 2, source_max_length,
                               (target_batch_size,))
    paddings = get_padding_from_length(length)

    # Convert paddings to atten_mask for LocalSelfAttentionXL:
    atten_mask_padding = attentions.convert_paddings_to_mask(
        paddings, np.float32)

    # Emulate local attention for DotProductAttentionXL.
    context_mask = attentions.limited_context_mask(left_context, right_context,
                                                   paddings.shape[1])
    rev_padding_mask = jnp.transpose(atten_mask_padding, (0, 1, 3, 2))
    dot_product_atten_mask_padding = jnp.minimum(atten_mask_padding,
                                                 rev_padding_mask)
    local_atten_mask_padding = jnp.minimum(dot_product_atten_mask_padding,
                                           context_mask)

    # Run LocalSelfAttentionXL which computes local self attention explicitly
    # vs DotProductAttentionXL which uses local_atten_mask_padding for emulation
    # of local self attention.
    global_layer = instantiate(global_layer_p)
    local_layer = instantiate(local_layer_p)
    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      prng_key, init_key = jax.random.split(prng_key)
      initial_vars = global_layer.init(init_key, query_vec, key_vec, value_vec,
                                       atten_mask_padding)
      out_local = local_layer.apply(
          initial_vars,
          query_vec=query_vec,
          key_vec=key_vec,
          value_vec=value_vec,
          atten_mask=atten_mask_padding)[0]
      out_global = global_layer.apply(
          initial_vars,
          query_vec=query_vec,
          key_vec=key_vec,
          value_vec=value_vec,
          atten_mask=local_atten_mask_padding)[0]

      # Attention mask is applied to prevent attention between unwanted pairs
      # but it does not apply paddings, so mask it out below:
      mask = 1.0 - paddings
      mask = np.reshape(mask,
                        mask.shape + (1,) * (out_local.ndim - mask.ndim))
      self.assertAllClose(out_local * mask, out_global * mask, atol=1e-5)


if __name__ == '__main__':
  absltest.main()
