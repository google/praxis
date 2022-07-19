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

"""Tests for ngrammer."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from lingvo.core import attention_util
import numpy as np
from praxis import base_layer
from praxis import py_utils
from praxis import test_utils
from praxis.layers import bregman
from praxis.layers import ngrammer
import tensorflow as tf

instantiate = base_layer.instantiate
NON_TRAINABLE = base_layer.NON_TRAINABLE
to_np = test_utils.to_np


class NgrammerTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  @parameterized.parameters(
      (10000),
      (1000),
      (320000),
      (500),
  )
  def test_get_bigram_ids(self, vocab_size):
    ids = np.random.randint(vocab_size, size=(2, 16), dtype=np.int64)
    ngram_ids = ngrammer.get_bigram_ids(ids, vocab_size)
    np_ngram_ids = to_np(ngram_ids)
    self.assertLess(np.max(np_ngram_ids), vocab_size**2)

  @parameterized.parameters(
      (10000),
      (1000),
      (320000),
      (500),
  )
  def test_get_bigram_ids_with_packing(self, vocab_size):
    ids = np.random.randint(vocab_size, size=(2, 8), dtype=np.int64)
    segment_pos = np.array([[0, 1, 2, 3, 0, 1, 2, 3], [0, 1, 2, 0, 1, 2, 3, 4]])
    ngram_ids = ngrammer.get_bigram_ids(ids, vocab_size, segment_pos)
    np_ngram_ids = to_np(ngram_ids)
    self.assertLess(np.max(np_ngram_ids), vocab_size**2)
    self.assertEqual(np_ngram_ids[0, 0], ids[0, 0])
    self.assertEqual(np_ngram_ids[1, 0], ids[1, 0])
    self.assertEqual(np_ngram_ids[0, 4], ids[0, 4])
    self.assertEqual(np_ngram_ids[1, 3], ids[1, 3])

  @parameterized.parameters(
      (10000),
      (1000),
      (320000),
      (500),
  )
  def test_get_bigram_ids_with_packing_and_pair_ids(self, vocab_size):
    ids = np.random.randint(vocab_size, size=(3, 8), dtype=np.int64)
    segment_pos = np.array([[0, 1, 2, 3, 0, 1, 2, 3], [0, 1, 2, 0, 1, 2, 3, 4],
                            [0, 1, 2, 3, 4, 5, 6, 7]])
    pair_ids = np.array([[8, 0, 1, 2, 3, 4, 5, 6]] * 3)
    ngram_ids = ngrammer.get_bigram_ids(ids, vocab_size, segment_pos)
    np_ngram_ids = to_np(ngram_ids)
    ngram_ids_pair_ids = ngrammer.get_bigram_ids(
        ids, vocab_size, segment_pos, pair_ids=pair_ids)
    np_ngram_ids_pair_ids = to_np(ngram_ids_pair_ids)
    self.assertArraysEqual(np_ngram_ids, np_ngram_ids_pair_ids)

  @parameterized.parameters(
      (16, 8, 32),
      (24, 4, 16),
      (32, 16, 8),
      (25, 2, 16),
  )
  def test_vq_layer_equivalence_with_tf(self, num_clusters, num_heads,
                                        dim_per_head):
    inputs = np.random.normal(1.5, 2.0, (2, 32, num_heads, dim_per_head))
    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)
    vq_layer_p = ngrammer.VectorQuantization.HParams(
        name='jax_vq_layer',
        num_clusters=num_clusters,
        num_heads=num_heads,
        dim_per_head=dim_per_head,
    )
    with base_layer.JaxContext.new_context():
      vq_layer = instantiate(vq_layer_p)
      initial_vars = vq_layer.init(init_key, inputs)
      jax_dists, _ = vq_layer.apply(initial_vars, inputs)
    # Now run TF based computation.
    tf_vq_layer_p = attention_util.KMeansClusteringForAtten.Params().Set(
        name='tf_vq_layer',
        num_clusters=num_clusters,
        num_heads=num_heads,
        dim_per_head=dim_per_head,
        apply_layer_norm=False)
    tf_vq_layer = tf_vq_layer_p.Instantiate()
    tf_initial_vars = py_utils.NestedMap.FromNestedDict(
        initial_vars['non_trainable'])
    tf_dists, _ = tf_vq_layer.FProp(tf_initial_vars, tf.constant(inputs))
    self.assertAllClose(to_np(jax_dists), to_np(tf_dists), atol=1e-5)

  @parameterized.parameters(
      (8, 32, 8),
      (4, 16, 4),
      (16, 8, 2),
      (2, 16, 1),
  )
  def test_bregman_compression(self, num_heads, dim_per_head, num_components):
    inputs = np.random.normal(1.5, 2.0, (2, 32, num_heads, dim_per_head))
    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)
    bregman_compression_layer_p = ngrammer.BregmanCompression.HParams(
        name='jax_bregman_compression_layer',
        num_heads=num_heads,
        dim_per_head=dim_per_head,
        num_components=num_components,
        activation_type=bregman.ActivationType.LEAKY_RELU,
        negative_slope=0.1,
        start_step=0,
        end_step=10,
        constant_lr_schedule=True)
    bregman_compression_layer_p = instantiate(bregman_compression_layer_p)
    with base_layer.JaxContext.new_context():
      initial_vars = bregman_compression_layer_p.init(init_key, inputs)
      coefficients = bregman_compression_layer_p.apply(initial_vars, inputs)
    self.assertEqual(
        list(coefficients.shape), [2, 32, num_heads, num_components])

  @parameterized.parameters(
      (16, 8, 2, 32, True),
      (24, 4, 4, 16, True),
      (32, 16, 1, 64, True),
      (25, 4, 2, 8, True),
      (16, 8, 2, 8, False),
      (24, 4, 4, 4, False),
      (32, 16, 1, 16, False),
      (25, 4, 2, 4, False),
  )
  def test_ngrammer_layer_exact_bigram(self, unigram_vocab_size, ngram_emb_dim,
                                       num_heads, dim_per_head, concat_ngrams):
    batch_size = 2
    seq_len = 8
    inputs = np.random.randint(
        unigram_vocab_size,
        size=[batch_size, seq_len, num_heads],
        dtype=np.int32)
    paddings = np.random.randint(1, size=[batch_size, seq_len])
    input_embs = np.random.normal(
        1.5, 2.0, (batch_size, seq_len, num_heads * dim_per_head))
    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)
    ngrammer_layer_p = ngrammer.Ngrammer.HParams(
        name='jax_ngrammer_layer',
        unigram_vocab_size=unigram_vocab_size,
        ngram_vocab_size=num_heads * unigram_vocab_size**2,
        ngram_emb_dim=ngram_emb_dim,
        num_heads=num_heads,
        dim_per_head=dim_per_head,
        concat_ngrams=concat_ngrams,
    )
    ngrammer_layer = instantiate(ngrammer_layer_p)
    with base_layer.JaxContext.new_context():
      initial_vars = ngrammer_layer.init(init_key, inputs, input_embs, paddings)
      # Bind makes ngrammer_layer a stateful layer with all its submodule
      # layer variables automatically bound. It simplfies calling of submodule
      # fprop because we no longer need to explicitly provide submodule layer
      # vars.
      ngrammer_layer = ngrammer_layer.bind(
          initial_vars, mutable=[base_layer.NON_TRAINABLE])
      ngram_embs = ngrammer_layer(inputs, input_embs, paddings)
    ngram_embs = np.reshape(ngram_embs,
                            [batch_size, seq_len, num_heads, dim_per_head])
    input_embs = np.reshape(input_embs,
                            [batch_size, seq_len, num_heads, dim_per_head])
    for i in range(num_heads):
      input_ids_per_head = inputs[:, :, i]
      ngram_ids_per_head = ngrammer.get_bigram_ids(input_ids_per_head,
                                                   unigram_vocab_size)
      ngram_ids_per_head *= (i + 1)
      ngram_ids_per_head += (i + 1)
      ngram_embs_expected = ngrammer_layer.ngram_table[i].emb_lookup(
          np.reshape(ngram_ids_per_head, [-1]))
      ngram_embs_expected = ngrammer_layer.ngram_layer_norm[i](
          ngram_embs_expected)
      ngram_embs_expected = jnp.reshape(ngram_embs_expected,
                                        [batch_size, seq_len, ngram_emb_dim])
      ngram_embs_expected *= (1 - paddings[:, :, np.newaxis])
      if concat_ngrams:
        ngram_embs_slice = ngram_embs[:, :, i, -ngram_emb_dim:]
      else:
        input_embs_ln = ngrammer_layer.emb_layer_norm[i](input_embs[:, :, i, :])
        ngram_embs_slice = ngram_embs[:, :, i, :] - input_embs_ln
      self.assertAllClose(to_np(ngram_embs_slice), to_np(ngram_embs_expected))

  @parameterized.parameters(
      (16, 8, 2, 32, True),
      (24, 4, 4, 16, True),
      (32, 16, 1, 64, True),
      (25, 4, 2, 8, True),
      (16, 8, 2, 8, False),
      (24, 4, 4, 4, False),
      (32, 16, 1, 16, False),
      (25, 4, 2, 4, False),
  )
  def test_ngrammer_layer_exact_bigram_2d(self, unigram_vocab_size,
                                          ngram_emb_dim, num_heads,
                                          dim_per_head, concat_ngrams):
    batch_size = 2
    seq_len = 8
    inputs = np.random.randint(
        unigram_vocab_size, size=[batch_size, seq_len], dtype=np.int32)
    paddings = np.random.randint(1, size=[batch_size, seq_len])
    input_embs = np.random.normal(
        1.5, 2.0, (batch_size, seq_len, num_heads * dim_per_head))
    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)
    ngrammer_layer_p = ngrammer.Ngrammer.HParams(
        name='jax_ngrammer_layer',
        unigram_vocab_size=unigram_vocab_size,
        ngram_vocab_size=num_heads * unigram_vocab_size**2,
        ngram_emb_dim=ngram_emb_dim,
        num_heads=num_heads,
        dim_per_head=dim_per_head,
        concat_ngrams=concat_ngrams,
    )
    ngrammer_layer = instantiate(ngrammer_layer_p)
    with base_layer.JaxContext.new_context():
      # Bind makes ngrammer_layer a stateful layer with all its submodule
      # layer variables automatically bound. It simplfies calling of submodule
      # fprop because we no longer need to explicitly provide submodule layer
      # vars.
      initial_vars = ngrammer_layer.init(init_key, inputs, input_embs, paddings)
      ngrammer_layer = ngrammer_layer.bind(
          initial_vars, mutable=[NON_TRAINABLE])
      ngram_embs = ngrammer_layer(inputs, input_embs, paddings)
    ngram_embs = np.reshape(ngram_embs,
                            [batch_size, seq_len, num_heads, dim_per_head])
    input_embs = np.reshape(input_embs,
                            [batch_size, seq_len, num_heads, dim_per_head])
    for i in range(num_heads):
      input_ids_per_head = inputs
      ngram_ids_per_head = ngrammer.get_bigram_ids(input_ids_per_head,
                                                   unigram_vocab_size)
      ngram_ids_per_head *= (i + 1)
      ngram_ids_per_head += (i + 1)
      ngram_embs_expected = ngrammer_layer.ngram_table[i].emb_lookup(
          np.reshape(ngram_ids_per_head, [-1]))
      ngram_embs_expected = ngrammer_layer.ngram_layer_norm[i](
          ngram_embs_expected)
      ngram_embs_expected = jnp.reshape(ngram_embs_expected,
                                        [batch_size, seq_len, ngram_emb_dim])
      ngram_embs_expected *= (1 - paddings[:, :, np.newaxis])
      if concat_ngrams:
        ngram_embs_slice = ngram_embs[:, :, i, -ngram_emb_dim:]
      else:
        input_embs_ln = ngrammer_layer.emb_layer_norm[i](input_embs[:, :, i, :])
        ngram_embs_slice = ngram_embs[:, :, i, :] - input_embs_ln
      self.assertAllClose(to_np(ngram_embs_slice), to_np(ngram_embs_expected))

  @parameterized.parameters(
      (8, 2, 4, 32, True, True),
      (4, 4, 32, 16, True, False),
      (16, 2, 8, 64, True, True),
      (4, 2, 8, 8, True, False),
      (8, 2, 4, 8, False, True),
      (4, 4, 32, 4, False, False),
      (16, 4, 16, 16, False, True),
      (16, 8, 16, 16, False, False),
  )
  def test_vq_ngrammer_layer_exact_bigram(self, ngram_emb_dim, num_heads,
                                          num_clusters, dim_per_head,
                                          concat_ngrams, use_attention_scores):
    batch_size = 2
    seq_len = 8
    paddings = np.random.randint(1, size=[batch_size, seq_len])
    input_embs = np.random.normal(
        1.5, 2.0, (batch_size, seq_len, num_heads * dim_per_head))
    pair_ids = None
    attention_scores = None
    if use_attention_scores:
      attention_scores = np.random.uniform(
          low=0, high=1, size=(batch_size, num_heads, seq_len, seq_len))
      pair_ids = np.argmax(attention_scores, axis=-1)
    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)
    vq_ngrammer_layer_p = ngrammer.VQNgrammer.HParams(
        name='jax_vq_ngrammer_layer',
        ngram_vocab_size=num_heads * num_clusters**2 + 1,
        ngram_emb_dim=ngram_emb_dim,
        num_heads=num_heads,
        num_clusters=num_clusters,
        dim_per_head=dim_per_head,
        concat_ngrams=concat_ngrams,
        ngram_using_attention_scores=use_attention_scores,
        causal_attention=False)
    vq_ngrammer_layer = instantiate(vq_ngrammer_layer_p)
    context_params = base_layer.JaxContext.HParams(do_eval=True)
    with base_layer.JaxContext.new_context(hparams=context_params):
      initial_vars = vq_ngrammer_layer.init(
          init_key,
          None,
          input_embs,
          paddings,
          attention_scores=attention_scores)
      # Bind makes vq_ngrammer_layer a stateful layer with all its submodule
      # layer variables automatically bound. It simplfies calling of submodule
      # fprop because we no longer need to explicitly provide submodule layer
      # vars.
      vq_ngrammer_layer = vq_ngrammer_layer.bind(
          initial_vars, mutable=[NON_TRAINABLE])
      ngram_embs = vq_ngrammer_layer(
          None, input_embs, paddings, attention_scores=attention_scores)
      dists, _ = vq_ngrammer_layer.vq_layer(input_embs)

    ngram_embs = np.reshape(ngram_embs,
                            [batch_size, seq_len, num_heads, dim_per_head])
    input_embs = jnp.reshape(input_embs,
                             [batch_size, seq_len, num_heads, dim_per_head])

    # [B, L, N].
    cluster_ids = jnp.argmin(dists, -1)
    for i in range(num_heads):
      input_ids_per_head = cluster_ids[:, :, i]
      pair_ids_per_head = None
      if use_attention_scores:
        pair_ids_per_head = pair_ids[:, i, :]
      ngram_ids_per_head = ngrammer.get_bigram_ids(
          input_ids_per_head, num_clusters, pair_ids=pair_ids_per_head)
      ngram_ids_per_head *= (i + 1)
      ngram_ids_per_head += (i + 1)
      ngram_embs_expected = vq_ngrammer_layer.ngram_layer.ngram_table[
          i].emb_lookup(np.reshape(ngram_ids_per_head, [-1]))
      ngram_embs_expected = (
          vq_ngrammer_layer.ngram_layer.ngram_layer_norm[i](ngram_embs_expected)
      )
      ngram_embs_expected = jnp.reshape(ngram_embs_expected,
                                        [batch_size, seq_len, ngram_emb_dim])
      ngram_embs_expected *= (1 - paddings[:, :, np.newaxis])
      if concat_ngrams:
        ngram_embs_slice = ngram_embs[:, :, i, -ngram_emb_dim:]
      else:
        input_embs_ln = vq_ngrammer_layer.ngram_layer.emb_layer_norm[i](
            input_embs[:, :, i, :])
        ngram_embs_slice = ngram_embs[:, :, i, :] - input_embs_ln
      self.assertAllClose(to_np(ngram_embs_slice), to_np(ngram_embs_expected))

  @parameterized.parameters(
      (8, 2, 4, 32, True, True),
      (4, 4, 32, 16, True, False),
      (16, 2, 8, 64, True, True),
      (4, 2, 8, 8, True, False),
      (8, 2, 4, 8, False, True),
      (4, 4, 32, 4, False, False),
      (16, 4, 16, 16, False, True),
      (16, 8, 16, 16, False, False),
  )
  def test_vq_ngrammer_layer_extend_step(self, ngram_emb_dim, num_heads,
                                         num_clusters, dim_per_head,
                                         concat_ngrams, use_attention_scores):
    batch_size = 2
    seq_len = 8
    paddings = np.random.randint(1, size=[batch_size, seq_len])
    input_embs = np.random.normal(
        1.5, 2.0, (batch_size, seq_len, num_heads * dim_per_head))
    attention_scores = None
    if use_attention_scores:
      attention_scores = np.random.uniform(
          low=0, high=1, size=(batch_size, num_heads, seq_len, seq_len))
    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)
    vq_ngrammer_layer_p = ngrammer.VQNgrammer.HParams(
        name='jax_vq_ngrammer_layer',
        ngram_vocab_size=num_heads * num_clusters**2 + 1,
        ngram_emb_dim=ngram_emb_dim,
        num_heads=num_heads,
        num_clusters=num_clusters,
        dim_per_head=dim_per_head,
        concat_ngrams=concat_ngrams,
        causal_attention=True,
        ngram_using_attention_scores=use_attention_scores)
    vq_ngrammer_layer = instantiate(vq_ngrammer_layer_p)
    context_params = base_layer.JaxContext.HParams(do_eval=True)
    with base_layer.JaxContext.new_context(hparams=context_params):
      initial_vars = vq_ngrammer_layer.init(
          init_key,
          None,
          input_embs,
          paddings,
          attention_scores=attention_scores)
      # Bind makes vq_ngrammer_layer a stateful layer with all its submodule
      # layer variables automatically bound. It simplfies calling of submodule
      # fprop because we no longer need to explicitly provide submodule layer
      # vars.
      vq_ngrammer_layer = vq_ngrammer_layer.bind(
          initial_vars, mutable=[NON_TRAINABLE])
      ngram_embs = vq_ngrammer_layer(
          None, input_embs, paddings, attention_scores=attention_scores)
      for step in range(seq_len):
        attention_score = None
        if use_attention_scores:
          attention_score = attention_scores[:, :, step, :]
        ngram_embs_extend_step = vq_ngrammer_layer.extend_step(
            input_embs, step=step, attention_score=attention_score)
        self.assertAllClose(
            to_np(ngram_embs[:, step, :]), to_np(ngram_embs_extend_step))

  def test_vq_ngrammer_layer_cache(self):
    ngram_emb_dim = 2
    num_heads = 2
    num_clusters = 4
    dim_per_head = 4
    concat_ngrams = True
    use_attention_scores = False
    batch_size = 2
    seq_len = 4
    unigram_vocab_size = 8
    paddings = np.random.randint(1, size=[batch_size, seq_len])
    input_ids = np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]])
    embeddings = np.random.normal(
        1.5, 2.0, (unigram_vocab_size, num_heads * dim_per_head))
    input_embs = embeddings[(input_ids,)]
    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)
    vq_ngrammer_layer_p = ngrammer.VQNgrammer.HParams(
        name='jax_vq_ngrammer_layer',
        ngram_vocab_size=num_heads * num_clusters**2 + 1,
        unigram_vocab_size=unigram_vocab_size,
        ngram_emb_dim=ngram_emb_dim,
        num_heads=num_heads,
        num_clusters=num_clusters,
        dim_per_head=dim_per_head,
        concat_ngrams=concat_ngrams,
        ngram_using_attention_scores=use_attention_scores,
        causal_attention=False,
        use_cached_input_ids_to_cluster_ids=False
        )
    vq_ngrammer_layer = instantiate(vq_ngrammer_layer_p)
    context_params = base_layer.JaxContext.HParams(do_eval=True)
    vq_ngrammer_layer_p_cached = ngrammer.VQNgrammer.HParams(
        name='jax_vq_ngrammer_layer',
        ngram_vocab_size=num_heads * num_clusters**2 + 1,
        unigram_vocab_size=unigram_vocab_size,
        ngram_emb_dim=ngram_emb_dim,
        num_heads=num_heads,
        num_clusters=num_clusters,
        dim_per_head=dim_per_head,
        concat_ngrams=concat_ngrams,
        ngram_using_attention_scores=use_attention_scores,
        causal_attention=False,
        use_cached_input_ids_to_cluster_ids=True
        )
    vq_ngrammer_layer_cached = instantiate(vq_ngrammer_layer_p_cached)
    with base_layer.JaxContext.new_context(hparams=context_params):
      initial_vars = vq_ngrammer_layer.init(
          init_key,
          input_ids,
          input_embs,
          paddings)
      # Bind makes vq_ngrammer_layer a stateful layer with all its submodule
      # layer variables automatically bound. It simplfies calling of submodule
      # fprop because we no longer need to explicitly provide submodule layer
      # vars.
      vq_ngrammer_layer = vq_ngrammer_layer.bind(
          initial_vars, mutable=[NON_TRAINABLE])
      vq_ngrammer_layer_cached = vq_ngrammer_layer_cached.bind(
          initial_vars, mutable=[NON_TRAINABLE])
      dists, _ = vq_ngrammer_layer.vq_layer(input_embs)
      cluster_ids = jnp.argmin(dists, -1)
      input_ids_flat = jnp.reshape(input_ids, [-1])
      cluster_ids_flat = jnp.reshape(cluster_ids, [-1, num_heads])
      cache = initial_vars['non_trainable']['input_id_to_cluster_id_cache']
      for i in range(num_heads):
        cache = cache.at[input_ids_flat, i].set(
            cluster_ids_flat[:, i])
      updated_vars = initial_vars
      updated_vars['non_trainable'][
          'input_id_to_cluster_id_cache'] = cache
      ngram_embs = vq_ngrammer_layer(input_ids, input_embs, paddings)
      ngram_embs_cached = vq_ngrammer_layer_cached.apply(
          updated_vars,
          input_ids, input_embs, paddings)
      self.assertAllClose(ngram_embs, ngram_embs_cached)

  @parameterized.parameters(
      (32, 8, 2, 4, 32, True),
      (32, 4, 4, 16, 16, True),
      (24, 16, 2, 8, 64, True),
      (16, 4, 2, 8, 8, True),
      (16, 8, 2, 4, 8, False),
      (32, 4, 4, 2, 4, False),
      (32, 16, 4, 8, 16, False),
      (32, 16, 8, 16, 16, False),
  )
  def test_bregman_ngrammer_layer_exact_bigram(self, ngram_vocab_size,
                                               ngram_emb_dim, num_heads,
                                               num_components, dim_per_head,
                                               concat_ngrams):
    batch_size = 2
    seq_len = 8
    paddings = np.random.randint(low=0, high=2, size=[batch_size, seq_len])
    inputs = np.tile(np.arange(seq_len)[np.newaxis, :],
                     [batch_size, 1]).astype(dtype=np.int32)
    inputs_pad = jnp.zeros([batch_size, 1], dtype=np.int32)
    inputs = np.concatenate([inputs_pad, inputs], 1)
    curr_token_id = inputs[:, 1:]
    prev_token_id = inputs[:, :-1]

    input_embs = np.random.normal(
        1.5, 2.0, (batch_size, seq_len, num_heads, dim_per_head))
    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)
    bregman_ngrammer_layer_p = ngrammer.BregmanNgrammer.HParams(
        name='jax_bregman_ngrammer_layer',
        ngram_vocab_size=ngram_vocab_size,
        ngram_emb_dim=ngram_emb_dim,
        num_heads=num_heads,
        num_components=num_components,
        dim_per_head=dim_per_head,
        concat_ngrams=concat_ngrams,
        start_step=0,
        end_step=10)
    bregman_ngrammer_layer = instantiate(bregman_ngrammer_layer_p)
    context_params = base_layer.JaxContext.HParams(do_eval=True)
    with base_layer.JaxContext.new_context(hparams=context_params):
      initial_vars = bregman_ngrammer_layer.init(
          init_key, inputs, input_embs, paddings=paddings)
      # Bind makes bregman_ngrammer_layer a stateful layer with all its
      # submodule layer variables automatically bound. It simplfies calling of
      # submodule __call__ because we no longer need to explicitly provide
      # submodule layer vars.
      bregman_ngrammer_layer = bregman_ngrammer_layer.bind(
          initial_vars, mutable=[NON_TRAINABLE])
      ngram_embs = bregman_ngrammer_layer(inputs, input_embs, paddings=paddings)
      # [B, L, N, C].
      input_coeffs = bregman_ngrammer_layer.bregman_compression_layer(
          input_embs, paddings=paddings)

    ngram_embs = np.reshape(ngram_embs,
                            [batch_size, seq_len, num_heads, dim_per_head])
    input_embs = np.reshape(input_embs,
                            [batch_size, seq_len, num_heads, dim_per_head])

    correlations = np.split(
        bregman_ngrammer_layer.theta.correlation, num_heads, axis=-1)
    correlations = [
        np.squeeze(correlation, axis=-1) for correlation in correlations
    ]
    embedding_tables = np.split(
        bregman_ngrammer_layer.theta.embedding_table, num_heads, axis=-1)
    embedding_tables = [
        np.squeeze(embedding_table, axis=-1)
        for embedding_table in embedding_tables
    ]

    input_embs_per_head = np.split(input_embs, num_heads, 2)
    for i in range(num_heads):
      # Reshape into [B * L, H]
      per_head_emb = np.reshape(input_embs_per_head[i], [-1, dim_per_head])
      input_embs_per_head[i] = bregman_ngrammer_layer.emb_layer_norm[i](
          per_head_emb)
      input_embs_per_head[i] = np.reshape(input_embs_per_head[i],
                                          [batch_size, seq_len, dim_per_head])

    # [B, L, N, H].
    input_embs = np.stack(input_embs_per_head, axis=2)

    for i in range(num_heads):
      # [B, L, C]
      curr_coeffs_i = np.take_along_axis(
          input_coeffs[:, :, i, :], curr_token_id[:, :, np.newaxis], axis=1)
      prev_coeffs_i = np.take_along_axis(
          input_coeffs[:, :, i, :], prev_token_id[:, :, np.newaxis], axis=1)
      # [B, L, C, V].
      inner_prod = np.einsum('BLC, CKV -> BLKV', prev_coeffs_i, correlations[i])
      # [B, L, V].
      ngram_corrs = np.einsum('BLC, BLCV -> BLV', curr_coeffs_i, inner_prod)
      # [B, L, H].
      ngram_embs_expected = np.einsum('BLV, VH -> BLH', ngram_corrs,
                                      embedding_tables[i])
      ngram_embs_expected = np.reshape(ngram_embs_expected, [-1, ngram_emb_dim])
      ngram_embs_expected = bregman_ngrammer_layer.ngram_layer_norm[i](
          ngram_embs_expected)
      ngram_embs_expected = jnp.reshape(ngram_embs_expected,
                                        [batch_size, seq_len, ngram_emb_dim])
      ngram_embs_expected *= (1 - paddings[:, :, np.newaxis])
      if concat_ngrams:
        ngram_embs_slice = ngram_embs[:, :, i, -ngram_emb_dim:]
      else:
        ngram_embs_slice = ngram_embs[:, :, i, :] - input_embs[:, :, i, :]
        ngram_embs_slice *= (1 - paddings[:, :, np.newaxis])
      self.assertAllClose(to_np(ngram_embs_slice), to_np(ngram_embs_expected))


if __name__ == '__main__':
  absltest.main()
