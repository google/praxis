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

"""Tests for Praxis transformer layers."""

import itertools

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from lingvo.core import gshard_builder
import numpy as np
from praxis import base_hyperparams
from praxis import base_layer
from praxis import py_utils
from praxis import test_utils
from praxis.layers import attentions
from praxis.layers import embedding_softmax
from praxis.layers import glam
from praxis.layers import ngrammer
from praxis.layers import transformer_models
from praxis.layers import transformers
import tensorflow.compat.v2 as tf

PARAMS = base_layer.PARAMS
RANDOM = base_layer.RANDOM
DECODE_CACHE = base_layer.DECODE_CACHE
instantiate = base_layer.instantiate


class TransformerModelsTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)
    tf.random.set_seed(123)

  @parameterized.parameters([True, False])
  def test_transformer_bert(self, trainable_position_emb):
    seq_len = 512
    if trainable_position_emb:
      position_emb_tpl = embedding_softmax.TrainablePositionalEmbedding.HParams(
      )
      position_emb_tpl.max_seq_length = seq_len
    else:
      position_emb_tpl = embedding_softmax.PositionalEmbedding.HParams()
    p = transformer_models.TransformerLm.HParams(
        name='bert_lm',
        model_dims=32,
        vocab_size=52,
        position_emb_tpl=position_emb_tpl)
    stacked_transformer_tpl = p.stacked_transformer_tpl
    stacked_transformer_tpl.model_dims = 32
    stacked_transformer_tpl.hidden_dims = 4 * 32
    stacked_transformer_tpl.num_heads = 4
    stacked_transformer_tpl.num_layers = 1
    p.softmax_tpl.scale_sqrt_depth = True
    batch_size = 8
    bert_lm = instantiate(p)
    input_ids = jax.random.randint(
        jax.random.PRNGKey(1234), [batch_size, seq_len], 0, 51)
    input_paddings = jnp.zeros([batch_size, seq_len])
    input_weights = jnp.ones([batch_size, seq_len])
    input_segment_ids = jnp.ones([batch_size, seq_len])
    input_segment_pos = jnp.tile(
        jnp.arange(0, seq_len)[jnp.newaxis, :], [batch_size, 1])

    labels = py_utils.NestedMap()
    labels.class_ids = input_ids
    labels.class_weights = input_weights

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = bert_lm.init(
          prng_key,
          input_ids,
          input_paddings,
          labels=labels,
          segment_ids=input_segment_ids,
          segment_pos=input_segment_pos)
      outputs = bert_lm.apply(
          initial_vars,
          input_ids,
          input_paddings,
          labels=labels,
          segment_ids=input_segment_ids,
          segment_pos=input_segment_pos)
      logging.info('outputs: %s', outputs)

  @parameterized.parameters(*list(itertools.product([True, False], repeat=5)))
  def test_ngrammer_lm_extendstep(self, use_vq_ngrams, use_rotary_position_emb,
                                  use_post_attention_ngrammer,
                                  ngram_using_attention_scores,
                                  share_embedding_and_softmax):
    vocab_size = 8
    num_layers = 2
    num_heads = 2
    dim_per_head = 8
    ngram_emb_dim = 4
    post_attention_ngrammer_tpls = None
    if use_vq_ngrams:
      ngrammer_params = ngrammer.VQNgrammer.HParams(
          ngram_vocab_size=64,
          ngram_emb_dim=ngram_emb_dim,
          num_heads=num_heads,
          concat_ngrams=True,
          num_clusters=2,
          dim_per_head=dim_per_head)
    else:
      ngrammer_params = ngrammer.Ngrammer.HParams(
          ngram_vocab_size=64,
          unigram_vocab_size=vocab_size,
          ngram_emb_dim=ngram_emb_dim,
          num_heads=num_heads,
          concat_ngrams=True,
          dim_per_head=dim_per_head)
    if use_post_attention_ngrammer:
      post_attn_ngrammer_params = ngrammer.VQNgrammer.HParams(
          ngram_vocab_size=8,
          ngram_emb_dim=ngram_emb_dim,
          ngram_using_attention_scores=ngram_using_attention_scores,
          num_heads=num_heads,
          concat_ngrams=True,
          causal_attention=True,
          num_clusters=2,
          dim_per_head=dim_per_head)
      post_attention_ngrammer_tpls = [post_attn_ngrammer_params] * num_layers
    p = transformer_models.TransformerLm.HParams(
        name='jax_ngrammer_layer',
        model_dims=num_heads * dim_per_head,
        model_type=transformer_models.LanguageModelType.CAUSAL,
        packed_input=False,
        ngrammer_tpl=ngrammer_params,
        post_attention_ngrammer_tpls=post_attention_ngrammer_tpls,
        vocab_size=vocab_size)
    stacked_transformer_tpl = p.stacked_transformer_tpl
    stacked_transformer_tpl.model_dims = num_heads * dim_per_head
    stacked_transformer_tpl.hidden_dims = 4 * num_heads * dim_per_head
    stacked_transformer_tpl.num_heads = num_heads
    stacked_transformer_tpl.num_layers = num_layers
    if not share_embedding_and_softmax:
      p.separate_embedding_tpl = embedding_softmax.Embedding.HParams()
      p.softmax_tpl = embedding_softmax.FullSoftmax.HParams()
    # Rotary position embedding.
    params = p.stacked_transformer_tpl.transformer_layer_params_tpl
    params.tr_atten_tpl.use_rotary_position_emb = use_rotary_position_emb
    seq_len = 4
    batch_size = 2
    transformer_lm = instantiate(p)
    npy_inputs = np.random.randint(
        vocab_size, size=(batch_size, seq_len)).astype('int32')
    inputs = jnp.asarray(npy_inputs)
    context_params = base_layer.JaxContext.HParams(do_eval=True)
    with base_layer.JaxContext.new_context(hparams=context_params):
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = transformer_lm.init(
          prng_key,
          inputs,
          jnp.zeros_like(inputs),
      )
      fprop_outputs = transformer_lm.apply(
          initial_vars,
          inputs,
          jnp.zeros_like(inputs),
          method=transformer_lm.__call__)
      _, decoder_state = transformer_lm.apply(
          initial_vars,
          jnp.zeros_like(inputs),
          jnp.ones_like(inputs),
          method=transformer_lm.__call__,
          mutable=[DECODE_CACHE])

      logits = fprop_outputs.logits

      updated_vars = py_utils.MergeDictsWithValueCheck(decoder_state,
                                                       initial_vars)
      for t in range(seq_len):
        if t > 0:
          inputs_prefix = inputs[:, t - 1:t + 1]
        else:
          inputs_prefix = inputs[:, t]
        xent_output, decoder_state = transformer_lm.apply(
            updated_vars,
            inputs_prefix,
            method=transformer_lm.extend_step,
            mutable=[DECODE_CACHE])
        updated_vars = py_utils.MergeDictsWithValueCheck(
            decoder_state, initial_vars)
        self.assertAllClose(logits[:, t, :], xent_output.logits)

  @parameterized.parameters(*list(itertools.product([True, False], repeat=2)))
  def test_primer_lm_extendstep(self, use_rotary_position_emb,
                                share_embedding_and_softmax):
    vocab_size = 8
    num_layers = 2
    num_heads = 2
    dim_per_head = 4
    dconv_kernel_size = 3
    p = transformer_models.TransformerLm.HParams(
        name='jax_primer_layer',
        model_dims=num_heads * dim_per_head,
        model_type=transformer_models.LanguageModelType.CAUSAL,
        packed_input=False,
        vocab_size=vocab_size)
    stacked_transformer_tpl = p.stacked_transformer_tpl
    stacked_transformer_tpl.model_dims = num_heads * dim_per_head
    stacked_transformer_tpl.hidden_dims = 2 * num_heads * dim_per_head
    stacked_transformer_tpl.num_heads = num_heads
    stacked_transformer_tpl.num_layers = num_layers
    if not share_embedding_and_softmax:
      p.separate_embedding_tpl = embedding_softmax.Embedding.HParams()
      p.softmax_tpl = embedding_softmax.FullSoftmax.HParams()
    seq_len = 4
    batch_size = 3
    # Turn on dconv as in Primer.
    params = p.stacked_transformer_tpl.transformer_layer_params_tpl
    params.tr_atten_tpl.dconv_qkv = True
    # Rotary position embedding.
    params = p.stacked_transformer_tpl.transformer_layer_params_tpl
    params.tr_atten_tpl.dconv_kernel_size = dconv_kernel_size
    params.tr_atten_tpl.use_rotary_position_emb = use_rotary_position_emb
    transformer_lm = instantiate(p)
    npy_inputs = np.random.randint(
        vocab_size, size=(batch_size, seq_len)).astype('int32')
    inputs = jnp.asarray(npy_inputs)
    context_params = base_layer.JaxContext.HParams(do_eval=True)
    with base_layer.JaxContext.new_context(hparams=context_params):
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = transformer_lm.init(
          prng_key,
          inputs,
          jnp.zeros_like(inputs),
      )
      fprop_outputs = transformer_lm.apply(
          initial_vars,
          inputs,
          jnp.zeros_like(inputs),
          method=transformer_lm.__call__)
      _, decoder_state = transformer_lm.apply(
          initial_vars,
          jnp.zeros_like(inputs),
          jnp.ones_like(inputs),
          method=transformer_lm.__call__,
          mutable=[DECODE_CACHE])
      logits = fprop_outputs.logits
      updated_vars = py_utils.MergeDictsWithValueCheck(decoder_state,
                                                       initial_vars)
      for t in range(seq_len):
        xent_output, decoder_state = transformer_lm.apply(
            updated_vars,
            inputs[:, t],
            method=transformer_lm.extend_step,
            mutable=[DECODE_CACHE])
        updated_vars = py_utils.MergeDictsWithValueCheck(
            decoder_state, initial_vars)
        self.assertAllClose(logits[:, t, :], xent_output.logits)

  @parameterized.parameters(*list(itertools.product([True, False], repeat=2)))
  def test_lm_extend_n_step(self, use_rotary_position_emb,
                            share_embedding_and_softmax):
    vocab_size = 8
    num_layers = 2
    num_heads = 2
    dim_per_head = 4
    p = transformer_models.TransformerLm.HParams(
        name='jax_lm_layer',
        model_dims=num_heads * dim_per_head,
        model_type=transformer_models.LanguageModelType.CAUSAL,
        packed_input=False,
        vocab_size=vocab_size)
    stacked_transformer_tpl = p.stacked_transformer_tpl
    stacked_transformer_tpl.model_dims = num_heads * dim_per_head
    stacked_transformer_tpl.hidden_dims = 2 * num_heads * dim_per_head
    stacked_transformer_tpl.num_heads = num_heads
    stacked_transformer_tpl.num_layers = num_layers
    if not share_embedding_and_softmax:
      p.separate_embedding_tpl = embedding_softmax.Embedding.HParams()
      p.softmax_tpl = embedding_softmax.FullSoftmax.HParams()
    seq_len = 4
    batch_size = 3
    # Turn on dconv as in Primer.
    params = p.stacked_transformer_tpl.transformer_layer_params_tpl
    params.tr_atten_tpl.dconv_qkv = False
    # Rotary position embedding.
    params = p.stacked_transformer_tpl.transformer_layer_params_tpl
    params.tr_atten_tpl = attentions.DotProductAttentionWithLPB.HParams(
        input_dim=num_heads * dim_per_head,
        hidden_dim=2 * num_heads * dim_per_head,
        num_heads=num_heads,
        dim_per_head=dim_per_head if use_rotary_position_emb else None,
        atten_logit_cap=20.0,
        combine_qkv=True,
        dconv_qkv=False,
        use_rotary_position_emb=use_rotary_position_emb)
    transformer_lm = instantiate(p)
    npy_inputs = np.random.randint(
        vocab_size, size=(batch_size, seq_len)).astype('int32')
    inputs = jnp.asarray(npy_inputs)
    ninf = py_utils.get_large_negative_number(jnp.float32)
    segment_mask = jnp.stack([
        jnp.array([[0, ninf, ninf, ninf], [0, 0, ninf, ninf], [0, 0, 0, ninf],
                   [0, 0, 0, 0]],
                  dtype=jnp.float32)
    ] * batch_size)
    segment_mask = segment_mask[:, jnp.newaxis, :, :]
    segment_pos = jnp.stack([jnp.arange(seq_len)] * batch_size)
    context_params = base_layer.JaxContext.HParams(do_eval=True)
    with base_layer.JaxContext.new_context(hparams=context_params):
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = transformer_lm.init(
          prng_key,
          inputs,
          jnp.zeros_like(inputs),
      )
      fprop_outputs = transformer_lm.apply(
          initial_vars,
          inputs,
          jnp.zeros_like(inputs),
          method=transformer_lm.__call__)
      _, decoder_state = transformer_lm.apply(
          initial_vars,
          jnp.zeros_like(inputs),
          jnp.ones_like(inputs),
          method=transformer_lm.__call__,
          mutable=[DECODE_CACHE])
      logits = fprop_outputs.logits
      updated_vars = py_utils.MergeDictsWithValueCheck(decoder_state,
                                                       initial_vars)
      xent_output, _ = transformer_lm.apply(
          updated_vars,
          inputs,
          method=transformer_lm.extend_step,
          segment_pos=segment_pos,
          atten_mask=segment_mask,
          mutable=[DECODE_CACHE])
      self.assertAllClose(logits, xent_output.logits)

  @parameterized.parameters(*list(itertools.product([True, False], repeat=5)))
  def test_ngrammer_primer_lm_extendstep(self, use_vq_ngrams,
                                         use_rotary_position_emb,
                                         use_post_attention_ngrammer,
                                         ngram_using_attention_scores,
                                         share_embedding_and_softmax):
    vocab_size = 8
    num_layers = 2
    num_heads = 2
    dim_per_head = 8
    ngram_emb_dim = 4
    dconv_kernel_size = 3
    post_attention_ngrammer_tpls = None
    ngram_using_attention_scores = False
    if use_vq_ngrams:
      ngrammer_params = ngrammer.VQNgrammer.HParams(
          ngram_vocab_size=64,
          ngram_emb_dim=ngram_emb_dim,
          num_heads=num_heads,
          concat_ngrams=True,
          num_clusters=2,
          dim_per_head=dim_per_head)
    else:
      ngrammer_params = ngrammer.Ngrammer.HParams(
          ngram_vocab_size=64,
          unigram_vocab_size=vocab_size,
          ngram_emb_dim=ngram_emb_dim,
          num_heads=num_heads,
          concat_ngrams=True,
          dim_per_head=dim_per_head)
    if use_post_attention_ngrammer:
      post_attn_ngrammer_params = ngrammer.VQNgrammer.HParams(
          ngram_vocab_size=8,
          ngram_emb_dim=ngram_emb_dim,
          ngram_using_attention_scores=ngram_using_attention_scores,
          num_heads=num_heads,
          concat_ngrams=True,
          causal_attention=True,
          num_clusters=2,
          dim_per_head=dim_per_head)
      post_attention_ngrammer_tpls = [post_attn_ngrammer_params] * num_layers
    p = transformer_models.TransformerLm.HParams(
        name='jax_ngrammer_layer',
        model_dims=num_heads * dim_per_head,
        model_type=transformer_models.LanguageModelType.CAUSAL,
        packed_input=False,
        ngrammer_tpl=ngrammer_params,
        post_attention_ngrammer_tpls=post_attention_ngrammer_tpls,
        vocab_size=vocab_size)
    stacked_transformer_tpl = p.stacked_transformer_tpl
    stacked_transformer_tpl.model_dims = num_heads * dim_per_head
    stacked_transformer_tpl.hidden_dims = 4 * num_heads * dim_per_head
    stacked_transformer_tpl.num_heads = num_heads
    stacked_transformer_tpl.num_layers = num_layers
    if not share_embedding_and_softmax:
      p.separate_embedding_tpl = embedding_softmax.Embedding.HParams()
      p.softmax_tpl = embedding_softmax.FullSoftmax.HParams()
    seq_len = 4
    batch_size = 2
    # Turn on dconv as in Primer.
    params = p.stacked_transformer_tpl.transformer_layer_params_tpl
    params.tr_atten_tpl.dconv_qkv = True
    params.tr_atten_tpl.dconv_kernel_size = dconv_kernel_size
    # Rotary position embedding.
    params = p.stacked_transformer_tpl.transformer_layer_params_tpl
    params.tr_atten_tpl.use_rotary_position_emb = use_rotary_position_emb
    transformer_lm = instantiate(p)
    npy_inputs = np.random.randint(
        vocab_size, size=(batch_size, seq_len)).astype('int32')
    inputs = jnp.asarray(npy_inputs)
    context_params = base_layer.JaxContext.HParams(do_eval=True)
    with base_layer.JaxContext.new_context(hparams=context_params):
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = transformer_lm.init(prng_key, inputs,
                                         jnp.zeros_like(inputs))
      fprop_outputs = transformer_lm.apply(
          initial_vars,
          inputs,
          jnp.zeros_like(inputs),
          method=transformer_lm.__call__)
      _, decoder_state = transformer_lm.apply(
          initial_vars,
          jnp.zeros_like(inputs),
          jnp.ones_like(inputs),
          method=transformer_lm.__call__,
          mutable=[DECODE_CACHE])
      logits = fprop_outputs.logits

      updated_vars = py_utils.MergeDictsWithValueCheck(decoder_state,
                                                       initial_vars)
      for t in range(seq_len):
        if t > 0:
          inputs_prefix = inputs[:, t - 1:t + 1]
        else:
          inputs_prefix = inputs[:, t]
        xent_output, decoder_state = transformer_lm.apply(
            updated_vars,
            inputs_prefix,
            method=transformer_lm.extend_step,
            mutable=[DECODE_CACHE])
        updated_vars = py_utils.MergeDictsWithValueCheck(
            decoder_state, initial_vars)
        self.assertAllClose(logits[:, t, :], xent_output.logits)

  @parameterized.parameters(*list(itertools.product([True, False], repeat=9)))
  def test_transformer_encoder_decoder_extendstep(
      self, use_encoder_ngrams, use_decoder_ngrams, use_encoder_vq_ngrams,
      use_decoder_vq_ngrams, use_post_attention_ngrammer,
      use_rotary_position_emb, separate_encoder_embedding,
      separate_decoder_embedding, use_stacked_transformer_repeated):
    vocab_size = 4
    num_layers = 2
    num_heads = 2
    dim_per_head = 4
    ngram_emb_dim = 2
    encoder_ngrammer_params = None
    decoder_ngrammer_params = None
    post_attention_ngrammer_tpls = None
    if use_encoder_vq_ngrams:
      encoder_ngrammer_params = ngrammer.VQNgrammer.HParams(
          ngram_vocab_size=8,
          ngram_emb_dim=ngram_emb_dim,
          num_heads=num_heads,
          concat_ngrams=True,
          num_clusters=2,
          dim_per_head=dim_per_head)
    if use_encoder_ngrams:
      encoder_ngrammer_params = ngrammer.Ngrammer.HParams(
          ngram_vocab_size=16,
          unigram_vocab_size=vocab_size,
          ngram_emb_dim=ngram_emb_dim,
          num_heads=num_heads,
          concat_ngrams=True,
          dim_per_head=dim_per_head)
    if use_decoder_vq_ngrams:
      decoder_ngrammer_params = ngrammer.VQNgrammer.HParams(
          ngram_vocab_size=8,
          ngram_emb_dim=ngram_emb_dim,
          num_heads=num_heads,
          concat_ngrams=True,
          num_clusters=2,
          dim_per_head=dim_per_head)
    if use_decoder_ngrams:
      decoder_ngrammer_params = ngrammer.Ngrammer.HParams(
          ngram_vocab_size=16,
          unigram_vocab_size=vocab_size,
          ngram_emb_dim=ngram_emb_dim,
          num_heads=num_heads,
          concat_ngrams=True,
          dim_per_head=dim_per_head)
    if use_post_attention_ngrammer:
      ngrammer_params = ngrammer.Ngrammer.HParams(
          ngram_vocab_size=4,
          unigram_vocab_size=vocab_size,
          ngram_emb_dim=ngram_emb_dim,
          num_heads=num_heads,
          concat_ngrams=True,
          dim_per_head=dim_per_head)
      post_attention_ngrammer_tpls = [ngrammer_params] * 2
    p = transformer_models.TransformerEncoderDecoder.HParams(
        name='jax_transformer_encoder_decoder',
        model_dims=num_heads * dim_per_head,
        decoder_ngrammer_tpl=decoder_ngrammer_params,
        encoder_ngrammer_tpl=encoder_ngrammer_params,
        encoder_post_attention_ngrammer_tpls=post_attention_ngrammer_tpls,
        decoder_post_attention_ngrammer_tpls=post_attention_ngrammer_tpls)

    # Encoder stack.
    if use_stacked_transformer_repeated:
      block_param = transformers.StackedTransformer.HParams(
          num_layers=num_layers,
          num_heads=num_heads,
          model_dims=num_heads * dim_per_head,
          hidden_dims=num_heads * dim_per_head,
          mask_self_attention=False,
          fold_padding_with_segment_mask=True)
      p.encoder_stacked_transformer_tpl = (
          transformers.StackedTransformerRepeated.HParams(
              block=block_param, x_times=1))
    else:
      p.encoder_stacked_transformer_tpl = (
          transformers.StackedTransformer.HParams(
              model_dims=num_heads * dim_per_head,
              hidden_dims=num_heads * dim_per_head,
              num_heads=num_heads,
              num_layers=num_layers,
              mask_self_attention=False,
              fold_padding_with_segment_mask=True))

    # Decoder stack.
    if use_stacked_transformer_repeated:
      block_param = transformers.StackedTransformer.HParams(
          num_layers=num_layers,
          num_heads=num_heads,
          model_dims=num_heads * dim_per_head,
          hidden_dims=num_heads * dim_per_head,
          mask_self_attention=True,
          fold_padding_with_segment_mask=True)
      p.decoder_stacked_transformer_tpl = (
          transformers.StackedTransformerRepeated.HParams(
              block=block_param, x_times=1))
    else:
      p.decoder_stacked_transformer_tpl = (
          transformers.StackedTransformer.HParams(
              model_dims=num_heads * dim_per_head,
              hidden_dims=num_heads * dim_per_head,
              num_heads=num_heads,
              num_layers=num_layers,
              mask_self_attention=True,
              fold_padding_with_segment_mask=True))

    if separate_encoder_embedding:
      p.encoder_embedding_tpl = (
          embedding_softmax.Embedding.HParams(
              num_classes=vocab_size, input_dims=num_heads * dim_per_head))

    if separate_decoder_embedding:
      p.decoder_embedding_tpl = (
          embedding_softmax.Embedding.HParams(
              num_classes=vocab_size, input_dims=num_heads * dim_per_head))

    # Softmax params.
    if separate_decoder_embedding:
      p.softmax_tpl = embedding_softmax.FullSoftmax.HParams(
          input_dims=num_heads * dim_per_head, num_classes=vocab_size)
    else:
      p.softmax_tpl = (
          embedding_softmax.SharedEmbeddingSoftmax.HParams(
              input_dims=num_heads * dim_per_head, num_classes=vocab_size))

    # Rotary position embedding.
    if use_rotary_position_emb:
      if use_stacked_transformer_repeated:
        params = p.encoder_stacked_transformer_tpl.block
      else:
        params = p.encoder_stacked_transformer_tpl
      params = params.transformer_layer_params_tpl
      params.tr_atten_tpl.use_rotary_position_emb = use_rotary_position_emb
      if use_stacked_transformer_repeated:
        params = p.decoder_stacked_transformer_tpl.block
      else:
        params = p.decoder_stacked_transformer_tpl
      params = params.transformer_layer_params_tpl
      params.tr_atten_tpl.use_rotary_position_emb = use_rotary_position_emb
    p.position_emb_tpl = None

    seq_len = 4
    batch_size = 1
    transformer_enc_dec = instantiate(p)
    npy_inputs = np.random.randint(
        vocab_size, size=(batch_size, seq_len)).astype('int32')
    npy_input_paddings = np.random.randint(
        0, 2, size=(batch_size, seq_len)).astype('int32')
    npy_targets = np.random.randint(
        vocab_size, size=(batch_size, seq_len)).astype('int32')
    inputs = jnp.asarray(npy_inputs, dtype=jnp.int32)
    input_paddings = jnp.asarray(npy_input_paddings, dtype=jnp.int32)
    targets = jnp.asarray(npy_targets, dtype=jnp.int32)
    context_params = base_layer.JaxContext.HParams(do_eval=True)
    with base_layer.JaxContext.new_context(hparams=context_params):
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = transformer_enc_dec.init(prng_key, inputs, input_paddings,
                                              targets, jnp.zeros_like(targets))
      fprop_outputs = transformer_enc_dec.apply(
          initial_vars,
          inputs,
          input_paddings,
          targets,
          jnp.zeros_like(targets),
          method=transformer_enc_dec.__call__)
      _, decoder_state = transformer_enc_dec.apply(
          initial_vars,
          inputs,
          input_paddings,
          targets,
          jnp.zeros_like(targets),
          start_time_step=0,
          method=transformer_enc_dec.__call__,
          mutable=[DECODE_CACHE])
      logits = fprop_outputs.logits
      updated_vars = py_utils.MergeDictsWithValueCheck(decoder_state,
                                                       initial_vars)
      for t in range(seq_len):
        targets_prefix = targets[:, t]
        if use_decoder_ngrams or use_decoder_vq_ngrams:
          if t > 0:
            targets_prefix = targets[:, t - 1:t + 1]
        xent_output, decoder_state = transformer_enc_dec.apply(
            updated_vars,
            targets_prefix,
            method=transformer_enc_dec.extend_step,
            mutable=[DECODE_CACHE])
        updated_vars = py_utils.MergeDictsWithValueCheck(
            decoder_state, initial_vars)
        self.assertAllClose(logits[:, t, :], xent_output.logits, atol=2e-6)

  @parameterized.parameters(
      itertools.product(['top2', 'dense_top2', 'expert_choice'],
                        ['token', 'sentence']))
  def test_glam_unitransformer_fprop(self, gating_func,
                                     moe_gating_embedding_level):
    batch = 2
    length = 3
    d_model = 6
    num_heads = 2
    vocab_size = 16
    ff_dim = 8
    c_dim = 3
    e_dim = 2
    num_layers = 4
    # Build jax layer
    jax_p = glam.GlamUniTransformerLmHParams(
        name='model',
        vocab_size=vocab_size,
        num_transformer_layers=num_layers,
        moe=True,
        model_dim=d_model,
        ff_dim=ff_dim,
        moe_hidden_dim=ff_dim,
        attention_num_heads=num_heads,
        attention_key_value_dim=d_model // num_heads,
        attention_extra_logit=0.0,
        use_tgt_labels_size_as_loss_denominator=True,
        moe_load_balance_loss_weight=0.01,
        z_loss_weight=1e-4,
        moe_gating_func=gating_func,
        moe_gating_embedding_level=moe_gating_embedding_level,
        c_dim=c_dim,
        e_dim=e_dim)
    jax_layer = instantiate(jax_p)
    # Build Jax Inputs
    np.random.seed(42)
    npy_ids = np.random.randint(0, vocab_size - 1, [batch, length])
    jax_ids = jnp.asarray(npy_ids)
    npy_paddings = np.array([[0, 0, 1], [0, 0, 1]], dtype=np.float32)

    jax_paddings = jnp.asarray(npy_paddings)
    npy_segment_ids = np.array([[1, 2, 0], [1, 1, 0]], dtype=np.int32)
    npy_segment_pos = np.array([[0, 0, 0], [0, 1, 0]], dtype=np.int32)
    npy_labels = np.roll(npy_ids, -1, axis=1)
    jax_labels = jnp.asarray(npy_labels)
    jax_seg_ids = jnp.asarray(npy_segment_ids)
    jax_seg_pos = jnp.asarray(npy_segment_pos)
    jax_label_weighs = jnp.asarray([[1, 1, 0], [1, 1, 0]])
    # Compute jax outputs
    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=42)
      prng_key, init_key = jax.random.split(prng_key)
      prng_key, random_key = jax.random.split(prng_key)
      jax_vars = jax_layer.init(
          {
              PARAMS: init_key,
              RANDOM: random_key
          },
          jax_ids,
          jax_paddings,
          labels=py_utils.NestedMap(
              class_ids=jax_labels,
              class_weights=jax_label_weighs,
          ),
          segment_ids=jax_seg_ids,
          segment_pos=jax_seg_pos,
      )
      jax_outputs, updated_vars = jax_layer.apply(
          jax_vars,
          jax_ids,
          jax_paddings,
          labels=py_utils.NestedMap(
              class_ids=jax_labels,
              class_weights=jax_label_weighs,
          ),
          segment_ids=jax_seg_ids,
          segment_pos=jax_seg_pos,
          rngs={RANDOM: random_key},
          mutable=[base_layer.AUX_LOSS])
      # There are two MOE load balancing loss + z-loss, for a total aux-loss
      # weight of 3.0
      print(jax_outputs)
      print(updated_vars)
      self.assertEqual(3.0, jax_outputs.aux_loss_weight)

  def test_glam_unitransformer(self):
    batch = 2
    length = 3
    d_model = 6
    num_heads = 2
    vocab_size = 16
    ff_dim = 8
    c_dim = 3
    e_dim = 2
    num_layers = 4
    # Build jax layer
    jax_p = glam.GlamUniTransformerLmHParams(
        name='model',
        vocab_size=vocab_size,
        num_transformer_layers=num_layers,
        moe=True,
        model_dim=d_model,
        ff_dim=ff_dim,
        moe_hidden_dim=ff_dim,
        attention_num_heads=num_heads,
        attention_key_value_dim=d_model // num_heads,
        attention_extra_logit=0.0,
        use_tgt_labels_size_as_loss_denominator=True,
        moe_load_balance_loss_weight=0.01,
        z_loss_weight=1e-4,
        c_dim=c_dim,
        e_dim=e_dim)
    assert jax_p.packed_input
    jax_layer = instantiate(jax_p)

    builder_p = gshard_builder.DenseBuilder.Params().Set(
        num_groups=1,
        second_expert_policy='all',
        relative_attention_type='bias',
        model_dim=d_model,
        attention_key_value_dim=d_model // num_heads,
        attention_num_heads=num_heads,
        attention_combine_dims=True,
        c_dim=c_dim,
        capacity_factor=None,
        attention_extra_logit=0.0,
        e_dim=e_dim,
        moe_hidden_dim=ff_dim,
        ff_dim=ff_dim)
    tf_layer = gshard_builder.UniTransformer.Params().Set(
        name='model',
        num_transformer_layers=num_layers,
        builder=builder_p,
        vocab_size=vocab_size,
        sequence_length=length,
        label_smoothing=0,
        aux_loss_coef=0.01,
        z_loss=1e-4,
        use_tgt_labels_size_as_loss_denominator=True,
        positional_embedding=False,
        gated_gelu=True,
        moe=True).Instantiate()

    # Build Jax Inputs
    np.random.seed(42)
    npy_ids = np.random.randint(0, vocab_size - 1, [batch, length])
    jax_ids = jnp.asarray(npy_ids)
    npy_paddings = np.array([[0, 0, 1], [0, 0, 1]], dtype=np.float32)

    jax_paddings = jnp.asarray(npy_paddings)
    npy_segment_ids = np.array([[1, 2, 0], [1, 1, 0]], dtype=np.int32)
    npy_segment_pos = np.array([[0, 0, 0], [0, 1, 0]], dtype=np.int32)
    npy_labels = np.roll(npy_ids, -1, axis=1)
    jax_labels = jnp.asarray(npy_labels)
    jax_seg_ids = jnp.asarray(npy_segment_ids)
    jax_seg_pos = jnp.asarray(npy_segment_pos)
    jax_label_weighs = jnp.asarray([[1, 1, 0], [1, 1, 0]])

    # Build TF Inputs
    tf_tgt_inputs = py_utils.NestedMap(
        ids=tf.convert_to_tensor(npy_ids, dtype=tf.int32),
        labels=tf.convert_to_tensor(npy_labels, dtype=tf.int32),
        segment_ids=tf.convert_to_tensor(npy_segment_ids, dtype=tf.int32),
        segment_pos=tf.convert_to_tensor(npy_segment_pos, dtype=tf.int32))
    tf_inputs = py_utils.NestedMap(tgt=tf_tgt_inputs)

    # Compute jax outputs
    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=42)
      prng_key, init_key = jax.random.split(prng_key)
      prng_key, random_key = jax.random.split(prng_key)
      jax_vars = jax_layer.init({
          PARAMS: init_key,
          RANDOM: random_key
      },
                                jax_ids,
                                jax_paddings,
                                labels=py_utils.NestedMap(
                                    class_ids=jax_labels,
                                    class_weights=jax_label_weighs,
                                ),
                                segment_ids=jax_seg_ids,
                                segment_pos=jax_seg_pos)
      jax_outputs, _ = jax_layer.apply(
          jax_vars,
          jax_ids,
          jax_paddings,
          labels=py_utils.NestedMap(
              class_ids=jax_labels,
              class_weights=jax_label_weighs,
          ),
          segment_ids=jax_seg_ids,
          segment_pos=jax_seg_pos,
          rngs={RANDOM: random_key},
          mutable=[base_layer.AUX_LOSS])

    # Copy jax vars to tf ones.
    tf_theta = tf_layer.theta.DeepCopy()
    jax_vars_nmap = py_utils.NestedMap.FromNestedDict(jax_vars[PARAMS])

    # GShardBuilder softmax weight use self.vars rather than theta.
    tf_layer.vars.dec_emb.w.embedding.assign(jax_vars_nmap.softmax.embedding.w)
    tf_theta.dec_emb.w.embedding = jax_vars_nmap.softmax.embedding.w
    tf_theta.dec.final_layer_norm.w.scale = jax_vars_nmap.final_ln.scale
    jax_layer_0_var = tf.nest.map_structure(
        lambda v: jnp.squeeze(jnp.split(v, 2)[0], axis=0),
        jax_vars_nmap.transformer.repeat.sub.x_layers_0)
    tf_theta.dec.layer_000.ln.w.scale = jax_layer_0_var.layer_norm.scale
    jax_atten_var = jax_layer_0_var.self_attention
    tf_atten_var = tf_theta.dec.layer_000.dec_self_attention
    tf_atten_var.w.wk = jax_atten_var.key.w
    tf_atten_var.w.wq = jax_atten_var.query.w
    tf_atten_var.w.wv = jax_atten_var.value.w
    tf_atten_var.w.wo = jax_atten_var.post.w
    tf_atten_var.wrb.wrb = jax_atten_var.relative_bias.wrb

    jax_moe_var = jax_layer_0_var.ff_layer
    tf_theta.dec.layer_001.ln.w.scale = jax_moe_var.layer_norm.scale
    tf_theta.dec.layer_001.moe.ffw.top_2_gating.w = jax_moe_var.gate
    tf_theta.dec.layer_001.moe.moe.wi = jax_moe_var.wi_0
    tf_theta.dec.layer_001.moe.moe.wo = jax_moe_var.wo_0

    jax_layer_1_var = tf.nest.map_structure(
        lambda v: jnp.squeeze(jnp.split(v, 2)[0], axis=0),
        jax_vars_nmap.transformer.repeat.sub.x_layers_1)
    tf_theta.dec.layer_002.ln.w.scale = jax_layer_1_var.layer_norm.scale
    jax_atten_var = jax_layer_1_var.self_attention
    tf_atten_var = tf_theta.dec.layer_002.dec_self_attention
    tf_atten_var.w.wk = jax_atten_var.key.w
    tf_atten_var.w.wq = jax_atten_var.query.w
    tf_atten_var.w.wv = jax_atten_var.value.w
    tf_atten_var.w.wo = jax_atten_var.post.w
    tf_atten_var.wrb.wrb = jax_atten_var.relative_bias.wrb

    jax_ffn_var = jax_layer_1_var.ff_layer
    tf_ffn_var = tf_theta.dec.layer_003.dense_relu_dense
    tf_ffn_var.w.wi_0 = jax_ffn_var.ffn_layer1_gate.linear.w
    tf_ffn_var.w.wi_1 = jax_ffn_var.ffn_layer1.linear.w
    tf_ffn_var.w.wo = jax_ffn_var.ffn_layer2.linear.w
    tf_theta.dec.layer_003.ln.w.scale = jax_ffn_var.layer_norm.scale

    jax_layer_2_var = tf.nest.map_structure(
        lambda v: jnp.squeeze(jnp.split(v, 2)[1], axis=0),
        jax_vars_nmap.transformer.repeat.sub.x_layers_0)
    tf_theta.dec.layer_004.ln.w.scale = jax_layer_2_var.layer_norm.scale
    jax_atten_var = jax_layer_2_var.self_attention
    tf_atten_var = tf_theta.dec.layer_004.dec_self_attention
    tf_atten_var.w.wk = jax_atten_var.key.w
    tf_atten_var.w.wq = jax_atten_var.query.w
    tf_atten_var.w.wv = jax_atten_var.value.w
    tf_atten_var.w.wo = jax_atten_var.post.w
    tf_atten_var.wrb.wrb = jax_atten_var.relative_bias.wrb

    jax_moe_var = jax_layer_2_var.ff_layer
    tf_theta.dec.layer_005.ln.w.scale = jax_moe_var.layer_norm.scale
    tf_theta.dec.layer_005.moe.ffw.top_2_gating.w = jax_moe_var.gate
    tf_theta.dec.layer_005.moe.moe.wi = jax_moe_var.wi_0
    tf_theta.dec.layer_005.moe.moe.wo = jax_moe_var.wo_0

    jax_layer_3_var = tf.nest.map_structure(
        lambda v: jnp.squeeze(jnp.split(v, 2)[1], axis=0),
        jax_vars_nmap.transformer.repeat.sub.x_layers_1)
    tf_theta.dec.layer_006.ln.w.scale = jax_layer_3_var.layer_norm.scale
    jax_atten_var = jax_layer_3_var.self_attention
    tf_atten_var = tf_theta.dec.layer_006.dec_self_attention
    tf_atten_var.w.wk = jax_atten_var.key.w
    tf_atten_var.w.wq = jax_atten_var.query.w
    tf_atten_var.w.wv = jax_atten_var.value.w
    tf_atten_var.w.wo = jax_atten_var.post.w
    tf_atten_var.wrb.wrb = jax_atten_var.relative_bias.wrb

    jax_ffn_var = jax_layer_3_var.ff_layer
    tf_ffn_var = tf_theta.dec.layer_007.dense_relu_dense
    tf_ffn_var.w.wi_0 = jax_ffn_var.ffn_layer1_gate.linear.w
    tf_ffn_var.w.wi_1 = jax_ffn_var.ffn_layer1.linear.w
    tf_ffn_var.w.wo = jax_ffn_var.ffn_layer2.linear.w
    tf_theta.dec.layer_007.ln.w.scale = jax_ffn_var.layer_norm.scale

    tf_theta = test_utils.to_tf_nmap(tf_theta)

    # Compute TF outputs
    tf_out, _ = tf_layer.FProp(tf_theta, tf_inputs)
    self.assertAllClose(
        test_utils.to_np(jax_outputs.total_loss),
        test_utils.to_np(tf_out['loss'][0]))

  @parameterized.parameters([True, False])
  def test_glam_unitransformer_extendstep(self, moe):
    batch = 1
    length = 3
    d_model = 6
    num_heads = 2
    vocab_size = 16
    ff_dim = 8
    c_dim = 3
    e_dim = 4
    num_layers = 4
    # Build jax layer
    transformer_lm = instantiate(
        glam.GlamUniTransformerLmHParams(
            name='model',
            vocab_size=vocab_size,
            num_transformer_layers=num_layers,
            moe=moe,
            model_dim=d_model,
            ff_dim=ff_dim,
            moe_hidden_dim=ff_dim,
            attention_num_heads=num_heads,
            attention_key_value_dim=d_model // num_heads,
            attention_extra_logit=0.0,
            use_tgt_labels_size_as_loss_denominator=True,
            moe_load_balance_loss_weight=0.01,
            num_groups=1,
            z_loss_weight=1e-4,
            c_dim=c_dim,
            e_dim=e_dim))
    npy_inputs = np.random.randint(
        vocab_size, size=(batch, length)).astype('int32')
    inputs = jnp.asarray(npy_inputs)
    context_params = base_layer.JaxContext.HParams(do_eval=True)
    with base_layer.JaxContext.new_context(hparams=context_params):
      prng_key = jax.random.PRNGKey(seed=123)
      prng_key, init_key = jax.random.split(prng_key)
      prng_key, random_key = jax.random.split(prng_key)
      initial_vars = transformer_lm.init(
          {
              PARAMS: init_key,
              RANDOM: random_key
          },
          inputs,
          jnp.zeros_like(inputs),
      )
      fprop_outputs = transformer_lm.apply(
          initial_vars,
          inputs,
          jnp.zeros_like(inputs),
          rngs={RANDOM: random_key},
          method=transformer_lm.__call__)
      _, decoder_state = transformer_lm.apply(
          initial_vars,
          jnp.zeros_like(inputs),
          jnp.ones_like(inputs),
          rngs={RANDOM: random_key},
          method=transformer_lm.__call__,
          mutable=[DECODE_CACHE])
      logits = fprop_outputs.logits
      updated_vars = py_utils.MergeDictsWithValueCheck(decoder_state,
                                                       initial_vars)
      for t in range(length):
        xent_output, decoder_state = transformer_lm.apply(
            updated_vars,
            inputs[:, t],
            rngs={RANDOM: random_key},
            mutable=[DECODE_CACHE],
            method=transformer_lm.extend_step)
        updated_vars = py_utils.MergeDictsWithValueCheck(
            decoder_state, initial_vars)
        self.assertAllClose(
            logits[:, t, :], xent_output.logits, rtol=1e-5, atol=1e-5)

  # TODO(wangtao): Fix this test for moe models.
  @parameterized.parameters([False])
  def test_glam_unitransformer_fprop_update_state_extendstep(self, moe):
    batch = 1
    length = 4
    prefix_len = 2
    d_model = 6
    num_heads = 2
    vocab_size = 16
    ff_dim = 8
    c_dim = 3
    e_dim = 4
    num_layers = 4
    # Build jax layer
    transformer_lm = instantiate(
        glam.GlamUniTransformerLmHParams(
            name='model',
            vocab_size=vocab_size,
            num_transformer_layers=num_layers,
            moe=moe,
            model_dim=d_model,
            ff_dim=ff_dim,
            moe_hidden_dim=ff_dim,
            attention_num_heads=num_heads,
            attention_key_value_dim=d_model // num_heads,
            attention_extra_logit=0.0,
            use_tgt_labels_size_as_loss_denominator=True,
            moe_load_balance_loss_weight=0.01,
            num_groups=1,
            z_loss_weight=1e-4,
            c_dim=c_dim,
            e_dim=e_dim))
    npy_inputs = np.random.randint(
        vocab_size, size=(batch, length)).astype('int32')
    inputs = jnp.asarray(npy_inputs)
    prefix = jnp.zeros_like(inputs)
    prefix = jax.lax.dynamic_update_slice(prefix, inputs[:, 0:prefix_len],
                                          [0, 0])
    context_params = base_layer.JaxContext.HParams(do_eval=True)
    with base_layer.JaxContext.new_context(hparams=context_params):
      prng_key = jax.random.PRNGKey(seed=123)
      prng_key, init_key = jax.random.split(prng_key)
      prng_key, random_key = jax.random.split(prng_key)
      initial_vars = transformer_lm.init({
          PARAMS: init_key,
          RANDOM: random_key
      }, inputs, jnp.zeros_like(inputs))
      # Run fprop without decode state on all the inputs.
      fprop_outputs = transformer_lm.apply(
          initial_vars,
          inputs,
          jnp.zeros_like(inputs),
          rngs={RANDOM: random_key},
          method=transformer_lm.__call__)
      logits = fprop_outputs.logits
      # Init states.
      _, decoder_state = transformer_lm.apply(
          initial_vars,
          prefix,
          jnp.zeros_like(prefix),
          start_time_step=prefix_len,
          rngs={RANDOM: random_key},
          method=transformer_lm.__call__,
          mutable=[DECODE_CACHE])

      # Run fprop on prefix only and update the decode states.
      updated_vars = py_utils.MergeDictsWithValueCheck(decoder_state,
                                                       initial_vars)

      # Run extend start from prefix_len and compares the result at each step.
      for t in range(prefix_len, length):
        xent_output, decoder_state = transformer_lm.apply(
            updated_vars,
            inputs[:, t],
            rngs={RANDOM: random_key},
            mutable=[DECODE_CACHE],
            method=transformer_lm.extend_step)
        updated_vars = py_utils.MergeDictsWithValueCheck(
            decoder_state, initial_vars)
        self.assertAllClose(
            logits[:, t, :], xent_output.logits, rtol=1e-5, atol=1e-5)

  def _set_transformer_lm_sharding_params_legacy(self, lm_p, *, replica_axis,
                                                 data_axis, mdl_axis,
                                                 ici_mesh_shape, dcn_mesh_shape,
                                                 mesh_axis_names):
    # In the following, model weights are layed out on the [data_axis, mdl_axis]
    # 2d mesh. Model weights are always replicated over the replica_axis mesh
    # axis.
    #
    # The batch axis of the activations are always sharded over the combination
    # of (replica_axis, data_axis).
    lm_p.ici_mesh_shape = ici_mesh_shape
    lm_p.dcn_mesh_shape = dcn_mesh_shape
    lm_p.mesh_axis_names = mesh_axis_names
    # TODO(zhangqiaorjc): Remove once scan no longer needs explicit weight
    # sharding annotations.
    lm_p.stacked_transformer_tpl.ici_mesh_shape = lm_p.ici_mesh_shape
    lm_p.stacked_transformer_tpl.dcn_mesh_shape = lm_p.dcn_mesh_shape
    lm_p.stacked_transformer_tpl.mesh_axis_names = mesh_axis_names

    # We assume activation batch is split on both replica_axis and data_axis.
    batch_split = (replica_axis, data_axis)

    if (lm_p.position_emb_tpl is not None and lm_p.position_emb_tpl.cls
        == embedding_softmax.TrainablePositionalEmbedding):
      pos_emb_p = lm_p.position_emb_tpl
      pos_emb_p.weight_split_dims_mapping.wt = [data_axis, mdl_axis]
      pos_emb_p.activation_split_dims_mapping.out = [data_axis, mdl_axis]

    # NGrammer embedding table is currently replicated.
    # TODO(aurkor): Explore different sharding configs for the table.
    # n-gram table is of shape [ngram_vocab_size, embedding_dims].
    if lm_p.ngrammer_tpl is not None:
      ngrammer_p = lm_p.ngrammer_tpl
      ngrammer_p.weight_split_dims_mapping.wt = [mdl_axis, data_axis]

    softmax_p = lm_p.softmax_tpl
    if softmax_p.cls == embedding_softmax.GShardSharedEmbeddingSoftmax:
      # Softmax weight is of shape [vocab_size, input_dim].
      softmax_p.weight_split_dims_mapping.wt = [mdl_axis, data_axis]
    else:
      # Softmax weight is of shape [input_dim, vocab_size].
      softmax_p.weight_split_dims_mapping.wt = [data_axis, mdl_axis]
      softmax_p.lookup_style = 'matmul'
    # During training, softmax output is 3d.
    softmax_p.activation_split_dims_mapping.out = [batch_split, None, mdl_axis]

    softmax_p.activation_split_dims_mapping.emb_out_split_dims_mapping = [
        batch_split, None, mdl_axis
    ]

    if lm_p.stacked_transformer_tpl.cls == transformers.PipelinedTransformer:
      stacked_transformer_tpl = lm_p.stacked_transformer_tpl.pipeline_stage
    else:
      stacked_transformer_tpl = lm_p.stacked_transformer_tpl

    if stacked_transformer_tpl.cls == transformers.StackedTransformer:
      xformer_p = stacked_transformer_tpl.transformer_layer_params_tpl
    elif stacked_transformer_tpl.cls == transformers.StackedTransformerRepeated:
      xformer_p = stacked_transformer_tpl.block.transformer_layer_params_tpl
    else:
      assert False, f'{stacked_transformer_tpl.cls} not supported.'

    xformer_p.tr_atten_tpl.activation_split_dims_mapping.blnh = [
        batch_split, None, mdl_axis, None
    ]
    xformer_p.tr_atten_tpl.activation_split_dims_mapping.bld = [
        batch_split, None, mdl_axis
    ]
    # Attention project weight matrix is of shape [data_dim, num_heads,
    # dim_per_head].
    xformer_p.tr_atten_tpl.weight_split_dims_mapping.proj = [
        data_axis, mdl_axis, None
    ]
    # Sharding for depth-wise conv weights. Depth-wise conv weights are of shape
    # [num_heads, dim_per_head].
    xformer_p.tr_atten_tpl.weight_split_dims_mapping.dconv = [mdl_axis, None]

    ffw_wp = xformer_p.tr_fflayer_tpl.weight_split_dims_mapping
    ffw_ap = xformer_p.tr_fflayer_tpl.activation_split_dims_mapping
    ffw_wp.ffn0 = [data_axis, mdl_axis]
    ffw_wp.ffn1 = [mdl_axis, data_axis]
    ffw_ap.ffn0 = [batch_split, None, mdl_axis]
    ffw_ap.ffn1 = [batch_split, None, mdl_axis]

    # MoE
    # Following GShard sharding settings for large 2D sharded models.
    #
    # TODO(lepikhin): Provide better reference.
    #   lingvo/core/gshard_builder.py and specifically MoE splits
    #     emh_split=[0, -1, 1],
    #     ehm_split=[0, 1, -1],
    #     egcm_split=[0, -1, -1, 1],
    #     gecm_split=[0, -1, -1, 1],
    #     gsec_split=[0, -1, -1, -1],
    # for mesh with 2 dimensions.
    if stacked_transformer_tpl.cls == transformers.StackedTransformer:
      moe_p = stacked_transformer_tpl.moe_layer_tpl
    elif stacked_transformer_tpl.cls == transformers.StackedTransformerRepeated:
      moe_p = stacked_transformer_tpl.block.moe_layer_tpl
    else:
      assert False, f'{stacked_transformer_tpl.cls} not supported.'
    # Weights
    moe_wp = moe_p.weight_split_dims_mapping
    # TODO(lepikhin): RET_CHECK with [data_axis, None] http://b/209481545
    moe_wp.me = [None, None]  # replicated
    moe_wp.emh = [data_axis, None, mdl_axis]
    moe_wp.ehm = [data_axis, mdl_axis, None]
    # Activations
    moe_ap = moe_p.activation_split_dims_mapping
    moe_ap.gsm = [data_axis, None, mdl_axis]
    moe_ap.gs = [data_axis, None]
    moe_ap.gsec = [data_axis, None, None, None]  # dispatch and combine tensors
    moe_ap.gecs = [data_axis, None, None, None]  # dispatch and combine tensors
    moe_ap.gec = [data_axis, None, None]  # dispatch and combine tensors
    moe_ap.egcm = [data_axis, None, None, mdl_axis]
    moe_ap.egch = [data_axis, None, None, mdl_axis]
    moe_ap.gecm = [data_axis, None, None, mdl_axis]

    return lm_p

  def test_transformer_sharding_refactor(self):
    seq_len = 512

    embedding_options = [
        embedding_softmax.TrainablePositionalEmbedding,
        embedding_softmax.RotaryPositionalEmbedding,
        embedding_softmax.PositionalEmbedding,
        None,
    ]

    ngrammer_options = [ngrammer.Ngrammer, ngrammer.VQNgrammer, None]
    tfm_options = [
        transformers.StackedTransformer,
        transformers.StackedTransformerRepeated,
    ]

    softmax_options = [
        embedding_softmax.GShardSharedEmbeddingSoftmax,
        embedding_softmax.SharedEmbeddingSoftmax,
    ]

    for emb_cls, ngram_cls, tfm_cls, softmax_cls in itertools.product(
        embedding_options, ngrammer_options, tfm_options, softmax_options):

      if emb_cls is not None:
        position_emb_tpl = emb_cls.HParams()
        if emb_cls == embedding_softmax.TrainablePositionalEmbedding:
          position_emb_tpl.max_seq_length = seq_len
      else:
        position_emb_tpl = None

      if ngram_cls is not None:
        ngrammer_tpl = ngram_cls.HParams()
      else:
        ngrammer_tpl = None

      stacked_transformer_tpl = tfm_cls.HParams()
      softmax_tpl = softmax_cls.HParams()

      lm_p = transformer_models.TransformerLm.HParams(
          name='bert_lm',
          model_dims=32,
          vocab_size=52,
          position_emb_tpl=position_emb_tpl,
          ngrammer_tpl=ngrammer_tpl,
          stacked_transformer_tpl=stacked_transformer_tpl,
          softmax_tpl=softmax_tpl,
      )

      replica_axis, data_axis, model_axis = ('replica', 'data', 'model')

      # Unnecessary to test the setting of mesh_shape and mesh_axis_names.
      # In pax, as long as these two are set at the model level, all submodules
      # will inherit the same values.

      lm_p_1 = transformer_models.TransformerLm.set_sharding_params_v1(
          lm_p.clone(),
          replica_axis=replica_axis,
          data_axis=data_axis,
          mdl_axis=model_axis,
          ici_mesh_shape=None,
          dcn_mesh_shape=None,
          mesh_axis_names=None,
          training_optimized=True)

      lm_p_2 = self._set_transformer_lm_sharding_params_legacy(
          lm_p.clone(),
          replica_axis=replica_axis,
          data_axis=data_axis,
          mdl_axis=model_axis,
          ici_mesh_shape=None,
          dcn_mesh_shape=None,
          mesh_axis_names=None)

      lm_p_1_txt = base_hyperparams.nested_struct_to_text(lm_p_1)
      lm_p_2_txt = base_hyperparams.nested_struct_to_text(lm_p_2)
      print('lm_p_1', lm_p_1_txt)
      print('lm_p_2', lm_p_2_txt)
      # Assert the sharding strategy of the two lms are the same!
      self.assertEqual(lm_p_1_txt, lm_p_2_txt)


if __name__ == '__main__':
  absltest.main()
