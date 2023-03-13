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

"""Tests for Praxis transformer layers (Encoder Decoder)."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized
import fiddle as fdl
import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import pax_fiddle
from praxis import py_utils
from praxis import test_utils
from praxis.layers import embedding_softmax
from praxis.layers import ngrammer
from praxis.layers import transformer_models
from praxis.layers import transformers
import tensorflow.compat.v2 as tf

DECODE_CACHE = base_layer.DECODE_CACHE
instantiate = base_layer.instantiate


class TransformerModelsEncoderDecoderTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)
    tf.random.set_seed(123)

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
      encoder_ngrammer_params = pax_fiddle.Config(
          ngrammer.VQNgrammer,
          ngram_vocab_size=8,
          ngram_emb_dim=ngram_emb_dim,
          num_heads=num_heads,
          concat_ngrams=True,
          num_clusters=2,
          dim_per_head=dim_per_head,
      )
    if use_encoder_ngrams:
      encoder_ngrammer_params = pax_fiddle.Config(
          ngrammer.Ngrammer,
          ngram_vocab_size=16,
          unigram_vocab_size=vocab_size,
          ngram_emb_dim=ngram_emb_dim,
          num_heads=num_heads,
          concat_ngrams=True,
          dim_per_head=dim_per_head,
      )
    if use_decoder_vq_ngrams:
      decoder_ngrammer_params = pax_fiddle.Config(
          ngrammer.VQNgrammer,
          ngram_vocab_size=8,
          ngram_emb_dim=ngram_emb_dim,
          num_heads=num_heads,
          concat_ngrams=True,
          num_clusters=2,
          dim_per_head=dim_per_head,
      )
    if use_decoder_ngrams:
      decoder_ngrammer_params = pax_fiddle.Config(
          ngrammer.Ngrammer,
          ngram_vocab_size=16,
          unigram_vocab_size=vocab_size,
          ngram_emb_dim=ngram_emb_dim,
          num_heads=num_heads,
          concat_ngrams=True,
          dim_per_head=dim_per_head,
      )
    if use_post_attention_ngrammer:
      ngrammer_params = pax_fiddle.Config(
          ngrammer.Ngrammer,
          ngram_vocab_size=4,
          unigram_vocab_size=vocab_size,
          ngram_emb_dim=ngram_emb_dim,
          num_heads=num_heads,
          concat_ngrams=True,
          dim_per_head=dim_per_head,
      )
      post_attention_ngrammer_tpls = [ngrammer_params] * 2
    p = pax_fiddle.Config(
        transformer_models.TransformerEncoderDecoder,
        name='jax_transformer_encoder_decoder',
        model_dims=num_heads * dim_per_head,
        decoder_ngrammer_tpl=decoder_ngrammer_params,
        encoder_ngrammer_tpl=encoder_ngrammer_params,
        encoder_post_attention_ngrammer_tpls=post_attention_ngrammer_tpls,
        decoder_post_attention_ngrammer_tpls=post_attention_ngrammer_tpls,
    )

    # Encoder stack.
    if use_stacked_transformer_repeated:
      block_param = pax_fiddle.Config(
          transformers.StackedTransformer,
          num_layers=num_layers,
          num_heads=num_heads,
          model_dims=num_heads * dim_per_head,
          hidden_dims=num_heads * dim_per_head,
          mask_self_attention=False,
          fold_padding_with_segment_mask=True,
      )
      p.encoder_stacked_transformer_tpl = pax_fiddle.Config(
          transformers.StackedTransformerRepeated, block=block_param, x_times=1
      )
    else:
      p.encoder_stacked_transformer_tpl = pax_fiddle.Config(
          transformers.StackedTransformer,
          model_dims=num_heads * dim_per_head,
          hidden_dims=num_heads * dim_per_head,
          num_heads=num_heads,
          num_layers=num_layers,
          mask_self_attention=False,
          fold_padding_with_segment_mask=True,
      )

    # Decoder stack.
    if use_stacked_transformer_repeated:
      block_param = pax_fiddle.Config(
          transformers.StackedTransformer,
          num_layers=num_layers,
          num_heads=num_heads,
          model_dims=num_heads * dim_per_head,
          hidden_dims=num_heads * dim_per_head,
          mask_self_attention=True,
          fold_padding_with_segment_mask=True,
      )
      p.decoder_stacked_transformer_tpl = pax_fiddle.Config(
          transformers.StackedTransformerRepeated, block=block_param, x_times=1
      )
    else:
      p.decoder_stacked_transformer_tpl = pax_fiddle.Config(
          transformers.StackedTransformer,
          model_dims=num_heads * dim_per_head,
          hidden_dims=num_heads * dim_per_head,
          num_heads=num_heads,
          num_layers=num_layers,
          mask_self_attention=True,
          fold_padding_with_segment_mask=True,
      )

    if separate_encoder_embedding:
      p.encoder_embedding_tpl = pax_fiddle.Config(
          embedding_softmax.Embedding,
          num_classes=vocab_size,
          input_dims=num_heads * dim_per_head,
      )

    if separate_decoder_embedding:
      p.decoder_embedding_tpl = pax_fiddle.Config(
          embedding_softmax.Embedding,
          num_classes=vocab_size,
          input_dims=num_heads * dim_per_head,
      )

    # Softmax params.
    if separate_decoder_embedding:
      p.softmax_tpl = pax_fiddle.Config(
          embedding_softmax.FullSoftmax,
          input_dims=num_heads * dim_per_head,
          num_classes=vocab_size,
      )
    else:
      p.softmax_tpl = pax_fiddle.Config(
          embedding_softmax.SharedEmbeddingSoftmax,
          input_dims=num_heads * dim_per_head,
          num_classes=vocab_size,
      )

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
      updated_vars = py_utils.merge_dict(decoder_state, initial_vars)
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
        updated_vars = py_utils.merge_dict(decoder_state, initial_vars)
        self.assertAllClose(logits[:, t, :], xent_output.logits, atol=2e-6)

if __name__ == '__main__':
  absltest.main()
