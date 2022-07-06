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

"""Tests for Praxis retrieval model layers."""

from absl.testing import absltest
import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import layers
from praxis import py_utils
from praxis import test_utils
from praxis.layers import activations
from praxis.layers import retrieval_models

RANDOM = base_layer.RANDOM
instantiate = base_layer.instantiate


class RetrievalModelsTest(test_utils.TestCase):
  """Unit tests for retrieval models."""

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  def _setup_stacked_encoder(self, num_layers, num_heads, model_dims,
                             hidden_dims):
    stacked_transformer_tpl = layers.StackedTransformer.HParams(
        model_dims=model_dims,
        hidden_dims=hidden_dims,
        num_layers=num_layers,
        num_heads=num_heads,
        cross_attention=False,
        mask_self_attention=False,
        dropout_prob=0.)
    lp = stacked_transformer_tpl.transformer_layer_params_tpl
    # Enable rotary position embedding for encoder.
    lp.tr_atten_tpl.use_rotary_position_emb = True

    lp.tr_fflayer_tpl.activation_tpl = activations.GELU.HParams()
    lp.tr_fflayer_tpl.use_gated_activation = True
    # Enable Gated Gelu activation.
    stacked_transformer_tpl.hidden_dims = (stacked_transformer_tpl.hidden_dims *
                                           2) // 3

    stacked_transformer_tpl.num_layers = 1
    repeated_stacked_transformer_tpl = (
        layers.StackedTransformerRepeated.HParams())
    repeated_stacked_transformer_tpl.block = stacked_transformer_tpl
    repeated_stacked_transformer_tpl.x_times = num_layers
    return repeated_stacked_transformer_tpl

  def _setup_stacked_decoder(self, num_layers, num_heads, model_dims,
                             hidden_dims):
    stacked_transformer_tpl = layers.StackedRetroTransformer.HParams(
        transformer_layer_params_tpl=layers.RetroTransformer.HParams(),
        model_dims=model_dims,
        hidden_dims=hidden_dims,
        num_layers=num_layers,
        num_heads=num_heads,
        cross_attention=False,
        mask_self_attention=True,
        dropout_prob=0.)
    lp = stacked_transformer_tpl.transformer_layer_params_tpl
    # Enable rotary position embedding for encoder.
    lp.tr_atten_tpl.use_rotary_position_emb = True

    lp.tr_fflayer_tpl.activation_tpl = activations.GELU.HParams()
    lp.tr_fflayer_tpl.use_gated_activation = True
    # Enable Gated Gelu activation.
    stacked_transformer_tpl.hidden_dims = (stacked_transformer_tpl.hidden_dims *
                                           2) // 3

    lp.cross_atten_tpl = lp.tr_atten_tpl.clone()
    lp.cross_atten_tpl.use_rotary_position_emb = False

    stacked_transformer_tpl.num_layers = 1
    repeated_stacked_transformer_tpl = (
        layers.StackedRetroTransformerRepeated.HParams())
    repeated_stacked_transformer_tpl.block = stacked_transformer_tpl
    repeated_stacked_transformer_tpl.x_times = num_layers
    return repeated_stacked_transformer_tpl

  def test_retro(self):
    batch = 2
    length = 3
    model_dim = 8
    num_heads = 2
    vocab_size = 16
    num_layers = 3
    # Build jax layer
    jax_p = layers.Retro.HParams(name='retro_test')
    jax_p.model_dims = model_dim
    jax_p.packed_input = False
    jax_p.encoder_stacked_transformer_tpl = self._setup_stacked_encoder(
        num_layers, num_heads, model_dim, model_dim * 2)
    jax_p.decoder_stacked_transformer_tpl = self._setup_stacked_decoder(
        num_layers, num_heads, model_dim, model_dim * 2)
    # Retrieval setup.
    jax_p.chunk_length = 64
    jax_p.num_neighbors = 2
    jax_p.neighbor_length = 5
    # Use DummyRetriever for testing.
    # TODO(yuancao): Replace with SCaM when ready.
    jax_p.retriever_tpl = retrieval_models.DummyRetriever.HParams(
        key_dim=model_dim,
        retrieve_value=retrieval_models.DummyRetrievedValue.SEQUENCE_OF_IDS,
        retrieve_id_length=jax_p.neighbor_length)

    # Disable position embeddings since we are using rotary.
    jax_p.position_emb_tpl = None

    # Separate input and target embeddings from softmax.
    jax_p.encoder_embedding_tpl = (
        layers.embedding_softmax.SharedEmbeddingSoftmax.HParams(
            num_classes=vocab_size, input_dims=model_dim))
    jax_p.decoder_embedding_tpl = (
        layers.embedding_softmax.SharedEmbeddingSoftmax.HParams(
            num_classes=vocab_size, input_dims=model_dim))

    # Softmax with scale sqrt depth.
    jax_p.softmax_tpl = (
        layers.embedding_softmax.SharedEmbeddingSoftmax.HParams(
            input_dims=model_dim, num_classes=vocab_size))

    jax_layer = instantiate(jax_p)
    # Build Jax Inputs
    np.random.seed(7232)
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
              'params': init_key,
              'random': random_key
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
          mutable=['non_trainable'])
    # [batch, length, vocab]
    self.assertEqual(jax_outputs.logits.shape, (2, 3, 16))


if __name__ == '__main__':
  absltest.main()
