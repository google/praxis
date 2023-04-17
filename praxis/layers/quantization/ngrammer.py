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

"""Quantized Ngrammer layers."""
from jax import numpy as jnp
from praxis import base_layer
from praxis import pax_fiddle
from praxis.layers import ngrammer
from praxis.layers import normalizations
from praxis.layers.quantization import embedding_softmax as quantized_embedding_softmax
from praxis.layers.quantization import quantization_hparams


QuantizationHParams = quantization_hparams.QuantizationHParams


WeightHParams = base_layer.WeightHParams


sub_config_field = base_layer.sub_config_field


class Ngrammer(ngrammer.Ngrammer):
  """Quantized Ngrammer."""

  quantization: QuantizationHParams = sub_config_field(QuantizationHParams)

  def setup(self) -> None:
    """Constructs an instance which looks up ngrams."""

    if self.concat_ngrams:
      # The ngram_emb_dim must be smaller than dim_per_head.
      assert self.ngram_emb_dim <= self.dim_per_head
    else:
      # If not concatenating ngram embeddings, check the dims are compatible.
      assert self.ngram_emb_dim == self.dim_per_head

    # Create a separate layer norm per head for embedding normalization.
    # Create a separate layer norm per head for ngram embedding normalization.
    emb_layer_norm_p = []
    ngram_emb_layer_norm_p = []
    ngram_emb_table_p = []
    for i in range(self.num_heads):
      layer_norm_p = pax_fiddle.Config(normalizations.LayerNorm).clone()
      layer_norm_p.dim = self.dim_per_head
      layer_norm_p.name = f'layer_norm_{i}'

      emb_layer_norm_p.append(layer_norm_p)
      ngram_layer_norm_p = pax_fiddle.Config(normalizations.LayerNorm).clone()
      ngram_layer_norm_p.dim = self.ngram_emb_dim
      ngram_emb_layer_norm_p.append(ngram_layer_norm_p)

      # Create embedding table for ngram lookup.
      embedding_p = pax_fiddle.Config(
          quantized_embedding_softmax.Embedding,
          quantization=self.quantization,
      ).clone()
      embedding_p.name = f'embedding_{i}'
      embedding_p.num_classes = self.ngram_vocab_size
      embedding_p.input_dims = self.ngram_emb_dim
      embedding_p.params_init = self.params_init
      # Copy sharding annotations.
      embedding_p.weight_split_dims_mapping = self.weight_split_dims_mapping
      ngram_emb_table_p.append(embedding_p)

    self.create_children('emb_layer_norm', emb_layer_norm_p)
    self.create_children('ngram_layer_norm', ngram_emb_layer_norm_p)
    self.create_children('ngram_table', ngram_emb_table_p)


class VQNgrammer(ngrammer.VQNgrammer):
  """Quantized VQNgrammer."""

  quantization: QuantizationHParams = sub_config_field(QuantizationHParams)

  def setup(self) -> None:
    """Constructs a VQ layer and an N-grammer layer."""

    if self.concat_ngrams:
      # The ngram_emb_dim must be smaller than dim_per_head.
      assert self.ngram_emb_dim <= self.dim_per_head
    else:
      # If not concatenating ngram embeddings, check the dims are compatible.
      assert self.ngram_emb_dim == self.dim_per_head

    # Create VQ layer.
    vq_layer_p = pax_fiddle.Config(
        ngrammer.VectorQuantization,
        num_clusters=self.num_clusters,
        num_heads=self.num_heads,
        dim_per_head=self.dim_per_head,
        decay=self.decay,
        epsilon=self.epsilon,
        params_init=self.params_init,
    )
    self.create_child('vq_layer', vq_layer_p)

    # Create the input id to cluster id cache.
    if self.unigram_vocab_size:
      input_id_to_cluster_id_cache = WeightHParams(
          shape=[self.unigram_vocab_size, self.num_heads],
          dtype=jnp.int32,
          init=base_layer.WeightInit.Constant(0),
      )
      self.create_variable(
          'input_id_to_cluster_id_cache',
          input_id_to_cluster_id_cache,
          trainable=False,
      )

    # Create N-gram lookup layer.
    ngram_layer_p = pax_fiddle.Config(
        Ngrammer,  # Quantized Ngrammer.
        quantization=self.quantization,
        ngram_vocab_size=self.ngram_vocab_size,
        unigram_vocab_size=self.num_clusters,
        ngram_emb_dim=self.ngram_emb_dim,
        concat_ngrams=self.concat_ngrams,
        num_heads=self.num_heads,
        dim_per_head=self.dim_per_head,
        params_init=self.params_init,
        weight_split_dims_mapping=self.weight_split_dims_mapping,
    )
    self.create_child('ngram_layer', ngram_layer_p)
