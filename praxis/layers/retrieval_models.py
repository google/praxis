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

"""Layers for retrieval language models."""

import enum
from typing import Optional
from jax import numpy as jnp
from praxis import base_layer
from praxis import py_utils
from praxis import pytypes
from praxis.layers import attentions
from praxis.layers import transformer_models

BaseHParams = base_layer.BaseLayer.HParams
NestedMap = py_utils.NestedMap
JTensor = pytypes.JTensor


class DummyRetrievedValue(enum.Enum):
  """Types of the dummy retrieved values.

  Attributes:
    ALL_ZERO: Retrieved value vectors are all zero.
    SAME_AS_KEY: Retrieved value vectors are the same as keys.
    SEQUENCE_OF_IDS: Retrieved value vectors are a sequence of ids.
  """
  ALL_ZERO = 1
  SAME_AS_KEY = 2
  SEQUENCE_OF_IDS = 3


class DummyRetriever(base_layer.BaseLayer):
  """Dummy retriever for testing."""

  class HParams(BaseHParams):
    """Params for the dummy retriever.

    Attributes:
      retrieve_value: Specify the retrieved values, one of the options in
        DummyRetrievedValue.
      retrieve_id_length: For SEQUENCE_OF_IDS: length of retrieved value id
        sequence.
    """
    key_dim: int = 128
    value_dim: int = 128
    retrieve_value: DummyRetrievedValue = DummyRetrievedValue.ALL_ZERO
    retrieve_id_length: int = 5

  def compute_keys(self, inputs: JTensor):
    # TODO(yuancao): This part is supposed to be replaced by an encoder like
    # BERT, to map inputs (token ids) to encodings.
    # Dummy keys (all zero) for inputs.
    p = self.hparams
    return jnp.zeros([inputs.shape[0], p.key_dim], p.fprop_dtype)

  def retrieve(self, keys: JTensor, inputs: JTensor, top_k):
    p = self.hparams
    if self.hparams.retrieve_value == DummyRetrievedValue.ALL_ZERO:
      value_shape = [top_k * keys.shape[0], p.value_dim]
      return jnp.zeros(value_shape, dtype=keys.dtype)
    elif self.hparams.retrieve_value == DummyRetrievedValue.SAME_AS_KEY:
      assert p.value_dim == p.key_dim, 'value_dim must be equal to key_dim.'
      return jnp.tile(keys, [top_k, 1])
    elif self.hparams.retrieve_value == DummyRetrievedValue.SEQUENCE_OF_IDS:
      inputs = jnp.resize(inputs, (inputs.shape[0], p.retrieve_id_length))
      return jnp.tile(inputs, [top_k, 1])

  def __call__(self, inputs: JTensor, input_paddings: JTensor, top_k: int):
    del input_paddings
    keys = self.compute_keys(inputs)
    return self.retrieve(keys, inputs, top_k)


class Retro(transformer_models.TransformerEncoderDecoder):
  """RETRO language model.

  Note that this model is inherited from the encoder-decoder model. However,
  the encoder is used only to encode retrieved neighbors, and the original input
  sequence is processed by the decoder rather than encoder. Therefore, although
  this model is an instance of EncoderDecoder, it behave just like a decoder-
  only model. The decoder is overloaded with cross-chunk attention layers
  enabled.
  """

  class HParams(transformer_models.TransformerEncoderDecoder.HParams):
    """Params for RETRO model.

    Attributes:
      chunk_length: Length of each chunk.
      num_neighbors: Number of retrieved neighbors for each chunk.
      neighbor_length: Length of the neighbor sequence.
      retriever_tpl: The retriever to use, as subclass of Retriever.
    """
    chunk_length: int = 64
    num_neighbors: int = 2
    neighbor_length: int = 5
    retriever_tpl: BaseHParams = base_layer.sub_config_field(
        DummyRetriever.HParams)

  def setup(self) -> None:
    super().setup()
    p = self.hparams
    if p.retriever_tpl is not None:
      retriever_params = p.retriever_tpl.clone()
      self.create_child('retriever', retriever_params)

  def _num_chunks(self, input_length):
    num_chunks = input_length // self.hparams.chunk_length
    if input_length % self.hparams.chunk_length:
      num_chunks += 1
    return num_chunks

  def _chunk_inputs(self, inputs: JTensor):
    """Split input sequence into chunks.

    Args:
      inputs: A JTensor of shape [batch, length], input ids.

    Returns:
      A tuple of chunks and associated chunk-level paddings.
    """
    chunk_length = self.hparams.chunk_length
    batch, input_length = inputs.shape
    num_chunks = self._num_chunks(input_length)
    pad_len = chunk_length * num_chunks - input_length
    paddings = jnp.zeros([batch, pad_len], dtype=inputs.dtype)
    # TODO(yuancao,zhouwk): Right now we are padding from right, it might be
    # helpful to switchto left-padding, especially for decoding.
    inputs_padded = jnp.concatenate([inputs, paddings], axis=-1)
    # [batch*num_chunk, chunk_len]
    chunks = jnp.reshape(
        jnp.transpose(
            jnp.array(jnp.split(inputs_padded, num_chunks, axis=-1)),
            [1, 0, 2]), [-1, chunk_length])
    # TODO(yuancao): Handle chunk paddings properly.
    chunk_paddings = jnp.zeros_like(chunks, dtype=self.fprop_dtype)
    return chunks, chunk_paddings

  def _retrieve_neighbors(self, chunk, chunk_paddings):
    p = self.hparams
    # [batch, num_neighbors, neighbor_length]
    neighbors = self.retriever(chunk, chunk_paddings, p.num_neighbors)
    return jnp.reshape(neighbors, [-1, p.neighbor_length])

  def _encode_neighbors(self, inputs: JTensor):
    """Retrieve neighbors for inputs, then encode each retrieved chunks.

    Args:
      inputs: A JTensor of shape [batch, length], input ids.

    Returns:
      A tuple of chunk encodings and associated paddings.
    """
    p = self.hparams
    batch, input_length = inputs.shape
    num_chunks = self._num_chunks(input_length)
    # Get retrieved examples.
    input_chunks, chunk_paddings = self._chunk_inputs(inputs)
    # [batch*num_chunks*num_neighbors, neighbor_length]
    chunk_neighbors = self._retrieve_neighbors(input_chunks, chunk_paddings)
    # TODO(yuancao): Handle paddings for neighbors.
    neighbor_paddings = jnp.zeros_like(chunk_neighbors, dtype=self.fprop_dtype)
    # [batch*num_chunks*num_neighbors, neighbor_length, dim]
    neighbor_encodings = self.encode(chunk_neighbors, neighbor_paddings)
    neighbor_encodings = jnp.reshape(
        neighbor_encodings,
        [batch, num_chunks, p.num_neighbors, p.neighbor_length, -1])
    neighbor_paddings = jnp.reshape(
        neighbor_paddings,
        [batch, num_chunks, p.num_neighbors, p.neighbor_length])
    return neighbor_encodings, neighbor_paddings

  def _decode(self, inputs: JTensor, paddings: JTensor,
              neighbor_encodings: JTensor, neighbor_paddings: JTensor):
    """Decode inputs.

    Args:
      inputs: Input ids. An int32 JTensor of shape [batch, time].
      paddings: A 0/1 JTensor of shape [batch, time] with 1 denoting padding.
      neighbor_encodings: A JTensor, encodings of the retrieved neighbor chunks.
      neighbor_paddings: A JTensor, paddings for neighbor_encodings.

    Returns:
      Outputs of the decoder layers, [batch, time, dim]
    """
    # TODO(yuancao):
    # The current implementation is just a placeholder that reuses the
    # standard decoder only to make the model go through. It will be replaced
    # by RETRO-specific decoder shortly!

    input_emb = self.decoder_embedding_lookup.emb_lookup(inputs)
    segment_ids = jnp.asarray(1 - paddings, jnp.int32)
    target_segment_mask = attentions.causal_segment_mask(
        segment_ids, input_emb.dtype)
    neighbor_encodings = jnp.multiply(
        neighbor_encodings, 1.0 - jnp.expand_dims(neighbor_paddings, -1))
    # TODO(yuancao): Handle segment_pos properly.
    output = self.decoder(
        input_emb, paddings, target_segment_mask, neighbors=neighbor_encodings)
    return self.decoder_ln(output)

  def __call__(self,
               inputs: JTensor,
               paddings: JTensor,
               labels: Optional[NestedMap] = None,
               segment_ids: Optional[JTensor] = None,
               segment_pos: Optional[JTensor] = None,
               causal_attention_mask: Optional[JTensor] = None,
               start_time_step: int = 0) -> NestedMap:
    """Computes xent loss given the language model inputs.

    Args:
      inputs: Input ids. An int32 JTensor of shape [B, T].
      paddings: A 0/1 JTensor of shape [B, T] with 1 denoting padding.
      labels: A `.NestedMap` containing the following fields: class_weights, a
        JTensor with shape [batch, seqlen] containing weights for each target
        word. class_ids, a JTensor with shape [B, T] of int32 dtype containing
        the target class labels. class_probabilities, a JTensor with shape [B,
        T, V] of float values indicating class-membership probabilities.
      segment_ids: A JTensor of shape [B, T]. The segment that each token
        belongs to.
      segment_pos: A JTensor of shape [B, T]. The position of each token in a
        segment.
      causal_attention_mask: A JTensor of shape [B, T] where 1 indicates a token
        position with causal attention and 0 indicates bidirectional attention.
        This overrides part of the causal mask.
      start_time_step: Decode extend_step start time step. When decoding after
        prefix, start_time_step will be prefix_len - 1.

    Returns:
      Returns xent_output, where
      `xent_output` is a `.NestedMap` as defined by `SoftmaxLayer`'s return. In
      addition, per_sequence_xent is added which equal to the sum of xent loss
      for tokens in a sequence.
    """
    neighbor_encodings, neighbor_paddings = self._encode_neighbors(inputs)
    output = self._decode(inputs, paddings, neighbor_encodings,
                          neighbor_paddings)
    return self.compute_loss(output, labels)
