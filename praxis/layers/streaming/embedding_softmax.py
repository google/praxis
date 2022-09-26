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

"""Streaming aware Embedding layers."""

from __future__ import annotations
import math
from jax import numpy as jnp
import numpy as np

from praxis import base_layer
from praxis import py_utils
from praxis import pytypes
from praxis.layers import embedding_softmax
from praxis.layers.streaming import streaming_base

NestedMap = py_utils.NestedMap
WeightHParams = base_layer.WeightHParams
JTensor = pytypes.JTensor


class PositionalEmbedding(embedding_softmax.PositionalEmbedding,  # pytype: disable=signature-mismatch
                          streaming_base.StreamingBase):
  """Streaming aware PositionalEmbedding layer."""

  @classmethod
  def get_right_context(cls, hparams: PositionalEmbedding.HParams) -> int:
    return 0

  @classmethod
  def get_stride(cls, hparams: PositionalEmbedding.HParams) -> int:
    return 1

  def init_states(self,
                  batch_size: int,
                  with_paddings=True):
    self._update_streaming_state('position', jnp.asarray(0, dtype=jnp.uint64))

  def streaming_step(
      self,
      inputs: NestedMap,
  ) -> JTensor:
    """Generates in streaming a JTensor of sinusoids with different frequencies.

    Args:
      inputs: NestedMap with
        seq_length - sequence length of the embeddings to be generated.
        position - optional position JTensor which denotes the position of each
        token in the sequence. It has to be None

    Returns:
      a JTensor of shape [batch, seq_length, embedding_dim] if position JTensor
      is specified, else of shape [1, seq_length, embedding_dim].
    """
    if 'position' in inputs:
      assert inputs.position is None
    assert inputs.seq_length is not None

    # TODO(b/248751079) It can overflow:
    position = self.get_streaming_state('position')
    new_position = inputs.seq_length + position
    self._update_streaming_state('position', new_position)

    position = jnp.arange(
        inputs.seq_length, dtype=jnp.float32)[jnp.newaxis, :] + position

    return super().__call__(position=position)
