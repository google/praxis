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

"""Tests for Praxis streaming embedding and softmax layers."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from praxis import base_layer
from praxis import py_utils
from praxis.layers import embedding_softmax
from praxis.layers import streaming
from praxis.layers.streaming import test_utils

NestedMap = py_utils.NestedMap

instantiate = base_layer.instantiate


class StreamingConformersTest(test_utils.TestCase):

  @parameterized.parameters(
      (1, 10, 10, 1),
      (1, 1e5, 100, 25),
      (10, 20, 200, 50),
      (10, 1e5, 400, 100)
      )
  def test_streaming_position_embedding(self, min_timescale, max_timescale,
                                        seq_length, step):
    p_non_stream = embedding_softmax.PositionalEmbedding.HParams(
        name='pos_non_stream',
        embedding_dims=4,
        min_timescale=min_timescale,
        max_timescale=max_timescale)
    pos_non_stream = instantiate(p_non_stream)
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = pos_non_stream.init(prng_key, seq_length)
    output_non_stream = pos_non_stream.apply(initial_vars, seq_length)

    # Streaming aware layer
    p_stream = streaming.PositionalEmbedding.HParams(
        name='pos_stream')
    p_stream.copy_fields_from(p_non_stream)
    pos_stream = instantiate(p_stream)

    # Get streaming states.
    _, streaming_states = pos_stream.apply(
        initial_vars,
        batch_size=0,
        with_paddings=False,
        method=pos_stream.init_states,
        mutable=[base_layer.DECODE_CACHE])

    stream_outputs = None
    for _ in range(0, seq_length, step):
      in_nmap = NestedMap(seq_length=step)

      # Combine streaming state with model vars and run one streaming step:
      updated_vars = py_utils.MergeDictsWithValueCheck(
          streaming_states, initial_vars)
      output_step, streaming_states = pos_stream.apply(
          updated_vars,
          in_nmap,
          method=pos_stream.streaming_step,
          mutable=[base_layer.DECODE_CACHE])

      # Concatenate streaming output with the final non streaming output.
      if stream_outputs is None:
        stream_outputs = output_step
      else:
        stream_outputs = np.concatenate([stream_outputs, output_step], axis=1)

    self.assertAllClose(stream_outputs, output_non_stream,)


if __name__ == '__main__':
  absltest.main()
