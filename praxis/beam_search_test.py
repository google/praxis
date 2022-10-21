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

"""Unit tests for beam_search."""

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import beam_search
from praxis import py_utils
from praxis import test_utils
from praxis.layers import models
from praxis.layers import transformer_models

NestedMap = py_utils.NestedMap
BaseHParams = base_layer.BaseLayer.HParams
instantiate = base_layer.instantiate
LanguageModelType = transformer_models.LanguageModelType

RANDOM = base_layer.RANDOM
DECODE_CACHE = base_layer.DECODE_CACHE


class BeamSearchHelperTest(test_utils.TestCase):

  def test_shuffle_state_r3(self):
    cache_state = jnp.array([[[0, 0], [1, 1], [2, 2], [3, 3]]],
                            dtype=jnp.float32)
    hyp_ids = jnp.array([[0, 0, 2, 1]], dtype=jnp.int32)
    transformed_cache_state = beam_search.shuffle_state(cache_state, hyp_ids)

    self.assertArraysEqual(
        transformed_cache_state,
        np.array([[[0, 0], [0, 0], [2, 2], [1, 1]]], dtype=np.float32))

  def test_shuffle_state_r1(self):
    cache_state = jnp.array([0, 1, 2, 3], dtype=jnp.float32)
    hyp_ids = jnp.array([[0, 1], [0, 2], [2, 0], [1, 1]], dtype=jnp.int32)
    transformed_cache_state = beam_search.shuffle_state(cache_state, hyp_ids)

    self.assertArraysEqual(transformed_cache_state,
                           np.array([0, 1, 2, 3], dtype=np.float32))

  def test_broadcast_beam_dim(self):
    x = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
    self.assertArraysEqual(
        beam_search.broadcast_beam_dim(x, beam_dim=0, beam_size=2),
        np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]], dtype=np.int32))

  def test_update_topk_scores_with_eos(self):
    end_ids = jnp.array([[[0, 1, 0, 0], [4, 5, 0, 0]]], dtype=np.int32)
    cur_ids = jnp.array([[[0, 1, 2, 3], [4, 5, 6, 7]]], dtype=np.int32)

    end_lengths = jnp.array([[2, 2]], dtype=np.int32)
    cur_lengths = jnp.array([[4, 4]], dtype=np.int32)

    end_scores = jnp.array([[0, 1]], dtype=np.float32)
    cur_scores = jnp.array([[2, 3]], dtype=np.float32)

    end_scores_norm = jnp.array([[2, 0]], dtype=np.float32)
    cur_scores_norm = jnp.array([[3, 1]], dtype=np.float32)

    end_logprobs = jnp.array(
        [[[-0.1, -1.1, 0., 0.], [-1.3, -2.3, 0., 0.]]], dtype=np.float32)
    cur_logprobs = jnp.array(
        [[[-0.1, -1.2, -1.1, 0.], [-1., -2., -0.3, 0.]]], dtype=np.float32)

    (output_ids, output_lengths, output_scores, output_scores_norm,
     output_logprobs) = beam_search.update_topk_scores_with_eos(
         (end_ids, end_lengths, end_scores, end_scores_norm, end_logprobs),
         (cur_ids, cur_lengths, cur_scores, cur_scores_norm, cur_logprobs))

    self.assertArraysEqual(
        output_ids, np.array([[[0, 1, 2, 3], [0, 1, 0, 0]]], dtype=np.int32))
    self.assertArraysEqual(
        output_logprobs, np.array(
            [[[-0.1, -1.2, -1.1, 0.], [-0.1, -1.1, 0., 0.]]], dtype=np.float32))
    self.assertArraysEqual(output_lengths, np.array([[4, 2]], dtype=np.int32))
    self.assertArraysEqual(output_scores, np.array([[2, 0]], dtype=np.float32))
    self.assertArraysEqual(output_scores_norm,
                           np.array([[3, 2]], dtype=np.float32))


class MockLM(base_layer.BaseLayer):

  class HParams(BaseHParams):
    """Associated hyper-params for this layer class.

    Attributes:
      logits: results returned by extend_step(), of shape [max step, batch size,
        vocab size].
    """
    logits: Any = None
    model_type: LanguageModelType = LanguageModelType.CAUSAL

  def setup(self) -> None:
    p = self.hparams
    self._logits = jnp.array(p.logits, dtype=jnp.float32)

  def __call__(self, *args: Any, **kwargs: Any) -> None:
    self.put_variable(DECODE_CACHE, 'time_step', 0)

  def transform_decode_state(self, *args: Any, **kwargs: Any) -> None:
    pass

  def extend_step(
      self,
      inputs: Any,
      **kwargs: Any,
  ) -> Any:
    ret = NestedMap()
    time_step = self.get_variable(DECODE_CACHE, 'time_step')
    ret.logits = jnp.take(self._logits, inputs, axis=0)
    self.put_variable(DECODE_CACHE, 'time_step', time_step + 1)
    return ret


class BeamSearchTest(test_utils.TestCase):

  def _run_decode(self, decoder_p, logits, input_batch):
    p = models.LanguageModel.HParams(
        name='mock_lm',
        decoder_tpl=decoder_p.clone(),
        lm_tpl=MockLM.HParams(logits=logits))
    lang_model = instantiate(p)
    theta = NestedMap(lm_tpl=NestedMap())
    # We fix seed to 9 to get the desired prefix lengths below.
    prng_key = jax.random.PRNGKey(seed=9)
    results, _ = lang_model.apply(
        theta,
        input_batch,
        rngs={RANDOM: prng_key},
        method=lang_model.decode,
        mutable=[DECODE_CACHE])
    _, results, _ = results
    return results

  @parameterized.parameters(
      ([], [[[2, 3, 4, 0, 0], [2, 3, 3, 4, 0], [2, 3, 0, 3, 4], [2, 3, 1, 0, 4]
            ]], [[3, 4, 5, 5]], [[[3, 4]], [[3, 3], [3, 4]],
                                 [[3, 0], [0, 3], [3, 4]],
                                 [[3, 1], [1, 0], [0, 4]]]),
      ([1, 3], [[[2, 3, 4, 3, 0], [2, 3, 4, 0, 3], [2, 3, 4, 2, 3],
                 [2, 3, 3, 0, 0]]
               ], [[4, 5, 5, 3]], [[[3, 4], [4, 1]], [[3, 4], [4, 0], [0, 3]],
                                   [[3, 4], [4, 1], [1, 1]], [[3, 3]]]))
  def test_vanilla_beam_search_base(self, parse_tokens, target_output_ids,
                                    target_decode_ids, target_logprob_indexes):
    # Set length_norm_alpha to maket length_norm = 1.0
    length_norm_alpha = 0.0
    seq_len = 5
    p = models.BeamSearchHParams(
        beam_size=4,
        eos_id=4,
        parse_tokens=parse_tokens,
        fprop_for_prefix=True,
        max_decode_steps=3,
        seqlen=seq_len,
        length_norm_alpha=length_norm_alpha)
    logits = [[1, 0, 0, 2, 0], [5, 1, 0, 0, 0], [5, 0, 0, 1, 0],
              [0, 1, 0, 2, 10], [0, 0, 0, 0, 0]]
    logprobs = jax.nn.log_softmax(jnp.array(logits, dtype=jnp.float32))
    input_batch = NestedMap(
        ids=jnp.array([[2, 3]], dtype=jnp.int32),
        paddings=jnp.zeros(shape=(1, 2), dtype=jnp.float32),
        prefix_lengths=2 * jnp.ones(shape=(1), dtype=jnp.int32),
    )
    results = self._run_decode(p, logits, input_batch)

    # If no EOS in the sequence, add EOS to the last postion of the sequence.
    self.assertArraysEqual(results.output_ids,
                           np.array(target_output_ids, dtype=np.int32))

    self.assertArraysEqual(results.decode_lengths,
                           np.array(target_decode_ids, dtype=np.int32))

    # If no EOS in the sequence, add EOS to the last postion of the sequence.
    target_logprobs = []
    for indexes_list in target_logprob_indexes:
      logprobs_list = [1., 1.]
      for index in indexes_list:
        logprobs_list.append(logprobs[index[0]][index[1]])
      while len(logprobs_list) < seq_len:
        logprobs_list.append(1.)
      target_logprobs.append(logprobs_list)
    self.assertArraysEqual(results.logprobs,
                           np.array([target_logprobs], dtype=np.float32))


if __name__ == '__main__':
  absltest.main()
