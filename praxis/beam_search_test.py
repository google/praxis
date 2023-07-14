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

"""Unit tests for beam_search."""

import heapq
import math
import random
import time
from typing import Any
from praxis import pax_fiddle

from absl import logging
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

  def test_shuffle_state_match(self):
    cache_state = jnp.array(
        np.random.normal(size=(2, 4, 4, 8)), dtype=jnp.float32
    )
    hyp_ids = jnp.array(
        np.random.randint(low=0, high=4, size=(2, 4)), dtype=jnp.int32
    )
    transformed_state = beam_search.shuffle_state(
        cache_state, hyp_ids, use_one_hot_matmul=False
    )
    one_hot_transformed_state = beam_search.shuffle_state(
        cache_state, hyp_ids, use_one_hot_matmul=True
    )

    self.assertAllClose(transformed_state, one_hot_transformed_state)

  def test_broadcast_beam_dim(self):
    x = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
    self.assertArraysEqual(
        beam_search.broadcast_beam_dim(x, beam_dim=0, beam_size=2),
        np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]], dtype=np.int32))

  def test_update_global_beam(self):
    end_ids = jnp.array([[[0, 1, 0, 0], [4, 5, 0, 0]]], dtype=np.int32)
    cur_ids = jnp.array([[[0, 1, 2, 3], [4, 5, 6, 7]]], dtype=np.int32)

    end_lengths = jnp.array([[2, 2]], dtype=np.int32)
    cur_lengths = jnp.array([[4, 4]], dtype=np.int32)

    end_scores = jnp.array([[2, 0]], dtype=np.float32)
    cur_scores = jnp.array([[3, 1]], dtype=np.float32)

    end_logprobs = jnp.array(
        [[[-0.1, -1.1, 0., 0.], [-1.3, -2.3, 0., 0.]]], dtype=np.float32)
    cur_logprobs = jnp.array(
        [[[-0.1, -1.2, -1.1, 0.], [-1., -2., -0.3, 0.]]], dtype=np.float32)

    (output_ids, output_lengths, output_scores,
     output_logprobs) = beam_search.update_global_beam(
         (end_ids, end_lengths, end_scores, end_logprobs),
         (cur_ids, cur_lengths, cur_scores, cur_logprobs))

    self.assertArraysEqual(
        output_ids, np.array([[[0, 1, 2, 3], [0, 1, 0, 0]]], dtype=np.int32))
    self.assertArraysEqual(
        output_logprobs, np.array(
            [[[-0.1, -1.2, -1.1, 0.], [-0.1, -1.1, 0., 0.]]], dtype=np.float32))
    self.assertArraysEqual(output_lengths, np.array([[4, 2]], dtype=np.int32))
    self.assertArraysEqual(output_scores, np.array([[3, 2]], dtype=np.float32))


class MockLM(base_layer.BaseLayer):
  """Mock l m.

  Attributes:
    stepwise_logits: If True, the logits are keyed by step; otherwise, they are
      keyed by input token.
    logits: Premade logits returned by extend_step().  If `stepwise_logits` is
      True, then `logits[step]` is returned for every batch item.  Otherwise,
      the `i`th batch has logits `logits[inputs[i]]`.
  """
  logits: Any = None
  stepwise_logits: bool = False
  model_type: LanguageModelType = LanguageModelType.CAUSAL

  def setup(self) -> None:
    self._logits = jnp.array(self.logits, dtype=jnp.float32)

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
    if self.stepwise_logits:
      logits = jnp.take(self._logits, time_step, axis=0)
      logits = jnp.expand_dims(logits, axis=0)
      ret.logits = jnp.repeat(logits, inputs.shape[0], axis=0)
    else:
      ret.logits = jnp.take(self._logits, inputs, axis=0)
    self.put_variable(DECODE_CACHE, 'time_step', time_step + 1)
    return ret


class BeamSearchTest(test_utils.TestCase):

  def _run_decode(self, decoder_p, logits, input_batch, stepwise_logits=False):
    p = pax_fiddle.Config(
        models.LanguageModel,
        name='mock_lm',
        decoder_tpl=decoder_p.clone(),
        lm_tpl=pax_fiddle.Config(
            MockLM, logits=logits, stepwise_logits=stepwise_logits
        ),
    )
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
      (
          [],
          0,
          False,
          [[
              [2, 3, 4, 0, 0],
              [2, 3, 3, 4, 0],
              [2, 3, 0, 3, 4],
              [2, 3, 1, 0, 4],
          ]],
          [[3, 4, 5, 5]],
          [
              [[3, 4]],
              [[3, 3], [3, 4]],
              [[3, 0], [0, 3], [3, 4]],
              [[3, 1], [1, 0], [0, 4]],
          ],
          [[False, False, False, False]],
      ),
      (
          [],
          1,
          False,
          [[
              [2, 3, 3, 4, 0],
              [2, 3, 0, 3, 4],
              [2, 3, 1, 0, 4],
              [2, 3, 0, 4, 0],
          ]],
          [[4, 5, 5, 4]],
          [
              [[3, 3], [3, 4]],
              [[3, 0], [0, 3], [3, 4]],
              [[3, 1], [1, 0], [0, 4]],
              [[3, 0], [0, 4]],
          ],
          [[False, False, False, False]],
      ),
      (
          [3, 1],
          0,
          False,
          [[
              [2, 3, 4, 3, 0],
              [2, 3, 4, 0, 3],
              [2, 3, 4, 2, 3],
              [2, 3, 3, 0, 0],
          ]],
          [[4, 5, 5, 3]],
          [
              [[3, 4], [4, 3]],
              [[3, 4], [4, 0], [0, 3]],
              [[3, 4], [4, 2], [2, 3]],
              [[3, 3]],
          ],
          [[False, False, False, False]],
      ),
      (
          [],
          0,
          True,
          [[
              [2, 3, 4, 0, 0],
              [2, 3, 4, 4, 0],
              [2, 3, 3, 4, 0],
              [2, 3, 0, 4, 0],
          ]],
          [[3, 4, 4, 4]],
          [
              [[3, 4]],
              [[3, 4], [4, 4]],
              [[3, 3], [3, 4]],
              [[3, 0], [0, 4]],
          ],
          [[True, True, True, True]],
      ),
      (
          [],
          1,
          True,
          [[
              [2, 3, 3, 4, 0],
              [2, 3, 3, 4, 4],
              [2, 3, 0, 3, 4],
              [2, 3, 1, 0, 4],
          ]],
          [[4, 5, 5, 5]],
          [
              [[3, 3], [3, 4]],
              [[3, 3], [3, 4], [4, 4]],
              [[3, 0], [0, 3], [3, 4]],
              [[3, 1], [1, 0], [0, 4]],
          ],
          [[True, True, True, True]],
      ),
  )
  def test_vanilla_beam_search_base(
      self,
      parse_tokens,
      min_decode_steps,
      early_exit,
      target_output_ids,
      target_decode_ids,
      target_logprob_indexes,
      target_done,
  ):
    # Set length_norm_alpha to make length_norm = 1.0
    length_norm_alpha = 0.0
    seq_len = 5
    p = models.BeamSearchHParams(
        beam_size=4,
        eos_id=parse_tokens if parse_tokens else 4,
        fprop_for_prefix=True,
        min_decode_steps=min_decode_steps,
        max_decode_steps=3,
        seqlen=seq_len,
        length_norm_alpha=length_norm_alpha,
        early_exit=early_exit,
    )
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

    self.assertArraysEqual(results.done, target_done)

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

  def test_beam_search_vs_brute_force_on_random_logits(self):
    seed = time.time_ns()
    random.seed(seed)
    logging.info('PRNG seed: %d', seed)  # for later repro

    batch_size = 1
    beam_size = 32

    eos_ids = [1, 2]  # exercise multiple end tokens
    random.shuffle(eos_ids)
    num_control_tokens = 1 + len(eos_ids)  # <pad> </s1> </s2>
    num_normal_tokens = 3
    vocab_size = num_control_tokens + num_normal_tokens
    prefix_len = 2
    decoded_len = 5
    seq_len = prefix_len + decoded_len

    # We use the logs of unique primes as logits, so a score of a decoded path
    # is a product of unique primes and every path has a provably unique score.
    # Uniqueness allows the test to avoid tie-breaking and numerical near-ties,
    # which would make it hard to compare brute force and beam search.
    primes = [
        # Since <pad> is not scored, each row is 1 shorter than expected.
          2,   3,   5,   7,  11,  # 5
         13,  17,  19,  23,  29,  # 10
         31,  37,  41,  43,  47,  # 15
         53,  59,  61,  67,  71,  # 20
         73,  79,  83,  89,  97,  # 25
        101, 103, 107, 109, 113,  # 30
    ]  # pyformat: disable
    random.shuffle(primes)
    log_primes = [math.log(prime) for prime in primes]
    logits = [
        # Start each row with an infeasible score for <pad>, then slice rows off
        # of the shuffled list of primes.
        [-1e20] + log_primes[i * (vocab_size - 1) : (i + 1) * (vocab_size - 1)]
        for i in range(decoded_len)
    ]
    logging.info('logits:\n%s', logits)

    # Pad the logits up to `beam_size`; otherwise, the beam cannot be filled.
    for row in logits:
      row.extend([-1e20] * (beam_size - vocab_size))
    scores = jax.nn.log_softmax(jnp.array(logits, dtype=jnp.float32))
    logging.info('scores:\n%s', scores[:, :vocab_size])

    # Perform brute-force inference to find the true top-k.  Since the logits
    # are non-contextual (i.e., they do not depend on the previously-decoded
    # tokens), beam_search() will also find the true top-k and we can compare
    # results between the two.
    topk_heap = []  # min-heap of (score, [ids...])
    for length in range(decoded_len):  # `length` excludes </s>
      logging.info("enumerating all sequences of length %d", length)

      # Note that beam_search() does not split hypotheses for different choices
      # of EOS.  That is, while [1, 2, 3] and [1, 2, 4] are distinct hypotheses,
      # [1, 2, </s1>] and [1, 2, </s2>] are not---beam_search() only tracks the
      # one with the highest-scoring EOS.  Correspondingly, we pre-maximize the
      # EOS choices here.
      best_eos_id = eos_ids[0]
      for eos_id in eos_ids[1:]:
        if scores[length, eos_id] > scores[length, best_eos_id]:
          best_eos_id = eos_id

      # Enumerate all sequences of N tokens from vocabulary V by numbering them
      # and interpreting each index as an N-digit base-|V| number.
      for index in range(num_normal_tokens**length):
        sequence = [0] * prefix_len
        for _ in range(length):
          sequence.append(num_control_tokens + index % num_normal_tokens)
          index //= num_normal_tokens
        sequence.append(best_eos_id)
        score = sum([scores[i, t] for i, t in enumerate(sequence[prefix_len:])])

        heapq.heappush(topk_heap, (float(score), sequence))
        if len(topk_heap) > beam_size:
          heapq.heappop(topk_heap)

    brute_force_topk = list(reversed(sorted(topk_heap)))
    for score, ids in brute_force_topk:
      ids.pop(0)  # treat the first prefix token as padding

    # Set length_norm_alpha to make length_norm = 1.0
    length_norm_alpha = 0.0
    p = models.BeamSearchHParams(
        beam_size=beam_size,
        eos_id=eos_ids,
        fprop_for_prefix=True,
        max_decode_steps=decoded_len,
        seqlen=seq_len,
        length_norm_alpha=length_norm_alpha,
    )
    batch_size = 1
    input_batch = NestedMap(
        ids=jnp.zeros((batch_size, prefix_len), dtype=jnp.int32),
        # Treat the first prefix token as padding.
        paddings=jnp.array(
            [[1] + [0] * (prefix_len - 1)] * batch_size, dtype=jnp.int32
        ),
        prefix_lengths=jnp.full((batch_size,), prefix_len - 1, dtype=jnp.int32),
    )
    results = self._run_decode(p, logits, input_batch, stepwise_logits=True)

    beam_search_topk = []
    for k in range(beam_size):
      score = float(results.scores[0, k])
      length = results.decode_lengths[0, k]
      sequence = [int(t) for t in results.output_ids[0, k, :length]]
      beam_search_topk.append((score, sequence))

    for k, (
        (brute_force_score, brute_force_sequence),
        (beam_search_score, beam_search_sequence),
    ) in enumerate(zip(brute_force_topk, beam_search_topk)):
      logging.info(
          'For k=%d\nBrute force: %f %s\nBeam search: %f %s',
          k,
          brute_force_score,
          brute_force_sequence,
          beam_search_score,
          beam_search_sequence,
      )

      self.assertNear(
          brute_force_score, beam_search_score, 1e-3, f'for hypothesis {k}'
      )
      self.assertEqual(
          brute_force_sequence, beam_search_sequence, f'for hypothesis {k}'
      )


if __name__ == '__main__':
  absltest.main()
