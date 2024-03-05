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

"""Unit tests for model."""

from typing import Any

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import decoder_utils
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis import sample_decode
from praxis import test_utils
from praxis.layers import attentions
from praxis.layers import embedding_softmax
from praxis.layers import models
from praxis.layers import ngrammer
from praxis.layers import resnets
from praxis.layers import transformer_models
from praxis.layers import transformers
import tensorflow.compat.v2 as tf


NestedMap = py_utils.NestedMap
instantiate = base_layer.instantiate
LanguageModelType = transformer_models.LanguageModelType
JTensor = pytypes.JTensor

RANDOM = base_layer.RANDOM
DECODE_CACHE = base_layer.DECODE_CACHE
PREFIX_DECODE_CACHE = base_layer.PREFIX_DECODE_CACHE


class MockLM(base_layer.BaseLayer):
  """Mock l m.

  Attributes:
    logits: results returned by extend_step(), of shape [max step, batch size,
      vocab size].
    model_type:
  """
  logits: Any = None
  model_type: LanguageModelType = LanguageModelType.CAUSAL

  def setup(self) -> None:
    self._logits = jnp.array(self.logits, dtype=jnp.float32)

  def __call__(self, *args: Any, **kwargs: Any) -> None:
    self.put_variable(DECODE_CACHE, 'time_step', 0)

  def extend_step(
      self,
      inputs: Any,
      segment_pos: Any,
  ) -> Any:
    del inputs
    ret = NestedMap()
    time_step = self.get_variable(DECODE_CACHE, 'time_step')
    if segment_pos is not None:
      logits = jnp.take_along_axis(
          self._logits, segment_pos[jnp.newaxis, :, jnp.newaxis], axis=0
      )
      ret.logits = jnp.squeeze(logits, axis=0)
    else:
      ret.logits = self._logits.at[time_step].get()
    self.put_variable(DECODE_CACHE, 'time_step', time_step + 1)
    return ret

  def transform_decode_state(self, transform_fn):
    """Transforms all decode state variables based on transform_fn."""
    batch_dim = -1
    time_dim = -1
    new_state = transform_fn(
        self.get_variable(DECODE_CACHE, 'time_step'), batch_dim, time_dim)
    self.update_decode_state('time_step', new_state)


class LanguageModelTest(test_utils.TestCase):

  def _run_decode(
      self,
      decoder_p,
      logits,
      input_batch,
      model_type=LanguageModelType.CAUSAL,
      prng_seed=9,
  ):
    p = pax_fiddle.Config(
        models.LanguageModel,
        name='mock_lm',
        decoder_tpl=decoder_p.clone(),
        lm_tpl=pax_fiddle.Config(MockLM, logits=logits),
        model_type=model_type,
    )
    lang_model = instantiate(p)
    theta = NestedMap(lm=NestedMap())
    # We fix seed to 9 to get the desired prefix lengths below.
    prng_key = jax.random.PRNGKey(seed=prng_seed)
    results, _ = lang_model.apply(
        theta,
        input_batch,
        rngs={RANDOM: prng_key},
        method=lang_model.decode,
        mutable=[DECODE_CACHE])
    _, results, _ = results
    return results

  @parameterized.named_parameters(('_with_eval_sample_weights', True),
                                  ('_without_eval_sample_weights', False))
  def test_fprop(self, apply_eval_sample_weights):
    p = pax_fiddle.Config(
        models.LanguageModel,
        name='LM',
        lm_tpl=pax_fiddle.Config(
            transformer_models.TransformerLm, model_dims=3, vocab_size=5
        ),
        apply_eval_sample_weights=apply_eval_sample_weights,
    )
    stacked_transformer_tpl = p.lm_tpl.stacked_transformer_tpl
    stacked_transformer_tpl.model_dims = 3
    stacked_transformer_tpl.hidden_dims = 4 * 3
    stacked_transformer_tpl.dim_per_head = 3
    stacked_transformer_tpl.num_heads = 1
    stacked_transformer_tpl.num_layers = 1
    # We use full paddings to force prefix lengths to be 0 (since it is capped
    # at the lengths of input ids.
    input_batch = NestedMap(
        ids=jnp.array([[11, 12, 13, 14, 15]], dtype=jnp.int32),
        paddings=jnp.array([[0, 1, 1, 1, 1]], dtype=jnp.float32),
        weights=jnp.array([[1., 1., 1., 1., 1.]], dtype=jnp.float32),
        labels=jnp.array([[0, 1, 2, 3, 4]], dtype=jnp.int32),
        eval_sample_weights=jnp.array([1.], dtype=jnp.float32),
    )
    lang_model = instantiate(p)
    prng_key = jax.random.PRNGKey(1234)
    with base_layer.JaxContext.new_context():
      initial_vars = lang_model.init(prng_key, input_batch)
      metrics, per_example_out = lang_model.apply(
          initial_vars, input_batch, rngs={RANDOM: prng_key})
    self.assertIn('total_loss', metrics)
    self.assertIn('avg_xent', metrics)
    self.assertIn('fraction_of_correct_next_step_preds', metrics)
    self.assertIn('num_predictions', metrics)
    self.assertIn('labels', per_example_out)
    self.assertIn('scores', per_example_out)
    if apply_eval_sample_weights:
      self.assertIn('eval_sample_weights', per_example_out)

  def test_fprop_eval_sample_weights(self):
    p = pax_fiddle.Config(
        models.LanguageModel,
        name='LM',
        lm_tpl=pax_fiddle.Config(
            transformer_models.TransformerLm, model_dims=3, vocab_size=5
        ),
        apply_eval_sample_weights=True,
    )
    stacked_transformer_tpl = p.lm_tpl.stacked_transformer_tpl
    stacked_transformer_tpl.model_dims = 3
    stacked_transformer_tpl.hidden_dims = 4 * 3
    stacked_transformer_tpl.dim_per_head = 3
    stacked_transformer_tpl.num_heads = 1
    stacked_transformer_tpl.num_layers = 1
    lang_model = instantiate(p)
    prng_key = jax.random.PRNGKey(1234)

    batch_size = 3
    seq_len = 5

    input_batch = NestedMap(
        ids=np.array(
            [[1, 2, 3, 4, 5], [10, 20, 30, 40, 50], [100, 200, 300, 400, 500]],
            dtype=np.int32),
        paddings=np.zeros(shape=(batch_size, seq_len), dtype=np.float32),
        weights=np.ones(shape=(batch_size, seq_len), dtype=np.float32),
        labels=np.array([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [1, 1, 1, 1, 1]],
                        dtype=np.int32),
        eval_sample_weights=np.ones(shape=(batch_size,), dtype=np.float32),
    )
    with base_layer.JaxContext.new_context():
      initial_vars = lang_model.init(prng_key, input_batch)
      metrics_all, per_example_out_all = lang_model.apply(
          initial_vars, input_batch, rngs={RANDOM: prng_key})
    total_loss_all, total_loss_w_all = metrics_all.total_loss
    avg_xent_all, avg_xent_w_all = metrics_all.avg_xent
    accuracy_all, accuracy_w_all = metrics_all.fraction_of_correct_next_step_preds
    num_predictions_all, _ = metrics_all.num_predictions
    scores_all = per_example_out_all.scores

    input_batch_1 = input_batch.copy()
    input_batch_1.eval_sample_weights = np.array([1., 0., 0.], dtype=np.float32)
    with base_layer.JaxContext.new_context():
      metrics_1, per_example_out_1 = lang_model.apply(
          initial_vars, input_batch_1, rngs={RANDOM: prng_key})
    total_loss_1, total_loss_w_1 = metrics_1.total_loss
    avg_xent_1, avg_xent_w_1 = metrics_1.avg_xent
    accuracy_1, accuracy_w_1 = metrics_1.fraction_of_correct_next_step_preds
    num_predictions_1, _ = metrics_1.num_predictions
    scores_1 = per_example_out_1.scores

    input_batch_2 = input_batch.copy()
    input_batch_2.eval_sample_weights = np.array([0., 1., 1.], dtype=np.float32)
    with base_layer.JaxContext.new_context():
      metrics_2, per_example_out_2 = lang_model.apply(
          initial_vars, input_batch_2, rngs={RANDOM: prng_key})
    total_loss_2, total_loss_w_2 = metrics_2.total_loss
    avg_xent_2, avg_xent_w_2 = metrics_2.avg_xent
    accuracy_2, accuracy_w_2 = metrics_2.fraction_of_correct_next_step_preds
    num_predictions_2, _ = metrics_2.num_predictions
    scores_2 = per_example_out_2.scores

    # Ensure that we can recover the eval metrics for the large unpadded batch
    # from the two smaller padded batches.
    self.assertAllClose(total_loss_all,
                        1 / 3 * total_loss_1 + 2 / 3 * total_loss_2)
    self.assertAllClose(total_loss_w_all, total_loss_w_1 + total_loss_w_2)
    self.assertAllClose(avg_xent_all, 1 / 3 * avg_xent_1 + 2 / 3 * avg_xent_2)
    self.assertAllClose(avg_xent_w_all, avg_xent_w_1 + avg_xent_w_2)
    self.assertAllClose(accuracy_all, 1 / 3 * accuracy_1 + 2 / 3 * accuracy_2)
    self.assertAllClose(accuracy_w_all, accuracy_w_1 + accuracy_w_2)
    self.assertAllClose(num_predictions_all,
                        num_predictions_1 + num_predictions_2)
    # The following per-example metric is purely additive, since padded examples
    # provide 0. values.
    self.assertAllClose(scores_all, scores_1 + scores_2)

  @parameterized.parameters([True, False])
  def test_base_case(self, fprop_for_prefix):
    p = pax_fiddle.Config(models.LanguageModel).decoder_tpl
    p.seqlen = 3
    p.min_prefix_len = 1
    p.fprop_for_prefix = fprop_for_prefix
    if fprop_for_prefix:
      p.max_decode_steps = 2
    logits = [
        [
            [0, 1, 0, 0],
        ],
        [
            [0, 0, 0, 1],
        ],
    ]
    # We use full paddings to force prefix lengths to be 0 (since it is capped
    # at the lengths of input ids.
    input_batch = NestedMap(
        ids=jnp.array([[11, 12, 13, 14, 15]], dtype=jnp.int32),
        paddings=jnp.array([[0, 1, 1, 1, 1]], dtype=jnp.float32),
    )
    results = self._run_decode(p, logits, input_batch)
    self.assertArraysEqual(results.prefix_lengths,
                           np.array([[1]], dtype=np.int32))
    # Decoding starts at 1 from input.ids, then each step uses argmax from the
    # provided logits, which are 1 and 3.
    if fprop_for_prefix:
      self.assertArraysEqual(
          results.output_ids,
          np.array([[[11, 1, 3, 0, 0, 0, 0]]], dtype=np.int32))
    else:
      self.assertArraysEqual(results.output_ids,
                             np.array([[[11, 1, 3]]], dtype=np.int32))
    self.assertArraysEqual(results.decode_lengths,
                           np.array([[3]], dtype=np.int32))

  def test_flat_beam_search_no_prefix(self):
    length_norm_alpha = 0.8
    p = models.FlatBeamSearchHParams(
        beam_size=4, eos_id=4, seqlen=4, length_norm_alpha=length_norm_alpha)
    logits = [
        [
            [2, 3, 1, 5, 0],  #  Top4: 3, 1, 0, 2
            [2, 3, 1, 5, 0],
            [2, 3, 1, 5, 0],
            [2, 3, 1, 5, 0],
        ],
        [
            [2, 3, 4, 1, 55],
            [25, 4, 3, 2, 1],
            [0, 0, 0, 0, 0],
            [1, 2, 26, 4, 5],
        ],
        # The last step doesn't matter as seqlen = 4 and flat beam search will
        # add EOS to the last step.
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
    ]
    input_batch = NestedMap(
        ids=jnp.array([[1, 0, 0, 0, 9]], dtype=jnp.int32),
        paddings=jnp.ones(shape=(1, 5), dtype=jnp.float32),
    )
    results = self._run_decode(p, logits, input_batch)
    # Decoding starts from the last element from input.ids, then each step uses
    # beam search from the provided logits.
    self.assertArraysEqual(
        results.output_ids,
        np.array([[[9, 3, 4, 0], [9, 1, 0, 4], [9, 2, 2, 4], [9, 3, 2, 4]]],
                 dtype=np.int32))
    self.assertArraysEqual(results.decode_lengths,
                           np.array([[3, 4, 4, 4]], dtype=np.int32))

    def length_norm(length):
      return decoder_utils.length_norm(np.array(length - 1), length_norm_alpha)

    # Get scores from output_ids sequence [3, 4], [1, 0], [2, 2] and [3, 2].
    self.assertArraysEqual(
        results.scores,
        np.array([[(5 + 55), (3 + 25), (1 + 26), (5 + 4)]], dtype=np.float32) /
        length_norm(results.decode_lengths))

  @parameterized.parameters([True, False])
  def test_prefix(self, fprop_for_prefix):
    p = pax_fiddle.Config(models.LanguageModel).decoder_tpl
    p.seqlen = 5
    p.min_prefix_len = 2
    p.fprop_for_prefix = fprop_for_prefix
    if fprop_for_prefix:
      p.max_decode_steps = 3
    logits = [
        [
            [0, 0, 0, 0, 0, 1],  # argmax=5, prefix
        ],
        [
            [0, 1, 0, 0, 0, 0],  # argmax=1
        ],
        [
            [0, 0, 0, 1, 0, 0],  # argmax=3
        ],
        [
            [0, 0, 0, 0, 1, 0],  # argmax=4
        ],
        [
            [0, 0, 0, 0, 0, 1],  # argmax=5
        ],
    ]
    input_batch = NestedMap(
        ids=jnp.array([[11, 5, 13, 14, 15]], dtype=jnp.int32),
        paddings=jnp.array([[0.0, 0.0, 1.0, 1.0, 1.0]], dtype=jnp.float32),
    )
    results = self._run_decode(p, logits, input_batch)
    self.assertArraysEqual(
        results.prefix_lengths, np.array([[2]], dtype=np.int32)
    )

    expected_output_ids = np.array([[[11, 5, 1, 3, 4]]], dtype=np.int32)
    expected_prefix_ids = np.array([[[11, 5, 0, 0, 0]]], dtype=np.int32)

    if fprop_for_prefix:
      total_len = p.seqlen + p.max_decode_steps
      expected_output_ids = py_utils.pad_or_trim_to(
          expected_output_ids, [1, 1, total_len], pad_val=0
      )
      expected_prefix_ids = py_utils.pad_or_trim_to(
          expected_prefix_ids, [1, 1, total_len], pad_val=0
      )

    self.assertArraysEqual(results.output_ids, expected_output_ids)
    self.assertArraysEqual(results.prefix_ids, expected_prefix_ids)
    self.assertArraysEqual(
        results.decode_lengths, np.array([[5]], dtype=np.int32)
    )

  @parameterized.parameters([True, False])
  def test_prefix_lm(self, fprop_for_prefix):
    p = pax_fiddle.Config(models.LanguageModel).decoder_tpl
    p.seqlen = 5
    p.min_prefix_len = 2
    p.fprop_for_prefix = fprop_for_prefix
    if fprop_for_prefix:
      p.max_decode_steps = 3
    logits = [
        [
            [0, 0, 0, 0, 0, 1],  # argmax=5, prefix
        ],
        [
            [0, 1, 0, 0, 0, 0],  # argmax=1
        ],
        [
            [0, 0, 0, 1, 0, 0],  # argmax=3
        ],
        [
            [0, 0, 0, 0, 1, 0],  # argmax=4
        ],
        [
            [0, 0, 0, 0, 0, 1],  # argmax=5
        ],
    ]
    input_batch = NestedMap(
        ids=jnp.array([[11, 5, 13, 14, 15]], dtype=jnp.int32),
        paddings=jnp.array([[0.0, 0.0, 1.0, 1.0, 1.0]], dtype=jnp.float32),
        inputs_indicator=jnp.array([[1, 1, 0, 0, 0]], dtype=jnp.float32),
    )
    results = self._run_decode(
        p, logits, input_batch, model_type=LanguageModelType.PREFIX)
    self.assertArraysEqual(results.prefix_lengths,
                           np.array([[2]], dtype=np.int32))

    expected_output_ids = np.array([[[11, 5, 1, 3, 4]]], dtype=np.int32)
    expected_prefix_ids = np.array([[[11, 5, 0, 0, 0]]], dtype=np.int32)

    if fprop_for_prefix:
      total_len = p.seqlen + p.max_decode_steps
      expected_output_ids = py_utils.pad_or_trim_to(
          expected_output_ids, [1, 1, total_len], pad_val=0
      )
      expected_prefix_ids = py_utils.pad_or_trim_to(
          expected_prefix_ids, [1, 1, total_len], pad_val=0
      )

    self.assertArraysEqual(results.output_ids, expected_output_ids)
    self.assertArraysEqual(results.prefix_ids, expected_prefix_ids)
    self.assertArraysEqual(
        results.decode_lengths, np.array([[5]], dtype=np.int32)
    )

  @parameterized.parameters([(True, True), (True, False), (False, False)])
  def test_sample_decoding_prefix(
      self, fprop_for_prefix, vanilla_sample_decode
  ):
    p = models.SampleDecoderHParams(
        seqlen=5,
        min_prefix_len=2,
        eos_id=2,
        k=1,  # greedy_decoding
        fprop_for_prefix=fprop_for_prefix,
        vanilla_sample_decode=vanilla_sample_decode,
    )
    p.max_decode_steps = 3 if fprop_for_prefix else p.seqlen
    logits = [
        [
            [0, 0, 0, 0, 0, 1],  # argmax=5, prefix
        ],
        [
            [0, 1, 0, 0, 0, 0],  # argmax=1
        ],
        [
            [0, 0, 0, 1, 0, 0],  # argmax=3
        ],
        [
            [0, 0, 0, 0, 1, 0],  # argmax=4
        ],
        [
            [0, 0, 0, 0, 0, 1],  # argmax=5
        ],
    ]
    # The test doesn't check (False, True) case, because vanilla_sample_decode
    # doesn't forgive the case where input_batch has prefix but fprop_for_prefix
    # is False.
    input_batch = NestedMap(
        ids=jnp.array([[11, 5, 13, 14, 15]], dtype=jnp.int32),
        paddings=jnp.array([[0.0, 0.0, 1.0, 1.0, 1.0]], dtype=jnp.float32),
    )
    results = self._run_decode(p, logits, input_batch)

    def to_expected(x):
      if vanilla_sample_decode:
        # Extend the sample axis, as vanilla doesn't have sample concept.
        return x[:, jnp.newaxis]
      return x

    results = results.Transform(to_expected)
    self.assertArraysEqual(
        results.prefix_lengths, np.array([[2]], dtype=np.int32)
    )

    expected_output_ids = np.array([[[11, 5, 1, 3, 4]]], dtype=np.int32)
    expected_prefix_ids = np.array([[[11, 5, 0, 0, 0]]], dtype=np.int32)

    if fprop_for_prefix:
      total_len = p.seqlen + p.max_decode_steps
      expected_output_ids = py_utils.pad_or_trim_to(
          expected_output_ids, [1, 1, total_len], pad_val=0
      )
      if not vanilla_sample_decode:
        expected_prefix_ids = py_utils.pad_or_trim_to(
            expected_prefix_ids, [1, 1, total_len], pad_val=0
        )

    self.assertArraysEqual(results.output_ids, expected_output_ids)
    self.assertArraysEqual(results.prefix_ids, expected_prefix_ids)
    self.assertArraysEqual(
        results.decode_lengths, np.array([[5]], dtype=np.int32)
    )

  def test_eos_terminate(self):
    p = pax_fiddle.Config(models.LanguageModel).decoder_tpl
    p.seqlen = 6
    p.min_prefix_len = 0
    p.eos_id = 2
    logits = [
        [
            [0, 0, 0, 0, 1],  # argmax=4
        ],
        [
            [0, 0, 1, 0, 0],  # argmax=2
        ],
        [
            [0, 0, 0, 1, 0],  # argmax=3
        ],
    ]
    input_batch = NestedMap(
        ids=jnp.array([[11, 13]], dtype=jnp.int32),
        paddings=jnp.ones(shape=(1, 2), dtype=jnp.float32),
    )
    results = self._run_decode(p, logits, input_batch)
    self.assertArraysEqual(results.prefix_lengths,
                           np.array([[0]], dtype=np.int32))
    # Decoding terminates after step 2 when eos_id=2 is encountered.
    self.assertArraysEqual(results.output_ids,
                           np.array([[[11, 4, 2, 0, 0, 0]]], dtype=np.int32))
    self.assertArraysEqual(results.decode_lengths,
                           np.array([[3]], dtype=np.int32))

  def test_eos_independent(self):
    p = pax_fiddle.Config(models.LanguageModel).decoder_tpl
    p.seqlen = 5
    p.min_prefix_len = 0
    p.eos_id = 2
    logits = [
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],  # argmax=[4, 3]
        ],
        [
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1],  # argmax=[2, 4]
        ],
        [
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],  # argmax=[3, 2]
        ],
    ]
    input_batch = NestedMap(
        ids=jnp.array([[11, 13], [12, 14]], dtype=jnp.int32),
        paddings=jnp.ones(shape=(2, 2), dtype=jnp.float32),
    )
    results = self._run_decode(p, logits, input_batch)
    self.assertArraysEqual(results.prefix_lengths,
                           np.array([[0], [0]], dtype=np.int32))
    # EOS termination are row independent: row 0 terminates at step 2 while
    # row 1 terminates at step 3.
    self.assertArraysEqual(
        results.output_ids,
        np.array([[[11, 4, 2, 0, 0]], [[12, 3, 4, 2, 0]]], dtype=np.int32))
    self.assertArraysEqual(results.decode_lengths,
                           np.array([[3], [4]], dtype=np.int32))

  def test_prefix_and_eos(self):
    p = pax_fiddle.Config(models.LanguageModel).decoder_tpl
    p.seqlen = 5
    p.min_prefix_len = 0
    p.eos_id = 2
    logits = [
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],  # argmax=[4, 3, 3]
        ],
        [
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0],  # argmax=[3, 4, 2]
        ],
        [
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],  # argmax=[3, 2, 3]
        ],
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],  # argmax=[4, 4, 3]
        ],
    ]
    input_batch = NestedMap(
        ids=jnp.array([[11, 13, 15], [12, 14, 16], [20, 30, 40]],
                      dtype=jnp.int32),
        paddings=jnp.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]],
                           dtype=jnp.float32),
    )
    results = self._run_decode(p, logits, input_batch)
    # This is fixed by the paddings provided.
    self.assertArraysEqual(results.prefix_lengths,
                           np.array([[2], [1], [0]], dtype=np.int32))
    # Row 0 copies 2 ids from the input as prefix, and continues without
    # ever hitting EOS. Row 1 and 2 only copies the first id from the input,
    # and continues until EOS is found.
    self.assertArraysEqual(
        results.output_ids,
        np.array([[[11, 13, 3, 3, 4]], [[12, 3, 4, 2, 0]], [[20, 3, 2, 0, 0]]],
                 dtype=np.int32))
    self.assertArraysEqual(results.decode_lengths,
                           np.array([[5], [4], [3]], dtype=np.int32))

  def test_prefix_and_eos_fprop_for_prefix(self):
    p = pax_fiddle.Config(models.LanguageModel).decoder_tpl
    p.seqlen = 7
    p.max_decode_steps = 4
    p.min_prefix_len = 0
    p.eos_id = 2
    p.fprop_for_prefix = True
    logits = [
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],  # argmax=[4, 3, 3]
        ],
        [
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0],  # argmax=[3, 4, 2]
        ],
        [
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],  # argmax=[3, 2, 3]
        ],
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],  # argmax=[4, 4, 3]
        ],
    ]
    input_batch = NestedMap(
        ids=jnp.array(
            [[11, 4, 15], [12, 14, 16], [20, 30, 40]], dtype=jnp.int32
        ),
        paddings=jnp.array(
            [[0, 0, 1], [0, 1, 1], [0, 1, 1]], dtype=jnp.float32
        ),
        prefix_lengths=jnp.array([2, 1, 1], dtype=jnp.int32),
    )
    results = self._run_decode(p, logits, input_batch)
    # This is fixed by the paddings provided.
    self.assertArraysEqual(results.prefix_lengths,
                           np.array([2, 1, 1], dtype=np.int32))
    # Row 0 copies 2 ids from the input as prefix, and continues without
    # ever hitting EOS. Row 1 and 2 only copies the first id from the input,
    # and continues until EOS is found.
    # The prefix is right aligned to the generated sequence.
    self.assertArraysEqual(
        results.output_ids,
        np.array(
            [
                [[11, 4, 3, 3, 4, 0, 0]],
                [[12, 3, 4, 2, 0, 0, 0]],
                [[20, 3, 2, 0, 0, 0, 0]],
            ],
            dtype=np.int32,
        ),
    )
    self.assertArraysEqual(results.decode_lengths,
                           np.array([[6], [4], [3]], dtype=np.int32))

  def test_prefix_has_eos(self):
    p = pax_fiddle.Config(models.LanguageModel).decoder_tpl
    p.seqlen = 4
    p.min_prefix_len = 0
    p.eos_id = 2
    logits = [
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],  # argmax=3
        ],
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],  # argmax=4
        ],
        [
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],  # argmax=[3, 2]
        ],
    ]
    input_batch = NestedMap(
        ids=jnp.array([[2, 2, 2], [2, 2, 2]], dtype=jnp.int32),
        paddings=jnp.array([[0, 0, 0], [0, 1, 1]], dtype=jnp.float32),
    )
    results = self._run_decode(p, logits, input_batch)
    # This is fixed by the paddings provided.
    self.assertArraysEqual(results.prefix_lengths,
                           np.array([[3], [1]], dtype=np.int32))
    # Row 0 copies the first 3 ids, and does not terminate even though these
    # ids are EOS. Row 1 copies the first EOS from ids, and uses argmax for the
    # remaining 3.
    self.assertArraysEqual(
        results.output_ids,
        np.array([[[2, 2, 2, 3]], [[2, 3, 4, 2]]], dtype=np.int32))
    self.assertArraysEqual(results.decode_lengths,
                           np.array([[4], [4]], dtype=np.int32))

  def test_max_decode_steps(self):
    p = pax_fiddle.Config(models.LanguageModel).decoder_tpl
    p.seqlen = 5
    p.min_prefix_len = 0
    p.eos_id = 2
    p.max_decode_steps = 2
    logits = [
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],  # argmax=[4, 3, 3]
        ],
        [
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],  # argmax=[2, 4, 4]
        ],
        [
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],  # argmax=[3, 4, 3]
        ],
    ]
    input_batch = NestedMap(
        ids=jnp.array([[11, 13, 15], [12, 14, 16], [20, 30, 40]],
                      dtype=jnp.int32),
        paddings=jnp.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]],
                           dtype=jnp.float32),
    )
    results = self._run_decode(p, logits, input_batch)
    # This is fixed by the paddings provided.
    self.assertArraysEqual(results.prefix_lengths,
                           np.array([[2], [1], [0]], dtype=np.int32))
    # Row 0 has prefix length 2, and hit EOS after decode for one step, so it
    # stops. Row 1 has prefix length 1, and hit max decode steps of 2, so it
    # stops at 3 decoded ids. Row 2 has prefix length 0, and stops after
    # hitting the max decode step of 2, ending with 2 decoded ids.
    # Note that logically prefix length 1 and 0 are equivalent, because
    # decoding always starts with the fixed first ids (BOS in practice), the
    # only difference is how they affect the counting of max_decode_steps.
    self.assertArraysEqual(
        results.output_ids,
        np.array([[[11, 13, 2, 0, 0]], [[12, 3, 4, 0, 0]], [[20, 3, 0, 0, 0]]],
                 dtype=np.int32))
    self.assertArraysEqual(results.decode_lengths,
                           np.array([[3], [3], [2]], dtype=np.int32))
    # softmax on logits of [0, 0, 0, 0, 1] reproduces:
    # [-1.904833   -1.904833   -1.904833   -1.904833   -0.90483296]
    self.assertAllClose(
        results.logprobs,
        np.array([[[1., -0.904832, -0.904832, 1., 1.]],
                  [[1., -0.904832, -0.904832, 1., 1.]],
                  [[1., -0.904832, 1., 1., 1.]]],
                 dtype=np.float32))

  @parameterized.parameters(
      (1),
      (2),
  )
  def test_sample_decoding_prefix_and_eos(self, k):
    p = models.SampleDecoderHParams(
        seqlen=5,
        min_prefix_len=0,
        eos_id=2,
        num_samples=2,
        k=k,
        temperature=0.5)
    logits = [
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],  # argmax=[4, 3, 3]
        ],
        [
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0],  # argmax=[3, 4, 2]
        ],
        [
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],  # argmax=[3, 2, 3]
        ],
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],  # argmax=[4, 4, 3]
        ],
    ]
    sample_logits = jnp.repeat(jnp.array(logits), axis=1, repeats=2)
    input_batch = NestedMap(
        ids=jnp.array([[11, 13, 15], [12, 14, 16], [20, 30, 40]],
                      dtype=jnp.int32),
        paddings=jnp.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]],
                           dtype=jnp.float32),
    )
    results = self._run_decode(p, sample_logits, input_batch)

    # This is fixed by the paddings provided.
    self.assertArraysEqual(results.prefix_lengths,
                           np.array([[2, 2], [1, 1], [0, 0]], dtype=np.int32))
    # Row 0 copies 2 ids from the input as prefix, and continues without
    # ever hitting EOS. Row 1 and 2 only copies the first id from the input,
    # and continues until EOS is found.
    if k == 1:
      self.assertArraysEqual(
          results.output_ids,
          np.array([[[11, 13, 3, 3, 4], [11, 13, 3, 3, 4]],
                    [[12, 3, 4, 2, 0], [12, 3, 4, 2, 0]],
                    [[20, 3, 2, 0, 0], [20, 3, 2, 0, 0]]],
                   dtype=np.int32))
      self.assertArraysEqual(
          results.decode_lengths,
          np.array([[5, 5], [4, 4], [3, 3]], dtype=np.int32),
      )
    else:
      # Gumbel noise will make some difference between samples.
      self.assertArraysEqual(
          results.output_ids,
          np.array([[[11, 13, 3, 3, 4], [11, 13, 0, 3, 4]],
                    [[12, 3, 4, 2, 0], [12, 3, 4, 0, 0]],
                    [[20, 3, 2, 0, 0], [20, 3, 2, 0, 0]]],
                   dtype=np.int32))
      self.assertArraysEqual(
          results.decode_lengths,
          np.array([[5, 5], [4, 5], [3, 3]], dtype=np.int32),
      )

  def test_sample_decoding_with_gumbel_prng(self):
    p = models.SampleDecoderHParams(
        seqlen=5,
        min_prefix_len=0,
        eos_id=1,
        num_samples=2,
        k=3,
        temperature=20.0,
    )

    prng_seed_a = 123
    prng_seed_b = 456
    logits = np.ones(shape=(4, 3, 5), dtype=np.float32) * 0.1
    sample_logits = jnp.repeat(jnp.array(logits), axis=1, repeats=2)

    # input batch with dummy gumbel_prng_key, gumbel_prng_key will be ignored.
    input_batch_c = NestedMap(
        ids=jnp.array(
            [[11, 13, 15], [12, 14, 16], [20, 30, 40]], dtype=jnp.int32
        ),
        paddings=jnp.zeros(shape=(3, 3), dtype=jnp.float32),
        gumbel_prng_key=jnp.array(
            [decoder_utils.DUMMY_PRNG_KEY] * 3, dtype=jnp.uint32
        ),
        prefix_lengths=jnp.array([3, 3, 3], dtype=jnp.int32),
    )
    results_ac = self._run_decode(
        p, sample_logits, input_batch_c, prng_seed=prng_seed_a
    )
    results_bc = self._run_decode(
        p, sample_logits, input_batch_c, prng_seed=prng_seed_b
    )

    # results will be different with different prng_seed
    self.assertNotAllClose(results_ac.output_ids, results_bc.output_ids)

    # input batch with deterministic gumbel_prng_key, prng_seed will be ignored.
    input_batch_d = NestedMap(
        ids=jnp.array(
            [[11, 13, 15], [12, 14, 16], [20, 30, 40]], dtype=jnp.int32
        ),
        paddings=jnp.zeros(shape=(3, 3), dtype=jnp.float32),
        gumbel_prng_key=jnp.array([123, 23, 56], dtype=jnp.uint32),
        prefix_lengths=jnp.array([3, 3, 3], dtype=jnp.int32),
    )
    results_ad = self._run_decode(
        p, sample_logits, input_batch_d, prng_seed=prng_seed_a
    )
    results_bd = self._run_decode(
        p, sample_logits, input_batch_d, prng_seed=prng_seed_b
    )
    self.assertArraysEqual(results_ad.output_ids, results_bd.output_ids)

  @parameterized.parameters(
      (1, False),
      (2, False),
      (1, True),
      (2, True),
  )
  def test_sample_decoding_prefix_and_eos_fprop_for_prefix(
      self, k, is_dynamic_input):
    p = models.SampleDecoderHParams(
        fprop_for_prefix=True,
        seqlen=7,
        max_decode_steps=4,
        min_prefix_len=0,
        eos_id=2,
        num_samples=2,
        k=k,
        p=0.9,
        temperature=0.5)
    logits = [
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],  # argmax=[4, 3, 3]
        ],
        [
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0],  # argmax=[3, 4, 2]
        ],
        [
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],  # argmax=[3, 2, 3]
        ],
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],  # argmax=[4, 4, 3]
        ],
    ]
    sample_logits = jnp.repeat(jnp.array(logits), axis=1, repeats=2)
    input_batch = NestedMap(
        ids=jnp.array([[11, 13, 15], [12, 14, 16], [20, 30, 40]],
                      dtype=jnp.int32),
        paddings=jnp.array([[0, 0, 1], [0, 1, 1], [0, 1, 1]],
                           dtype=jnp.float32),
        prefix_lengths=jnp.array([2, 1, 1], dtype=jnp.int32),
    )

    if is_dynamic_input:
      # Test if JTensor type temperature could work.
      input_batch['temperature'] = jnp.array([0.5, 0.5, 0.5], dtype=jnp.float32)
      input_batch['stop_decode_steps'] = jnp.array([4, 4, 3], dtype=jnp.int32)
      input_batch['per_example_top_p'] = jnp.array(
          [0.9, 0.9, 0.9], dtype=jnp.float32
      )
      input_batch['per_example_top_k'] = jnp.array([2, 2, 2], dtype=jnp.int32)
      input_batch['eos_id'] = jnp.array(
          [[0, 2], [0, 2], [0, 2]], dtype=jnp.int32
      )

    results = self._run_decode(p, sample_logits, input_batch)

    # This is fixed by the paddings provided.
    self.assertArraysEqual(results.prefix_lengths,
                           np.array([2, 1, 1], dtype=np.int32))
    # Row 0 copies 2 ids from the input as prefix, and continues without
    # ever hitting EOS. Row 1 and 2 only copies the first id from the input,
    # and continues until EOS is found.
    if k == 1:
      self.assertArraysEqual(
          results.output_ids,
          np.array(
              [
                  [[11, 13, 3, 3, 4, 0, 0], [11, 13, 3, 3, 4, 0, 0]],
                  [[12, 3, 4, 2, 0, 0, 0], [12, 3, 4, 2, 0, 0, 0]],
                  [[20, 3, 2, 0, 0, 0, 0], [20, 3, 2, 0, 0, 0, 0]],
              ],
              dtype=np.int32,
          ),
      )
      self.assertArraysEqual(
          results.decode_lengths,
          np.array([[6, 6], [4, 4], [3, 3]], dtype=np.int32),
      )
    else:
      # Gumbel noise will make some difference between samples.
      self.assertArraysEqual(
          results.output_ids,
          np.array(
              [
                  [[11, 13, 3, 3, 4, 0, 0], [11, 13, 3, 0, 4, 0, 0]],
                  [[12, 3, 4, 2, 0, 0, 0], [12, 3, 4, 0, 0, 0, 0]],
                  [[20, 3, 2, 0, 0, 0, 0], [20, 3, 2, 0, 0, 0, 0]],
              ],
              dtype=np.int32,
          ),
      )
      self.assertArraysEqual(
          results.decode_lengths,
          np.array([[6, 6], [4, 5], [3, 3]], dtype=np.int32),
      )

  def test_sample_decoding_prefix_and_eos_sample_equal_one(self):
    p = models.SampleDecoderHParams(
        seqlen=5,
        min_prefix_len=0,
        eos_id=2,
        num_samples=1,
        k=2,
        temperature=0.5)
    logits = [
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],  # argmax=[4, 3, 3]
        ],
        [
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0],  # argmax=[3, 4, 2]
        ],
        [
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],  # argmax=[3, 2, 3]
        ],
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],  # argmax=[4, 4, 3]
        ],
    ]
    input_batch = NestedMap(
        ids=jnp.array([[11, 13, 15], [12, 14, 16], [20, 30, 40]],
                      dtype=jnp.int32),
        paddings=jnp.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]],
                           dtype=jnp.float32),
    )
    results = self._run_decode(p, logits, input_batch)

    # This is fixed by the paddings provided.
    self.assertArraysEqual(results.prefix_lengths,
                           np.array([[2], [1], [0]], dtype=np.int32))
    # Row 0 copies 2 ids from the input as prefix, and continues without
    # ever hitting EOS. Row 1 and 2 only copies the first id from the input,
    # and continues until EOS is found.
    self.assertArraysEqual(
        results.output_ids,
        np.array([[[11, 13, 0, 3, 0]], [[12, 3, 4, 2, 0]], [[20, 3, 2, 0, 0]]],
                 dtype=np.int32))
    self.assertArraysEqual(results.decode_lengths,
                           np.array([[5], [4], [3]], dtype=np.int32))

  def test_sample_decoding_multi_stop_tokens(self):
    p = models.SampleDecoderHParams(
        seqlen=7,
        min_prefix_len=0,
        eos_id=[1, 2],
        num_samples=1,
        k=2,
        temperature=0.5,
        fprop_for_prefix=True,
        max_decode_steps=4,
    )
    logits = [
        [
            [0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],  # argmax=[4, 1, 3]
        ],
        [
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],  # argmax=[1, 4, 3]
        ],
        [
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],  # argmax=[3, 2, 3]
        ],
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],  # argmax=[4, 4, 3]
        ],
    ]
    input_batch = NestedMap(
        ids=jnp.array(
            [[11, 13, 15], [12, 14, 16], [20, 30, 40]], dtype=jnp.int32
        ),
        paddings=jnp.zeros(shape=(3, 3), dtype=jnp.float32),
        prefix_lengths=jnp.array([2, 2, 1], dtype=jnp.int32),
    )
    results = self._run_decode(p, logits, input_batch)

    self.assertArraysEqual(
        results.output_ids,
        np.array(
            [
                [[11, 13, 1, 0, 0, 0, 0]],
                [[12, 14, 4, 2, 0, 0, 0]],
                [[20, 3, 3, 3, 0, 0, 0]],
            ],
            dtype=np.int32,
        ),
    )
    self.assertArraysEqual(
        results.decode_lengths, np.array([[3], [4], [5]], dtype=np.int32)
    )

  def test_sample_decoding_with_entropy_score(self):
    p = models.SampleDecoderHParams(
        seqlen=7,
        min_prefix_len=0,
        eos_id=[1, 2],
        num_samples=1,
        k=2,
        temperature=0.5,
        fprop_for_prefix=True,
        max_decode_steps=4,
    )
    logits = [
        [
            [0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],  # argmax=[4, 1, 3]
        ],
        [
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0],  # argmax=[1, 4, 2]
        ],
        [
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],  # argmax=[3, 2, 3]
        ],
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],  # argmax=[4, 4, 3]
        ],
    ]
    input_batch = NestedMap(
        ids=jnp.array(
            [[11, 13, 15], [12, 14, 16], [20, 30, 40]], dtype=jnp.int32
        ),
        paddings=jnp.zeros(shape=(3, 3), dtype=jnp.float32),
        prefix_lengths=jnp.array([2, 2, 1], dtype=jnp.int32),
        return_entropy_score=jnp.array([1.0], dtype=jnp.float32),
    )
    results = self._run_decode(p, logits, input_batch)
    self.assertLen(results.entropy, 3)

  def test_sample_decoding_with_num_per_token_logprobs(self):
    p = models.SampleDecoderHParams(
        seqlen=7,
        min_prefix_len=0,
        eos_id=[1, 2],
        num_samples=1,
        k=2,
        temperature=0.5,
        fprop_for_prefix=True,
        max_decode_steps=4,
    )
    logits = [
        [
            [0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],  # argmax=[4, 1, 3]
        ],
        [
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0],  # argmax=[1, 4, 2]
        ],
        [
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],  # argmax=[3, 2, 3]
        ],
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],  # argmax=[4, 4, 3]
        ],
    ]
    num_per_token_logprobs = 3
    input_batch = NestedMap(
        ids=jnp.array(
            [[11, 13, 15], [12, 14, 16], [20, 30, 40]], dtype=jnp.int32
        ),
        paddings=jnp.zeros(shape=(3, 3), dtype=jnp.float32),
        prefix_lengths=jnp.array([2, 2, 1], dtype=jnp.int32),
        num_per_token_logprobs=jnp.array(
            [num_per_token_logprobs], dtype=jnp.int32
        ),
    )
    results = self._run_decode(p, logits, input_batch)
    top_candidate_ids = results.top_candidate_ids
    top_candidate_logprobs = results.top_candidate_logprobs
    # Check shape.
    shape = (
        3,  # batch_size
        1,  # num_samples
        7,  # seq_len
        sample_decode.MAX_NUM_PER_TOKEN_LOGPROBS,
    )
    self.assertEqual(shape, top_candidate_ids.shape)
    self.assertEqual(shape, top_candidate_logprobs.shape)
    # Check that values outside of the top `num_per_token_logprobs` are 0.
    self.assertArraysEqual(
        top_candidate_ids[:, :, :, num_per_token_logprobs:], 0
    )
    self.assertArraysEqual(
        top_candidate_logprobs[:, :, :, num_per_token_logprobs:], 1.0
    )
    # Check that logprobs are sorted in descending order.
    logprobs = top_candidate_logprobs[:, :, :, :num_per_token_logprobs]
    self.assertArraysEqual(
        jnp.flip(jnp.sort(logprobs), -1),
        logprobs,
    )

  def test_cf_guidance_unimplemented_exception(self):
    p = models.SampleDecoderHParams(seqlen=5, cf_guidance_scale=2.0)
    input_batch = NestedMap(
        ids=jnp.array([[11, 13]], dtype=jnp.int32),
        paddings=jnp.ones(shape=(1, 2), dtype=jnp.float32),
    )

    with self.assertRaisesRegex(NotImplementedError,
                                'LanguageModel does not support guidance.'):
      self._run_decode(p, [], input_batch)

  def test_ngrammer_multi_sample_lpb(self):
    decode_max_len = 8
    max_decode_steps = 4
    num_layers = 2
    batch_size = 2
    num_heads = 2
    dim_per_head = 2
    vocab_size = 8
    ngrammer_params = pax_fiddle.Config(
        ngrammer.VQNgrammer,
        ngram_vocab_size=vocab_size**2,
        ngram_emb_dim=2,
        num_heads=num_heads,
        concat_ngrams=True,
        num_clusters=2,
        dim_per_head=dim_per_head,
    )
    lpb_lm_p = pax_fiddle.Config(
        transformer_models.TransformerLm,
        name='jax_lm_layer',
        model_dims=num_heads * dim_per_head,
        model_type=transformer_models.LanguageModelType.CAUSAL,
        packed_input=False,
        ngrammer_tpl=ngrammer_params,
        vocab_size=vocab_size,
    )
    stacked_transformer_tpl = lpb_lm_p.stacked_transformer_tpl
    stacked_transformer_tpl.model_dims = num_heads * dim_per_head
    stacked_transformer_tpl.hidden_dims = 2 * num_heads * dim_per_head
    stacked_transformer_tpl.num_heads = num_heads
    stacked_transformer_tpl.num_layers = num_layers
    params = stacked_transformer_tpl.transformer_layer_params_tpl
    params.tr_atten_tpl = pax_fiddle.Config(
        attentions.DotProductAttentionWithLPB
    )

    lm_p = pax_fiddle.Config(
        transformer_models.TransformerLm,
        name='jax_lm_layer',
        model_dims=num_heads * dim_per_head,
        model_type=transformer_models.LanguageModelType.CAUSAL,
        packed_input=False,
        ngrammer_tpl=ngrammer_params,
        vocab_size=vocab_size,
    )
    stacked_transformer_tpl = lm_p.stacked_transformer_tpl
    stacked_transformer_tpl.model_dims = num_heads * dim_per_head
    stacked_transformer_tpl.hidden_dims = 2 * num_heads * dim_per_head
    stacked_transformer_tpl.num_heads = num_heads
    stacked_transformer_tpl.num_layers = num_layers
    lpb_p = pax_fiddle.Config(
        models.LanguageModel,
        name='lpb_lm',
        lm_tpl=lpb_lm_p,
        model_type=LanguageModelType.CAUSAL,
    )

    lpb_p.decoder_tpl = models.SampleDecoderHParams(
        seqlen=decode_max_len + max_decode_steps,
        max_decode_steps=max_decode_steps,
        min_prefix_len=decode_max_len,
        fprop_for_prefix=True,
        lazy_prefix_broadcast=True,
        num_samples=2,
        temperature=1,
        k=8,
        p=0,
    )

    non_lpb_p = pax_fiddle.Config(
        models.LanguageModel,
        name='non_lpb_lm',
        lm_tpl=lm_p,
        model_type=LanguageModelType.CAUSAL,
    )

    non_lpb_p.decoder_tpl = models.SampleDecoderHParams(
        seqlen=decode_max_len + max_decode_steps,
        max_decode_steps=max_decode_steps,
        min_prefix_len=decode_max_len,
        fprop_for_prefix=True,
        lazy_prefix_broadcast=False,
        num_samples=2,
        temperature=1,
        k=8,
        p=0,
    )

    # EOS_ID is 1.
    input_ids = jnp.arange(2, decode_max_len + 2, dtype=jnp.int32)
    input_ids = jnp.expand_dims(input_ids, axis=0)
    prefix_lengths = jnp.array([decode_max_len], dtype=jnp.int32)
    labels = jnp.arange(start=1, stop=decode_max_len + 1, dtype=jnp.int32)
    labels = jnp.expand_dims(labels, axis=0)
    weights = jnp.ones_like(input_ids, dtype=jnp.bfloat16)
    paddings = jnp.zeros_like(input_ids, dtype=jnp.bfloat16)
    input_batch = NestedMap(
        ids=input_ids,
        prefix_lengths=prefix_lengths,
        paddings=paddings,
        weights=weights,
        labels=labels,
    )

    segment_ids = np.maximum(
        np.random.randint(0, 2, [batch_size, decode_max_len]),
        paddings.astype('int32'),
    )
    segment_ids = np.cumsum(segment_ids, axis=1)
    segment_pos = np.zeros_like(segment_ids)
    for b in range(batch_size):
      for t in range(1, decode_max_len):
        if segment_ids[b, t] == segment_ids[b, t - 1]:
          segment_pos[b, t] = segment_pos[b, t - 1] + 1
    segment_pos = jnp.asarray(segment_pos)
    input_batch['segment_pos'] = segment_pos
    input_batch['segment_ids'] = segment_ids

    prng_key = jax.random.PRNGKey(seed=123)
    with base_layer.JaxContext.new_context():
      lpb_lm = instantiate(lpb_p)
      non_lpb_lm = instantiate(non_lpb_p)
      initial_vars = lpb_lm.init(prng_key, input_batch)

      lpb_decodes, _ = lpb_lm.apply(
          initial_vars,
          input_batch,
          rngs={RANDOM: prng_key},
          method=lpb_lm.decode,
          mutable=[DECODE_CACHE, PREFIX_DECODE_CACHE],
      )

      non_lpb_decodes, _ = non_lpb_lm.apply(
          initial_vars,
          input_batch,
          rngs={RANDOM: prng_key},
          method=non_lpb_lm.decode,
          mutable=[DECODE_CACHE],
      )
      # Decode log probs and output ids must match.
      self.assertAllClose(
          lpb_decodes[1]['logprobs'], non_lpb_decodes[1]['logprobs']
      )
      self.assertAllClose(
          lpb_decodes[1]['output_ids'], non_lpb_decodes[1]['output_ids']
      )


class ClassifierModelTest(test_utils.TestCase):

  @parameterized.parameters([2, 6])
  def test_fprop(self, num_classes: int):

    p = pax_fiddle.Config(
        models.ClassificationModel,
        name='classifier',
        network_tpl=resnets.ResNet.HParamsResNet5(),
    )
    p.softmax_tpl.num_classes = num_classes
    p.softmax_tpl.input_dims = 16

    inputs = NestedMap(
        image=jnp.zeros((1, 25, 25, 3), jnp.float32),
        label_probs=jax.nn.one_hot(
            jnp.array([0]), num_classes, dtype=jnp.float32))
    model = instantiate(p)
    with base_layer.JaxContext.new_context():
      (metrics, _), _ = model.init_with_output(jax.random.PRNGKey(42), inputs)

    self.assertContainsSubset(['accuracy', 'error'], metrics)
    if num_classes > 5:
      self.assertContainsSubset(['acc5', 'error5'], metrics)


class SequenceModelTest(test_utils.TestCase):

  def test_encode_runs(self):
    data = NestedMap(
        ids=jnp.array([[11, 12, 13, 14, 15]], dtype=jnp.int32),
        paddings=jnp.array([[0, 1, 1, 1, 1]], dtype=jnp.float32),
        labels=jnp.ones([1, 5], jnp.float32),
        weights=jnp.ones([1, 5], jnp.float32),
    )

    input_batch = NestedMap(src=data, tgt=data)
    model_dims = 8
    model_p = pax_fiddle.Config(models.SequenceModel, name='test')
    model_p.model_tpl = pax_fiddle.Config(
        models.transformer_models.TransformerEncoderDecoder,
        model_dims=model_dims,
    )
    encoder_stacked_transformer_tpl = pax_fiddle.Config(
        transformers.StackedTransformer
    )
    encoder_stacked_transformer_tpl.num_layers = 2
    encoder_stacked_transformer_tpl.num_heads = 4
    encoder_stacked_transformer_tpl.model_dims = model_dims
    encoder_stacked_transformer_tpl.hidden_dims = model_dims * 4
    encoder_stacked_transformer_tpl.mask_self_attention = False
    decoder_stacked_transformer_tpl = pax_fiddle.Config(
        transformers.StackedTransformer
    )
    decoder_stacked_transformer_tpl.num_layers = 2
    decoder_stacked_transformer_tpl.num_heads = 4
    decoder_stacked_transformer_tpl.model_dims = model_dims
    decoder_stacked_transformer_tpl.hidden_dims = model_dims * 4
    decoder_stacked_transformer_tpl.mask_self_attention = True
    model_p.model_tpl.encoder_stacked_transformer_tpl = (
        encoder_stacked_transformer_tpl
    )
    model_p.model_tpl.decoder_stacked_transformer_tpl = (
        decoder_stacked_transformer_tpl
    )
    model_p.model_tpl.softmax_tpl = pax_fiddle.Config(
        embedding_softmax.SharedEmbeddingSoftmax,
        input_dims=model_dims,
        num_classes=16,
    )
    seq_model = instantiate(model_p)
    prng_key = jax.random.PRNGKey(seed=9)
    initial_vars = seq_model.init(prng_key, input_batch)
    results = seq_model.apply(
        initial_vars,
        input_batch.src,
        rngs={RANDOM: prng_key},
        method=seq_model.encode,
    )

    self.assertIn('embeddings', results)
    self.assertSequenceEqual(results.embeddings.shape, (1, 5, 8))

  def _run_decode(self, decoder_p, input_batch):
    model_dims = 8
    model_p = pax_fiddle.Config(models.SequenceModel, name='test')
    model_p.model_tpl = pax_fiddle.Config(
        models.transformer_models.TransformerEncoderDecoder,
        model_dims=model_dims,
    )
    encoder_stacked_transformer_tpl = pax_fiddle.Config(
        transformers.StackedTransformer
    )
    encoder_stacked_transformer_tpl.num_layers = 2
    encoder_stacked_transformer_tpl.num_heads = 4
    encoder_stacked_transformer_tpl.model_dims = model_dims
    encoder_stacked_transformer_tpl.hidden_dims = model_dims * 4
    encoder_stacked_transformer_tpl.mask_self_attention = False
    decoder_stacked_transformer_tpl = pax_fiddle.Config(
        transformers.StackedTransformer
    )
    decoder_stacked_transformer_tpl.num_layers = 2
    decoder_stacked_transformer_tpl.num_heads = 4
    decoder_stacked_transformer_tpl.model_dims = model_dims
    decoder_stacked_transformer_tpl.hidden_dims = model_dims * 4
    decoder_stacked_transformer_tpl.mask_self_attention = True
    model_p.model_tpl.encoder_stacked_transformer_tpl = (
        encoder_stacked_transformer_tpl
    )
    model_p.model_tpl.decoder_stacked_transformer_tpl = (
        decoder_stacked_transformer_tpl
    )
    model_p.model_tpl.softmax_tpl = pax_fiddle.Config(
        embedding_softmax.SharedEmbeddingSoftmax,
        input_dims=model_dims,
        num_classes=16,
    )
    seq_model = instantiate(model_p)
    prng_key = jax.random.PRNGKey(seed=9)
    mdl_vars = seq_model.init(prng_key, input_batch)
    results, _ = seq_model.apply(
        mdl_vars,
        mutable=[
            DECODE_CACHE,
        ],
        rngs={RANDOM: prng_key},
        method=seq_model.decode_with_params,
        input_batch=input_batch,
        decoder_params=decoder_p,
    )
    _, results, _ = results
    return results

  def test_greedy_decode(self):
    data = NestedMap(
        ids=jnp.array([[11, 12, 13, 14, 15]], dtype=jnp.int32),
        paddings=jnp.array([[0, 1, 1, 1, 1]], dtype=jnp.float32),
        labels=jnp.ones([1, 5], jnp.float32),
        weights=jnp.ones([1, 5], jnp.float32),
    )
    input_batch = NestedMap(src=data, tgt=data)
    decoder_p = models.GreedyDecoderHParams(seqlen=5, max_decode_steps=5)
    results = self._run_decode(decoder_p, input_batch)
    self.assertIn('output_ids', results)
    self.assertSequenceEqual(results.output_ids.shape, (1, 1, 5))
    self.assertArraysEqual(results.output_ids, [[[11, 10, 10, 10, 10]]])

  def test_sample_decode(self):
    data = NestedMap(
        ids=jnp.array([[11, 12, 13, 14, 15]], dtype=jnp.int32),
        paddings=jnp.array([[0, 1, 1, 1, 1]], dtype=jnp.float32),
        labels=jnp.ones([1, 5], jnp.float32),
        weights=jnp.ones([1, 5], jnp.float32),
    )
    input_batch = NestedMap(src=data, tgt=data)
    decoder_p = models.SampleDecoderHParams(
        num_samples=3, k=0, seqlen=5, max_decode_steps=5
    )
    results = self._run_decode(decoder_p, input_batch)
    self.assertIn('output_ids', results)
    self.assertSequenceEqual(results.output_ids.shape, (1, 3, 5))
    self.assertArraysEqual(
        results.output_ids,
        [[[11, 10, 9, 1, 14], [11, 14, 14, 7, 14], [11, 11, 13, 3, 6]]],
    )

  def test_beam_search_decode(self):
    data = NestedMap(
        ids=jnp.array([[11, 12, 13, 14, 15]], dtype=jnp.int32),
        paddings=jnp.array([[0, 1, 1, 1, 1]], dtype=jnp.float32),
        labels=jnp.ones([1, 5], jnp.float32),
        weights=jnp.ones([1, 5], jnp.float32),
    )
    input_batch = NestedMap(src=data, tgt=data)
    decoder_p = models.BeamSearchHParams(
        beam_size=4, seqlen=5, max_decode_steps=5
    )
    results = self._run_decode(decoder_p, input_batch)
    self.assertIn('output_ids', results)
    self.assertSequenceEqual(results.output_ids.shape, (1, 4, 6))
    self.assertArraysEqual(
        results.output_ids,
        [[
            [2, 0, 0, 0, 0, 0],
            [11, 2, 0, 0, 0, 0],
            [15, 2, 0, 0, 0, 0],
            [15, 15, 2, 0, 0, 0],
        ]],
    )


def _flatten_input_data(lm_input):
  return {
      'y_w/inputs': lm_input.inputs,
      'y_w/paddings': lm_input.paddings,
      'y_w/labels/ids': lm_input.labels.class_ids,
      'y_w/labels/weights': lm_input.labels.class_weights,
      'y_w/segment_ids': lm_input.segment_ids,
      'y_w/segment_pos': lm_input.segment_pos,
      'y_w/inputs_indicator': lm_input.inputs_indicator,
      'y_l/inputs': lm_input.inputs,
      'y_l/paddings': lm_input.paddings,
      'y_l/labels/ids': lm_input.labels.class_ids,
      'y_l/labels/weights': lm_input.labels.class_weights,
      'y_l/segment_ids': lm_input.segment_ids,
      'y_l/segment_pos': lm_input.segment_pos,
      'y_l/inputs_indicator': lm_input.inputs_indicator,
  }


class TransformerLmDpoTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)
    tf.random.set_seed(123)

  def test_transformer_lm_dpo(self):
    seq_len = 512
    model_config = transformer_models.TransformerLm.config(
        model_dims=32,
        vocab_size=52,
        position_emb_tpl=embedding_softmax.PositionalEmbedding.config(),
        stacked_transformer_tpl=transformers.StackedTransformer.config(
            model_dims=32,
            hidden_dims=4 * 32,
            num_heads=4,
            num_layers=1,
        ),
    )
    model_config.softmax_tpl.scale_sqrt_depth = True
    p = models.LanguageModelDPO.config(ref_mdl=model_config, mdl=model_config)

    batch_size = 8
    dpo_lm = instantiate(p)

    input_ids = jax.random.randint(
        jax.random.PRNGKey(1234), [batch_size, seq_len], 0, 51
    )
    labels = jnp.concatenate((input_ids[:, 1:], input_ids[:, :1]), axis=1)
    weights = jnp.ones([batch_size, seq_len])
    lm_input = models.DPOExampleHalf(
        inputs=input_ids,
        inputs_indicator=jnp.ones([batch_size, seq_len]),
        labels=models.Labels(class_ids=labels, class_weights=weights),
        paddings=jnp.zeros([batch_size, seq_len]),
        segment_ids=jnp.ones([batch_size, seq_len]),
        segment_pos=jnp.tile(jnp.arange(0, seq_len), [batch_size, 1]),
    )

    input_batch = _flatten_input_data(lm_input)

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = dpo_lm.init(prng_key, input_batch)
      outputs, _ = dpo_lm.apply(initial_vars, input_batch)
      logging.info('outputs: %s', outputs)
      self.assertEqual(0.6931472, outputs.total_loss[0])


if __name__ == '__main__':
  absltest.main()
