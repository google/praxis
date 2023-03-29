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
from praxis import pax_fiddle

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import decoder_utils
from praxis import py_utils
from praxis import pytypes
from praxis import test_utils
from praxis.layers import models
from praxis.layers import resnets
from praxis.layers import transformers
from praxis.layers import transformer_models
from praxis.layers import embedding_softmax

NestedMap = py_utils.NestedMap
instantiate = base_layer.instantiate
LanguageModelType = transformer_models.LanguageModelType
JTensor = pytypes.JTensor

RANDOM = base_layer.RANDOM
DECODE_CACHE = base_layer.DECODE_CACHE


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
        ids=jnp.array([[11, 12, 13, 14, 15]], dtype=jnp.int32),
        paddings=jnp.array([[0., 0., 1., 1., 1.]], dtype=jnp.float32),
    )
    results = self._run_decode(p, logits, input_batch)
    self.assertArraysEqual(results.prefix_lengths,
                           np.array([[2]], dtype=np.int32))
    # We copy prefix of length 2 from input.ids, so the first argmax
    # from logits is unused. Remaining 3 ids are from argmax.
    if fprop_for_prefix:
      self.assertArraysEqual(
          results.output_ids,
          np.array([[[11, 12, 1, 3, 4, 0, 0, 0]]], dtype=np.int32))
      self.assertArraysEqual(
          results.prefix_ids,
          np.array([[[11, 12, 0, 0, 0, 0, 0, 0]]], dtype=np.int32))
    else:
      self.assertArraysEqual(results.output_ids,
                             np.array([[[11, 12, 3, 4, 5]]], dtype=np.int32))
      self.assertArraysEqual(results.prefix_ids,
                             np.array([[[11, 12, 0, 0, 0]]], dtype=np.int32))

    self.assertArraysEqual(results.decode_lengths,
                           np.array([[5]], dtype=np.int32))

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
        ids=jnp.array([[11, 12, 13, 14, 15]], dtype=jnp.int32),
        paddings=jnp.array([[0., 0., 1., 1., 1.]], dtype=jnp.float32),
        inputs_indicator=jnp.array([[1, 1, 0, 0, 0]], dtype=jnp.float32),
    )
    results = self._run_decode(
        p, logits, input_batch, model_type=LanguageModelType.PREFIX)
    self.assertArraysEqual(results.prefix_lengths,
                           np.array([[2]], dtype=np.int32))
    # We copy prefix of length 2 from input.ids, so the first argmax
    # from logits is unused. Remaining 3 ids are from argmax.
    if fprop_for_prefix:
      self.assertArraysEqual(
          results.output_ids,
          np.array([[[11, 12, 1, 3, 4, 0, 0, 0]]], dtype=np.int32))
      self.assertArraysEqual(
          results.prefix_ids,
          np.array([[[11, 12, 0, 0, 0, 0, 0, 0]]], dtype=np.int32))
    else:
      self.assertArraysEqual(results.output_ids,
                             np.array([[[11, 12, 3, 4, 5]]], dtype=np.int32))
      self.assertArraysEqual(results.prefix_ids,
                             np.array([[[11, 12, 0, 0, 0]]], dtype=np.int32))

    self.assertArraysEqual(results.decode_lengths,
                           np.array([[5]], dtype=np.int32))

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
        ids=jnp.array([[11, 13, 15], [12, 14, 16], [20, 30, 40]],
                      dtype=jnp.int32),
        paddings=jnp.array([[0, 0, 1], [0, 1, 1], [0, 1, 1]],
                           dtype=jnp.float32),
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
        np.array([[[11, 13, 4, 3, 3, 4, 0]], [[12, 3, 4, 2, 0, 0, 0]],
                  [[20, 3, 2, 0, 0, 0, 0]]],
                 dtype=np.int32))
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
        gumbel_prng_key=jnp.array([0, 0, 0], dtype=jnp.uint32),
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
          np.array([[[11, 13, 4, 3, 3, 4, 0], [11, 13, 4, 3, 3, 4, 0]],
                    [[12, 3, 4, 2, 0, 0, 0], [12, 3, 4, 2, 0, 0, 0]],
                    [[20, 3, 2, 0, 0, 0, 0], [20, 3, 2, 0, 0, 0, 0]]],
                   dtype=np.int32))
      self.assertArraysEqual(
          results.decode_lengths,
          np.array([[6, 6], [4, 4], [3, 3]], dtype=np.int32),
      )
    else:
      # Gumbel noise will make some difference between samples.
      self.assertArraysEqual(
          results.output_ids,
          np.array([[[11, 13, 4, 3, 3, 4, 0], [11, 13, 4, 0, 3, 4, 0]],
                    [[12, 3, 4, 2, 0, 0, 0], [12, 3, 4, 0, 0, 0, 0]],
                    [[20, 3, 2, 0, 0, 0, 0], [20, 3, 2, 0, 0, 0, 0]]],
                   dtype=np.int32))
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
    )
    results = self._run_decode(p, logits, input_batch)

    self.assertArraysEqual(
        results.output_ids,
        np.array(
            [
                [[11, 13, 4, 0, 3, 0, 0]],
                [[12, 14, 1, 0, 0, 0, 0]],
                [[20, 3, 2, 0, 0, 0, 0]],
            ],
            dtype=np.int32,
        ),
    )
    self.assertArraysEqual(
        results.decode_lengths, np.array([[6], [3], [3]], dtype=np.int32)
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


if __name__ == '__main__':
  absltest.main()
