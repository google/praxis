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

"""Unit tests for sample_decode."""
from typing import Sequence

from absl.testing import absltest
from absl.testing import parameterized
import flax.linen as nn
import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import base_model
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis import sample_decode
from praxis import test_utils

NestedMap = py_utils.NestedMap
WeightHPrams = base_layer.WeightHParams
instantiate = base_layer.instantiate
JTensor = pytypes.JTensor

RANDOM = base_layer.RANDOM
DECODE_CACHE = base_layer.DECODE_CACHE
SUMMARIES = base_layer.SUMMARIES


class TestNextTokenSampler(sample_decode.BaseNextTokenSampler):

  def __call__(
      self,
      mdl,
      logits,
      temperature,
      decode_loop_state,
      per_example_top_p,
      per_example_top_k,
      gumbel_prng_key,
  ):
    del (
        mdl,
        temperature,
        decode_loop_state,
        per_example_top_p,
        gumbel_prng_key,
    )
    return NestedMap(new_ids=jnp.array([1234, 2345]), logits=logits)


class TestNextTokenSamplerWithAdditionalState(
    sample_decode.BaseNextTokenSampler
):

  def init_decode_loop_state(
      self,
      decode_loop_state: NestedMap,
      model: base_layer.BaseLayerApi | None = None,
      batch_size: int | None = None,
      eos_id: int | Sequence[int] | JTensor | None = None,
  ) -> NestedMap:
    """Initialize addition decode loop state."""
    decode_loop_state.mock_state = jnp.zeros(shape=batch_size, dtype=jnp.bool_)
    return decode_loop_state

  def __call__(
      self,
      mdl,
      logits,
      temperature,
      decode_loop_state,
      per_example_top_p,
      per_example_top_k,
      gumbel_prng_key,
  ):
    del (
        mdl,
        temperature,
        per_example_top_p,
        per_example_top_k,
        gumbel_prng_key,
    )
    # Update additional states (Change additional states here).
    return NestedMap(
        new_ids=jnp.array([1234, 2345]),
        logits=logits,
        mock_state=jnp.ones_like(decode_loop_state.mock_state, dtype=jnp.bool_),
    )


class TestModelWithLogits(base_model.BaseModel):
  use_dummy_next_token_sampler: bool = True
  vocab_size: int = 0
  num_samples: int = 0
  seq_len: int = 0
  batch_size: int = 0
  logits: jnp.ndarray = None

  def setup(self) -> None:
    super().setup()
    assert self.logits is not None
    expected_shape = (
        self.seq_len,
        self.batch_size * self.num_samples,
        self.vocab_size,
    )
    assert self.logits.shape == expected_shape, (
        self.logits.shape,
        expected_shape,
    )
    self.next_token_sampler = base_layer.instantiate(
        pax_fiddle.Config(
            sample_decode.DefaultNextTokenSampler, top_k=0, top_p=1.0
        )
    )

  def __call__(self, *args, **kwargs):
    del args, kwargs

  def extend_step(self, ids, segment_pos):
    print(ids.shape, segment_pos.shape)
    assert segment_pos.shape == (self.batch_size * self.num_samples,), (
        segment_pos.shape,
        (self.batch_size * self.num_samples,),
    )
    time_step = segment_pos[0] + 1
    logits_at_t = self.logits[time_step, :, :]
    self.add_summary('logits', logits_at_t)
    self.add_summary('time_step', time_step)
    return logits_at_t


class TestModel(base_model.BaseModel):
  use_dummy_next_token_sampler: bool = True
  vocab_size: int = 0
  num_samples: int = 0
  seq_len: int = 0
  batch_size: int = 0

  def setup(self) -> None:
    super().setup()
    logits_wp = base_layer.WeightHParams(
        shape=[
            self.seq_len,
            self.batch_size * self.num_samples,
            self.vocab_size,
        ]
    )
    self.create_variable('logits', logits_wp)
    if self.use_dummy_next_token_sampler:
      self.next_token_sampler = base_layer.instantiate(
          pax_fiddle.Config(TestNextTokenSampler)
      )
    else:
      self.next_token_sampler = base_layer.instantiate(
          pax_fiddle.Config(
              sample_decode.DefaultNextTokenSampler, top_k=0, top_p=1.0
          )
      )

  def __call__(self, *args, **kwargs):
    # A dummy __call__ function
    del args, kwargs

  # do something here
  def extend_step(self, ids, segment_pos):
    assert segment_pos.shape == (self.batch_size * self.num_samples,), (
        segment_pos.shape,
        (self.batch_size * self.num_samples,),
    )
    time_step = segment_pos[0] + 1
    logits_at_t = self.theta.logits[time_step, :, :]
    self.add_summary('logits', logits_at_t)
    self.add_summary('time_step', time_step)
    return logits_at_t


class TestModelWithAdditionalState(base_model.BaseModel):
  vocab_size: int = 0
  num_samples: int = 0
  seq_len: int = 0
  batch_size: int = 0

  def setup(self) -> None:
    super().setup()
    logits_wp = base_layer.WeightHParams(
        shape=[
            self.seq_len,
            self.batch_size * self.num_samples,
            self.vocab_size,
        ]
    )
    self.create_variable('logits', logits_wp)
    self.next_token_sampler = base_layer.instantiate(
        pax_fiddle.Config(TestNextTokenSamplerWithAdditionalState)
    )

  def __call__(self, *args, **kwargs):
    # A dummy __call__ function
    del args, kwargs

  # do something here
  def extend_step(self, ids, segment_pos):
    assert segment_pos.shape == (self.batch_size * self.num_samples,), (
        segment_pos.shape,
        (self.batch_size * self.num_samples,),
    )
    time_step = segment_pos[0] + 1
    logits_at_t = self.theta.logits[time_step, :, :]
    return logits_at_t


class SampleDecodeHelperTest(test_utils.TestCase):

  def test_split_batch_dim(self):
    x = jnp.array([[1, 2], [1, 2], [3, 4], [3, 4]], dtype=np.int32)
    self.assertArraysEqual(
        sample_decode.split_batch_dim(x, batch_dim=0, num_samples=2),
        np.array([[[1, 2], [1, 2]], [[3, 4], [3, 4]]], dtype=np.int32),
    )

  def test_reorder_with_indices(self):
    indices = jnp.array([[0, 2, 1], [2, 0, 1]], dtype=jnp.int32)
    x = jnp.array(
        [[[0, 0, 0], [1, 1, 1], [2, 2, 2]], [[0, 0, 0], [1, 1, 1], [2, 2, 2]]],
        dtype=jnp.int32,
    )

    reordered_x = sample_decode.reorder_with_indices(x, indices)

    self.assertArraysEqual(
        reordered_x,
        jnp.array(
            [
                [[0, 0, 0], [2, 2, 2], [1, 1, 1]],
                [[2, 2, 2], [0, 0, 0], [1, 1, 1]],
            ],
            dtype=jnp.int32,
        ),
    )

  def test_sort_samples_by_scores(self):
    logprobs = jnp.array(
        [
            [
                [0.1, 0.1, 1.0],  #  sum is 0.2
                [0.3, 0.3, 0.3],  #  sum is 0.9
                [0.2, 0.2, 1.0],  #  sum is 0.4
            ],
            [
                [0.9, 0.9, 0.9],  # sum is 2.7
                [0.1, 1.0, 1.0],  # sum is 0.1
                [0.2, 0.2, 1.0],  # sum is 0.4
            ],
        ],
        dtype=jnp.float32,
    )
    x = jnp.array(
        [[[0, 0, 0], [1, 1, 1], [2, 2, 2]], [[0, 0, 0], [1, 1, 1], [2, 2, 2]]],
        dtype=jnp.int32,
    )
    result = NestedMap()
    result.logprobs = logprobs
    result.x = x

    result = sample_decode.sort_samples_by_scores(result)

    # Verify values in result are sorted and ordered at dimension 1.
    self.assertArraysEqual(
        result.x,
        jnp.array(
            [
                [[1, 1, 1], [2, 2, 2], [0, 0, 0]],
                [[0, 0, 0], [2, 2, 2], [1, 1, 1]],
            ],
            dtype=jnp.int32,
        ),
    )
    self.assertArraysEqual(
        result.logprobs,
        jnp.array(
            [
                [
                    [0.3, 0.3, 0.3],  #  sum is 0.9
                    [0.2, 0.2, 1.0],  #  sum is 0.4
                    [0.1, 0.1, 1.0],  #  sum is 0.2
                ],
                [
                    [0.9, 0.9, 0.9],  # sum is 2.7
                    [0.2, 0.2, 1.0],  # sum is 0.4
                    [0.1, 1.0, 1.0],  # sum is 0.1
                ],
            ],
            dtype=jnp.float32,
        ),
    )

  def test_right_align_prefix_ids(self):
    prefix_ids = jnp.array([[1, 2, 0], [1, 0, 0], [0, 1, 2]], dtype=jnp.int32)
    prefix_lengths = jnp.array([2, 1, 3], dtype=jnp.int32)

    (right_align_prefix_ids, right_align_prefix_paddings) = (
        sample_decode.right_align_prefix_ids(
            prefix_ids, prefix_lengths, jnp.int32
        )
    )

    self.assertArraysEqual(
        right_align_prefix_ids,
        jnp.array([[0, 1, 2], [0, 0, 1], [0, 1, 2]], dtype=jnp.int32),
    )

    self.assertArraysEqual(
        right_align_prefix_paddings,
        jnp.array([[1, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=jnp.int32),
    )

  def test_right_align_segment_position(self):
    lengths = jnp.array([5, 4, 6], dtype=jnp.int32)

    right_align_segment_pos = sample_decode.right_align_segment_position(
        lengths, max_length=6
    )

    self.assertArraysEqual(
        right_align_segment_pos,
        jnp.array(
            [[0, 0, 1, 2, 3, 4], [0, 0, 0, 1, 2, 3], [0, 1, 2, 3, 4, 5]],
            dtype=jnp.int32,
        ),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='use_dummy_next_token_sampler',
          use_dummy_next_token_sampler=True,
          use_gumbel_prng_key=False,  # doesn't matter
          num_per_token_logprobs=None,
      ),
      dict(
          testcase_name='without_gumbel_prng_key',
          use_dummy_next_token_sampler=False,
          use_gumbel_prng_key=False,
          num_per_token_logprobs=1,
      ),
      dict(
          testcase_name='with_gumbel_prng_key',
          use_dummy_next_token_sampler=False,
          use_gumbel_prng_key=True,
          num_per_token_logprobs=2,
      ),
      dict(
          testcase_name='return_entropy_score',
          use_dummy_next_token_sampler=True,
          use_gumbel_prng_key=False,  # doesn't matter
          return_entropy_score=True,
          num_per_token_logprobs=3,
      ),
  )
  def test_sample_decode(
      self,
      use_dummy_next_token_sampler,
      use_gumbel_prng_key,
      return_entropy_score=False,
      num_per_token_logprobs=None,
  ):
    batch_size = 1
    num_samples = 2
    seq_len = 3
    vocab_size = 4
    model_p = pax_fiddle.Config(
        TestModel,
        name='test_model',
        batch_size=batch_size,
        num_samples=num_samples,
        seq_len=seq_len,
        vocab_size=vocab_size,
        use_dummy_next_token_sampler=use_dummy_next_token_sampler,
    )

    def extend_step_fn(mdl, ids, segment_pos):
      logits = mdl.extend_step(ids, segment_pos=segment_pos)
      return logits

    def transform_decode_state_fn(mdl, transform_fn):
      del mdl
      del transform_fn

    model = instantiate(model_p)
    init_vars = model.init(rngs=jax.random.PRNGKey(1234))
    logits_var = init_vars['params']['logits']
    # One can override logits to inject logits to be used during decoding.

    input_ids = jnp.zeros([batch_size, seq_len], dtype=jnp.int32)
    input_paddings = jnp.zeros([batch_size, seq_len], dtype=jnp.float32)
    def decode_fn(model, input_ids, input_paddings):
      gumbel_prng_key = None
      if use_gumbel_prng_key:
        gumbel_prng_key = jax.vmap(jax.random.PRNGKey)(
            jnp.arange(1002, 1002 + batch_size)
        )
      return sample_decode.sample_decode(
          model,
          extend_step_fn,
          transform_decode_state_fn,
          None,
          model.next_token_sampler,
          input_ids,
          input_paddings,
          prefix_lengths=jnp.zeros([batch_size], dtype=jnp.int32),
          seq_len=seq_len,
          num_samples=num_samples,
          gumbel_prng_key=gumbel_prng_key,
          max_prefix_len=0,
          max_decode_steps=seq_len,
          fprop_for_prefix=True,
          # Call the scan loop.
          early_exit=False,
          return_entropy_score=return_entropy_score,
          num_per_token_logprobs=None
          if num_per_token_logprobs is None
          else jnp.array([num_per_token_logprobs], dtype=jnp.int32),
      )

    mutables = [SUMMARIES, DECODE_CACHE]
    rngs = {'random': jax.random.PRNGKey(9382)}

    # test that we can fetch arbitrary summary out.
    result, updated_vars = nn.apply(decode_fn, model, mutable=mutables)(
        init_vars, input_ids, input_paddings, rngs=rngs
    )
    logits_summary = updated_vars['summaries']['logits_scalar']
    new_ids_summary = updated_vars['summaries']['new_ids_scalar']
    time_step_summary = updated_vars['summaries']['time_step_scalar']
    print('logits_var', logits_var)
    print('logits_summary', logits_summary)
    print('time_step_summary', time_step_summary)
    self.assertAllClose(logits_var, logits_summary)
    if use_dummy_next_token_sampler:
      self.assertAllClose(
          new_ids_summary, jnp.tile(jnp.array([1234, 2345]), [3, 1])
      )
    elif use_gumbel_prng_key:
      self.assertAllClose(new_ids_summary, jnp.array([[1, 2], [3, 0], [0, 0]]))
    else:
      self.assertAllClose(new_ids_summary, jnp.array([[3, 0], [2, 1], [0, 1]]))
    # Check score
    if return_entropy_score:
      prob = jax.nn.softmax(logits_var)
      entropy = jnp.transpose(-jnp.sum(prob * jnp.log(prob), axis=-1))
      self.assertEqual(entropy.shape, result['entropy'][0].shape)
      np.testing.assert_array_almost_equal(
          entropy, result['entropy'][0], decimal=5
      )
    else:
      self.assertNotIn('entropy', result)

    if num_per_token_logprobs is None:
      self.assertNotIn('top_candidate_logprobs', result)
      self.assertNotIn('top_candidate_ids', result)
    else:
      top_candidate_ids = result['top_candidate_ids']
      top_candidate_logprobs = result['top_candidate_logprobs']
      # Check shape.
      shape = (
          batch_size,
          num_samples,
          seq_len,
          sample_decode.MAX_NUM_PER_TOKEN_LOGPROBS,
      )
      self.assertEqual(shape, top_candidate_ids.shape)
      self.assertEqual(shape, top_candidate_logprobs.shape)
      # Check that ids outside of the top `num_per_token_logprobs` are 0.
      self.assertArraysEqual(
          top_candidate_ids[:, :, :, num_per_token_logprobs:], 0
      )
      # Check that logprobs outside of the top `num_per_token_logprobs` are 1.
      self.assertArraysEqual(
          top_candidate_logprobs[:, :, :, num_per_token_logprobs:], 1.0
      )
      # Check that logprobs are sorted in descending order.
      logprobs = top_candidate_logprobs[:, :, :, :num_per_token_logprobs]
      self.assertArraysEqual(
          jnp.flip(jnp.sort(logprobs), -1),
          logprobs,
      )
      # Check that the top logprobs and ids are correct.
      # swap seq_len and num_samples -> (sample, seq, vocab).
      logits_var = jnp.swapaxes(logits_var, 0, 1)
      logprobs = jax.nn.log_softmax(logits_var)
      logprobs, ids = jax.lax.top_k(logprobs, num_per_token_logprobs)
      # batch size is 1.
      self.assertEqual(1, top_candidate_logprobs.shape[0])
      self.assertEqual(1, top_candidate_ids.shape[0])
      self.assertArraysEqual(
          logprobs, top_candidate_logprobs[0, :, :, :num_per_token_logprobs]
      )
      self.assertArraysEqual(
          ids, top_candidate_ids[0, :, :, :num_per_token_logprobs]
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='cf_guidance_scale_1d',
          cf_guidance_scale_1d=True,
      ),
      dict(
          testcase_name='cf_guidance_scale_2d',
          cf_guidance_scale_1d=False,
      ),
  )
  def test_sample_decode_cf_guidance(self, cf_guidance_scale_1d):
    batch_size = 1
    num_samples = 2
    seq_len = 3
    vocab_size = 4
    model_p = pax_fiddle.Config(
        TestModel,
        name='test_model',
        batch_size=batch_size * 2,
        num_samples=num_samples,
        seq_len=seq_len,
        vocab_size=vocab_size,
        use_dummy_next_token_sampler=False,
    )

    def extend_step_fn(mdl, ids, segment_pos):
      logits = mdl.extend_step(ids, segment_pos=segment_pos)
      return logits

    def transform_decode_state_fn(mdl, transform_fn):
      del mdl
      del transform_fn

    model = instantiate(model_p)
    init_vars = model.init(rngs=jax.random.PRNGKey(1234))
    logits_var = init_vars['params']['logits']
    # One can override logits to inject logits to be used during decoding.

    input_ids = jnp.zeros([batch_size * 2, seq_len], dtype=jnp.int32)
    input_paddings = jnp.zeros([batch_size * 2, seq_len], dtype=jnp.float32)
    if cf_guidance_scale_1d:
      cf_guidance_scale = jnp.ones([batch_size], dtype=jnp.float32) * 1.75
    else:
      cf_guidance_scale = (
          jnp.ones([batch_size, num_samples], dtype=jnp.float32) * 1.75
      )

    def decode_fn(model, input_ids, input_paddings):
      return sample_decode.sample_decode(
          model,
          extend_step_fn,
          transform_decode_state_fn,
          None,
          model.next_token_sampler,
          input_ids,
          input_paddings,
          prefix_lengths=jnp.zeros([batch_size * 2], dtype=jnp.int32),
          seq_len=seq_len,
          num_samples=num_samples,
          max_prefix_len=0,
          max_decode_steps=seq_len,
          cf_guidance_scale=cf_guidance_scale,
          fprop_for_prefix=True,
          # Call the scan loop.
          early_exit=False,
      )

    mutables = [SUMMARIES, DECODE_CACHE]
    rngs = {'random': jax.random.PRNGKey(9382)}

    # test that we can fetch arbitrary summary out.
    _, updated_vars = nn.apply(decode_fn, model, mutable=mutables)(
        init_vars, input_ids, input_paddings, rngs=rngs
    )
    logits_summary = updated_vars['summaries']['logits_scalar']
    new_ids_summary = updated_vars['summaries']['new_ids_scalar']
    time_step_summary = updated_vars['summaries']['time_step_scalar']
    print('logits_var', logits_var)
    print('logits_summary', logits_summary)
    print('time_step_summary', time_step_summary)
    self.assertAllClose(logits_var, logits_summary)
    self.assertAllClose(new_ids_summary, jnp.array([[3, 3], [2, 1], [1, 1]]))

  def test_sample_decode_with_optimize_eos(self):
    batch_size = 1
    num_samples = 2
    seq_len = 4
    vocab_size = 4
    model_p = pax_fiddle.Config(
        TestModelWithLogits,
        name='test_model',
        batch_size=batch_size,
        num_samples=num_samples,
        seq_len=seq_len,
        vocab_size=vocab_size,
        use_dummy_next_token_sampler=False,
        logits=jnp.array([
            [[1, -1e4, 0, 1], [1, -1e4, -1e4, 1]],
            [[1, -1e4, 1, 2], [1, -1e4, -1e4, 2]],
            [[-1e4, 1, -1e4, -1e4], [1, -1e4, -1e4, 3]],
            [[1, 2, 1, 2], [-1e4, 1, 3, -1e4]],
        ]),
    )

    def extend_step_fn(mdl, ids, segment_pos):
      return mdl.extend_step(ids, segment_pos=segment_pos)

    def transform_decode_state_fn(mdl, transform_fn):
      del mdl
      del transform_fn

    model = instantiate(model_p)
    init_vars = model.init(rngs=jax.random.PRNGKey(1234))
    # One can override logits to inject logits to be used during decoding.

    input_ids = jnp.zeros([batch_size, seq_len], dtype=jnp.int32)
    input_paddings = jnp.zeros([batch_size, seq_len], dtype=jnp.float32)

    def decode_fn(model, input_ids, input_paddings):
      return sample_decode.sample_decode(
          model,
          extend_step_fn,
          transform_decode_state_fn,
          None,
          model.next_token_sampler,
          input_ids,
          input_paddings,
          seq_len=seq_len,
          num_samples=num_samples,
          prefix_lengths=jnp.zeros([batch_size], dtype=jnp.int32),
          gumbel_prng_key=None,
          max_prefix_len=0,
          eos_id=[1, 2],
          max_decode_steps=seq_len,
          fprop_for_prefix=True,
          # Call the scan loop.
          early_exit=False,
          return_entropy_score=True,
          optimize_eos=True,
      )

    mutables = [SUMMARIES, DECODE_CACHE]
    rngs = {'random': jax.random.PRNGKey(9382)}

    # test that we can fetch arbitrary summary out.
    result, updated_vars = nn.apply(decode_fn, model, mutable=mutables)(
        init_vars, input_ids, input_paddings, rngs=rngs
    )
    new_ids_summary = updated_vars['summaries']['new_ids_scalar']
    time_step_summary = updated_vars['summaries']['time_step_scalar']
    print('new_ids_summary', new_ids_summary)
    print('time_step_summary', time_step_summary)
    self.assertAllClose(
        new_ids_summary, jnp.array([[3, 3], [3, 3], [0, 3], [0, 3]])
    )
    self.assertAllClose(
        result.output_ids, jnp.array([[[3, 3, 3, 2], [3, 3, 1, 0]]])
    )

  def test_sample_decode_with_additional_states(self):
    batch_size = 1
    num_samples = 2
    seq_len = 3
    vocab_size = 4
    model_p = pax_fiddle.Config(
        TestModelWithAdditionalState,
        batch_size=batch_size,
        num_samples=num_samples,
        seq_len=seq_len,
        vocab_size=vocab_size,
        name='test_model_additional_states',
    )

    def extend_step_fn(mdl, ids, segment_pos):
      logits = mdl.extend_step(ids, segment_pos=segment_pos)
      return logits

    def transform_decode_state_fn(mdl, transform_fn):
      del mdl
      del transform_fn

    model = instantiate(model_p)
    init_vars = model.init(rngs=jax.random.PRNGKey(1234))

    input_ids = jnp.zeros([batch_size, seq_len], dtype=jnp.int32)
    input_paddings = jnp.zeros([batch_size, seq_len], dtype=jnp.float32)

    def decode_fn(model, input_ids, input_paddings):
      return sample_decode.sample_decode(
          model,
          extend_step_fn,
          transform_decode_state_fn,
          None,
          model.next_token_sampler,
          input_ids,
          input_paddings,
          prefix_lengths=jnp.zeros([batch_size], dtype=jnp.int32),
          seq_len=seq_len,
          num_samples=num_samples,
          gumbel_prng_key=None,
          max_prefix_len=0,
          max_decode_steps=seq_len,
          fprop_for_prefix=True,
          # Call the scan loop.
          early_exit=False,
      )

    mutables = [DECODE_CACHE]
    rngs = {'random': jax.random.PRNGKey(9382)}

    result, _ = nn.apply(decode_fn, model, mutable=mutables)(
        init_vars, input_ids, input_paddings, rngs=rngs
    )
    # Check updated states.
    self.assertArraysEqual(
        result.mock_state,
        np.ones_like(result.mock_state, dtype=np.bool_),
    )

  def test_sample_decode_with_single_sample(self):
    batch_size = 1
    num_samples = 1
    seq_len = 3
    vocab_size = 4
    model_p = pax_fiddle.Config(
        TestModel,
        batch_size=batch_size,
        num_samples=num_samples,
        seq_len=seq_len,
        vocab_size=vocab_size,
        use_dummy_next_token_sampler=False,
        name='test_model_single_sample',
    )

    def extend_step_fn(mdl, ids, segment_pos):
      logits = mdl.extend_step(ids, segment_pos=segment_pos)
      return logits

    def transform_decode_state_fn(mdl, transform_fn):
      del mdl
      del transform_fn

    model = instantiate(model_p)
    init_vars = model.init(rngs=jax.random.PRNGKey(1234))

    input_ids = jnp.zeros([batch_size, seq_len], dtype=jnp.int32)
    input_paddings = jnp.zeros([batch_size, seq_len], dtype=jnp.float32)

    def decode_fn(model, input_ids, input_paddings):
      return sample_decode.sample_decode(
          model,
          extend_step_fn,
          transform_decode_state_fn,
          None,
          model.next_token_sampler,
          input_ids,
          input_paddings,
          prefix_lengths=jnp.zeros([batch_size], dtype=jnp.int32),
          seq_len=seq_len,
          num_samples=num_samples,
          gumbel_prng_key=None,
          max_prefix_len=0,
          max_decode_steps=seq_len,
          fprop_for_prefix=True,
          temperature=jnp.ones([batch_size, num_samples], dtype=jnp.float32),
          per_example_top_p=jnp.ones([batch_size], dtype=jnp.float32),
          # Call the scan loop.
          early_exit=False,
      )

    mutables = [DECODE_CACHE]
    rngs = {'random': jax.random.PRNGKey(9382)}

    nn.apply(decode_fn, model, mutable=mutables)(
        init_vars, input_ids, input_paddings, rngs=rngs
    )

  def test_vanilla_sample_decode(self):
    batch_size = 6
    prefix_len = 1
    max_decoding_steps = 3
    seq_len = prefix_len + max_decoding_steps
    vocab_size = 5
    np.random.seed(987656)
    logits = jax.nn.log_softmax(
        10.0 * jnp.array(np.random.rand(seq_len, batch_size, vocab_size))
    )

    model_p = pax_fiddle.Config(
        TestModelWithLogits,
        batch_size=batch_size,
        num_samples=1,
        seq_len=seq_len,
        vocab_size=vocab_size,
        name='test_model',
        logits=logits,
    )

    def transform_decode_state_fn(mdl, transform_fn):
      del mdl
      del transform_fn

    model = instantiate(model_p)
    init_vars = model.init(rngs=jax.random.PRNGKey(1234))

    input_ids = jnp.zeros([batch_size, prefix_len], dtype=jnp.int32) + 2
    input_paddings = jnp.zeros([batch_size, prefix_len], dtype=jnp.float32)

    def fprop_fn(*args, **kwargs):
      pass

    def extend_step_fn(mdl, ids, segment_pos):
      logits = mdl.extend_step(ids, segment_pos=segment_pos)
      return logits

    def decode_fn(model, input_ids, input_paddings):
      return sample_decode.vanilla_sample_decode(
          model,
          fprop_fn,
          extend_step_fn,
          transform_decode_state_fn,
          model.next_token_sampler,
          input_ids,
          input_paddings,
          max_decode_steps=max_decoding_steps,
          gumbel_prng_key=None,
          # Call the scan loop.
          temperature=0.0001,
      )

    mutables = [DECODE_CACHE]
    rngs = {'random': jax.random.PRNGKey(9382)}

    result, _ = nn.apply(decode_fn, model, mutable=mutables)(
        init_vars, input_ids, input_paddings, rngs=rngs
    )
    expected_out_ids = np.array([
        [2, 2, 3, 3],
        [2, 4, 1, 2],
        [2, 4, 4, 1],
        [2, 4, 3, 4],
        [2, 4, 0, 1],
        [2, 0, 3, 0],
    ])
    self.assertArraysEqual(expected_out_ids, result.output_ids)


if __name__ == '__main__':
  absltest.main()
