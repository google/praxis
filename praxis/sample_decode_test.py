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

"""Unit tests for sample_decode."""

from absl.testing import absltest
import flax.linen as nn
import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import base_model
from praxis import py_utils
from praxis import sample_decode
from praxis import test_utils

NestedMap = py_utils.NestedMap
BaseHParams = base_layer.BaseLayer.HParams
WeightHPrams = base_layer.WeightHParams
instantiate = base_layer.instantiate

RANDOM = base_layer.RANDOM
DECODE_CACHE = base_layer.DECODE_CACHE
SUMMARIES = base_layer.SUMMARIES


class TestModel(base_model.BaseModel):

  class HParams(base_model.BaseModel.HParams):
    vocab_size: int = 0
    num_samples: int = 0
    seq_len: int = 0
    batch_size: int = 0

  def setup(self) -> None:
    p = self.hparams
    super().setup()
    logits_wp = base_layer.WeightHParams(
        shape=[p.seq_len, p.batch_size * p.num_samples, p.vocab_size])
    self.create_variable('logits', logits_wp)

  def __call__(self, *args, **kwargs):
    # A dummy __call__ function
    del args, kwargs

  # do something here
  def extend_step(self, ids, segment_pos):
    p = self.hparams
    assert segment_pos.shape == (p.batch_size *
                                 p.num_samples,), (segment_pos.shape,
                                                   (p.batch_size *
                                                    p.num_samples,))
    time_step = segment_pos[0] + 1
    logits_at_t = self.theta.logits[time_step, :, :]
    self.add_summary('logits', logits_at_t)
    self.add_summary('time_step', time_step)
    return logits_at_t


class SampleDecodeHelperTest(test_utils.TestCase):

  def test_split_batch_dim(self):
    x = jnp.array([[1, 2], [1, 2], [3, 4], [3, 4]], dtype=np.int32)
    self.assertArraysEqual(
        sample_decode.split_batch_dim(x, batch_dim=0, num_samples=2),
        np.array([[[1, 2], [1, 2]], [[3, 4], [3, 4]]], dtype=np.int32))

  def test_sample_from_topk_with_gumbel_noise(self):
    logits = jnp.array(
        [[0, 0, 1, 1, 0], [1, 0, 0, 0, 1], [0, 1, 1, 0, 0], [1, 1, 0, 0, 0]],
        dtype=jnp.float32)
    noise = jnp.array([[0.5, 0], [-0.5, 0], [-0.5, -1], [1, 0.5]],
                      dtype=jnp.float32)
    # logits + noise =
    # [[0, 0, 1.5, 1, 0], # argmax: 2
    #  [0.5, 0, 0, 0, 1]  # argmax: 4
    #  [0, 0.5, 0, 0, 0], # argmax: 1
    #  [2, 1.5, 0, 0, 0]] # argmax: 0
    new_ids = sample_decode.sample_from_topk_with_gumbel_noise(
        logits, noise, temperature=1.0, topk=2)
    self.assertArraysEqual(new_ids, np.array([2, 4, 1, 0], dtype=np.int32))

  def test_sample_from_topk_with_gumbel_noise_dyn_temp(self):
    logits = jnp.array([[[0, 0, 1, 1, 0], [1, 0, 0, 0, 1]],
                        [[0, 1, 1, 0, 0], [1, 1, 0, 0, 0]]],
                       dtype=jnp.float32)
    noise = jnp.array([[[0.5, 0], [-0.5, 0]], [[-0.5, -1], [1, 0.5]]],
                      dtype=jnp.float32)
    temperature = jnp.array([[0.1], [0.2]], dtype=jnp.float32)
    new_ids = sample_decode.sample_from_topk_with_gumbel_noise(
        logits, noise, temperature=temperature, topk=2)
    # logits + noise =
    # [[[0, 0, 1.5, 1, 0], # argmax: 2
    #  [0.5, 0, 0, 0, 1]]  # argmax: 4
    #  [[0, 0.5, 0, 0, 0], # argmax: 1
    #  [2, 1.5, 0, 0, 0]]] # argmax: 0
    self.assertArraysEqual(new_ids, np.array([[2, 4], [1, 0]], dtype=np.int32))

  def test_sample_from_topk(self):
    logits = jnp.array(
        [
            [0, 0, 1, 0, 0],  # argmax: 2
            [1, 0, 0, 0, 0],  # argmax: 0
            [0, 1, 0, 0, 0],  # argmax: 1
            [1, 0, 0, 0, 0],  # argmax: 0
        ],
        dtype=jnp.float32)
    new_ids = sample_decode.sample_from_topk(
        logits, jax.random.PRNGKey(seed=123), temperature=1.0, topk=2)
    # gumbel noise is relatively smaller compared to the logits value.
    self.assertArraysEqual(new_ids, np.array([2, 0, 1, 0], dtype=np.int32))

  def test_sample_from_topk_dyn_temp(self):
    logits = jnp.array(
        [
            [
                [0, 0, 1, 0, 0],  # argmax: 2
                [1, 0, 0, 0, 0],  # argmax: 0
            ],
            [
                [0, 1, 0, 0, 0],  # argmax: 1
                [1, 0, 0, 0, 0],  # argmax: 0
            ],
        ],
        dtype=jnp.float32)

    temperature = jnp.array([[0.1], [0.2]], dtype=jnp.float32)
    new_ids = sample_decode.sample_from_topk(
        logits, jax.random.PRNGKey(seed=123), temperature=temperature, topk=2)
    # gumbel noise is relatively smaller compared to the logits value.
    self.assertArraysEqual(new_ids, np.array([[2, 0], [1, 0]], dtype=np.int32))

  def test_sample_from_topk_distribution(self):
    logits = jnp.array([
        [0, 0.25, 0.2, 0.15, 0.4],
    ], dtype=jnp.float32)
    count = [0] * 5

    for i in range(100):
      new_ids = sample_decode.sample_from_topk(
          logits, jax.random.PRNGKey(seed=i), temperature=1.0, topk=4)
      count[new_ids[0]] += 1

    # Top4 value won't choose token 0.
    self.assertEqual(count[0], 0)
    # Token #1, #2, #3 should be chosen more than 10%.
    self.assertGreaterEqual(count[1], 10)
    self.assertGreaterEqual(count[2], 10)
    self.assertGreaterEqual(count[3], 10)
    # Token #4 should be chosen more than 25%.
    self.assertGreaterEqual(count[4], 25)

  def test_reorder_with_indices(self):
    indices = jnp.array([[0, 2, 1], [2, 0, 1]], dtype=jnp.int32)
    x = jnp.array(
        [[[0, 0, 0], [1, 1, 1], [2, 2, 2]], [[0, 0, 0], [1, 1, 1], [2, 2, 2]]],
        dtype=jnp.int32)

    reordered_x = sample_decode.reorder_with_indices(x, indices)

    self.assertArraysEqual(
        reordered_x,
        jnp.array([[[0, 0, 0], [2, 2, 2], [1, 1, 1]],
                   [[2, 2, 2], [0, 0, 0], [1, 1, 1]]],
                  dtype=jnp.int32))

  def test_sort_samples_by_scores(self):
    logprobs = jnp.array(
        [
            [
                [0.1, 0.1, 1.0],  #  sum is 0.2
                [0.3, 0.3, 0.3],  #  sum is 0.9
                [0.2, 0.2, 1.0]  #  sum is 0.4
            ],
            [
                [0.9, 0.9, 0.9],  # sum is 2.7
                [0.1, 1.0, 1.0],  # sum is 0.1
                [0.2, 0.2, 1.0]  # sum is 0.4
            ]
        ],
        dtype=jnp.float32)
    x = jnp.array(
        [[[0, 0, 0], [1, 1, 1], [2, 2, 2]], [[0, 0, 0], [1, 1, 1], [2, 2, 2]]],
        dtype=jnp.int32)
    result = NestedMap()
    result.logprobs = logprobs
    result.x = x

    result = sample_decode.sort_samples_by_scores(result)

    # Verify values in result are sorted and ordered at dimension 1.
    self.assertArraysEqual(
        result.x,
        jnp.array([[[1, 1, 1], [2, 2, 2], [0, 0, 0]],
                   [[0, 0, 0], [2, 2, 2], [1, 1, 1]]],
                  dtype=jnp.int32))
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
                ]
            ],
            dtype=jnp.float32))

  def test_right_align_prefix_ids(self):
    prefix_ids = jnp.array([[1, 2, 0], [1, 0, 0], [0, 1, 2]], dtype=jnp.int32)
    prefix_lengths = jnp.array([2, 1, 3], dtype=jnp.int32)

    (right_align_prefix_ids,
     right_align_prefix_paddings) = sample_decode.right_align_prefix_ids(
         prefix_ids, prefix_lengths, jnp.int32)

    self.assertArraysEqual(
        right_align_prefix_ids,
        jnp.array([[0, 1, 2], [0, 0, 1], [0, 1, 2]], dtype=jnp.int32))

    self.assertArraysEqual(
        right_align_prefix_paddings,
        jnp.array([[1, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=jnp.int32))

  def test_right_align_segment_position(self):
    lengths = jnp.array([5, 4, 6], dtype=jnp.int32)

    right_align_segment_pos = sample_decode.right_align_segment_position(
        lengths, max_length=6)

    self.assertArraysEqual(
        right_align_segment_pos,
        jnp.array([[0, 0, 1, 2, 3, 4], [0, 0, 0, 1, 2, 3], [0, 1, 2, 3, 4, 5]],
                  dtype=jnp.int32))

  def test_top_p_mask_logits(self):
    logits = jnp.array([[1.0, 1.0, 0.5, -1e6]])
    masked = sample_decode.top_p_mask_logits(logits, p=0.99)
    self.assertAllClose(logits[:, :-1], masked[:, :-1])
    self.assertLess(masked[0, -1], 1e-10)

  def test_sample_decode(self):
    batch_size = 1
    num_samples = 2
    seq_len = 3
    vocab_size = 4
    model_p = TestModel.HParams(
        name='test_model',
        batch_size=batch_size,
        num_samples=num_samples,
        seq_len=seq_len,
        vocab_size=vocab_size)

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
      return sample_decode.sample_decode(
          model,
          extend_step_fn,
          transform_decode_state_fn,
          None,
          input_ids,
          input_paddings,
          prefix_lengths=jnp.zeros([batch_size], dtype=jnp.int32),
          seq_len=seq_len,
          num_samples=num_samples,
          k=0,
          max_prefix_len=0,
          max_decode_steps=seq_len,
          fprop_for_prefix=True,
          # Call the scan loop.
          early_exit=False)

    mutables = [SUMMARIES, DECODE_CACHE]
    rngs = {'random': jax.random.PRNGKey(9382)}

    # test that we can fetch arbitrary summary out.
    _, updated_vars = nn.apply(
        decode_fn, model, mutable=mutables)(
            init_vars, input_ids, input_paddings, rngs=rngs)
    logits_summary = updated_vars['summaries']['logits_scalar']
    time_step_summary = updated_vars['summaries']['time_step_scalar']
    print('logits_var', logits_var)
    print('logits_summary', logits_summary)
    print('time_step_summary', time_step_summary)
    self.assertAllClose(logits_var, logits_summary)


if __name__ == '__main__':
  absltest.main()
