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

"""Unit tests for token_samplers."""
from absl.testing import absltest
import jax
from jax import numpy as jnp
import numpy as np
from praxis import py_utils
from praxis import test_utils
from praxis import token_samplers


class TokenSamplersTest(test_utils.TestCase):

  def test_sample_from_top_k(self):
    logits = jnp.array(
        [
            [0, 0, 1, 0, 0],  # argmax: 2
            [1, 0, 0, 0, 0],  # argmax: 0
            [0, 1, 0, 0, 0],  # argmax: 1
            [1, 0, 0, 0, 0],  # argmax: 0
        ],
        dtype=jnp.float32,
    )
    new_ids, _ = token_samplers.sample_from_top_k_and_top_p(
        logits, jax.random.PRNGKey(seed=123), temperature=1.0, top_k=2
    )
    # gumbel noise is relatively smaller compared to the logits value.
    self.assertArraysEqual(new_ids, np.array([2, 0, 1, 0], dtype=np.int32))

  def test_sample_from_top_k_with_topk_logprobs(self):
    logits = jnp.array(
        [
            [0, 0, 1, 0, 0],  # argmax: 2
            [1, 0, 0, 0, 0],  # argmax: 0
            [0, 1, 0, 0, 0],  # argmax: 1
            [1, 0, 0, 0, 0],  # argmax: 0
        ],
        dtype=jnp.float32,
    )
    new_ids, top_k_logprobs = token_samplers.sample_from_top_k_and_top_p(
        logits,
        jax.random.PRNGKey(seed=123),
        temperature=1.0,
        top_k=2,
    )
    expected_logprobs = jax.nn.log_softmax(jax.lax.top_k(logits, 2)[0])
    expected_logprobs = expected_logprobs[jnp.arange(4), 0]
    # gumbel noise is relatively smaller compared to the logits value.
    self.assertArraysEqual(new_ids, np.array([2, 0, 1, 0], dtype=np.int32))
    self.assertAllClose(top_k_logprobs, expected_logprobs)

  def test_sample_from_top_k_and_top_p_scalar(self):
    logits = jnp.array(
        [
            [0.1, 0.7, 0.2, 0, 0],
            [0.3, 0.1, 0, 0.2, 0.5],
            [0.2, 0.6, 0.1, 0, 0.1],
            [0.5, 0, 0.5, 0, 0],
        ],
        dtype=jnp.float32,
    )
    new_ids, _ = token_samplers.sample_from_top_k_and_top_p(
        logits,
        jax.random.PRNGKey(seed=123),
        temperature=1.0,
        top_k=2,
        top_p=0.5,
    )
    # gumbel noise is relatively smaller compared to the logits value.
    self.assertArraysEqual(new_ids, np.array([1, 4, 1, 0], dtype=np.int32))

  def test_sample_from_top_k_and_top_p_global_normalize(self):
    logits = jnp.array(
        [
            [0.1, 0.7, 0.2, 0, 0],
            [0.1, 0, 0, 0.4, 0.5],
            [0.2, 0.6, 0.1, 0, 0.1],
            [0.5, 0, 0.5, 0, 0],
        ],
        dtype=jnp.float32,
    )
    new_ids, _ = token_samplers.sample_from_top_k_and_top_p(
        logits,
        jax.random.PRNGKey(seed=123),
        temperature=1.0,
        top_k=3,
        top_p=0.9,
        global_normalize=True,
    )
    self.assertArraysEqual(new_ids, np.array([0, 3, 1, 0], dtype=np.int32))

  def test_sample_from_top_k_and_top_p_scalar_false_fn(self):
    logits = jnp.array(
        [
            [0.1, 0.7, 0.2, 0, 0],
            [0.3, 0.1, 0, 0.2, 0.5],
            [0.2, 0.6, 0.1, 0, 0.1],
            [0.5, 0, 0.5, 0, 0],
        ],
        dtype=jnp.float32,
    )
    new_ids, _ = token_samplers.sample_from_top_k_and_top_p(
        logits,
        jax.random.PRNGKey(seed=123),
        temperature=1.0,
        top_k=2,
        top_p=1.0,
    )
    # gumbel noise is relatively smaller compared to the logits value.
    self.assertArraysEqual(new_ids, np.array([1, 4, 1, 0], dtype=np.int32))

  def test_sample_from_top_k_and_top_p_tensor(self):
    logits = jnp.array(
        [
            [0.1, 0.7, 0.2, 0, 0],
            [0.2, 0.2, 0.2, 0.2, 0.2],
        ],
        dtype=jnp.float32,
    )
    new_ids, _ = token_samplers.sample_from_top_k_and_top_p(
        logits,
        jax.random.PRNGKey(seed=123),
        temperature=1.0,
        top_k=4,
        top_p=jnp.array([[0.75], [0.3]], dtype=jnp.float32),
    )
    # gumbel noise is relatively smaller compared to the logits value.
    self.assertArraysEqual(new_ids, np.array([1, 0], dtype=np.int32))

  def test_sample_from_top_k_and_top_p_tensor_false_fn(self):
    logits = jnp.array(
        [
            [0.1, 0.7, 0.2, 0, 0],
            [0.3, 0.1, 0, 0.2, 0.5],
            [0.2, 0.6, 0.1, 0, 0.1],
            [0.5, 0, 0.5, 0, 0],
        ],
        dtype=jnp.float32,
    )
    new_ids, _ = token_samplers.sample_from_top_k_and_top_p(
        logits,
        jax.random.PRNGKey(seed=123),
        temperature=1.0,
        top_k=2,
        top_p=jnp.array([[1.0], [1.0], [1.0], [1.0]], dtype=jnp.float32),
    )
    # gumbel noise is relatively smaller compared to the logits value.
    self.assertArraysEqual(new_ids, np.array([1, 4, 1, 0], dtype=np.int32))

  def test_apply_top_k(self):
    """Tests apply_top_k_and_top_p helper for top_k sampling only."""
    logits = jnp.array(
        [
            [0.1, 0.7, 0.2, 0, 0],
            [0.3, 0.1, 0, 0.2, 0.5],
            [0.2, 0.6, 0.1, 0, 0.1],
            [0.5, 0, 0.5, 0, 0],
        ],
        dtype=jnp.float32,
    )
    top_p_logits, top_k_logprobs, top_k_indices = (
        token_samplers._apply_top_k_and_top_p(logits, top_k=2)
    )
    expected_top_k_logits = np.array(
        [[0.7, 0.2], [0.5, 0.3], [0.6, 0.2], [0.5, 0.5]], dtype=np.float32
    )
    self.assertArraysEqual(
        top_k_logprobs, jax.nn.log_softmax(expected_top_k_logits)
    )
    self.assertArraysEqual(
        top_k_indices,
        np.array([[1, 2], [4, 0], [1, 0], [0, 2]], dtype=np.int32),
    )
    self.assertArraysEqual(top_p_logits, expected_top_k_logits)

  def test_apply_top_k_and_top_p(self):
    logits = jnp.array(
        [
            [0.1, 0.7, 0.2, 0, 0],
            [0.3, 0.1, 0, 0.2, 0.5],
            [0.2, 0.6, 0.1, 0, 0.1],
            [0.5, 0, 0.5, 0, 0],
        ],
        dtype=jnp.float32,
    )
    top_p_logits, top_k_logprobs, top_k_indices = (
        token_samplers._apply_top_k_and_top_p(logits, top_k=2, top_p=0.5)
    )
    self.assertArraysEqual(
        top_k_logprobs,
        jax.nn.log_softmax(
            np.array(
                [[0.7, 0.2], [0.5, 0.3], [0.6, 0.2], [0.5, 0.5]],
                dtype=np.float32,
            )
        ),
    )
    self.assertArraysEqual(
        top_k_indices,
        np.array([[1, 2], [4, 0], [1, 0], [0, 2]], dtype=np.int32),
    )
    large_neg = py_utils.get_large_negative_number(np.float32)
    self.assertAllClose(
        top_p_logits,
        np.array(
            [[0.7, large_neg], [0.5, large_neg], [0.6, large_neg], [0.5, 0.5]],
            dtype=np.float32,
        ),
    )

  def test_sample_from_top_k_dyn_temp(self):
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
        dtype=jnp.float32,
    )

    temperature = jnp.array([[0.1], [0.2]], dtype=jnp.float32)
    new_ids, _ = token_samplers.sample_from_top_k_and_top_p(
        logits, jax.random.PRNGKey(seed=123), temperature=temperature, top_k=2
    )
    # gumbel noise is relatively smaller compared to the logits value.
    self.assertArraysEqual(new_ids, np.array([[2, 0], [1, 0]], dtype=np.int32))

  def test_sample_from_top_k_distribution(self):
    logits = jnp.array(
        [
            [0, 0.25, 0.2, 0.15, 0.4],
        ],
        dtype=jnp.float32,
    )
    count = [0] * 5

    for i in range(100):
      new_ids, _ = token_samplers.sample_from_top_k_and_top_p(
          logits, jax.random.PRNGKey(seed=i), temperature=1.0, top_k=4
      )
      count[new_ids[0]] += 1

    # Top4 value won't choose token 0.
    self.assertEqual(count[0], 0)
    # Token #1, #2, #3 should be chosen more than 10%.
    self.assertGreaterEqual(count[1], 10)
    self.assertGreaterEqual(count[2], 10)
    self.assertGreaterEqual(count[3], 10)
    # Token #4 should be chosen more than 25%.
    self.assertGreaterEqual(count[4], 25)

  def test_get_argmax_ids(self):
    top_k_argmax_ids = jnp.array([2, 1], dtype=jnp.int32)
    top_k_indices = jnp.array(
        [[12, 200, 300, 9], [500, 608, 9000, 7]], dtype=jnp.int32
    )
    argmax_ids = token_samplers._get_argmax_ids(top_k_argmax_ids, top_k_indices)
    self.assertArraysEqual(argmax_ids, jnp.array([300, 608], dtype=jnp.int32))

  def test_top_p_mask_logits(self):
    logits = jnp.array([[1.0, 1.0, 1.0, -1e6]])
    masked = token_samplers.top_p_mask_logits(logits, p=0.99)
    self.assertAllClose(logits[:, :-1], masked[:, :-1])
    self.assertLess(masked[0, -1], 1e-10)

  def test_top_p_mask_logits_logits_unsorted(self):
    logits = jnp.array([[0.2, 2.0, 0.5, 1.5, 1.0]])
    # softmax of logits:
    # [0.06995774 0.42321967 0.09443307 0.2566957  0.15569381]
    masked = token_samplers.top_p_mask_logits(logits, p=0.4)
    ninf = -2.381976e38
    self.assertAllClose(masked[0], np.array([ninf, 2.0, ninf, ninf, ninf]))

    masked = token_samplers.top_p_mask_logits(logits, p=0.6)
    self.assertAllClose(masked[0], np.array([ninf, 2.0, ninf, 1.5, ninf]))

    masked = token_samplers.top_p_mask_logits(logits, p=0.8)
    self.assertAllClose(masked[0], np.array([ninf, 2.0, ninf, 1.5, 1.0]))

    masked = token_samplers.top_p_mask_logits(logits, p=0.9)
    self.assertAllClose(masked[0], np.array([ninf, 2.0, 0.5, 1.5, 1.0]))

  def test_top_p_mask_logits_logits_sorted(self):
    logits = jnp.array([[2.0, 1.5, 1.0, 0.5, 0.2]])
    # softmax of logits:
    # [0.42321967 0.2566957  0.15569381 0.09443307 0.06995774]
    masked = token_samplers.top_p_mask_logits(
        logits, p=0.4, logits_sorted_in_descending_order=True
    )
    ninf = -2.381976e38
    self.assertAllClose(masked[0], np.array([2.0, ninf, ninf, ninf, ninf]))

    masked = token_samplers.top_p_mask_logits(
        logits, p=0.6, logits_sorted_in_descending_order=True
    )
    self.assertAllClose(masked[0], np.array([2.0, 1.5, ninf, ninf, ninf]))

    masked = token_samplers.top_p_mask_logits(
        logits, p=0.8, logits_sorted_in_descending_order=True
    )
    self.assertAllClose(masked[0], np.array([2.0, 1.5, 1.0, ninf, ninf]))

    masked = token_samplers.top_p_mask_logits(
        logits, p=0.9, logits_sorted_in_descending_order=True
    )
    self.assertAllClose(masked[0], np.array([2.0, 1.5, 1.0, 0.5, ninf]))

  def test_epsilon_mask_logits(self):
    logits = jnp.array([[1.0, 1.0, 0.5, -1e6]])
    masked = token_samplers.epsilon_mask_logits(logits, epsilon=0.1)
    self.assertAllClose(logits[:, :-1], masked[:, :-1])
    self.assertLess(masked[0, -1], 1e-10)


if __name__ == '__main__':
  absltest.main()
