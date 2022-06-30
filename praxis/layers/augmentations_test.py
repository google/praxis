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

"""Tests for Praxis augmentation layers."""

from absl import logging
from absl.testing import absltest
import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import test_utils
from praxis.layers import augmentations

instantiate = base_layer.instantiate
PARAMS = base_layer.PARAMS
RANDOM = base_layer.RANDOM
to_np = test_utils.to_np


class AugmentationsTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  def testMaskedLmDataAugmenterSmall(self):
    p = augmentations.MaskedLmDataAugmenter.HParams(
        name='mlm', vocab_size=32000, mask_token_id=0)
    layer = instantiate(p)
    inputs = jnp.arange(10, dtype=jnp.int32)
    paddings = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 1.0, 1.0], dtype=jnp.float32)
    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=12345)
      prng_key, compute_key = jax.random.split(prng_key)
      initial_vars = layer.init({
          PARAMS: prng_key,
          RANDOM: compute_key
      }, inputs, paddings)
      augmented_ids, augmented_pos = layer.apply(
          initial_vars, inputs, paddings, rngs={RANDOM: compute_key})
    logging.info('augmented_ids: %s', augmented_ids)
    logging.info('augmented_pos: %s', augmented_pos)
    expected_ids = np.array([0, 1, 0, 3, 4, 5, 0, 7, 8, 9])
    expected_pos = np.array([0., 0., 1., 0., 0., 0., 1., 0., 0., 0.])
    self.assertAllClose(to_np(expected_ids), to_np(augmented_ids))
    self.assertAllClose(to_np(expected_pos), to_np(augmented_pos))

  def testMaskedLmDataAugmenterLarge(self):
    p = augmentations.MaskedLmDataAugmenter.HParams(
        name='mlm', vocab_size=32000, mask_token_id=0)
    layer = instantiate(p)
    inputs = jnp.arange(100, dtype=jnp.int32)
    paddings = jnp.zeros_like(inputs).astype(jnp.float32)
    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      prng_key, compute_key = jax.random.split(prng_key)
      initial_vars = layer.init({
          PARAMS: prng_key,
          RANDOM: compute_key
      }, inputs, paddings)
      augmented_ids, augmented_pos = layer.apply(
          initial_vars, inputs, paddings, rngs={RANDOM: compute_key})
    logging.info('augmented_ids: %s', np.array_repr(augmented_ids))
    logging.info('augmented_pos: %s', np.array_repr(augmented_pos))
    np.set_printoptions(threshold=np.inf)
    expected_ids = np.array([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 0, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
        38, 39, 40, 41, 42, 43, 26592, 45, 46, 11329, 48, 49, 50, 51, 52, 53,
        54, 55, 56, 57, 58, 59, 60, 0, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
        19237, 73, 0, 0, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
        90, 91, 92, 93, 94, 95, 0, 97, 98, 99
    ])
    expected_pos = np.array([
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
        1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 1., 0., 0., 0.
    ])
    self.assertAllClose(to_np(expected_ids), to_np(augmented_ids))
    self.assertAllClose(to_np(expected_pos), to_np(augmented_pos))


if __name__ == '__main__':
  absltest.main()
