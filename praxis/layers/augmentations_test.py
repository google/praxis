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

"""Tests for Praxis augmentation layers."""

from absl import logging
from praxis import pax_fiddle
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
    p = pax_fiddle.Config(
        augmentations.MaskedLmDataAugmenter,
        name='mlm',
        vocab_size=32000,
        mask_token_id=0,
    )
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
    p = pax_fiddle.Config(
        augmentations.MaskedLmDataAugmenter,
        name='mlm',
        vocab_size=32000,
        mask_token_id=0,
    )
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

  def test_shifting(self):
    p = pax_fiddle.Config(
        augmentations.TemporalShifting,
        name='shifting',
        shift_range_ms=13.3,
        sample_rate=1000.0,
    )
    layer = instantiate(p)
    batch_size = 2
    audio_len = 20
    channels = 2
    shape = [batch_size, audio_len, channels]
    # Add 1 to distinguish the shifted signal from the added zeros.
    features = jnp.arange(np.prod(shape)).reshape(shape) + 1
    # Use 2 to distinguish the default np.pad value from the signal paddings.
    paddings = 2.0 * jnp.ones(shape[:-1])

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(12345)
      prng_key, init_key = jax.random.split(prng_key)
      prng_key, shift_key1, shift_key2 = jax.random.split(prng_key, 3)
      initial_vars = layer.init({
          'random': shift_key1,
          'params': init_key
      }, features, paddings)
      logging.info('initial_vars: %s', initial_vars)
      output1, out_paddings1 = layer.apply(
          initial_vars, features, paddings, rngs={'random': shift_key1})
      output2, out_paddings2 = layer.apply(
          initial_vars, features, paddings, rngs={'random': shift_key2})

    logging.info('output1:\n%s', output1)
    logging.info('output2:\n%s', output2)

    # Check that the outputs are shifted by the expected ammount along the batch
    # dimension. Also checks that each channel is shifted by the same amount.
    self.assertArraysEqual(output1[0, 2:], features[0, :-2])
    self.assertArraysEqual(output1[1, :-4], features[1, 4:])

    self.assertArraysEqual(output2[0, :-4], features[0, 4:])
    self.assertArraysEqual(output2[1, 8:], features[1, :-8])

    # Check that the zeros added to the signal are padded and that the shifted
    # signal is unpadded (which artificially corresponds to a padding value of
    # 2.0).
    self.assertArraysEqual(output1[..., 0] == 0.0, out_paddings1 == 1.0)
    self.assertArraysEqual(output1[..., 0] != 0.0, out_paddings1 == 2.0)

    self.assertArraysEqual(output2[..., 0] == 0.0, out_paddings2 == 1.0)
    self.assertArraysEqual(output2[..., 0] != 0.0, out_paddings2 == 2.0)


if __name__ == '__main__':
  absltest.main()
