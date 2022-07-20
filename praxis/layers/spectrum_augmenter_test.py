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

"""Tests for spectrum_augmenter."""

from absl.testing import absltest
import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import test_utils
from praxis.layers import spectrum_augmenter

instantiate = base_layer.instantiate
to_np = test_utils.to_np


class SpectrumAugmenterTest(test_utils.TestCase):

  def testSpectrumAugmenterWithTimeMask(self):
    batch_size = 5
    inputs = jnp.ones([batch_size, 20, 2], dtype=jnp.float32)
    paddings = []
    for i in range(batch_size):
      paddings.append(
          jnp.concatenate([jnp.zeros([1, i + 12]),
                           jnp.ones([1, 8 - i])],
                          axis=1))
    paddings = jnp.concatenate(paddings, axis=0)

    p = spectrum_augmenter.SpectrumAugmenter.HParams(
        name='specAug_layers',
        freq_mask_max_bins=0,
        time_mask_max_frames=5,
        time_mask_count=2,
        time_mask_max_ratio=1.)
    specaug_layer = instantiate(p)
    expected_output = np.array(
        [[[1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.],
          [1., 1.], [0., 0.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.],
          [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.], [0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.],
          [0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.], [1., 1.],
          [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.], [1., 1.], [0., 0.], [0., 0.], [0., 0.], [0., 0.],
          [0., 0.], [0., 0.], [0., 0.], [0., 0.], [1., 1.], [1., 1.], [1., 1.],
          [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.]],
         [[1., 1.], [0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.],
          [0., 0.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.],
          [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.], [1., 1.], [0., 0.], [0., 0.], [0., 0.], [0., 0.],
          [0., 0.], [0., 0.], [0., 0.], [0., 0.], [1., 1.], [1., 1.], [1., 1.],
          [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.]]])
    context_p = base_layer.JaxContext.HParams(do_eval=False)
    prng_key = jax.random.PRNGKey(seed=23456)
    prng_key, compute_key = jax.random.split(prng_key)
    with base_layer.JaxContext.new_context(hparams=context_p):
      initial_vars = specaug_layer.init(
          {
              'params': prng_key,
              'random': compute_key
          }, inputs, paddings)
      actual_layer_output, _ = specaug_layer.apply(
          initial_vars, inputs, paddings, rngs={'random': compute_key})
    self.assertAllClose(actual_layer_output, expected_output)

  def testSpectrumAugmenterWithFrequencyMask(self):
    inputs = jnp.ones([3, 5, 10], dtype=jnp.float32)
    paddings = jnp.zeros([3, 5])
    p = spectrum_augmenter.SpectrumAugmenter.HParams(
        name='specAug_layers',
        freq_mask_max_bins=6,
        freq_mask_count=2,
        time_mask_max_frames=0)
    specaug_layer = instantiate(p)
    expected_output = np.array([[[1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
                                 [1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
                                 [1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
                                 [1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
                                 [1., 1., 1., 0., 0., 0., 0., 0., 0., 0.]],
                                [[0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
                                 [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
                                 [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
                                 [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
                                 [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.]],
                                [[1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
                                 [1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
                                 [1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
                                 [1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
                                 [1., 1., 1., 0., 0., 0., 0., 0., 0., 0.]]])
    context_p = base_layer.JaxContext.HParams(do_eval=False)
    prng_key = jax.random.PRNGKey(seed=34567)
    prng_key, compute_key = jax.random.split(prng_key)
    with base_layer.JaxContext.new_context(hparams=context_p):
      initial_vars = specaug_layer.init(
          {
              'params': prng_key,
              'random': compute_key
          }, inputs, paddings)
      actual_layer_output, _ = specaug_layer.apply(
          initial_vars, inputs, paddings, rngs={'random': compute_key})
    self.assertAllClose(actual_layer_output, expected_output)

  def testSpectrumEvalMode(self):
    inputs = jnp.ones([3, 5, 10], dtype=jnp.float32)
    paddings = jnp.zeros([3, 5])
    p = spectrum_augmenter.SpectrumAugmenter.HParams(
        name='specAug_layers',
        freq_mask_max_bins=6,
        freq_mask_count=2,
        time_mask_max_frames=0)
    specaug_layer = instantiate(p)
    context_p = base_layer.JaxContext.HParams(do_eval=True)
    prng_key = jax.random.PRNGKey(seed=34567)
    prng_key, compute_key = jax.random.split(prng_key)
    with base_layer.JaxContext.new_context(hparams=context_p):
      initial_vars = specaug_layer.init(
          {
              'params': prng_key,
              'random': compute_key
          }, inputs, paddings)
      actual_layer_output, _ = specaug_layer.apply(
          initial_vars, inputs, paddings, rngs={'random': compute_key})
    self.assertAllClose(actual_layer_output, inputs)


if __name__ == '__main__':
  absltest.main()
