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

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from praxis import base_layer
from praxis import pax_fiddle
from praxis import py_utils
from praxis import test_utils
from praxis.layers.video import losses
from praxis.layers.video import vqvae


class LossesTest(test_utils.TestCase):

  def test_r1_gradient_penalty(self):
    prng_key = jax.random.PRNGKey(seed=123)
    x = jax.random.normal(prng_key, (2, 5, 16, 16, 3))
    # Create a pax layer and get the output from the random input.
    p = pax_fiddle.Config(
        vqvae.Discriminator,
        name='magvit',
        num_frames=5,
        image_height=16,
        image_width=16,
        filters=32,
        channel_multipliers=(2, 4),
    )
    context_p = base_layer.JaxContext.HParams(do_eval=False)
    with base_layer.JaxContext.new_context(hparams=context_p):
      pax_layer = base_layer.instantiate(p)
      pax_vars = pax_layer.init(prng_key, x)
      logit_fn = functools.partial(pax_layer.apply, pax_vars)
      logits, penalty = losses.r1_gradient_penalty(x, logit_fn)
      self.assertEqual(logits.shape, (2, 1))
      self.assertEqual(penalty.shape, ())

  @parameterized.parameters(True, False)
  def test_vqgan_loss(self, do_eval):
    batch_size, num_frames, height, width, channels = 2, 5, 128, 128, 3
    video_shape = (batch_size, num_frames, height, width, channels)
    np.random.seed(12345)
    input_batch = py_utils.NestedMap(
        video=np.random.randint(0, 255, size=video_shape)
    )
    predictions = py_utils.NestedMap(
        reconstructed=np.random.normal(size=video_shape),
        logits_real=np.random.normal(size=(batch_size, 1)),
        logits_fake=np.random.normal(size=(batch_size, 1)),
        quantizer_loss=np.random.normal(size=[]),
        r1_gradient_penalty=np.random.normal(size=[]),
    )

    loss_p = pax_fiddle.Config(
        losses.VQGANLoss,
        name='loss',
    )
    loss_layer = loss_p.Instantiate()
    prng_key = jax.random.PRNGKey(seed=123)
    context_p = base_layer.JaxContext.HParams(do_eval=do_eval)
    with base_layer.JaxContext.new_context(hparams=context_p):
      init_vars = loss_layer.init(prng_key, predictions, input_batch)
      loss_dict, updated_vars = loss_layer.apply(
          init_vars, predictions, input_batch, mutable=base_layer.NON_TRAINABLE
      )
      for loss in loss_dict.values():
        self.assertEqual((), loss.shape)
    self.assertNotEqual(
        updated_vars[base_layer.NON_TRAINABLE]['ema_fake_pred'], 0.0
    )
    self.assertNotEqual(
        updated_vars[base_layer.NON_TRAINABLE]['ema_real_pred'], 0.0
    )


if __name__ == '__main__':
  absltest.main()
