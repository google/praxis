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

from absl.testing import absltest
import jax
from praxis import base_layer
from praxis import pax_fiddle
from praxis import test_utils
from praxis.layers.video import quantizer
from praxis.layers.video import vqvae


class VQVAETest(test_utils.TestCase):

  def test_auto_encoder(self):
    prng_key = jax.random.PRNGKey(seed=123)
    x = jax.random.normal(prng_key, (1, 5, 16, 16, 3))
    config = pax_fiddle.Config(
        vqvae.VQVaeModel,
        name="vqvae",
        encoder_tpl=pax_fiddle.Config(
            vqvae.Encoder,
            name="encoder",
            filters=128,
            input_dim=3,
            embedding_dim=8,
            num_res_blocks=4,
            temporal_downsample=(False, True, True),
            channel_multipliers=(1, 2, 2, 4),
        ),
        decoder_tpl=pax_fiddle.Config(
            vqvae.Decoder,
            name="decoder",
            filters=128,
            embedding_dim=8,
            output_dim=3,
            num_res_blocks=4,
            temporal_downsample=(False, True, True),
            channel_multipliers=(1, 2, 2, 4),
        ),
        quantizer_tpl=pax_fiddle.Config(
            quantizer.LookupFreeQuantizer,
            name="quantizer",
            embedding_dim=8,
        ),
    )
    vqvae_model = base_layer.instantiate(config)
    pax_vars = vqvae_model.init(prng_key, x)
    value = vqvae_model.apply(pax_vars, x)
    self.assertIsInstance(value, tuple)
    self.assertLen(value, 2)
    self.assertEqual(value[0].shape, x.shape)
    self.assertAllClose(value[1].quantizer_loss, -0.09399976)
    self.assertAllClose(value[1].e_latent_loss, 0.07907465)
    self.assertAllClose(value[1].entropy_loss, -0.17307441)

  def test_discriminator(self):
    prng_key = jax.random.PRNGKey(seed=123)
    x = jax.random.normal(prng_key, (1, 5, 16, 16, 3))
    config = pax_fiddle.Config(
        vqvae.Discriminator,
        name="discriminator",
        filters=128,
        num_frames=5,
        image_height=16,
        image_width=16,
        input_dim=3,
        blur_filter_size=3,
        channel_multipliers=(2, 4, 4, 4),
    )
    discriminator = base_layer.instantiate(config)
    pax_vars = discriminator.init(prng_key, x)
    value = discriminator.apply(pax_vars, x)
    self.assertAllClose(value, -4.26183e-05)


if __name__ == "__main__":
  absltest.main()
