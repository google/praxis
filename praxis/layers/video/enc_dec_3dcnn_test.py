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

from absl import logging
from absl.testing import absltest
import jax
from jax import numpy as jnp
from praxis import base_layer
from praxis import pax_fiddle
from praxis import test_utils
from praxis.layers.video import enc_dec_3dcnn


class EndcDec3dcnnTest(test_utils.TestCase):

  def test_depth_to_space(self):
    x = jnp.ones((2, 5, 3, 3, 32))
    y = enc_dec_3dcnn.depth_to_space(x, 2, 4)
    self.assertEqual(y.shape, (2, 10, 6, 6, 4))

  def test_gan_res_block(self):
    prng_key, _ = jax.random.split(jax.random.PRNGKey(1234))
    pax_x = jax.random.normal(prng_key, (1, 17, 256, 257, 32))
    res_block_p = pax_fiddle.Config(
        enc_dec_3dcnn.DiscriminatorResBlock,
        name='res_block',
        input_dim=32,
        output_dim=32,
    )
    pax_layer = base_layer.instantiate(res_block_p)
    init_vars = pax_layer.init(prng_key, pax_x)
    logging.info(
        'init_vars: %s', jax.tree_util.tree_map(lambda x: x.shape, init_vars)
    )
    pax_y = pax_layer.apply(init_vars, pax_x)
    self.assertEqual(pax_y.shape, (1, 9, 128, 129, 32))

  def test_res_block(self):
    prng_key, _ = jax.random.split(jax.random.PRNGKey(1234))
    pax_x = jax.random.normal(prng_key, (1, 5, 3, 3, 32))
    res_block_p = pax_fiddle.Config(
        enc_dec_3dcnn.ResBlock,
        name='res_block',
        input_dim=32,
        output_dim=64,
        use_conv_shortcut=True,
    )
    pax_layer = base_layer.instantiate(res_block_p)
    init_vars = pax_layer.init(prng_key, pax_x)
    logging.info(
        'init_vars: %s', jax.tree_util.tree_map(lambda x: x.shape, init_vars)
    )
    pax_y = pax_layer.apply(init_vars, pax_x)
    self.assertEqual(pax_y.shape, pax_x.shape[:-1] + (64,))


if __name__ == '__main__':
  absltest.main()
