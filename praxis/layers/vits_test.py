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

"""Tests for Praxis vit model."""

from typing import Tuple
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import py_utils
from praxis import test_utils
from praxis.layers import activations
from praxis.layers import transformers
from praxis.layers import vits

PARAMS = base_layer.PARAMS
RANDOM = base_layer.RANDOM
instantiate = base_layer.instantiate


class VitTest(test_utils.TestCase, parameterized.TestCase):

  def _exp_params(self):
    return py_utils.NestedMap(
        batch_size=3,
        patch_size=4,
        hidden_dim=24,
        num_tokens=16,
        num_heads=4,
        num_xformer_layers=4,
        atten_dropout_prob=0.1,
        stochastic_depth_dropout_prob=0.1,
    )

  def _vit_entry_layers(self, exp_params):
    p_entry = vits.VitEntryLayers.HParams(
        name='entry',
        pos_embed_shapes=exp_params.pos_embed_shapes,
        patch_size=exp_params.patch_size,
        dim_per_patch=exp_params.hidden_dim,
        image_channels=3,
        pos_emb_dropout_prob=0.1,
    )
    return p_entry

  def _vit_transformer_layers(self, exp_params):
    p_stacked_tfm = transformers.StackedTransformer.HParams(
        model_dims=exp_params.hidden_dim,
        hidden_dims=exp_params.hidden_dim * 4,
        num_heads=exp_params.num_heads,
        mask_self_attention=False,
        cross_attention=False,
        packed_input=False,
        num_layers=exp_params.num_xformer_layers,
        dropout_prob=exp_params.atten_dropout_prob,
        residual_droppath_prob=exp_params.stochastic_depth_dropout_prob,
    )

    p_tfm = p_stacked_tfm.transformer_layer_params_tpl
    p_tfm.norm_policy = 'pre'
    p_tfm.tr_fflayer_tpl.activation_tpl = activations.GELU.HParams()
    p_tfm.tr_atten_tpl.atten_logit_cap = 0.0
    p_tfm.tr_atten_tpl.internal_enable_per_dim_scale = False

    return p_stacked_tfm

  def _vit_exit_layers(self, exp_params):
    p_exit = vits.VitExitLayers.HParams(
        name='exit',
        hidden_dim=exp_params.hidden_dim,
        output_dim=exp_params.hidden_dim,
        output_dropout_prob=0.1)
    return p_exit

  @parameterized.named_parameters(
      ('square_image_square_embeddings_no_interp', (32, 32), (8, 8)),
      ('square_image_square_embeddings_interp', (64, 64), (8, 8)),
      ('rec_image_rec_embeddings_interp', (16, 24), (2, 3)))
  def test_vit_entry_layers(self, image_sizes, pos_embed_shapes):
    exp_params = self._exp_params()
    exp_params.pos_embed_shapes = pos_embed_shapes
    p_entry = self._vit_entry_layers(exp_params)
    entry = instantiate(p_entry)

    inputs_np = np.random.normal(
        size=[exp_params.batch_size, image_sizes[0], image_sizes[1], 3])
    inputs = jnp.asarray(inputs_np)

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      prng_key, subkey = jax.random.split(prng_key)
      initial_vars = entry.init({PARAMS: prng_key, RANDOM: subkey}, inputs)

      features = entry.apply(initial_vars, inputs, rngs={RANDOM: subkey})

    row_patch_count = image_sizes[0] // exp_params.patch_size
    col_patch_count = image_sizes[1] // exp_params.patch_size

    self.assertEqual(features.shape,
                     (exp_params.batch_size, row_patch_count * col_patch_count,
                      exp_params.hidden_dim))

  def test_vit_exit_layers(self):
    exp_params = self._exp_params()
    p_exit = self._vit_exit_layers(exp_params)
    exit_module = instantiate(p_exit)

    inputs_np = np.random.normal(size=[
        exp_params.batch_size, exp_params.num_tokens, exp_params.hidden_dim
    ])
    inputs = jnp.asarray(inputs_np)

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      prng_key, subkey = jax.random.split(prng_key)
      initial_vars = exit_module.init({
          PARAMS: prng_key,
          RANDOM: subkey
      }, inputs)
      features = exit_module.apply(initial_vars, inputs, rngs={RANDOM: subkey})

    self.assertEqual(features.shape,
                     (exp_params.batch_size, exp_params.hidden_dim))

  def test_interpolate_embedding_2d(self):
    source_shapes = (3, 5)
    target_shapes = (6, 10)
    emb_dims = 7

    emb = jnp.tile(
        jnp.arange(np.prod(source_shapes))[:, jnp.newaxis], (1, emb_dims))
    emb = emb[jnp.newaxis, ...]

    resized_emb = vits.interpolate_embedding_2d(emb, source_shapes,
                                                target_shapes)

    resized_emb_expected = np.array(
        [[0.0, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.0],
         [1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.25],
         [3.75, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 7.75],
         [6.25, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.25],
         [8.75, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 12.75],
         [10.0, 10.25, 10.75, 11.25, 11.75, 12.25, 12.75, 13.25, 13.75, 14.0]])

    resized_emb_expected = np.reshape(resized_emb_expected, (-1,))
    resized_emb_expected = np.tile(resized_emb_expected[:, np.newaxis],
                                   (1, emb_dims))
    resized_emb_expected = resized_emb_expected[np.newaxis, :, :]

    self.assertAllClose(resized_emb, test_utils.to_np(resized_emb_expected))

  @parameterized.product(pooled=[False, True], image_sizes=[(16, 16), (20, 24)])
  def test_vit(self, pooled: bool, image_sizes: Tuple[int, int]):
    exp_params = self._exp_params()
    exp_params.pos_embed_shapes = (3, 5)

    entry_p = self._vit_entry_layers(exp_params)
    transformer_p = self._vit_transformer_layers(exp_params)
    exit_p = self._vit_exit_layers(exp_params).set(pooled=pooled)

    p_vit = vits.VisionTransformer.HParams().set(
        name='vit',
        entry_layers_tpl=entry_p,
        transformer_layers_tpl=transformer_p,
        exit_layers_tpl=exit_p)

    vit_model = instantiate(p_vit)

    inputs_np = np.random.normal(
        size=[exp_params.batch_size, image_sizes[0], image_sizes[1], 3])
    inputs = jnp.asarray(inputs_np)

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      prng_key, subkey = jax.random.split(prng_key)
      initial_vars = vit_model.init({PARAMS: prng_key, RANDOM: subkey}, inputs)

      features = vit_model.apply(initial_vars, inputs, rngs={RANDOM: subkey})

    if pooled:
      self.assertEqual(features.shape,
                       (exp_params.batch_size, exp_params.hidden_dim))
    else:
      num_patches = np.prod(image_sizes) / (exp_params.patch_size**2)
      self.assertEqual(
          features.shape,
          (exp_params.batch_size, num_patches, exp_params.hidden_dim))


if __name__ == '__main__':
  absltest.main()
