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

"""Tests for flaxformer_model."""

from absl.testing import absltest
from absl.testing import parameterized
import fiddle as fdl
import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import pax_fiddle
from praxis import py_utils
from praxis import test_utils
from praxis.layers import flaxformer_models
from t5x import models as t5x_models
from t5x import partitioning as t5x_partitioning
from flaxformer import testing_utils
from flaxformer.t5x.fiddle.architectures import t5_1_1
from flaxformer.t5x.fiddle.models import t5x_model as t5x_model_configs

instantiate = base_layer.instantiate

expected_files = testing_utils.ExpectedJsonFiles(
    '__main__/praxis/layers/testdata')


def fiddle_configured_model():
  flaxformer_model = t5_1_1.small_fixture()
  # The current Praxis bridge uses partials instead of top-level-attribute
  # submodules, but we could change this in the future.
  as_partial = fdl.cast(fdl.Partial, flaxformer_model)
  encdec_p = flaxformer_models.EncoderDecoder.HParams.config(
      encoder_decoder_factory=as_partial, name='encoder_decoder')
  mdl_p = flaxformer_models.EncoderDecoderModel.config(
      encoder_decoder_tpl=encdec_p, name='mdl')
  return instantiate(mdl_p)


def fiddle_configured_scanned_model():
  flaxformer_model = t5_1_1.small_fixture(
      scanned=True,
      per_layer_relpos_bias=True,
  )
  as_partial = fdl.cast(fdl.Partial, flaxformer_model)
  encdec_p = flaxformer_models.EncoderDecoder.HParams.config(
      encoder_decoder_factory=as_partial, name='encoder_decoder')
  mdl_p = flaxformer_models.EncoderDecoderModel.config(
      encoder_decoder_tpl=encdec_p, name='mdl')
  return instantiate(mdl_p)


def factory_configured_model():
  encdec_p = flaxformer_models.FactoryBasedEncoderDecoder.HParams(
      name='encoder_decoder',
      num_encoder_layers=2,
      num_decoder_layers=1,
      activation_dtype='bfloat16',
      embed_dim=32,
      num_embeddings=65536,
      num_heads=2,
      head_dim=16,
      init_scale=0.1,
      dropout_rate=0.1,
      mlp_dim=64,
      activation_partitioning_dims=2)
  mdl_p = flaxformer_models.EncoderDecoderModel.config(
      encoder_decoder_tpl=encdec_p, name='mdl')
  return instantiate(mdl_p)


class FlaxFormerModelsTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  def test_flaxformer(self):
    decoder_p = flaxformer_models.FlaxFormerDecoder.HParams(
        name='flaxformer_decoder', num_layers=2)
    decoder = instantiate(decoder_p)

    seq_len = 128
    batch_size = 2
    # TODO(yonghui): Feed in real data.
    decoder_input_tokens = jnp.tile(
        jnp.arange(0, seq_len)[jnp.newaxis, :], (batch_size, 1))
    decoder_target_tokens = jnp.tile(
        jnp.arange(1, seq_len + 1)[jnp.newaxis, :], (batch_size, 1))
    decoder_segment_ids = jnp.ones((batch_size, seq_len), jnp.int32)
    decoder_positions = jnp.tile(
        jnp.arange(0, seq_len)[jnp.newaxis, :], (batch_size, 1))
    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      init_vars = decoder.init(prng_key, decoder_input_tokens,
                               decoder_target_tokens, decoder_segment_ids,
                               decoder_positions)
      out = decoder.apply(init_vars, decoder_input_tokens,
                          decoder_target_tokens, decoder_segment_ids,
                          decoder_positions)
      del out  # unused

  def test_flaxformer_model(self):
    decoder_p = flaxformer_models.FlaxFormerDecoder.HParams(num_layers=2)
    decoder_mdl_p = flaxformer_models.LanguageModel.HParams(
        flax_decoder_tpl=decoder_p,
        z_loss=0.0001,
        label_smoothing=0.1,
        name='mdl')
    decoder_mdl = instantiate(decoder_mdl_p)

    seq_len = 8
    batch_size = 2
    # TODO(yonghui): Feed in real data.
    decoder_input_tokens = jnp.tile(
        jnp.arange(0, seq_len)[jnp.newaxis, :], (batch_size, 1))
    decoder_target_tokens = jnp.tile(
        jnp.arange(1, seq_len + 1)[jnp.newaxis, :], (batch_size, 1))
    decoder_segment_ids = jnp.ones((batch_size, seq_len), jnp.int32)
    decoder_positions = jnp.tile(
        jnp.arange(0, seq_len)[jnp.newaxis, :], (batch_size, 1))
    decoder_loss_weights = jnp.ones((batch_size, seq_len), jnp.float32)
    input_batch = py_utils.NestedMap(
        decoder_input_tokens=decoder_input_tokens,
        decoder_target_tokens=decoder_target_tokens,
        decoder_segment_ids=decoder_segment_ids,
        decoder_positions=decoder_positions,
        decoder_loss_weights=decoder_loss_weights)
    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      init_vars = decoder_mdl.init(prng_key, input_batch)
      expected_files.check_params(init_vars['params'],
                                  'language_model_param_shapes.json')
      out = decoder_mdl.apply(init_vars, input_batch)
      del out  # unused

  @parameterized.named_parameters(
      {
          'testcase_name': 'fiddle_config',
          'get_model_fn': fiddle_configured_model,
      },
      {
          'testcase_name': 'fiddle_scanned_config',
          'get_model_fn': fiddle_configured_scanned_model,
      },
      {
          'testcase_name': 'factory_config',
          'get_model_fn': factory_configured_model,
      },
  )
  def test_encoder_decoder_model(self, get_model_fn):
    model = get_model_fn()

    encoder_seq_len = 32
    decoder_seq_len = 16
    batch_size = 2

    encoder_input_tokens = jnp.tile(
        jnp.arange(0, encoder_seq_len)[jnp.newaxis, :], (batch_size, 1))
    decoder_input_tokens = jnp.tile(
        jnp.arange(0, decoder_seq_len)[jnp.newaxis, :], (batch_size, 1))
    decoder_target_tokens = jnp.tile(
        jnp.arange(1, decoder_seq_len + 1)[jnp.newaxis, :], (batch_size, 1))
    encoder_segment_ids = jnp.ones((batch_size, encoder_seq_len), jnp.int32)
    decoder_segment_ids = jnp.ones((batch_size, decoder_seq_len), jnp.int32)
    encoder_positions = jnp.tile(
        jnp.arange(0, encoder_seq_len)[jnp.newaxis, :], (batch_size, 1))
    decoder_positions = jnp.tile(
        jnp.arange(0, decoder_seq_len)[jnp.newaxis, :], (batch_size, 1))
    decoder_loss_weights = jnp.ones((batch_size, decoder_seq_len), jnp.float32)

    input_batch = py_utils.NestedMap(
        decoder_input_tokens=decoder_input_tokens,
        decoder_target_tokens=decoder_target_tokens,
        decoder_segment_ids=decoder_segment_ids,
        decoder_positions=decoder_positions,
        encoder_input_tokens=encoder_input_tokens,
        encoder_segment_ids=encoder_segment_ids,
        encoder_positions=encoder_positions,
        decoder_loss_weights=decoder_loss_weights)

    with base_layer.JaxContext.new_context():
      prng_key = {
          'params': jax.random.PRNGKey(seed=123),
          'dropout': jax.random.PRNGKey(seed=456)
      }
      init_vars = model.init(prng_key, input_batch)

      out = model.apply(
          init_vars, input_batch, rngs={'dropout': jax.random.PRNGKey(456)})
      del out  # unused

  def test_encoder_decoder_loss(self):
    """Tests that the `compute_loss()` function is equivalent to T5X."""
    model = fiddle_configured_model()

    batch_size = 5
    seq_len = 7
    vocab_size = 13

    # Setup shared inputs.
    np.random.seed(0)
    logits = np.random.normal(size=(batch_size, seq_len, vocab_size))
    target_tokens = np.argmax(logits, axis=-1)
    dummy_input_tokens = target_tokens
    loss_weights = np.ones((batch_size, seq_len))

    predictions = py_utils.NestedMap(logits=logits)
    input_batch = py_utils.NestedMap(
        encoder_input_tokens=dummy_input_tokens,
        decoder_input_tokens=dummy_input_tokens,
        decoder_target_tokens=target_tokens,
        decoder_loss_weights=loss_weights)
    # Need to run model.compute_predictions to trigger self.decoder.
    init_vars = model.init(
        jax.random.PRNGKey(0), input_batch, method=model.compute_predictions)
    # model.compute_loss(predictions, input_batch)
    result, variables = model.apply(
        init_vars,
        predictions,
        input_batch,
        method=model.compute_loss,
        mutable=['summaries'])
    metrics, _ = result
    self.assertEqual(metrics['z_loss'][0],
                     variables['summaries']['z_loss_scalar'])

    class FakeLogitsEncoder(t5x_models.EncoderDecoderModel):

      def _compute_logits(self, params, batch, dropout_rng):
        return logits

    t5x_model_config = t5x_model_configs.t5x_model_fixture.as_buildable(
        t5_1_1.small_fixture(),)
    fdl.update_callable(t5x_model_config, FakeLogitsEncoder)
    t5x_model_config.loss_normalizing_factor = 'NUM_REAL_TARGET_TOKENS'
    t5x_model = pax_fiddle.build(t5x_model_config)
    t5x_loss, unused_t5x_metrics = t5x_model.loss_fn(
        params=init_vars['params']['encoder_decoder']['enc_dec']['cld'],
        batch={
            'decoder_target_tokens': target_tokens,
            'decoder_loss_weights': loss_weights,
        },
        dropout_rng=jax.random.PRNGKey(0),
    )

    np.testing.assert_allclose(t5x_loss, metrics['total_loss'][0])

  def test_encoder_decoder_decode(self):
    pax_model = fiddle_configured_model()
    t5x_model_cfg = t5x_model_configs.t5x_model_fixture.as_buildable(
        t5_1_1.small_fixture(),)
    t5x_model = pax_fiddle.build(t5x_model_cfg)

    prng_key = {
        'params': jax.random.PRNGKey(seed=123),
        'dropout': jax.random.PRNGKey(seed=456)
    }

    encoder_seq_len = 32
    decoder_seq_len = 16
    batch_size = 2

    encoder_input_tokens = jnp.tile(
        jnp.arange(0, encoder_seq_len)[jnp.newaxis, :], (batch_size, 1))
    decoder_input_tokens = jnp.tile(
        jnp.arange(0, decoder_seq_len)[jnp.newaxis, :], (batch_size, 1))
    decoder_target_tokens = jnp.tile(
        jnp.arange(1, decoder_seq_len + 1)[jnp.newaxis, :], (batch_size, 1))
    encoder_segment_ids = jnp.ones((batch_size, encoder_seq_len), jnp.int32)
    decoder_segment_ids = jnp.ones((batch_size, decoder_seq_len), jnp.int32)
    encoder_positions = jnp.tile(
        jnp.arange(0, encoder_seq_len)[jnp.newaxis, :], (batch_size, 1))
    decoder_positions = jnp.tile(
        jnp.arange(0, decoder_seq_len)[jnp.newaxis, :], (batch_size, 1))
    decoder_loss_weights = jnp.ones((batch_size, decoder_seq_len), jnp.float32)

    input_batch = py_utils.NestedMap(
        decoder_input_tokens=decoder_input_tokens,
        decoder_target_tokens=decoder_target_tokens,
        decoder_segment_ids=decoder_segment_ids,
        decoder_positions=decoder_positions,
        encoder_input_tokens=encoder_input_tokens,
        encoder_segment_ids=encoder_segment_ids,
        encoder_positions=encoder_positions,
        decoder_loss_weights=decoder_loss_weights)
    init_vars = pax_model.init(prng_key, input_batch)

    # Run Pax version of decoding (originally predict_batch_with_aux)
    pax_result = pax_model.apply(
        init_vars, input_batch, method=pax_model.decode)
    pax_result_ids = pax_result[1].output_ids

    # Run the original T5X version of decoding
    t5x_result = t5x_model.predict_batch_with_aux(
        init_vars['params']['encoder_decoder']['enc_dec']['cld'],
        input_batch,
        decoder_params={'eos_id': 1})
    t5x_result_ids = t5x_result[0]

    # Test whether they are the same.
    np.testing.assert_array_equal(pax_result_ids, t5x_result_ids)

  def test_sharded_flaxformer_model(self):
    decoder_p = flaxformer_models.FlaxFormerDecoder.HParams(num_layers=2)
    decoder_mdl_p = flaxformer_models.LanguageModel.HParams(
        flax_decoder_tpl=decoder_p,
        z_loss=0.0001,
        label_smoothing=0.1,
        name='mdl',
        mesh_axis_names=['replica', 'data', 'model'],
        ici_mesh_shape=[1, 2, 4],
        logical_axes_rules=t5x_partitioning.standard_logical_axis_rules())
    decoder_mdl = instantiate(decoder_mdl_p)

    seq_len = 8
    batch_size = 2
    # TODO(yonghui): Feed in real data.
    decoder_input_tokens = jnp.tile(
        jnp.arange(0, seq_len)[jnp.newaxis, :], (batch_size, 1))
    decoder_target_tokens = jnp.tile(
        jnp.arange(1, seq_len + 1)[jnp.newaxis, :], (batch_size, 1))
    decoder_segment_ids = jnp.ones((batch_size, seq_len), jnp.int32)
    decoder_positions = jnp.tile(
        jnp.arange(0, seq_len)[jnp.newaxis, :], (batch_size, 1))
    decoder_loss_weights = jnp.ones((batch_size, seq_len), jnp.float32)
    input_batch = py_utils.NestedMap(
        decoder_input_tokens=decoder_input_tokens,
        decoder_target_tokens=decoder_target_tokens,
        decoder_segment_ids=decoder_segment_ids,
        decoder_positions=decoder_positions,
        decoder_loss_weights=decoder_loss_weights)

    init_vars = decoder_mdl.abstract_init_with_metadata(input_batch)

    dec = init_vars['params']['decoder']['dec']['cld']['decoder']
    # Tests sharding for embedding variable.
    self.assertEqual(
        dec['token_embedder']['embedding'].tensor_split_dims_mapping,
        ['model', None])
    # Tests sharding for layernorm.
    self.assertEqual(dec['decoder_norm']['scale'].tensor_split_dims_mapping,
                     [None])

  def test_sharded_encoder_decoder_model(self):
    flaxformer_model = t5_1_1.small_fixture()
    as_partial = fdl.cast(fdl.Partial, flaxformer_model)
    encdec_p = flaxformer_models.EncoderDecoder.HParams.config(
        encoder_decoder_factory=as_partial, name='encoder_decoder')
    mdl_p = flaxformer_models.EncoderDecoderModel.config(
        encoder_decoder_tpl=encdec_p,
        name='mdl',
        mesh_axis_names=['replica', 'data', 'model'],
        ici_mesh_shape=[1, 2, 4],
        logical_axes_rules=t5x_partitioning.standard_logical_axis_rules())
    encdec_mdl = instantiate(mdl_p)

    encoder_seq_len = 32
    decoder_seq_len = 16
    batch_size = 2

    encoder_input_tokens = jnp.tile(
        jnp.arange(0, encoder_seq_len)[jnp.newaxis, :], (batch_size, 1))
    decoder_input_tokens = jnp.tile(
        jnp.arange(0, decoder_seq_len)[jnp.newaxis, :], (batch_size, 1))
    decoder_target_tokens = jnp.tile(
        jnp.arange(1, decoder_seq_len + 1)[jnp.newaxis, :], (batch_size, 1))
    encoder_segment_ids = jnp.ones((batch_size, encoder_seq_len), jnp.int32)
    decoder_segment_ids = jnp.ones((batch_size, decoder_seq_len), jnp.int32)
    encoder_positions = jnp.tile(
        jnp.arange(0, encoder_seq_len)[jnp.newaxis, :], (batch_size, 1))
    decoder_positions = jnp.tile(
        jnp.arange(0, decoder_seq_len)[jnp.newaxis, :], (batch_size, 1))
    decoder_loss_weights = jnp.ones((batch_size, decoder_seq_len), jnp.float32)

    input_batch = py_utils.NestedMap(
        decoder_input_tokens=decoder_input_tokens,
        decoder_target_tokens=decoder_target_tokens,
        decoder_segment_ids=decoder_segment_ids,
        decoder_positions=decoder_positions,
        encoder_input_tokens=encoder_input_tokens,
        encoder_segment_ids=encoder_segment_ids,
        encoder_positions=encoder_positions,
        decoder_loss_weights=decoder_loss_weights)
    init_vars = encdec_mdl.abstract_init_with_metadata(input_batch)

    encdec = init_vars['params']['encoder_decoder']['enc_dec']['cld']
    # Tests sharding for embedding variable.
    self.assertEqual(
        encdec['token_embedder']['embedding'].tensor_split_dims_mapping,
        ['model', None])
    # Tests sharding for layernorm.
    self.assertEqual(
        encdec['decoder']['decoder_norm']['scale'].tensor_split_dims_mapping,
        [None])


if __name__ == '__main__':
  absltest.main()
