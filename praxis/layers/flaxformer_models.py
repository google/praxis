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

"""An inter-op wrapper that directly instantiates flaxformer models in pax."""

from typing import Any, Callable, Dict, Optional, Tuple

import clu.metrics as clu_metrics
from flax import linen
from jax import numpy as jnp
from praxis import base_layer
from praxis import base_model
from praxis import py_utils
from praxis import pytypes
from praxis.layers import flax_adapter
from t5x import losses as t5x_losses

from flaxformer.architectures.t5 import t5_architecture
from flaxformer.components import dense
from flaxformer.components import embedding
from flaxformer.components import layer_norm
from flaxformer.components import relative_position_biases
from flaxformer.components.attention import dense_attention

NestedMap = py_utils.NestedMap
Predictions = base_model.Predictions
Metrics = pytypes.Metrics
BaseHParams = base_layer.BaseLayer.HParams
sub_config_field = base_layer.sub_config_field


class FlaxFormerDecoder(base_layer.BaseLayer):
  """A wrapper of a Flaxformer decoder.

  This model architecture is derived from the following gin config:
  # Hidden encoder-decoder config.gin

  TODO(Pax): Create the model directly following the gin config without
  manual translation.
  TODO(Pax): Enable spmd sharding of the flaxformer models / t5x
  models.
  """

  class HParams(BaseHParams):
    """Associated hyperparams for this layer class.

    Attributes:
      num_layers: Number of decoder layers.
    """
    num_layers: int = 2
    activation_dtype: str = 'bfloat16'
    embed_dim: int = 2048
    num_embeddings: int = 32128
    num_heads: int = 32
    head_dim: int = 64
    init_scale: float = 1.0
    dropout_rate: float = 0.0
    mlp_dim: int = 5120
    activation_partitioning_dims: int = 1

  def setup(self) -> None:
    p = self.hparams
    super().setup()
    activation_dtype = p.activation_dtype
    embed_dim = p.embed_dim
    num_embeddings = p.num_embeddings
    num_heads = p.num_heads
    head_dim = p.head_dim
    init_scale = p.init_scale
    dropout_rate = p.dropout_rate
    mlp_dim = p.mlp_dim
    activation_partitioning_dims = p.activation_partitioning_dims
    num_decoder_layers = p.num_layers

    def token_embedder_factory():
      emb_init_kwargs = dict(
          attend_dtype='float32',
          cast_input_dtype='int32',
          dtype=activation_dtype,
          embedding_init=linen.initializers.normal(stddev=1.0),
          features=embed_dim,
          name='token_embedder',
          num_embeddings=num_embeddings,
          one_hot=False)
      return embedding.Embed(**emb_init_kwargs)

    def relative_position_emb_factory():
      init_kwargs = dict(
          dtype=activation_dtype,
          embedding_init=linen.initializers.variance_scaling(
              distribution='uniform', mode='fan_avg', scale=init_scale),
          max_distance=128,
          num_buckets=32,
          num_heads=num_heads)
      return relative_position_biases.RelativePositionBiases(**init_kwargs)

    def layer_norm_factory():
      return layer_norm.T5LayerNorm(dtype=activation_dtype)

    def self_attention_factory():
      init_kwargs = dict(
          bias_init=linen.initializers.normal(stddev=1e-06),
          broadcast_dropout=True,
          dropout_rate=dropout_rate,
          dtype=activation_dtype,
          head_dim=head_dim,
          kernel_init=linen.initializers.variance_scaling(
              distribution='normal', mode='fan_in', scale=init_scale),
          num_heads=num_heads,
          use_bias=False)
      return dense_attention.MultiHeadDotProductAttention(**init_kwargs)

    def mlp_factory():
      init_kwargs = dict(
          activations=('gelu', 'linear'),
          bias_init=linen.initializers.normal(stddev=1e-06),
          dtype=activation_dtype,
          final_dropout_rate=0,
          intermediate_dim=mlp_dim,
          intermediate_dropout_rate=dropout_rate,
          kernel_init=linen.initializers.variance_scaling(
              distribution='truncated_normal', mode='fan_in', scale=init_scale),
          use_bias=False)
      return dense.MlpBlock(**init_kwargs)

    def output_logits_factory():
      init_kwargs = dict(
          bias_init=linen.initializers.normal(stddev=1e-06),
          dtype='float32',
          features=num_embeddings,
          kernel_axis_names=['embed', 'vocab'],
          kernel_init=linen.initializers.variance_scaling(
              distribution='truncated_normal', mode='fan_in', scale=init_scale),
          use_bias=False)
      return dense.DenseGeneral(**init_kwargs)

    def dropout_factory():
      return linen.Dropout(broadcast_dims=(-2,), rate=dropout_rate)

    def decoder_layer_factory(shared_relative_position_bias=None):
      init_kwargs = dict(
          activation_partitioning_dims=activation_partitioning_dims,
          dropout_factory=dropout_factory,
          encoder_decoder_attention=None,
          layer_norm_factory=layer_norm_factory,
          mlp=mlp_factory(),
          self_attention=self_attention_factory(),
          shared_relative_position_bias=shared_relative_position_bias)
      return t5_architecture.DecoderLayer(**init_kwargs)

    def decoder_factory(shared_token_embedder=None):
      init_kwargs = dict(
          dropout_factory=dropout_factory,
          dtype=activation_dtype,
          layer_factory=decoder_layer_factory,
          layer_norm_factory=layer_norm_factory,
          num_layers=num_decoder_layers,
          output_logits_factory=output_logits_factory,
          position_embedder_factory=None,
          shared_relative_position_bias_factory=relative_position_emb_factory,
          token_embedder_factory=token_embedder_factory,
          shared_token_embedder=shared_token_embedder)
      return t5_architecture.Decoder(**init_kwargs)

    def decoder_only_factory():
      init_kwargs = dict(
          decoder_factory=decoder_factory, dtype=activation_dtype)
      return t5_architecture.DecoderOnly(**init_kwargs)

    def var_init_args_fn():
      # model initialization shouldn't be sensitive to batch size and sequence
      # length.
      batch_size = 2
      seq_length = 2
      return (jnp.zeros((batch_size, seq_length), dtype=jnp.int32),
              jnp.zeros((batch_size, seq_length), dtype=jnp.int32))

    flaxformer_decoder = flax_adapter.FlaxModuleAdapter.HParams(
        module_factory_method=decoder_only_factory,
        var_init_args=var_init_args_fn)

    self.create_child('dec', flaxformer_decoder)

  def __call__(self, *args, **kwargs):
    return self.dec(*args, **kwargs)


class EncoderDecoder(base_layer.BaseLayer):
  """A wrapper of a T5 Encoder Decoder."""

  class HParams(BaseHParams):
    """Associated hyperparams for this layer class.

    Attributes:
      encoder_decoder_factory: Callable which will generate a Flaxformer model.
    """
    encoder_decoder_factory: Optional[Callable[[], linen.Module]] = None

  def _build_wrapped_module(self) -> linen.Module:
    if self.hparams.encoder_decoder_factory is None:
      raise ValueError('encoder_decoder_factory is required!')
    return self.hparams.encoder_decoder_factory()

  def setup(self) -> None:
    super().setup()

    def var_init_args_fn():
      # model initialization shouldn't be sensitive to batch size and sequence
      # length.
      batch_size = 2
      encoder_seq_length = 2
      decoder_seq_length = 4
      return (
          # encoder_input_tokens
          jnp.zeros((batch_size, encoder_seq_length), dtype=jnp.int32),
          # decoder_input_tokens
          jnp.zeros((batch_size, decoder_seq_length), dtype=jnp.int32),
          # decoder_target_tokens
          jnp.zeros((batch_size, decoder_seq_length), dtype=jnp.int32),
          # encoder_segment_ids
          jnp.zeros((batch_size, encoder_seq_length), dtype=jnp.int32),
          # decoder_segment_ids
          jnp.zeros((batch_size, decoder_seq_length), dtype=jnp.int32),
          # encoder_positions
          jnp.zeros((batch_size, encoder_seq_length), dtype=jnp.int32),
          # decoder_positions
          jnp.zeros((batch_size, decoder_seq_length), dtype=jnp.int32))

    encoder_decoder = flax_adapter.FlaxModuleAdapter.HParams(
        module_factory_method=self._build_wrapped_module,
        var_init_args=var_init_args_fn)

    self.create_child('enc_dec', encoder_decoder)

  def __call__(self, *args, **kwargs):
    return self.enc_dec(*args, **kwargs)


class FactoryBasedEncoderDecoder(EncoderDecoder):
  """Legacy EncoderDecoder that exposes a few common settings.

  In general, we recommend using Fiddle to configure Flaxformer models; this
  allows deep overrides in model settings.
  """

  class HParams(BaseHParams):
    """Associated hyperparams for this layer class.

    Attributes:
      num_encoder_layers: Number of encoder layers.
      num_decoder_layers: Number of decoder layers.
    """
    num_encoder_layers: int = 12
    num_decoder_layers: int = 12
    activation_dtype: str = 'bfloat16'
    embed_dim: int = 768
    num_embeddings: int = 250112
    num_heads: int = 12
    head_dim: int = 64
    init_scale: float = 1.0
    dropout_rate: float = 0.1
    mlp_dim: int = 2048
    activation_partitioning_dims: int = 2

  def _build_wrapped_module(self) -> linen.Module:
    p: FactoryBasedEncoderDecoder.HParams = self.hparams
    activation_dtype = p.activation_dtype
    embed_dim = p.embed_dim
    num_embeddings = p.num_embeddings
    num_heads = p.num_heads
    head_dim = p.head_dim
    init_scale = p.init_scale
    dropout_rate = p.dropout_rate
    mlp_dim = p.mlp_dim
    activation_partitioning_dims = p.activation_partitioning_dims
    num_encoder_layers = p.num_encoder_layers
    num_decoder_layers = p.num_decoder_layers

    def shared_token_embedder_factory():
      init_kwargs = dict(
          attend_dtype='float32',
          cast_input_dtype='int32',
          dtype=activation_dtype,
          embedding_init=linen.initializers.normal(stddev=1.0),
          features=embed_dim,
          name='token_embedder',
          num_embeddings=num_embeddings,
          one_hot=True)
      return embedding.Embed(**init_kwargs)

    def relative_position_bias_factory():
      init_kwargs = dict(
          dtype=activation_dtype,
          embedding_init=linen.initializers.variance_scaling(
              distribution='uniform', mode='fan_avg', scale=init_scale),
          max_distance=128,
          num_buckets=32,
          num_heads=num_heads)
      return relative_position_biases.RelativePositionBiases(**init_kwargs)

    def layer_norm_factory():
      return layer_norm.T5LayerNorm(dtype=activation_dtype)

    def self_attention_factory():
      init_kwargs = dict(
          bias_init=linen.initializers.normal(stddev=1e-06),
          broadcast_dropout=True,
          dropout_rate=dropout_rate,
          dtype=activation_dtype,
          head_dim=head_dim,
          kernel_init=linen.initializers.variance_scaling(
              distribution='normal', mode='fan_in', scale=init_scale),
          num_heads=num_heads,
          use_bias=False)
      return dense_attention.MultiHeadDotProductAttention(**init_kwargs)

    def mlp_factory():
      init_kwargs = dict(
          activations=('gelu', 'linear'),
          bias_init=linen.initializers.normal(stddev=1e-06),
          dtype=activation_dtype,
          final_dropout_rate=0,
          intermediate_dim=mlp_dim,
          intermediate_dropout_rate=dropout_rate,
          kernel_init=linen.initializers.variance_scaling(
              distribution='normal', mode='fan_in', scale=init_scale),
          use_bias=False)
      return dense.MlpBlock(**init_kwargs)

    def output_logits_factory():
      init_kwargs = dict(
          bias_init=linen.initializers.normal(stddev=1e-06),
          dtype='float32',
          features=num_embeddings,
          kernel_axis_names=['embed', 'vocab'],
          kernel_init=linen.initializers.variance_scaling(
              distribution='truncated_normal', mode='fan_in', scale=init_scale),
          use_bias=False)
      return dense.DenseGeneral(**init_kwargs)

    def dropout_factory():
      # TODO(jqmu): dropout is not needed for inference.
      return linen.Dropout(broadcast_dims=(-2,), rate=dropout_rate)

    def encoder_layer_factory(shared_relative_position_bias=None):
      init_kwargs = dict(
          activation_partitioning_dims=activation_partitioning_dims,
          attention=self_attention_factory(),
          dropout_factory=dropout_factory,
          layer_norm_factory=layer_norm_factory,
          mlp=mlp_factory(),
          relative_position_bias_factory=relative_position_bias_factory,
          scanned=True,
          shared_relative_position_bias=shared_relative_position_bias)
      return t5_architecture.EncoderLayer(**init_kwargs)

    def encoder_factory(shared_token_embedder):
      init_kwargs = dict(
          dtype=activation_dtype,
          input_dropout_factory=dropout_factory,
          layer_factory=encoder_layer_factory,
          layer_norm_factory=layer_norm_factory,
          layer_remat='full',
          num_layers=num_encoder_layers,
          output_dropout_factory=dropout_factory,
          position_embedder_factory=None,
          scan_layers=True,
          shared_relative_position_bias_factory=None,
          shared_token_embedder=shared_token_embedder,
      )
      return t5_architecture.Encoder(**init_kwargs)

    def decoder_layer_factory(shared_relative_position_bias=None):
      init_kwargs = dict(
          activation_partitioning_dims=activation_partitioning_dims,
          dropout_factory=dropout_factory,
          encoder_decoder_attention=self_attention_factory(),
          layer_norm_factory=layer_norm_factory,
          mlp=mlp_factory(),
          relative_position_bias_factory=relative_position_bias_factory,
          scanned=True,
          self_attention=self_attention_factory(),
          shared_relative_position_bias=shared_relative_position_bias,
      )
      return t5_architecture.DecoderLayer(**init_kwargs)

    def decoder_factory(shared_token_embedder):
      init_kwargs = dict(
          dropout_factory=dropout_factory,
          dtype=activation_dtype,
          layer_factory=decoder_layer_factory,
          layer_norm_factory=layer_norm_factory,
          layer_remat='full',
          num_layers=num_decoder_layers,
          output_logits_factory=output_logits_factory,
          position_embedder_factory=None,
          scan_layers=True,
          shared_relative_position_bias_factory=None,
          shared_token_embedder=shared_token_embedder,
      )
      return t5_architecture.Decoder(**init_kwargs)

    return t5_architecture.EncoderDecoder(
        encoder_factory=encoder_factory,
        decoder_factory=decoder_factory,
        dtype=activation_dtype,
        scan_layers=True,
        shared_token_embedder_factory=shared_token_embedder_factory,
    )


class LanguageModel(base_model.BaseModel):
  """Language Model base task."""

  class HParams(base_model.BaseModel.HParams):
    """Associated hyperparams for this model class.

    Attributes:
      decoder: Flaxformer decoder params.
      loss_normalizing_factor: Normalization factor for loss.
      label_smoothing: Amount of label smoothing to apply.
      z_loss: Coefficient for auxiliary z-loss loss term.
    """
    decoder: base_layer.BaseLayer.HParams = sub_config_field(
        FlaxFormerDecoder.HParams)
    loss_normalizing_factor: str = 'NUM_REAL_TARGET_TOKENS'
    label_smoothing: float = 0.0
    z_loss: float = 0.0001

  def setup(self):
    p = self.hparams
    self.create_child('decoder', p.decoder)

  def compute_predictions(self, input_batch: NestedMap) -> Predictions:
    """Compute model predictions.

    Args:
      input_batch: A NestedMap of an input batch. It should contain the
        following elements. "decoder_input_tokens" - the input tokens, of shape
        [batch, seq_len]. "decoder_target_tokens" - the target tokens to predict
        , of shape [batch_size, seq_len]. "decoder_segment_ids" - the segment
        that each token belongs to, of shape [batch_size, seq_len].
        "decoder_positions" - the position from the beginning of a segment that
        token is at. 'decoder_loss_weights' - the weight of each target token.

    Returns:
      A NestedMap of predictions.
    """
    get_elem = lambda x, k: x[k] if k in x else None
    logits = self.decoder(
        decoder_input_tokens=input_batch.decoder_input_tokens,
        decoder_target_tokens=input_batch.decoder_target_tokens,
        decoder_segment_ids=get_elem(input_batch, 'decoder_segment_ids'),
        decoder_positions=get_elem(input_batch, 'decoder_positions'))
    return NestedMap(logits=logits)

  def compute_loss(self, predictions: Predictions,
                   input_batch: NestedMap) -> Tuple[Metrics, Dict[str, Any]]:
    p = self.hparams
    assert 'decoder_loss_weights' in input_batch
    loss_weights = input_batch.decoder_loss_weights

    if p.loss_normalizing_factor == 'NUM_REAL_TARGET_TOKENS':
      loss_normalizing_factor = jnp.sum(loss_weights)
    else:
      assert NotImplementedError('loss_normalizing_factor: %s not implemented' %
                                 p.loss_normalizing_factor)

    # TODO(yonghui): reimplement the loss in pax instead of having a dependency
    # on t5x for loss computations.
    targets = input_batch['decoder_target_tokens']
    loss, z_loss, weight_sum = t5x_losses.compute_weighted_cross_entropy(
        predictions.logits,
        targets=targets,
        weights=loss_weights,
        label_smoothing=p.label_smoothing,
        z_loss=p.z_loss,
        loss_normalizing_factor=loss_normalizing_factor)
    accuracy = clu_metrics.Accuracy.from_model_output(
        logits=predictions.logits,
        labels=targets.astype(jnp.int32),
        mask=loss_weights).compute()

    metrics = {
        'total_loss': (loss, weight_sum),
        'z_loss': (z_loss, weight_sum),
        'cross_entropy': (loss - z_loss, weight_sum),
        'accuracy': (accuracy, weight_sum),
    }
    self.add_summary('z_loss', z_loss)
    self.add_summary('cross_entropy', loss - z_loss)
    self.add_summary('accuracy', accuracy)
    # loss already contains z_loss
    return metrics, NestedMap()


class EncoderDecoderModel(LanguageModel):
  """EncoderDecoder base task."""

  class HParams(base_model.BaseModel.HParams):
    """Associated hyperparams for this model class.

    Attributes:
      encoder_decoder: Flaxformer encoder decoder params.
      loss_normalizing_factor: Normalization factor for loss.
      label_smoothing: Amount of label smoothing to apply.
      z_loss: Coefficient for auxiliary z-loss loss term.
    """
    encoder_decoder: base_layer.BaseLayer.HParams = sub_config_field(
        EncoderDecoder.HParams)
    loss_normalizing_factor: str = 'NUM_REAL_TARGET_TOKENS'
    label_smoothing: float = 0.0
    z_loss: float = 0.0001

  def setup(self):
    p = self.hparams
    self.create_child('encoder_decoder', p.encoder_decoder)

  def compute_predictions(self, input_batch: NestedMap) -> Predictions:
    """Compute model predictions.

    Args:
      input_batch: A NestedMap of an input batch. It should contain the
        following elements. "encoder_input_tokens" - the encoder input tokens,
        of shape [batch, encoder_len]. "decoder_input_tokens" - the decoder
        input tokens, of shape [batch, decoder_len]. "decoder_target_tokens" -
        the target tokens to predict, of shape [batch_size, decoder_len].
        "encoder_segment_ids" - the segment that each encoder token belongs to,
        of shape [batch_size, encoder_len]. "decoder_segment_ids" - the segment
        that each decoder token belongs to, of shape [batch_size, decoder_len].
        "encoder_positions" - the position from the beginning of a encoder
        segment that token is at. "decoder_positions" - the position from the
        beginning of a decoder segment that token is at. 'decoder_loss_weights'
        - the weight of each target token.

    Returns:
      A NestedMap of predictions.
    """
    get_elem = lambda x, k: x[k] if k in x else None

    logits = self.encoder_decoder(
        encoder_input_tokens=input_batch.encoder_input_tokens,
        decoder_input_tokens=input_batch.decoder_input_tokens,
        decoder_target_tokens=input_batch.decoder_target_tokens,
        encoder_segment_ids=get_elem(input_batch, 'encoder_segment_ids'),
        decoder_segment_ids=get_elem(input_batch, 'decoder_segment_ids'),
        encoder_positions=get_elem(input_batch, 'encoder_positions'),
        decoder_positions=get_elem(input_batch, 'decoder_positions'))
    return NestedMap(logits=logits)
