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

import functools
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import clu.metrics as clu_metrics
from flax import linen
from flax.linen import partitioning as flax_partitioning
import jax
from jax import numpy as jnp
from praxis import base_layer
from praxis import base_model
from praxis import decoder_hparams
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis.layers import flax_adapter
from t5x import decoding as t5x_decoding
from t5x import losses as t5x_losses

from flaxformer.architectures.t5 import parallel_fused_decoder
from flaxformer.architectures.t5 import t5_architecture
from flaxformer.components import dense
from flaxformer.components import embedding
from flaxformer.components import layer_norm
from flaxformer.components import relative_position_biases
from flaxformer.components.attention import dense_attention

NestedMap = py_utils.NestedMap
Predictions = base_model.Predictions
WeightedScalars = pytypes.WeightedScalars
BaseHParams = base_layer.BaseLayer.HParams
sub_config_field = base_layer.sub_config_field
LogicalAxisRules = pytypes.LogicalAxisRules
DecodeOut = Tuple[WeightedScalars, NestedMap, Any]
PyTreeDef = type(jax.tree_util.tree_structure(None))
SampleDecoderHParams = decoder_hparams.SampleDecoderHParams
DecoderHParams = decoder_hparams.DecoderHParams
GreedyDecoderHParams = decoder_hparams.GreedyDecoderHParams


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
    mlp_activations: Sequence[str] = ('gelu', 'linear')
    mlp_dim: int = 5120
    mlp_out_dim: Optional[int] = None
    mlp_precomputed_intermediates: bool = False
    activation_partitioning_dims: int = 1
    logical_axes_rules: Optional[LogicalAxisRules] = None
    scan_layers: bool = False
    shared_relative_bias: bool = True
    decode_layer_norm_use_scale: bool = True
    final_layer_norm_use_scale: bool = True
    layer_norm_center_scale_at_zero: bool = False
    use_multi_query_attention: bool = False
    use_rotary_embedding: bool = False
    parallel_fused_decoder_layer: bool = False
    use_output_logits: bool = True

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
    mlp_activations = p.mlp_activations
    mlp_dim = p.mlp_dim
    activation_partitioning_dims = p.activation_partitioning_dims
    num_decoder_layers = p.num_layers
    scan_layers = p.scan_layers
    shared_relative_bias = p.shared_relative_bias
    decode_layer_norm_use_scale = p.decode_layer_norm_use_scale
    final_layer_norm_use_scale = p.final_layer_norm_use_scale
    layer_norm_center_scale_at_zero = p.layer_norm_center_scale_at_zero
    use_multi_query_attention = p.use_multi_query_attention
    use_rotary_embedding = p.use_rotary_embedding
    parallel_fused_decoder_layer = p.parallel_fused_decoder_layer
    mlp_out_dim = p.mlp_out_dim
    mlp_precomputed_intermediates = p.mlp_precomputed_intermediates
    use_output_logits = p.use_output_logits

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
      t5_layer_norm = layer_norm.T5LayerNorm(dtype=activation_dtype)
      t5_layer_norm.center_scale_at_zero = layer_norm_center_scale_at_zero
      t5_layer_norm.use_scale = decode_layer_norm_use_scale
      return t5_layer_norm

    def final_layer_norm_factory():
      t5_layer_norm = layer_norm.T5LayerNorm(dtype=activation_dtype)
      t5_layer_norm.use_scale = final_layer_norm_use_scale
      return t5_layer_norm

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
          use_bias=False,
          use_rotary_embedding=use_rotary_embedding)
      if use_multi_query_attention:
        init_kwargs['rescale_logits'] = True
        init_kwargs['split_head_kernel'] = True
        init_kwargs['out_features'] = embed_dim
        return dense_attention.MultiQueryDotProductAttention(**init_kwargs)
      return dense_attention.MultiHeadDotProductAttention(**init_kwargs)

    def mlp_factory():
      init_kwargs = dict(
          activations=mlp_activations,
          bias_init=linen.initializers.normal(stddev=1e-06),
          dtype=activation_dtype,
          final_dropout_rate=0,
          intermediate_dim=mlp_dim,
          intermediate_dropout_rate=dropout_rate,
          kernel_init=linen.initializers.variance_scaling(
              distribution='truncated_normal', mode='fan_in', scale=init_scale),
          use_bias=False,
          out_dim=mlp_out_dim,
          precomputed_intermediates=mlp_precomputed_intermediates)
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
          layer_norm_factory=layer_norm_factory,
          mlp=mlp_factory(),
          self_attention=self_attention_factory())
      if parallel_fused_decoder_layer:
        init_kwargs['scanned'] = scan_layers
        return parallel_fused_decoder.ParallelFusedDecoderLayer(**init_kwargs)
      init_kwargs['encoder_decoder_attention'] = None
      init_kwargs[
          'shared_relative_position_bias'] = shared_relative_position_bias
      return t5_architecture.DecoderLayer(**init_kwargs)

    def decoder_factory(shared_token_embedder=None):
      init_kwargs = dict(
          dropout_factory=dropout_factory,
          dtype=activation_dtype,
          layer_factory=decoder_layer_factory,
          layer_norm_factory=final_layer_norm_factory,
          num_layers=num_decoder_layers,
          output_logits_factory=output_logits_factory
          if use_output_logits else None,
          position_embedder_factory=None,
          shared_relative_position_bias_factory=(
              relative_position_emb_factory if shared_relative_bias else None),
          token_embedder_factory=token_embedder_factory,
          shared_token_embedder=shared_token_embedder,
          scan_layers=scan_layers)
      return t5_architecture.Decoder(**init_kwargs)

    def decoder_only_factory():
      init_kwargs = dict(
          decoder_factory=decoder_factory,
          dtype=activation_dtype,
          scan_layers=scan_layers)
      return t5_architecture.DecoderOnly(**init_kwargs)

    flaxformer_decoder = flax_adapter.FlaxModuleAdapter.HParams(
        module_factory_method=decoder_only_factory,
        logical_axes_rules=p.logical_axes_rules)

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
    logical_axes_rules: Optional[LogicalAxisRules] = None

  def _build_wrapped_module(self) -> linen.Module:
    if self.hparams.encoder_decoder_factory is None:
      raise ValueError('encoder_decoder_factory is required!')
    return self.hparams.encoder_decoder_factory()

  def setup(self) -> None:
    super().setup()

    encoder_decoder_tpl = flax_adapter.EncoderDecoderFlaxModuleAdaptor.HParams(
        module_factory_method=self._build_wrapped_module,
        logical_axes_rules=self.hparams.logical_axes_rules)

    self.create_child('enc_dec', encoder_decoder_tpl)

  def __call__(self, *args, **kwargs):
    return self.enc_dec(*args, **kwargs)

  def encode(self, *args, **kwargs):
    return self.enc_dec.encode(*args, **kwargs)

  def decode(self, *args, **kwargs):
    return self.enc_dec.decode(*args, **kwargs)

  def compute_logits(self, *args, **kwargs):
    return self.enc_dec.compute_logits(*args, **kwargs)


class FactoryBasedEncoderDecoder(EncoderDecoder):
  """Legacy EncoderDecoder that exposes a few common settings.

  In general, we recommend using Fiddle to configure Flaxformer models; this
  allows deep overrides in model settings.
  """

  class HParams(EncoderDecoder.HParams):
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
      flax_decoder_tpl: Flaxformer decoder params.
      loss_normalizing_factor: Normalization factor for loss.
      label_smoothing: Amount of label smoothing to apply.
      z_loss: Coefficient for auxiliary z-loss loss term.
      decoding_fn: Decoding function used in autoregressive decoding.
      decoder_tpl: Parameterization of the autoregressive decoder.
    """
    flax_decoder_tpl: base_layer.BaseLayer.HParams = sub_config_field(
        FlaxFormerDecoder.HParams)
    loss_normalizing_factor: str = 'NUM_REAL_TARGET_TOKENS'
    label_smoothing: float = 0.0
    z_loss: float = 0.0001
    logical_axes_rules: Optional[LogicalAxisRules] = None
    decoding_fn: Optional[Callable[..., Any]] = t5x_decoding.temperature_sample
    decoder_tpl: DecoderHParams = pax_fiddle.sub_field(GreedyDecoderHParams)

  def setup(self):
    p = self.hparams
    self._decoding_fn = p.decoding_fn
    # Propagate partitioning information from BaseModel to BaseLayer.
    decoder_p = p.flax_decoder_tpl.clone()
    decoder_p.logical_axes_rules = p.logical_axes_rules
    self.create_child('decoder', decoder_p)

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
        If the input_batch comes from serving, there will be "ids" for the input
        tokens and "labels" for the target tokens.

    Returns:
      A NestedMap of predictions.
    """
    p = self.hparams
    get_elem = lambda x, k: x[k] if k in x else None
    decoder_input_tokens = (
        input_batch.ids
        if 'ids' in input_batch else input_batch.decoder_input_tokens)
    decoder_target_tokens = (
        input_batch.labels
        if 'labels' in input_batch else input_batch.decoder_target_tokens)
    logits = self.decoder(
        decoder_input_tokens=decoder_input_tokens,
        decoder_target_tokens=decoder_target_tokens,
        decoder_segment_ids=get_elem(input_batch, 'decoder_segment_ids'),
        decoder_positions=get_elem(input_batch, 'decoder_positions'))
    class_probabilities = jax.nn.one_hot(
        decoder_target_tokens,
        p.flax_decoder_tpl.num_embeddings,
        dtype=jnp.float32)
    class_probabilities = jax.lax.stop_gradient(class_probabilities)
    log_probs = jax.nn.log_softmax(logits)

    per_example_xent = -jnp.sum(
        log_probs * class_probabilities, axis=-1, dtype=jnp.float32)

    if 'weights' in input_batch:
      decoder_loss_weight = input_batch.weights
    elif 'decoder_loss_weight' in input_batch:
      decoder_loss_weight = input_batch.decoder_loss_weight
    else:
      decoder_loss_weight = jnp.ones_like(per_example_xent)
    per_token_xent = per_example_xent * decoder_loss_weight

    return NestedMap(logits=logits, per_token_xent=per_token_xent)

  def compute_loss(
      self, predictions: Predictions,
      input_batch: NestedMap) -> Tuple[WeightedScalars, Dict[str, Any]]:
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

  def decode(self, input_batch: NestedMap) -> DecodeOut:
    """Mimic `predict_batch_with_aux` function in t5x models.

    Predict with sample decode on a batch. Unlike
    `compute_predictions` that provides logits for training, this function
    provides predictions for inference.

    Args:
      input_batch: A NestedMap of an input batch. It should contain the
        following elements. "ids" - the prefix input tokens, of shape [batch,
        perfix_len]., "paddings", the paddings for input prefix.

    Returns:
      A 3-tuple with:
      - weighted scalars, a NestedMap containing str keys and (value, weight)
        pairs for the current batch (a tuple of two scalars).
      - results, a `.NestedMap` as decoder output.
      - metrics, a NestedMap containing str keys and clu_metrics.Metric
        objects. This is currently optional.
    """
    num_decodes = self.hparams.decoder_tpl.num_samples
    params = self.decoder.variables['params']
    decoder_params = {'eos_id': self.hparams.decoder_tpl.eos_id}
    max_decode_length = self.hparams.decoder_tpl.max_decode_steps

    # Prepare zeroed-out autoregressive cache.
    # [batch, input_len]
    batch_size, input_seq_len = input_batch.ids.shape
    inputs = jnp.pad(input_batch.ids, [[0, 0], [1, 0]])
    inputs = jax.lax.slice(inputs, [0, 0], [batch_size, input_seq_len])
    inputs = jnp.pad(inputs, [[0, 0], [0, max_decode_length]])
    inputs = jnp.repeat(inputs, repeats=num_decodes, axis=0)

    decoder_causal_attention = jnp.pad(
        1 - input_batch.paddings, [[0, 0], [1, 0]],
        'constant',
        constant_values=1).astype(jnp.int32)
    decoder_causal_attention = jax.lax.slice(decoder_causal_attention, [0, 0],
                                             [batch_size, input_seq_len])
    decoder_causal_attention = jnp.pad(decoder_causal_attention,
                                       [[0, 0], [0, max_decode_length]])
    decoder_causal_attention = jnp.repeat(
        decoder_causal_attention, repeats=num_decodes, axis=0)

    inputs_lengths = jnp.sum(decoder_causal_attention, axis=-1) - 1

    # Compute the key/value cache on the input prefix."""
    _, initial_variables = self.decoder.apply({'params': params},
                                              jnp.ones_like(inputs),
                                              jnp.ones_like(inputs),
                                              enable_dropout=False,
                                              decode=True,
                                              mutable=['cache'])
    cache = initial_variables['cache']
    if 'cache_axes' in initial_variables:
      cache_axes = initial_variables['cache_axes']

      cache = jax.tree_util.tree_map(
          flax_partitioning.with_sharding_constraint, cache,
          flax_partitioning.get_axis_names(cache_axes))

    # Initialize decode cache with prefix.
    _, variables_with_cache = self.decoder.apply(
        {
            'params': params,
            'cache': cache
        },
        decoder_input_tokens=inputs,
        decoder_target_tokens=decoder_causal_attention,
        decoder_causal_attention=None,
        mutable=['cache'],
        enable_dropout=False,
        prefill=True,
        prefill_lengths=inputs_lengths)
    prefilled_cache = variables_with_cache['cache']

    scanned = self.hparams.flax_decoder_tpl.scan_layers

    # Single step decoder function.
    tokens_ids_to_logits = functools.partial(
        self._compute_logits_from_slice,
        params=params,
        max_decode_length=max_decode_length)

    # Using the above-defined single-step decoder function, run a
    # sample_decode over possible sequences given input encoding.
    # decodes: [batch, num_decodes, max_decode_len + 1]
    # scores: [batch, num_decodes]
    decodes, scores = self._decoding_fn(
        inputs=inputs,
        cache=prefilled_cache,
        tokens_to_logits=tokens_ids_to_logits,
        num_decodes=1,
        cache_offset=1 if scanned else 0,
        topk=self.hparams.decoder_tpl.k,
        **decoder_params)

    eos_position = jnp.argmax(
        jnp.equal(decodes, decoder_params['eos_id']), axis=-1)
    decode_lengths = jnp.where(eos_position == 0,
                               jnp.ones_like(eos_position) * decodes.shape[-1],
                               eos_position + 1)
    decode_out = (NestedMap(
        num_decoded=(num_decodes, jnp.array(1, jnp.float32))),
                  NestedMap(
                      output_ids=jnp.reshape(decodes,
                                             (batch_size, num_decodes, -1)),
                      scores=jnp.reshape(scores, (batch_size, num_decodes)),
                      prefix_lengths=jnp.reshape(inputs_lengths + 1,
                                                 (batch_size, num_decodes)),
                      decode_lengths=jnp.reshape(decode_lengths,
                                                 (batch_size, num_decodes)),
                  ), None)
    return decode_out

  def _compute_logits_from_slice(
      self,
      decoding_state: t5x_decoding.DecodingState,
      params: PyTreeDef,
      max_decode_length: int,
  ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """This mimics _compute_logits_from_slice in t5x."""
    flat_ids = decoding_state.cur_token
    flat_cache = decoding_state.cache
    # flat_ids: [batch, seq_len=1]
    # flat_cache['cached_(keys|values)']:
    #   [batch, num_heads, depth_per_head, max_decode_length]
    # flat_cache['cache_index']: [batch]
    # flat_logits: [batch, seq_len=1, vocab]
    flat_logits, new_vars = self.decoder.apply(
        {
            'params': params,
            'cache': flat_cache
        },
        flat_ids,
        flat_ids,
        enable_dropout=False,
        decode=True,
        max_decode_length=max_decode_length,
        mutable=['cache'])
    # Remove sequence length dimension since it's always 1 during decoding.
    flat_logits = jnp.squeeze(flat_logits, axis=1)
    new_flat_cache = new_vars['cache']
    return flat_logits, new_flat_cache


class EncoderDecoderModel(base_model.BaseModel):
  """EncoderDecoder base task."""

  class HParams(base_model.BaseModel.HParams):
    """Associated hyperparams for this model class.

    Attributes:
      encoder_decoder_tpl: Flaxformer encoder decoder params.
      loss_normalizing_factor: Normalization factor for loss.
      label_smoothing: Amount of label smoothing to apply.
      z_loss: Coefficient for auxiliary z-loss loss term.
      decoding_fn: Decoding function to be used during the prediction. The
        default is t5x_decoding.beam_search.
    """
    encoder_decoder_tpl: base_layer.BaseLayer.HParams = sub_config_field(
        EncoderDecoder.HParams)
    loss_normalizing_factor: str = 'NUM_REAL_TARGET_TOKENS'
    label_smoothing: float = 0.0
    z_loss: float = 0.0001
    logical_axes_rules: Optional[LogicalAxisRules] = None
    decoding_fn: Optional[Callable[..., Any]] = t5x_decoding.beam_search

  def setup(self):
    p = self.hparams
    # Propagate partitioning information from BaseModel to BaseLayer.
    encoder_decoder_p = p.encoder_decoder_tpl.clone()
    encoder_decoder_p.logical_axes_rules = p.logical_axes_rules
    self._decoding_fn = p.decoding_fn
    self.create_child('encoder_decoder', encoder_decoder_p)

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

  def compute_loss(
      self, predictions: Predictions,
      input_batch: NestedMap) -> Tuple[WeightedScalars, Dict[str, Any]]:
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

  def decode(self, input_batch: NestedMap) -> DecodeOut:
    """Mimic `predict_batch_with_aux` function in t5x models.

    Predict with fast decoding beam search on a batch. Unlike
    `compute_predictions` that provides logits for training, this function
    provides predictions for inference.

    Args:
      input_batch: A NestedMap of an input batch. It should contain the
        following elements. "encoder_input_tokens" - the encoder input tokens,
        of shape [batch, encoder_len]. "decoder_input_tokens" - the decoder
        input tokens of shape [batch, target_len].

    Returns:
      A 3-tuple with:
      - weighted scalars, a NestedMap containing str keys and (value, weight)
        pairs for the current batch (a tuple of two scalars).
      - results, a `.NestedMap` as decoder output.
      - metrics, a NestedMap containing str keys and clu_metrics.Metric
        objects. This is currently optional.
    """
    num_decodes = 1  # We only have 1
    params = self.encoder_decoder.variables['params']
    decoder_params = {'eos_id': 1}

    # Prepare zeroed-out autoregressive cache.
    # [batch, input_len]
    inputs = input_batch['encoder_input_tokens']
    # [batch, target_len]
    target_shape = input_batch['decoder_input_tokens'].shape
    target_type = input_batch['decoder_input_tokens'].dtype

    _, variables_with_cache = self.encoder_decoder.apply(
        {'params': params},
        jnp.ones(inputs.shape, inputs.dtype),
        jnp.ones(target_shape, target_type),
        jnp.ones(target_shape, target_type),
        decode=True,
        enable_dropout=False,
        mutable=['cache'])

    cache = variables_with_cache['cache']

    # Prepare transformer fast-decoder call for beam search: for beam search, we
    # need to set up our decoder model to handle a batch size equal to
    # batch_size * num_decodes, where each batch item's data is expanded
    # in-place rather than tiled.
    # i.e. if we denote each batch element subtensor as el[n]:
    # [el0, el1, el2] --> beamsize=2 --> [el0,el0,el1,el1,el2,el2]
    # [batch * num_decodes, input_len, emb_dim]
    encoded_inputs = t5x_decoding.flat_batch_beam_expand(
        self.encoder_decoder.apply({'params': params},
                                   inputs,
                                   enable_dropout=False,
                                   method=self.encoder_decoder.encode),
        num_decodes)

    # [batch * num_decodes, input_len]
    raw_inputs = t5x_decoding.flat_batch_beam_expand(inputs, num_decodes)

    tokens_ids_to_logits = functools.partial(
        self._compute_logits_from_slice,
        params=params,
        encoded_inputs=encoded_inputs,
        raw_inputs=raw_inputs,
        max_decode_length=target_shape[1])

    # Currently, we do not support the prompt in the decoder.
    empty_decoder_prompt_inputs = jnp.zeros_like(
        input_batch['decoder_input_tokens'])

    # TODO(hwchung): rename the returned value names to more generic ones.
    # Using the above-defined single-step decoder function, run a
    # beam search over possible sequences given input encoding.
    # decodes: [batch, num_decodes, max_decode_len + 1]
    # scores: [batch, num_decodes]
    scanned = hasattr(self.encoder_decoder,
                      'scan_layers') and self.module.scan_layers

    decodes, scores = self._decoding_fn(
        inputs=empty_decoder_prompt_inputs,
        cache=cache,
        tokens_to_logits=tokens_ids_to_logits,
        num_decodes=num_decodes,
        cache_offset=1 if scanned else 0,
        **decoder_params)

    # Beam search returns [n_batch, n_beam, n_length] with beam dimension sorted
    # in increasing order of log-probability.
    # Return the highest scoring beam sequence.
    # pyformat: disable
    decode_out = (
        NestedMap(num_decoded=(num_decodes, jnp.array(1, jnp.float32))),
        NestedMap(output_ids=decodes[:, -1, :],
                  logprobs=scores[:, -1]),
        None)
    # pyformat: enable
    return decode_out

  def _compute_logits_from_slice(
      self, decoding_state: t5x_decoding.DecodingState, params: Any,
      encoded_inputs: jnp.ndarray, raw_inputs: jnp.ndarray,
      max_decode_length: int):
    """This mimics _compute_logits_from_slice in t5x."""
    params = self.encoder_decoder.variables['params']
    flat_ids = decoding_state.cur_token
    flat_cache = decoding_state.cache

    # flat_ids: [batch * beam, seq_len=1]
    # cache is expanded inside beam_search to become flat_cache
    # flat_cache: [batch * beam, num_heads, depth_per_head, max_decode_len]
    # flat_logits: [batch * beam, seq_len=1, vocab]
    flat_logits, new_vars = self.encoder_decoder.apply(
        {
            'params': params,
            'cache': flat_cache
        },
        encoded_inputs,
        raw_inputs,  # only needed for encoder padding mask
        flat_ids,
        flat_ids,
        enable_dropout=False,
        decode=True,
        max_decode_length=max_decode_length,
        mutable=['cache'],
        method=self.encoder_decoder.decode)

    # Remove sequence length dimension since it's always 1 during decoding.
    flat_logits = jnp.squeeze(flat_logits, axis=1)
    new_flat_cache = new_vars['cache']
    return flat_logits, new_flat_cache
