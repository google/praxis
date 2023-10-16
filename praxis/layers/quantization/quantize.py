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

"""Rewriter decorator(s) for quantizing models.

Those functions help creating quantized models/params for pax system.

Example usage:

class XYZModel():
  ...

@for_transformer()
class QuantizedXYZModel(XYZModel):
  pass

This creates a quantized model for the original XYZModel configuration by
quantizing all transformer blocks.

"""
import functools
from typing import Sequence, Type, cast

import fiddle as fdl
from jax import numpy as jnp
from praxis import base_layer
from praxis import layers
from praxis import pax_fiddle
from praxis.layers import quantization
from praxis.layers.quantization import quantization_hparams
from praxis.layers.quantization import utils

# Internal import for internal quantization hyper parameters.
# Internal import for internal quantization long seq support.

LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]
QuantizationParams = quantization_hparams.QuantizationParams
QuantizationType = quantization_hparams.QuantizationType
QuantizationMode = quantization_hparams.QuantizationMode
WeightQuantizationParams = quantization_hparams.WeightQuantizationParams
ActQuantizationParams = quantization_hparams.ActQuantizationParams


def _quantize_embedding_softmax_layer_weights(
    layer_tpl: pax_fiddle.Config[layers.TransformerLm]
    | pax_fiddle.Config[layers.TransformerEncoderDecoder],
    quantization_type: QuantizationType,
    mode: QuantizationMode,
    weight_quantization_params: WeightQuantizationParams,
    transposed_embedding_softmax: bool,
    softmax_only: bool = True,
) -> None:
  """Rewrites Embedding HParam for weight only quantization."""
  if transposed_embedding_softmax:
    # Transposed embedding quantization.
    # Replace softmax_tpl to quantized NClassMajorSharedEmbeddingSoftmax.
    quant_embedding_softmax_tpl = pax_fiddle.Config(
        quantization.NClassMajorSharedEmbeddingSoftmax,
        quantization=QuantizationParams(
            quantization_type=quantization_type,
            mode=mode,
            weight_params=weight_quantization_params,
        ),
    )
  else:
    # Non-transposed embedding quantization.
    # Replace softmax_tpl to quantized SharedEmbeddingSoftmax.
    quant_embedding_softmax_tpl = pax_fiddle.Config(
        quantization.SharedEmbeddingSoftmax,
        quantization=QuantizationParams(
            quantization_type=quantization_type,
            mode=mode,
            weight_params=weight_quantization_params,
        ),
    )

  def _quantize(name: str):
    embedding_softmax_tpl = getattr(layer_tpl, name)
    if embedding_softmax_tpl is None:
      return
    if issubclass(embedding_softmax_tpl.cls, layers.SharedEmbeddingSoftmax):
      new_embedding_softmax_tpl = quant_embedding_softmax_tpl.clone()
      new_embedding_softmax_tpl.copy_fields_from(embedding_softmax_tpl)
      setattr(layer_tpl, name, new_embedding_softmax_tpl)
    else:
      raise ValueError(
          f'layer_tpl.{name}.cls is : {embedding_softmax_tpl.cls}'
          'but has to be layers.SharedEmbeddingSoftmax'
      )

  _quantize('softmax_tpl')
  if not softmax_only:
    if issubclass(layer_tpl.cls, layers.TransformerLm):
      _quantize('separate_embedding_tpl')
    if issubclass(layer_tpl.cls, layers.TransformerEncoderDecoder):
      _quantize('encoder_embedding_tpl')
      _quantize('decoder_embedding_tpl')


def _quantize_ngrammer_embedding_weights(
    layer_tpl: pax_fiddle.Config[layers.TransformerLm]
    | pax_fiddle.Config[layers.TransformerEncoderDecoder],
    quantization_type: QuantizationType,
    mode: QuantizationMode,
    weight_quantization_params: WeightQuantizationParams,
) -> None:
  """Rewrites Ngrammer Embedding HParam for weight only quantization."""

  def _quantize(name: str):
    ngrammer_tpl = getattr(layer_tpl, name)
    if ngrammer_tpl is None:
      return
    if issubclass(ngrammer_tpl.cls, layers.ngrammer.Ngrammer):
      new_ngrammer_cls = quantization.Ngrammer
    elif issubclass(ngrammer_tpl.cls, layers.ngrammer.VQNgrammer):
      new_ngrammer_cls = quantization.VQNgrammer
    else:
      return

    new_ngrammer_tpl = pax_fiddle.Config(
        new_ngrammer_cls,
        quantization=QuantizationParams(
            quantization_type=quantization_type,
            mode=mode,
            weight_params=weight_quantization_params,
        ),
    )
    new_ngrammer_tpl.copy_fields_from(ngrammer_tpl)
    setattr(layer_tpl, name, new_ngrammer_tpl)

  if issubclass(layer_tpl.cls, layers.TransformerLm):
    _quantize('ngrammer_tpl')
  if issubclass(layer_tpl.cls, layers.TransformerEncoderDecoder):
    _quantize('encoder_ngrammer_tpl')
    _quantize('decoder_ngrammer_tpl')


# TODO(jianlijianli): mark quantize_* as private.
def quantize_transformer_layer_weights(
    tr_tpl: pax_fiddle.Config[layers.transformers.Transformer],
    quantization_type: QuantizationType,
    mode: QuantizationMode,
    weight_quantization_params: WeightQuantizationParams,
    act_quantization_params: ActQuantizationParams | None = None,
    linear_only: bool = False,
    rank: int = -1,
    quantize_self_attention: bool = True,
    quantize_cross_attention: bool = True,
) -> None:
  """Rewrites Transformer HParam for weight and act quantization."""
  if not linear_only:
    # If not linear only, quantize attentions (MHA / MQA) as well.
    quantize_attention_layer_weights(
        tr_tpl,
        quantization_type,
        mode,
        weight_quantization_params,
        act_quantization_params,
        quantize_self_attention,
        quantize_cross_attention,
    )
  # Always quantize linears layers.
  tr_fflayer_tpl = cast(
      pax_fiddle.Config[layers.transformers.TransformerFeedForward],
      tr_tpl.tr_fflayer_tpl,
  )
  quantize_transformer_feed_forward_layer_weights(
      tr_fflayer_tpl,
      quantization_type,
      mode,
      weight_quantization_params,
      act_quantization_params,
      rank,
  )


def quantize_attention_layer_weights(
    tr_tpl: pax_fiddle.Config[layers.transformers.Transformer],
    quantization_type: QuantizationType,
    mode: QuantizationMode,
    weight_quantization_params: WeightQuantizationParams,
    act_quantization_params: ActQuantizationParams | None = None,
    quantize_self_attention: bool = True,
    quantize_cross_attention: bool = True,
) -> None:
  """Rewrites Attention HParam for weight and act quantization."""

  def _quantize(name: str):
    atten_tpl = getattr(tr_tpl, name)
    if atten_tpl is None:
      return

    if issubclass(atten_tpl.cls, layers.attentions.DotProductAttention):
      quantize_dot_product_attention_layer_weights(
          atten_tpl,
          quantization_type,
          mode,
          weight_quantization_params,
          act_quantization_params,
      )
    elif (
        issubclass(
            atten_tpl.cls,
            layers.multi_query_attention.MultiQueryDotProductAttention,
        )
        # Internal quantization long seq support.
    ):
      quantize_mq_dot_product_attention_layer_weights(
          atten_tpl,
          quantization_type,
          mode,
          weight_quantization_params,
          act_quantization_params,
      )
    else:
      raise ValueError(
          f'tr_tpl.{name}.cls is : {atten_tpl.cls}'
          'but has to be DotProductAttention or MultiQueryDotProductAttention'
      )

  if quantize_cross_attention != quantize_self_attention:
    if tr_tpl.cross_atten_tpl is None:
      # Note that we do not depends on the value of tr_tpl.use_cross_attention
      # to decide whether a copy needs to be made, because this value is not
      # reliable before running layer initialization. If this value is
      # eventually not being set, e.g. in the case of transformer encoder, then
      # this copy operation is a noop, i.e. no cross attention layer will
      # be created.
      tr_tpl.cross_atten_tpl = tr_tpl.tr_atten_tpl.clone()

  if quantize_self_attention:
    _quantize('tr_atten_tpl')
  if quantize_cross_attention:
    _quantize('cross_atten_tpl')


def quantize_dot_product_attention_layer_weights(
    attn_tpl: pax_fiddle.Config[layers.attentions.DotProductAttention],
    quantization_type: QuantizationType,
    mode: QuantizationMode,
    weight_quantization_params: WeightQuantizationParams,
    act_quantization_params: ActQuantizationParams | None = None,
) -> None:
  """Rewrites DotProductAttention HParam for weight only quantization."""

  quant_proj_tpl = pax_fiddle.Config(
      quantization.AttentionProjection,
      quantization=QuantizationParams(
          quantization_type=quantization_type,
          mode=mode,
          act_params=act_quantization_params,
          weight_params=weight_quantization_params,
      ),
  )
  if issubclass(attn_tpl.proj_tpl.cls, layers.attentions.AttentionProjection):
    quant_proj_tpl.copy_fields_from(attn_tpl.proj_tpl)
  attn_tpl.proj_tpl = quant_proj_tpl

  quant_combined_qkv_proj_tpl = pax_fiddle.Config(
      quantization.attentions.CombinedQKVProjectionLayer,
      quantization=QuantizationParams(
          quantization_type=quantization_type,
          mode=mode,
          act_params=act_quantization_params,
          weight_params=weight_quantization_params,
      ),
  )
  if issubclass(
      attn_tpl.combined_qkv_proj_tpl.cls,
      layers.attentions.CombinedQKVProjectionLayer,
  ):
    quant_combined_qkv_proj_tpl.copy_fields_from(attn_tpl.combined_qkv_proj_tpl)
  attn_tpl.combined_qkv_proj_tpl = quant_combined_qkv_proj_tpl


def quantize_mq_dot_product_attention_layer_weights(
    attn_tpl: pax_fiddle.Config[
        layers.multi_query_attention.MultiQueryDotProductAttention
    ],
    quantization_type: QuantizationType,
    mode: QuantizationMode,
    weight_quantization_params: WeightQuantizationParams,
    act_quantization_params: ActQuantizationParams | None = None,
) -> None:
  """Rewrites MultiQueryDotProductAttention HParam."""

  quant_proj_tpl = pax_fiddle.Config(
      quantization.attentions.AttentionProjection,
      quantization=QuantizationParams(
          quantization_type=quantization_type,
          mode=mode,
          act_params=act_quantization_params,
          weight_params=weight_quantization_params,
      ),
  )
  if issubclass(attn_tpl.proj_tpl.cls, layers.attentions.AttentionProjection):
    quant_proj_tpl.copy_fields_from(attn_tpl.proj_tpl)
  attn_tpl.proj_tpl = quant_proj_tpl

  quant_headless_proj_tpl = pax_fiddle.Config(
      quantization.multi_query_attention.OneHeadedAttentionProjection,
      quantization=QuantizationParams(
          quantization_type=quantization_type,
          mode=mode,
          act_params=act_quantization_params,
          weight_params=weight_quantization_params,
      ),
  )
  if issubclass(
      attn_tpl.headless_proj_tpl.cls,
      layers.multi_query_attention.OneHeadedAttentionProjection,
  ):
    quant_headless_proj_tpl.copy_fields_from(attn_tpl.headless_proj_tpl)
  attn_tpl.headless_proj_tpl = quant_headless_proj_tpl


def quantize_transformer_feed_forward_layer_weights(
    tr_fflayer_tpl: pax_fiddle.Config[
        layers.transformers.TransformerFeedForward
    ],
    quantization_type: QuantizationType,
    mode: QuantizationMode,
    weight_quantization_params: WeightQuantizationParams,
    act_quantization_params: ActQuantizationParams | None = None,
    rank: int = -1,
) -> None:
  """Rewrites TransformerFeedForward HParam for weight only quantization."""

  tr_fflayer_tpl.fflayer_tpl.linear_tpl = pax_fiddle.Config(
      quantization.Linear,
      quantization=QuantizationParams(
          quantization_type=quantization_type,
          mode=mode,
          act_params=act_quantization_params,
          weight_params=weight_quantization_params,
      ),
      rank=rank,
  )


# TODO(jianlijianli): Add decorator for other model architectures.
# Ready-to-use quantization decorators for quantizing transformer.
def for_transformer(
    num_bits: int = 8,
    quantization_type: QuantizationType = QuantizationType.FQ,
    mode: QuantizationMode = QuantizationMode.TRAINING,
    use_symmetric: bool = True,
    rank: int = -1,
    *,
    weight_quant_only: bool = True,
    quantize_embedding_softmax: bool = False,
    transposed_embedding_softmax: bool = False,
    softmax_only: bool = True,
    quantize_ngrammer_embedding: bool = False,
    linear_only: bool = False,
    quantize_self_attention: bool = True,
    quantize_cross_attention: bool = True,
    dtype: jnp.dtype = jnp.int8,
    quantize_init_from_checkpoint_rules_task: bool = False,
    block_size: int = 0,
    # Internal quantization parameters.
    num_bits_act: int | None = None,
    use_symmetric_act: bool | None = None,
):
  """Find and quantize transformer.

  If there are transformers that shouldn't be quantized, use lowers level APIs
  and manually/selectively quantize the model.

  If there are no transformers in the model, it's a no-op.

  TODO(jianlijianli): pass in additional parameters.

  Args:
    num_bits: Number of bits for quantized weight. Currently supports 8 and 4
      but any integer [1, 8] works.
    quantization_type: Indicates the quantization type among PTQ, FQ, and AQT.
    mode: Indicates the quantization mode. Only TRAINING and INFERENCE
      (excluding MATERIALIZE) are valid for non-servable models.
    use_symmetric: If true, do symmetric quantization for weights, otherwise
      asymmetric quantization.
    rank: If positive, factorize weight matrix for linear layers to two [in_dim,
      rank], [rank, out_dim] matrices.
    weight_quant_only: If true, quantize weight only, otherweise quantize both
      weight and activation except that softmax, embedding, Ngrammer/VQNgrammer
      layer only support weight quantization regardless of this option.
    quantize_embedding_softmax: If true, Quantize embedding table of embedding
      softmax layer. This applies to both softmax and embedding layers unless
      softmax_only is set to True.
    transposed_embedding_softmax: If the model is using transposed embedding for
      embedding softmax layer. This applies to both softmax and embedding layers
      unless softmax_only is set to True.
    softmax_only: Only quantize softmax layers and leave embedding layers
      untouched. This option only works if softmax and embedding layers are not
      sharing the same weights. This argument does not impact whether layers
      other than softmax and embedding are quantized or not.
    quantize_ngrammer_embedding: Quantize embedding table of each embedding in
      Ngrammer/VQNgrammer layer.
    linear_only: If True, quantize only the linear layers inside the transforemr
      layer. If False, linear layers inside the transformer layer are still
      quantized, self attention and cross attention layers inside the
      transformer layer maybe quantized. This argument does not impact whether
      layers outside of transformer layer are quantized or not.
    quantize_self_attention: Quantize the self attention layer inside the
      transformer layer. Must set linear_only to false to take effect.
    quantize_cross_attention: Quantize the cross attention layer inside the
      transformer layer. Must set linear_only to false to take effect.
    dtype: Dtype of the quantized variables.
    quantize_init_from_checkpoint_rules_task: Apply quantization to the tasks
      that are defined in task_p.train.init_from_checkpoint_rules.values()
    block_size: block size for sub-channel quantization. Defaults to 0, which
      means off.
    num_bits_act: The number of bits used for activation quantization. Only
      valid when weight_quant_only is false.
    use_symmetric_act: Use symmetric activation quantization.Only valid when
      weight_quant_only is false.

  Returns:
    A modifier that quantizes transformers when applied to a config.
  """

  def decorator(cls):
    """decorator that quantize transformers."""

    @functools.wraps(cls, updated=())  # to keep original class name.
    class Wrapper(cls):
      """Wrapper class for cls with Quantization enabled."""

      def task(self):
        config = super()
        task_p = config.task()
        assert num_bits in [2, 4, 8]
        models = [task_p.model]
        if (
            quantize_init_from_checkpoint_rules_task
            and hasattr(task_p, 'train')
            and hasattr(task_p.train, 'init_from_checkpoint_rules')
        ):
          for _, ckpt_rules in task_p.train.init_from_checkpoint_rules.items():
            models.append(ckpt_rules.task_p.model)
        for model in models:
          set_transformer_quantization(
              model,
              quantization_type=quantization_type,
              mode=mode,
              num_bits=num_bits,
              linear_only=linear_only,
              use_symmetric=use_symmetric,
              rank=rank,
              weight_quant_only=weight_quant_only,
              quantize_embedding_softmax=quantize_embedding_softmax,
              transposed_embedding_softmax=transposed_embedding_softmax,
              quantize_ngrammer_embedding=quantize_ngrammer_embedding,
              dtype=dtype,
              block_size=block_size,
              # Pass internal quantization parameters.
              quantize_self_attention=quantize_self_attention,
              quantize_cross_attention=quantize_cross_attention,
              softmax_only=softmax_only,
              use_symmetric_act=use_symmetric_act,
              num_bits_act=num_bits_act,
          )
        return task_p

    return Wrapper

  return decorator


# Ready-to-use quantization decorators for quantizing diffusion.
def for_diffusion(
    target: Type[base_layer.BaseLayer],
    num_bits: int = 8,
    quantization_type: QuantizationType = QuantizationType.FQ,
    mode: QuantizationMode = QuantizationMode.TRAINING,
    use_symmetric: bool = True,
    dtype: jnp.dtype = jnp.int8,
    weight_quant_only: bool = True,
    quantize_init_from_checkpoint_rules_task: bool = False,
):
  """Find and quantize Unet.

  If there are diffusion that shouldn't be quantized, use lowers level APIs
  and manually/selectively quantize the model.

  If there are no diffusion in the model, it's a no-op.

  TODO(jianlijianli): pass in additional parameters.

  Args:
    target: Target tpl,
    num_bits: Number of bits for quantized weight. Currently supports 8 and 4
      but any integer [1, 8] works.
    quantization_type: Indicates the quantization type among PTQ, FQ, and AQT.
    mode: Indicates the quantization mode. Only TRAINING and INFERENCE
      (excluding MATERIALIZE) are valid for non-servable models.
    use_symmetric: If true, do symmetric quantization for weights, otherwise
      asymmetric quantization.
    dtype: Dtype of the quantized variables.
    weight_quant_only: If true, quantize weight only, otherweise quantize both
      weight and activation.
    quantize_init_from_checkpoint_rules_task: Apply quantization to the tasks
      that are defined in task_p.train.init_from_checkpoint_rules.values()

  Returns:
    A modifier that quantizes diffusion when applied to a config.
  """

  def decorator(cls):
    """decorator that quantize diffusions."""

    @functools.wraps(cls, updated=())  # to keep original class name.
    class Wrapper(cls):
      """Wrapper class for cls with Quantization enabled."""

      def task(self):
        config = super()
        task_p = config.task()
        assert num_bits in [2, 4, 8]
        models = [task_p.model]
        if (
            quantize_init_from_checkpoint_rules_task
            and hasattr(task_p, 'train')
            and hasattr(task_p.train, 'init_from_checkpoint_rules')
        ):
          for _, ckpt_rules in task_p.train.init_from_checkpoint_rules.items():
            models.append(ckpt_rules.task_p.model)
        for model in models:
          set_diffusion_quantization(
              model,
              target,
              quantization_type=quantization_type,
              mode=mode,
              num_bits=num_bits,
              use_symmetric=use_symmetric,
              weight_quant_only=weight_quant_only,
              dtype=dtype,
          )
        return task_p

    return Wrapper

  return decorator


def set_transformer_quantization(
    config: LayerTpl,
    quantization_type: QuantizationType = QuantizationType.PTQ,
    mode: QuantizationMode = QuantizationMode.INFERENCE,
    num_bits: int = 8,
    linear_only: bool = False,
    use_symmetric: bool = True,
    rank: int = -1,
    *,
    weight_quant_only: bool = True,
    quantize_self_attention: bool = True,
    quantize_cross_attention: bool = True,
    quantize_embedding_softmax: bool = False,
    transposed_embedding_softmax: bool = False,
    softmax_only: bool = True,
    quantize_ngrammer_embedding: bool = False,
    dtype: jnp.dtype = jnp.int8,
    block_size: int = 0,
    use_int4_packed_weights: bool = True,
    int4_packed_weights_container_dtype: jnp.dtype = jnp.int32,
    # Internal quantization parameters.
    num_bits_act: int | None = None,
    use_symmetric_act: bool | None = None,
):
  """Sets quantization params for TransformerLm or TransformerEncoderDecoder.

  Args:
    config: The config to apply quantization on.
    quantization_type: The quantization types (PTQ, FQ, AQT etc)
    mode: The quantization modes (INFERENCE, TRAINING, MATERIALIZE etc)
    num_bits: The number of bits used for quantization.
    linear_only: If True, quantize only the linear layers inside the transforemr
      layer. If False, linear layers inside the transformer layer are still
      quantized, self attention and cross attention layers inside the
      transformer layer maybe quantized. This argument does not impact whether
      layers outside of transformer layer are quantized or not.
    use_symmetric: Use symmetric weight quantization.
    rank: If positive, factorize weight matrix for linear layers to two [in_dim,
      rank], [rank, out_dim] matrices.
    weight_quant_only: If true, quantize weight only, otherweise quantize both
      weight and activation except that softmax, embedding, Ngrammer/VQNgrammer
      layer only support weight quantization regardless of this option.
    quantize_self_attention: Quantize the self attention layer inside the
      transformer layer. Must set linear_only to false to take effect.
    quantize_cross_attention: Quantize the cross attention layer inside the
      transformer layer. Must set linear_only to false to take effect.
    quantize_embedding_softmax: If true, Quantize embedding table of embedding
      softmax layer. This applies to both softmax and embedding layers unless
      softmax_only is set to True.
    transposed_embedding_softmax: If the model is using transposed embedding for
      embedding softmax layer. This applies to both softmax and embedding layers
      unless softmax_only is set to True.
    softmax_only: Only quantize softmax layers and leave embedding layers
      untouched. This option only works if softmax and embedding layers are not
      sharing the same weights. This argument does not impact whether layers
      other than softmax and embedding are quantized or not.
    quantize_ngrammer_embedding: If true, Quantize embedding table of each
      embedding in Ngrammer/VQNgrammer layer. This rewrites
      TransformerLm.ngrammer_tpl in `config`.
    dtype: Dtype of the quantized variables.
    block_size: Block size for sub-channel quantization. Defaults to 0.
    use_int4_packed_weights: If True, pack/unpack int4 weights into int32 or
      int8. It is for int4 weights only and has not effect on other type. If
      False int4 weights will be kept in int8.
    int4_packed_weights_container_dtype: Container type for int4 weights: int32
      to pack 8 int4s, or int8 to pack 2 int4s.
    num_bits_act: The number of bits used for activation quantization. Only
      valid when weight_quant_only is false.
    use_symmetric_act: Use symmetric activation quantization. Only valid when
      weight_quant_only is false.
  """
  weight_quantization_params = WeightQuantizationParams(
      precision=num_bits,
      use_symmetric=use_symmetric,
      dtype=dtype,
      block_size=block_size,
      use_int4_packed_weights=use_int4_packed_weights,
      int4_packed_weights_container_dtype=int4_packed_weights_container_dtype,
      # Pass internal quantization parameters.
  )
  act_quantization_params = None
  if (
      num_bits_act is not None or use_symmetric_act is not None
  ) and weight_quant_only:
    raise ValueError(
        f'Activation quantization params (`num_bits_act` and'
        f' `use_symmetric_act`) should not be set when `weight_quant_only` is'
        f' set to True.'
    )
  if not weight_quant_only:
    if num_bits_act == None or use_symmetric_act == None:
      raise ValueError(
          f'Activation quantization params (`num_bits_act` and'
          f' `use_symmetric_act`) have to be set when  `weight_quant_only` is'
          f' set to false.'
      )
    act_quantization_params = ActQuantizationParams(
        precision=num_bits_act,
        symmetric=use_symmetric_act,
    )

  transformer_tpls = utils.find_target_tpl(
      config, layers.transformers.Transformer
  )
  for transformer_tpl in transformer_tpls:
    quantize_transformer_layer_weights(
        transformer_tpl,
        quantization_type,
        mode,
        weight_quantization_params,
        act_quantization_params,
        linear_only,
        rank,
        quantize_self_attention,
        quantize_cross_attention,
    )  # pytype: disable=wrong-arg-types  # py310-upgrade

  if quantize_embedding_softmax or quantize_ngrammer_embedding:
    lm_or_encdec_tpls = utils.find_target_tpl(
        config, [layers.TransformerLm, layers.TransformerEncoderDecoder]
    )
    for lm_or_encdec_tpl in lm_or_encdec_tpls:
      if quantize_embedding_softmax:
        _quantize_embedding_softmax_layer_weights(
            lm_or_encdec_tpl,
            quantization_type,
            mode,
            weight_quantization_params,
            transposed_embedding_softmax,
            softmax_only,
        )  # pytype: disable=wrong-arg-types  # py310-upgrade
      if quantize_ngrammer_embedding:
        _quantize_ngrammer_embedding_weights(
            lm_or_encdec_tpl,
            quantization_type,
            mode,
            weight_quantization_params,
        )  # pytype: disable=wrong-arg-types  # py310-upgrade


def set_diffusion_quantization(
    config: LayerTpl,
    target: Type[base_layer.BaseLayer],
    quantization_type: QuantizationType = QuantizationType.PTQ,
    mode: QuantizationMode = QuantizationMode.INFERENCE,
    num_bits: int = 8,
    use_symmetric: bool = True,
    weight_quant_only: bool = True,
    dtype: jnp.dtype = jnp.int8,
):
  """Sets quantization parameters for Diffusion in 'config'.

  Args:
    config: The config to apply quantization on.
    target: The target tpl.
    quantization_type: The quantization types (PTQ, FQ, AQT etc)
    mode: The quantization modes (INFERENCE, TRAINING, MATERIALIZE etc)
    num_bits: The number of bits used for quantization.
    use_symmetric: Use symmetric weight quantization.
    weight_quant_only: If true, quantize weight only, otherweise quantize both
      weight and activation.
    dtype: Dtype of the quantized variables.
  """
  weight_quantization_params = WeightQuantizationParams(
      precision=num_bits,
      use_symmetric=use_symmetric,
      dtype=dtype,
  )
  act_quantization_params = (
      None if weight_quant_only else ActQuantizationParams(precision=num_bits)
  )

  diffusion_tpls = utils.find_target_tpl(config, target)
  for diffusion_tpl in diffusion_tpls:
    if hasattr(diffusion_tpl, 'conv_tpl'):
      diffusion_tpl.conv_tpl = pax_fiddle.Config(
          quantization.Conv2D,
          quantization=QuantizationParams(
              quantization_type=quantization_type,
              mode=mode,
              act_params=act_quantization_params,
              weight_params=weight_quantization_params,
          ),
      )


def set_inference_mode(
    config: LayerTpl,
):
  """Sets quantization mode to be INFERENCE while keeping other quantization configs unchanged.

  Args:
    config: The config to apply quantization on.
  """
  def set_quantization_mode_inference(tpl):
    if hasattr(tpl, 'quantization'):
      tpl.quantization.mode = QuantizationMode.INFERENCE

  to_process = [config]
  while to_process:
    param = to_process.pop()
    params = param if isinstance(param, Sequence) else [param]
    for param_elem in params:
      set_quantization_mode_inference(param_elem)
      if isinstance(param_elem, fdl.Config):
        to_process.extend(fdl.ordered_arguments(param_elem).values())
