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
from typing import cast, Optional, Type, Sequence
import fiddle as fdl
from jax import numpy as jnp
from praxis import base_layer
from praxis import layers
from praxis import pax_fiddle
from praxis.layers import quantization
from praxis.layers.quantization import quantization_hparams

LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]
QuantizationHParams = quantization_hparams.QuantizationHParams
QuantizationType = quantization_hparams.QuantizationType
QuantizationMode = quantization_hparams.QuantizationMode
WeightQuantizationParams = quantization_hparams.WeightQuantizationParams
ActQuantizationParams = quantization_hparams.ActQuantizationParams


def _quantize_embedding_softmax_layer_weights(
    lm_tpl: pax_fiddle.Config[layers.TransformerLm],
    quantization_type: QuantizationType,
    mode: QuantizationMode,
    weight_quantization_params: WeightQuantizationParams,
    transposed_embedding_softmax: bool,
) -> None:
  """Rewrites Embedding HParam for weight only quantization."""
  if issubclass(lm_tpl.softmax_tpl.cls, layers.SharedEmbeddingSoftmax):
    if transposed_embedding_softmax:
      # Transposed embedding quantization.
      # Replace softmax_tpl to quantized NClassMajorSharedEmbeddingSoftmax.
      new_softmax_tpl = pax_fiddle.Config(
          quantization.NClassMajorSharedEmbeddingSoftmax,
          quantization=QuantizationHParams(
              quantization_type=quantization_type,
              mode=mode,
              weight_params=weight_quantization_params,
          ),
      )
    else:
      # Non-transposed embedding quantization.
      # Replace softmax_tpl to quantized SharedEmbeddingSoftmax.
      new_softmax_tpl = pax_fiddle.Config(
          quantization.SharedEmbeddingSoftmax,
          quantization=QuantizationHParams(
              quantization_type=quantization_type,
              mode=mode,
              weight_params=weight_quantization_params,
          ),
      )
    new_softmax_tpl.copy_fields_from(lm_tpl.softmax_tpl)
    lm_tpl.softmax_tpl = new_softmax_tpl


def _quantize_ngrammer_embedding_weights(
    lm_tpl: pax_fiddle.Config[layers.TransformerLm],
    quantization_type: QuantizationType,
    mode: QuantizationMode,
    weight_quantization_params: WeightQuantizationParams,
) -> None:
  """Rewrites Ngrammer Embedding HParam for weight only quantization."""
  if lm_tpl.ngrammer_tpl is None:
    return
  if issubclass(lm_tpl.ngrammer_tpl.cls, layers.ngrammer.Ngrammer):
    new_ngrammer_cls = quantization.Ngrammer
  elif issubclass(lm_tpl.ngrammer_tpl.cls, layers.ngrammer.VQNgrammer):
    new_ngrammer_cls = quantization.VQNgrammer
  else:
    return

  new_ngrammer_tpl = pax_fiddle.Config(
      new_ngrammer_cls,
      quantization=QuantizationHParams(
          quantization_type=quantization_type,
          mode=mode,
          weight_params=weight_quantization_params,
      ),
  )
  new_ngrammer_tpl.copy_fields_from(lm_tpl.ngrammer_tpl)
  lm_tpl.ngrammer_tpl = new_ngrammer_tpl


# TODO(jianlijianli): mark quantize_* as private.
def quantize_transformer_layer_weights(
    tr_tpl: pax_fiddle.Config[layers.transformers.Transformer],
    quantization_type: QuantizationType,
    mode: QuantizationMode,
    weight_quantization_params: WeightQuantizationParams,
    act_quantization_params: Optional[ActQuantizationParams] = None,
    linear_only: bool = False,
) -> None:
  """Rewrites Transformer HParam for weight only quantization."""
  if not linear_only:
    # If not linear only, quantize attentions (MHA / MQA) as well.
    if issubclass(
        tr_tpl.tr_atten_tpl.cls, layers.attentions.DotProductAttention
    ):
      tr_atten_tpl = cast(
          pax_fiddle.Config[layers.attentions.DotProductAttention],
          tr_tpl.tr_atten_tpl,
      )
      quantize_dot_product_attention_layer_weights(
          tr_atten_tpl,
          quantization_type,
          mode,
          weight_quantization_params,
          act_quantization_params,
      )

    if issubclass(
        tr_tpl.tr_atten_tpl.cls,
        layers.multi_query_attention.MultiQueryDotProductAttention,
    ):
      tr_atten_tpl = cast(
          pax_fiddle.Config[
              layers.multi_query_attention.MultiQueryDotProductAttention
          ],
          tr_tpl.tr_atten_tpl,
      )
      quantize_mq_dot_product_attention_layer_weights(
          tr_atten_tpl,
          quantization_type,
          mode,
          weight_quantization_params,
          act_quantization_params,
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
  )


def quantize_dot_product_attention_layer_weights(
    attn_tpl: pax_fiddle.Config[layers.attentions.DotProductAttention],
    quantization_type: QuantizationType,
    mode: QuantizationMode,
    weight_quantization_params: WeightQuantizationParams,
    act_quantization_params: Optional[ActQuantizationParams] = None,
) -> None:
  """Rewrites DotProductAttention HParam for weight only quantization."""

  attn_tpl.proj_tpl = pax_fiddle.Config(
      quantization.AttentionProjection,
      quantization=QuantizationHParams(
          quantization_type=quantization_type,
          mode=mode,
          act_params=act_quantization_params,
          weight_params=weight_quantization_params,
      ),
  )

  attn_tpl.combined_qkv_proj_tpl = pax_fiddle.Config(
      quantization.attentions.CombinedQKVProjectionLayer,
      quantization=QuantizationHParams(
          quantization_type=quantization_type,
          mode=mode,
          act_params=act_quantization_params,
          weight_params=weight_quantization_params,
      ),
  )


def quantize_mq_dot_product_attention_layer_weights(
    attn_tpl: pax_fiddle.Config[
        layers.multi_query_attention.MultiQueryDotProductAttention
    ],
    quantization_type: QuantizationType,
    mode: QuantizationMode,
    weight_quantization_params: WeightQuantizationParams,
    act_quantization_params: Optional[ActQuantizationParams] = None,
) -> None:
  """Rewrites MultiQueryDotProductAttention HParam."""

  attn_tpl.proj_tpl = pax_fiddle.Config(
      quantization.attentions.AttentionProjection,
      quantization=QuantizationHParams(
          quantization_type=quantization_type,
          mode=mode,
          act_params=act_quantization_params,
          weight_params=weight_quantization_params,
      ),
  )

  attn_tpl.headless_proj_tpl = pax_fiddle.Config(
      quantization.multi_query_attention.OneHeadedAttentionProjection,
      quantization=QuantizationHParams(
          quantization_type=quantization_type,
          mode=mode,
          act_params=act_quantization_params,
          weight_params=weight_quantization_params,
      ),
  )


def quantize_transformer_feed_forward_layer_weights(
    tr_fflayer_tpl: pax_fiddle.Config[
        layers.transformers.TransformerFeedForward
    ],
    quantization_type: QuantizationType,
    mode: QuantizationMode,
    weight_quantization_params: WeightQuantizationParams,
    act_quantization_params: Optional[ActQuantizationParams] = None,
) -> None:
  """Rewrites TransformerFeedForward HParam for weight only quantization."""

  tr_fflayer_tpl.fflayer_tpl.linear_tpl = pax_fiddle.Config(
      quantization.Linear,
      quantization=QuantizationHParams(
          quantization_type=quantization_type,
          mode=mode,
          act_params=act_quantization_params,
          weight_params=weight_quantization_params,
      ),
  )


# TODO(jianlijianli): Add decorator for other model architectures.
# Ready-to-use quantization decorators for quantizing transformer.
def for_transformer(
    num_bits: int = 8,
    quantization_type: QuantizationType = QuantizationType.FQ,
    use_symmetric: bool = True,
    *,
    weight_quant_only: bool = True,
    quantize_embedding_softmax: bool = False,
    transposed_embedding_softmax: bool = False,
    quantize_ngrammer_embedding: bool = False,
    linear_only: bool = False,
    dtype: jnp.dtype = jnp.int8,
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
    use_symmetric: If true, do symmetric quantization for weights, otherwise
      asymmetric quantization.
    weight_quant_only: If true, quantize weight only, otherweise quantize both
      weight and activation.
    quantize_embedding_softmax: Quantize embedding table of embedding softmax
      layer.
    transposed_embedding_softmax: If the model is using transposed embedding for
      embedding softmax layer.
    quantize_ngrammer_embedding: Quantize embedding table of each embedding in
      Ngrammer/VQNgrammer layer.
    linear_only: quantize only linear layer.
    dtype: Dtype of the quantized variables.

  Returns:
    A modifier that quantizes transformers when applied to a config.
  """

  def decorator(cls):
    """decorator that quantize transformers."""

    @functools.wraps(cls, updated=())  # to keep original class name.
    class Wrapper(cls):
      """Wrapper class for cls with Quantization enabled."""

      def task(self):
        config = super(Wrapper, self)
        task_p = config.task()
        mode = QuantizationMode.TRAINING
        assert num_bits in [2, 4, 8]
        set_quantization(
            task_p.model,
            layers.transformers.Transformer,
            quantization_type=quantization_type,
            mode=mode,
            num_bits=num_bits,
            linear_only=linear_only,
            use_symmetric=use_symmetric,
            weight_quant_only=weight_quant_only,
            quantize_embedding_softmax=quantize_embedding_softmax,
            transposed_embedding_softmax=transposed_embedding_softmax,
            quantize_ngrammer_embedding=quantize_ngrammer_embedding,
            dtype=dtype,
        )
        return task_p

    return Wrapper

  return decorator


def set_quantization(
    config: LayerTpl,
    target: Type[base_layer.BaseLayer] = layers.transformers.Transformer,
    quantization_type: QuantizationType = QuantizationType.PTQ,
    mode: QuantizationMode = QuantizationMode.INFERENCE,
    num_bits: int = 8,
    linear_only: bool = False,
    use_symmetric: bool = True,
    *,
    weight_quant_only: bool = True,
    quantize_embedding_softmax: bool = False,
    transposed_embedding_softmax: bool = False,
    quantize_ngrammer_embedding: bool = False,
    dtype: jnp.dtype = jnp.int8,
):
  """Sets quantization parameters for 'target' in 'config'.

  NOTE: If `quantize_embedding_softmax` is True, this rewrites
  TransformerLm.softmax_tpl
  if it is in `config`, regardless of `target` argument.

  Args:
    config: The config to apply quantization on.
    target: The target component to be replaced.
    quantization_type: The quantization types (PTQ, FQ, AQT etc)
    mode: The quantization modes (INFERENCE, TRAINING, MATERIALIZE etc)
    num_bits: The number of bits used for quantization.
    linear_only: Quantize only the linear layers.
    use_symmetric: Use symmetric weight quantization.
    weight_quant_only: If true, quantize weight only, otherweise quantize both
      weight and activation.
    quantize_embedding_softmax: If true, Quantize embedding table of embedding
      softmax layer. Regardless of `target` argument, this results in rewriting
      TransformerLm.softmax_tpl in `config`.
    transposed_embedding_softmax: If the model is using transposed embedding for
      embedding softmax layer.
    quantize_ngrammer_embedding: If true, Quantize embedding table of each
      embedding in Ngrammer/VQNgrammer layer. Regardless of `target` argument,
      this results in rewriting TransformerLm.ngrammer_tpl in `config`.
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

  target_tpls = find_target_tpl(config, target)
  for target_tpl in target_tpls:
    quantize_transformer_layer_weights(
        target_tpl,
        quantization_type,
        mode,
        weight_quantization_params,
        act_quantization_params,
        linear_only,
    )  # pytype: disable=wrong-arg-types  # py310-upgrade

  if quantize_embedding_softmax or quantize_ngrammer_embedding:
    lm_tpls = find_target_tpl(config, layers.TransformerLm)
    for lm_tpl in lm_tpls:
      if quantize_embedding_softmax:
        _quantize_embedding_softmax_layer_weights(
            lm_tpl,
            quantization_type,
            mode,
            weight_quantization_params,
            transposed_embedding_softmax,
        )  # pytype: disable=wrong-arg-types  # py310-upgrade
      if quantize_ngrammer_embedding:
        _quantize_ngrammer_embedding_weights(
            lm_tpl,
            quantization_type,
            mode,
            weight_quantization_params,
        )  # pytype: disable=wrong-arg-types  # py310-upgrade


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


# Traverse entire config HParam and find the tpl of the target type.
def find_target_tpl(config: LayerTpl, target: Type[base_layer.BaseLayer]):
  """Find and return target tpl from the config."""
  to_process = [config]
  target_tpl = []
  while to_process:
    param = to_process.pop(0)
    if isinstance(param, fdl.Config):
      if issubclass(fdl.get_callable(param), target):
        target_tpl.append(param)
        continue
      else:
        to_process.extend(fdl.ordered_arguments(param).values())
  return target_tpl
