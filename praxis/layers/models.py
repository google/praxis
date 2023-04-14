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

"""Definition of specific models."""

from typing import Any, Dict, Optional, Sequence, Tuple

from absl import logging
import clu.metrics as clu_metrics
import jax
from jax import numpy as jnp
import numpy as np
from praxis import asserts
from praxis import base_input
from praxis import base_layer
from praxis import base_model
from praxis import beam_search
from praxis import decoder_hparams
from praxis import decoder_utils
from praxis import flat_beam_search
from praxis import metric_utils
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis import sample_decode
from praxis.layers import augmentations
from praxis.layers import embedding_softmax
from praxis.layers import linears
from praxis.layers import resnets
from praxis.layers import transformer_models

NestedMap = py_utils.NestedMap
JTensor = pytypes.JTensor
Predictions = base_model.Predictions
Metrics = base_model.Metrics
WeightedScalars = base_model.WeightedScalars
DecodeOut = base_model.DecodeOut
ProcessDecodeOut = base_model.ProcessDecodeOut
DecoderHParams = decoder_hparams.DecoderHParams
BeamSearchHParams = decoder_hparams.BeamSearchHParams
FlatBeamSearchHParams = decoder_hparams.FlatBeamSearchHParams
SampleDecoderHParams = decoder_hparams.SampleDecoderHParams
GreedyDecoderHParams = decoder_hparams.GreedyDecoderHParams
LanguageModelType = transformer_models.LanguageModelType
LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]
template_field = base_layer.template_field


def _merge_per_token_and_per_example_weights(
    per_token_weights: JTensor, per_example_weights: JTensor) -> JTensor:
  """Merges the per token and per example weights into per token weights.

  Args:
    per_token_weights: A float32 tensor of shape [B, T].
    per_example_weights: A float32 tensor of shape [B].

  Returns:
    A merged per token weights tensor of shape [B, T].
  """
  seq_len = per_token_weights.shape[1]
  # Shape [B, T].
  per_example_weights_tiled = jnp.tile(
      jnp.expand_dims(per_example_weights, axis=-1), (1, seq_len))
  return per_token_weights * per_example_weights_tiled


def compute_xent_loss_helper(
    predictions: NestedMap,
    input_batch: NestedMap,
    return_predictions: bool,
    apply_eval_sample_weights: bool = False,
    report_strict_acc: bool = False,
) -> Tuple[WeightedScalars, Dict[str, Any]]:
  """Helper for computing the xent loss for Language model and Sequence model.

  Args:
    predictions: A `.NestedMap` containing the keys `per_example_argmax`,
      `total_loss`, `avg_xent`, `aux_loss`, `total_weight` which corresponds to
      the output of the Softmax layer.
    input_batch: A `.NestedMap` object containing input tensors which contains
      the keys `labels` and `weights` which corresponds to the labels and the
      `weights` for each token in the sequence.
    return_predictions: Whether to return predictions, which can be more
      expensive.
    apply_eval_sample_weights: Boolean indicating whether to apply the per
      example weights from the input `eval_sample_weights` or not. When enabled,
      these per-example weights will be merged with the per token
      `input_batch.weights`.
    report_strict_acc: Whether to report strict accuracy. In general, this requires 
      the entire portion of the sequence with nonzero weight be predicted correctly.
      Frequently used for eval on the Lambada dataset, in which case this metric is 
      equivalent to full-word matching. 

  Returns:
    - A dict or NestedMap containing str keys and (value, weight) pairs as
      values, where one of the entries is expected to correspond to the loss.
    - A dict containing arbitrary tensors describing something about each
      training example, where the first dimension of each tensor is the batch
      index. The base class just returns an empty dict.
  """

  labels = input_batch.labels
  weights = input_batch.weights
  if apply_eval_sample_weights:
    if not hasattr(input_batch, 'eval_sample_weights'):
      logging.warning(
          '`apply_eval_sample_weights` enabled, but the input batch does not '
          'provide the necessary `eval_sample_weights` field.')
    weights = _merge_per_token_and_per_example_weights(
        weights, input_batch.eval_sample_weights)
  predicted_labels = predictions.per_example_argmax.astype(labels.dtype)
  num_preds = predictions.total_weight
  mean_acc = jnp.sum(
      (labels == predicted_labels) * weights) / jnp.maximum(num_preds, 1)
  metric_weight = jnp.array(num_preds, predictions.avg_xent.dtype)

  if hasattr(predictions, 'avg_xent_weight'):
    avg_xent_weight = predictions.avg_xent_weight
  else:
    avg_xent_weight = metric_weight

  metrics = NestedMap(
      total_loss=(predictions.total_loss, metric_weight),
      avg_xent=(predictions.avg_xent, avg_xent_weight),
      aux_loss=(predictions.aux_loss, jnp.array(1.0,
                                                predictions.aux_loss.dtype)),
      log_pplx=(predictions.avg_xent, avg_xent_weight),
      fraction_of_correct_next_step_preds=(mean_acc, metric_weight),
      num_predictions=(num_preds, jnp.array(1.0, num_preds.dtype)),
  )
  if report_strict_acc:
    num_acc = jnp.sum(weights, axis=-1, dtype=jnp.float32)
    ## mask out padding examples
    num_acc = jax.lax.select(input_batch.eval_sample_weights.astype(jnp.int32),
                             num_acc, jnp.inf*jnp.ones_like(num_acc))
    num_nonpadding = jnp.sum(input_batch.eval_sample_weights)

    mean_acc_strict = (jnp.sum(jnp.sum((labels == predicted_labels)
                                       * weights, axis=-1) == num_acc)
                       /jnp.maximum(num_nonpadding, 1))
    strict_weight = jnp.array(num_nonpadding, predictions.avg_xent.dtype)

    metrics.acc_strict=(mean_acc_strict, strict_weight)

  # The score for the sequence is the negative of the sum of per token cross
  # entropy, which is the (weighted) sum of log probs on the tokens.
  per_example_output = NestedMap(
      labels=labels, scores=-predictions.per_sequence_xent)
  if apply_eval_sample_weights and hasattr(input_batch, 'eval_sample_weights'):
    per_example_output.eval_sample_weights = input_batch.eval_sample_weights
  if return_predictions:
    per_example_output = predictions
  return metrics, per_example_output


class LanguageModel(base_model.BaseModel):
  """Language Model base task.

  Attributes:
    lm_tpl: LM layer.
    return_predictions: Whether to return predictions during eval. Returning
      predictions is more expensive, but may be useful for debugging.
    decoder_tpl: Parameterization of the decoder.
    model_type: The type of language model based on the tokens visibility.
    count_tokens: Whether to track total tokens trained on in the checkpoint.
    apply_eval_sample_weights: Boolean indicating whether to apply the per
      example weights from the input `eval_sample_weights` or not.
    report_strict_acc: Whether to report strict accuracy. Used for eval on 
      Lambada dataset.
  """
  lm_tpl: LayerTpl = template_field(transformer_models.TransformerLm)
  return_predictions: bool = False
  decoder_tpl: DecoderHParams = base_layer.instance_field(GreedyDecoderHParams)
  model_type: LanguageModelType = LanguageModelType.CAUSAL
  count_tokens: bool = False
  apply_eval_sample_weights: bool = False
  report_strict_acc: bool = False

  def setup(self) -> None:
    super().setup()

    if self.count_tokens:
      tc_p = pax_fiddle.Config(embedding_softmax.TokenCounter)
      self.create_child('token_counter', tc_p)

    # Construct the model.
    lm_p = self.lm_tpl.clone()
    lm_p.model_type = self.model_type
    self.create_child('lm', lm_p)

  def _prepare_predict_data(self, input_batch: NestedMap) -> NestedMap:
    paddings = input_batch.paddings
    weights = input_batch.weights
    if self.apply_eval_sample_weights:
      if not hasattr(input_batch, 'eval_sample_weights'):
        logging.warning(
            '`apply_eval_sample_weights` enabled, but the input batch does not '
            'provide the necessary `eval_sample_weights` field.')
      weights = _merge_per_token_and_per_example_weights(
          weights, input_batch.eval_sample_weights)
    inputs = input_batch.ids
    if self.count_tokens:
      self.token_counter(inputs, paddings)
    labels = NestedMap(class_ids=input_batch.labels, class_weights=weights)

    extra_input_kwargs = {}
    if self.lm_tpl.packed_input:
      extra_input_kwargs = {
          'segment_ids': input_batch.segment_ids,
          'segment_pos': input_batch.segment_pos,
      }
      if 'segment_mask' in input_batch:
        # Note that the "real" segment mask is inferred from segment ids (and
        # possibly paddings.) If segment mask is explicitly specified here, what
        # we are saying is "I want to provide my own attention mask."
        extra_input_kwargs['segment_mask'] = input_batch.segment_mask

    if self.model_type == LanguageModelType.BIDIRECTIONAL:
      causal_attention_mask = jnp.zeros_like(inputs)
    elif self.model_type == LanguageModelType.PREFIX:
      causal_attention_mask = 1 - input_batch.inputs_indicator
    else:
      causal_attention_mask = None
    return NestedMap(inputs=inputs, paddings=paddings, labels=labels,
                     causal_attention_mask=causal_attention_mask,
                     extra_input_kwargs=extra_input_kwargs)

  def compute_predictions(self, input_batch: NestedMap) -> Predictions:
    """Computes predictions for `input_batch`."""
    predict_data = self._prepare_predict_data(input_batch)
    predictions = self.lm(
        inputs=predict_data.inputs,
        paddings=predict_data.paddings,
        labels=predict_data.labels,
        causal_attention_mask=predict_data.causal_attention_mask,
        **predict_data.extra_input_kwargs)

    return predictions

  def compute_loss(  # pytype: disable=signature-mismatch  # jax-ndarray
      self, predictions: NestedMap,
      input_batch: NestedMap) -> Tuple[WeightedScalars, Dict[str, Any]]:
    """Computes the loss and other metrics for the given predictions.

    Args:
      predictions: The output of `compute_predictions`.
      input_batch: A `.NestedMap` object containing input tensors to this tower.

    Returns:
      - A dict or NestedMap containing str keys and (value, weight) pairs as
        values, where one of the entries is expected to corresponds to the loss.
      - A dict containing arbitrary tensors describing something about each
        training example, where the first dimension of each tensor is the batch
        index.
    """
    return compute_xent_loss_helper(
        predictions,
        input_batch,
        self.return_predictions,
        self.apply_eval_sample_weights,
        self.report_strict_acc,
    )

  def _prepare_guidance_decode_data(self, decode_data: NestedMap) -> NestedMap:
    raise NotImplementedError('LanguageModel does not support guidance.')

  def _prepare_decode_data(
      self, input_batch: NestedMap, decoder_params: DecoderHParams
  ) -> NestedMap:
    batch_size = input_batch.ids.shape[0]

    # TODO(b/229679837): unify prefix_lengths logic to depend on
    # inputs_indicator or paddings only.
    # Get prefix_lengths from inputs_indicator or paddings.
    if 'prefix_lengths' in input_batch:
      prefix_lengths = input_batch.prefix_lengths
    elif self.model_type == transformer_models.LanguageModelType.PREFIX:
      asserts.eq(('inputs_indicator' in input_batch),
                 True,
                 msg='inputs_indicator should be in input batch for prefix LM.')
      prefix_lengths = jnp.sum(
          input_batch.inputs_indicator.astype(jnp.int32), axis=1)
    else:
      # The max lengths of the prefix, which are the number of unpadded tokens.
      # Note that computing the sum with bf16 is not precise enough, so convert
      # paddings to integers first.
      maxval = jnp.sum(1 - input_batch.paddings.astype(jnp.int32), axis=1)
      prefix_lengths = maxval

    if self.model_type == LanguageModelType.BIDIRECTIONAL:
      raise NotImplementedError(type(self))
    elif self.model_type == LanguageModelType.PREFIX:
      if 'inputs_indicator' in input_batch:
        causal_attention_mask = 1 - input_batch.inputs_indicator
      else:
        causal_attention_mask = (
            jnp.arange(input_batch.ids.shape[-1])[jnp.newaxis, :] >=
            prefix_lengths[:, jnp.newaxis])
    else:
      causal_attention_mask = None

    max_prefix_len = input_batch.ids.shape[1]
    if decoder_params.fprop_for_prefix:
      asserts.not_none(decoder_params.max_decode_steps)
      max_decode_steps = decoder_params.max_decode_steps
      last_max_decode_steps = (
          max(max_decode_steps)
          if isinstance(max_decode_steps, Sequence)
          else max_decode_steps
      )
      first_max_decode_steps = (
          min(max_decode_steps)
          if isinstance(max_decode_steps, Sequence)
          else max_decode_steps
      )
      seqlen = max_prefix_len + last_max_decode_steps
      start_time_step = max_prefix_len - 1
      # Change prefix to be right-aligned.
      fprop_input_ids, fprop_input_paddings = (
          sample_decode.right_align_prefix_ids(input_batch.ids, prefix_lengths,
                                               self.fprop_dtype))
      fprop_segment_pos = sample_decode.right_align_segment_position(
          prefix_lengths, max_prefix_len)
      # Make the left-padding into a separate segment. Some models may use only
      # segment mask instead of paddings.
      fprop_segment_ids = jnp.where(
          jnp.arange(max_prefix_len) <
          (max_prefix_len - prefix_lengths)[:, jnp.newaxis],
          jnp.zeros_like(fprop_segment_pos), jnp.ones_like(fprop_segment_pos))
      state_padding_size = first_max_decode_steps
      # Init input ids and paddings for extend_step.
      input_ids = jnp.pad(
          fprop_input_ids,
          [[0, 0], [0, first_max_decode_steps]],
      )
      input_paddings = jnp.pad(
          fprop_input_paddings,
          [[0, 0], [0, first_max_decode_steps]],
          constant_values=1.0,
      )
      if causal_attention_mask is not None:
        # Pad 1 to the left of causal_attention_mask.
        causal_attention_mask = decoder_utils.right_align_tensors(
            causal_attention_mask, prefix_lengths)
    else:
      seqlen = decoder_params.seqlen
      start_time_step = 0
      input_ids = input_batch.ids
      input_paddings = input_batch.paddings
      fprop_input_ids = jnp.zeros((batch_size, 1), input_batch.ids.dtype)
      fprop_input_paddings = jnp.ones((batch_size, 1), self.fprop_dtype)
      fprop_segment_pos = None
      fprop_segment_ids = None
      state_padding_size = seqlen - 1

    decode_data = NestedMap(
        seqlen=seqlen,
        start_time_step=start_time_step,
        input_ids=input_ids,
        input_paddings=input_paddings,
        fprop_input_ids=fprop_input_ids,
        fprop_input_paddings=fprop_input_paddings,
        fprop_segment_ids=fprop_segment_ids,
        fprop_segment_pos=fprop_segment_pos,
        causal_attention_mask=causal_attention_mask,
        state_padding_size=state_padding_size,
        prefix_lengths=prefix_lengths,
        extra_input_kwargs={},
    )

    if (template_has_type(decoder_params, SampleDecoderHParams) and
        decoder_params.cf_guidance_scale is not None):
      decode_data = self._prepare_guidance_decode_data(decode_data)

    return decode_data

  def decode(
      self,
      input_batch: NestedMap,
      result_callback: Optional[decoder_utils.StreamingResultCallback] = None,
      return_result_for_suffix_score=False,
  ) -> DecodeOut:
    """Decodes the input_batch with specified decoder params.

    Args:
      input_batch: The input batch, with fields `.ids` and `.paddings`. It may
        have an optional `.prefix_lengths` field indicating the lengths of
        prefixes in the ids used as decoding inputs. Optional `.suffix` for the
        suffix_ids with shape [num_suffix, suffix_length]. Optional
        `.temperature` of shape [batch_size] has the temperature for each
        example. If `.temperature` is not set in the input_batch,
        p.decoder_tpl.temperature will be used in sampling decode. Optional
        `.per_example_max_decode_steps` of shape [batch_size] has the maximum
        decoding steps for each example.
      result_callback: Optional callback function to be called for intermediate
        decoding results.
      return_result_for_suffix_score: Whether return results for suffix score.

    Returns:
      A 3-tuple with:
      - weighted_scalars, a NestedMap containing str keys and (metrics, weight)
        pairs.
      - A NestedMap like `input_batch`, with `.prefix_lengths` (vector of
        specified or randomly generated ints indicating the lengths of prefixes
        for each row), and `.output_ids` (matrix of int ids with the decoded
        output). If `.suffix` exists in the `input_batch` and uses sample
        decode function, will return the decoded results with suffix and
        logprobs of the sequence with suffix, the return `.output_ids` and
        `.logprobs` will have the shape of
        [batch, num_samples, num_suffix, seq_len].
      - A dict of [str, clu_metrics.Metric] objects with metric objects.
    """
    if not isinstance(self.decoder_tpl, DecoderHParams):
      raise ValueError(
          'p.decoder_tpl must be DecoderHParams type, but it is a '
          f'type of {type(self.decoder_tpl)}'
      )
    return self.decode_with_params(
        self.decoder_tpl,
        input_batch,
        result_callback=result_callback,
        return_result_for_suffix_score=return_result_for_suffix_score,
    )

  def decode_with_params(
      self,
      decoder_params: DecoderHParams,
      input_batch: NestedMap,
      result_callback: Optional[decoder_utils.StreamingResultCallback] = None,
      return_result_for_suffix_score=False,
  ) -> DecodeOut:
    """Same as decode but with specified DecoderHParams."""
    if decoder_params.seqlen <= 0:
      raise ValueError(
          'Must set p.decoder_tpl.seqlen > 0, current value = '
          f'{decoder_params.seqlen}'
      )
    max_prefix_len = input_batch.ids.shape[1]
    decode_data = self._prepare_decode_data(input_batch, decoder_params)

    decode_mesh_transpose = decoder_params.decode_loop_mesh_axes_transpose
    if decode_mesh_transpose:
      lm_var_hparams = self.lm.abstract_init_with_metadata(
          decode_data.fprop_input_ids,
          decode_data.fprop_input_paddings,
          segment_ids=decode_data.fprop_segment_ids,
          segment_pos=decode_data.fprop_segment_pos,
          start_time_step=decode_data.start_time_step,
          causal_attention_mask=decode_data.causal_attention_mask,
          **decode_data.extra_input_kwargs,
      )

      lm_var_pspecs = base_layer.var_partition_specs(
          lm_var_hparams, self.lm.mesh_shape, self.lm.mesh_axis_names
      )
    else:
      lm_var_pspecs = None
    logging.info('decode_mesh_transpose: %s', decode_mesh_transpose)

    def extend_step_fn(mdl, ids, segment_pos):
      xent = mdl.extend_step(ids, segment_pos=segment_pos)
      return xent.logits

    def transform_decode_state_fn(mdl, transform_fn):
      mdl.transform_decode_state(transform_fn)

    def lazy_broadcast_prefix_fn(mdl, num_suffix_samples, suffix_length):
      mdl.lazy_broadcast_prefix(num_suffix_samples, suffix_length)

    # Flat beam search doesn't work yet.
    if template_has_type(decoder_params, FlatBeamSearchHParams):
      assert isinstance(decoder_params, FlatBeamSearchHParams)
      # Init cache states.
      self.lm(
          decode_data.fprop_input_ids,
          decode_data.fprop_input_paddings,
          segment_ids=decode_data.fprop_segment_ids,
          segment_pos=decode_data.fprop_segment_pos,
          start_time_step=decode_data.start_time_step,
          causal_attention_mask=decode_data.causal_attention_mask,
          **decode_data.extra_input_kwargs
      )
      mdl_for_decode = decoder_utils.maybe_reshard_mdl_for_decode(
          self.lm,
          decode_mesh_transpose,
          lm_var_pspecs,
          transform_decode_state_fn,
      )
      with decoder_utils.maybe_decode_mesh_transpose(
          mdl_for_decode, decode_mesh_transpose
      ):
        # Pad to full-sequence length.
        mdl_for_decode.transform_decode_state(
            decoder_utils.pad_state_fn(decode_data.state_padding_size)
        )
        result = flat_beam_search.flat_beam_search(
            mdl_for_decode,
            extend_step_fn,
            decode_data.input_ids,
            decode_data.input_paddings,
            decode_data.seqlen,
            beam_size=decoder_params.beam_size,
            fprop_dtype=self.fprop_dtype,
            max_decode_steps=decoder_params.max_decode_steps,
            eos_id=decoder_params.eos_id,
            length_norm_alpha=decoder_params.length_norm_alpha,
        )
    elif template_has_type(decoder_params, BeamSearchHParams):
      assert isinstance(decoder_params, BeamSearchHParams)
      assert decoder_params.fprop_for_prefix

      def fprop_fn(mdl, ids, paddings):
        mdl(
            ids,
            paddings,
            segment_ids=decode_data.fprop_segment_ids,
            segment_pos=decode_data.fprop_segment_pos,
            start_time_step=decode_data.start_time_step,
            causal_attention_mask=decode_data.causal_attention_mask,
            **decode_data.extra_input_kwargs
        )
      assert isinstance(decoder_params, BeamSearchHParams)
      result = beam_search.beam_search(
          self.lm,
          extend_step_fn,
          fprop_fn,
          transform_decode_state_fn,
          decode_data.fprop_input_ids,
          decode_data.fprop_input_paddings,
          decoder_params,
          decode_loop_mesh_axes_transpose=decode_mesh_transpose,
          model_var_pspecs=lm_var_pspecs,
      )
    elif template_has_type(decoder_params, SampleDecoderHParams):
      assert isinstance(decoder_params, SampleDecoderHParams)
      def fprop_fn(mdl, ids, paddings):
        del ids, paddings
        mdl(
            decode_data.fprop_input_ids,
            decode_data.fprop_input_paddings,
            segment_ids=decode_data.fprop_segment_ids,
            segment_pos=decode_data.fprop_segment_pos,
            start_time_step=decode_data.start_time_step,
            causal_attention_mask=decode_data.causal_attention_mask,
            **decode_data.extra_input_kwargs,
        )

      # Fetch dynamic temperature from input_batch if the input_batch has this
      # information.
      if hasattr(input_batch, 'temperature'):
        temperature = input_batch.temperature
      else:
        temperature = decoder_params.temperature

      # Fetch dynamic per params from input_batch if the
      # input_batch has this information.
      per_example_max_decode_steps = getattr(
          input_batch, 'per_example_max_decode_steps', None
      )
      per_example_top_p = getattr(input_batch, 'per_example_top_p', None)
      per_example_top_k = getattr(input_batch, 'per_example_top_k', None)
      eos_id = getattr(input_batch, 'eos_id', decoder_params.eos_id)
      gumbel_prng_key = getattr(input_batch, 'gumbel_prng_key', None)

      next_token_sampler_p = decoder_params.next_token_sampler_tpl.clone()
      # TODO(b/260646361): Avoid this param propagation.
      next_token_sampler_p.top_k = decoder_params.k
      next_token_sampler_p.top_p = decoder_params.p
      next_token_sampler_p.global_normalize = decoder_params.global_normalize
      next_token_sampler_p.top_k_recall_target = (
          decoder_params.top_k_recall_target
      )
      next_token_sampler_p.use_top_k_for_logprobs = (
          decoder_params.use_top_k_for_logprobs
      )
      next_token_sampler = base_layer.instantiate(next_token_sampler_p)

      result = sample_decode.sample_decode(
          self.lm,
          extend_step_fn,
          transform_decode_state_fn,
          lazy_broadcast_prefix_fn
          if decoder_params.lazy_prefix_broadcast
          else None,
          next_token_sampler,
          decode_data.input_ids,
          decode_data.input_paddings,
          decode_data.seqlen,
          fprop_fn=fprop_fn,
          num_samples=decoder_params.num_samples,
          fprop_for_prefix=decoder_params.fprop_for_prefix,
          temperature=temperature,
          per_example_top_p=per_example_top_p,
          per_example_top_k=per_example_top_k,
          max_prefix_len=max_prefix_len,
          max_decode_steps=decoder_params.max_decode_steps,
          per_example_max_decode_steps=per_example_max_decode_steps,
          prefix_lengths=decode_data.prefix_lengths,
          eos_id=eos_id,
          return_result_for_suffix_score=return_result_for_suffix_score,
          result_callback=result_callback,
          cf_guidance_scale=decoder_params.cf_guidance_scale,
          gumbel_prng_key=gumbel_prng_key,
          controlled_decoding=decoder_params.controlled_decoding,
          decode_loop_mesh_axes_transpose=decode_mesh_transpose,
          model_var_pspecs=lm_var_pspecs,
          sort_samples=decoder_params.sort_samples,
          use_top_k_for_logprobs=decoder_params.use_top_k_for_logprobs,
      )

    elif template_has_type(decoder_params, GreedyDecoderHParams):
      assert isinstance(decoder_params, GreedyDecoderHParams)

      def fprop_fn(mdl, ids, paddings):
        del ids, paddings
        mdl(
            decode_data.fprop_input_ids,
            decode_data.fprop_input_paddings,
            segment_ids=decode_data.fprop_segment_ids,
            segment_pos=decode_data.fprop_segment_pos,
            start_time_step=decode_data.start_time_step,
            causal_attention_mask=decode_data.causal_attention_mask,
            **decode_data.extra_input_kwargs,
        )

      result = sample_decode.greedy_decode(
          self.lm,
          extend_step_fn,
          decode_data.input_ids,
          decode_data.input_paddings,
          decode_data.seqlen,
          fprop_fn=fprop_fn,
          fprop_for_prefix=decoder_params.fprop_for_prefix,
          transform_state_fn=transform_decode_state_fn,
          max_prefix_len=max_prefix_len,
          max_decode_steps=decoder_params.max_decode_steps,
          prefix_lengths=decode_data.prefix_lengths,
          eos_id=decoder_params.eos_id,
          decode_loop_mesh_axes_transpose=decode_mesh_transpose,
          model_var_pspecs=lm_var_pspecs,
      )
    else:
      # Needs to define a decoding algorithm.
      raise NotImplementedError(
          f'Decoding algorithm {type(decoder_params)} is not implemented.'
      )

    result.update(input_batch)

    if hasattr(result, 'eval_sample_weights'):
      num_decoded = jnp.sum(result.eval_sample_weights)
    else:
      num_decoded = jnp.array(result.ids.shape[0], jnp.float32)
    metrics = NestedMap(num_decoded=(num_decoded, jnp.array(1, jnp.float32)))
    out_clu_metrics = NestedMap()
    return metrics, result, out_clu_metrics

  def process_decode_out(self, input_obj: base_input.BaseInput,
                         decode_out: NestedMap) -> ProcessDecodeOut:
    """Processes one batch of decoded outputs.

    Args:
      input_obj: The input object where a tokenizer is accessible.
      decode_out: The output from decode(). May have an extra leading axis.

    Returns:
      A 3-tuple with:
      - metrics, a NestedMap containing str keys and (metric, weight) pairs for
        the current batch (a tuple of two scalars).
      - A list of dict where each entry corresponds to a row in the batch. The
        keys should be unique across the entire decode dataset.
      - out_clu_metrics, a NestedMap containing str keys and clu_metrics.Metric
        objects. This is currently unused.
    """
    # Optionally get all samples output in text format.
    batch_size, num_samples, max_len = decode_out.output_ids.shape
    if num_samples > 1:
      sampled_ids = np.reshape(decode_out.output_ids, [-1, max_len])
      sampled_lengths = np.reshape(decode_out.decode_lengths, [-1])
      sampled_strs = input_obj.ids_to_strings(sampled_ids, sampled_lengths)
      sampled_strs = np.reshape(sampled_strs, [batch_size, num_samples])
    else:
      sampled_strs = None

    # Get the first output within a batch.
    decode_out.output_ids = decode_out.output_ids[:, 0, :]
    decode_out.decode_lengths = decode_out.decode_lengths[:, 0]
    decode_out.original_lengths = decode_out.original_lengths[:, 0]
    decode_out.prefix_ids = decode_out.prefix_ids[:, 0, :]
    decode_out.logprobs = decode_out.logprobs[:, 0, :]
    if decode_out.prefix_lengths.ndim == 2:
      decode_out.prefix_lengths = decode_out.prefix_lengths[:, 0]

    decoded_strs = input_obj.ids_to_strings(decode_out.output_ids,
                                            decode_out.decode_lengths)
    original_strs = input_obj.ids_to_strings(decode_out.ids,
                                             decode_out.original_lengths)
    prefix_strs = input_obj.ids_to_strings(decode_out.prefix_ids,
                                           decode_out.prefix_lengths)

    ret = []
    for idx, decoded_str in enumerate(decoded_strs):
      if ('eval_sample_weights' in decode_out and
          not decode_out.eval_sample_weights[idx]):
        # skip padded examples
        continue

      prefix_length = decode_out.prefix_lengths[idx]
      decode_length = decode_out.decode_lengths[idx]
      # Note that this field has varying lengths.
      decoded_ids = decode_out.output_ids[idx][prefix_length:decode_length]
      decoded_substr = input_obj.ids_to_strings(
          decoded_ids[None, :],
          np.array([decode_length - prefix_length], dtype=np.int32))[0]

      ret_dict = {
          'prefix': prefix_strs[idx],
          'decoded': decoded_str,
          'original': original_strs[idx],
          'ids': decode_out.output_ids[idx],
          'decoded_ids': decoded_ids,
          'decoded_substr': decoded_substr,
          'logprobs': decode_out.logprobs[idx],
          'prefix_length': prefix_length,
          'decode_length': decode_length,
      }
      if sampled_strs is not None:
        ret_dict['sampled'] = list(sampled_strs[idx])
      ret.append((prefix_strs[idx], ret_dict))

    decoded_lengths = np.average(decode_out.decode_lengths).astype(np.float32)
    metrics = NestedMap(
        decoded_length=(decoded_lengths, np.array(1.0, np.float32)))
    out_clu_metrics = NestedMap()
    return metrics, ret, out_clu_metrics


class SequenceModel(base_model.BaseModel):
  """Sequence Model base task.

  Attributes:
    model_tpl: Sequence model layer for this task.
    return_predictions: Whether to return predictions during eval. Returning
      predictions is more expensive, but may be useful for debugging.
    decoder_tpl: Parameterization of the decoder.
    label_smoothing_prob: If > 0.0, smooth out one-hot prob by spreading this
      amount ofprob mass to all other tokens.
  """
  model_tpl: LayerTpl = template_field(
      transformer_models.TransformerEncoderDecoder
  )
  return_predictions: bool = False
  decoder_tpl: DecoderHParams = base_layer.instance_field(GreedyDecoderHParams)
  label_smoothing_prob: float = 0.0

  def setup(self) -> None:
    # Construct the model.
    model_p = self.model_tpl.clone()
    self.create_child('model', model_p)

  def compute_predictions(self, input_batch):
    """Computes predictions for `input_batch`."""
    if self.model_tpl.packed_input:
      packed_input_kwargs = {
          'input_segment_ids': input_batch.src.segment_ids,
          'input_segment_pos': input_batch.src.segment_pos,
          'target_segment_ids': input_batch.tgt.segment_ids,
          'target_segment_pos': input_batch.tgt.segment_pos,
      }
    else:
      packed_input_kwargs = {}

    labels = NestedMap(
        class_ids=input_batch.tgt.labels, class_weights=input_batch.tgt.weights)
    if self.label_smoothing_prob > 0.0:
      vocab_size = self.model_tpl.softmax_tpl.num_classes
      class_probabilities = jax.nn.one_hot(labels.class_ids, vocab_size)
      fill_prob = self.label_smoothing_prob / (vocab_size - 1)
      class_probabilities = (
          (1.0 - self.label_smoothing_prob) * class_probabilities
          + fill_prob * (1.0 - class_probabilities)
      ).astype(self.fprop_dtype)
      labels.class_probabilities = class_probabilities

    return self.model(
        inputs=input_batch.src.ids,
        input_paddings=input_batch.src.paddings,
        targets=input_batch.tgt.ids,
        target_paddings=input_batch.tgt.paddings,
        labels=labels,
        **packed_input_kwargs)

  def compute_loss(self, predictions, input_batch):
    """Computes the loss and other metrics for the given predictions.

    Args:
      predictions: The output of `ComputePredictions`.
      input_batch: A `.NestedMap` object containing input tensors to this tower.

    Returns:
      - A dict or NestedMap containing str keys and (metric, weight) pairs as
        values, where one of the entries is expected to corresponds to the loss.
      - A dict containing arbitrary tensors describing something about each
        training example, where the first dimension of each tensor is the batch
        index.
    """
    # TODO(pax-dev): Add support for eval_sample_weights.
    return compute_xent_loss_helper(
        predictions, input_batch.tgt, self.return_predictions
    )

  def encode(self, input_batch: NestedMap) -> NestedMap:
    """Encodes input_batch.

    Args:
      input_batch: The input batch containing the `.ids` and `.paddings.`

    Returns:
      A NestedMap containing '.embeddings' that contains the model's embedding
      of the text.
    """
    embeddings = self.model.encode(input_batch.ids, input_batch.paddings)
    return py_utils.NestedMap(embeddings=embeddings)

  def decode(self, input_batch: NestedMap) -> DecodeOut:
    """Decodes input_batch.

    Args:
      input_batch: The input batch, with a field `.src` and `.tgt` corresponding
        to source and target, which itself contains the `.ids` and `.paddings.`

    Returns:
      A 3-tuple with:
      - weighted_scalars, a nestedmap of (scalar, weight) pairs.
      - results, a NestedMap like `input_batch`, with `.output_ids` (matrix of
        int ids with the decoded output) as well as the decoded length.
      - metrics, A NestedMap of clu.metrics objects.
    """
    if not template_has_type(self.decoder_tpl, DecoderHParams):
      raise ValueError(
          'p.decoder must be DecoderHParams type, but it is a '
          f'type of {type(self.decoder_tpl)}'
      )
    return self.decode_with_params(input_batch, self.decoder_tpl)

  def decode_with_params(
      self, input_batch: NestedMap, decoder_params: DecoderHParams
  ) -> DecodeOut:
    """Same as decode but with specified DecoderHParams."""
    if decoder_params.seqlen <= 0:
      raise ValueError(
          'Must set p.decoder_tpl.seqlen > 0, current value = '
          f'{decoder_params.seqlen}'
      )
    batch_size = input_batch.tgt.ids.shape[0]

    def extend_step_fn(mdl, ids, segment_pos):
      del segment_pos
      xent = mdl.model.extend_step(ids)
      return xent.logits

    def transform_decode_state_fn(mdl, transform_fn):
      mdl.model.transform_decode_state(transform_fn)

    # Flat beam search doesn't work yet.
    if template_has_type(decoder_params, FlatBeamSearchHParams):
      raise NotImplementedError('flat beam search not supported')
    elif template_has_type(decoder_params, BeamSearchHParams):
      assert isinstance(decoder_params, BeamSearchHParams)
      assert not decoder_params.fprop_for_prefix
      start_time_step = 0

      # Prefix decoding is not fully supported.
      # This fprop_fn is currently used for initializing the decoder states.
      def fprop_fn(mdl, ids, paddings):
        mdl.model(
            inputs=input_batch.src.ids,
            input_paddings=input_batch.src.paddings,
            targets=ids,
            target_paddings=paddings,
            start_time_step=start_time_step)

      fprop_input_ids = input_batch.tgt.ids[:, :1]
      fprop_input_paddings = jnp.ones((batch_size, 1),
                                      input_batch.tgt.paddings.dtype)
      result = beam_search.beam_search(
          self,
          extend_step_fn,
          fprop_fn,
          transform_decode_state_fn,
          fprop_input_ids,
          fprop_input_paddings,
          decoder_params,
      )
    elif template_has_type(decoder_params, SampleDecoderHParams):
      raise NotImplementedError('sample decode not supported')
    elif template_has_type(decoder_params, GreedyDecoderHParams):

      def fprop_fn(mdl, ids, paddings):
        del ids, paddings
        mdl.model(
            inputs=input_batch.src.ids,
            input_paddings=input_batch.src.paddings,
            targets=input_batch.tgt.ids[:, :1],
            target_paddings=input_batch.tgt.paddings[:, :1],
        )

      result = sample_decode.greedy_decode(
          self,
          extend_step_fn,
          input_batch.tgt.ids,
          input_batch.tgt.paddings,
          decoder_params.seqlen,
          eos_id=decoder_params.eos_id,
          fprop_fn=fprop_fn,
          transform_state_fn=transform_decode_state_fn,
      )
    else:
      # Needs to define a decoding algorithm.
      raise NotImplementedError(
          f'Decoding algorithm {type(decoder_params)} is not implemented.'
      )

    result.update(input_batch)
    if hasattr(result, 'eval_sample_weights'):
      num_decoded = jnp.sum(result.eval_sample_weights)
    else:
      num_decoded = jnp.array(batch_size, jnp.float32)
    metrics = NestedMap(num_decoded=(num_decoded, jnp.array(1, jnp.float32)))
    out_clu_metrics = NestedMap()
    return metrics, result, out_clu_metrics

  def process_decode_out(self, input_obj: base_input.BaseInput,
                         decode_out: NestedMap) -> ProcessDecodeOut:
    """Processes one batch of decoded outputs.

    Args:
      input_obj: The input object where a tokenizer is accessible.
      decode_out: The output from decode(). May have an extra leading axis.

    Returns:
      A 3-tuple with:
      - metrics, a NestedMap containing str keys and (metric, weight) pairs for
        the current batch (a tuple of two scalars).
      - A list of dict where each entry corresponds to a row in the batch. The
        keys should be unique across the entire decode dataset.
      - out_clu_metrics, a NestedMap containing str keys and clu_metrics.Metric
        objects. This is currently unused.
    """
    # Optionally get all samples output in text format.
    batch_size, num_samples, max_len = decode_out.output_ids.shape
    if num_samples > 1:
      sampled_ids = np.reshape(decode_out.output_ids, [-1, max_len])
      sampled_lengths = np.reshape(decode_out.decode_lengths, [-1])
      sampled_strs = input_obj.ids_to_strings(
          sampled_ids, sampled_lengths, key='tgt')
      sampled_strs = np.reshape(sampled_strs, [batch_size, num_samples])
    else:
      sampled_strs = None

    # Get the first output within a batch.
    decode_out.output_ids = decode_out.output_ids[:, 0, :]
    decode_out.decode_lengths = decode_out.decode_lengths[:, 0]
    decode_out.original_lengths = decode_out.original_lengths[:, 0]
    decode_out.logprobs = decode_out.logprobs[:, :]
    decoded_strs = input_obj.ids_to_strings(
        decode_out.output_ids, decode_out.decode_lengths, key='tgt')
    source_lengths = np.sum(
        1.0 - decode_out.src.paddings, axis=1).astype(np.int32)
    source_strs = input_obj.ids_to_strings(
        decode_out.src.ids, source_lengths, key='src')
    target_lengths = np.sum(
        1.0 - decode_out.tgt.paddings, axis=1).astype(np.int32)
    target_strs = input_obj.ids_to_strings(
        decode_out.tgt.ids, target_lengths, key='tgt')
    ret = list()
    for idx, decoded_str in enumerate(decoded_strs):
      if (hasattr(decode_out, 'eval_sample_weights') and
          not decode_out.eval_sample_weights[idx]):
        continue

      logging.info('SRC: %s\n', source_strs[idx])
      logging.info('TGT: %s\n', target_strs[idx])
      logging.info('OUT: %s\n', decoded_str)

      ret_dict = {
          'source': source_strs[idx],
          'decoded': decoded_str,
          'target': target_strs[idx],
          'ids': decode_out.output_ids[idx],
          'logprobs': decode_out.logprobs[idx],
          'decode_length': decode_out.decode_lengths[idx],
          # TODO(b/244434890): remove workaround with more robust integration
          'prefix': source_strs[idx],  # for seqio metrics
          'decoded_substr': decoded_str,  # for seqio metrics
      }
      if sampled_strs is not None:
        ret_dict['sampled'] = list(sampled_strs[idx])
      ret.append((source_strs[idx], ret_dict))
    decode_lengths = np.average(decode_out.decode_lengths).astype(np.float32)
    metrics = NestedMap(
        decode_length=(decode_lengths, np.array(1.0, np.float32)))
    out_clu_metrics = NestedMap()
    return metrics, ret, out_clu_metrics


class ClassificationModel(base_model.BaseModel):
  """Classification task for images and video.

  Attributes:
    network_tpl: The classifier network_tpl, which is ResNet-50 by default.
    softmax_tpl: The softmax_tpl layer used for the classification.
    input_field: The input field which contains the image or video features to
      pass to the classification network.
  """
  network_tpl: LayerTpl = template_field(resnets.ResNet)
  softmax_tpl: LayerTpl = template_field(embedding_softmax.FullSoftmax)
  input_field: str = 'image'
  label_field: str = 'label_probs'

  def setup(self) -> None:
    super().setup()
    self.create_child('network', self.network_tpl)
    self.create_child('softmax', self.softmax_tpl)

  def compute_predictions(self, input_batch: NestedMap) -> Predictions:
    """Computes predictions for `input_batch`.

    Args:
      input_batch: A `.NestedMap` object containing input tensors to this tower.

    Returns:
      - A NestedMap containing str keys and features, softmax output and the
        class weights as values.
    """
    inputs = input_batch[self.input_field]
    label_probs = input_batch[self.label_field]
    features = self.network(inputs)
    batch_size = inputs.shape[0]
    example_weights = jnp.ones([batch_size])
    if 'weight' in input_batch:
      example_weights = input_batch.weight
      if example_weights.shape != (batch_size,):
        raise ValueError(
            f'Shape of example weights should be ({batch_size},), but instead'
            f'is {example_weights.shape}')
    # Softmax expects weights to be of shape [..., 1].
    softmax_output = self.softmax(
        inputs=features,
        class_weights=example_weights[:, jnp.newaxis],
        class_probabilities=label_probs)
    return NestedMap(
        features=features,
        softmax_output=softmax_output,
        example_weights=example_weights)

  def compute_loss(  # pytype: disable=signature-mismatch  # jax-ndarray
      self, predictions: NestedMap,
      input_batch: NestedMap) -> Tuple[WeightedScalars, Dict[str, Any]]:
    """Computes the loss and other metrics for the given predictions.

    Args:
      predictions: The output of `compute_predictions`.
      input_batch: A `.NestedMap` object containing input tensors to this tower.

    Returns:
      - A dict or NestedMap containing str keys and (value, weight) pairs as
        values, where one of the entries is expected to correspond to the loss.
      - A dict containing arbitrary tensors describing something about each
        training example, where the first dimension of each tensor is the batch
        index. The base class just returns an empty dict.
    """
    label_probs = input_batch[self.label_field]
    avg_xent = predictions.softmax_output.avg_xent
    total_weight = predictions.softmax_output.total_weight
    metrics = NestedMap(
        avg_xent=(avg_xent, total_weight),
        num_predictions=(total_weight, jnp.array(1.0, total_weight.dtype)))
    # Compute top-1 and top-5 accuracy and add summary.
    acc1 = metric_utils.top_k_accuracy(
        1,
        predictions.softmax_output.logits,
        label_probs=label_probs,
        weights=predictions.example_weights)
    metrics.update(
        accuracy=(acc1, predictions.softmax_output.total_weight),
        error=(1.0 - acc1, predictions.softmax_output.total_weight),
    )
    self.add_summary('acc1', acc1)

    num_classes = predictions.softmax_output.logits.shape[-1]
    if num_classes > 5:
      acc5 = metric_utils.top_k_accuracy(
          5,
          predictions.softmax_output.logits,
          label_probs=label_probs,
          weights=predictions.example_weights)
      metrics.update(
          acc5=(acc5, predictions.softmax_output.total_weight),
          error5=(1.0 - acc5, predictions.softmax_output.total_weight),
      )
      self.add_summary('acc5', acc5)

    per_example_out = NestedMap(
        labels=label_probs, scores=predictions.softmax_output.logits
        )
    return metrics, per_example_out

  def predict(self, input_batch: NestedMap) -> Predictions:
    """Computes logits from `input_batch`.

    Args:
      input_batch: A `.NestedMap` object containing input tensors to this tower.

    Returns:
      - A NestedMap containing logits and logp scores.
    """
    inputs = input_batch[self.input_field]
    features = self.network(inputs)
    logits = self.softmax.get_logits(inputs=features)
    logp = self.softmax.logits_to_logp(logits)
    return py_utils.NestedMap(logits=logits, logp=logp)

  def decode(self, input_batch: NestedMap) -> DecodeOut:
    """Computes predictions and runs metrics."""
    label_probs = input_batch[self.label_field]
    predictions = self.compute_predictions(input_batch)
    losses, _ = self.compute_loss(predictions, input_batch)

    per_example_out = NestedMap()
    eval_metrics = NestedMap()

    eval_metrics.accuracy = clu_metrics.Accuracy.from_model_output(
        logits=predictions.softmax_output.logits,  # pytype: disable=attribute-error  # jax-ndarray
        labels=jnp.argmax(label_probs, axis=-1))
    return losses, per_example_out, eval_metrics

  def process_decode_out(self, input_obj: base_input.BaseInput,
                         decode_out: NestedMap) -> ProcessDecodeOut:
    return NestedMap(), [], NestedMap()


class BertModel(base_model.BaseModel):
  """Bert Model base task.

  Attributes:
    lm: BERT LM layer.
    label_smoothing_prob: If > 0.0, smooth out one-hot prob by spreading this
      amount of prob mass to all other tokens.
    mask_token_id: Mask token id.
    force_mask_generation: if True, always use runtime generated
      random mask for training, even if pre-generated masks exist in the
      training examples.
  """
  lm_tpl: LayerTpl = template_field(transformer_models.TransformerLm)
  label_smoothing_prob: float = 0.0
  mask_token_id: int = 0
  force_mask_generation: bool = False

  def setup(self) -> None:
    super().setup()
    assert self.lm_tpl.model_type == LanguageModelType.BIDIRECTIONAL
    assert self.lm_tpl.packed_input

    self.create_child('lm', self.lm_tpl)

    mlm_augment_p = pax_fiddle.Config(augmentations.MaskedLmDataAugmenter)
    mlm_augment_p.vocab_size = self.lm_tpl.vocab_size
    mlm_augment_p.mask_token_id = self.mask_token_id
    self.create_child('mlm_augmenter', mlm_augment_p)

  def compute_predictions(self, input_batch: NestedMap) -> Predictions:
    """Computes predictions for `input_batch`."""
    assert self.lm_tpl.packed_input
    segment_ids = input_batch.segment_ids
    segment_pos = input_batch.segment_pos
    paddings = input_batch.paddings
    # Note that internal BertTransformer uses input_batch.ids instead.
    labels = input_batch.labels
    if not self.force_mask_generation and 'masked_ids' in input_batch:
      # Input data already has masking done.
      augmented_labels = input_batch.masked_ids
      augmented_pos = input_batch.masked_pos
    else:
      augmented_labels, augmented_pos = self.mlm_augmenter(labels, paddings)

    if self.label_smoothing_prob > 0.0:
      class_probabilities = jax.nn.one_hot(labels, self.lm.vocab_size)
      fill_prob = self.label_smoothing_prob / (self.lm.vocab_size - 1)
      class_probabilities = (
          (1.0 - self.label_smoothing_prob) * class_probabilities
          + fill_prob * (1.0 - class_probabilities)
      ).astype(self.fprop_dtype)

      # Only compute loss on masked pos.
      labels = NestedMap(
          class_probabilities=class_probabilities, class_weights=augmented_pos)
    else:
      # Only compute loss on masked pos.
      labels = NestedMap(class_ids=labels, class_weights=augmented_pos)

    lm_out = self.lm(
        inputs=augmented_labels,
        paddings=paddings,
        labels=labels,
        segment_ids=segment_ids,
        segment_pos=segment_pos)
    lm_out.augmented_labels = augmented_labels
    lm_out.augmented_pos = augmented_pos
    return lm_out

  def compute_loss(  # pytype: disable=signature-mismatch  # jax-ndarray
      self, predictions: NestedMap,
      input_batch: NestedMap) -> Tuple[WeightedScalars, Dict[str, Any]]:
    """Computes the loss and other metrics for the given predictions.

    Args:
      predictions: The output of `compute_predictions`.
      input_batch: A `.NestedMap` object containing input tensors to this tower.

    Returns:
      - A dict or NestedMap containing str keys and (metric, weight) pairs as
        values, where one of the entries is expected to corresponds to the loss.
      - A dict containing arbitrary tensors describing something about each
        training example, where the first dimension of each tensor is the batch
        index.
    """
    labels = input_batch.labels
    num_tokens = jnp.sum(1.0 - input_batch.paddings.astype(jnp.float32))
    num_seqs = jnp.sum(
        jnp.amax(input_batch.segment_ids.astype(jnp.float32), axis=1))
    weights = predictions.augmented_pos.astype(jnp.float32)
    predicted_labels = predictions.per_example_argmax.astype(labels.dtype)
    num_preds = predictions.total_weight.astype(jnp.float32)
    mean_acc = jnp.sum(
        (labels == predicted_labels) * weights) / jnp.maximum(num_preds, 1)
    metric_weight = jnp.array(num_preds, predictions.avg_xent.dtype)
    metrics = py_utils.NestedMap(
        total_loss=(predictions.total_loss, metric_weight),
        avg_xent=(predictions.avg_xent, metric_weight),
        aux_loss=(predictions.aux_loss, metric_weight),
        log_pplx=(predictions.avg_xent, metric_weight),
        fraction_of_correct_preds=(mean_acc, jnp.array(num_preds,
                                                       mean_acc.dtype)),
        num_predictions=(num_preds, jnp.array(1.0, num_preds.dtype)),
        num_tokens=(num_tokens, jnp.array(1.0, num_tokens.dtype)),
        num_seqs=(num_seqs, jnp.array(1.0, num_seqs.dtype)),
    )

    per_example_output = py_utils.NestedMap()
    return metrics, per_example_output


class ClassificationMLPModel(base_model.BaseModel):
  """Language Model task with a simple MLP model.

  Attributes:
    mlp_tpl: MLP model parameters.
    softmax_tpl: Input softmax_tpl embedding lookup layer.
  """
  mlp_tpl: LayerTpl = template_field(linears.MLPBlock)
  softmax_tpl: LayerTpl = template_field(
      embedding_softmax.SharedEmbeddingSoftmax
  )

  def setup(self) -> None:
    super().setup()
    # Note: We add a `_0` suffix here because this child was previously created
    # using create_children; and we want to ensure that its name stays the
    # same (for checkpoints, etc).
    self.create_child('mlp_layers_0', self.mlp_tpl)
    self.create_child('softmax', self.softmax_tpl)

  def compute_predictions(self, input_batch: NestedMap) -> Predictions:

    input_emb = self.softmax.emb_lookup(input_batch.ids)

    output = self.mlp_layers_0(input_emb)
    predictions = self.softmax(
        inputs=output,
        class_weights=input_batch.weights[:, :, jnp.newaxis],
        class_ids=input_batch.ids[:, :, jnp.newaxis])
    return predictions

  def compute_loss(  # pytype: disable=signature-mismatch  # jax-ndarray
      self, predictions: NestedMap,
      input_batch: NestedMap) -> Tuple[WeightedScalars, Dict[str, Any]]:
    labels = input_batch.labels
    weights = input_batch.weights
    class_weights = weights[:, :, jnp.newaxis]
    num_preds = jnp.sum(class_weights)
    predicted_labels = predictions.per_example_argmax.astype(labels.dtype)
    mean_acc = jnp.sum(
        (labels == predicted_labels) * weights) / jnp.maximum(num_preds, 1)
    metrics = NestedMap(total_loss=(mean_acc, mean_acc),)

    return metrics, NestedMap()


def template_has_type(template, cls):
  return (isinstance(template, cls) or
          (isinstance(template, pax_fiddle.Config) and template.cls == cls))
