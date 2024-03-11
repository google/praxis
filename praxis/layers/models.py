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

import dataclasses
from typing import Any, Mapping, Sequence, Union

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
from praxis.pytypes import ArrayT, Float32, Int32  # pylint: disable=g-importing-member,g-multiple-import,line-too-long

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
template_field = pax_fiddle.template_field


def _merge_per_token_and_per_example_weights(
    per_token_weights: JTensor, per_example_weights: JTensor
) -> JTensor:
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
      jnp.expand_dims(per_example_weights, axis=-1), (1, seq_len)
  )
  return per_token_weights * per_example_weights_tiled


def compute_xent_loss_helper(
    predictions: NestedMap,
    input_batch: NestedMap,
    return_predictions: bool,
    apply_eval_sample_weights: bool = False,
    report_strict_acc: bool = False,
) -> tuple[WeightedScalars, dict[str, Any]]:
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
    report_strict_acc: Whether to report strict accuracy. In general, this
      requires the entire portion of the sequence with nonzero weight be
      predicted correctly. Frequently used for eval on the Lambada dataset, in
      which case this metric is equivalent to full-word matching.

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
          'provide the necessary `eval_sample_weights` field.'
      )
    weights = _merge_per_token_and_per_example_weights(
        weights, input_batch.eval_sample_weights
    )
  predicted_labels = predictions.per_example_argmax.astype(labels.dtype)
  num_preds = predictions.total_weight
  mean_acc = jnp.sum((labels == predicted_labels) * weights) / jnp.maximum(
      num_preds, 1
  )
  metric_weight = jnp.array(num_preds, predictions.avg_xent.dtype)

  if hasattr(predictions, 'avg_xent_weight'):
    avg_xent_weight = predictions.avg_xent_weight
  else:
    avg_xent_weight = metric_weight

  metrics = NestedMap(
      total_loss=(predictions.total_loss, metric_weight),
      avg_xent=(predictions.avg_xent, avg_xent_weight),
      aux_loss=(
          predictions.aux_loss,
          jnp.array(1.0, predictions.aux_loss.dtype),
      ),
      log_pplx=(predictions.avg_xent, avg_xent_weight),
      fraction_of_correct_next_step_preds=(mean_acc, metric_weight),
      num_predictions=(num_preds, jnp.array(1.0, num_preds.dtype)),
  )
  if report_strict_acc:
    num_acc = jnp.sum(weights, axis=-1, dtype=jnp.float32)
    ## mask out padding examples
    num_acc = jax.lax.select(
        input_batch.eval_sample_weights.astype(jnp.int32),
        num_acc,
        jnp.inf * jnp.ones_like(num_acc),
    )
    num_nonpadding = jnp.sum(input_batch.eval_sample_weights)

    mean_acc_strict = jnp.sum(
        jnp.sum((labels == predicted_labels) * weights, axis=-1) == num_acc
    ) / jnp.maximum(num_nonpadding, 1)
    strict_weight = jnp.array(num_nonpadding, predictions.avg_xent.dtype)

    metrics.acc_strict = (mean_acc_strict, strict_weight)

  # The score for the sequence is the negative of the sum of per token cross
  # entropy, which is the (weighted) sum of log probs on the tokens.
  per_example_output = NestedMap(
      labels=labels, scores=-predictions.per_sequence_xent
  )
  if apply_eval_sample_weights and hasattr(input_batch, 'eval_sample_weights'):
    per_example_output.eval_sample_weights = input_batch.eval_sample_weights
  if return_predictions:
    per_example_output = predictions
  return metrics, per_example_output


def add_hist(
    layer: base_layer.BaseLayer,
    name: str,
    expression: jax.Array,
    verbosity: int = 2,
) -> None:
  layer.add_summary(
      name,
      expression,
      summary_type=base_layer.SummaryType.HISTOGRAM,
      verbosity=verbosity,
  )


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
            'provide the necessary `eval_sample_weights` field.'
        )
      weights = _merge_per_token_and_per_example_weights(
          weights, input_batch.eval_sample_weights
      )
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
    return NestedMap(
        inputs=inputs,
        paddings=paddings,
        labels=labels,
        causal_attention_mask=causal_attention_mask,
        extra_input_kwargs=extra_input_kwargs,
    )

  def compute_predictions(self, input_batch: NestedMap) -> Predictions:
    """Computes predictions for `input_batch`."""
    predict_data = self._prepare_predict_data(input_batch)
    predictions = self.lm(
        inputs=predict_data.inputs,
        paddings=predict_data.paddings,
        labels=predict_data.labels,
        causal_attention_mask=predict_data.causal_attention_mask,
        **predict_data.extra_input_kwargs,
    )

    return predictions

  def compute_loss(  # pytype: disable=signature-mismatch  # jax-ndarray
      self, predictions: NestedMap, input_batch: NestedMap
  ) -> tuple[Union[WeightedScalars, Metrics], dict[str, Any]]:
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
      asserts.eq(
          ('inputs_indicator' in input_batch),
          True,
          msg='inputs_indicator should be in input batch for prefix LM.',
      )
      prefix_lengths = jnp.sum(
          input_batch.inputs_indicator.astype(jnp.int32), axis=1
      )
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
            jnp.arange(input_batch.ids.shape[-1])[jnp.newaxis, :]
            >= prefix_lengths[:, jnp.newaxis]
        )
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
          sample_decode.right_align_prefix_ids(
              input_batch.ids, prefix_lengths, self.fprop_dtype
          )
      )
      fprop_segment_pos = sample_decode.right_align_segment_position(
          prefix_lengths, max_prefix_len
      )
      # Make the left-padding into a separate segment. Some models may use only
      # segment mask instead of paddings.
      fprop_segment_ids = jnp.where(
          jnp.arange(max_prefix_len)
          < (max_prefix_len - prefix_lengths)[:, jnp.newaxis],
          jnp.zeros_like(fprop_segment_pos),
          jnp.ones_like(fprop_segment_pos),
      )
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
            causal_attention_mask, prefix_lengths
        )
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

    extra_input_kwargs = {}
    if decoder_params.use_extra_input_kwargs:
      extra_input_kwargs = input_batch.get('extra_input_kwargs', {})

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
        extra_input_kwargs=extra_input_kwargs,
    )

    if (
        template_has_type(decoder_params, SampleDecoderHParams)
        and hasattr(decoder_params, 'cf_guidance_scale')
        and decoder_params.cf_guidance_scale is not None
    ):
      decode_data = self._prepare_guidance_decode_data(decode_data)

    return decode_data

  def decode(
      self,
      input_batch: NestedMap,
      result_callback: decoder_utils.StreamingResultCallback | None = None,
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
      result_callback: decoder_utils.StreamingResultCallback | None = None,
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
          **decode_data.extra_input_kwargs,
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
        if decoder_params.process_result_fn is not None:
          result = decoder_params.process_result_fn(mdl_for_decode, result)
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
            **decode_data.extra_input_kwargs,
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
          process_result_fn=decoder_params.process_result_fn,
          lazy_broadcast_prefix_fn=lazy_broadcast_prefix_fn
          if decoder_params.lazy_prefix_broadcast
          else None,
      )
    elif template_has_type(decoder_params, SampleDecoderHParams):
      assert isinstance(decoder_params, SampleDecoderHParams)
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
      return_entropy_score = hasattr(input_batch, 'return_entropy_score')
      cf_guidance_scale = getattr(
          input_batch, 'cf_guidance_scale', decoder_params.cf_guidance_scale
      )
      enforce_sample_constraints = getattr(
          input_batch, 'enforce_sample_constraints', None
      )
      num_per_token_logprobs = getattr(
          input_batch, 'num_per_token_logprobs', None
      )

      next_token_sampler_p = decoder_params.next_token_sampler_tpl.clone()
      # TODO(b/260646361): Avoid this param propagation.
      if decoder_params.override_next_token_sampler_params:
        next_token_sampler_p.top_k = decoder_params.k
        next_token_sampler_p.top_p = decoder_params.p
        next_token_sampler_p.global_normalize = decoder_params.global_normalize
        next_token_sampler_p.top_k_recall_target = (
            decoder_params.top_k_recall_target
        )
        next_token_sampler_p.use_top_k_for_logprobs = (
            decoder_params.use_top_k_for_logprobs
        )
      next_token_sampler = base_layer.instantiate(
          next_token_sampler_p,
          **next_token_sampler_p.cls.get_extra_kwargs(
              input_batch, decoder_params.num_samples
          ),
      )

      if decoder_params.vanilla_sample_decode:
        # TODO(b/289423925): All decoders should remove the last id. Currently,
        # few sax unittests fail in non-trivial way.
        # The last prefix token is the start id of decoding in fprop_for_prefix.
        def _remove_last(ids, paddings, segment_pos, segment_ids, causal_mask):
          if ids.shape[1] == 1:
            # Prevent empty tensor.
            ids = jnp.zeros_like(ids)
            paddings = jnp.ones_like(paddings)
          else:
            ids = ids[:, :-1]
            paddings = paddings[:, :-1]
            segment_pos = segment_pos[:, :-1]
            segment_ids = segment_ids[:, :-1]
            if causal_mask is not None:
              causal_mask = causal_mask[:, :-1]
          return (ids, paddings, segment_pos, segment_ids, causal_mask)

        (
            fprop_input_ids,
            fprop_input_paddings,
            fprop_segment_pos,
            fprop_segment_ids,
            causal_attention_mask,
        ) = _remove_last(
            decode_data.fprop_input_ids,
            decode_data.fprop_input_paddings,
            decode_data.fprop_segment_pos,
            decode_data.fprop_segment_ids,
            decode_data.causal_attention_mask,
        )

        def fprop_fn(mdl, ids, paddings):
          del ids, paddings
          mdl(
              fprop_input_ids,
              fprop_input_paddings,
              segment_ids=fprop_segment_ids,
              segment_pos=fprop_segment_pos,
              start_time_step=decode_data.start_time_step,
              causal_attention_mask=causal_attention_mask,
              **decode_data.extra_input_kwargs,
          )

        result = sample_decode.vanilla_sample_decode(
            model=self.lm,
            fprop_fn=fprop_fn,
            extend_step_fn=extend_step_fn,
            transform_state_fn=transform_decode_state_fn,
            next_token_sampler=next_token_sampler,
            prefix_ids=decode_data.fprop_input_ids,
            prefix_paddings=decode_data.fprop_input_paddings,
            temperature=temperature,
            gumbel_prng_key=gumbel_prng_key,
            max_decode_steps=decoder_params.max_decode_steps,
            eos_id=eos_id,
            decode_loop_mesh_axes_transpose=decode_mesh_transpose,
            model_var_pspecs=lm_var_pspecs,
        )
      else:

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
            cf_guidance_scale=cf_guidance_scale,
            gumbel_prng_key=gumbel_prng_key,
            controlled_decoding=decoder_params.controlled_decoding,
            decode_loop_mesh_axes_transpose=decode_mesh_transpose,
            model_var_pspecs=lm_var_pspecs,
            sort_samples=decoder_params.sort_samples,
            top_k_recall_target=decoder_params.top_k_recall_target,
            use_top_k_for_logprobs=decoder_params.use_top_k_for_logprobs,
            return_entropy_score=return_entropy_score,
            process_result_fn=decoder_params.process_result_fn,
            optimize_eos=decoder_params.optimize_eos,
            sample_constraint=decoder_params.sample_constraint,
            enforce_sample_constraints=enforce_sample_constraints,
            num_per_token_logprobs=num_per_token_logprobs,
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
          process_result_fn=decoder_params.process_result_fn,
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

  def process_decode_out(
      self, input_obj: base_input.BaseInput, decode_out: NestedMap
  ) -> ProcessDecodeOut:
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
    # In vanilla_sample_decode, num_samples = 1.
    if len(decode_out.output_ids.shape) == 2:
      batch_size, max_len = decode_out.output_ids.shape
      num_samples = 1
    elif len(decode_out.output_ids.shape) == 3:
      batch_size, num_samples, max_len = decode_out.output_ids.shape
    else:
      raise ValueError(
          'output shape is incorrect, expected a 2 or 3 tuple'
          f' but got {len(decode_out.output_ids.shape)}'
      )
    if num_samples > 1:
      sampled_ids = np.reshape(decode_out.output_ids, [-1, max_len])
      sampled_lengths = np.reshape(decode_out.decode_lengths, [-1])
      sampled_strs = input_obj.ids_to_strings(sampled_ids, sampled_lengths)
      sampled_strs = np.reshape(sampled_strs, [batch_size, num_samples])
    else:
      sampled_strs = None

    # Get the first output within a batch.
    if len(decode_out.output_ids.shape) == 3:
      decode_out.output_ids = decode_out.output_ids[:, 0, :]
      decode_out.decode_lengths = decode_out.decode_lengths[:, 0]
      decode_out.original_lengths = decode_out.original_lengths[:, 0]
      decode_out.prefix_ids = decode_out.prefix_ids[:, 0, :]
      decode_out.logprobs = decode_out.logprobs[:, 0, :]
    if decode_out.prefix_lengths.ndim == 2:
      decode_out.prefix_lengths = decode_out.prefix_lengths[:, 0]

    # for vanilla_sample_decode
    if 'original_lengths' not in decode_out:
      decode_out.original_lengths = decode_out.prefix_lengths
    decoded_strs = input_obj.ids_to_strings(
        decode_out.output_ids, decode_out.decode_lengths
    )
    original_strs = input_obj.ids_to_strings(
        decode_out.ids, decode_out.original_lengths
    )
    prefix_strs = input_obj.ids_to_strings(
        decode_out.prefix_ids, decode_out.prefix_lengths
    )

    ret = []
    for idx, decoded_str in enumerate(decoded_strs):
      if (
          'eval_sample_weights' in decode_out
          and not decode_out.eval_sample_weights[idx]
      ):
        # skip padded examples
        continue

      prefix_length = decode_out.prefix_lengths[idx]
      decode_length = decode_out.decode_lengths[idx]
      # Note that this field has varying lengths.
      decoded_ids = decode_out.output_ids[idx][prefix_length:decode_length]
      decoded_substr = input_obj.ids_to_strings(
          decoded_ids[None, :],
          np.array([decode_length - prefix_length], dtype=np.int32),
      )[0]

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
        decoded_length=(decoded_lengths, np.array(1.0, np.float32))
    )
    out_clu_metrics = NestedMap()
    return metrics, ret, out_clu_metrics


# TODO(@jwyang): add support for sample decoding and beam search
class LanguageModelContinuousBatching(LanguageModel):
  """Language model that uses continuous batching."""

  def _last_decode_step(self, decoder_params) -> int:
    max_decode_steps = decoder_params.max_decode_steps
    if isinstance(decoder_params.max_decode_steps, int):
      max_decode_steps = [decoder_params.max_decode_steps]
    max_decode_steps = sorted(max_decode_steps)
    return max(max_decode_steps)

  def sample_init_decode_state(
      self, decode_data: NestedMap, decoder_params: DecoderHParams
  ) -> NestedMap:
    max_prefix_len = decoder_params.seqlen - decoder_params.max_decode_steps

    def transform_decode_state_fn(mdl, transform_fn):
      mdl.transform_decode_state(transform_fn)

    decode_state = sample_decode.sample_init_decode_state(
        model=self.lm,
        prefix_ids=decode_data.input_ids,
        max_prefix_len=max_prefix_len,
        max_decode_steps=self._last_decode_step(decoder_params),
        top_k=getattr(decoder_params, 'k', 1),
        top_p=getattr(decoder_params, 'p', None),
        prefix_lengths=decode_data.prefix_lengths,
        eos_id=decoder_params.eos_id,
        transform_state_fn=transform_decode_state_fn,
    )
    return decode_state

  def sample_prefill(
      self,
      input_batch: NestedMap,
      decoder_params: DecoderHParams,
  ) -> NestedMap:
    if decoder_params.seqlen <= 0:
      raise ValueError(
          'Must set p.decoder_tpl.seqlen > 0, current value = '
          f'{decoder_params.seqlen}'
      )
    decode_data = self._prepare_decode_data(input_batch, decoder_params)

    # run prefill
    def fprop_fn(mdl, ids, paddings):
      del ids, paddings
      output = mdl(
          decode_data.fprop_input_ids,
          decode_data.fprop_input_paddings,
          segment_ids=decode_data.fprop_segment_ids,
          segment_pos=decode_data.fprop_segment_pos,
          start_time_step=decode_data.start_time_step,
          causal_attention_mask=decode_data.causal_attention_mask,
          **decode_data.extra_input_kwargs,
      )
      return output.logits

    logits = fprop_fn(
        self.lm, decode_data.input_ids, decode_data.input_paddings
    )
    last_prefix_logits = logits[:, -1, :]

    # init prefix decode state
    prefill_decode_state = self.sample_init_decode_state(
        decode_data, decoder_params
    )

    batch_size = input_batch.ids.shape[0]
    logging.info('Prefill batch_size is %s', batch_size)
    # Fetch dynamic per params from input_batch if the
    # input_batch has this information.
    last_decode_step = self._last_decode_step(decoder_params)
    temperature = getattr(decoder_params, 'temperature', 0.0)
    prefill_decode_state.temperature = getattr(
        input_batch,
        'temperature',
        jnp.ones(shape=(batch_size,), dtype=jnp.float32) * temperature,
    )
    prefill_decode_state.per_example_max_decode_steps = getattr(
        input_batch,
        'per_example_max_decode_steps',
        jnp.ones(shape=(batch_size,), dtype=jnp.int32) * last_decode_step,
    )
    prefill_decode_state.per_example_top_p = getattr(
        input_batch,
        'per_example_top_p',
        jnp.ones(shape=(batch_size,), dtype=jnp.float32),
    )
    prefill_decode_state.per_example_top_k = getattr(
        input_batch,
        'per_example_top_k',
        jnp.ones(shape=(batch_size,), dtype=jnp.int32),
    )

    def extend_step_fn(mdl, ids, segment_pos):
      del mdl, ids, segment_pos
      return last_prefix_logits

    last_decode_steps = self._last_decode_step(decoder_params)
    max_prefix_len = decoder_params.seqlen - last_decode_steps

    prefill_decode_state = sample_decode.sample_decoding_step(
        model=self.lm,
        extend_step_fn=extend_step_fn,
        decode_state=prefill_decode_state,
        max_prefix_len=max_prefix_len,
        eos_id=decoder_params.eos_id,
        decode_loop_mesh_axes_transpose=decoder_params.decode_loop_mesh_axes_transpose,
        max_decode_steps=decoder_params.max_decode_steps,
    )
    return prefill_decode_state

  def sample_insert(
      self,
      decoder_params,
      prefix_decode_state,
      prefix_decode_cache,
      decode_state,
      prefix_slot,
      slot,
  ):
    # update decode_state
    decode_state.per_sample_steps = decode_state.per_sample_steps.at[slot].set(
        prefix_decode_state.per_sample_steps[prefix_slot]
    )

    # set 0 to start decoding phase
    decode_state.done = decode_state.done.at[slot].set(0)

    attrs = [
        'has_eos',
        'prefix_lengths',
        'segment_pos',
        'decode_lengths',
        'output_ids',
        'logprobs',
        'temperature',
        'per_example_max_decode_steps',
        'per_example_top_p',
        'per_example_top_k',
    ]

    for attr in attrs:
      update = getattr(prefix_decode_state, attr)
      update = update[prefix_slot]
      state_dtype = getattr(decode_state, attr).dtype
      if state_dtype != update.dtype:
        logging.info(
            '%s changed from %s to %s:', attr, state_dtype, update.dtype
        )
        update = update.astype(state_dtype)

      if update.ndim == 0:
        update = jnp.expand_dims(update, axis=0)
      ret = jax.lax.dynamic_update_slice_in_dim(
          getattr(decode_state, attr), update, slot, axis=0
      )
      setattr(decode_state, attr, ret)

    # update kv_cache (need to right aligned)
    max_prefix_len = decoder_params.seqlen - decoder_params.max_decode_steps
    sequence_len = decoder_params.seqlen

    right_aligned_length = sequence_len - (decode_state.step - max_prefix_len)
    for i in range(self.lm_tpl.stacked_transformer_tpl.num_layers):
      layer_kv_cache_key = 'x_layers_{}'.format(i)
      per_layer_prefix_decode_cache = prefix_decode_cache['decoder_cache'][
          'lm'
      ]['transformer'][layer_kv_cache_key]['self_attention']
      atten_state = self.variables[base_layer.DECODE_CACHE]['lm'][
          'transformer'
      ][layer_kv_cache_key]['self_attention']

      for name in atten_state.keys():
        new_state = per_layer_prefix_decode_cache[name][prefix_slot]
        new_state = jnp.expand_dims(new_state, axis=0)
        atten_state[name] = jax.lax.dynamic_update_slice_in_dim(
            atten_state[name],
            decoder_utils.right_align_tensors(new_state, right_aligned_length),
            slot,
            axis=0,
        )

    return decode_state

  def left_align_decode_state(
      self, max_prefix_len, max_decode_steps, decode_state, batch_size
  ):
    # when reach end of sequence, align all tensors to left end, reset step
    decode_state.per_sample_steps = jnp.where(
        decode_state.done, max_prefix_len, decode_state.per_sample_steps
    )
    left_align_steps = jnp.max(decode_state.per_sample_steps)
    left_align_steps_arr = (
        jnp.ones_like(decode_state.prefix_lengths) * left_align_steps
    )

    row_length = max_prefix_len + max_decode_steps

    transformer_kv_cache = self.variables[base_layer.DECODE_CACHE]['lm'][
        'transformer'
    ]
    for i in range(self.lm_tpl.stacked_transformer_tpl.num_layers):
      layer_kv_cache_key = 'x_layers_{}'.format(i)
      atten_kv = transformer_kv_cache[layer_kv_cache_key]['self_attention']
      for name in atten_kv.keys():
        atten_kv[name] = decoder_utils.left_align_kv_cache(
            atten_kv[name],
            left_align_steps_arr,
            row_length - 1,
            batch_size=batch_size,
        )

    decode_state.step = jnp.where(
        decode_state.step < row_length - 1, decode_state.step, left_align_steps
    )
    self.variables[base_layer.DECODE_CACHE]['lm']['time_step'] = (
        decode_state.step[0]
    )

    return decode_state

  def sample_generate(
      self,
      decode_state: NestedMap,
      decoder_params: DecoderHParams,
      align_decode_state: bool = False,
  ) -> NestedMap:
    def extend_step_fn(mdl, ids, segment_pos):
      xent = mdl.extend_step(ids, segment_pos=segment_pos)
      return xent.logits

    model = self.lm
    decode_mesh_transpose = decoder_params.decode_loop_mesh_axes_transpose
    last_decode_steps = self._last_decode_step(decoder_params)

    max_prefix_len = decoder_params.seqlen - last_decode_steps
    if align_decode_state:
      decode_state = self.left_align_decode_state(
          max_prefix_len,
          last_decode_steps,
          decode_state,
          decoder_params.num_cache_slots,
      )

    decode_state = sample_decode.sample_decoding_step(
        model=model,
        extend_step_fn=extend_step_fn,
        decode_state=decode_state,
        max_prefix_len=max_prefix_len,
        eos_id=decoder_params.eos_id,
        decode_loop_mesh_axes_transpose=decode_mesh_transpose,
        max_decode_steps=decoder_params.max_decode_steps,
    )
    return decode_state


@dataclasses.dataclass(kw_only=True, slots=True)
class Labels:
  class_ids: Int32[ArrayT, 'B T']
  class_weights: Float32[ArrayT, 'B T']


@dataclasses.dataclass(kw_only=True, slots=True)
class DPOExampleHalf:
  """Represents a rated DPO example."""

  inputs: Int32[ArrayT, 'B T']
  labels: Labels
  paddings: Int32[ArrayT, 'B T']
  segment_ids: Int32[ArrayT, 'B T']  # Packing is currently unsupported.
  segment_pos: Int32[ArrayT, 'B T']  # Packing is currently unsupported.
  inputs_indicator: Int32[ArrayT, 'B T'] = None
  causal_attention_mask: Int32[ArrayT, 'B T'] | None = None

  def as_xformer_lm_input(self) -> dict[str, ArrayT | NestedMap]:
    nestedmap = NestedMap.FromNestedDataclass(self)
    del nestedmap['inputs_indicator']
    return nestedmap


@dataclasses.dataclass(kw_only=True, slots=True)
class DPOExample:
  """The structure of an example batch fed to a LanguageModelDPO instance.

  Different from the RM data structure, we separate out exactly two generations
  into y_w and y_l, where the former is the 'winner' in the pairwise matchup.
  """

  y_w: DPOExampleHalf
  y_l: DPOExampleHalf

  @classmethod
  def from_feature_converter(cls, example: Mapping[str, JTensor]):
    return cls(
        y_w=DPOExampleHalf(
            inputs=example['y_w/inputs'],
            inputs_indicator=example['y_w/inputs_indicator'],
            labels=Labels(
                class_ids=example['y_w/labels/ids'],
                class_weights=example['y_w/labels/weights'],
            ),
            paddings=example['y_w/paddings'],
            segment_ids=example['y_w/segment_ids'],
            segment_pos=example['y_w/segment_pos'],
        ),
        y_l=DPOExampleHalf(
            inputs=example['y_l/inputs'],
            inputs_indicator=example['y_l/inputs_indicator'],
            labels=Labels(
                class_ids=example['y_l/labels/ids'],
                class_weights=example['y_l/labels/weights'],
            ),
            paddings=example['y_l/paddings'],
            segment_ids=example['y_l/segment_ids'],
            segment_pos=example['y_l/segment_pos'],
        ),
    )


class LanguageModelDPO(base_model.BaseModel):
  """Contains a pair of TransformerLM for direct preference optimization.

  This model implicitly optimizes this standard RLHF objective.

  .. math::
    max_{pi} = E_{x ~ D, y ~ pi(x)}(r(y, x) - beta * kl(pi(y|x) | ref(y|x)

  Reference: https://arxiv.org/abs/2305.18290

  Attributes:
    ref_mdl: the reference model.
    mdl: the mdl to be optimized.
    beta: kl regularization weight.
    apply_eval_sample_weights: Boolean indicating whether to apply the per
      example weights from the input `eval_sample_weights` or not.
    model_type: The type of language model based on the tokens visibility.
  """

  ref_mdl: transformer_models.TransformerLm = pax_fiddle.instance_field(
      transformer_models.TransformerLm
  )
  """The reference model. Not back-proped into."""

  mdl: transformer_models.TransformerLm = pax_fiddle.instance_field(
      transformer_models.TransformerLm
  )
  """The model to be optimized."""

  beta: float = 0.1
  """kl divergence regularization weight."""

  token_counter: embedding_softmax.TokenCounter = pax_fiddle.instance_field(
      embedding_softmax.TokenCounter
  )
  """Simple counter for tracking the number of tokens; used for fine-tuning."""

  apply_eval_sample_weights: bool = False
  model_type: LanguageModelType = LanguageModelType.CAUSAL

  def _prepare_predict_data(self, batch: DPOExampleHalf) -> DPOExampleHalf:
    if self.apply_eval_sample_weights:
      assert hasattr(batch, 'eval_sample_weights'), (
          '`apply_eval_sample_weights` enabled, but the input batch does not '
          'provide the necessary `eval_sample_weights` field.'
      )
      batch.weights = _merge_per_token_and_per_example_weights(
          batch.weights, batch.eval_sample_weights
      )

    self.token_counter(batch.inputs, batch.paddings)

    match self.model_type:
      case LanguageModelType.BIDIRECTIONAL:
        batch.causal_attention_mask = jnp.zeros_like(batch.inputs)
      case LanguageModelType.PREFIX:
        batch.causal_attention_mask = 1 - batch.inputs_indicator
      case LanguageModelType.CAUSAL:
        batch.causal_attention_mask = None

    return batch

  # Each batch of input contains two examples: (x, y_l) and (x, y_w) where
  # y_w is preferred over y_l per pairwise preference rating.
  #
  # Since this is a decoder only language model, (x, y_l) and (x, y_w) are both
  # concatenated as one single sequence.
  #
  # Note(yonghui): this implementation doesn't support packing. It is assumed
  # that one sequence (batch element) contains one single example.
  def compute_predictions(self, input_batch: NestedMap) -> Predictions:
    batch = DPOExample.from_feature_converter(input_batch)
    y_l = self._prepare_predict_data(batch.y_l)
    y_w = self._prepare_predict_data(batch.y_w)

    # Ref = reference policy; pi = current policy.
    return NestedMap(
        y_l_ref=self.ref_mdl(**y_l.as_xformer_lm_input()),
        y_l_pi=self.mdl(**y_l.as_xformer_lm_input()),
        y_w_ref=self.ref_mdl(**y_w.as_xformer_lm_input()),
        y_w_pi=self.mdl(**y_w.as_xformer_lm_input()),
    )

  def compute_loss(
      self, predictions, input_batch: NestedMap
  ) -> tuple[Union[WeightedScalars, Metrics], dict[str, Any]]:
    def per_seq_log_p(softmax_out: NestedMap):
      # assert per_example_xent is float32, learning might be unstable in
      # bfloat16.
      assert softmax_out.per_example_xent.dtype == jnp.float32
      assert softmax_out.per_sequence_xent.dtype == jnp.float32
      assert len(softmax_out.per_sequence_xent.shape) == 1
      return -1.0 * softmax_out.per_sequence_xent

    # Prevent backprop into reference model
    y_w_ref_log_p = jax.lax.stop_gradient(per_seq_log_p(predictions.y_w_ref))
    y_l_ref_log_p = jax.lax.stop_gradient(per_seq_log_p(predictions.y_l_ref))
    # Allow backprop into policy model
    y_w_pi_log_p = per_seq_log_p(predictions.y_w_pi)
    y_l_pi_log_p = per_seq_log_p(predictions.y_l_pi)

    self.add_summary('dpo/y_w_ref_log_p', jnp.mean(y_w_ref_log_p))
    self.add_summary('dpo/y_w_pi_log_p', jnp.mean(y_w_pi_log_p))
    self.add_summary('dpo/y_l_ref_log_p', jnp.mean(y_l_ref_log_p))
    self.add_summary('dpo/y_l_pi_log_p', jnp.mean(y_l_pi_log_p))
    add_hist(self, 'dpo/y_w_ref_log_p', y_w_ref_log_p)
    add_hist(self, 'dpo/y_w_pi_log_p', y_w_pi_log_p)
    add_hist(self, 'dpo/y_l_ref_log_p', y_l_ref_log_p)
    add_hist(self, 'dpo/y_l_pi_log_p', y_l_pi_log_p)

    r_hat_y_w = self.beta * (y_w_pi_log_p - y_w_ref_log_p)
    r_hat_y_l = self.beta * (y_l_pi_log_p - y_l_ref_log_p)

    # This is the dpo_loss, same as what equation 7 in the paper computes.
    loss = -1.0 * jnp.mean(jax.nn.log_sigmoid(r_hat_y_w - r_hat_y_l))

    self.add_summary('dpo/r_hat_y_w', jnp.mean(r_hat_y_w))
    self.add_summary('dpo/r_hat_y_w_std', jnp.std(r_hat_y_w))
    self.add_summary('dpo/r_hat_y_l', jnp.mean(r_hat_y_l))
    self.add_summary('dpo/r_hat_y_l_std', jnp.std(r_hat_y_l))
    self.add_summary('dpo/delta_r_hat', jnp.mean(r_hat_y_w - r_hat_y_l))
    self.add_summary('dpo/delta_r_hat_std', jnp.std(r_hat_y_w - r_hat_y_l))
    self.add_summary('dpo/dpo_loss', loss)
    self.add_summary(
        '_dpo_topline/p_correct_ranking',
        jnp.mean(jax.nn.sigmoid(r_hat_y_w - r_hat_y_l)),
    )
    add_hist(self, 'dpo/r_hat_y_w', r_hat_y_w)
    add_hist(self, 'dpo/r_hat_y_l', r_hat_y_l)
    add_hist(self, 'dpo/delta_r_hat', r_hat_y_w - r_hat_y_l)

    batch_size = predictions.y_l_ref.per_example_xent.shape[0]

    # TODO(yonghui): Add diagnostic summaries.
    return (
        NestedMap(
            total_loss=(loss, jnp.array(batch_size, loss.dtype)),
        ),
        {},
    )


class SequenceModel(base_model.BaseModel):
  """Sequence Model base task.

  Attributes:
    model_tpl: Sequence model layer for this task.
    return_predictions: Whether to return predictions during eval. Returning
      predictions is more expensive, but may be useful for debugging.
    decoder_tpl: Parameterization of the decoder.
    label_smoothing_prob: If > 0.0, smooth out one-hot prob by spreading this
      amount of prob mass to all other tokens.
  """

  model_tpl: LayerTpl = template_field(
      transformer_models.TransformerEncoderDecoder
  )
  return_predictions: bool = False
  decoder_tpl: DecoderHParams = base_layer.instance_field(GreedyDecoderHParams)
  label_smoothing_prob: float | None = None

  def setup(self) -> None:
    # Construct the model.
    model_p = self.model_tpl.clone()
    if self.label_smoothing_prob is not None:
      model_p.softmax_tpl.label_smoothing_prob = self.label_smoothing_prob
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
        class_ids=input_batch.tgt.labels, class_weights=input_batch.tgt.weights
    )

    return self.model(
        inputs=input_batch.src.ids,
        input_paddings=input_batch.src.paddings,
        targets=input_batch.tgt.ids,
        target_paddings=input_batch.tgt.paddings,
        labels=labels,
        **packed_input_kwargs,
    )

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
            start_time_step=start_time_step,
        )

      fprop_input_ids = jax.lax.dynamic_slice_in_dim(
          input_batch.tgt.ids, 0, 1, axis=1
      )
      fprop_input_paddings = jnp.ones(
          (batch_size, 1), input_batch.tgt.paddings.dtype
      )
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
      assert isinstance(decoder_params, SampleDecoderHParams)

      def fprop_fn(mdl, ids, paddings):
        del ids, paddings
        mdl.model(
            inputs=input_batch.src.ids,
            input_paddings=input_batch.src.paddings,
            targets=jax.lax.dynamic_slice_in_dim(
                input_batch.tgt.ids, 0, 1, axis=1
            ),
            target_paddings=jax.lax.dynamic_slice_in_dim(
                input_batch.tgt.paddings, 0, 1, axis=1
            ),
        )

      temperature = decoder_params.temperature

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

      prefix_lengths = None
      if 'prefix_lengths' in input_batch:
        prefix_lengths = input_batch.prefix_lengths
      result = sample_decode.sample_decode(
          self,
          extend_step_fn,
          transform_state_fn=transform_decode_state_fn,
          lazy_broadcast_prefix_fn=None,
          next_token_sampler=next_token_sampler,
          prefix_ids=input_batch.tgt.ids,
          prefix_paddings=input_batch.tgt.paddings,
          prefix_lengths=prefix_lengths,
          seq_len=decoder_params.seqlen,
          fprop_fn=fprop_fn,
          num_samples=decoder_params.num_samples,
          fprop_for_prefix=decoder_params.fprop_for_prefix,
          temperature=temperature,
          max_decode_steps=decoder_params.max_decode_steps,
          eos_id=decoder_params.eos_id,
          cf_guidance_scale=decoder_params.cf_guidance_scale,
          controlled_decoding=decoder_params.controlled_decoding,
          sort_samples=decoder_params.sort_samples,
          top_k_recall_target=decoder_params.top_k_recall_target,
          use_top_k_for_logprobs=decoder_params.use_top_k_for_logprobs,
          return_entropy_score=False,
          process_result_fn=decoder_params.process_result_fn,
      )
    elif template_has_type(decoder_params, GreedyDecoderHParams):

      def fprop_fn(mdl, ids, paddings):
        del ids, paddings
        mdl.model(
            inputs=input_batch.src.ids,
            input_paddings=input_batch.src.paddings,
            targets=jax.lax.dynamic_slice_in_dim(
                input_batch.tgt.ids, 0, 1, axis=1
            ),
            target_paddings=jax.lax.dynamic_slice_in_dim(
                input_batch.tgt.paddings, 0, 1, axis=1
            ),
        )

      prefix_lengths = None
      if 'prefix_lengths' in input_batch:
        prefix_lengths = input_batch.prefix_lengths
      result = sample_decode.greedy_decode(
          self,
          extend_step_fn,
          input_batch.tgt.ids,
          input_batch.tgt.paddings,
          decoder_params.seqlen,
          prefix_lengths=prefix_lengths,
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

  def process_decode_out(
      self, input_obj: base_input.BaseInput, decode_out: NestedMap
  ) -> ProcessDecodeOut:
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
          sampled_ids, sampled_lengths, key='tgt'
      )
      sampled_strs = np.reshape(sampled_strs, [batch_size, num_samples])
    else:
      sampled_strs = None

    # Get the first output within a batch.
    decode_out.output_ids = decode_out.output_ids[:, 0, :]
    decode_out.decode_lengths = decode_out.decode_lengths[:, 0]
    decode_out.original_lengths = decode_out.original_lengths[:, 0]
    decode_out.logprobs = decode_out.logprobs[:, :]
    decoded_strs = input_obj.ids_to_strings(
        decode_out.output_ids, decode_out.decode_lengths, key='tgt'
    )
    source_lengths = np.sum(1.0 - decode_out.src.paddings, axis=1).astype(
        np.int32
    )
    source_strs = input_obj.ids_to_strings(
        decode_out.src.ids, source_lengths, key='src'
    )
    target_lengths = np.sum(1.0 - decode_out.tgt.paddings, axis=1).astype(
        np.int32
    )
    target_strs = input_obj.ids_to_strings(
        decode_out.tgt.ids, target_lengths, key='tgt'
    )
    ret = list()
    for idx, decoded_str in enumerate(decoded_strs):
      if (
          hasattr(decode_out, 'eval_sample_weights')
          and not decode_out.eval_sample_weights[idx]
      ):
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
        decode_length=(decode_lengths, np.array(1.0, np.float32))
    )
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
            f'is {example_weights.shape}'
        )
    # Softmax expects weights to be of shape [..., 1].
    softmax_output = self.softmax(
        inputs=features,
        class_weights=example_weights[:, jnp.newaxis],
        class_probabilities=label_probs,
    )
    return NestedMap(
        features=features,
        softmax_output=softmax_output,
        example_weights=example_weights,
    )

  def compute_loss(  # pytype: disable=signature-mismatch  # jax-ndarray
      self, predictions: NestedMap, input_batch: NestedMap
  ) -> tuple[Union[WeightedScalars, Metrics], dict[str, Any]]:
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
        num_predictions=(total_weight, jnp.array(1.0, total_weight.dtype)),
    )
    # Compute top-1 and top-5 accuracy and add summary.
    acc1 = metric_utils.top_k_accuracy(
        1,
        predictions.softmax_output.logits,
        label_probs=label_probs,
        weights=predictions.example_weights,
    )
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
          weights=predictions.example_weights,
      )
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
        labels=jnp.argmax(label_probs, axis=-1),
    )
    return losses, per_example_out, eval_metrics

  def process_decode_out(
      self, input_obj: base_input.BaseInput, decode_out: NestedMap
  ) -> ProcessDecodeOut:
    return NestedMap(), [], NestedMap()


class BertModel(base_model.BaseModel):
  """Bert Model base task.

  Attributes:
    lm: BERT LM layer.
    label_smoothing_prob: If > 0.0, smooth out one-hot prob by spreading this
      amount of prob mass to all other tokens.
    mask_token_id: Mask token id.
    force_mask_generation: if True, always use runtime generated random mask for
      training, even if pre-generated masks exist in the training examples.
  """

  lm_tpl: LayerTpl = template_field(transformer_models.TransformerLm)
  label_smoothing_prob: float | None = None
  mask_token_id: int = 0
  force_mask_generation: bool = False

  def setup(self) -> None:
    super().setup()
    assert self.lm_tpl.model_type == LanguageModelType.BIDIRECTIONAL
    assert self.lm_tpl.packed_input

    lm_tpl = self.lm_tpl.clone()
    if self.label_smoothing_prob is not None:
      lm_tpl.softmax_tpl.label_smoothing_prob = self.label_smoothing_prob

    self.create_child('lm', lm_tpl)

    mlm_augment_p = augmentations.MaskedLmDataAugmenter.config()
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

    labels = NestedMap(class_ids=labels, class_weights=augmented_pos)

    lm_out = self.lm(
        inputs=augmented_labels,
        paddings=paddings,
        labels=labels,
        segment_ids=segment_ids,
        segment_pos=segment_pos,
    )
    lm_out.augmented_labels = augmented_labels
    lm_out.augmented_pos = augmented_pos
    return lm_out

  def compute_loss(  # pytype: disable=signature-mismatch  # jax-ndarray
      self, predictions: NestedMap, input_batch: NestedMap
  ) -> tuple[Union[WeightedScalars, Metrics], dict[str, Any]]:
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
        jnp.amax(input_batch.segment_ids.astype(jnp.float32), axis=1)
    )
    weights = predictions.augmented_pos.astype(jnp.float32)
    predicted_labels = predictions.per_example_argmax.astype(labels.dtype)
    num_preds = predictions.total_weight.astype(jnp.float32)
    mean_acc = jnp.sum((labels == predicted_labels) * weights) / jnp.maximum(
        num_preds, 1
    )
    metric_weight = jnp.array(num_preds, predictions.avg_xent.dtype)
    metrics = py_utils.NestedMap(
        total_loss=(predictions.total_loss, metric_weight),
        avg_xent=(predictions.avg_xent, metric_weight),
        aux_loss=(predictions.aux_loss, metric_weight),
        log_pplx=(predictions.avg_xent, metric_weight),
        fraction_of_correct_preds=(
            mean_acc,
            jnp.array(num_preds, mean_acc.dtype),
        ),
        num_predictions=(num_preds, jnp.array(1.0, num_preds.dtype)),
        num_tokens=(num_tokens, jnp.array(1.0, num_tokens.dtype)),
        num_seqs=(num_seqs, jnp.array(1.0, num_seqs.dtype)),
    )
    per_example_output = py_utils.NestedMap(
        per_sequence_xent=predictions.per_sequence_xent,
        predicted_labels=predicted_labels,
        augmented_labels=predictions.augmented_labels,
    )
    if 'activations' in predictions:
      per_example_output.activations = predictions.activations
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
        class_ids=input_batch.ids[:, :, jnp.newaxis],
    )
    return predictions

  def compute_loss(  # pytype: disable=signature-mismatch  # jax-ndarray
      self, predictions: NestedMap, input_batch: NestedMap
  ) -> tuple[Union[WeightedScalars, Metrics], dict[str, Any]]:
    labels = input_batch.labels
    weights = input_batch.weights
    class_weights = weights[:, :, jnp.newaxis]
    num_preds = jnp.sum(class_weights)
    predicted_labels = predictions.per_example_argmax.astype(labels.dtype)
    mean_acc = jnp.sum((labels == predicted_labels) * weights) / jnp.maximum(
        num_preds, 1
    )
    metrics = NestedMap(
        total_loss=(mean_acc, mean_acc),
    )

    return metrics, NestedMap()


def template_has_type(template, cls):
  return isinstance(template, cls) or (
      isinstance(template, pax_fiddle.Config) and template.cls == cls
  )
