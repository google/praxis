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

"""Definition of specific models."""

from typing import Any, Dict, Tuple

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

BaseHParams = base_layer.BaseLayer.HParams
sub_config_field = base_layer.sub_config_field


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
  """Language Model base task."""

  class HParams(base_model.BaseModel.HParams):
    """Associated hyper-params for this layer class.

    Attributes:
      lm_tpl: LM layer.
      return_predictions: Whether to return predictions during eval. Returning
        predictions is more expensive, but may be useful for debugging.
      decoder_tpl: Parameterization of the decoder.
      model_type: The type of language model based on the tokens visibility.
      count_tokens: Whether to track total tokens trained on in the checkpoint.
      apply_eval_sample_weights: Boolean indicating whether to apply the per
        example weights from the input `eval_sample_weights` or not.
    """
    lm_tpl: BaseHParams = sub_config_field(
        transformer_models.TransformerLm.HParams)
    return_predictions: bool = False
    decoder_tpl: DecoderHParams = pax_fiddle.sub_field(
        GreedyDecoderHParams)
    model_type: LanguageModelType = LanguageModelType.CAUSAL
    count_tokens: bool = False
    apply_eval_sample_weights: bool = False

  def setup(self) -> None:
    super().setup()
    p = self.hparams

    if p.count_tokens:
      tc_p = embedding_softmax.TokenCounter.HParams()
      self.create_child('token_counter', tc_p)

    # Construct the model.
    lm_p = p.lm_tpl.clone()
    lm_p.model_type = p.model_type
    self.create_child('lm', lm_p)

  def _prepare_predict_data(self, input_batch: NestedMap) -> NestedMap:
    p = self.hparams
    paddings = input_batch.paddings
    weights = input_batch.weights
    if p.apply_eval_sample_weights:
      if not hasattr(input_batch, 'eval_sample_weights'):
        logging.warning(
            '`apply_eval_sample_weights` enabled, but the input batch does not '
            'provide the necessary `eval_sample_weights` field.')
      weights = _merge_per_token_and_per_example_weights(
          weights, input_batch.eval_sample_weights)
    inputs = input_batch.ids
    if p.count_tokens:
      self.token_counter(inputs, paddings)
    labels = NestedMap(class_ids=input_batch.labels, class_weights=weights)

    extra_input_kwargs = {}
    if p.lm_tpl.packed_input:
      extra_input_kwargs = {
          'segment_ids': input_batch.segment_ids,
          'segment_pos': input_batch.segment_pos,
      }

    if p.model_type == LanguageModelType.BIDIRECTIONAL:
      causal_attention_mask = jnp.zeros_like(inputs)
    elif p.model_type == LanguageModelType.PREFIX:
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

  def compute_loss(
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
    p = self.hparams
    return compute_xent_loss_helper(predictions, input_batch,
                                    p.return_predictions,
                                    p.apply_eval_sample_weights)

  def _prepare_decode_data(self, input_batch: NestedMap) -> NestedMap:
    p = self.hparams
    batch_size = input_batch.ids.shape[0]

    # TODO(b/229679837): unify prefix_lengths logic to depend on
    # inputs_indicator or paddings only.
    # Get prefix_lengths from inputs_indicator or paddings.
    if 'prefix_lengths' in input_batch:
      prefix_lengths = input_batch.prefix_lengths
    elif p.model_type == transformer_models.LanguageModelType.PREFIX:
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
      minval = jnp.minimum(maxval, p.decoder_tpl.min_prefix_len)
      prefix_lengths = jax.random.randint(self.next_prng_key(), [batch_size],
                                          minval, maxval + 1,
                                          input_batch.ids.dtype)

    if p.model_type == LanguageModelType.BIDIRECTIONAL:
      raise NotImplementedError(type(self))
    elif p.model_type == LanguageModelType.PREFIX:
      if 'inputs_indicator' in input_batch:
        causal_attention_mask = 1 - input_batch.inputs_indicator
      else:
        causal_attention_mask = (
            jnp.arange(input_batch.ids.shape[-1])[jnp.newaxis, :] >=
            prefix_lengths[:, jnp.newaxis])
    else:
      causal_attention_mask = None

    max_prefix_len = input_batch.ids.shape[1]
    if p.decoder_tpl.fprop_for_prefix:
      asserts.not_none(p.decoder_tpl.max_decode_steps)
      seqlen = max_prefix_len + p.decoder_tpl.max_decode_steps
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
      state_padding_size = p.decoder_tpl.max_decode_steps
      # Init input ids and paddings for extend_step.
      input_ids = jnp.pad(fprop_input_ids,
                          [[0, 0], [0, p.decoder_tpl.max_decode_steps]])
      input_paddings = jnp.pad(
          fprop_input_paddings, [[0, 0], [0, p.decoder_tpl.max_decode_steps]],
          constant_values=1.)
      if causal_attention_mask is not None:
        # Pad 1 to the left of causal_attention_mask.
        causal_attention_mask = decoder_utils.right_align_tensors(
            causal_attention_mask, prefix_lengths)
    else:
      seqlen = p.decoder_tpl.seqlen
      start_time_step = 0
      input_ids = input_batch.ids
      input_paddings = input_batch.paddings
      fprop_input_ids = jnp.zeros((batch_size, 1), input_batch.ids.dtype)
      fprop_input_paddings = jnp.ones((batch_size, 1), self.fprop_dtype)
      fprop_segment_pos = None
      fprop_segment_ids = None
      state_padding_size = seqlen - 1

    return NestedMap(
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

  def decode(self,
             input_batch: NestedMap,
             return_result_for_suffix_score=False) -> DecodeOut:
    """Greedy decodes the input_batch.

    Args:
      input_batch: The input batch, with fields `.ids` and `.paddings`. It may
        have an optional `.prefix_lengths` field indicating the lengths of
        prefixes in the ids used as decoding inputs. Optional `.suffix` for the
        suffix_ids with shape [num_suffix, suffix_length]. Optional
        `.temperature` of shape [batch_size] has the temperature for each
        example. If `.temperature` is not set in the input_batch,
        p.decoder_tpl.temperature will be used in sampling decode.
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
    p = self.hparams
    if not isinstance(p.decoder_tpl, DecoderHParams):
      raise ValueError('p.decoder_tpl must be DecoderHParams type, but it is a '
                       f'type of {type(p.decoder_tpl)}')
    if p.decoder_tpl.seqlen <= 0:
      raise ValueError('Must set p.decoder_tpl.seqlen > 0, current value = '
                       f'{p.decoder_tpl.seqlen}')
    max_prefix_len = input_batch.ids.shape[1]
    decode_data = self._prepare_decode_data(input_batch)

    def extend_step_fn(mdl, ids, segment_pos):
      xent = mdl.lm.extend_step(ids, segment_pos=segment_pos)
      return xent.logits

    def transform_decode_state_fn(mdl, transform_fn):
      mdl.lm.transform_decode_state(transform_fn)

    def lazy_broadcast_prefix_fn(mdl, num_suffix_samples, suffix_length):
      mdl.lm.lazy_broadcast_prefix(num_suffix_samples, suffix_length)

    # Flat beam search doesn't work yet.
    if template_has_type(p.decoder_tpl, FlatBeamSearchHParams):
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
      # Pad to full-sequence length.
      self.lm.transform_decode_state(
          decoder_utils.pad_state_fn(decode_data.state_padding_size))
      result = flat_beam_search.flat_beam_search(
          self,
          extend_step_fn,
          decode_data.input_ids,
          decode_data.input_paddings,
          decode_data.seqlen,
          beam_size=p.decoder_tpl.beam_size,
          fprop_dtype=self.fprop_dtype,
          max_decode_steps=p.decoder_tpl.max_decode_steps,
          eos_id=p.decoder_tpl.eos_id,
          length_norm_alpha=p.decoder_tpl.length_norm_alpha)
    elif template_has_type(p.decoder_tpl, BeamSearchHParams):
      assert p.decoder_tpl.fprop_for_prefix

      def fprop_fn(mdl, ids, paddings):
        mdl.lm(
            ids,
            paddings,
            segment_ids=decode_data.fprop_segment_ids,
            segment_pos=decode_data.fprop_segment_pos,
            start_time_step=decode_data.start_time_step,
            causal_attention_mask=decode_data.causal_attention_mask,
        )

      result = beam_search.beam_search(self, extend_step_fn, fprop_fn,
                                       transform_decode_state_fn,
                                       decode_data.fprop_input_ids,
                                       decode_data.fprop_input_paddings,
                                       p.decoder_tpl)
    elif template_has_type(p.decoder_tpl, SampleDecoderHParams):
      # Init cache states, batch size needs to multiply by num_samples.
      self.lm(
          decode_data.fprop_input_ids,
          decode_data.fprop_input_paddings,
          segment_ids=decode_data.fprop_segment_ids,
          segment_pos=decode_data.fprop_segment_pos,
          start_time_step=decode_data.start_time_step,
          causal_attention_mask=decode_data.causal_attention_mask,
          **decode_data.extra_input_kwargs
      )

      if not p.decoder_tpl.lazy_prefix_broadcast:
        # Pad to full-sequence length.
        self.lm.transform_decode_state(
            decoder_utils.pad_state_fn(decode_data.state_padding_size))

      if p.decoder_tpl.k > 0 and p.decoder_tpl.p:
        raise ValueError(
            'In decoder_tpl hparams, k and p should not be set at the'
            ' same time. Currently k is: %s and p is: %s' %
            (p.decoder_tpl.k, p.decoder_tpl.p))

      # Fetch dynamic temperature from input_batch if the input_batch has this
      # information.
      if hasattr(input_batch, 'temperature'):
        temperature = input_batch.temperature
      else:
        temperature = p.decoder_tpl.temperature

      result = sample_decode.sample_decode(
          self,
          extend_step_fn,
          transform_decode_state_fn,
          lazy_broadcast_prefix_fn
          if p.decoder_tpl.lazy_prefix_broadcast else None,
          decode_data.input_ids,
          decode_data.input_paddings,
          decode_data.seqlen,
          num_samples=p.decoder_tpl.num_samples,
          k=p.decoder_tpl.k,
          p=p.decoder_tpl.p,
          fprop_for_prefix=p.decoder_tpl.fprop_for_prefix,
          temperature=temperature,
          max_prefix_len=max_prefix_len,
          max_decode_steps=p.decoder_tpl.max_decode_steps,
          prefix_lengths=decode_data.prefix_lengths,
          eos_id=p.decoder_tpl.eos_id,
          return_result_for_suffix_score=return_result_for_suffix_score,
      )
    elif template_has_type(p.decoder_tpl, GreedyDecoderHParams):
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
      # Pad to full-sequence length.
      self.lm.transform_decode_state(
          decoder_utils.pad_state_fn(decode_data.state_padding_size))
      result = sample_decode.greedy_decode(
          self,
          extend_step_fn,
          decode_data.input_ids,
          decode_data.input_paddings,
          decode_data.seqlen,
          fprop_for_prefix=p.decoder_tpl.fprop_for_prefix,
          max_prefix_len=max_prefix_len,
          max_decode_steps=p.decoder_tpl.max_decode_steps,
          prefix_lengths=decode_data.prefix_lengths,
          eos_id=p.decoder_tpl.eos_id)
    else:
      # Needs to define a decoding algorithm.
      raise NotImplementedError(
          f'Decoding algorithm {type(p.decoder_tpl)} is not implemented.')

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
    # Get the first output within a batch.
    decode_out.output_ids = decode_out.output_ids[:, 0, :]
    decode_out.decode_lengths = decode_out.decode_lengths[:, 0]
    decode_out.original_lengths = decode_out.original_lengths[:, 0]
    decode_out.prefix_ids = decode_out.prefix_ids[:, 0, :]
    decode_out.prefix_lengths = decode_out.prefix_lengths[:, 0]
    decode_out.logprobs = decode_out.logprobs[:, 0, :]
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

      ex = jax.tree_map(lambda x: x[idx], decode_out)  # pylint: disable=cell-var-from-loop
      key = py_utils.get_enumeration_id(ex)
      if not key:
        # not using seqio input's use_enumeration matching
        key = prefix_strs[idx]

      ret.append((key, {
          'prefix': prefix_strs[idx],
          'decoded': decoded_str,
          'original': original_strs[idx],
          'ids': decode_out.output_ids[idx],
          'decoded_ids': decoded_ids,
          'decoded_substr': decoded_substr,
          'logprobs': decode_out.logprobs[idx],
          'prefix_length': prefix_length,
          'decode_length': decode_length,
      }))

    decoded_lengths = np.average(decode_out.decode_lengths).astype(np.float32)
    metrics = NestedMap(
        decoded_length=(decoded_lengths, np.array(1.0, np.float32)))
    out_clu_metrics = NestedMap()
    return metrics, ret, out_clu_metrics


class SequenceModel(base_model.BaseModel):
  """Sequence Model base task."""

  class HParams(base_model.BaseModel.HParams):
    """Associated hyper-params for this layer class.

    Attributes:
      model_tpl: Sequence model layer for this task.
      return_predictions: Whether to return predictions during eval. Returning
        predictions is more expensive, but may be useful for debugging.
      decoder_tpl: Parameterization of the decoder.
      label_smoothing_prob: If > 0.0, smooth out one-hot prob by spreading this
        amount ofprob mass to all other tokens.
    """
    model_tpl: BaseHParams = sub_config_field(
        transformer_models.TransformerEncoderDecoder.HParams)
    return_predictions: bool = False
    decoder_tpl: DecoderHParams = pax_fiddle.sub_field(
        GreedyDecoderHParams)
    label_smoothing_prob: float = 0.0

  def setup(self) -> None:
    super().setup()
    p = self.hparams

    # Construct the model.
    model_p = p.model_tpl.clone()
    self.create_child('model', model_p)

  def compute_predictions(self, input_batch):
    """Computes predictions for `input_batch`."""
    p = self.hparams
    if p.model_tpl.packed_input:
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
    if p.label_smoothing_prob > 0.0:
      vocab_size = p.model_tpl.softmax_tpl.num_classes
      class_probabilities = jax.nn.one_hot(labels.class_ids, vocab_size)
      fill_prob = p.label_smoothing_prob / (vocab_size - 1)
      class_probabilities = (
          (1.0 - p.label_smoothing_prob) * class_probabilities + fill_prob *
          (1.0 - class_probabilities)).astype(self.fprop_dtype)
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
    return compute_xent_loss_helper(predictions, input_batch.tgt,
                                    self.hparams.return_predictions)

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
    p = self.hparams
    if not template_has_type(p.decoder_tpl, DecoderHParams):
      raise ValueError('p.decoder must be DecoderHParams type, but it is a '
                       f'type of {type(p.decoder_tpl)}')
    if p.decoder_tpl.seqlen <= 0:
      raise ValueError('Must set p.decoder_tpl.seqlen > 0, current value = '
                       f'{p.decoder_tpl.seqlen}')
    batch_size = input_batch.tgt.ids.shape[0]

    def extend_step_fn(mdl, ids, segment_pos):
      del segment_pos
      xent = mdl.model.extend_step(ids)
      return xent.logits

    def transform_decode_state_fn(mdl, transform_fn):
      mdl.model.transform_decode_state(transform_fn)

    # Flat beam search doesn't work yet.
    if template_has_type(p.decoder_tpl, FlatBeamSearchHParams):
      raise NotImplementedError('flat beam search not supported')
    elif template_has_type(p.decoder_tpl, BeamSearchHParams):
      assert not p.decoder_tpl.fprop_for_prefix
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
      result = beam_search.beam_search(self, extend_step_fn, fprop_fn,
                                       transform_decode_state_fn,
                                       fprop_input_ids, fprop_input_paddings,
                                       p.decoder_tpl)
    elif template_has_type(p.decoder_tpl, SampleDecoderHParams):
      raise NotImplementedError('sample decode not supported')
    elif template_has_type(p.decoder_tpl, GreedyDecoderHParams):
      self.model(
          inputs=input_batch.src.ids,
          input_paddings=input_batch.src.paddings,
          targets=input_batch.tgt.ids,
          target_paddings=input_batch.tgt.paddings)
      result = sample_decode.greedy_decode(
          self,
          extend_step_fn,
          input_batch.tgt.ids,
          input_batch.tgt.paddings,
          p.decoder_tpl.seqlen,
          eos_id=p.decoder_tpl.eos_id)
    else:
      # Needs to define a decoding algorithm.
      raise NotImplementedError(
          f'Decoding algorithm {type(p.decoder_tpl)} is not implemented.')

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

      ex = jax.tree_map(lambda x: x[idx], decode_out)  # pylint: disable=cell-var-from-loop
      key = py_utils.get_enumeration_id(ex)
      if not key:
        # not using seqio input's use_enumeration matching
        key = source_strs[idx]

      logging.info('SRC: %s\n', source_strs[idx])
      logging.info('TGT: %s\n', target_strs[idx])
      logging.info('OUT: %s\n', decoded_str)
      ret.append((key, {
          'source': source_strs[idx],
          'decoded': decoded_str,
          'target': target_strs[idx],
          'ids': decode_out.output_ids[idx],
          'logprobs': decode_out.logprobs[idx],
          'decode_length': decode_out.decode_lengths[idx],
          # TODO(b/244434890): remove workaround with more robust integration
          'prefix': source_strs[idx],  # for seqio metrics
          'decoded_substr': decoded_str,  # for seqio metrics
      }))
    decode_lengths = np.average(decode_out.decode_lengths).astype(np.float32)
    metrics = NestedMap(
        decode_length=(decode_lengths, np.array(1.0, np.float32)))
    out_clu_metrics = NestedMap()
    return metrics, ret, out_clu_metrics


class ClassificationModel(base_model.BaseModel):
  """Classification task for images and video."""

  class HParams(base_model.BaseModel.HParams):
    """Associated hyper-params for this layer class.

    Attributes:
      network_tpl: The classifier network_tpl, which is ResNet-50 by default.
      softmax_tpl: The softmax_tpl layer used for the classification.
      input_field: The input field which contains the image or video features to
        pass to the classification network.
    """
    network_tpl: BaseHParams = sub_config_field(resnets.ResNet.HParams)
    softmax_tpl: BaseHParams = sub_config_field(
        embedding_softmax.FullSoftmax.HParams)
    input_field: str = 'image'

  def setup(self) -> None:
    super().setup()
    p = self.hparams
    self.create_child('network', p.network_tpl)
    self.create_child('softmax', p.softmax_tpl)

  def compute_predictions(self, input_batch: NestedMap) -> Predictions:
    """Computes predictions for `input_batch`.

    Args:
      input_batch: A `.NestedMap` object containing input tensors to this tower.

    Returns:
      - A NestedMap containing str keys and features, softmax output and the
        class weights as values.
    """
    p = self.hparams
    inputs = input_batch.Get(p.input_field)
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
        class_probabilities=input_batch.label_probs)
    return NestedMap(
        features=features,
        softmax_output=softmax_output,
        example_weights=example_weights)

  def compute_loss(
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
    avg_xent = predictions.softmax_output.avg_xent
    total_weight = predictions.softmax_output.total_weight
    metrics = NestedMap(
        avg_xent=(avg_xent, total_weight),
        num_predictions=(total_weight, jnp.array(1.0, total_weight.dtype)))
    # Compute top-1 and top-5 accuracy and add summary.
    acc1 = metric_utils.top_k_accuracy(
        1,
        predictions.softmax_output.logits,
        label_probs=input_batch.label_probs,
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
          label_probs=input_batch.label_probs,
          weights=predictions.example_weights)
      metrics.update(
          acc5=(acc5, predictions.softmax_output.total_weight),
          error5=(1.0 - acc5, predictions.softmax_output.total_weight),
      )
      self.add_summary('acc5', acc5)
    return metrics, {}

  def predict(self, input_batch: NestedMap) -> Predictions:
    """Computes logits from `input_batch`.

    Args:
      input_batch: A `.NestedMap` object containing input tensors to this tower.

    Returns:
      - A NestedMap containing logits and logp scores.
    """
    p = self.hparams
    inputs = input_batch.Get(p.input_field)
    features = self.network(inputs)
    logits = self.softmax.get_logits(inputs=features)
    logp = self.softmax.logits_to_logp(logits)
    return py_utils.NestedMap(logits=logits, logp=logp)

  def decode(self, input_batch: NestedMap) -> DecodeOut:
    """Computes predictions and runs metrics."""
    predictions = self.compute_predictions(input_batch)
    losses, _ = self.compute_loss(predictions, input_batch)

    per_example_out = NestedMap()
    eval_metrics = NestedMap()

    eval_metrics.accuracy = clu_metrics.Accuracy.from_model_output(
        logits=predictions.softmax_output.logits,
        labels=jnp.argmax(input_batch.label_probs, axis=-1))
    return losses, per_example_out, eval_metrics

  def process_decode_out(self, input_obj: base_input.BaseInput,
                         decode_out: NestedMap) -> ProcessDecodeOut:
    return NestedMap(), [], NestedMap()


class BertModel(base_model.BaseModel):
  """Bert Model base task."""

  class HParams(base_model.BaseModel.HParams):
    """Associated hyper-params for this layer class.

    Attributes:
      lm: BERT LM layer.
      label_smoothing_prob: If > 0.0, smooth out one-hot prob by spreading this
        amount of prob mass to all other tokens.
      mask_token_id: Mask token id.
    """
    lm_tpl: BaseHParams = sub_config_field(
        transformer_models.TransformerLm.HParams)
    label_smoothing_prob: float = 0.0
    mask_token_id: int = 0

  def setup(self) -> None:
    super().setup()
    p = self.hparams
    assert p.lm_tpl.model_type == LanguageModelType.BIDIRECTIONAL
    assert p.lm_tpl.packed_input

    self.create_child('lm', p.lm_tpl)

    mlm_augment_p = augmentations.MaskedLmDataAugmenter.HParams()
    mlm_augment_p.vocab_size = p.lm_tpl.vocab_size
    mlm_augment_p.mask_token_id = p.mask_token_id
    self.create_child('mlm_augmenter', mlm_augment_p)

  def compute_predictions(self, input_batch: NestedMap) -> Predictions:
    """Computes predictions for `input_batch`."""
    p = self.hparams
    assert p.lm_tpl.packed_input
    segment_ids = input_batch.segment_ids
    segment_pos = input_batch.segment_pos
    paddings = input_batch.paddings
    # Note that internal BertTransformer uses input_batch.ids instead.
    labels = input_batch.labels
    if 'masked_ids' in input_batch:
      # Input data already has masking done.
      augmented_labels = input_batch.masked_ids
      augmented_pos = input_batch.masked_pos
    else:
      augmented_labels, augmented_pos = self.mlm_augmenter(labels, paddings)

    if p.label_smoothing_prob > 0.0:
      class_probabilities = jax.nn.one_hot(labels, p.lm.vocab_size)
      fill_prob = p.label_smoothing_prob / (p.lm.vocab_size - 1)
      class_probabilities = (
          (1.0 - p.label_smoothing_prob) * class_probabilities + fill_prob *
          (1.0 - class_probabilities)).astype(self.fprop_dtype)

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

  def compute_loss(
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
  """Language Model task with a simple MLP model."""

  class HParams(base_model.BaseModel.HParams):
    """Associated hyper-params for this layer class.

    Attributes:
      mlp_tpl: MLP model parameters.
      softmax_tpl: Input softmax_tpl embedding lookup layer.
    """
    mlp_tpl: BaseHParams = sub_config_field(linears.MLPBlock.HParams)
    softmax_tpl: BaseHParams = sub_config_field(
        embedding_softmax.SharedEmbeddingSoftmax.HParams)

  def setup(self) -> None:
    super().setup()
    p = self.hparams
    self.create_children('mlp_layers', p.mlp_tpl.clone())
    self.create_child('softmax', p.softmax_tpl.clone())

  def compute_predictions(self, input_batch: NestedMap) -> Predictions:

    input_emb = self.softmax.emb_lookup(input_batch.ids)

    output = self.mlp_layers(input_emb)
    predictions = self.softmax(
        inputs=output,
        class_weights=input_batch.weights[:, :, jnp.newaxis],
        class_ids=input_batch.ids[:, :, jnp.newaxis])
    return predictions

  def compute_loss(
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
