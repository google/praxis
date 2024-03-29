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

from typing import Any, Union

from absl import logging
import jax
import jax.numpy as jnp
from praxis import base_layer
from praxis import base_model
from praxis import decoder_hparams
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis.layers import transformer_models
from praxis.layers.models import LanguageModel, _merge_per_token_and_per_example_weights

NestedMap = py_utils.NestedMap
JTensor = pytypes.JTensor
Metrics = base_model.Metrics
WeightedScalars = base_model.WeightedScalars
DecoderHParams = decoder_hparams.DecoderHParams
GreedyDecoderHParams = decoder_hparams.GreedyDecoderHParams
LanguageModelType = transformer_models.LanguageModelType
LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]
template_field = pax_fiddle.template_field


def compute_xent_loss_helper(
    predictions: NestedMap,
    input_batch: NestedMap,
    return_predictions: bool,
    apply_eval_sample_weights: bool = False,
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


def compute_eval_metrics_helper(
    metrics: WeightedScalars,
    predictions: NestedMap,
    input_batch: NestedMap,
    apply_eval_sample_weights: bool = False,
    eval_task: str = None,
    boolq_yn_tokens: JTensor = None,
    report_strict_acc: bool = False,  ## legacy
) -> WeightedScalars:
  """Helper for computing the eval metrics for Language model and Sequence model.

  Args:
    predictions: A `.NestedMap` containing the keys `per_example_argmax`,
      `total_loss`, `avg_xent`, `aux_loss`, `total_weight` which corresponds to
      the output of the Softmax layer.
    input_batch: A `.NestedMap` object containing input tensors which contains
      the keys `labels` and `weights` which corresponds to the labels and the
      `weights` for each token in the sequence.
    eval_task: Optional. Supported eval tasks are 'lambada' and 'boolq'.
    report_strict_acc: Legacy. Whether to report strict accuracy. In general,
      this requires the entire portion of the sequence with nonzero weight be
      predicted correctly. Frequently used for eval on the Lambada dataset, in
      which case this metric is equivalent to full-word matching. This is
      equivalent to setting eval_task=='lambada'.
    boolq_yn_tokens: Required when 'eval_task' == 'boolq'. Integers
      corresponding to the tokenizer's "yes" and "no" tokens.

  Returns:
    Metrics dict with eval metrics appended.
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

  if report_strict_acc:
    logging.warning(
        '"report_strict_acc" is present for legacy purposes only. '
        'Please use "eval_task=\'lambada\'" instead.'
    )

  if eval_task == 'lambada' or report_strict_acc:
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

  elif eval_task == 'boolq':

    assert (
        boolq_yn_tokens is not None
    ), "'boolq_yn_tokens' must be set when eval_task=='bool_q'"

    sum_of_labels = jnp.sum(boolq_yn_tokens)

    logits = predictions.logits
    reshaped_logits = jnp.reshape(logits, (-1, logits.shape[-1]))
    reshaped_labels = jnp.reshape(labels, (-1,))
    ll_correct = reshaped_logits[
        jnp.arange(reshaped_labels.shape[0]), reshaped_labels
    ].reshape(
        labels.shape
    )  ## extract the ll of the label for each token
    ll_incorrect = reshaped_logits[
        jnp.arange(reshaped_labels.shape[0]), sum_of_labels - reshaped_labels
    ].reshape(
        labels.shape
    )  ## this will pull out the ll of the "wrong" choice
    num_examples = jnp.sum(input_batch.eval_sample_weights).astype(jnp.float32)

    avg_correct = (
        jnp.sum(((ll_correct - ll_incorrect) > 0) * weights.astype(jnp.float32))
        / num_examples
    )  ## pull out the ll of the target tokens and average
    avg_correct_weight = jnp.array(num_examples, predictions.avg_xent.dtype)
    metrics.boolq_acc = (avg_correct, avg_correct_weight)

  return metrics


class CustomMetricsLM(LanguageModel):

  lm_tpl: LayerTpl = template_field(transformer_models.TransformerLm)
  return_predictions: bool = False
  decoder_tpl: DecoderHParams = base_layer.instance_field(GreedyDecoderHParams)
  model_type: LanguageModelType = LanguageModelType.CAUSAL
  count_tokens: bool = False
  apply_eval_sample_weights: bool = False
  eval_task: str = None
  boolq_yn_tokens: JTensor = None
  report_strict_acc: bool = False

  def setup(self) -> None:
    super().setup()

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
    metrics, per_example_output = compute_xent_loss_helper(
        predictions,
        input_batch,
        self.return_predictions,
        self.apply_eval_sample_weights,
    )
    metrics = compute_eval_metrics_helper(
        metrics,
        predictions,
        input_batch,
        self.apply_eval_sample_weights,
        self.eval_task,
        self.boolq_yn_tokens,
        self.report_strict_acc,
    )
    return metrics, per_example_output
