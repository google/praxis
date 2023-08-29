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

"""Base class for all models.

The model solely consists of the network, while the task combines one or several
models with one or several learners/optimizers.
"""

from typing import Any, Sequence

from praxis import base_input
from praxis import base_layer
from praxis import py_utils
from praxis import pytypes

NestedMap = py_utils.NestedMap
JTensor = pytypes.JTensor
Metrics = pytypes.Metrics
WeightedScalars = pytypes.WeightedScalars
Predictions = JTensor | NestedMap | dict[str, Any] | dict[int, Any]

DecodeOut = tuple[WeightedScalars | Metrics, NestedMap, Metrics]
ProcessDecodeOut = tuple[WeightedScalars, Sequence[tuple[str, Any]], Metrics]


class BaseModel(base_layer.BaseLayer):
  """An API that every model should be derived from."""

  def compute_predictions(self, input_batch: NestedMap) -> Predictions:
    """Computes predictions for `input_batch`.

    This method must be defined in a concrete derived class.

    The output can be in the form of probablistic distributions, e.g., softmax
    logits for discrete outputs, mixture of logistics for continuous values, or
    regression values.

    For training/evaluation, the output will be used for computing loss and
    gradient updates, including comparing predicted distributions between
    teacher and student for distillation. During inference the output can be
    used to compute final outputs, perhaps with sampling.

    Args:
      input_batch: A `.NestedMap` object containing input tensors.

    Returns:
      Predictions, either a single Tensor, a `.NestedMap`, or a namedtuple.
    """
    raise NotImplementedError('Abstract method')

  def compute_loss(
      self, predictions: Predictions, input_batch: NestedMap
  ) -> tuple[WeightedScalars | Metrics, dict[str, Any]]:
    """Computes the loss and other metrics for the given predictions.

    This method must be defined in a concrete derived class.

    Args:
      predictions: The output of `compute_predictions`.
      input_batch: A `.NestedMap` object containing input tensors to this tower.

    Returns:
      - Metrics data: This can be either:
        1. WeightedScalars - A dict or NestedMap containing str keys and
          (value, weight) pairs as values, where one or more entries are
          expected to correspond to the loss (or losses).  These values will be
          aggregated upstream as: sum(weight * value) / sum(weight).
        2. clu_metrics - a NestedMap containing str keys and clu_metrics.Metric
          objects.
      - A dict containing arbitrary tensors describing something about each
        training example, where the first dimension of each tensor is the batch
        index.
    """
    raise NotImplementedError('Abstract method')

  def __call__(
      self, input_batch: NestedMap
  ) -> tuple[WeightedScalars | Metrics, dict[str, Any]]:
    """Forward propagation through one tower of the model.

    Args:
      input_batch: A `.NestedMap` object containing input tensors to this tower.

    Returns:
      (dict, dict):

      - Metrics data: This can be either:
        1. WeightedScalars - A dict or NestedMap containing str keys and
          (value, weight) pairs as values, where one or more entries are
          expected to correspond to the loss (or losses).  These values will be
          aggregated upstream as: sum(weight * value) / sum(weight).
        2. clu_metrics - a NestedMap containing str keys and clu_metrics.Metric
          objects.
      - A dict containing arbitrary tensors describing something about each
        training example, where the first dimension of each tensor is the batch
        index.
    """
    predictions = self.compute_predictions(input_batch)
    return self.compute_loss(predictions, input_batch)

  def decode(self, input_batch: NestedMap) -> DecodeOut:
    """Decodes input_batch.

    Args:
      input_batch: The input batch. A `NestedMap` of tensors. Or, if input batch
        spiltting is used, a list of `NestedMap`, one for each split.

    Returns:
      A 3-tuple with:
      - weighted scalars, a NestedMap containing str keys and (value, weight)
        pairs for the current batch (a tuple of two scalars).
      - results, a `.NestedMap` as decoder output.
      - metrics, a NestedMap containing str keys and clu_metrics.Metric
        objects.
    """
    raise NotImplementedError('Abstract method')

  def process_decode_out(self, input_obj: base_input.BaseInput,
                         decode_out: NestedMap) -> ProcessDecodeOut:
    """Processes one batch of decoded outputs.

    Args:
      input_obj: The input object where a tokenizer is accessible.
      decode_out: The output from decode(). May have an extra leading axis.

    Returns:
      A 3-tuple with:
      - weighted scalars, a NestedMap containing str keys and (value, weight)
        pairs for the current batch (a tuple of two scalars).
      - A list of tuples where each element corresponds to a row in the batch.
        Each tuple is a key value pair.
      - metrics, a NestedMap containing str keys and clu_metrics.Metric
        objects. These will run outside of pmap/pjit.
    """
    raise NotImplementedError('Abstract method')
