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

"""Loss functions."""

from jax import numpy as jnp
from praxis import base_layer
from praxis import pytypes

from jax_bitempered_loss import loss

WeightHParams = base_layer.WeightHParams
WeightInit = base_layer.WeightInit
JTensor = pytypes.JTensor


class BiTemperedLoss(base_layer.BaseLayer):
  """Bi-tempered logitstic loss.

  Bi-Tempered logistic loss is a generalized softmax cross-entropy loss function
  with a bounded loss value per sample and a heavy-tail softmax probability
  function. Temperature t1 < 1.0 controls the boundedness and t2 > 1.0 controls
  the tail heaviness of the softmax probabilities.

  Temperature pairs (1, 1) correspond to CE loss. The schedule sets the
  temperatures to (1, 1) initially and transitions to (t1, t2) linearly over
  the interval [start_step, end_step]. If end_step == 0, then the temperatures
  are set to (t1, t2) throughout.

  Source: https://bit.ly/3jSol8T
  """

  class HParams(base_layer.BaseLayer.HParams):
    """Associated hyper-params for this layer class.

    Attributes:
      t1: Temperature 1 (log).
      t2: Temperature 2 (exp).
      label_smoothing: Label smoothing.
      start_step: Step number to start transitioning from CE loss.
      end_step: Step number to reach the final temperature pairs (t1, t2). When
        end_step == 0, the temperatures are set to (t1, t2) throughout.
    """
    t1: float = 1.0
    t2: float = 1.0
    label_smoothing: float = 0.0
    start_step: int = 0
    end_step: int = 0

  def setup(self) -> None:
    """Initialize the step variable."""
    p = self.hparams
    assert p.end_step >= p.start_step
    count = WeightHParams(
        shape=[],
        init=WeightInit.Constant(0.0),
        dtype=jnp.float32,
        collections=[base_layer.WeightHParamsCollection.REQUIRES_MEAN_SYNC])
    self.create_variable('count', count, trainable=False)

  def temperature_schedule(self, count: JTensor) -> JTensor:
    """Temperature schedule.

    The temperatures will be set to the final values if end_step == 0.

    Args:
      count: Step number.

    Returns:
      Base schedule.
    """
    p = self.hparams
    count = jnp.array(count).astype(jnp.float32)
    schedule = jnp.where(
        jnp.logical_and(p.end_step > 0, count < p.end_step), 1.0, 0.0)
    schedule = jnp.where(
        count >= p.start_step,
        jnp.maximum(
            1. - (count - p.start_step) /
            jnp.maximum(p.end_step - p.start_step, 1.0), 0.0), schedule)
    return schedule

  def __call__(self, logits: JTensor, labels: JTensor) -> JTensor:
    """Applies bi-tempered loss.

    Args:
      logits: The logits JTensor.  Shaped [..., num_classes].
      labels: The one-hot labels JTensor.  Shaped [..., num_classes].

    Returns:
      Loss values. Shaped either [...] or same as logits/labels but without the
      last dimension of size `num_classes`.
    """
    p = self.hparams
    base_schedule = 0.0
    if not self.do_eval:
      count = self.get_var('count')
      self.update_var('count', count + 1.0)
      base_schedule = self.temperature_schedule(count)
    t1 = 1.0 * base_schedule + p.t1 * (1.0 - base_schedule)
    t2 = 1.0 * base_schedule + p.t2 * (1.0 - base_schedule)
    loss_vals = loss.bi_tempered_logistic_loss(
        logits, labels, t1, t2, label_smoothing=p.label_smoothing)
    return loss_vals
