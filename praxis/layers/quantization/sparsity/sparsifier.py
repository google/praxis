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

"""Sparse base layer."""

import copy
from typing import Optional

import jax
from jax import numpy as jnp
from praxis import base_layer
from praxis import pytypes
from praxis.layers.quantization.sparsity import sparsity
from praxis.layers.quantization.sparsity import sparsity_hparams
from praxis.layers.quantization.sparsity import sparsity_modes


SparsityType = sparsity_hparams.SparsityType
SparsityHParams = sparsity_hparams.SparsityHParams
WeightHParams = base_layer.WeightHParams
WeightInit = base_layer.WeightInit
WeightSparsityParams = sparsity_hparams.WeightSparsityParams
# Postfix for sparsity mask
SPARSITY_NAME_POSTFIX = base_layer.SPARSITY_NAME_POSTFIX
JTensor = pytypes.JTensor
InferenceMode = sparsity_modes.InferenceMode
FewShotMode = sparsity_modes.FewShotMode
OneShotMode = sparsity_modes.OneShotMode
MaterializeMode = sparsity_modes.MaterializeMode

instance_field = base_layer.instance_field


class SparsityBaseLayer(base_layer.BaseLayer):
  """Sparsity base layer.

  Attributes:
    sparsity: The relevant information related to the kind of sparsity that is
      applied to this layer.
  """

  sparsity: Optional[SparsityHParams] = None

  def create_aux_variables(
      self, name: str, weight_hparams: WeightHParams
  ) -> None:
    if self.sparsity is None or isinstance(self.sparsity.mode, InferenceMode):
      return
    self._create_masks_variables(name, weight_hparams)
    self._create_counter_variables()

  def _create_masks_variables(self, name: str, weight_hp: WeightHParams):
    """Creates mask tensors for sparse variables.

    Args:
      name: Variable name for the weight tensor.
      weight_hp: HParams for weight.
    """
    assert self.sparsity is not None
    sparsity_weight_hp = copy.deepcopy(weight_hp)
    sparsity_weight_hp.init = WeightInit.Constant(True)
    sparsity_weight_hp.dtype = jnp.bool_
    self.create_variable(
        name=name + SPARSITY_NAME_POSTFIX,
        var_hparams=sparsity_weight_hp,
        trainable=False,
    )

  def _create_counter_variables(self):
    """Create variable of num_shot and mask update count."""
    assert self.sparsity is not None
    num_shots = self.sparsity.mode.get_num_shots()
    num_shots_hp = WeightHParams(
        shape=[], init=WeightInit.Constant(num_shots), dtype=jnp.int32
    )
    self.create_variable('num_shots', num_shots_hp, trainable=False)

    count_hp = WeightHParams(
        shape=[], init=WeightInit.Constant(0), dtype=jnp.int32
    )
    # TODO(zhonglinhan): remove these variable to callable sparse scheduler.
    self.create_variable('mask_update_count', count_hp, trainable=False)
    self.create_variable('step', copy.deepcopy(count_hp), trainable=False)

  # TODO(shivaniagrawal): add base layer tests for boundary conditions.
  def _get_sparsity_mask(self, score, mask, step):
    assert self.sparsity is not None

    if self.sparsity.sparsity_type == SparsityType.STRUCTURED_NM:
      if (
          self.sparsity.weight_params is None
          or self.sparsity.weight_params.prune_rate is None
      ):
        return mask
      return sparsity.get_sparsity_mask(
          score,
          n_sparsity=self.sparsity.weight_params.prune_rate[0],
          m_sparsity=self.sparsity.weight_params.prune_rate[1],
          order=self.sparsity.order,
      )

    assert (
        self.sparsity.sparsity_type == SparsityType.UNSTRUCTURED
    ), f'invalid sparsity type {self.sparsity.sparsity_type}'

    prune_rate = self._get_prune_rate_unstructured(step)
    if prune_rate is None:
      return mask
    return sparsity.get_sparsity_mask_unstructured(score, mask, prune_rate)

  def _get_prune_rate_unstructured(self, step):
    prune_rate = (
        self.sparsity.weight_params.prune_rate
        if self.sparsity and self.sparsity.weight_params
        else None
    )
    if self.sparsity is None or self.sparsity.polynomial_decay_schedule is None:
      return prune_rate
    else:
      final_sparsity = self.sparsity.polynomial_decay_schedule.final_sparsity
      init_sparsity = self.sparsity.polynomial_decay_schedule.initial_sparsity
      begin_step = self.sparsity.polynomial_decay_schedule.begin_step
      end_step = self.sparsity.polynomial_decay_schedule.end_step
      exponent = self.sparsity.polynomial_decay_schedule.exponent

      sparsity_active = jax.lax.cond(
          jnp.greater_equal(step, begin_step),
          lambda: 1.0,
          lambda: 0.0,
      )

      return sparsity_active * (
          final_sparsity
          + (init_sparsity - final_sparsity)
          * (
              1
              - (jax.lax.min(step, end_step) - begin_step)
              / (end_step - begin_step)
          )
          ** exponent
      )

  def _maybe_update_mask(
      self,
      weight: JTensor,
      inputs: JTensor,
      name: str,
      step: int,
  ):
    assert self.sparsity is not None

    # Return without updating mask if in MATERIALIZE MODE
    if isinstance(self.sparsity.mode, MaterializeMode):
      return

    mask_var_name = name + SPARSITY_NAME_POSTFIX
    mask = self.get_var(mask_var_name)

    # Reshape if mask and weight have shape mismatch.
    # E.g., this happens in attentions.AttentionProjection when setting
    # attention_combine_dims=True.
    # TODO(shaojinding): Move this reshape to attentions.py if it blocks
    # future refactors on sparse_base_layer.py.
    if mask.shape != weight.shape:
      mask = jnp.reshape(mask, weight.shape)

    update_cnt = self.get_var('mask_update_count')

    if isinstance(self.sparsity.mode, FewShotMode):
      self.sparsity.mode.increment_target_step(update_cnt)

    def mask_update(w, inputs, mask, update_cnt, step):  # pylint: disable=unused-argument
      score = sparsity.compute_score(
          w, score_func=self.sparsity.score, inputs=inputs
      )
      return self._get_sparsity_mask(score, mask, step), update_cnt + 1

    def no_mask_update(w, inputs, mask, update_cnt, step):  # pylint: disable=unused-argument
      return mask, update_cnt

    new_mask, update_cnt = jax.lax.cond(
        self.sparsity.mode.update_cond(step, update_cnt),
        mask_update,
        no_mask_update,
        weight,
        inputs,
        mask,
        update_cnt,
        step,
    )

    self.update_var('mask_update_count', update_cnt)
    self.update_var(mask_var_name, new_mask)

    if isinstance(self.sparsity.mode, OneShotMode) or isinstance(
        self.sparsity.mode, FewShotMode
    ):
      self.add_summary('mask_update_count', update_cnt, verbosity=4)

  def sparsifiy(
      self,
      weight: JTensor,
      name: str,
      inputs: JTensor | None = None,
      layer_idx: int | None = None,
  ) -> JTensor:
    """Get weight of this layer based on mode and other conditions.

    Args:
      weight: tensor to be sparsified, it can be a weight variable.
      name: name of inputs to be sparsified, this is to get corresponding mask.
      inputs: input tensor, i.e., activation of the given weight.
      layer_idx: Layer index.

    Returns:
      variables weights.
    """

    if self.sparsity is None or (
        isinstance(self.sparsity.mode, InferenceMode)
        or self.sparsity.weight_params is None
    ):
      return weight

    step = self.get_var('step')

    # Return without updating mask if in we want to do mixed sparsity for
    # layers and layer index is not in layers to be sparsified
    if (
        self.sparsity.sparsified_layers is not None and layer_idx is not None
    ) and (layer_idx not in self.sparsity.sparsified_layers):
      self.update_var('step', step + 1)
      return weight

    self._maybe_update_mask(
        weight=weight,
        inputs=inputs,
        name=name,
        step=step,
    )

    # NOTE: Mask will be all True (as initialized) for steps before target step
    # [case of few shot/one shot]; and for layer we dont want to sparsify
    # so we apply mask for all the cases.
    mask_var_name = name + SPARSITY_NAME_POSTFIX
    mask = self.get_var(mask_var_name)

    if mask.shape != weight.shape:
      mask = jnp.reshape(mask, weight.shape)
    weight = sparsity.apply_sparsity(weight, mask)

    self.update_var('step', step + 1)

    return weight
