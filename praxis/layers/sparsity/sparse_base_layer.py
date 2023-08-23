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

import jax
from jax import numpy as jnp
from praxis import base_layer
from praxis import pytypes
from praxis.layers.sparsity import sparsity
from praxis.layers.sparsity import sparsity_hparams

SparsityMode = sparsity_hparams.SparsityMode
SparsityType = sparsity_hparams.SparsityType
SparsityHParams = sparsity_hparams.SparsityHParams
WeightHParams = base_layer.WeightHParams
WeightInit = base_layer.WeightInit
WeightSparsityParams = sparsity_hparams.WeightSparsityParams
# Postfix for sparsity mask
SPARSITY_NAME_POSTFIX = base_layer.SPARSITY_NAME_POSTFIX
JTensor = pytypes.JTensor

instance_field = base_layer.instance_field


class SparsityBaseLayer(base_layer.BaseLayer):
  """Sparsity base layer.

  Attributes:
    sparsity: The relevant information related to the kind of sparsity that is
      applied to this layer.
  """

  sparsity: SparsityHParams = instance_field(SparsityHParams)

  def create_aux_variables(
      self, name: str, weight_hparams: WeightHParams
  ) -> None:
    if self.sparsity.mode == SparsityMode.INFERENCE:
      return
    self._create_masks_variables(name, weight_hparams)
    self._create_counter_variables()

  def _create_masks_variables(self, name: str, weight_hp: WeightHParams):
    """Creates mask tensors for sparse variables.

    Args:
      name: Variable name for the weight tensor.
      weight_hp: HParams for weight.
    """
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
    num_shots = self.sparsity.get_num_shots()
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

  def _update_schedule_cond(
      self,
      step: int,
      target_step: int,
      mask_update_times: int,
      num_shots: int,
      is_update_mode: bool = False,
  ) -> JTensor:
    """Sparse mask update schedule.

    Make mask update when following conditions are all satisfied 1) mode is not
    equal to materialize or inference and 2) mask_update_times <= num_shots for
    one(few) shot, or num_shots == -1 for training mode and 3) current step is
    the target step to do pruning.

    Args:
      step: current step.
      target_step: step to do pruner.
      mask_update_times: number of times sparse masks has been updated.
      num_shots: the target number of times to prune.
      is_update_mode: If this mode to update mask, for inference and
        materialization this should be false.

    Returns:
      A boolean condition results.
    """
    should_do_pruning = jnp.logical_or(
        jnp.equal(num_shots, -1),  # SparsityMode.Training/MATERIALIZE
        jnp.less_equal(mask_update_times, num_shots),  # OneShot/FewShot
    )
    should_pruning_step = jnp.equal(step, target_step)
    return jnp.logical_and(
        jnp.logical_and(should_pruning_step, should_do_pruning), is_update_mode
    )

  def _apply_schedule_cond(self, step, is_always_apply=False):
    """Sparse mask apply schedule.

    Apply mask to tensor when one of following condition is satisfied 1) for
    training or materialize mode; 2) for one(few) shot when current step is
    greater or equal than sparsity.target_step, when is the starting step for
    sparse training.

    Args:
      step: current step.
      is_always_apply: If mask is always applied, for materiaization and
        training this is true.

    Returns:
      A boolean condition results
    """
    should_do_step = jnp.greater_equal(step, self.sparsity.target_step)
    return jnp.logical_or(is_always_apply, should_do_step)

  def _layer_cond(
      self, layer_idx: int, sparsified_layers: jax.Array
  ) -> JTensor:
    """Materialize sparsified weight by applying mask.

    Args:
      layer_idx: Layer index.
      sparsified_layers: Layer indices of sparsified layers.

    Returns:
      Weights maybe materialized by sparsity mask.
    """
    return jnp.logical_or(
        jnp.any(sparsified_layers == layer_idx),
        jnp.any(sparsified_layers == -1),
    )

  def _get_sparsity_mask(self, score, mask, step):
    if self.sparsity.sparsity_type == SparsityType.STRUCTURED_NM:
      return sparsity.get_sparsity_mask(
          score,
          n_sparsity=self.sparsity.weight_params.prune_rate[0],  # pytype: disable=attribute-error
          m_sparsity=self.sparsity.weight_params.prune_rate[1],  # pytype: disable=attribute-error
      )

    # self.sparsity.sparsity_type == SparsityType.UNSTRUCTURED

    prune_rate = self._get_prune_rate_polynomial(step)

    return sparsity.get_sparsity_mask_unstructured(score, mask, prune_rate)

  def _get_prune_rate_polynomial(self, step):
    final_sparsity = self.sparsity.polynomial_decay_schedule.final_sparsity  # pytype: disable=attribute-error
    initial_sparsity = self.sparsity.polynomial_decay_schedule.initial_sparsity  # pytype: disable=attribute-error
    begin_step = self.sparsity.polynomial_decay_schedule.begin_step  # pytype: disable=attribute-error
    end_step = self.sparsity.polynomial_decay_schedule.end_step  # pytype: disable=attribute-error
    exponent = self.sparsity.polynomial_decay_schedule.exponent  # pytype: disable=attribute-error

    sparsity_active = jax.lax.cond(
        jnp.greater_equal(step, begin_step),
        lambda: 1.0,
        lambda: 0.0,
    )

    return sparsity_active * (
        final_sparsity
        + (initial_sparsity - final_sparsity)
        * (
            1
            - (jax.lax.min(step, end_step) - begin_step)
            / (end_step - begin_step)
        )
        ** exponent
    )

  def _get_sparsified_layers(self):
    # Sparsified layers could be set as [1, 2, 3 ...], or None, if None, then
    # set sparsified_layers = [-1] to sparsified all layers.
    if self.sparsity.sparsified_layers is None:
      self.sparsity.sparsified_layers = [-1]
    sparsified_layers = jnp.asarray(self.sparsity.sparsified_layers)
    return sparsified_layers

  def _maybe_update_mask(
      self,
      weight: JTensor,
      inputs: JTensor,
      name: str,
      layer_idx: int,
      step: int,
  ):
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

    num_shots = self.sparsity.get_num_shots()
    target_step = (
        self.sparsity.target_step
        + self.sparsity.mask_update_interval * update_cnt
    )

    sparsified_layers = self._get_sparsified_layers()

    def mask_update(w, inputs, mask, update_cnt, step):  # pylint: disable=unused-argument
      score = sparsity.compute_score(
          w, score_func=self.sparsity.score, inputs=inputs
      )
      return self._get_sparsity_mask(score, mask, step), update_cnt + 1

    def no_mask_update(w, inputs, mask, update_cnt, step):  # pylint: disable=unused-argument
      return mask, update_cnt

    is_update_mode = self.sparsity.mode not in [
        SparsityMode.MATERIALIZE,
        SparsityMode.INFERENCE,
    ]
    new_mask, update_cnt = jax.lax.cond(
        jnp.logical_and(
            self._update_schedule_cond(
                step, target_step, update_cnt, num_shots, is_update_mode
            ),
            self._layer_cond(layer_idx, sparsified_layers),
        ),
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

    if num_shots > 0:
      self.add_summary('mask_update_count', update_cnt, verbosity=4)

  def _maybe_sparsify(
      self,
      weight: JTensor,
      inputs: JTensor,
      name: str,
      layer_idx: int,
      step: int,
  ):
    mask_var_name = name + SPARSITY_NAME_POSTFIX
    mask = self.get_var(mask_var_name)

    if mask.shape != weight.shape:
      mask = jnp.reshape(mask, weight.shape)

    sparsified_layers = self._get_sparsified_layers()

    # Apply mask to weight.
    no_op = lambda inputs, mask: inputs
    is_always_apply = self.sparsity.mode in [
        SparsityMode.MATERIALIZE,
        SparsityMode.TRAINING,
    ]
    weight = jax.lax.cond(
        jnp.logical_and(
            self._apply_schedule_cond(step, is_always_apply),
            self._layer_cond(layer_idx, sparsified_layers),
        ),
        sparsity.apply_sparsity,
        no_op,
        weight,
        mask,
    )
    return weight

  def sparsifiy(
      self,
      weight: JTensor,
      name: str,
      inputs: JTensor | None = None,
      layer_idx: int | None = -1,
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

    if self.sparsity.mode == SparsityMode.INFERENCE:
      return weight

    step = self.get_var('step')

    self._maybe_update_mask(
        weight=weight,
        inputs=inputs,
        name=name,
        layer_idx=layer_idx,
        step=step,
    )

    weight = self._maybe_sparsify(
        weight=weight, inputs=inputs, name=name, layer_idx=layer_idx, step=step
    )

    self.update_var('step', step + 1)

    return weight
