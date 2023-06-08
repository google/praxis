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
    sparsity_weight_hp.init = WeightInit.Constant(False)
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

  def _schedule_cond(
      self, step: int, target_step: int, mask_update_times: int, num_shots: int
  ) -> JTensor:
    """Sparse mask schedule for one shot and few shot.

    Args:
      step:
      target_step: step to do pruner.
      mask_update_times:
      num_shots:

    Returns:
      A tuple of current mask, and updated count.
    """
    should_do_pruning = jnp.logical_or(
        jnp.equal(num_shots, -1),  # SparsityMode.Training
        jnp.less_equal(mask_update_times, num_shots),  # OneShot/FewShot
    )
    should_pruning_step = jnp.equal(step, target_step)
    return jnp.logical_and(should_pruning_step, should_do_pruning)

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
    return jnp.any(sparsified_layers == layer_idx)

  def _get_sparsity_mask(self, inputs):
    return sparsity.get_sparsity_mask(
        inputs,
        n_sparsity=self.sparsity.weight_params.prune_rate[0],  # pytype: disable=attribute-error
        m_sparsity=self.sparsity.weight_params.prune_rate[1],  # pytype: disable=attribute-error
    )

  def _maybe_sparsify(
      self, inputs: JTensor, name: str, layer_idx: int
  ):
    # Get variables
    mask_var_name = name + SPARSITY_NAME_POSTFIX
    mask = self.get_var(mask_var_name)
    # Reshape if mask and weight have shape mismatch.
    # E.g., this happens in attentions.AttentionProjection when setting
    # attention_combine_dims=True.
    # TODO(shaojinding): Move this reshape to attentions.py if it blocks
    # future refactors on sparse_base_layer.py.
    if mask.shape != inputs.shape:
      mask = jnp.reshape(mask, inputs.shape)
    update_cnt = self.get_var('mask_update_count')
    step = self.get_var('step')

    num_shots = self.sparsity.get_num_shots()
    target_step = (
        self.sparsity.target_step
        + self.sparsity.mask_update_interval * update_cnt
    )
    # Sparsified layers could be set as [1, 2, 3 ...], or None, if None, then
    # set sparsified_layers = [-1] to sparsified all layers.
    if self.sparsity.sparsified_layers is None:
      self.sparsity.sparsified_layers = [-1]
    sparsified_layers = jnp.asarray(self.sparsity.sparsified_layers)

    def mask_update(x, mask, update_cnt):  # pylint: disable=unused-argument
      return self._get_sparsity_mask(x), update_cnt + 1

    def no_mask_update(x, mask, update_cnt):  # pylint: disable=unused-argument
      return mask, update_cnt

    new_mask, update_cnt = jax.lax.cond(
        jnp.logical_and(
            self._schedule_cond(step, target_step, update_cnt, num_shots),
            self._layer_cond(layer_idx, sparsified_layers),
        ),
        mask_update,
        no_mask_update,
        inputs,
        mask,
        update_cnt,
    )
    self.update_var('mask_update_count', update_cnt)
    self.update_var('step', step + 1)
    self.update_var(mask_var_name, new_mask)

    no_op = lambda inputs, new_mask: inputs
    inputs = jax.lax.cond(
        self._layer_cond(layer_idx, sparsified_layers),
        sparsity.apply_sparsity,
        no_op,
        inputs,
        new_mask,
    )
    return inputs

  def sparsifiy(
      self, inputs: JTensor, name: str, layer_idx: Optional[int] = -1
  ) -> JTensor:
    """Get weight of this layer based on mode and other conditions.

    Args:
      inputs: input tensor to be sparsified, it can be a weight variable or
        activation output.
      name: name of inputs to be sparsified, this is to get corresponding mask.
      layer_idx: Layer index.

    Returns:
      variables weights.
    """

    if self.sparsity.mode == SparsityMode.INFERENCE:
      return inputs
    else:
      if self.sparsity.sparsity_type != SparsityType.STRUCTURED_NM:
        raise NotImplementedError(
            'Only structured sparsity is currently supported.'
        )
      return self._maybe_sparsify(inputs=inputs, name=name, layer_idx=layer_idx)
