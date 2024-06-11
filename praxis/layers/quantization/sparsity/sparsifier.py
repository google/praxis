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
import functools
from typing import Optional, Sequence

import jax
from jax import numpy as jnp
from praxis import base_layer
from praxis import pytypes
from praxis.layers.quantization.sparsity import sparsity
from praxis.layers.quantization.sparsity import sparsity_hparams
from praxis.layers.quantization.sparsity import sparsity_modes


SPARSITY_METADATA_SUFFIX = '_sparsity_metadata'
SPARSITY_NZ_SUFFIX = '_sparsity_nz'
SPARSITY_CONFIG_SUFFIX = '_sparsity_config'
SPARSITY_PRUNED_VALUE_SUFFIX = '_sparsity_mask_pruned_value'

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


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def sr_ste(
    inputs: jnp.ndarray,
    mask: jnp.ndarray,
    sparse_ste_weight: float,
):
  """Wrapper function for custom derivative rule for structured sparsity.

  Algorithm description: https://arxiv.org/abs/2102.04010

  The last argument is forced to be static to simplify
    the implementation.

  Args:
    inputs: Input array for which N:M pruning mask is computed.
    mask: The mask matrix which defines which elements to be pruned.
    sparse_ste_weight: Denotes the relative weight for the sparse-refined term.
      As mentioned in the paper (https://arxiv.org/abs/2102.04010), the best
      default value is 0.0002 (lambda_w in the paper).

  Returns:
    The updated input values after applying sparsity.
  """

  return sr_ste_fwd(
      inputs=inputs,
      mask=mask,
      sparse_ste_weight=sparse_ste_weight,
  )[0]


@functools.partial(jax.jit, static_argnums=(2,))
def sr_ste_fwd(
    inputs: jnp.ndarray,
    mask: jnp.ndarray,
    sparse_ste_weight: float,
) -> tuple[
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
]:
  """Custom forward pass for structured sparsity."""
  updated_inputs = jnp.multiply(mask, inputs)
  # pylint:enable=g-long-lambda
  return (updated_inputs, mask, jnp.array(sparse_ste_weight)), (
      inputs,
      mask,
      jnp.array(sparse_ste_weight),
  )


def sr_ste_bwd(sparsity_params, res, g):
  """Implements custom gradient for backward pass.

  Args:
    sparsity_params: Non-diff arguments as defined in `sr_ste`.
    res: Residuals computed in sr_ste_fwd.
    g: Default calculated gradients.

  Returns:
    Gradients for differentiable inputs:
      - inputs
      - mask
  """
  del sparsity_params
  inputs, mask, ste_weight = res
  # g contains a list of gradients, one per output.
  # g1: updated_inputs
  g1, _, _ = g
  g1 = g1 + ste_weight * jnp.multiply(~mask, inputs)
  return (g1, None)


sr_ste.defvjp(sr_ste_fwd, sr_ste_bwd)


class SparsityBaseLayer(base_layer.BaseLayer):
  """Sparsity base layer.

  Attributes:
    sparsity: The relevant information related to the kind of sparsity that is
      applied to this layer.
  """

  sparsity: Optional[SparsityHParams] = None

  def create_sparsity_variables(
      self,
      name: str,
      weight_hparams: WeightHParams,
      scale_shape: Optional[Sequence[int]] = None,
  ) -> None:
    if self.sparsity is None or isinstance(self.sparsity.mode, InferenceMode):
      return
    self._create_masks_variables(name, weight_hparams, scale_shape=scale_shape)
    self._create_counter_variables()

  def _create_masks_variables(
      self,
      name: str,
      weight_hp: WeightHParams,
      scale_shape: Optional[Sequence[int]] = None,
  ):
    """Creates mask tensors for sparse variables.

    Args:
      name: Variable name for the weight tensor.
      weight_hp: HParams for weight.
      scale_shape: Shape of the scales.
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
    if self.sparsity.topk_estimator_type:
      # create learnable mask parameters for top-k methods
      assert (
          scale_shape is not None
      ), 'scale_shape is required for top-k methods.'
      sparsity_mask_hp = copy.deepcopy(weight_hp)
      self.set_up_weights(
          weight_name='w_mask',
          weight_params=sparsity_mask_hp,
          scale_shape=scale_shape,
      )
      # store intermediate masks in floats generated by top-k methods.
      sparsity_mask_hp = copy.deepcopy(weight_hp)
      sparsity_mask_hp.init = WeightInit.Constant(1.0)
      sparsity_mask_hp.dtype = jnp.float_
      sparsity_mask_hp.collections = [
          base_layer.WeightHParamsCollection.REQUIRES_MEAN_SYNC
      ]
      self.create_variable(
          name=name + SPARSITY_NAME_POSTFIX + '_float',
          var_hparams=sparsity_mask_hp,
          trainable=False,
      )
    if self.sparsity.weight_params is not None and (
        self.sparsity.weight_params.pruned_value != 0.0
        or self.sparsity.weight_params.pruned_value_trainable
    ):
      prune_value_hp = WeightHParams(
          shape=(),
          init=WeightInit.Constant(
              scale=self.sparsity.weight_params.pruned_value
          ),
          dtype=jnp.float32,
      )
      self.create_variable(
          name=name + SPARSITY_PRUNED_VALUE_SUFFIX,
          var_hparams=prune_value_hp,
          trainable=self.sparsity.weight_params.pruned_value_trainable,
      )

  def _create_counter_variables(self):
    """Create variable of num_shot and mask update count."""
    assert self.sparsity is not None
    num_shots = self.sparsity.mode.get_num_shots()
    num_shots_hp = WeightHParams(
        shape=[],
        init=WeightInit.Constant(num_shots),
        dtype=jnp.int32,
        collections=[base_layer.WeightHParamsCollection.REQUIRES_MEAN_SYNC],
    )
    self.create_variable('num_shots', num_shots_hp, trainable=False)

    count_hp = WeightHParams(
        shape=[],
        init=WeightInit.Constant(0),
        dtype=jnp.int32,
        collections=[base_layer.WeightHParamsCollection.REQUIRES_MEAN_SYNC],
    )
    # TODO(zhonglinhan): remove these variable to callable sparse scheduler.
    self.create_variable('mask_update_count', count_hp, trainable=False)
    self.create_variable('step', copy.deepcopy(count_hp), trainable=False)

    # create variable for tracking sparse architecture divergence (SAD) metric
    # proposed in https://arxiv.org/abs/2102.04010
    if self.sparsity.track_sad_metric:
      sad_hp = WeightHParams(
          shape=[],
          init=WeightInit.Constant(0.0),
          dtype=jnp.float32,
          collections=[base_layer.WeightHParamsCollection.REQUIRES_MEAN_SYNC],
      )
      self.create_variable(
          'sparse_architecture_divergence', sad_hp, trainable=False
      )

  # TODO(shivaniagrawal): add base layer tests for boundary conditions.
  def _get_sparsity_mask(self, score, mask, step, input_dtype):
    assert self.sparsity is not None

    if self.sparsity.sparsity_type == SparsityType.STRUCTURED_NM:
      if (
          self.sparsity.weight_params is None
          or self.sparsity.weight_params.prune_rate is None
      ):
        return mask
      if (
          sparsity.is_optimized_offset(
              self.sparsity.order,
              self.sparsity.weight_params.offset,
              input_dtype,
          )
          and not self.sparsity.block_size
      ):
        return sparsity.get_sparsity_mask_optimized_for_offset(
            score,
            n_sparsity=self.sparsity.weight_params.prune_rate[0],
            m_sparsity=self.sparsity.weight_params.prune_rate[1],
            order=self.sparsity.order,
            offset=self.sparsity.weight_params.offset,
        )

      return sparsity.get_sparsity_mask(
          score,
          n_sparsity=self.sparsity.weight_params.prune_rate[0],
          m_sparsity=self.sparsity.weight_params.prune_rate[1],
          order=self.sparsity.order,
          offset=self.sparsity.weight_params.offset,
          block_size=self.sparsity.block_size,
      )

    if self.sparsity.sparsity_type == SparsityType.CHANNELWISE_PRUNING:
      if (
          self.sparsity.weight_params is None
          or self.sparsity.weight_params.prune_rate is None
      ):
        return mask
      return sparsity.get_sparsity_mask_channelwise(
          score,
          self.sparsity.weight_params.prune_rate,
          self.sparsity.channelwise_pruning_dim,
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
      input_dtype = w.dtype
      new_mask = self._get_sparsity_mask(score, mask, step, input_dtype)

      sad_score = None
      if self.sparsity.track_sad_metric:
        # compute sad = #mask-changes / #params
        sad_score = jnp.sum(jnp.logical_xor(mask, new_mask)) / jnp.size(mask)

      return (
          new_mask,
          update_cnt + 1,
          sad_score,
      )

    def no_mask_update(w, inputs, mask, update_cnt, step):  # pylint: disable=unused-argument
      sad_score = None
      if self.sparsity.track_sad_metric:
        # sad metric becomes zero as there are no mask-changes
        sad_score = jnp.zeros((), dtype=jnp.float32)

      return mask, update_cnt, sad_score

    new_mask, update_cnt, sad_score = jax.lax.cond(
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
    if self.sparsity.track_sad_metric:
      self.update_var('sparse_architecture_divergence', sad_score)

    if isinstance(self.sparsity.mode, OneShotMode) or isinstance(
        self.sparsity.mode, FewShotMode
    ):
      self.add_summary('mask_update_count', update_cnt, verbosity=4)

  def sparsify_via_top_k_estimator(
      self,
      weight: JTensor,
      name: str,
      step: int,
  ) -> JTensor:
    """Sparsify weight based on top-k estimator type and other conditions.

    Args:
      weight: tensor to be sparsified, it can be a weight variable.
      name: name of inputs to be sparsified, this is to get corresponding mask.
      step: current step.

    Returns:
      variables weights.
    """
    assert self.sparsity is not None

    binary_mask_var_name = name + SPARSITY_NAME_POSTFIX
    binary_mask = self.get_var(binary_mask_var_name)

    float_mask_var_name = name + SPARSITY_NAME_POSTFIX + '_float'
    float_mask = self.get_var(float_mask_var_name)

    mask_params = self.theta.w_mask

    # Reshape if mask and weight have shape mismatch.
    # E.g., this happens in attentions.AttentionProjection when setting
    # attention_combine_dims=True.
    if binary_mask.shape != weight.shape:
      binary_mask = jnp.reshape(binary_mask, weight.shape)
    if float_mask.shape != weight.shape:
      float_mask = jnp.reshape(float_mask, weight.shape)

    update_cnt = self.get_var('mask_update_count')

    if isinstance(self.sparsity.mode, FewShotMode):
      self.sparsity.mode.increment_target_step(update_cnt)

    def no_mask_update(
        binary_mask, weight, update_cnt, mask_params, float_mask  # pylint: disable=unused-argument
    ):
      sad_score = None
      if self.sparsity.track_sad_metric:
        # sad metric becomes zero as there are no mask-changes
        sad_score = jnp.zeros((), dtype=jnp.float32)

      # apply mask on weight
      weight = weight * float_mask

      return weight, update_cnt, float_mask, binary_mask, sad_score

    def mask_update(binary_mask, weight, update_cnt, mask_params, float_mask):  # pylint: disable=unused-argument
      # get sparsity constraints
      sparsity_n = self.sparsity.weight_params.prune_rate[0]
      sparsity_m = self.sparsity.weight_params.prune_rate[1]

      # reshape the mask_params to groups
      num_params = jnp.size(mask_params)
      num_groups = int(num_params / sparsity_m)
      mask_params_grouped = mask_params.reshape(num_groups, sparsity_m)

      # identify top n items from each group
      _, top_k_indices = jax.lax.top_k(mask_params_grouped, k=sparsity_n)

      # create binary mask based on top n items
      new_mask = jnp.any(
          jax.nn.one_hot(top_k_indices, sparsity_m, dtype=jnp.bool_), axis=-2
      )

      cur_binary_mask = None
      cur_float_mask = None
      if self.sparsity.topk_estimator_type == 'BINARY_MASK':
        # reshape binary_mask to original weight shape
        new_mask = new_mask.reshape(mask_params.shape)

        # straight-through estimator logic
        zero = mask_params - jax.lax.stop_gradient(mask_params)
        cur_float_mask = zero + jax.lax.stop_gradient(new_mask)

        # create binary mask
        cur_binary_mask = cur_float_mask == 1.0
      elif self.sparsity.topk_estimator_type == 'PROB_MASK':
        # set the binary mask
        cur_binary_mask = new_mask.reshape(weight.shape)

        # renormalize mask_params by setting -inf for pruned weight
        renorm_mask_params_grouped = jnp.where(
            new_mask,
            mask_params_grouped,
            jnp.full(
                mask_params_grouped.shape,
                -jnp.inf,
                dtype=mask_params_grouped.dtype,
            ),
        )

        # convert to prob distrib. via softmax
        prob_mask_params_grouped = jax.nn.softmax(
            renorm_mask_params_grouped, axis=-1
        )

        # scale probs by #unpruned params
        prob_mask_params_grouped = prob_mask_params_grouped * sparsity_n

        # reshape it to original size
        cur_float_mask = prob_mask_params_grouped.reshape(weight.shape)

      # apply mask on weight
      weight = weight * cur_float_mask

      sad_score = None
      if self.sparsity.track_sad_metric:
        # compute sad = #mask-changes / #params
        sad_score = jnp.sum(
            jnp.logical_xor(binary_mask, cur_binary_mask)
        ) / jnp.size(cur_binary_mask)

      return weight, update_cnt, cur_float_mask, cur_binary_mask, sad_score

    weight, update_cnt, new_float_mask, new_binary_mask, sad_score = (
        jax.lax.cond(
            self.sparsity.mode.update_cond(step, update_cnt),
            mask_update,
            no_mask_update,
            binary_mask,
            weight,
            update_cnt,
            mask_params,
            float_mask,
        )
    )

    self.update_var('mask_update_count', update_cnt)
    self.update_var(binary_mask_var_name, new_binary_mask)
    self.update_var(float_mask_var_name, new_float_mask)
    if self.sparsity.track_sad_metric:
      self.update_var('sparse_architecture_divergence', sad_score)

    if isinstance(self.sparsity.mode, OneShotMode) or isinstance(
        self.sparsity.mode, FewShotMode
    ):
      self.add_summary('mask_update_count', update_cnt, verbosity=4)

    return weight

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

    assert (
        self.sparsity.sparsity_type != SparsityType.CHANNELWISE_PRUNING
    ), 'Channel-wise pruning is temporarily disabled'

    step = self.get_var('step')

    # Return without updating mask if in we want to do mixed sparsity for
    # layers and layer index is not in layers to be sparsified
    if (
        self.sparsity.sparsified_layers is not None and layer_idx is not None
    ) and (layer_idx not in self.sparsity.sparsified_layers):
      self.update_var('step', step + 1)
      return weight

    # Perform mask learning with top-k estimator if topk_estimator_type is set
    if self.sparsity.topk_estimator_type:
      weight = self.sparsify_via_top_k_estimator(
          weight=weight, name=name, step=step
      )
      self.update_var('step', step + 1)
      return weight

    self._maybe_update_mask(
        weight=weight,
        inputs=inputs,
        name=name,
        step=step,
    )

    # NOTE: Mask will be all True (as initialized) for steps before target step
    # [case of few shot/one shot]; and for layer we don't want to sparsify
    # so we apply mask for all the cases.
    mask_var_name = name + SPARSITY_NAME_POSTFIX
    mask = self.get_var(mask_var_name)

    if mask.shape != weight.shape:
      mask = jnp.reshape(mask, weight.shape)

    pruned_value = None
    if (
        self.sparsity.weight_params.pruned_value != 0.0
        or self.sparsity.weight_params.pruned_value_trainable
    ):
      pruned_value_var_name = name + SPARSITY_PRUNED_VALUE_SUFFIX
      if self.sparsity.weight_params.pruned_value_trainable:
        pruned_value = getattr(self.theta, pruned_value_var_name)
      else:
        pruned_value = self.get_var(pruned_value_var_name)

    if self.sparsity.weight_params.sparse_ste:
      weight, _, _ = sr_ste(
          weight, mask, self.sparsity.weight_params.sparse_ste_weight
      )
    else:
      weight = sparsity.apply_sparsity(
          weight,
          mask,
          pruned_value=pruned_value,
      )

    self.update_var('step', step + 1)

    return weight
