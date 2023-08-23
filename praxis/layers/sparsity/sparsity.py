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

"""Basic functionalities for pruning neural networks implemented in jax."""

import functools
import math

from absl import logging
from flax import linen as nn
import jax
import jax.numpy as jnp
from praxis.layers.sparsity import sparsity_hparams


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
def sr_ste(
    inputs: jnp.ndarray,
    mask: jnp.ndarray,
    update_mask: bool,
    apply_mask: bool,
    sparse_ste_weight: float,
    n_sparsity: int = 0,
    m_sparsity: int = 0,
):
  """Wrapper function for custom derivative rule for structured sparsity.

  Algorithm description: https://arxiv.org/abs/2102.04010

  The last three arguments are forced to be static to simplify
    the implementation.

  Args:
    inputs: Input array for which N:M pruning mask is computed.
    mask: The mask matrix which defines which elements to be pruned.
    update_mask: If True, the mask pattern gets updated.
    apply_mask: If True, the mask is applied to input.
    sparse_ste_weight: Denotes the relative weight for the sparse-refined term.
      As mentioned in the paper (https://arxiv.org/abs/2102.04010), the best
      default value is 0.0002 (lambda_w in the paper).
    n_sparsity: Integer value for N in N:M sparsity.
    m_sparsity: Integer value for M in N:M sparsity.

  Returns:
    The updated input values after applying sparsity.
  """

  return sr_ste_fwd(
      inputs=inputs,
      mask=mask,
      update_mask=update_mask,
      apply_mask=apply_mask,
      sparse_ste_weight=sparse_ste_weight,
      n_sparsity=n_sparsity,
      m_sparsity=m_sparsity,
  )[0]


@functools.partial(jax.jit, static_argnums=(4, 5, 6))
def sr_ste_fwd(
    inputs: jnp.ndarray,
    mask: jnp.ndarray,
    update_mask: bool,
    apply_mask: bool,
    sparse_ste_weight: float,
    n_sparsity: int = 0,
    m_sparsity: int = 0,
) -> tuple[
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
]:
  """Custom forward pass for structured sparsity."""
  # pylint:disable=g-long-lambda
  updated_mask = jax.lax.cond(
      update_mask,
      lambda: get_sparsity_mask(inputs, n_sparsity, m_sparsity),
      lambda: mask,
  )
  updated_inputs = jax.lax.cond(
      apply_mask, lambda: jnp.multiply(updated_mask, inputs), lambda: inputs
  )
  # pylint:enable=g-long-lambda
  return (updated_inputs, updated_mask, jnp.array(sparse_ste_weight)), (
      inputs,
      updated_mask,
      jnp.array(sparse_ste_weight),
  )


def sr_ste_bwd(sparsity_params, n_sparsity, m_sparsity, res, g):
  """Implements custom gradient for backward pass.

  Args:
    sparsity_params: Non-diff arguments as defined in `sr_ste`.
    n_sparsity: Non-diff arguments as defined in `sr_ste`.
    m_sparsity: Non-diff arguments as defined in `sr_ste`.
    res: Residuals computed in sr_ste_fwd.
    g: Default calculated gradients.

  Returns:
    Gradients for differentiable inputs:
      - inputs
      - mask
      - update_mask
      - apply_mask
  """
  del sparsity_params, n_sparsity, m_sparsity
  inputs, updated_mask, ste_weight = res
  # g contains a list of gradients, one per output.
  # g1: updated_inputs
  g1, _, _ = g
  g1 = g1 + ste_weight * jnp.multiply(~updated_mask, inputs)
  return (g1, None, None, None)


sr_ste.defvjp(sr_ste_fwd, sr_ste_bwd)


class Sparsity(nn.Module):
  """Abstract class sparsity for applying sparsity."""

  sparsity_hparams: sparsity_hparams.SparsityHParams

  @nn.compact
  def __call__(
      self,
      inputs: jnp.ndarray,
      *,
      update_mask: bool,
      apply_mask: bool,
      num_update_sparsity: int = 0,
  ) -> jnp.ndarray:
    # TODO(shivaniagrawal): make a decision on creating/not creating mask for
    # when sparsity hparams is None itself.
    if (
        self.sparsity_hparams.weight_params is None
        or self.sparsity_hparams.weight_params.prune_rate is None
    ):
      return inputs

    if self.sparsity_hparams.sparsity_type == 'STRUCTURED_NM':
      n_sparsity = self.sparsity_hparams.weight_params.prune_rate[0]
      m_sparsity = self.sparsity_hparams.weight_params.prune_rate[1]
      if self.sparsity_hparams.weight_params.structure_decay:
        if num_update_sparsity == 1:
          n_sparsity = n_sparsity - 1
        else:
          n_sparsity = int(
              math.ceil(n_sparsity / math.pow(2, num_update_sparsity))
          )
    else:
      logging.info('Unstructured sparsity does not support structure decaying.')
      n_sparsity = 0
      m_sparsity = 0

    # Due to overhead of jit, we limited the number of updates to
    # `num_update_sparsity` to 16. Once we reach to 16, we forcefully set
    # `mask_decay_value` to zero.
    # TODO(ayazdan): Support more than 16 decay.
    weight_params = self.sparsity_hparams.weight_params
    mask_decay_value = 1.0
    if weight_params.mask_decay_weight != 0.0:
      if num_update_sparsity < 16:
        mask_decay_value = max(
            mask_decay_value
            - (num_update_sparsity * weight_params.mask_decay_weight),
            0.0,
        )
      else:
        mask_decay_value = 0.0
    mask = self.variable('sparsity', 'mask', jnp.ones, inputs.shape, jnp.bool_)

    if weight_params.sparse_ste:
      updated_inputs, updated_mask, _ = sr_ste(
          inputs=inputs,
          mask=mask.value,
          update_mask=update_mask,
          apply_mask=apply_mask,
          sparse_ste_weight=weight_params.sparse_ste_weight,
          n_sparsity=n_sparsity,
          m_sparsity=m_sparsity,
      )
      if update_mask and self.has_variable('sparsity', 'mask'):
        mask.value = updated_mask
      return updated_inputs
    else:
      if update_mask and self.has_variable('sparsity', 'mask'):
        mask.value = get_sparsity_mask(inputs, n_sparsity, m_sparsity)

      if apply_mask and self.has_variable('sparsity', 'mask'):
        if weight_params.mask_decay_weight != 0.0:
          return jnp.multiply(
              ~mask.value * mask_decay_value + mask.value, inputs
          )
        else:
          return jnp.where(
              mask.value, inputs, jnp.zeros(inputs.shape, inputs.dtype)
          )
      return inputs


def apply_sparsity(inputs: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
  """Returns sparsified inputs based on input mask."""
  return jnp.where(mask, inputs, jnp.zeros(inputs.shape, inputs.dtype))


def get_sparsity_mask(
    inputs: jnp.ndarray,
    n_sparsity: int = 0,
    m_sparsity: int = 0,
    order: str = 'R',
) -> jnp.ndarray:
  """Returns sparsified inputs for n:m structured pruning.

  Args:
    inputs: Input array for which N:M pruning mask is computed.
    n_sparsity: Maximum number of non-zero values in each block.
    m_sparsity: Number of values in each block.
    order: Apply pruning using this index order. Supported values are `C`, `R`.
      `C` and `R` indicate column-wise and row-wise masking, respectively.
      Default is `R` indicating to applying N:M sparsity across rows of the
      input matrix.

  Returns:
    A mask that indicates the pruning locations (`0`: no pruning, `1`: pruned).
  """
  assert (
      n_sparsity <= m_sparsity
  ), f'N must be lower than M for N:M ({n_sparsity}:{m_sparsity}) sparsity.'
  length = jnp.size(inputs)
  if length % m_sparsity != 0:
    raise ValueError(
        f'inputs size must be divisible by m, provided {length} and'
        f' {m_sparsity}'
    )
  if order not in ['C', 'R']:
    raise ValueError(f'Index order {order} not supported.')

  group = int(length / m_sparsity)
  inputs = jnp.abs(inputs)
  if order == 'R':
    inputs_temp = inputs.reshape(group, m_sparsity, order='C')
  else:
    inputs_temp = jnp.einsum('...ij->...ji', inputs).reshape(
        group, m_sparsity, order='C'
    )
  # Extract the smallest elements and forcefully make them zero.
  _, top_k_indices = jax.lax.top_k(inputs_temp, k=n_sparsity)
  mask = jnp.any(
      jax.nn.one_hot(top_k_indices, m_sparsity, dtype=jnp.bool_), axis=-2
  )

  if order == 'R':
    return mask.reshape(inputs.shape, order='C')
  else:
    if len(inputs.shape) > 2:
      return jnp.einsum('...ij->...ji', mask.reshape(inputs.shape, order='F'))
    else:
      return jnp.einsum('ij->ji', mask).reshape(inputs.shape, order='F')


# TODO(ayazdan): Add support for fast top-k.


def get_sparsity_mask_unstructured(
    inputs: jnp.ndarray,
    mask: jnp.ndarray | None,
    prune_rate: jnp.ndarray | float,
) -> jnp.ndarray:
  """Computes a sparisty mask to prune the required percentage of weights.

  The mask is calculated by thresholding the absolute values of inputs. The
  threshold is the lowest value greater than prune_rate percent of weights, i.e.
  the corresponding percentile.

  The newly pruned weights form a superset of the currently pruned weights if
  the current mask is provided.

  Args:
      inputs: Input tensor.
      mask: Current mask.
      prune_rate: Percentage of weights to prune, value between 0 and 100.

  Returns:
      Sparsity mask.
  """
  if mask is not None:
    inputs = apply_sparsity(inputs, mask)
  inputs_abs = jnp.abs(inputs)
  threshold = jnp.percentile(inputs_abs, prune_rate)
  return jnp.greater(inputs_abs, threshold)


# TODO(shivaniagrawal): Only used for testing the functionality of
# get_prune_mask; update the test to call get_pruning_n_m_mask instead.
def prune_inputs_n_m(
    inputs: jnp.ndarray, *, n: int, m: int, order: str = 'R'
) -> jnp.ndarray:
  """Returns pruned array with N:M (structured) pruning.

  N:M pruning makes at most N values non-zero in each block of M consecutive
  values.

  Args:
    inputs: Input array for which N:M pruning mask is computed.
    n: Maximum number of non-zero values in each block.
    m: Number of values in each block.
    order: Apply pruning using this index order. Supported values are `C`, `R`.
      `C` and `R` indicate column-wise and row-wise masking, respectively.
      Default is `R` indicating to applying N:M sparsity across rows of the
      input matrix.

  Returns:
    An array with the same shape as inputs pruned with N:M strategy.
  """
  mask = get_sparsity_mask(inputs, n, m, order=order)
  return jnp.where(mask, inputs, jnp.zeros(inputs.shape, inputs.dtype))


SparsityScore = sparsity_hparams.SparsityScore


def compute_score(
    weights: jnp.ndarray,
    score_func: SparsityScore = SparsityScore.MAGNITUDE,
    inputs: jnp.ndarray | None = None,
) -> jnp.ndarray:
  """Compute importance score of weight before pruning."""
  if score_func == SparsityScore.ACTIVATION_WEIGHTED:
    if inputs is None:
      raise ValueError('`inputs` must be given for `ACTIVATION_WEIGHTED`.')
    else:
      return score_activation_weighted(weights, inputs)
  elif score_func == SparsityScore.MAGNITUDE:
    return score_weight_magnitude(weights)
  else:
    raise ValueError('Unknown sparsity score function.')


def score_weight_magnitude(weight: jnp.ndarray) -> jnp.ndarray:  # pylint: disable=unused-argument
  """This function returns score based on the magnitude of weights."""

  return jnp.abs(weight)


def score_activation_weighted(
    weight: jnp.ndarray, inputs: jnp.ndarray
) -> jnp.ndarray:
  """This function returns a weighted score of weights based on the average activation magnitude.

  The score is calculated as the product of the weight magnitude and the mean
  magnitude of the activation tensor.

  Args:
    weight: A 2-D weight matrix of shape (C_in, C_out).
    inputs: A N-D tensor where the last channel is C_in.

  Returns:
    A score with the same shape as weight.
  """

  # TODO(wppark): Add support for attention layers as well.
  if not (jnp.ndim(weight) == 2 and inputs.shape[-1] == weight.shape[0]):
    raise ValueError(
        'ACTIVATION_WEIGHTED score function only supports Linear layers for'
        ' now. Weight must be 2-dimensional matrices, and the last channel of'
        ' inputs must have the same number of dimension of the first channel of'
        ' weight.'
    )
  score = jnp.einsum('...j,jk->jk', jnp.abs(inputs), jnp.abs(weight))
  return score
