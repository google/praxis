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

"""Quantizer for vqvae."""

from collections.abc import Sequence
import math

import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import py_utils
from praxis import pytypes

JTensor = pytypes.JTensor
NestedMap = py_utils.NestedMap


def _get_indices(
    dim_indices: JTensor, grouped_base: JTensor, group_size: int
) -> JTensor:
  """Convert from per-dimension indices into global indices in groups.

  The number of dimensions does not need to be divisible by group_size.

  Args:
    dim_indices: per-dimension indices, e.g. [0, 1, 0, 1, 0]
    grouped_base: grouped per-dimension bases, e.g. [1, 2, 4, 1, 2]
    group_size: size of each group, e.g., 3

  Returns:
    global indices, e.g., [2, 1]
  """
  indices = dim_indices * grouped_base
  pad_len = (group_size - indices.shape[-1] % group_size) % group_size
  indices = jnp.concatenate(
      [indices, jnp.zeros((*indices.shape[:-1], pad_len), jnp.uint32)], axis=-1
  )
  indices = indices.reshape(*indices.shape[:-1], -1, group_size).sum(axis=-1)
  if indices.shape[-1] == 1:
    indices = indices[..., 0]
  return indices


def _cartesian_prod(x: JTensor) -> JTensor:
  """Cartesian prod from N x 2 to 2^N."""
  grid = jnp.meshgrid(*x, indexing='ij')
  return jnp.prod(jnp.stack(grid), axis=0).reshape(-1)


def _cal_entropy(prob_fracs: JTensor) -> JTensor:
  """Entropy from probability fractions of bs x N x 2."""
  probs = jax.vmap(_cartesian_prod)(prob_fracs)
  avg_probs = jnp.mean(probs, axis=0)
  group_entropy = -(avg_probs * jnp.log(avg_probs + 1e-5)).sum()
  return group_entropy


def _factorized_entropy_loss(
    affinity: JTensor,
    group_size: int,
    temperature: float,
    average_entropy_weight: float = 1.0,
) -> JTensor:
  """Factorized entropy loss for binary dimensions.

  Sample entropy is exactly factorized into per-dimension calculation.
  Average entropy is approximated within groups of a given size.

  Args:
    affinity: shape [..., dims, num_values]
    group_size: size of each group for average entropy
    temperature: softmax temperature
    average_entropy_weight: weight for average entropy

  Returns:
    sample_entropy - average_entropy
  """
  affinity = affinity.reshape(-1, *affinity.shape[-2:]) / temperature
  pfrac = jax.nn.softmax(affinity, axis=-1)
  log_pfrac = jax.nn.log_softmax(affinity + 1e-5, axis=-1)
  entropy = -(pfrac * log_pfrac).sum(axis=(-1, -2)).mean()

  bs = pfrac.shape[0]
  groups = pfrac.reshape(bs, -1, group_size, 2)
  if groups.shape[1] > 1 and groups.shape[2] > 1:
    groups_2 = pfrac.reshape(bs, group_size, -1, 2).transpose((0, 2, 1, 3))
    groups = jnp.concatenate([groups, groups_2], axis=1)
  group_entropy = jax.vmap(_cal_entropy, in_axes=1)(groups)
  avg_entropy = group_entropy.mean() * average_entropy_weight
  return entropy - avg_entropy


def _entropy_loss(
    affinity,
    loss_type='softmax',
    temperature=1.0,
    sample_minimization_weight=1.0,
    batch_maximization_weight=1.0,
):
  """Calculate the entropy loss."""
  flat_affinity = affinity.reshape(-1, affinity.shape[-1])
  flat_affinity /= temperature
  probs = jax.nn.softmax(flat_affinity, axis=-1)
  log_probs = jax.nn.log_softmax(flat_affinity + 1e-5, axis=-1)
  if loss_type == 'softmax':
    target_probs = probs
  elif loss_type == 'argmax':
    codes = jnp.argmax(flat_affinity, axis=-1)
    onehots = jax.nn.one_hot(
        codes, flat_affinity.shape[-1], dtype=flat_affinity.dtype
    )
    onehots = probs - jax.lax.stop_gradient(probs - onehots)
    target_probs = onehots
  else:
    raise ValueError('Entropy loss {} not supported'.format(loss_type))
  avg_probs = jnp.mean(target_probs, axis=0)
  avg_entropy = -jnp.sum(avg_probs * jnp.log(avg_probs + 1e-5))
  sample_entropy = -jnp.mean(jnp.sum(target_probs * log_probs, axis=-1))
  loss = (sample_minimization_weight * sample_entropy) - (
      batch_maximization_weight * avg_entropy
  )
  return loss


class ScalarQuantizer(base_layer.BaseLayer):
  """Scalar quantizer (SQ)."""

  embedding_dim: int = 8
  dim_widths: Sequence[int] = (8, 8, 4, 4, 4, 4, 4, 4)
  bound_method: str = 'sine'  # 'sine' or 'tanh'
  eps: float = 1e-3  # relatively large due to use of float16

  def _bound_embedding(self, z: JTensor) -> JTensor:
    """Applies bounding function (e.g., tanh, sine)."""
    dim_widths = jnp.asarray(self.dim_widths, np.int32)
    k = dim_widths * (1 - self.eps) / 2
    offset = jnp.where(dim_widths % 2 == 1, 0.0, 0.5)
    if self.bound_method == 'sine':
      return jnp.sin(z) * k - offset
    elif self.bound_method == 'tanh':
      return jnp.tanh(z) * k - offset
    else:
      raise ValueError(f'Bound method {self.bound_method} not supported')

  def _quantize_embedding(self, z: JTensor) -> JTensor:
    """Returns quantized values like cdx.ops.ste_round(z)."""
    # use cdx.ops.ste_round(z) for training
    z_q = jnp.round(z)
    z_q = z + jax.lax.stop_gradient(z_q - z)
    return z_q

  def _get_indices(self, z: JTensor) -> tuple[JTensor, JTensor]:
    """Returns indices (SQ or implied codebook) from quantized values."""
    dim_widths = jnp.asarray(self.dim_widths, np.int32)
    left = -(dim_widths // 2)
    right = dim_widths + left - 1
    # In {left, ..., right}
    clipped = jnp.clip(z, left, right)
    # In {0, ..., left + right}
    zeroed = clipped - left
    if len(self.dim_widths) != zeroed.shape[-1]:
      raise ValueError('Sum of dim_widths and codebook sizes do not match.')
    if zeroed.dtype != jnp.int32:
      zeroed = jnp.round(zeroed).astype(jnp.int32)
      zeroed = jnp.clip(zeroed, 0, dim_widths - 1)
    indices = jnp.zeros(zeroed.shape[:-1], dtype=jnp.int32)
    for i, n in enumerate(self.dim_widths):
      indices *= n
      indices += zeroed[..., i]
    return indices, clipped

  def __call__(self, inputs: JTensor) -> tuple[JTensor, NestedMap]:
    if len(self.dim_widths) != inputs.shape[-1]:
      raise ValueError(
          'Number of dim widths must match channels: '
          f'{len(self.dim_widths)} vs {inputs.shape}'
      )

    dim_widths = jnp.asarray(self.dim_widths, np.int32)
    unquantized = self._bound_embedding(inputs)
    # In {left, ..., right}.
    quantized = self._quantize_embedding(unquantized)
    indices, clipped = self._get_indices(quantized)

    if self.do_eval:
      quantized = clipped
    middle = (dim_widths - dim_widths // 2 * 2 - 1) / 2
    # This might not be integers.
    quantized_centered = quantized - middle
    unquantized_centered = unquantized - middle

    result_dict = NestedMap(
        raw=unquantized_centered,
        quantized=quantized_centered,
        encoding_indices=indices,
        quantizer_loss=0.0,
    )

    return quantized_centered, result_dict

  def decode_ids(self, ids: JTensor) -> JTensor:
    if ids.dtype != jnp.int32:
      ids = jnp.round(ids).astype(jnp.int32)
    vals = []
    dim_widths = jnp.asarray(self.dim_widths, np.int32)
    for dim_width in dim_widths[::-1]:
      vals.append(jnp.mod(ids, dim_width))
      ids = ids // dim_width
    vals = jnp.stack(vals[::-1], axis=-1)
    vals -= dim_widths // 2  # tokens are [0..N), vals are [-N/2..N/2)

    middle = (dim_widths - dim_widths // 2 * 2 - 1) / 2
    vals -= middle
    return vals


class LookupFreeQuantizer(base_layer.BaseLayer):
  """Lookup free quantizer with fixed value sets."""

  embedding_dim: int = 8
  num_token_groups: int = 1
  entropy_loss_ratio: float = 0.1
  commitment_cost: float = 0.25
  entropy_loss_groups: int = (
      math.ceil(embedding_dim / 8) if embedding_dim > 18 else 1
  )
  entropy_loss_type: str = 'softmax'
  entropy_temperature: float = 0.01
  entropy_loss_balance: float = 1.0

  def __call__(self, inputs: JTensor) -> tuple[JTensor, NestedMap]:
    result_dict = NestedMap()
    group_size = int(math.ceil(self.embedding_dim / self.num_token_groups))
    base = jnp.power(
        2, jnp.arange(self.embedding_dim, dtype=jnp.uint32) % group_size
    )
    samples = inputs >= 0
    quantized = jnp.where(samples, 1.0, -1.0)
    indices = _get_indices(samples, base, group_size)
    if not self.do_eval:
      # Commitment loss
      e_latent_loss = (
          jnp.mean((jax.lax.stop_gradient(quantized) - inputs) ** 2)
          * self.commitment_cost
      )

      # Entropy loss
      entropy_loss = jnp.asarray(0.0, dtype=jnp.float32)
      if self.entropy_loss_ratio != 0:
        groups = self.entropy_loss_groups
        loss_type = self.entropy_loss_type
        temperature = self.entropy_temperature
        weight = self.entropy_loss_balance
        if groups > 1:
          # Factorized entropy loss
          assert (
              loss_type == 'softmax'
          ), f'Entropy loss type {loss_type} not supported in grouped mode.'
          group_size = self.embedding_dim // groups
          affinity = jnp.stack([2 * inputs, -2 * inputs], axis=-1)
          entropy_loss = _factorized_entropy_loss(
              affinity, group_size, temperature, weight
          )
        else:  # groups == 1:
          codebook_base = [
              np.array([-1.0, 1.0]) for _ in range(self.embedding_dim)
          ]
          codebook = np.stack(
              [*reversed(np.meshgrid(*codebook_base, indexing='ij'))], axis=-1
          )
          codebook = codebook.reshape(-1, self.embedding_dim)
          distances = -2 * inputs @ jnp.asarray(codebook.T)
          entropy_loss = _entropy_loss(
              -distances,
              loss_type=loss_type,
              temperature=temperature,
              batch_maximization_weight=weight,
          )
        entropy_loss *= self.entropy_loss_ratio

      loss = e_latent_loss + entropy_loss
      result_dict.quantizer_loss = loss
      result_dict.e_latent_loss = e_latent_loss
      result_dict.entropy_loss = entropy_loss

      # Straight-through gradient
      quantized = inputs + jax.lax.stop_gradient(quantized - inputs)

    result_dict.encodings = quantized
    result_dict.encoding_indices = indices
    result_dict.raw = inputs
    return quantized, result_dict

  def decode_ids(self, inputs: JTensor) -> JTensor:
    group_size = int(math.ceil(self.embedding_dim / self.num_token_groups))
    base = jnp.power(
        2, jnp.arange(self.embedding_dim, dtype=jnp.uint32) % group_size
    )
    if self.num_token_groups == 1:
      inputs = inputs[..., None]
    expanded_x = jnp.repeat(inputs, group_size, axis=-1)[..., : base.shape[0]]
    return jnp.where(expanded_x & base, 1.0, -1.0)
