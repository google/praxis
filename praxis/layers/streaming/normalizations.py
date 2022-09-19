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

"""Streaming aware normalization layers."""

from jax import numpy as jnp

from praxis import py_utils
from praxis import pytypes
from praxis.layers import normalizations
from praxis.layers.streaming import streaming_base

NestedJTensor = pytypes.NestedJTensor


class GroupNorm(normalizations.GroupNorm,  # pytype: disable=signature-mismatch
                streaming_base.StreamingBase):
  """Streaming aware GroupNorm layer."""

  @property
  def group_size(self):
    p = self.hparams
    assert p.min_group_size <= p.dim
    return max(p.dim // p.num_groups, p.min_group_size)

  @property
  def num_groups(self):
    p = self.hparams
    return p.dim // self.group_size

  @classmethod
  def get_right_context(cls, hparams):
    return 0

  @classmethod
  def get_stride(cls, hparams):
    return 1

  def init_states(self,
                  batch_size: int,
                  with_paddings=True):
    p = self.hparams
    num_groups = self.num_groups

    if p.cumulative:
      cached_count_shape = [batch_size, 1, 1]
      cached_moment_shape = [batch_size, 1, num_groups]
      self._update_streaming_state('cached_sum',
                                   jnp.zeros(cached_moment_shape, p.dtype))
      self._update_streaming_state('cached_count',
                                   jnp.zeros(cached_count_shape, p.dtype))
      self._update_streaming_state('cached_var',
                                   jnp.zeros(cached_moment_shape, p.dtype))

  def _streaming_moments(self, inputs, paddings):
    """Computes mean and variance over the valid data points in inputs.

    Args:
      inputs: [B, T, F, N, G] or [B, T, N, G]
      paddings: [B, T, 1, 1, 1] or [B, T, 1, 1] (same rank as inputs)

    Returns:
      mean: [B, T, 1, N, 1] or [B, T, N, 1] (same rank as inputs)
      variance: same shape as mean.
    """

    input_rank = inputs.ndim
    assert paddings.ndim == input_rank

    input_shape = inputs.shape
    output_shape = list(input_shape[:])
    if input_rank == 4:
      # Skip {B,T,N}. Reduce just G.
      reduce_over_dims = [3]
      multiplier = input_shape[3]
      output_shape[3] = 1
    else:
      assert input_rank == 5
      # Skip {B,T,N}. Reduce {F,G}.
      reduce_over_dims = [2, 4]
      multiplier = input_shape[2] * input_shape[4]
      output_shape[2] = 1
      output_shape[4] = 1

    # [B, T, N]
    sum_v = jnp.sum(
        inputs * (1.0 - paddings),
        reduce_over_dims,
        keepdims=False)
    sum_v = jnp.cumsum(sum_v, axis=1)
    sum_v += self.get_streaming_state('cached_sum')

    # [B, T, 1]
    count_v = jnp.sum(
        (1.0 - paddings) * multiplier,
        reduce_over_dims,
        keepdims=False)
    count_v = jnp.cumsum(count_v, axis=1)
    count_v += self.get_streaming_state('cached_count')

    # [B, T, 1, N, 1] or [B, T, N, 1]
    mean = jnp.reshape(sum_v / jnp.maximum(count_v, 1.0), output_shape)

    # [B, T, N]
    sum_vv = jnp.sum(
        jnp.square(inputs-mean) * (1.0 - paddings),
        reduce_over_dims,
        keepdims=False)
    sum_vv = jnp.cumsum(sum_vv, axis=1)
    sum_vv += self.get_streaming_state('cached_var')

    # [B, 1, N]
    self._update_streaming_state('cached_sum', sum_v[:, -1:])

    # [B, 1, 1]
    self._update_streaming_state('cached_count', count_v[:, -1:])

    # [B, 1, N]
    self._update_streaming_state('cached_var', sum_vv[:, -1:])

    # [B, T, 1, N, 1] or [B, T, N, 1]
    variance = jnp.reshape(sum_vv / jnp.maximum(count_v, 1.0), output_shape)
    return mean, variance

  def streaming_step(
      self,
      inputs: NestedJTensor,
  ) -> NestedJTensor:
    """GroupNorm layer in streaming mode.

    Note: even thought it computes normalization coefficients incrementally,
    this approach can have some issues in continuous streaming:
      * It can be applied only on limited in time sequences because if it runs
        in streaming mode too long then it can overflow.
      * It incrementally computes normalization coefficients in time,
        so if properties of the signal changes then it will take some time
        before these coefficients become representative.

    Args:
      inputs: NestedMap with
        * features with shape [batch_size, height, width, channel]
          if p.rank == 4, else [batch, height, channel].
        * paddings with shape [batch_size, height]. Intended to
          be used for sequence processing where `height` is `time`.

    Returns:
      NestedMap with normalized output (same shape with input features)
        and paddings.
    """

    p = self.hparams
    assert p.cumulative
    features = inputs.features
    assert features.ndim == p.input_rank

    group_size = self.group_size
    num_groups = self.num_groups

    input_shape = features.shape

    features = jnp.reshape(features,
                           list(input_shape[:-1]) + [num_groups, group_size])
    expanded_rank = p.input_rank + 1
    expanded_paddings = jnp.reshape(
        inputs.paddings, list(input_shape[:2]) + [1] * (expanded_rank - 2))

    (group_mean,
     group_variance) = self._streaming_moments(features, expanded_paddings)
    features = self._normalize(features, group_mean, group_variance)

    if p.set_padded_output_to_zero and inputs.paddings is not None:
      expanded_paddings = jnp.reshape(
          inputs.paddings,
          list(input_shape[:2]) + [1] * (expanded_rank - 3))
      features *= 1.0 - expanded_paddings
    return py_utils.NestedMap(features=features, paddings=inputs.paddings)
