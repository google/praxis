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

"""Stochastic layers."""

import numbers
from typing import List, Optional, Sequence

import jax
from jax import numpy as jnp
from praxis import base_layer
from praxis import py_utils
from praxis import pytypes

NestedMap = py_utils.NestedMap
JTensor = pytypes.JTensor

BaseHParams = base_layer.BaseLayer.HParams


class Dropout(base_layer.BaseLayer):
  """Apply dropout during training."""

  class HParams(BaseHParams):
    """Associated hyper-params for this layer class.

    Attributes:
      keep_prob: Keep probability.
      noise_shape: A 1-D list of type `int32`, representing the shape for
        randomly generated keep/drop flags. Note that this noise_shape is
        unknown, when building layer params.
      noise_shape_broadcast_dims: A list of dimension where the noise shape is
        broadcasted. For example, noise_shape = [n, h, w, 1] when
        noise_shape_broadcast_dims=[-1].
      dropout_at_eval: Whether or not to also perform dropout at eval time. We
        typically want to replace dropout by expectation during eval. However,
        in certain cases E(f(x)) != f(E(x)), and replacing dropout by its
        expectation during eval leads to worse quality.
    """
    keep_prob: float = 1.0
    noise_shape: Optional[Sequence[int]] = None
    noise_shape_broadcast_dims: Optional[Sequence[int]] = None
    dropout_at_eval: bool = False

  def _dropout(self, inputs: JTensor, noise_shape: List[int]) -> JTensor:
    p = self.hparams
    if noise_shape is None:
      noise_shape = inputs.shape
    prng_key = self.next_prng_key()
    keep_prob = p.keep_prob
    assert keep_prob > 0.0
    random_nums = keep_prob + jax.random.uniform(
        prng_key, noise_shape, inputs.dtype, minval=0.0, maxval=1.0)
    binary_mask = jnp.floor(random_nums)
    return inputs * binary_mask / keep_prob

  def __call__(self, inputs: JTensor) -> JTensor:
    """Applies dropout to inputs.

    Args:
      inputs: The inputs JTensor.

    Returns:
      inputs with dropout applied at training time.
    """
    p = self.hparams
    if isinstance(p.keep_prob, numbers.Real) and p.keep_prob == 1.0:
      return inputs

    if self.do_eval and not p.dropout_at_eval:
      return inputs

    if not p.noise_shape_broadcast_dims:
      noise_shape = p.noise_shape
    else:
      noise_shape = p.noise_shape or list(inputs.shape)
      if isinstance(noise_shape, tuple):
        noise_shape = list(noise_shape)
      for dim in p.noise_shape_broadcast_dims:
        if dim >= len(noise_shape):
          raise ValueError('Invalid broadcasted dim {}'.format(dim))
        noise_shape[dim] = 1

    ret = self._dropout(inputs, noise_shape)
    return ret


class StochasticResidual(base_layer.BaseLayer):
  """Stochastic residual layer that randomly drops the residual branch."""

  class HParams(BaseHParams):
    """Associated hyper-params for this layer class.

    Attributes:
      residual_weight: Residual weight with which to add the reisdual back to
        the input.
      survival_prob: Survival probability of the residual branch while dropping
        out.
    """
    residual_weight: float = 1.0
    survival_prob: float = 1.0

  def _drop_connect(self, inputs: JTensor) -> JTensor:
    """Drops the entire residual layer with given survival probability.

    Args:
      inputs: input `.JTensor` which is on the residual branch which is dropped.

    Returns:
      Dropped out inputs.
    """
    if self.do_eval:
      return inputs

    # Compute tensor.
    prng_key = self.next_prng_key()
    batch_size = inputs.shape[0]
    shape = [batch_size] + [1] * (len(inputs.shape) - 1)
    random_tensor = self.hparams.survival_prob + jax.random.uniform(
        prng_key, shape, dtype=inputs.dtype)
    binary_tensor = jnp.floor(random_tensor)
    # Unlike conventional way that multiply survival_prob at test time, here we
    # divide survival_prob at training time, such that no additional compute is
    # needed at test time.
    output = inputs / self.hparams.survival_prob * binary_tensor
    return output

  def __call__(self, inputs: JTensor, residual: JTensor) -> JTensor:
    """Returns inputs + residual with stochastic dropout.

    Args:
      inputs: input `.JTensor`.
      residual: residual `.JTensor` which is added to input with dropout.

    Returns:
      Output `.JTensor` which is residual added to inputs with dropout.
    """
    return inputs + self.hparams.residual_weight * self._drop_connect(residual)
