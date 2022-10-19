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

"""Module with the stochastic gradient function classes."""

from __future__ import annotations

import abc
from typing import Any, Callable, Tuple

from flax import struct
import jax
from jax import numpy as jnp
import optax
from praxis import base_hyperparams
from praxis import py_utils
from praxis import pytypes

JTensor = pytypes.JTensor
NestedMap = py_utils.NestedMap
PRNGKey = pytypes.PRNGKey


@struct.dataclass
class GradAuxInfo:
  aux_info: Any
  loss_weight: JTensor = 1.0


class BaseStochasticGradient(base_hyperparams.BaseParameterizable,
                             metaclass=abc.ABCMeta):
  """Stochastic gradient function."""

  def process_aux_info(self, aux_info: Any) -> Any:
    """Processes auxiliary info returned by `grad_fn`.

    Args:
      aux_info: Auxiliary info to be processed.

    Returns:
      Processed version of auxiliary info.
    """
    return aux_info

  @abc.abstractmethod
  def grad_fn(self, loss_fn: Callable[..., Any], mdl_vars: Any,
              inputs: NestedMap, prng_key: PRNGKey) -> Tuple[Any, Any]:
    """Main gradients function.

    Intended to accept a loss function, model parameters, input data, and
    a pseudorandom key, and return the loss (possibly with auxiliary info)
    and the gradient of the loss. Based on `jax.value_and_grad`.

    Args:
      loss_fn: Loss function.
      mdl_vars: Model variables.
      inputs: Input examples on which to call `loss_fn`.
      prng_key: A pseudorandom key.

    Returns:
      A tuple ((loss, auxiliary info), gradients).
    """


class StandardGradient(BaseStochasticGradient):
  """Standard gradient function."""

  def grad_fn(self, loss_fn: Callable[..., Any], mdl_vars: Any,
              inputs: NestedMap, prng_key: PRNGKey) -> Tuple[Any, Any]:
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True, allow_int=True)
    (values, aux), grads = grad_fn(mdl_vars, inputs, prng_key)
    aux = self.process_aux_info(aux)
    return (values, aux), grads


class DpSgdStochasticGradient(BaseStochasticGradient):
  """DP-SGD stochastic gradient function."""

  class HParams(BaseStochasticGradient.HParams):
    """Returns the PrivateGradient params."""
    l2_norm_clip: float = None
    noise_multiplier: float = 0.0

  def _clip_and_noise(
      self,
      grads: NestedMap,
      prng_key: PRNGKey = None,
      loss_weight: float = 1.0) -> Tuple[NestedMap, float]:
    p = self.hparams

    grads_flat, grads_treedef = jax.tree_flatten(grads)
    sum_clipped, num_clipped = optax.per_example_global_norm_clip(
        grads=grads_flat, l2_norm_clip=loss_weight * p.l2_norm_clip)
    sum_grads = jax.tree_unflatten(grads_treedef, sum_clipped)
    # Average gradients across all examples
    batch_size = grads_flat[0].shape[0]
    clipped_grads_mean = jax.tree_map(lambda x: x / batch_size, sum_grads)
    frac_clipped = num_clipped / batch_size

    if p.noise_multiplier == 0.0:
      final_grads = clipped_grads_mean
    else:
      noise_stddev = (p.noise_multiplier * loss_weight * p.l2_norm_clip /
                      batch_size)
      prng_keys = jax.random.split(prng_key,
                                   len(jax.tree_leaves(clipped_grads_mean)))
      prng_tree = jax.tree_unflatten(
          jax.tree_structure(clipped_grads_mean), prng_keys)
      final_grads = jax.tree_map(
          lambda x, prng: x + noise_stddev * jax.random.normal(
              prng, shape=x.shape), clipped_grads_mean, prng_tree)

    return final_grads, frac_clipped

  def process_aux_info(self, aux_info):
    aux_info = jax.tree_map(jax.tree_util.Partial(jnp.mean, axis=0), aux_info)
    return aux_info

  def grad_fn(self, loss_fn: Callable[..., Any], mdl_vars: Any,
              inputs: NestedMap, prng_key: PRNGKey) -> Tuple[Any, Any]:
    grad_fn = jax.vmap(
        jax.value_and_grad(loss_fn, has_aux=True, allow_int=True),
        in_axes=(None, 0, None),
        out_axes=0)
    inputs = jax.tree_map(
        jax.tree_util.Partial(jnp.expand_dims, axis=1), inputs)
    (values, aux), grads = grad_fn(mdl_vars, inputs, prng_key)

    aux = self.process_aux_info(aux)
    grads, _ = self._clip_and_noise(grads, prng_key, aux.loss_weight)
    return (values, aux), grads
