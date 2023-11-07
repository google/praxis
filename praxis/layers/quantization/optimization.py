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

"""Optimizations for quantization."""

import jax
from jax import numpy as jnp
import numpy as np
from praxis import pytypes

JTensor = pytypes.JTensor


def _get_scan_range() -> np.ndarray:
  # Produce candidate scan values.
  return np.linspace(1.0, 0.5, num=11)


def _get_mean_error(bound, t, min_value, max_value, p_value):
  scale = bound / max_value
  candidate = jnp.divide(t, scale)
  candidate = jnp.clip(jnp.round(candidate), min_value, max_value)
  candidate = jnp.multiply(candidate, scale)
  pmean_error = jnp.mean(jnp.abs(jnp.subtract(candidate, t)) ** p_value)
  return pmean_error


def _quantrow(
    vec: JTensor,
    bound: JTensor,
    min_value: float,
    max_value: float,
    p_value: float,
    factors: np.ndarray,
) -> JTensor:
  """Get best rescaling factor from a list of factors applied a channel.

  Args:
    vec: The vector in a channel.
    bound: The hard bound (max(abs(vec))) of the vector.
    min_value: The target min value.
    max_value: The target max value.
    p_value: Exponent of the p-mean error metric.
    factors: The values to be applied on top of bound.

  Returns:
    adjusted bound value out of the list of factors applied to bound.
  """

  def _quant(bounds):
    return _get_mean_error(bounds, vec, min_value, max_value, p_value)

  diffs = jax.vmap(_quant)(bound * factors)
  best_scaling = factors[jnp.argmin(diffs)]
  return bound * best_scaling


def _get_best_bound_per_tensor(
    t: JTensor,
    bound: JTensor,
    min_value: float,
    max_value: float,
    p_value: float = 1.0,
) -> JTensor:
  """Scan around [0.5, 1] * hard max value to get bound value for whole tensor.

  This does a scan to get bound value(s) that minimize mean absolute error (MAE)
  between original tensor 't' and quantized tensor. It's (almost) equivalent to
  maximizing entropy.

  Args:
    t: The input float tensor.
    bound: The hard max value for tensor 't'. It has the same length as shape.
    min_value: Minimal value for the quantization bound.
    max_value: Maximal value for the quantization bound.
    p_value: Exponent of the p-mean error metric. Default to 1.0 which is MAE.

  Returns:
    The best bound values for 't', that minimize p-mean error.
  """

  def _quant(scaling_factors):
    return _get_mean_error(
        bound * scaling_factors, t, min_value, max_value, p_value
    )

  scaling_factors = _get_scan_range()
  diffs = jax.vmap(_quant)(scaling_factors)
  best_scaling = scaling_factors[jnp.argmin(diffs)].astype(bound.dtype)
  return bound * best_scaling


def _get_best_bound_per_channel(
    t: JTensor,
    bound: JTensor,
    min_value: float,
    max_value: float,
    p_value: float = 1.0,
) -> JTensor:
  """Scan around [0.5, 1] * hard max value to get bound value for each channel.

  This does a scan to get bound value(s) that minimize mean absolute error (MAE)
  between original tensor 't' and quantized tensor. It's (almost) equivalent to
  maximizing entropy.

  Args:
    t: The input float tensor.
    bound: The hard max value for tensor 't'. It has the same length as shape.
    min_value: Minimal value for the quantization bound.
    max_value: Maximal value for the quantization bound.
    p_value: Exponent of the p-mean error metric. Default to 1.0 which is MAE.

  Returns:
    The best bound values for 't', that minimize p-mean error.
  """
  assert len(t.shape) == 2
  assert len(bound.shape) == 2
  assert t.shape[1] == bound.shape[1]
  assert bound.shape[0] == 1
  scans = _get_scan_range()

  def _quant(tensor, bound, min_value, max_value, p_value, factors):
    ret = np.zeros(bound.shape)
    for i in range(len(tensor)):
      best = _quantrow(
          tensor[i], bound[i], min_value, max_value, p_value, factors
      )
      ret[i] = best
    return ret

  t = t.transpose()
  t_split = list(t)
  res = _quant(t_split, bound[0, :], min_value, max_value, p_value, scans)
  res = res.reshape(bound.shape)
  return res


def get_best_bound(
    t: JTensor,
    bound: JTensor,
    min_value: float,
    max_value: float,
    p_value: float = 1.0,
    per_channel: bool = False,
) -> JTensor:
  """Scan mutliple factors on max value to get best bound value.

  This does a scan to get bound value(s) that minimize mean absolute error (MAE)
  between original tensor 't' and quantized tensor. It's (almost) equivalent to
  maximizing entropy.

  Args:
    t: The input float tensor.
    bound: The hard max value for tensor 't'. It has the same length as shape.
    min_value: Minimal value for the quantization bound.
    max_value: Maximal value for the quantization bound.
    p_value: Exponent of the p-mean error metric. Default to 1.0 which is MAE.
    per_channel: if get best bound for entire tensor or per channel.

  Returns:
    The best bound values for 't', that minimize p-mean error.
  """
  if per_channel:
    return _get_best_bound_per_channel(t, bound, min_value, max_value, p_value)
  else:
    return _get_best_bound_per_tensor(t, bound, min_value, max_value, p_value)
