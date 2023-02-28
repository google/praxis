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


def get_best_bound(
    t: JTensor, bound: JTensor, min_value: float, max_value: float
) -> JTensor:
  """Scan around [0.95, 1] * hard max value to get bound value.

  This does a scan to get bound value(s) that minimize mean absolute error (MAE)
  between original tensor 't' and quantized tensor. It's (almost) equivalent to
  maximizing entropy.

  TODO(jianlijianli): optimize this at per-channel level.

  Args:
    t: The input float tensor.
    bound: The hard max value for tensor 't'. It has the same length as shape.
    min_value: Minimal value for the quantization bound.
    max_value: Maximal value for the quantization bound.

  Returns:
    The best bound values for 't', that minimize average error (MAE).
  """

  def quantize(scaling_factor):
    scale = bound * scaling_factor / max_value
    candidate = jnp.divide(t, scale)
    candidate = jnp.clip(jnp.round(candidate), min_value, max_value)
    candidate = jnp.multiply(candidate, scale)
    mean_error = jnp.mean(jnp.abs(jnp.subtract(candidate, t)))
    return mean_error, jnp.array(scaling_factor)

  scaling_factors = np.linspace(1.0, 0.95, num=11)
  res = jax.vmap(quantize)(scaling_factors)
  best_scaling = res[1].at[jnp.argmin(res[0])].get().astype(bound.dtype)
  return jnp.multiply(bound, best_scaling)
