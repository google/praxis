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

"""Op wrappers that can be used instead of native JAX ops to allow injection of custom ops.

This is useful for quantization, sparsity and possibly other techniques.
"""

import jax.numpy as jnp
from praxis import base_layer
from praxis import pytypes

JTensor = pytypes.JTensor


# These wrappers allow (with a help of fiddle) to inject custom ops
# implementations into various Pax layers.
# Base ops injection are useful if one wants to inject quantized or sparse op.
# The main benefit is a reduction of the need for layer forking.
#
# It also allows these custom ops to use state e.g.:
# to use variables for calibration stats and
# to use random numbers e.g.: for stochastic rounding.


class EinsumOp(base_layer.BaseLayer):
  """Wrapper around jnp.einsum used in standard Pax layers."""

  def __call__(self, equation: str, *args: JTensor) -> JTensor:
    return jnp.einsum(equation, *args)


class EinsumGatedOp(base_layer.BaseLayer):
  """Wrapper around two jnp.einsum for gated FFN."""

  def __call__(self, equation: str, *args: JTensor) -> tuple[JTensor, JTensor]:
    assert len(args) == 3
    x, k, k_gated = args
    y = jnp.einsum(equation, x, k)
    y_gated = jnp.einsum(equation, x, k_gated)
    return y, y_gated


class ArrayLookup(base_layer.BaseLayer):
  """Wrapper around array indexing as used in embedding lookup."""

  def __call__(self, x: JTensor, idx) -> JTensor:
    return x[idx]
