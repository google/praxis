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


class Einsum(base_layer.BaseLayer):
  """Einsum wrapper of jnp.einsum used in standard Pax layers.

  This wrapper allows (with a help of fiddle) to inject custom einsum
  implementations into various Pax layers.
  The main benefit is reduction of the need for layer forking.

  Einsum injection is useful if one wants to inject quantized or sparse einsum.
  It also allows these custom einsums to use state e.g.:
  to use variables for calibration stats and
  to use random numbers e.g.: for stochastic rounding.
  """

  def __call__(self, equation: str, lhs: JTensor, rhs: JTensor) -> JTensor:
    return jnp.einsum(equation, lhs, rhs)
