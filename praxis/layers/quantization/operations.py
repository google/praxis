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

"""Operations for quantization."""

from typing import List, Tuple

from jax import numpy as jnp
from praxis import pytypes

JTensor = pytypes.JTensor


def _get_expand_dims(eqn: str) -> List[int]:
  """Potentially expand dimentions for scale.

  It handles cases such as ABD,KDNH->KABNH and AD,KDNH->KANH, where weight is
  quantized to KNH and need to expand to K11NH and K1NH.

  Args:
    eqn: the equation for einsum.

  Returns:
    the expansion dimensions. Can be empty if no expansion is needed.
  """
  # TODO(jianlijianli): this is sufficient now but might improve to cover more
  # corner cases.
  segs = eqn.split('->')
  ins = segs[0].split(',')
  w, out = ins[1].replace('.', ''), segs[1].replace('.', '')

  filling_dims = [i for i, val in enumerate(out) if val not in w]
  # Avoid expanding dim for ab,bc->ac where scale of dim c doesn't need to be
  # expaneded to (1,c) to be applied on ac.
  if filling_dims and filling_dims[-1] == len(filling_dims) - 1:
    filling_dims = []
  return filling_dims


def einsum(eqn: str, x: JTensor, w: JTensor, scale: JTensor) -> JTensor:
  """Performs quantized einsum.

  Quantized einsum consists in a regular Einsum on lower precision types,
  followed by a rescaling operation of element-wise multiplcation.

  Args:
    eqn: The equation for the einsum between x and w.
    x: The input to the einsum; can be unquantized or quantized.
    w: The weight to the einsum; usually in quantized format.
    scale: The rescaling factor for the einsum. After applying this, the result
      is brought back to true value (no longer associated with scaling factors).

  Returns:
    A JTensor

  """
  ret = jnp.einsum(eqn, x, w)

  # Potentially expand dimentions of scale to match einsum output.
  filling_dims = _get_expand_dims(eqn)
  if filling_dims:
    scale = jnp.expand_dims(scale, filling_dims)

  return jnp.multiply(ret, scale)


def reduce_einsum_weight_precision(
    eqn: str,
    t: JTensor,
    calculation_type: jnp.dtype = jnp.bfloat16,
    output_type: jnp.dtype = jnp.bfloat16) -> Tuple[JTensor, JTensor]:
  """Reduce the precision of the weight of einsum.

  It uses per-channel quantization so einsum equantion is passed in as well.

  Args:
    eqn: the equation for the einsum.
    t: the weight tensor for the einsum.
    calculation_type: the type for calculation.
    output_type: the output type of scale.

  Returns:
    A tuple of JTensors. The first one is the quantized weight and the second
    one is the scaling factor.
  """
  segs = eqn.split('->')
  ins = segs[0].split(',')
  w, out = ins[1].replace('.', ''), segs[1].replace('.', '')

  contract_dims = [i for i, val in enumerate(w) if val not in out]

  if t.dtype != calculation_type:
    t = t.astype(calculation_type)
  bound = jnp.maximum(
      jnp.abs(jnp.max(t, axis=contract_dims, keepdims=True)),
      jnp.abs(jnp.min(t, axis=contract_dims, keepdims=True)))
  scale = bound / 127.0
  t = jnp.divide(t, scale)
  t = jnp.round(t)
  t = jnp.clip(t, -128.0, 127.0).astype(jnp.int8)
  scale = jnp.squeeze(scale).astype(output_type)
  return t, scale
