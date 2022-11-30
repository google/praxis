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

import functools
from typing import List, Tuple

import jax
from jax import lax
from jax import numpy as jnp
from praxis import pytypes
from praxis.layers.quantization import aqt

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


def _round_with_gradient(x):
  zero = x - jax.lax.stop_gradient(x)
  return zero + jax.lax.stop_gradient(jnp.round(x))


def reduce_einsum_weight_precision(
    eqn: str,
    t: JTensor,
    calculation_type: jnp.dtype = jnp.bfloat16,
    output_type: jnp.dtype = jnp.bfloat16,
    squeeze: bool = True,
    need_gradient=False) -> Tuple[JTensor, JTensor]:
  """Reduce the precision of the weight of einsum.

  It uses per-channel quantization so einsum equantion is passed in as well.

  Args:
    eqn: the equation for the einsum.
    t: the weight tensor for the einsum.
    calculation_type: the type for calculation.
    output_type: the output type of scale.
    squeeze: if the output scale is squeezed.
    need_gradient: if gradient is needed out of this function.

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
  if need_gradient:
    t = _round_with_gradient(t)
  else:
    t = jnp.round(t)
  t = jnp.clip(t, -128.0, 127.0).astype(jnp.int8)
  if squeeze:
    scale = jnp.squeeze(scale).astype(output_type)
  return t, scale


def fakequant_einsum(eqn: str, t: JTensor) -> JTensor:
  """Nudges weight of einsum with FakeQuant.

  Args:
    eqn: the equantion for the einsum. Determines the channel dimension.
    t: the weight tensor for the einsum.

  Returns:
    The nudged weight tensor.
  """
  q, scale = reduce_einsum_weight_precision(
      eqn, t, squeeze=False, need_gradient=True)
  return jnp.multiply(q, scale).astype(jnp.float32).astype(t.dtype)


def reduce_precision_activation(t: JTensor,
                                need_gradient=False) -> Tuple[JTensor, JTensor]:
  """Reduce the precision of activation.

  Args:
    t: input tensor.
    need_gradient: if gradient is needed out of this function.

  Returns:
    A tuple of JTensors. The first one is the quantized activation and the
    second one is the scaling factor.
  """
  # TODO(jianlijianli): enable zero point as well.
  bound = jnp.maximum(
      jnp.abs(jnp.max(t, keepdims=True)), jnp.abs(jnp.min(t, keepdims=True)))
  scale = bound / 127.0
  t = jnp.divide(t, scale)
  if need_gradient:
    t = _round_with_gradient(t)
  else:
    t = jnp.round(t)
  t = jnp.clip(t, -128.0, 127.0).astype(jnp.int8)
  return t, scale


def fakequant_activation(t: JTensor) -> JTensor:
  """FakeQuant activation.

  Args:
    t: activation tensor

  Returns:
    nudged activation.
  """
  qt, scale = reduce_precision_activation(t, need_gradient=True)
  return jnp.multiply(qt, scale).astype(t.dtype)


@functools.partial(jax.custom_jvp, nondiff_argnums=(2, 3))
def _dot_general_aqt(lhs, rhs, dimension_numbers, should_int8_quantize):
  """Wrapper around lax.dot_general, but with option to use integer dot."""
  def dot_general_float(ops):
    lhs_, rhs_ = ops
    return lax.dot_general(lhs_, rhs_, dimension_numbers=dimension_numbers)

  def dot_general_int(ops):
    lhs_, rhs_ = ops
    lhs_int = lhs_.astype(jnp.int8)
    rhs_int = rhs_.astype(jnp.int8)
    return lax.dot_general(
        lhs_int,
        rhs_int,
        dimension_numbers=dimension_numbers,
        preferred_element_type=jnp.int32).astype(jnp.float32)

  if should_int8_quantize:
    return dot_general_int((lhs, rhs))
  else:
    return dot_general_float((lhs, rhs))


@_dot_general_aqt.defjvp
def _dot_general_aqt_jvp(
    dimension_numbers,
    should_int8_quantize,
    primals,
    tangents):
  """Custom gradient for dot_general_aqt that ignores integer casts."""
  lhs, rhs = primals
  lhs_dot, rhs_dot = tangents
  y = _dot_general_aqt(
      lhs,
      rhs,
      dimension_numbers=dimension_numbers,
      should_int8_quantize=should_int8_quantize)

  def differentiable_dot_general(lhs_, rhs_):
    return lax.dot_general(lhs_, rhs_, dimension_numbers=dimension_numbers)

  _, y_tangent = jax.jvp(
      differentiable_dot_general,
      (lhs, rhs),
      (lhs_dot, rhs_dot))
  return y, y_tangent


def dot_general(
    lhs: JTensor,
    rhs: JTensor,
    lhs_quantizer: aqt.TensorQuantizer,
    rhs_quantizer: aqt.TensorQuantizer,
    dimension_numbers: lax.DotDimensionNumbers,
    train: bool
) -> JTensor:
  """Quantized lax.dot_general."""
  if train:
    lhs_quantizer.update(lhs)
    rhs_quantizer.update(rhs)

  lhs_contract_dims, rhs_contract_dims = dimension_numbers[0]
  lhs_scale = lhs_quantizer.get_quant_scale(lhs, lhs_contract_dims)
  rhs_scale = rhs_quantizer.get_quant_scale(rhs, rhs_contract_dims)

  lhs = lhs_scale * lhs
  rhs = rhs_scale * rhs

  lhs = lhs_quantizer.to_quant(lhs)
  rhs = rhs_quantizer.to_quant(rhs)

  out = _dot_general_aqt(
      lhs,
      rhs,
      dimension_numbers=dimension_numbers,
      should_int8_quantize=False)

  inv_scale = 1 / (lhs_scale * rhs_scale)
  return out * inv_scale

