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
from typing import List, Optional, Sequence, Tuple

import jax
from jax import lax
from jax import numpy as jnp
from praxis import pytypes
from praxis.layers.quantization import aqt
from praxis.layers.quantization import optimization

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


def reduce_precision(
    t: JTensor,
    bound: JTensor,
    need_gradient: bool = False,
    optimization_on_bound: bool = False,
) -> Tuple[JTensor, JTensor]:
  """Reduce the precision of a tensor.

  Generic for all tensors.

  TODO(jianlijianli): support lower bits.

  Args:
    t: input tensor
    bound: the bound value for the tensor.
    need_gradient: if gradient is needed out of this function.
    optimization_on_bound: if MAE bound optimizer is used.

  Returns:
    the quantized tensor.
    the quantization scale.
  """
  if optimization_on_bound:
    bound = optimization.get_best_bound(t, bound)
  scale = bound / 127.0
  t = jnp.divide(t, scale)
  if need_gradient:
    t = _round_with_gradient(t)
    t = jnp.clip(t, -128.0, 127.0)
  else:
    t = jnp.round(t)
    t = jnp.clip(t, -128.0, 127.0).astype(jnp.int8)
  return t, scale


def reduce_einsum_weight_precision(
    eqn: str,
    t: JTensor,
    calculation_type: jnp.dtype = jnp.bfloat16,
    squeeze: bool = True,
    need_gradient: bool = False,
    optimization_on_bound: bool = False,
) -> Tuple[JTensor, JTensor]:
  """Reduce the precision of the weight of einsum.

  It uses per-channel quantization so einsum equantion is passed in as well.

  Args:
    eqn: the equation for the einsum.
    t: the weight tensor for the einsum.
    calculation_type: the type for calculation.
    squeeze: if the output scale is squeezed.
    need_gradient: if gradient is needed out of this function.
    optimization_on_bound: if MAE bound optimizer is used.

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

  t, scale = reduce_precision(t, bound, need_gradient, optimization_on_bound)
  if squeeze:
    scale = jnp.squeeze(scale)
  return t, scale


def fakequant_einsum(eqn: str,
                     t: JTensor,
                     calculation_type: jnp.dtype = jnp.bfloat16) -> JTensor:
  """Nudges weight of einsum with FakeQuant.

  Args:
    eqn: the equantion for the einsum. Determines the channel dimension.
    t: the weight tensor for the einsum.
    calculation_type: the type for calculation.

  Returns:
    The nudged weight tensor.
  """
  q, scale = reduce_einsum_weight_precision(
      eqn,
      t,
      calculation_type=calculation_type,
      squeeze=False,
      need_gradient=True)
  return jnp.multiply(q, scale).astype(t.dtype)


def reduce_precision_activation(
    t: JTensor, need_gradient: bool = False
) -> Tuple[JTensor, JTensor]:
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
  return reduce_precision(t, bound, need_gradient, False)


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
    # The dtype of output is determined by the dtype of activation if the
    # activation and weight have different dtypes.
    input_dtype = lhs_.dtype
    lhs_int = lhs_.astype(jnp.int8)
    rhs_int = rhs_.astype(jnp.int8)
    return lax.dot_general(
        lhs_int,
        rhs_int,
        dimension_numbers=dimension_numbers,
        preferred_element_type=jnp.int32).astype(input_dtype)

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
    rhs: Optional[JTensor],
    lhs_quantizer: aqt.TensorQuantizer,
    rhs_quantizer: aqt.TensorQuantizer,
    dimension_numbers: lax.DotDimensionNumbers,
    is_eval: bool,
    perm: Optional[Sequence[int]] = None,
    rhs_quantized: Optional[Tuple[JTensor, JTensor]] = None,
) -> JTensor:
  """Quantized jax.lax.dot_general.

  Args:
    lhs: Left-hand side of the dot_general.
    rhs: Right-hand side of the dot_general.
    lhs_quantizer: The tensor quantizer for lhs.
    rhs_quantizer: The tensor quantizer for rhs.
    dimension_numbers: a tuple of tuples of the form `((lhs_contracting_dims,
      rhs_contracting_dims), (lhs_batch_dims, rhs_batch_dims))`
    is_eval: If False, update the statistics in the tensor quantizers based on
      lhs and rhs.
    perm: Permutation of the desired axes of the output. For some einsum
      equations, it is not possible to obtain their corresponding
      dimension_numbers that yield the same results using dot_general. If it is
      possible with a proper transposition, then one should pass a list of
      dimenions to permute. This should be None if transposition is not needed.
    rhs_quantized: A pair of quantized rhs and its scale. It should exist only
      in the inference mode and both rhs and rhs_quantized cannot be passed
      together.

  Returns:
    An array containing the result with the same dtype as 'lhs' and 'rhs'.
  """
  assert ((rhs is None and rhs_quantized is not None) or
          (rhs is not None and rhs_quantized is None))

  if not is_eval:
    # TODO(jihwanlee): Stats should be updated during training.
    pass

  input_dtype = lhs.dtype
  lhs_contract_dims, rhs_contract_dims = dimension_numbers[0]

  lhs_scale = lhs_quantizer.get_quant_scale(lhs, lhs_contract_dims, input_dtype)
  lhs = lhs_scale * lhs
  lhs = lhs_quantizer.to_quant(lhs, input_dtype)

  if rhs_quantized is None:
    rhs_scale = rhs_quantizer.get_quant_scale(rhs, rhs_contract_dims,
                                              input_dtype)
    rhs = rhs_scale * rhs
    rhs = rhs_quantizer.to_quant(rhs, input_dtype)
  elif rhs is None:
    assert is_eval, 'Expected inference when rhs_quantized is passed.'
    # If rhs_quantized is passed, then it means the rhs is already quantized and
    # its scale is provided. Thus, no need to get scale and quantize rhs again.
    rhs, rhs_scale = rhs_quantized
    # Make sure lhs and rhs have the same dtype.
    rhs = rhs.astype(input_dtype)
    rhs_scale = rhs_scale.astype(input_dtype)
    # Restore the contracting dimension.
    rhs_scale = jnp.expand_dims(rhs_scale, axis=rhs_contract_dims)
  else:
    raise ValueError('Cannot reach here.')

  should_int8_quantize = (
      lhs_quantizer.hparams.precision is not None and
      rhs_quantizer.hparams.precision is not None)

  out = _dot_general_aqt(
      lhs,
      rhs,
      dimension_numbers=dimension_numbers,
      should_int8_quantize=should_int8_quantize)

  inv_scale = lax.dot_general(
      1 / lhs_scale, 1 / rhs_scale, dimension_numbers=dimension_numbers)

  if perm is not None:
    assert len(perm) == len(out.shape)
    out = lax.transpose(out, perm)
    inv_scale = lax.transpose(inv_scale, perm)

  return out * inv_scale
