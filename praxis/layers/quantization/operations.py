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
from typing import List, Optional, Tuple

import jax
from jax import lax
from jax import numpy as jnp
from praxis import pytypes
from praxis.layers.quantization import aqt
from praxis.layers.quantization import optimization
from praxis.layers.quantization import utils

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


def _get_min_max(bits):
  """Gets the min/max range for a given number of bits.

  Args:
    bits: target number of bits for quantization.

  Returns:
    min/max values for the provide number of bits.
  """
  return -1 * 2 ** (bits - 1), 2 ** (bits - 1) - 1


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
  if x.dtype == jnp.int8 and w.dtype == jnp.int8:
    # upcast to int32 so einsum uses int32 as accumulator.
    # TODO(jianlijianli): allow preferred type to pass in as parameter.
    # TODO(jianlijianli): expand to cover for potentially int4.
    # TODO(jianlijianli): check if int32 is necessary since it will cast to
    # bf16/f32 for next op (accuracy-wise).
    x = x.astype(jnp.int32)
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
    bits: int = 8,
    optimization_on_bound: bool = False,
    percentile: float = 1.0,
) -> Tuple[JTensor, JTensor]:
  """Reduce the precision of a tensor.

  Generic for all tensors.

  Args:
    t: input tensor
    bound: the bound value for the tensor.
    need_gradient: if gradient is needed out of this function.
    bits: target number of bits.
    optimization_on_bound: if MAE bound optimizer is used.
    percentile: percentile factor to apply on the min/max range. Setting this to
      other than 1.0 disables optimization_on_bound.

  Returns:
    the quantized tensor.
    the quantization scale.
  """
  min_value, max_value = _get_min_max(bits)
  if percentile < 1.0:
    bound = jnp.multiply(bound, percentile)
  elif optimization_on_bound:
    bound = optimization.get_best_bound(t, bound, min_value, max_value)
  scale = bound / max_value
  t = jnp.divide(t, scale)
  if need_gradient:
    t = _round_with_gradient(t)
    t = jnp.clip(t, min_value, max_value)
  else:
    t = jnp.round(t)
    # Use int8 as container.
    t = jnp.clip(t, min_value, max_value).astype(jnp.int8)
  return t, scale


def reduce_einsum_weight_precision(
    eqn: str,
    t: JTensor,
    calculation_type: jnp.dtype = jnp.bfloat16,
    squeeze: bool = True,
    need_gradient: bool = False,
    bits: int = 8,
    optimization_on_bound: bool = False,
    percentile: float = 1.0,
) -> Tuple[JTensor, JTensor]:
  """Reduce the precision of the weight of einsum.

  It uses per-channel quantization so einsum equantion is passed in as well.

  Args:
    eqn: the equation for the einsum.
    t: the weight tensor for the einsum.
    calculation_type: the type for calculation.
    squeeze: if the output scale is squeezed.
    need_gradient: if gradient is needed out of this function.
    bits: target number of bits.
    optimization_on_bound: if MAE bound optimizer is used.
    percentile: percentile factor to apply on the min/max range.

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
      jnp.abs(jnp.min(t, axis=contract_dims, keepdims=True)),
  )

  t, scale = reduce_precision(
      t,
      bound,
      need_gradient,
      bits,
      optimization_on_bound,
      percentile=percentile,
  )
  if squeeze:
    scale = jnp.squeeze(scale)
  return t, scale


def fakequant_einsum(
    eqn: str,
    t: JTensor,
    bits: int = 8,
    calculation_type: jnp.dtype = jnp.bfloat16,
) -> JTensor:
  """Nudges weight of einsum with FakeQuant.

  Args:
    eqn: the equantion for the einsum. Determines the channel dimension.
    t: the weight tensor for the einsum.
    bits: target number of bits.
    calculation_type: the type for calculation.

  Returns:
    The nudged weight tensor.
  """
  q, scale = reduce_einsum_weight_precision(
      eqn,
      t,
      calculation_type=calculation_type,
      squeeze=False,
      need_gradient=True,
      bits=bits,
      optimization_on_bound=False,
  )
  return jnp.multiply(q, scale).astype(t.dtype)


def reduce_precision_activation(
    t: JTensor,
    need_gradient: bool = False,
    bits: int = 8,
) -> Tuple[JTensor, JTensor]:
  """Reduce the precision of activation.

  Args:
    t: input tensor.
    need_gradient: if gradient is needed out of this function.
    bits: target number of bits.

  Returns:
    A tuple of JTensors. The first one is the quantized activation and the
    second one is the scaling factor.
  """
  # TODO(jianlijianli): enable zero point as well.
  bound = jnp.maximum(
      jnp.abs(jnp.max(t, keepdims=True)), jnp.abs(jnp.min(t, keepdims=True)))
  return reduce_precision(t, bound, need_gradient, bits, False)


def fakequant_activation(t: JTensor, bits: int = 8) -> JTensor:
  """FakeQuant activation.

  Args:
    t: activation tensor
    bits: target number of bits.

  Returns:
    nudged activation.
  """
  qt, scale = reduce_precision_activation(t, need_gradient=True, bits=bits)
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
  lhs = lhs / lhs_scale
  lhs = lhs_quantizer.to_quant(lhs, input_dtype)

  if rhs_quantized is None:
    rhs_scale = rhs_quantizer.get_quant_scale(rhs, rhs_contract_dims,
                                              input_dtype)
    rhs = rhs / rhs_scale
    rhs = rhs_quantizer.to_quant(rhs, input_dtype)
  elif rhs is None:
    assert (
        is_eval
    ), f'Expected is_eval={is_eval} == True when rhs_quantized is passed.'
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
      lhs_quantizer.hparams.precision is not None
      and lhs_quantizer.hparams.precision <= 8
      and rhs_quantizer.hparams.precision is not None
      and rhs_quantizer.hparams.precision <= 8
  )

  out = _dot_general_aqt(
      lhs,
      rhs,
      dimension_numbers=dimension_numbers,
      should_int8_quantize=should_int8_quantize)

  out_scale = lax.dot_general(
      lhs_scale, rhs_scale, dimension_numbers=dimension_numbers
  )

  return out * out_scale


# TODO(b/262309036): Make dot_general the default quantized einsum
# implementation.
def aqt_einsum(
    eqn: str,
    lhs: JTensor,
    rhs: Optional[JTensor],
    *,
    lhs_quantizer: aqt.TensorQuantizer,
    rhs_quantizer: aqt.TensorQuantizer,
    is_eval: bool,
    rhs_quantized: Optional[Tuple[JTensor, JTensor]] = None,
) -> JTensor:
  """Quantized einsum using jax.lax.dot_general.

  Args:
    eqn: The valid binary einsum equation to use.
    lhs: Left-hand side of the einsum.
    rhs: Right-hand side of the einsum.
    lhs_quantizer: The tensor quantizer for lhs.
    rhs_quantizer: The tensor quantizer for rhs.
    is_eval: If False, update the statistics in the tensor quantizers based on
      lhs and rhs.
    rhs_quantized: A pair of quantized rhs and its scale. It should exist only
      in the inference mode and both rhs and rhs_quantized cannot be passed
      together.

  Returns:
    An array containing the result with the same dtype as 'lhs' and 'rhs'.
  """
  dimension_numbers, perm = utils.einsum_eqn_to_dimension_numbers(eqn)
  out = dot_general(
      lhs=lhs,
      rhs=rhs,
      lhs_quantizer=lhs_quantizer,
      rhs_quantizer=rhs_quantizer,
      dimension_numbers=dimension_numbers,
      is_eval=is_eval,
      rhs_quantized=rhs_quantized,
  )
  if perm is not None:
    out = lax.transpose(out, perm)
  return out
