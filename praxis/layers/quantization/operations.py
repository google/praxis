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

"""Operations for quantization."""

import functools
import string
from typing import List, Optional, Sequence, Tuple

import jax
from jax import lax
from jax import numpy as jnp
from praxis import pytypes
from praxis.layers.quantization import aqt
from praxis.layers.quantization import optimization
from praxis.layers.quantization import utils

JTensor = pytypes.JTensor

QUANTIZED_TYPES = [jnp.int8, jnp.uint8]


def _get_expand_dims(eqn: str) -> List[int]:
  """Potentially expand dimensions for scale.

  It handles cases such as ABD,KDNH->KABNH and AD,KDNH->KANH, where weight is
  quantized to KNH and need to expand to K11NH and K1NH.

  Args:
    eqn: The equation for einsum.

  Returns:
    The expansion dimensions. Can be empty if no expansion is needed.
  """
  # TODO(jianlijianli): this is sufficient now but might improve to cover more
  # corner cases.
  segs = eqn.split('->')
  ins = segs[0].split(',')
  w, out = ins[1].replace('.', ''), segs[1].replace('.', '')

  filling_dims = [i for i, val in enumerate(out) if val not in w]
  # Avoid expanding dim for ab,bc->ac where scale of dim c doesn't need to be
  # expanded to (1,c) to be applied on ac.
  if filling_dims and filling_dims[-1] == len(filling_dims) - 1:
    filling_dims = []
  return filling_dims


def _get_min_max(bits):
  """Gets the min/max range for a given number of bits.

  Args:
    bits: Target number of bits for quantization.

  Returns:
    min/max values for the provide number of bits.
  """
  return -1 * 2 ** (bits - 1), 2 ** (bits - 1) - 1


def compute_offset(x: JTensor, zp: JTensor, eqn: str):
  """Computes offset: product of activation x with zero point of weight.

  Args:
    x: Not quantized activation.
    zp: Not quantized zero point of weight.
    eqn: The equation for the einsum between x and w.

  Returns:
    Offset tensor.
  """

  def _get_x_reduce_axis(eqn: str, x_dims: int) -> List[int]:
    """Get reduction axis on activation."""
    if eqn == 'ANH,DNH->AD' or eqn == 'ABNH,DNH->ABD':
      return [x_dims - 2, x_dims - 1]
    else:
      return [x_dims-1]

  def _get_offset_eqn(eqn: str) -> str:
    """Gets the offset equation dimensions."""
    if eqn == 'AD,KDNH->KANH':
      return 'A,KNH->KANH'
    if eqn == 'ANH,DNH->AD':
      return 'A,D->AD'
    if eqn == '...y,yz->...z':
      return '...y,z->...yz'
    if eqn == 'ABD,KDNH->KABNH':
      return 'AB,KNH->KABNH'
    if eqn == 'ABNH,DNH->ABD':
      return 'AB,D->ABD'
    if eqn == 'ABD,DNH->ABNH':
      return 'AB,NH->ABNH'
    if eqn == 'AD,DNH->ANH':
      return 'A,NH->ANH'
    if eqn == '...D,DH->...H':
      return '...D,H->...DH'
    # Add new equations as needed.
    raise NotImplementedError(f'eqn {eqn} not supported for asymmetric weight.')

  reduce_axis = _get_x_reduce_axis(eqn, len(x.shape))
  x_reduce = jnp.sum(x, axis=reduce_axis, keepdims=False)
  offset_eqn = _get_offset_eqn(eqn)
  offset = jnp.einsum(offset_eqn, x_reduce, zp)
  return offset


def einsum(
    eqn: str,
    x: JTensor,
    w: JTensor,
    scale: JTensor,
    zp: Optional[JTensor] = None,
) -> JTensor:
  """Performs quantized einsum.

  Quantized einsum consists in a regular Einsum on lower precision types,
  followed by a rescaling operation of element-wise multiplication.

  Args:
    eqn: The equation for the einsum between x and w.
    x: The input to the einsum; can be unquantized or quantized.
    w: The weight to the einsum; usually in quantized format.
    scale: The rescaling factor for the einsum. After applying this, the result
      is brought back to true value (no longer associated with scaling factors).
    zp: Optional zero point tensor.

  Returns:
    A JTensor.
  """
  if x.dtype in QUANTIZED_TYPES and w.dtype in QUANTIZED_TYPES:
    # upcast to int32 so einsum uses int32 as accumulator.
    # TODO(jianlijianli): allow preferred type to pass in as parameter.
    # TODO(jianlijianli): expand to cover for potentially int4.
    # TODO(jianlijianli): check if int32 is necessary since it will cast to
    # bf16/f32 for next op (accuracy-wise).
    x = x.astype(jnp.int32)
  ret = jnp.einsum(eqn, x, w)

  # Potentially expand dimensions of scale to match einsum output.
  filling_dims = _get_expand_dims(eqn)
  if filling_dims:
    scale = jnp.expand_dims(scale, filling_dims)

  ret = jnp.multiply(ret, scale)

  if zp is not None:
    offset = compute_offset(x, zp, eqn)
    ret = ret - offset
  return ret


def _round_with_gradient(x):
  zero = x - jax.lax.stop_gradient(x)
  return zero + jax.lax.stop_gradient(jnp.round(x))


def _reduce_precision(
    t: JTensor,
    contract_dims: Optional[Sequence[int]],
    need_gradient: bool = False,
    bits: int = 8,
    optimization_on_bound: bool = False,
    percentile: float = 1.0,
    use_symmetric: bool = True,
) -> Tuple[JTensor, JTensor, Optional[JTensor]]:
  """Reduce the precision of a tensor.

  Generic for all tensors.

  Args:
    t: Input tensor.
    contract_dims: Speficies contracting dimesnions of the input tensor.
    need_gradient: If gradient is needed out of this function.
    bits: target number of bits.
    optimization_on_bound: If MAE bound optimizer is used.
    percentile: percentile Factor to apply on the min/max range. Setting this to
      other than 1.0 disables optimization_on_bound.
    use_symmetric: If the input tensor is quantized symmetrically.

  Returns:
    A tuple of quantized tensor, quantization scale
      and quantization zero point (optional).
  """
  min_value, max_value = _get_min_max(bits)

  if use_symmetric:
    bound = jnp.max(jnp.abs(t), axis=contract_dims, keepdims=True)
    scale_bound = max_value
  else:
    t_max = jnp.max(t, axis=contract_dims, keepdims=True)
    t_min = jnp.min(t, axis=contract_dims, keepdims=True)
    bound = t_max - t_min
    scale_bound = 2**bits - 1.0

  if percentile < 1.0:
    bound = jnp.multiply(bound, percentile)
  elif optimization_on_bound:
    bound = optimization.get_best_bound(t, bound, min_value, max_value)

  scale = bound / scale_bound

  if use_symmetric:
    zp = None
    t = jnp.divide(t, scale)
  else:
    zp = min_value - t_min / scale
    t = jnp.divide(t, scale) + zp
    zp = jnp.multiply(scale, zp)

  if need_gradient:
    t = _round_with_gradient(t)
    t = jnp.clip(t, min_value, max_value)
  else:
    t = jnp.round(t)
    # Use int8 as container.
    t = jnp.clip(t, min_value, max_value).astype(jnp.int8)

  return t, scale, zp


def eqn_to_weight_contract_dims(eqn: str):
  segs = eqn.split('->')
  ins = segs[0].split(',')
  w, out = ins[1].replace('.', ''), segs[1].replace('.', '')
  return [i for i, val in enumerate(w) if val not in out]


def reduce_einsum_weight_precision(
    eqn: str,
    t: JTensor,
    calculation_type: jnp.dtype = jnp.bfloat16,
    squeeze: bool = True,
    need_gradient: bool = False,
    bits: int = 8,
    optimization_on_bound: bool = False,
    percentile: float = 1.0,
    use_symmetric: bool = True,
) -> Tuple[JTensor, JTensor, Optional[JTensor]]:
  """Reduce the precision of the weight of einsum.

  It uses per-channel quantization so einsum equation is passed in as well.

  Args:
    eqn: The equation for the einsum.
    t: The weight tensor for the einsum.
    calculation_type: The type for calculation.
    squeeze: If the output scale is squeezed.
    need_gradient: If gradient is needed out of this function.
    bits: Target number of bits.
    optimization_on_bound: If MAE bound optimizer is used.
    percentile: Percentile factor to apply on the min/max range.
    use_symmetric: If weights are quantized symmetrically.

  Returns:
    A tuple of JTensors. The first one is the quantized weight and the second
    one is the scaling factor.
  """
  contract_dims = eqn_to_weight_contract_dims(eqn)

  if t.dtype != calculation_type:
    t = t.astype(calculation_type)

  t, scale, zp = _reduce_precision(
      t,
      contract_dims,
      need_gradient,
      bits,
      optimization_on_bound,
      percentile=percentile,
      use_symmetric=use_symmetric,
  )
  if squeeze:
    scale = jnp.squeeze(scale)
    if zp is not None:
      zp = jnp.squeeze(zp)
  return t, scale, zp


def fakequant_einsum(
    eqn: str,
    t: JTensor,
    bits: int = 8,
    calculation_type: jnp.dtype = jnp.bfloat16,
    use_symmetric: bool = True,
) -> JTensor:
  """Nudges weight of einsum with FakeQuant.

  Args:
    eqn: The equation for the einsum. Determines the channel dimension.
    t: The weight tensor for the einsum.
    bits: Target number of bits.
    calculation_type: The type for calculation.
    use_symmetric: Use symmetric quantization for weights.

  Returns:
    The nudged weight tensor.
  """
  q, scale, zp = reduce_einsum_weight_precision(
      eqn,
      t,
      calculation_type=calculation_type,
      squeeze=False,
      need_gradient=True,
      bits=bits,
      optimization_on_bound=False,
      use_symmetric=use_symmetric,
  )
  res = jnp.multiply(q, scale)
  if zp is not None:
    res = jnp.subtract(res, zp)
  return res.astype(t.dtype)


def reduce_precision_activation(
    t: JTensor,
    need_gradient: bool = False,
    bits: int = 8,
) -> Tuple[JTensor, JTensor]:
  """Reduce the precision of activation.

  Args:
    t: Input tensor.
    need_gradient: If gradient is needed out of this function.
    bits: Target number of bits.

  Returns:
    A tuple of JTensors. The first one is the quantized activation and the
    second one is the scaling factor.
  """
  qt, scale, _ = _reduce_precision(t, None, need_gradient, bits, False)
  return qt, scale


def fakequant_activation(t: JTensor, bits: int = 8) -> JTensor:
  """FakeQuant activation.

  Args:
    t: Activation tensor
    bits: Target number of bits.

  Returns:
    Nudged activation.
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
    eqn: str,
    perm: Optional[Sequence[int]] = None,
) -> JTensor:
  """Quantized jax.lax.dot_general.

  Args:
    lhs: Left-hand side of the dot_general (mostly activation).
    rhs: Right-hand side of the dot_general (mostly weight, can be activation).
    lhs_quantizer: The tensor quantizer for lhs.
    rhs_quantizer: The tensor quantizer for rhs.
    dimension_numbers: A tuple of tuples of the form `((lhs_contracting_dims,
      rhs_contracting_dims), (lhs_batch_dims, rhs_batch_dims))`
    is_eval: If False, update the statistics in the tensor quantizers based on
      lhs and rhs.
    eqn: The valid binary einsum equation to use.
    perm: the dimenions to be permuated to align with the einsum equation.

  Returns:
    An array containing the result with the same dtype as 'lhs' and 'rhs'.
  """
  if not is_eval:
    # TODO(jihwanlee): Stats should be updated during training.
    pass

  input_dtype = lhs.dtype
  lhs_contract_dims, rhs_contract_dims = dimension_numbers[0]

  lhs, lhs_scale, _ = lhs_quantizer.quantize(
      lhs, lhs_contract_dims, squeeze_scale=False, quantized_dtype=input_dtype
  )

  rhs, rhs_scale, rhs_zp = rhs_quantizer.quantize(
      rhs, rhs_contract_dims, squeeze_scale=False, quantized_dtype=input_dtype)

  should_int8_quantize = (
      lhs_quantizer.precision is not None
      and lhs_quantizer.precision <= 8
      and rhs_quantizer.precision is not None
      and rhs_quantizer.precision <= 8
  )

  out = _dot_general_aqt(
      lhs,
      rhs,
      dimension_numbers=dimension_numbers,
      should_int8_quantize=should_int8_quantize)

  out_scale = lax.dot_general(
      lhs_scale, rhs_scale, dimension_numbers=dimension_numbers
  )

  ret = out * out_scale

  if perm is not None:
    ret = lax.transpose(ret, perm)

  if rhs_zp is not None:
    if lhs_quantizer.precision is not None:
      raise NotImplementedError(
          'Activation quantization with weight zero point is not supported yet.'
      )
    offset = compute_offset(lhs, rhs_zp, eqn)
    ret = ret - offset

  return ret


def aqt_einsum(
    eqn: str,
    lhs: JTensor,
    rhs: JTensor,
    *,
    lhs_quantizer: aqt.TensorQuantizer,
    rhs_quantizer: aqt.TensorQuantizer,
) -> JTensor:
  """Quantized einsum with AQT style.

  Args:
    eqn: The valid binary einsum equation to use.
    lhs: Left-hand side of the einsum (mostly activation).
    rhs: Right-hand side of the einsum (mostly weight, can be activation).
    lhs_quantizer: The tensor quantizer for lhs.
    rhs_quantizer: The tensor quantizer for rhs.

  Returns:
    An array containing the result with the same dtype as 'lhs' and 'rhs'.
  """
  if '.' in eqn:
    # Replace the ellipsis with arbitrary symbols.
    eqn_sym = ''.join(sorted(set(string.ascii_uppercase) - set('yz')))
    rank = len(lhs.shape)
    batch_eqn = eqn_sym[:(rank - 1)] if rank else '...'
    eqn_edited = f'{batch_eqn}y,yz->{batch_eqn}z'
    dimension_numbers, _ = utils.einsum_eqn_to_dimension_numbers(eqn_edited)
  else:
    dimension_numbers, _ = utils.einsum_eqn_to_dimension_numbers(eqn)

  lhs_contract_dims, rhs_contract_dims = dimension_numbers[0]

  lhs, lhs_scale, _ = lhs_quantizer.quantize(
      lhs, lhs_contract_dims, squeeze_scale=False, quantized_dtype=lhs.dtype
  )

  rhs, rhs_scale, rhs_zp = rhs_quantizer.quantize(
      rhs, rhs_contract_dims, squeeze_scale=False, quantized_dtype=rhs.dtype)

  out = jnp.einsum(eqn, lhs, rhs)
  out_scale = jnp.einsum(eqn, lhs_scale, rhs_scale)

  ret = out * out_scale

  if rhs_zp is not None:
    if lhs_quantizer.precision is not None:
      raise NotImplementedError(
          'Activation quantization with weight zero point is not supported yet.'
      )
    offset = compute_offset(lhs, rhs_zp, eqn)
    ret = ret - offset

  return ret
