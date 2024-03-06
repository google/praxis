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
from typing import Any, Sequence

from absl import logging
import jax
from jax import lax
from jax import numpy as jnp
import numpy as np
from opt_einsum import parser as einsum_parser
from praxis import pytypes
from praxis.layers.quantization import optimization
from praxis.layers.quantization import quantization_hparams
from praxis.layers.quantization import utils


JTensor = pytypes.JTensor
PRNGKey = pytypes.PRNGKey
WeightQuantizationParams = quantization_hparams.WeightQuantizationParams
INT4_TYPES = utils.INT4_TYPES
INT_TYPES = utils.INT_TYPES


class FP4:
  """FP4 quantization."""

  int4_val = [
      -6.0,
      -4.0,
      -3.0,
      -2.0,
      -1.5,
      -1.0,
      -0.5,
      0.0,
      0.5,
      1.0,
      1.5,
      2.0,
      3.0,
      4.0,
      6.0,
  ]

  def __init__(self, val=None):
    self.v = jnp.array(val if val else self.int4_val, dtype=jnp.float32)

  def round(self, x):
    # TODO(jianlijianli): make a faster version.
    diff = jnp.abs(jnp.subtract(jnp.expand_dims(x, axis=-1), self.v))
    argmin = jnp.argmin(diff, axis=-1)
    return jnp.take(self.v, argmin.flatten()).reshape(x.shape)

  def nudge(self, x, contract_dim):
    # TODO(jianlijianli): allow symmetric, sub-channel etc.
    target_min, target_max = self.v[0], self.v[-1]
    value_min = jnp.min(x, axis=contract_dim, keepdims=True)
    value_max = jnp.max(x, axis=contract_dim, keepdims=True)
    scale = (value_max - value_min) / (target_max - target_min)
    scale = scale + jnp.finfo(x.dtype).eps
    zp = target_min - value_min / scale

    q = jnp.divide(x, scale) + zp
    rounded = self.round(q)
    recover = jnp.multiply(jnp.subtract(rounded, zp), scale)
    return recover.astype(x.dtype)


def _get_expand_dims_rhs(eqn: str) -> list[int]:
  """Potentially expand dimensions for scale of right-hand-side tensor.

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


def _get_expand_dims_lhs(eqn: str) -> list[int]:
  """Potentially expand dimensions for scale of left-hand-side tensor.

  It handles cases such as ABD,KDNH->KABNH and ABD,DKNH->ABKNH, where activation
  is quantized to AB and need to expand to 1AB11 and AB111.

  Args:
    eqn: The equation for einsum.

  Returns:
    The expansion dimensions. Can be empty if no expansion is needed.
  """
  # TODO(wppark): this is sufficient now but might improve to cover more
  # corner cases.
  segs = eqn.split('->')
  ins = segs[0].split(',')
  act, out = ins[0].replace('.', ''), segs[1].replace('.', '')

  filling_dims = [
      -(i + 1) for i, val in enumerate(reversed(out)) if val not in act
  ]
  return filling_dims


def get_min_max(
    bits: int = 8,
    unsigned: bool = False,
    use_fp: bool = False,
) -> tuple[float, float]:
  """Gets the min/max range for a given number of bits.

  Args:
    bits: Target number of bits for quantization.
    unsigned: If True compute min and max for unsigned number, else for signed.
    use_fp: in floating point.

  Returns:
    min/max values for the provide number of bits.
  """
  if use_fp:
    # TODO(jianlijianli): support other fp types.
    return -448.0, 448.0
  # Calculation instead of jax.iinfo is used to support bits beside 4 and 8.
  if unsigned:
    # For unsigned 8 bits precision it is [0, 255]
    return 0, 2**bits - 1
  else:
    # For signed 8 bits precision it is [-128, 127]
    return -1 * 2 ** (bits - 1), 2 ** (bits - 1) - 1


def compute_offset(eqn_normalized: str, x: JTensor, zp: JTensor) -> JTensor:
  """Computes offset: product of activation x with zero point of weight.

  Args:
    eqn_normalized: The equation for the einsum between x and w. eqn_normalized
      should not contain any '...'.
    x: Not quantized activation.
    zp: Not quantized zero point of weight.

  Returns:
    Offset tensor.
  """
  if '.' in eqn_normalized:
    raise ValueError(
        'eqn_normalized should not contain broadcast ellipsis "...". Use'
        ' opt_einsum.parser to normalize the eqn before using this function.'
    )
  ins, out = eqn_normalized.split('->')
  lhs, rhs = ins.split(',')
  rhs_out_dims = ''.join([c for c in out if c in rhs])
  offset_eqn = lhs + ',' + rhs_out_dims + '->' + out
  offset = jnp.einsum(offset_eqn, x, zp)
  return offset


@functools.partial(jax.custom_jvp, nondiff_argnums=(2,))
def dot_general_int(lhs, rhs, dimension_numbers):
  """Wrapper around lax.dot_general, with int dot."""

  def _dot_general_int(ops):
    lhs_, rhs_ = ops
    if lhs_.dtype != rhs_.dtype:
      # XLA will automatically cast operands with non-matching dtypes up to
      # int32, which is often suboptimal.
      dtype = utils.get_smallest_matching_dtype(lhs_, rhs_)
      lhs_ = lhs_.astype(dtype)
      rhs_ = rhs_.astype(dtype)

    return lax.dot_general(
        lhs_,
        rhs_,
        dimension_numbers=dimension_numbers,
        preferred_element_type=jnp.int32)

  if lhs.dtype not in INT_TYPES:
    raise ValueError(f'{lhs.dtype=} is not an int type: {INT_TYPES} ')
  if rhs.dtype not in INT_TYPES:
    raise ValueError(f'{rhs.dtype=} is not an int type: {INT_TYPES} ')
  return _dot_general_int((lhs, rhs))


@dot_general_int.defjvp
def dot_general_int_jvp(
    dimension_numbers,
    primals,
    tangents):
  """Custom gradient for dot_general_int that ignores integer casts."""
  lhs, rhs = primals
  lhs_dot, rhs_dot = tangents
  y = dot_general_int(
      lhs,
      rhs,
      dimension_numbers=dimension_numbers)

  def differentiable_dot_general_int(lhs_, rhs_):
    return lax.dot_general(lhs_, rhs_, dimension_numbers=dimension_numbers)

  _, y_tangent = jax.jvp(
      differentiable_dot_general_int,
      (lhs, rhs),
      (lhs_dot, rhs_dot))
  return y, y_tangent


@jax.custom_vjp
def custom_einsum(x: JTensor, w: JTensor, key: jax.Array) -> jnp.ndarray:
  return jnp.einsum('abc,cd->abd', x, w)


def custom_einsum_fwd(x: JTensor, w: JTensor, key: jax.Array):
  """Custom forward pass for custom_einsum."""
  # Currently support only abc,cd->abd
  # TODO(jianlijianli): make this more general.
  assert x.ndim == 3
  assert w.ndim == 2
  assert x.shape[2] == w.shape[0]
  qx, sx, _ = reduce_precision(x, bits=8, contract_dims=[2])
  qw, sw, _ = reduce_precision(w, bits=8, contract_dims=[0])
  acc = jnp.einsum('abc,cd->abd', qx, qw, preferred_element_type=jnp.bfloat16)
  res = jnp.multiply(sx, jnp.multiply(acc, sw))
  return res, (qx, qw, sx, sw, key)


def custom_einsum_bwd(res: Any, g: Any):
  """Custom gradient for custom_einsum."""
  qx, qw, sx, sw, key = res
  g_with_sw = jnp.multiply(g, sw)
  g_with_sx = jnp.multiply(g, sx)
  qg_for_w, sg_for_w, _ = reduce_precision(
      t=g_with_sw, bits=8, contract_dims=[2], random_rounding=True, key=key
  )
  qg_for_x, sg_for_x, _ = reduce_precision(
      t=g_with_sx, bits=8, contract_dims=[0, 1], random_rounding=True, key=key
  )
  gx = jnp.einsum(
      'abd,cd->abc', qg_for_w, qw, preferred_element_type=jnp.bfloat16
  )
  gw = jnp.einsum(
      'abc,abd->cd', qx, qg_for_x, preferred_element_type=jnp.bfloat16
  )
  gx = jnp.multiply(gx, sg_for_w)
  gw = jnp.multiply(gw, jnp.squeeze(sg_for_x))
  return gx, gw, None


custom_einsum.defvjp(custom_einsum_fwd, custom_einsum_bwd)


def einsum(
    eqn: str,
    x: JTensor,
    w: JTensor,
    scale: JTensor,
    zp: JTensor | None = None,
    scale_act: JTensor | None = None,
    zp_act: JTensor | None = None,
    scale_eqn: str | None = None,
    zp_eqn: str | None = None,
    swap_xw: bool = False,
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
      It is the scale factor of the weights in most of cases.
    zp: Optional zero point tensor.
    scale_act: Optional rescaling factor for the einsum on the left-hand-side
      tensor which is activation in most of cases. After applying this, the
      result is brought back to true value.
    zp_act: Optional zero point for the einsum on the right-hand-side  tensor
      which is activation in most cases.
    scale_eqn: Optional. Let tmp = einsum(eqn, x, w), scale_out =
      einsum(scale_eqn, tmp, out). Default scale_eqn act as '...z,z->...z'
    zp_eqn: Optional. ret = scale_out - einsum(zp_eqn, x, zp) Default zp_eqn act
      as '...y,z->...z'
    swap_xw: Swap the input and weight tensor in einsum for performance,

  Returns:
    A JTensor.
  """
  if swap_xw:
    input_str, output_str, _ = einsum_parser.parse_einsum_input((eqn, w, x))
  else:
    input_str, output_str, _ = einsum_parser.parse_einsum_input((eqn, x, w))
  eqn_normalized = input_str + '->' + output_str

  # Non performent equation for inference testing purposes
  # TODO: b/305735188 - Improve the performance by using the integer einsum op.
  if zp_act is not None:
    dequantized_x = jnp.multiply(x, scale_act) - zp_act
    # explicit broadcast if necessary.
    if w.ndim == 3 and scale.ndim == 1:
      scale = jnp.expand_dims(scale, (1, 2))
    dequantized_w = jnp.multiply(w, scale)
    if zp is not None:
      dequantized_w = dequantized_w - zp
    return jnp.einsum(eqn, dequantized_x, dequantized_w)

  if (
      jax.dtypes.scalar_type_of(w.dtype) == float
      and jnp.finfo(w.dtype).bits == 8
  ):
    w = w.astype(jnp.bfloat16)

  if x.dtype in INT_TYPES and w.dtype in INT_TYPES:
    assert not swap_xw, 'No need to swap x and w when both are int types.'
    dimension_numbers, perm = utils.einsum_eqn_to_dimension_numbers(
        eqn_normalized
    )
    ret = dot_general_int(
        x,
        w,
        dimension_numbers=dimension_numbers,
    )
    if perm is not None:
      ret = lax.transpose(ret, perm)
  else:
    # jnp.einsum does not support implicit promotion of (u)int4 types.
    w = w.astype(jnp.int8) if w.dtype in INT4_TYPES else w
    x = x.astype(jnp.int8) if x.dtype in INT4_TYPES else x
    if swap_xw:
      ret = jnp.einsum(eqn_normalized, w, x)
    else:
      ret = jnp.einsum(eqn_normalized, x, w)

  if scale_act is not None:
    if scale_act.ndim == 0:
      scale *= scale_act
    else:
      filling_dims_lhs = _get_expand_dims_lhs(eqn)
      if filling_dims_lhs:
        scale_act = jnp.expand_dims(scale_act, filling_dims_lhs)
      ret = jnp.multiply(ret, scale_act)

  # Potentially expand dimensions of scale to match einsum output.
  filling_dims_rhs = _get_expand_dims_rhs(eqn)
  if not swap_xw and filling_dims_rhs:
    scale = jnp.expand_dims(scale, filling_dims_rhs)

  if scale_eqn is not None:
    ret = jnp.einsum(scale_eqn, ret, scale)
  else:
    ret = jnp.multiply(ret, scale)

  if zp is not None:
    if zp_eqn is not None:
      offset = jnp.einsum(zp_eqn, x, zp)
    else:
      offset = compute_offset(eqn_normalized, x, zp)
    ret = ret - offset

  return ret


def pass_through(x: JTensor, fn: Any) -> JTensor:
  # Create an exactly-zero expression with Sterbenz lemma that has an
  # exactly-one gradient.
  return x - jax.lax.stop_gradient(x) + jax.lax.stop_gradient(fn(x))


def reduce_precision(
    t: JTensor,
    contract_dims: Sequence[int] | None,
    need_gradient: bool = False,
    bits: int = 8,
    optimization_on_bound: bool = False,
    p_value: float = 1.0,
    percentile: float = 1.0,
    use_symmetric: bool = True,
    use_fp: bool = False,
    add_scale_eps: bool = False,
    per_channel: bool = False,
    random_rounding: bool = False,
    key: jax.Array | None = None,
    save_fp8_to_int8: bool = True,
    quant_method: str = 'default',
) -> tuple[JTensor, JTensor, JTensor | None]:
  """Reduce the precision of a tensor.

  Generic for all tensors.

  Args:
    t: Input tensor.
    contract_dims: Speficies contracting dimesnions of the input tensor.
    need_gradient: If gradient is needed out of this function.
    bits: Target number of bits.
    optimization_on_bound: If MAE bound optimizer is used.
    p_value: Exponent of the p-mean error metric. Default to 1.0 which is MAE.
    percentile: Percentile Factor to apply on the min/max range. Setting this to
      other than 1.0 disables optimization_on_bound.
    use_symmetric: If the input tensor is quantized symmetrically.
    use_fp: Use floating point.
    add_scale_eps: Add eps value or replace zero value by 1 to avoid division by
      zero.
    per_channel: use per-channel clipping optimization.
    random_rounding: round with uniform random.
    key: rng key for rounding.
    save_fp8_to_int8: If fp8 will be saved as int8. Only works when use_fp is
      true and should be removed eventually.
    quant_method: Quantization method: * 'default' - extracts min and max for
      quantization scale estimation. It is well applied for int8, in4, int2
      quantization. * 'bin' - binarization, where scale is defined by mean|w|. *
      'bin_norm' - binarization with weight normalization.

  Returns:
    A tuple of quantized tensor, quantization scale
      and quantization zero point (optional).
  """

  if bits == 1 and quant_method in ['bin', 'bin_norm']:
    if quant_method == 'bin_norm':
      mean = jnp.mean(t, axis=contract_dims, keepdims=True)
      t = t - mean

    # Remove zeros, so that below jnp.sign return only 1, -1.
    t = jnp.where(t == 0.0, 1e-6, t)
    scale = jnp.mean(jnp.abs(t), axis=contract_dims, keepdims=True)

    # Binarize, (conditioned that all zeros are removed above).
    t = pass_through(t, jnp.sign)
    return t, scale, None
  else:
    min_value, max_value = get_min_max(bits, use_fp=use_fp)

    if use_symmetric:
      bound = jnp.max(jnp.abs(t), axis=contract_dims, keepdims=True)
      scale_bound = max_value
    else:
      t_max = jnp.max(t, axis=contract_dims, keepdims=True)
      t_min = jnp.min(t, axis=contract_dims, keepdims=True)
      bound = t_max - t_min
      scale_bound = max_value - min_value

    if isinstance(percentile, JTensor) or percentile < 1.0:
      bound = jnp.multiply(bound, percentile)
    elif optimization_on_bound:
      bound = optimization.get_best_bound(
          t, bound, min_value, max_value, p_value, per_channel=per_channel
      )

    scale = bound / scale_bound

    if add_scale_eps:
      # Add epsilon to avoid divide-by-zero.
      scale = scale + jnp.finfo(t.dtype).eps
    else:
      scale = jnp.where(scale == 0.0, 1.0, scale)

    if use_symmetric:
      zp = None
      t = jnp.divide(t, scale)
    else:
      zp = min_value - t_min / scale
      t = jnp.divide(t, scale) + zp
      zp = jnp.multiply(scale, zp)

    if use_fp:
      # No need to round.
      t = jnp.clip(t, min_value, max_value).astype(jnp.float8_e4m3fn)
      # TODO(jianlijianli): refactor to remove this logic.
      if save_fp8_to_int8:
        # This is needed since fp8 cannot be saved.
        t = jax.lax.bitcast_convert_type(t, new_dtype=jnp.int8)
      else:
        # This is needed since bf16 x fp8 is not allowed.
        t = t.astype(jnp.bfloat16)
    else:
      if need_gradient:
        t = pass_through(t, jnp.round)
        t = jnp.clip(t, min_value, max_value)
      else:
        if random_rounding:
          t = t + jax.random.uniform(
              key=key, shape=t.shape, minval=-0.5, maxval=0.5
          )
        t = jnp.round(t)
        container_dtype = (
            jnp.int8 if bits <= 8 else jnp.int16 if bits <= 16 else jnp.int32
        )
        t = jnp.clip(t, min_value, max_value).astype(container_dtype)

    return t, scale, zp


def eqn_to_weight_contract_dims(eqn: str) -> list[int]:
  segs = eqn.split('->')
  ins = segs[0].split(',')
  w, out = ins[1].replace('.', ''), segs[1].replace('.', '')
  return [i for i, val in enumerate(w) if val not in out]


def eqn_to_activation_contract_dims(eqn: str) -> list[int]:
  segs = eqn.split('->')
  ins = segs[0].split(',')
  act, out = ins[0].replace('.', ''), segs[1].replace('.', '')
  return [-(i + 1) for i, val in enumerate(reversed(act)) if val not in out]


def reduce_einsum_weight_precision(
    eqn: str | None,
    t: JTensor,
    calculation_dtype: jnp.dtype = jnp.bfloat16,
    squeeze: bool = True,
    need_gradient: bool = False,
    bits: int = 8,
    optimization_on_bound: bool = False,
    percentile: float = 1.0,
    use_symmetric: bool = True,
    quant_method: str = 'default',
    contract_dims: Sequence[int] | None = None,
) -> tuple[JTensor, JTensor, JTensor | None]:
  """Reduce the precision of the weight of einsum.

  It uses per-channel quantization so einsum equation is passed in as well.

  Args:
    eqn: The equation for the einsum.
    t: The weight tensor for the einsum.
    calculation_dtype: The type for calculation.
    squeeze: If the output scale is squeezed.
    need_gradient: If gradient is needed out of this function.
    bits: Target number of bits.
    optimization_on_bound: If MAE bound optimizer is used.
    percentile: Percentile factor to apply on the min/max range.
    use_symmetric: If weights are quantized symmetrically.
    quant_method: Quantization method.
    contract_dims: Contraction dims. It can be used if eqn is not defined.

  Returns:
    A tuple of JTensors. The first one is the quantized weight and the second
    one is the scaling factor.
  """
  assert not (
      contract_dims is not None and eqn is not None
  ), 'both contract_dims and eqn can not be defined'

  if eqn is not None:
    contract_dims = eqn_to_weight_contract_dims(eqn)
  else:
    assert contract_dims, 'contract_dims must be defined if eqn is None'

  if t.dtype != calculation_dtype:
    t = t.astype(calculation_dtype)

  t, scale, zp = reduce_precision(
      t,
      contract_dims,
      need_gradient,
      bits,
      optimization_on_bound,
      percentile=percentile,
      use_symmetric=use_symmetric,
      quant_method=quant_method,
  )
  if squeeze:
    scale = jnp.squeeze(scale)
    if zp is not None:
      zp = jnp.squeeze(zp)
  return t, scale, zp


def clip_to_fp16(t: JTensor) -> JTensor:
  """Clip a tensor to fp16 gradually with f32 precision."""

  def _skip(t):
    return t

  def _clip(t):
    # TODO(jianlijianli): explore other ideas.
    return t * 0.99

  max_fp16 = jnp.finfo(jnp.float16).max  # min = -max for jnp.f16
  should_clip = jnp.abs(t) > max_fp16
  t = jax.lax.select(should_clip, _clip(t), _skip(t))
  return t


def get_scale_shape(
    weight_shape: Sequence[int], contract_dims: Sequence[int]
) -> Sequence[int]:
  """Gets scaler shape from weight_shape and contract_dims.

  Args:
    weight_shape: Weights shape.
    contract_dims: List of contraction dims.

  Returns:
    A scale shape.
  """
  return [
      dim_size
      for i, dim_size in enumerate(weight_shape)
      if i not in contract_dims
  ]


def get_sub_channel_shape(
    shape: Sequence[int],
    block_size: int,
    contract_dims: Sequence[int],
    insert_sub_channel: bool = True,
    error_on_misaligned_shape: bool = False,
) -> tuple[Sequence[int], Sequence[int]]:
  """Converts a shape's contract dim into sub-channel and block_size.

  It can be useful for reducing quantization error in post training quantization
  or quantization aware training, shown in https://arxiv.org/pdf/2305.16619.pdf.

  Args:
    shape: Tensor shape.
    block_size: Block size, it defines number of sub-channels.
    contract_dims: List of contraction dims.
    insert_sub_channel: If True it will insert new dim for sub channel, else it
      will use existing feature dim.
    error_on_misaligned_shape: If True it will raise an error for size not
      aligned with block_size. By default it is False. It allows to apply sub
      channel on layers with aligned shape and ignore layers with non aligned
      shape, so it will not block an experiment.

  Returns:
    A tuple of new shape with new contract_dims.
  """

  new_contract_dims = list(contract_dims)
  sub_channel_shape = list(shape)

  if block_size <= 0:
    return sub_channel_shape, new_contract_dims

  contract_shape = [shape[i] for i in new_contract_dims]
  # Index of dim in new_contract_dims, which corresponds to max dim among
  # contraction dims of input shape.
  max_contract_dim_ind = np.argmax(contract_shape)

  contract_dim = new_contract_dims[max_contract_dim_ind]
  if block_size >= shape[contract_dim]:
    logging.warning(
        'block_size %d is smaller than max input dim: %s; of contract dims: %s',
        block_size,
        str(shape),
        str(contract_dims),
    )
    return sub_channel_shape, new_contract_dims

  sub_channels, rem = divmod(shape[contract_dim], block_size)
  if rem > 0:
    if error_on_misaligned_shape:
      raise ValueError(
          f'block_size {block_size} must fully divide shape: {shape}'
          f'with contract dims: {contract_dims}'
      )
    else:
      logging.warning(
          'block_size %d must fully divide shape: %s; of contract dims: %s',
          block_size,
          str(shape),
          str(contract_dims),
      )
      return sub_channel_shape, new_contract_dims

  if insert_sub_channel:
    sub_channel_shape[contract_dim] = block_size
    sub_channel_shape.insert(contract_dim, sub_channels)

    # Shift all contract dims starting from max_contract_dim_ind:
    for i in range(max_contract_dim_ind, len(new_contract_dims)):
      new_contract_dims[i] += 1
  else:
    feature_dims = tuple(i for i in range(len(shape)) if i not in contract_dims)
    sub_channel_shape[contract_dim] = block_size
    sub_channel_shape[feature_dims[0]] *= sub_channels

  return sub_channel_shape, new_contract_dims


def fakequant_einsum(
    eqn: str,
    t: JTensor,
    bits: int = 8,
    calculation_dtype: jnp.dtype = jnp.float32,
    use_symmetric: bool = True,
    block_size: int = 0,
    use_fp: bool = False,
    quant_method: str = 'default',
) -> JTensor:
  """Nudges weight of einsum with FakeQuant.

    It quantizes weights (using einsum equation for getting contract_dims) then
    de-quantizes it and returns it as an output.
  Args:
    eqn: The equation for the einsum. Determines the channel dimension.
    t: The weight tensor for the einsum.
    bits: Target number of bits.
    calculation_dtype: The type for calculation.
    use_symmetric: Use symmetric quantization for weights.
    block_size: Block wise quantization size. 0 to turn if off.
    use_fp: Use floating point.

  quant_method: Quantization method:
    * 'default' - extracts min and max for quantization scale estimation.
      It is well applied for int8, in4, int2 quantization.
    * 'bin' - binarization, where scale is defined by mean|w|.
    * 'bin_norm' - binarization with weight normalization.

  Returns:
    The nudged weight tensor.
  """
  ret_type = t.dtype
  contract_dims = eqn_to_weight_contract_dims(eqn)
  original_shape = list(t.shape)
  if block_size > 0:
    sub_channel_shape, contract_dims = get_sub_channel_shape(
        original_shape, block_size, contract_dims
    )
    t = jnp.reshape(t, sub_channel_shape)

  if t.dtype != calculation_dtype:
    t = t.astype(calculation_dtype)

  if use_fp and bits == 4:
    # Short cut for fp4.
    fp4 = FP4()
    return fp4.nudge(t, contract_dims)

  # Quantize input tensor.
  q, scale, zp = reduce_precision(
      t,
      contract_dims,
      need_gradient=True,
      bits=bits,
      optimization_on_bound=False,
      use_symmetric=use_symmetric,
      quant_method=quant_method,
  )
  # De-quantized q using scale and zp.
  res = jnp.multiply(q, scale)
  if zp is not None:
    res = jnp.subtract(res, zp)
  if block_size > 0:
    res = jnp.reshape(res, original_shape)
  return res.astype(ret_type)


def reduce_precision_activation(
    t: JTensor,
    need_gradient: bool = False,
    bits: int = 8,
    contract_dims: Sequence[int] | None = None,
    symmetric: bool = True,
    percentile: float = 1.0,
) -> tuple[JTensor, JTensor, JTensor | None]:
  """Reduce the precision of activation.

  Args:
    t: Input tensor.
    need_gradient: If gradient is needed out of this function.
    bits: Target number of bits.
    contract_dims: Contract dimension.
    symmetric: If the activation is quantized symmetrically.
    percentile: Percentile Factor to apply on the min/max range. Setting this to
      other than 1.0 disables optimization_on_bound.

  Returns:
    A tuple of JTensors. The first one is the quantized activation, the
    second one is the scaling factor, and the last is the zero point.
  """
  qt, scale, zp = reduce_precision(
      t,
      contract_dims,
      need_gradient,
      bits,
      False,
      use_symmetric=symmetric,
      percentile=percentile,
  )
  return qt, scale, zp


# TODO(wppark): support clipping for activation, e.g. using standard deviation.
def reduce_einsum_activation_precision(
    eqn: str,
    t: JTensor,
    bits: int = 8,
    squeeze: bool = True,
    per_channel: bool = False,
    symmetric: bool = True,
    percentile: float = 1.0,
) -> tuple[JTensor, JTensor, JTensor | None]:
  """Reduce the precision of the activation of einsum.

  It uses per-tensor or per-toeken quantization so einsum equation is passed in
  as well.

  Args:
    eqn: The equation for the einsum.
    t: The activation tensor for the einsum.
    bits: Target number of bits.
    squeeze: If the output scale is squeezed.
    per_channel: Whether or not to quantize activation channel-wisely.
    symmetric: If the activation is quantized symmetrically.
    percentile: Percentile Factor to apply on the min/max range. Setting this to
      other than 1.0 disables optimization_on_bound.

  Returns:
    A tuple of JTensors. The first one is the quantized activation and the
    second one is the scaling factor.
  """
  if per_channel:
    contract_dims = eqn_to_activation_contract_dims(eqn)
  else:
    contract_dims = None

  t, scale, zp = reduce_precision(
      t,
      contract_dims,
      bits=bits,
      use_symmetric=symmetric,
      percentile=percentile,
  )

  if squeeze:
    scale = jnp.squeeze(scale, axis=contract_dims)
    if zp is not None:
      zp = jnp.squeeze(zp, axis=contract_dims)
  return t, scale, zp


def fakequant_activation(
    t: JTensor,
    bits: int = 8,
    eqn: str | None = None,
    per_channel: bool = False,
    symmetric: bool = True,
    percentile: float = 1.0,
) -> JTensor:
  """FakeQuant activation.

  Args:
    t: Activation tensor.
    bits: Target number of bits.
    eqn: Einsum equation. If None, do per-tensor quantization.
    per_channel: Whether or not to quantize activation channel-wisely.
    symmetric: If the activation is quantized symmetrically.
    percentile: Percentile Factor to apply on the min/max range. Setting this to
      other than 1.0 disables optimization_on_bound.

  Returns:
    Nudged activation.
  """
  contract_dims = None
  if per_channel:
    if eqn is None:
      raise ValueError('eqn should be defined with per_channel = True.')
    contract_dims = eqn_to_activation_contract_dims(eqn)
  qt, scale, zp = reduce_precision_activation(
      t,
      need_gradient=True,
      bits=bits,
      contract_dims=contract_dims,
      symmetric=symmetric,
      percentile=percentile,
  )
  res = jnp.multiply(qt, scale)
  if zp is not None:
    res = jnp.subtract(res, zp)
  return res.astype(t.dtype)


def compute_shape_with_subchannels(
    sub_channels: int,
    inputs_shape: Sequence[int],
    contract_dims: Sequence[int],
    min_sub_channel_size: int = -1,
) -> list[int]:
  """Computes new shape of input tensor for subchannel quantization.

  Args:
    sub_channels: Number of subchannels for splitting reduction dimension.
    inputs_shape: Input tensor shape.
    contract_dims: Axis along which to quantize acts (the non-feature axis).
    min_sub_channel_size: Minimum feature size, after which there will be
      no more sub channel division.

  Returns:
    New shape for subchannel quantization.
  """
  ndims = len(inputs_shape)

  feature_axis = tuple(i for i in range(ndims) if i not in contract_dims)

  new_inputs_shape = list(inputs_shape)
  # Find index of the max size
  axis_ind_max_size = 0
  max_size = 0
  for axis in contract_dims:
    if new_inputs_shape[axis]:
      if max_size < new_inputs_shape[axis]:
        max_size = new_inputs_shape[axis]
        axis_ind_max_size = axis

  if max_size < sub_channels:
    raise ValueError(f'Maximum dimension: {max_size} can not be '
                     f'smaller than sub_channels: {sub_channels}.')

  remainder = sub_channels
  while remainder > 1:
    # Split largest reduction dimension into sub channels
    # and increase the first feature dim proportionally.
    new_size = new_inputs_shape[axis_ind_max_size] // 2

    # The right way to do it is to introduce another dimension for
    # sub channel, but we also will have to modify all downstream ops.
    # To avoid it, we do feature size redistribution,
    # so feature size has to be divisible by 2:
    if new_size*2 != new_inputs_shape[axis_ind_max_size]:
      break

    if new_size < min_sub_channel_size:
      break

    new_inputs_shape[axis_ind_max_size] = new_size
    new_inputs_shape[feature_axis[0]] *= 2

    remainder /= 2
  return new_inputs_shape


def aqt_einsum(
    eqn: str,
    lhs: JTensor,
    rhs: JTensor,
    *,
    lhs_quantizer: Any,
    rhs_quantizer: Any,
    scale_eqn: str | None = None,
    zp_eqn: str | None = None,
) -> JTensor:
  """Quantized einsum with AQT style.

  Args:
    eqn: The valid binary einsum equation to use.
    lhs: Left-hand side of the einsum (mostly activation).
    rhs: Right-hand side of the einsum (mostly weight, can be activation).
    lhs_quantizer: The tensor quantizer for lhs.
    rhs_quantizer: The tensor quantizer for rhs.
    scale_eqn: Optional. Let tmp = einsum(eqn, x, w), scale_out =
      einsum(scale_eqn, tmp, out). Default scale_eqn act as '...z,z->...z'
    zp_eqn: Optional. ret = scale_out - einsum(zp_eqn, x, zp) Default zp_eqn act
      as '...y,z->...z'

  Returns:
    An array containing the result with the same dtype as 'lhs' and 'rhs'.
  """
  input_str, output_str, _ = einsum_parser.parse_einsum_input((eqn, lhs, rhs))
  eqn_normalized = input_str + '->' + output_str
  dimension_numbers, _ = utils.einsum_eqn_to_dimension_numbers(eqn_normalized)
  lhs_contract_dims, rhs_contract_dims = dimension_numbers[0]

  if (
      hasattr(rhs_quantizer, 'sub_channels')
      and rhs_quantizer.sub_channels is not None
  ):
    logging.warning(
        'sub_channels option will deprecate. Use the block_size API for sub'
        ' channel support instead.'
    )
    if lhs_quantizer.precision is not None:
      raise ValueError(
          'sub_channels is not implemented for activation quantization yet.'
      )

    input_shape = rhs.shape
    new_shape = compute_shape_with_subchannels(
        rhs_quantizer.sub_channels, rhs.shape, rhs_contract_dims
    )
    rhs = jnp.reshape(rhs, new_shape)

    # It is weights only fake quantization for evaluation purposes.
    rhs, rhs_scale, rhs_zp = rhs_quantizer.quantize(
        rhs, rhs_contract_dims, squeeze_scale=False, quantized_dtype=rhs.dtype
    )
    deq_rhs = rhs_quantizer.dequantize(
        rhs, rhs_scale, rhs_contract_dims, rhs_zp
    )
    deq_rhs = jnp.reshape(deq_rhs, input_shape)
    ret = jnp.einsum(eqn, lhs, deq_rhs)
  else:
    if scale_eqn is not None or zp_eqn is not None:
      if lhs_quantizer.precision is not None:
        raise NotImplementedError(
            'Activation quantization with custom scale_eqn/zp_eqn is not'
            ' implemented.'
        )
      rhs, rhs_scale, rhs_zp = rhs_quantizer.quantize(
          rhs, rhs_contract_dims, squeeze_scale=True, quantized_dtype=rhs.dtype
      )
      out_scale = rhs_scale
    else:
      lhs, lhs_scale, _ = lhs_quantizer.quantize(
          lhs, lhs_contract_dims, squeeze_scale=False, quantized_dtype=lhs.dtype
      )
      rhs, rhs_scale, rhs_zp = rhs_quantizer.quantize(
          rhs, rhs_contract_dims, squeeze_scale=False, quantized_dtype=rhs.dtype
      )
      out_scale = jnp.einsum(eqn, lhs_scale, rhs_scale)

    ret = jnp.einsum(eqn, lhs, rhs)

    if scale_eqn is not None:
      ret = jnp.einsum(scale_eqn, ret, out_scale)
    else:
      ret = jnp.multiply(ret, out_scale)

    if rhs_zp is not None:
      if (
          hasattr(lhs_quantizer, 'precision')
          and lhs_quantizer.precision is not None
      ):
        raise NotImplementedError(
            'Activation quantization with weight zero point '
            'is not supported yet.'
        )
      if zp_eqn is not None:
        offset = jnp.einsum(zp_eqn, lhs, rhs_zp)
      else:
        offset = compute_offset(eqn_normalized, lhs, jnp.squeeze(rhs_zp))
      ret = ret - offset

  return ret


def fakequant_vn(
    eqn: str,
    w: JTensor,
    next_prng_key: PRNGKey,
    wp: WeightQuantizationParams,
    step: JTensor | None = None,
    do_eval: bool = False,
    bits: int = 8,
    calculation_dtype: jnp.dtype = jnp.bfloat16,
    use_symmetric: bool = True,
):
  """Add variational noise to weight w.

  Args:
    eqn: The equation for the einsum between x and w.
    w: Input weight to add variational noise to.
    next_prng_key: RNG key.
    wp: Weight quantization parameters.
    step: Current training step.
    do_eval: Evaluation mode, if True.
    bits: Target number of bits.
    calculation_dtype: The type for calculation.
    use_symmetric: Use symmetric quantization for weights.

  Returns:
    The input with variational noise added according to params.
  """

  assert wp.vn_weight_norm_type in ('L2', 'Linf', 'PerChannelLinf')

  if None in [
      wp.vn_scale,
      wp.vn_start_step,
      wp.vn_noise_type,
      wp.vn_weight_norm_type,
      wp.stop_scale_gradient,
  ]:
    raise ValueError('VN parameter must be set.')

  if not use_symmetric:
    raise ValueError('Asymmetric quantization with VN is not supported yet.')

  if do_eval:
    # TODO(rybakov): replace fakequant by native quantization.
    logging.info('Eval fakequant_vn with quantization')
    return fakequant_einsum(
        eqn,
        w,
        bits=bits,
        calculation_dtype=calculation_dtype,
        use_symmetric=use_symmetric,
    )
  else:
    if wp.vn_start_step > 0:
      if step is None:
        raise ValueError('step can not be None if wp.vn_start_step > 0.')

    if wp.vn_noise_type == 'uniform':
      noises = jax.random.uniform(
          next_prng_key, shape=w.shape, minval=-0.5, maxval=0.5, dtype=w.dtype
      )
    elif wp.vn_noise_type == 'normal':
      noises = jax.random.normal(next_prng_key, shape=w.shape, dtype=w.dtype)
    else:
      raise ValueError('Unsupported noise type.')

    # During warmup period (defined by vn_start_step) there is no noise addition
    if wp.vn_start_step > 0:
      scale = jax.lax.select(step >= wp.vn_start_step, wp.vn_scale, 0.0)
    else:
      scale = wp.vn_scale

    if wp.vn_weight_norm_type == 'L2':
      raise ValueError('Not implemented.')
    elif wp.vn_weight_norm_type == 'Linf':
      # Per tensor scaling.
      scale *= jnp.max(jnp.abs(w))
    elif wp.vn_weight_norm_type == 'PerChannelLinf':
      # Per channel scaling.
      contract_dims = eqn_to_weight_contract_dims(eqn)
      scale *= jnp.max(jnp.abs(w), axis=contract_dims, keepdims=True)

    if wp.stop_scale_gradient:
      scale = jax.lax.stop_gradient(scale)

    return w + scale.astype(w.dtype) * noises  # pytype: disable=attribute-error


def factorize_weight(var: JTensor, rank: int) -> tuple[JTensor, JTensor]:
  """Apply SVD to variable and return two matrices.

  Args:
    var: JTensor to be factorized.
    rank: Inner rank of the factorized output. In terms of SVD, keeps top "rank"

  singular values and zeros out the rest.

  Returns:
    Two JTensors representing the truncated SVD version of var with rank "rank".
  The singular values are folded into the second matrix.
  """
  u, s, vh = jnp.linalg.svd(var, full_matrices=False)
  u_truncated, s_truncated, vh_truncated = (
      u[..., :, :rank],
      s[..., :rank],
      vh[..., :rank, :],
  )
  return u_truncated, jnp.einsum('...i,...ij->...ij', s_truncated, vh_truncated)
