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

"""Utilities for quantization."""

from typing import List, Sequence, Type

import fiddle as fdl
from jax import lax
import jax.numpy as jnp
from praxis import base_layer
from praxis import pax_fiddle


JTensor = jnp.ndarray

INT4_TYPES = [jnp.int4, jnp.uint4]
INT_TYPES = [
    jnp.int4,
    jnp.uint4,
    jnp.int8,
    jnp.uint8,
    jnp.int16,
    jnp.uint16,
    jnp.int32,
    jnp.uint32,
]


def einsum_eqn_to_dimension_numbers(
    eqn: str,
) -> tuple[lax.DotDimensionNumbers, tuple[int, ...] | None]:
  """Convert einsum equation to dot_general dimension numbers and a permutation.

  Only supports equations meeting the following conditions:
    - there are two inputs
    - there are no repeated dims
    - there are no dynamic batch dims ('...')
    - all contractions dims are shared between lhs and rhs

  Args:
    eqn: The valid einsum equation to convert.

  Returns:
    lax.DotDimensionNumbers and an optional permutation to pass to
    jax.lax.transpose after jax.lax.dot_general.
  """
  if '.' in eqn:
    raise NotImplementedError(
        f"Dynamic batch dims ('...') are not supported. Input eqn: {eqn}. "
    )
  inputs, out_names = eqn.split('->')
  num_commas = inputs.count(',')
  if num_commas != 1:
    raise ValueError(
        f'einsum equation ({eqn}) expected two arguments, '
        f'but {num_commas+1} found.'
    )
  lhs_names, rhs_names = inputs.split(',')
  if len(lhs_names) != len(set(lhs_names)):
    raise ValueError(f'Repeated dim names are not supported, got {lhs_names=}')
  if len(rhs_names) != len(set(rhs_names)):
    raise ValueError(f'Repeated dim names are not supported, got {rhs_names=}')

  # Order batch_names by their appearance in out_names to minimize the need for
  # permutation.
  batch_names = [
      name for name in out_names if name in lhs_names and name in rhs_names
  ]
  lhs_batch_dims = [lhs_names.index(name) for name in batch_names]
  rhs_batch_dims = [rhs_names.index(name) for name in batch_names]

  # dot_general only supports contraction dims which exist in both arguments.
  lhs_contraction_names = set(lhs_names) - set(out_names)
  rhs_contraction_names = set(rhs_names) - set(out_names)
  if lhs_contraction_names != rhs_contraction_names:
    raise ValueError(
        'Contraction dims must be present in both lhs and rhs, but got '
        f'{lhs_contraction_names} and {rhs_contraction_names}'
    )
  # The order of the contraction dims does not matter to dot_general so long as
  # it is the same for both arguments, but ensuring that the order is
  # deterministic allows for easier validation of exported model artifacts.
  contraction_names = sorted(lhs_contraction_names)

  lhs_contraction_dims = [lhs_names.index(name) for name in contraction_names]
  rhs_contraction_dims = [rhs_names.index(name) for name in contraction_names]
  dimension_numbers = (
      (tuple(lhs_contraction_dims), tuple(rhs_contraction_dims)),
      (tuple(lhs_batch_dims), tuple(rhs_batch_dims)),
  )

  # Compute the order of the dot_general output names.
  shared_names = set(batch_names).union(contraction_names)
  lhs_unshared_names = [name for name in lhs_names if name not in shared_names]
  rhs_unshared_names = [name for name in rhs_names if name not in shared_names]
  dot_general_out_names = batch_names + lhs_unshared_names + rhs_unshared_names

  # Permute the dot_general output to match the einsum output if necessary.
  assert set(dot_general_out_names) == set(out_names), dot_general_out_names
  perm = None
  if dot_general_out_names != list(out_names):
    perm = tuple(dot_general_out_names.index(name) for name in out_names)
  return dimension_numbers, perm


def pack_4bit(
    x: JTensor, pack_dim: int, packed_dtype: jnp.dtype = jnp.int32
) -> JTensor:
  """Pack int8 or uint8 tensor where its values are actually int4 or uint4, to int32 or int8 nibble format along pack_dim.

  Args:
    x: Original int8 or uint8 tensor to pack.
    pack_dim: Dimension to pack along. x.shape[pack_dim] must be divisible by 8,
      when packed_dtype is int32 and divisible by 2 when target_type is int8.
      Also pack_dim must be < x.ndim - 1.
    packed_dtype: Target type to pack to, int32 or int8.

  Returns:
    int32 or int8 packed tensor where the pack_dim size is divided by 8
    from the original tensor x.
  """
  if packed_dtype == jnp.int8 and x.dtype == jnp.uint8:
    # It doesn't make sense to pack uint8 numbers into int4 as we'll
    # the range overlap between uint8 and int4 is [0..7].
    raise ValueError(
        'only int8 input dtype is supported when packing into int8. '
        f'Given {x.dtype}'
    )

  if x.dtype != jnp.int8 and x.dtype != jnp.uint8:
    raise ValueError(
        f'input dtype must be either int8 or uint8. Given {x.dtype}'
    )
  if pack_dim >= x.ndim - 1:
    raise ValueError(
        f'pack_dim must be < input ndim - 1. input shape {x.shape} and pack_dim'
        f' {pack_dim}'
    )
  if packed_dtype != jnp.int32 and packed_dtype != jnp.int8:
    raise ValueError(
        f'packed_dtype must be either int32 or int8. Given {packed_dtype}'
    )
  if packed_dtype == jnp.int32 and x.shape[pack_dim] % 8 != 0:
    raise ValueError(
        'input shape[pack_dim] must be divisible by 8 when target_type '
        f'is int32. Given shape {x.shape}'
    )
  if packed_dtype == jnp.int8 and x.shape[pack_dim] % 2 != 0:
    raise ValueError(
        'input shape[pack_dim] must be divisible by 2 when target_type '
        f'is int8. Given shape {x.shape}'
    )

  int4s_per_packed_type = 8 if packed_dtype == jnp.int32 else 2

  rep_shape = list(x.shape)
  rep_shape.insert(pack_dim + 1, int4s_per_packed_type)
  rep_shape[pack_dim] //= int4s_per_packed_type

  shifts = lax.broadcasted_iota(packed_dtype, rep_shape, pack_dim + 1)
  shifts <<= 2

  # Promote x to packed_dtype
  x = x & jnp.array(0x0F, packed_dtype)
  x = lax.reshape(x, rep_shape)
  x = x << shifts
  x = lax.reduce(x, jnp.array(0x0, packed_dtype), lax.add, [pack_dim + 1])
  return x


def unpack_4bit(
    packed: JTensor, pack_dim: int, original_dtype: jnp.dtype
) -> JTensor:
  """Unpack int32/int8 tensor packed by pack_4bit() to uint8/int8 tensor.

  Args:
    packed: int32 or int8 tensor that was packed by pack_4bit() function.
    pack_dim: Dimension that was used to pack along. pack_dim must be <
      packed.ndim - 1.
    original_dtype: dtype of the original tensor that was packed by pack_4bit()
      function. Must be either int8 or uint8.

  Returns:
    uint8/int8 unpack tensor where the pack_dim size is multiplied by 8/2 from
    the packed tensor. Which means that the returned shape is identical to the
    original shape before pack_4bit().
    Note that original input to pack_4bit() is int8 or uint8, so the unpacked
    tensor returned by unpack_4bit() is uint8/int8 with same values
    and shape of the original tensor.
  """
  if packed.dtype != jnp.int32 and packed.dtype != jnp.int8:
    raise ValueError(
        f'packed dtype must be either int32 or int8. Given {packed.dtype}'
    )
  if original_dtype != jnp.int8 and original_dtype != jnp.uint8:
    raise ValueError(
        f'original_dtype must be either int8 or uint8. Given {original_dtype}'
    )
  if pack_dim >= packed.ndim - 1:
    raise ValueError(
        f'pack_dim must be < input ndim - 1. input shape {packed.shape} and'
        f' pack_dim {pack_dim}'
    )

  packet_type_bits = 32 if packed.dtype == jnp.int32 else 8
  int4s_per_packed_type = packet_type_bits // 4

  rep_shape = list(packed.shape)
  rep_shape.insert(pack_dim + 1, int4s_per_packed_type)
  rep = jnp.broadcast_to(jnp.expand_dims(packed, pack_dim + 1), rep_shape)
  shifts = lax.broadcasted_iota(packed.dtype, rep_shape, pack_dim + 1)

  rep = lax.collapse(rep, pack_dim, pack_dim + 2)
  shifts = lax.collapse(shifts, pack_dim, pack_dim + 2)
  # Invert shifts table:
  # 0,1 -> 1,0 for int8
  # 0..7 -> 7..0 for int32
  shifts = int4s_per_packed_type - 1 - shifts
  # Multiply shifts table by 4
  shifts <<= 2
  rep <<= shifts
  if jnp.issubdtype(original_dtype, jnp.signedinteger):
    # Arithmetic shift is required to repsect negative numbers
    return lax.shift_right_arithmetic(
        rep, jnp.array(packet_type_bits - 4, packed.dtype)
    ).astype(original_dtype)
  else:
    return lax.shift_right_logical(
        rep, jnp.array(packet_type_bits - 4, packed.dtype)
    ).astype(original_dtype)


def dtype_to_bits(dtype: jnp.dtype) -> int:
  dtype = jnp.dtype(dtype) if not isinstance(dtype, jnp.dtype) else dtype
  # dtype.itemsize does not reflect int4 being smaller than int8.
  return 4 if dtype in INT4_TYPES else dtype.itemsize * 8


def bits_to_dtype(bits: int, signed: bool = True) -> jnp.dtype:
  """Returns the smallest int dtype that can represent a specific precision."""
  assert 1 <= bits <= 32, f'{bits=} must be between 1 and 32'
  if bits <= 4:
    return jnp.int4 if signed else jnp.uint4
  elif bits <= 8:
    return jnp.int8 if signed else jnp.uint8
  elif bits <= 16:
    return jnp.int16 if signed else jnp.uint16
  else:
    return jnp.int32 if signed else jnp.uint32


def get_smallest_matching_dtype(lhs: JTensor, rhs: JTensor) -> jnp.dtype:
  """Returns the smallest integer dtype that can represent both lhs and rhs."""
  if lhs.dtype not in INT_TYPES:
    raise ValueError(f'{lhs.dtype=} is not an int type: {INT_TYPES} ')
  if rhs.dtype not in INT_TYPES:
    raise ValueError(f'{rhs.dtype=} is not an int type: {INT_TYPES} ')

  lhs_signed = jnp.issubdtype(lhs.dtype, jnp.signedinteger)
  rhs_signed = jnp.issubdtype(rhs.dtype, jnp.signedinteger)

  if (lhs_signed and rhs_signed) or (not lhs_signed and not rhs_signed):
    # If the signedness matches, simply use the larger dtype.
    lhs_bits = dtype_to_bits(lhs.dtype)
    rhs_bits = dtype_to_bits(rhs.dtype)
    return lhs.dtype if lhs_bits >= rhs_bits else rhs.dtype
  else:
    signed = rhs if rhs_signed else lhs
    unsigned = lhs if rhs_signed else rhs
    signed_bits = dtype_to_bits(signed.dtype)
    unsigned_bits = dtype_to_bits(unsigned.dtype)
    if unsigned_bits < signed_bits:
      # If the unsigned dtype is smaller, it fits in the larger signed dtype.
      return signed.dtype
    else:
      # If the unsigned dtype is as big or larger than the signed dtype, then
      # represent both with the next largest signed dtype.
      if unsigned_bits < 32:
        return bits_to_dtype(unsigned_bits * 2, signed=True)
      else:
        # Since i64 support is usually disabled, just match XLA's behavior of
        # truncating ui32 to i32.
        return jnp.int32


def get_packed_shape(shape: Sequence[int], pack_dim: int, packing_factor: int):
  """Get packed shape where the original shape's pack_dim size is dividened by packing_factor."""
  if shape[pack_dim] % packing_factor != 0:
    raise ValueError(
        f'Packing supported for dim {pack_dim} size % {packing_factor} == 0'
        f' cases only. Given shape {shape}.'
    )
  return [
      d // packing_factor if i == pack_dim else d for i, d in enumerate(shape)
  ]


def find_target_tpl(
    config: (
        pax_fiddle.Config[base_layer.BaseLayer]
        | Sequence[pax_fiddle.Config[base_layer.BaseLayer]]
    ),
    targets: Type[base_layer.BaseLayer] | Sequence[Type[base_layer.BaseLayer]],
) -> Sequence[fdl.Config]:
  """Traverses the entire config tree to find Configs of the target types."""
  targets = list(targets) if hasattr(targets, '__iter__') else [targets]
  target_tpl = []
  for node, _ in fdl.daglish.iterate(config):
    if isinstance(node, fdl.Config) and any(
        issubclass(fdl.get_callable(node), target) for target in targets
    ):
      target_tpl.append(node)
  return target_tpl


def get_lora_shape_and_eqn(
    shape: Sequence[int], lora_size: int, eqn: str, max_reduction=True
) -> tuple[str, str, List[int], List[int], List[int], List[int]]:
  """Gets equations and shapes for LoRA weights of einsum equation.

  Args:
    shape: Weight shape.
    lora_size: Size of LoRA dimension.
    eqn: Einsum equation.
    max_reduction: It is used only for a case when there are two reduction dims.
      If True, then max reduction dim will be used as LoRA dim, else it will use
      both dims for two LoRA dims.

  Returns:
    Einsum equation for w_left.
    Einsum equation for w_right.
    Weight shape for w_left.
    Weight shape for w_right.
    Index of LoRA dim in w_left.
    Index of LoRA dim in w_right.
  """
  # Below comments are for example of eqn='...y,yz->...z'.
  eqn_split = eqn.split('->')
  assert len(eqn_split) == 2
  left_right = eqn_split[0]  # '...y,yz'
  left_right = left_right.split(',')
  assert len(left_right) == 2
  left, right = left_right[0], left_right[1]  # ('...y', 'yz')

  def map_str2ind(eqn_part):
    ch_map = {}
    ind = 0
    for ch in eqn_part:
      if ch != '.':
        ch_map[ch] = ind
        ind += 1
    return ch_map

  left_map = map_str2ind(left)
  right_map = map_str2ind(right)

  # Find unique character which is not part of eqn.
  lora_ch1 = None
  lora_ch2 = None
  for x in range(97, 123):
    ch = chr(x)
    if ch not in left_map and ch not in right_map:
      if lora_ch1 is None:
        lora_ch1 = ch
      elif lora_ch2 is None:
        lora_ch2 = ch
      else:
        break
  assert lora_ch1 is not None
  assert lora_ch2 is not None

  # Select reduction dimension.
  ch_reductions = []
  ch_reduction2 = None
  for ch in left_map:
    if ch in right_map:
      ch_reductions.append(ch)
  assert ch_reductions
  if len(ch_reductions) == 1:
    ch_reduction1 = ch_reductions[0]
  elif len(ch_reductions) == 2:
    # If there are several reduction dimensions then select the largest one
    # as LoRA dim.
    if max_reduction:
      max_reduction_size = 0
      for ch in ch_reductions:
        eqn_right_ind = right_map[ch]
        eqn_right_size = shape[eqn_right_ind]
        if max_reduction_size < eqn_right_size:
          max_reduction_size = eqn_right_size
          ch_reduction1 = ch
    else:
      ch_reduction1 = ch_reductions[0]
      ch_reduction2 = ch_reductions[1]
  else:
    raise ValueError(
        f'Unsupported number of reduction dims: {len(ch_reductions)}'
    )

  offset = 0
  if len(left) >= 3:
    if left[:3] == '...':
      offset = 3

  if ch_reduction2 is None:
    # Equation for w_left
    eqn_left_ind1 = left_map[ch_reduction1]
    new_right = [ch_reduction1, lora_ch1]
    new_left = list(left)
    new_left[eqn_left_ind1 + offset] = lora_ch1
    new_eqn_left = list(left) + [','] + new_right + ['->'] + new_left
    new_eqn_left = ''.join(new_eqn_left)

    # Equation for w_right
    new_right = list(right)
    eqn_right_ind1 = right_map[ch_reduction1]
    assert new_right[0] != '.'
    new_right[eqn_right_ind1] = lora_ch1
    new_eqn_right = new_left + [','] + new_right + ['->'] + list(eqn_split[1])
    new_eqn_right = ''.join(new_eqn_right)

    # Shapes for w_left and w_right
    left_shape = [shape[eqn_right_ind1], lora_size]
    right_shape = list(shape)
    right_shape[eqn_right_ind1] = lora_size
    return (
        new_eqn_left,
        new_eqn_right,
        left_shape,
        right_shape,
        [1],
        [eqn_right_ind1],
    )
  else:
    # Equation for w_left
    eqn_left_ind1 = left_map[ch_reduction1]
    eqn_left_ind2 = left_map[ch_reduction2]
    new_right = [ch_reduction1, ch_reduction2, lora_ch1, lora_ch2]
    new_left = list(left)
    new_left[eqn_left_ind1 + offset] = lora_ch1
    new_left[eqn_left_ind2 + offset] = lora_ch2
    new_eqn_left = list(left) + [','] + new_right + ['->'] + new_left
    new_eqn_left = ''.join(new_eqn_left)

    # Equation for w_right
    new_right = list(right)
    eqn_right_ind1 = right_map[ch_reduction1]
    eqn_right_ind2 = right_map[ch_reduction2]
    assert new_right[0] != '.'
    new_right[eqn_right_ind1] = lora_ch1
    new_right[eqn_right_ind2] = lora_ch2
    new_eqn_right = new_left + [','] + new_right + ['->'] + list(eqn_split[1])
    new_eqn_right = ''.join(new_eqn_right)

    # Shapes for w_left and w_right
    left_shape = [
        shape[eqn_right_ind1],
        shape[eqn_right_ind2],
        lora_size,
        lora_size,
    ]
    right_shape = list(shape)
    right_shape[eqn_right_ind1] = lora_size
    right_shape[eqn_right_ind2] = lora_size
    return (
        new_eqn_left,
        new_eqn_right,
        left_shape,
        right_shape,
        [2, 3],
        [eqn_right_ind1, eqn_right_ind2],
    )


def get_left_weight_split_dims_mapping(
    weight_split_dims_mapping: tuple[str | None, ...] | None,
    eqn_left_ind: List[int],
) -> tuple[str | None, ...] | None:
  if weight_split_dims_mapping is None:
    return None
  else:
    if len(eqn_left_ind) == 1:
      return (weight_split_dims_mapping[0], None)
    elif len(eqn_left_ind) == 2:
      return (
          weight_split_dims_mapping[0],
          weight_split_dims_mapping[1],
          None,
          None,
      )
    else:
      raise ValueError(
          f'Usupported number of reduction dims {len(eqn_left_ind)}'
      )


def get_right_weight_split_dims_mapping(
    weight_split_dims_mapping: tuple[str | None, ...] | None,
    eqn_right_ind: List[int],
) -> tuple[str | None, ...] | None:

  if weight_split_dims_mapping is None:
    return None
  else:
    out_weight_split_dims_mapping = list(weight_split_dims_mapping)
    for ind in eqn_right_ind:
      out_weight_split_dims_mapping[ind] = None
    return tuple(out_weight_split_dims_mapping)
