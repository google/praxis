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

from typing import Optional, Tuple, Sequence

from jax import lax
import jax.numpy as jnp

JTensor = jnp.ndarray


def einsum_eqn_to_dimension_numbers(
    eqn: str,
) -> Tuple[lax.DotDimensionNumbers, Optional[Tuple[int, ...]]]:
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
    raise NotImplementedError("Dynamic batch dims ('...') are not supported.")
  inputs, out_names = eqn.split('->')
  num_commas = inputs.count(',')
  if num_commas != 1:
    raise ValueError(f'einsum equation ({eqn}) expected two arguments, '
                     f'but {num_commas+1} found.')
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
  contraction_names = lhs_contraction_names

  # The order of the contraction dims does not matter so long as it is the same
  # for both arguments.
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


def pack_4bit(x: JTensor, pack_dim: int) -> JTensor:
  """Pack int8 or uint8 tensor where its values are actually int4 or uint4, to int32 nibble format along pack_dim.

  Args:
    x: Original int8 or uint8 tensor to pack.
    pack_dim: Dimension to pack along. x.shape[pack_dim] must be divisible by 8,
      and pack_dim must be < x.ndim - 1.

  Returns:
    int32 packed tensor where the pack_dim size is dividened by 8 from the
    original tensor x.
  """
  if x.dtype != jnp.int8 and x.dtype != jnp.uint8:
    raise ValueError(
        f'input dtype must be either int8 or uint8. Given {x.dtype}'
    )
  if pack_dim >= x.ndim - 1:
    raise ValueError(
        f'pack_dim must be < input ndim - 1. input shape {x.shape} and pack_dim'
        f' {pack_dim}'
    )
  if x.shape[pack_dim] % 8 != 0:
    raise ValueError(
        f'input shape[pack_dim] must be divisible by 8. Given shape {x.shape}'
    )

  packed_dtype = jnp.int32
  rows = x.shape[pack_dim]
  cols = x.shape[pack_dim + 1]
  blocks = rows // 8

  rep_shape = list(x.shape)
  rep_shape.insert(pack_dim + 1, 8)
  rep_shape[pack_dim] //= 8

  shifts = lax.broadcasted_iota(jnp.int32, rep_shape, pack_dim + 1)
  shifts *= 4

  # Promote x to int32
  x = x & jnp.array(0x0F, packed_dtype)
  x = lax.reshape(x, rep_shape)
  x = x << shifts
  x = lax.reduce(x, jnp.array(0x0, packed_dtype), lax.add, [pack_dim + 1])
  return x


def unpack_4bit(
    packed: JTensor, pack_dim: int, original_dtype: jnp.dtype
) -> JTensor:
  """Unpack int32 tensor packed by pack_4bit() to int32 tensor.

  Args:
    packed: int32 tensor that was packed by pack_4bit() function.
    pack_dim: Dimension that was used to pack along. pack_dim must be <
      packed.ndim - 1.
    original_dtype: dtype of the original tensor that was packed by pack_4bit()
      function. Must be either int8 or uint8.

  Returns:
    int32 unpack tensor where the pack_dim size is multipled by 8 from the
    packed tensor. Which means that the returned shape is identical to the
    original shape before pack_4bit().
    Note that original input to pack_4bit() is int8 or uint8, but the unpacked
    tensor returned by unpack_4bit() is int32 with same values and shape of the
    original tensor.
  """
  if packed.dtype != jnp.int32:
    raise ValueError(f'packed dtype must be either int32. Given {packed.dtype}')
  if original_dtype != jnp.int8 and original_dtype != jnp.uint8:
    raise ValueError(
        f'original_dtype must be either int8 or uint8. Given {original_dtype}'
    )
  if pack_dim >= packed.ndim - 1:
    raise ValueError(
        f'pack_dim must be < input ndim - 1. input shape {packed.shape} and'
        f' pack_dim {pack_dim}'
    )

  rep_shape = list(packed.shape)
  rep_shape.insert(pack_dim + 1, 8)
  rep = jnp.broadcast_to(jnp.expand_dims(packed, pack_dim + 1), rep_shape)
  shifts = lax.broadcasted_iota(jnp.int32, rep_shape, pack_dim + 1)

  rep = lax.collapse(rep, pack_dim, pack_dim + 2)
  shifts = lax.collapse(shifts, pack_dim, pack_dim + 2)
  shifts = 7 - shifts
  shifts *= 4
  rep <<= shifts
  if original_dtype == jnp.int8:
    return lax.shift_right_arithmetic(rep, 28)
  else:
    return lax.shift_right_logical(rep, 28)


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
