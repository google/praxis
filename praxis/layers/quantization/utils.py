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

"""Utilities for quantization."""

from typing import Optional, Tuple

from jax import lax


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
