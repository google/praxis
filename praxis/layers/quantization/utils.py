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

from typing import Optional, Sequence, Tuple

from jax import lax


def convert_einsum_eqn_to_dimension_numbers(
    eqn) -> Tuple[lax.DotDimensionNumbers, Optional[Sequence[int]]]:
  """Convert einsum equantion to dot_general dimenion numbers."""
  # Assume that the einsum equation is valid and has two arguments.
  # Note that this function provides conversion from einsum equation to
  # corresponding dimension numbers only for limited cases. It is supposed to
  # work well for this quantized attentions at least.
  inputs, out = eqn.split('->')
  num_commas = inputs.count(',')
  if num_commas != 1:
    raise ValueError(f'einsum equation ({eqn}) expected two arguments, '
                     f'but {num_commas+1} found.')
  lhs_names, rhs_names = inputs.split(',')
  lhs_names = lhs_names.replace('.', '')
  rhs_names = rhs_names.replace('.', '')
  out = out.replace('.', '')

  common_names = []
  for i, lhs_name in enumerate(lhs_names):
    for j, rhs_name in enumerate(rhs_names):
      if lhs_name == rhs_name:
        common_names.append((i, j, lhs_name))

  lhs_contract_dims = []
  rhs_contract_dims = []
  lhs_batch_dims = []
  rhs_batch_dims = []

  for lhs_dim_num, rhs_dim_num, name in common_names:
    if name in out:
      lhs_batch_dims.append(lhs_dim_num)
      rhs_batch_dims.append(rhs_dim_num)
    else:
      lhs_contract_dims.append(lhs_dim_num)
      rhs_contract_dims.append(rhs_dim_num)

  dimension_numbers = ((tuple(lhs_contract_dims), tuple(rhs_contract_dims)),
                       (tuple(lhs_batch_dims), tuple(rhs_batch_dims)))

  # Check if a transpose is needed or not.
  common_names = [name for _, _, name in common_names]
  lhs_remaining_names = [name for name in lhs_names if name not in common_names]
  rhs_remaining_names = [name for name in rhs_names if name not in common_names]
  result_names = lhs_remaining_names + rhs_remaining_names
  perm = None
  if result_names != list(out):
    perm = tuple(out.index(name) for name in result_names)
  return dimension_numbers, perm