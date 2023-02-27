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

"""Test utils for quantization test."""
import itertools
from typing import Any, Dict, Sequence, Optional, List


def to_list(w):
  """Convert the given nested array type into a list.

  Args:
    w: Target array-like object.

  Returns:
    Value converted into python list.
  """
  if w is None:
    return None

  try:
    ret = float(w)
    if ret.is_integer():
      return int(ret)
    # TODO(dhchoi): Instead of rounding value here, implement a custom function
    # to approximately compare the two values of the nested lists.
    return round(ret, 5)
  except Exception:  # pylint: disable=broad-except
    pass

  return [to_list(v) for v in w]


def generate_attention_projection_test_config(
    additional_feature_funcs: Optional[List[Any]] = None
) -> Sequence[Dict[str, Any]]:
  """Function to generate test configurations for AttentionProjection layer.

  Args:
    additional_feature_funcs: Additional functions to further populate the
    configuration.

  Returns:
    Test configurations for AttentionProjection layer.
  """
  keys = [
      'is_output_projection',
      'use_bias',
      'attention_combine_dims',
      'use_nhd_shape',
      'is_weight_symmetric',
  ]

  # If attention_combine_dims is set to True, use_bias should be set to False.
  boolean_flags = [
      [True, True, False, True],
      [True, False, True, True],
      [True, True, False, False],
      [True, False, True, False],
      [False, True, False, True],
      [False, False, True, True],
      [False, False, False, True],
      [False, False, True, False],
      [False, True, False, False],
      [True, False, False, False],
      [False, False, False, False],
  ]

  weight_symmetric = [True, False]
  cases = []
  for case in itertools.product(boolean_flags, weight_symmetric):
    cases.append(case[0] + [case[1]])

  if additional_feature_funcs is not None:
    for func in additional_feature_funcs:
      keys, cases = func(keys, cases)

  return [dict(zip(keys, case)) for case in cases]


def generate_combined_qkv_projection_test_config(
    additional_feature_funcs: Optional[List[Any]] = None
) -> Sequence[Dict[str, Any]]:
  """Function to generate test configurations for CombinedQKVProjection layer.

  Args:
    additional_feature_funcs: Additional functions to further populate the
    configuration.

  Returns:
    Test configurations for CombinedQKVProjection layer.
  """
  keys = [
      'use_bias',
      'attention_combine_dims',
      'is_weight_symmetric',
  ]

  boolean_flags = [
      [True, True],
      [True, False],
      [False, True],
      [False, False],
  ]

  weight_symmetric = [True, False]
  cases = []
  for case in itertools.product(boolean_flags, weight_symmetric):
    cases.append(case[0] + [case[1]])

  if additional_feature_funcs is not None:
    for func in additional_feature_funcs:
      keys, cases = func(keys, cases)

  return [dict(zip(keys, case)) for case in cases]
