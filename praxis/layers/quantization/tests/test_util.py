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

"""Test utils for quantization test."""
import itertools
from typing import Any, Dict, List, Optional, Sequence

from praxis import test_utils


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
    return ret
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


def generate_linears_test_config(
    additional_feature_funcs: Optional[List[Any]] = None,
) -> Sequence[Dict[str, Any]]:
  """Function to generate test configurations for Linears layer.

  Args:
    additional_feature_funcs: Additional functions to further populate the
      configuration.

  Returns:
    Test configurations for Linears layer.
  """
  keys = [
      'is_weight_symmetric',
  ]

  cases = [[True], [False]]

  if additional_feature_funcs is not None:
    for func in additional_feature_funcs:
      keys, cases = func(keys, cases)

  return [dict(zip(keys, case)) for case in cases]


def generate_dotproduct_attention_test_config(
    additional_feature_funcs: Optional[List[Any]] = None
) -> Sequence[Dict[str, Any]]:
  """Function to generate test configurations for DotProductAttention layer.

  Args:
    additional_feature_funcs: Additional functions to further populate the
    configuration.

  Returns:
    Test configurations for DotProductAttention layer.
  """
  keys = [
      'dconv_qkv',
      'combine_qkv',
      'output_proj_use_nhd_shape',
      'use_rotary_position_emb',
      'cast_rotary_position_emb',
      'zero_fully_masked',
      'simulate_packed'
  ]

  flags = [[True, False] for _ in range(len(keys))]

  cases = []
  for case in itertools.product(*flags):
    cases.append(list(case))

  if additional_feature_funcs is not None:
    for func in additional_feature_funcs:
      keys, cases = func(keys, cases)

  return [dict(zip(keys, case)) for case in cases]


def generate_one_headed_attention_projection_test_config(
    additional_feature_funcs: Optional[List[Any]] = None
) -> Sequence[Dict[str, Any]]:
  """Generate test configurations for OneHeadedAttentionProjection layer.

  Args:
    additional_feature_funcs: Additional functions to further populate the
    configuration.

  Returns:
    Test configurations for OneHeadedAttentionProjection layer.
  """
  keys = [
      'use_bias',
      'is_weight_symmetric',
  ]

  bias = [True, False]
  weight_symmetric = [True, False]

  cases = []
  for case in itertools.product(bias, weight_symmetric):
    cases.append([case[0], case[1]])

  if additional_feature_funcs is not None:
    for func in additional_feature_funcs:
      keys, cases = func(keys, cases)

  return [dict(zip(keys, case)) for case in cases]


class QuantizationTestCase(test_utils.TestCase):
  """Test case class for quantized layers.
  """

  def assertNestedListClose(self, list1, list2, places=4):
    """Function to compare the two nested lists if they are close enough.

    Args:
      list1: First list to compare.
      list2: Second list to compare.
      places: Places to match.
    """
    if list1 is None and list2 is None:
      return

    if (list1 is not None and list2 is None) or (
        list1 is None and list2 is not None
    ):
      self.fail('Comparing None with not-None value.')

    if isinstance(list1, list) and isinstance(list2, list):
      self.assertEqual(len(list1), len(list2))
    elif not isinstance(list1, list) and not isinstance(list2, list):
      self.assertAlmostEqual(list1, list2, places)
      return
    else:
      self.fail(f'Comparing non-list with list: {list1} and {list2}.')

    for v1, v2 in zip(list1, list2):
      self.assertNestedListClose(v1, v2, places)
