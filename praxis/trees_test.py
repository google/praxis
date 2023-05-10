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

"""Tests for trees."""
from typing import NamedTuple
from absl.testing import absltest
from absl.testing import parameterized

from praxis import py_utils
from praxis import pytypes
from praxis import test_utils
from praxis import trees

Nested = pytypes.Nested
NestedMap = py_utils.NestedMap


class TestPair(NamedTuple):
  subset: Nested
  superset: Nested


class TreesTest(test_utils.TestCase):

  @parameterized.named_parameters(
      ('trivial', TestPair(123, 123)),
      ('lists_prefix', TestPair([1, 2], [1, 2, 3])),
      ('lists', TestPair([1, 2, 3], [1, 2, 3])),
      (
          'mixed_types',
          TestPair(
              {'a': [1, 2, 3], 'b': {'c': 'hello', 'd': [123]}},
              {'a': [1, 2, 3], 'b': {'c': 'hello', 'd': [123]}},
          ),
      ),
      (
          'mixed_types_strict',
          TestPair(
              {'a': [1, 2, 3]},
              {'a': [1, 2, 3], 'b': {'c': 'hello', 'd': [123]}},
          ),
      ),
      (
          'nestedmap',
          TestPair(
              NestedMap.FromNestedDict({
                  'a': [1, 2, 3],
                  'b': {'c': 'hello', 'd': [123]},
              }),
              NestedMap.FromNestedDict({
                  'a': [1, 2, 3],
                  'b': {'c': 'hello', 'd': [123]},
              }),
          ),
      ),
      (
          'nestedmap_strict',
          TestPair(
              NestedMap.FromNestedDict({
                  'a': [1, 2, 3],
                  'b': {'c': 'hello', 'd': [123]},
              }),
              NestedMap.FromNestedDict({
                  'a': [1, 2, 3],
                  'b': {'c': 'hello', 'd': [123]},
                  'e': 'not in subset',
              }),
          ),
      ),
  )
  def test_is_subset(self, pair):
    self.assertTrue(trees.is_subset(pair.subset, pair.superset))

  @parameterized.named_parameters(
      ('trivial', TestPair(345, 123)),
      ('lists_reorder', TestPair([1, 3, 2], [1, 2, 3])),
      ('lists_strict', TestPair([1, 3], [1, 2, 3])),
      ('lists_flipped', TestPair([1, 2, 3], [2])),
      ('lists_types', TestPair([1, 2, 3, 'dev'], [2, 3, 'testing'])),
      (
          'mixed_types',
          TestPair(
              {'a': [12345678, 456]},
              {'a': [1, 2, 3], 'b': {'c': 'hello', 'd': [123]}},
          ),
      ),
      (
          'nestedmap',
          TestPair(
              NestedMap.FromNestedDict({
                  'a': [999, 444],
                  'b': {'c': 'hello', 'd': [123]},
              }),
              NestedMap.FromNestedDict({
                  'a': [1, 2, 3],
                  'b': {'c': 'hello', 'd': [123]},
              }),
          ),
      ),
  )
  def test_is_not_subset(self, pair):
    self.assertFalse(trees.is_subset(pair.subset, pair.superset))


if __name__ == '__main__':
  absltest.main()
