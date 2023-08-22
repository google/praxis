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
import jax
from jax import numpy as jnp
import numpy as np
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
      ('trivial', TestPair(subset=123, superset=123)),
      ('lists_prefix', TestPair(subset=[1, 2], superset=[1, 2, 3])),
      ('lists', TestPair(subset=[1, 2, 3], superset=[1, 2, 3])),
      (
          'mixed_types',
          TestPair(
              subset={'a': [1, 2, 3], 'b': {'c': 'hello', 'd': [123]}},
              superset={'a': [1, 2, 3], 'b': {'c': 'hello', 'd': [123]}},
          ),
      ),
      (
          'mixed_types_strict',
          TestPair(
              subset={'a': [1, 2, 3]},
              superset={'a': [1, 2, 3], 'b': {'c': 'hello', 'd': [123]}},
          ),
      ),
      (
          'nestedmap',
          TestPair(
              subset=NestedMap.FromNestedDict({
                  'a': [1, 2, 3],
                  'b': {'c': 'hello', 'd': [123]},
              }),
              superset=NestedMap.FromNestedDict({
                  'a': [1, 2, 3],
                  'b': {'c': 'hello', 'd': [123]},
              }),
          ),
      ),
      (
          'nestedmap_strict',
          TestPair(
              subset=NestedMap.FromNestedDict({
                  'a': [1, 2, 3],
                  'b': {'c': 'hello', 'd': [123]},
              }),
              superset=NestedMap.FromNestedDict({
                  'a': [1, 2, 3],
                  'b': {'c': 'hello', 'd': [123]},
                  'e': 'not in subset',
              }),
          ),
      ),
      (
          'mixed_nested_dtypes',
          TestPair(
              subset={'a': [1, 2, 3]},
              superset=NestedMap.FromNestedDict(
                  {'a': [1, 2, 3, 4], 'b': 'hello'}
              ),
          ),
      ),
  )
  def test_is_subset(self, pair):
    self.assertTrue(trees.is_subset(pair.subset, pair.superset))

  @parameterized.named_parameters(
      ('trivial', TestPair(subset=345, superset=123)),
      ('lists_reorder', TestPair(subset=[1, 3, 2], superset=[1, 2, 3])),
      ('lists_strict', TestPair(subset=[1, 3], superset=[1, 2, 3])),
      ('lists_flipped', TestPair(subset=[1, 2, 3], superset=[2])),
      ('lists_prefix_flipped', TestPair(subset=[1, 2, 3], superset=[1, 2])),
      (
          'lists_types',
          TestPair(subset=[1, 2, 3, 'dev'], superset=[2, 3, 'testing']),
      ),
      (
          'mixed_types',
          TestPair(
              subset={'a': [12345678, 456]},
              superset={'a': [1, 2, 3], 'b': {'c': 'hello', 'd': [123]}},
          ),
      ),
      (
          'nestedmap',
          TestPair(
              subset=NestedMap.FromNestedDict({
                  'a': [999, 444],
                  'b': {'c': 'hello', 'd': [123]},
              }),
              superset=NestedMap.FromNestedDict({
                  'a': [1, 2, 3],
                  'b': {'c': 'hello', 'd': [123]},
              }),
          ),
      ),
      ('mixed_dtypes', TestPair(subset=(1, 2, 3, 4), superset=[1, 2, 3])),
      (
          'mixed_nested_dtypes',
          TestPair(
              subset=NestedMap.FromNestedDict(
                  {'a': (1, 2, 3, 4), 'b': 'hello'}
              ),
              superset={'a': [1, 2, 3]},
          ),
      ),
      (
          'dicts_simple',
          TestPair(subset={'a': 123, 'b': 456}, superset={'a': 123}),
      ),
      (
          'dicts_overlapping',
          TestPair(
              subset={'a': 123, 'b': 456},
              superset={'a': 123, 'c': 456},
          ),
      ),
      (
          'partitionspec',
          TestPair(subset=(), superset=jax.sharding.PartitionSpec()),
      ),
  )
  def test_is_not_subset(self, pair):
    self.assertFalse(trees.is_subset(pair.subset, pair.superset))

  def test_special_case(self):
    subset = {
        'eval_sample_weights': jax.ShapeDtypeStruct(
            shape=(128,), dtype=np.float32
        ),
        'ids': jax.ShapeDtypeStruct(shape=(128, 8208), dtype=np.int32),
        'inputs_indicator': jax.ShapeDtypeStruct(
            shape=(128, 8208), dtype=np.int32
        ),
        'labels': jax.ShapeDtypeStruct(shape=(128, 8208), dtype=np.int32),
        'paddings': jax.ShapeDtypeStruct(shape=(128, 8208), dtype=np.float32),
        'segment_ids': jax.ShapeDtypeStruct(shape=(128, 8208), dtype=np.int32),
        'segment_pos': jax.ShapeDtypeStruct(shape=(128, 8208), dtype=np.int32),
        'weights': jax.ShapeDtypeStruct(shape=(128, 8208), dtype=np.float32),
    }
    superset = {
        '_seqio_provenance/index_within_shard': jax.ShapeDtypeStruct(
            shape=(128,), dtype=np.int32
        ),
        '_seqio_provenance/num_shards': jax.ShapeDtypeStruct(
            shape=(128,), dtype=np.int32
        ),
        '_seqio_provenance/shard_index': jax.ShapeDtypeStruct(
            shape=(128,), dtype=np.int32
        ),
        'eval_sample_weights': jax.ShapeDtypeStruct(
            shape=(128,), dtype=np.float32
        ),
        'ids': jax.ShapeDtypeStruct(shape=(128, 8208), dtype=np.int32),
        'inputs_indicator': jax.ShapeDtypeStruct(
            shape=(128, 8208), dtype=np.int32
        ),
        'labels': jax.ShapeDtypeStruct(shape=(128, 8208), dtype=np.int32),
        'paddings': jax.ShapeDtypeStruct(shape=(128, 8208), dtype=np.float32),
        'segment_ids': jax.ShapeDtypeStruct(shape=(128, 8208), dtype=np.int32),
        'segment_pos': jax.ShapeDtypeStruct(shape=(128, 8208), dtype=np.int32),
        'weights': jax.ShapeDtypeStruct(shape=(128, 8208), dtype=np.float32),
    }

    self.assertTrue(trees.is_subset(subset, superset))

  @parameterized.named_parameters(
      ('trivial_int', 1, 1),
      ('trivial_list', [1, 2, 3], [1, 2, 3]),
      ('nestedmap', NestedMap(a=[1, 2, 3]), NestedMap(a=[1, 2, 3])),
  )
  def test_copy(self, orig, expected):
    arrayify = lambda t: jax.tree_util.tree_map(jnp.array, t)
    self.assertArraysEqual(trees.copy(arrayify(orig)), arrayify(expected))

  def test_copy_complex(self):
    original = NestedMap(
        a=jnp.array([1, 2, 3]),
        b=NestedMap(c=jnp.array([5, 5, 5])),
    )
    copy = trees.copy(original)

    copy.b.c += 1  # Does not modify original's version of b.c.

    self.assertArraysEqual(original.a, copy.a)
    self.assertArraysEqual(original.b.c, jnp.array([5, 5, 5]))
    self.assertArraysEqual(copy.b.c, jnp.array([6, 6, 6]))


if __name__ == '__main__':
  absltest.main()
