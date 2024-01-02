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

import collections
import re
from typing import Any, List, NamedTuple

from absl.testing import absltest
from absl.testing import parameterized
from flax import struct
import jax
from jax import numpy as jnp
from jax import tree_util as jtu
import numpy as np
from praxis import base_layer
from praxis import pytypes
from praxis import test_utils
from praxis import trees


class TestPair(NamedTuple):
  subset: pytypes.Nested
  superset: pytypes.Nested


class TrainState(struct.PyTreeNode):
  """Simple train state."""

  step: base_layer.JTensorOrPartitionSpec
  mdl_vars: base_layer.NestedJTensorOrPartitionSpec
  opt_states: List[base_layer.NestedJTensorOrPartitionSpec]


def _is_subset_pairs():
  return (
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
              subset=pytypes.NestedMap.FromNestedDict({
                  'a': [1, 2, 3],
                  'b': {'c': 'hello', 'd': [123]},
              }),
              superset=pytypes.NestedMap.FromNestedDict({
                  'a': [1, 2, 3],
                  'b': {'c': 'hello', 'd': [123]},
              }),
          ),
      ),
      (
          'nestedmap_strict',
          TestPair(
              subset=pytypes.NestedMap.FromNestedDict({
                  'a': [1, 2, 3],
                  'b': {'c': 'hello', 'd': [123]},
              }),
              superset=pytypes.NestedMap.FromNestedDict({
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
              superset=pytypes.NestedMap.FromNestedDict(
                  {'a': [1, 2, 3, 4], 'b': 'hello'}
              ),
          ),
      ),
  )


def _is_not_structural_subset_pairs():
  return (
      ('lists_flipped', TestPair(subset=[1, 2, 3], superset=[2])),
      ('lists_prefix_flipped', TestPair(subset=[1, 2, 3], superset=[1, 2])),
      (
          'lists_types',
          TestPair(subset=[1, 2, 3, 'dev'], superset=[2, 3, 'testing']),
      ),
      ('mixed_dtypes', TestPair(subset=(1, 2, 3, 4), superset=[1, 2, 3])),
      (
          'mixed_nested_dtypes',
          TestPair(
              subset=pytypes.NestedMap.FromNestedDict(
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


def _is_not_elementwise_subset_pairs():
  return (
      ('trivial', TestPair(subset=345, superset=123)),
      ('lists_reorder', TestPair(subset=[1, 3, 2], superset=[1, 2, 3])),
      ('lists_strict', TestPair(subset=[1, 3], superset=[1, 2, 3])),
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
              subset=pytypes.NestedMap.FromNestedDict({
                  'a': [999, 444],
                  'b': {'c': 'hello', 'd': [123]},
              }),
              superset=pytypes.NestedMap.FromNestedDict({
                  'a': [1, 2, 3],
                  'b': {'c': 'hello', 'd': [123]},
              }),
          ),
      ),
  )


class TreesTest(test_utils.TestCase):

  def assert_treedefs_match(self, structure1, structure2):
    _, treedef1 = jax.tree_util.tree_flatten(structure1)
    _, treedef2 = jax.tree_util.tree_flatten(structure2)
    self.assertEqual(treedef1, treedef2)

  @parameterized.named_parameters(*_is_subset_pairs())
  def test_is_subset(self, pair):
    self.assertTrue(trees.is_subset(pair.subset, pair.superset))

  @parameterized.named_parameters(*_is_subset_pairs())
  def test_extract_subset(self, pair):
    subset_extracted = trees.extract_elements_matching_subset_structure(
        pair.subset, pair.superset
    )
    self.assertTrue(trees.is_subset(subset_extracted, pair.superset))
    self.assert_treedefs_match(pair.subset, subset_extracted)

  @parameterized.named_parameters(
      *(_is_not_structural_subset_pairs() + _is_not_elementwise_subset_pairs())
  )
  def test_is_not_subset(self, pair):
    self.assertFalse(trees.is_subset(pair.subset, pair.superset))

  @parameterized.named_parameters(*_is_not_structural_subset_pairs())
  def test_extract_subset_raises_on_structural_mismatch(self, pair):
    with self.assertRaises(ValueError):
      subset_extracted = trees.extract_elements_matching_subset_structure(
          pair.subset, pair.superset
      )

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
    subset_extracted = trees.extract_elements_matching_subset_structure(
        subset, superset
    )
    self.assertTrue(trees.is_subset(subset_extracted, superset))
    self.assert_treedefs_match(subset, subset_extracted)

  @parameterized.named_parameters(
      ('trivial_int', 1, 1),
      ('trivial_list', [1, 2, 3], [1, 2, 3]),
      (
          'nestedmap',
          pytypes.NestedMap(a=[1, 2, 3]),
          pytypes.NestedMap(a=[1, 2, 3]),
      ),
  )
  def test_copy(self, orig, expected):
    arrayify = lambda t: jtu.tree_map(jnp.array, t)
    self.assertArraysEqual(trees.copy(arrayify(orig)), arrayify(expected))

  def test_copy_complex(self):
    original = pytypes.NestedMap(
        a=jnp.array([1, 2, 3]),
        b=pytypes.NestedMap(c=jnp.array([5, 5, 5])),
    )
    copy = trees.copy(original)

    copy.b.c += 1  # Does not modify original's version of b.c.

    self.assertArraysEqual(original.a, copy.a)
    self.assertArraysEqual(original.b.c, jnp.array([5, 5, 5]))
    self.assertArraysEqual(copy.b.c, jnp.array([6, 6, 6]))

  def test_to_path_tree_from_state_specs(self):
    w_sepc = base_layer.var_partition_specs(
        {'w': base_layer.WeightHParams(shape=(4, 8))},
        mesh_shape=[1, 1],
        device_axis_names=['a', 'b'],
    )
    train_state_partition_specs = TrainState(
        step=jax.sharding.PartitionSpec(), mdl_vars=w_sepc, opt_states=[]
    )
    nested_names = trees.to_path_tree(train_state_partition_specs)
    self.assertListEqual(['step', 'mdl_vars/w'], jtu.tree_leaves(nested_names))

  def test_to_path_tree_from_nested_map(self):
    Point = collections.namedtuple('Point', ['x', 'y'])

    inputs = {'a': [1, 2, Point(x=3, y=4), (5, 6)], 'b': ('c', 'd')}
    outputs = trees.to_path_tree(inputs)
    self.assertEqual(
        {
            'a': [
                'a[0]',
                'a[1]',
                Point(x='a[2]/x', y='a[2]/y'),
                ('a[3][0]', 'a[3][1]'),
            ],
            'b': ('b[0]', 'b[1]'),
        },
        outputs,
    )

  def test_to_path_tree_from_dataclass(self):
    @struct.dataclass
    class GlobalShardedParameterStats:
      statistics: np.ndarray  # Statistics
      preconditioners: np.ndarray  # Preconditioners
      exponents: np.ndarray  # exponents
      index_start: int = struct.field(pytree_node=False)
      sizes: Any = struct.field(pytree_node=False)

    stats0 = GlobalShardedParameterStats(
        statistics=np.array([0], dtype=np.float32),
        preconditioners=np.array([1, 1], dtype=np.float32),
        exponents=np.array([2, 2, 2], dtype=np.float32),
        index_start=0,
        sizes=0,
    )
    # Even though the `preconditioners` is first here, the order is decided
    # by the order in `GlobalShardedParameterStats` class.
    stats1 = GlobalShardedParameterStats(
        preconditioners=np.array([5, 5], dtype=np.float32),
        statistics=np.array([4], dtype=np.float32),
        exponents=np.array([6, 6, 6], dtype=np.float32),
        index_start=1,
        sizes=1,
    )

    nested_data = pytypes.NestedMap(stats0=stats0, stats1=stats1)
    nested_names = trees.to_path_tree(nested_data)
    flattened_nested_names, _ = jax.tree_util.tree_flatten(nested_names)

    self.assertListEqual(
        [
            'stats0/statistics',
            'stats0/preconditioners',
            'stats0/exponents',
            'stats1/statistics',
            'stats1/preconditioners',
            'stats1/exponents',
        ],
        flattened_nested_names,
    )

  def test_to_path_tree_using_is_leaf(self):
    class Masked:
      """Test class."""

    Point = collections.namedtuple('Point', ['x', 'y'])

    inputs = {
        'a': [1, 2, Point(x=3, y=4), (5, 6, Masked())],
        'b': ('c', 'd'),
        'e': Masked(),
    }
    outputs = trees.to_path_tree(
        inputs, is_leaf=lambda x: isinstance(x, Masked)
    )
    self.assertEqual(
        {
            'a': [
                'a[0]',
                'a[1]',
                Point(x='a[2]/x', y='a[2]/y'),
                ('a[3][0]', 'a[3][1]', None),
            ],
            'b': ('b[0]', 'b[1]'),
            'e': None,
        },
        outputs,
    )

  def test_match_variable_names(self):
    tree = pytypes.NestedMap(
        a=pytypes.NestedMap(x=0, y=1, zz=2),
        b=pytypes.NestedMap(z=1),
    )
    expected = pytypes.NestedMap(
        a=pytypes.NestedMap(x=True, y=True, zz=False),
        b=pytypes.NestedMap(z=False),
    )
    result = trees.fullmatch_path(tree, r'a/.')
    self.assertEqual(result, expected)
    expected.a.zz = True
    result = trees.fullmatch_path(tree, [r'a/.', re.compile('.*zz')])
    self.assertEqual(result, expected)


if __name__ == '__main__':
  absltest.main()
