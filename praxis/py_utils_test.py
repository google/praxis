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

"""Tests for Python utils."""

import collections
import re
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from flax import struct
import jax
from jax import numpy as jnp
from jax.experimental import pjit
import numpy as np
from praxis import base_layer
from praxis import py_utils
from praxis import test_utils
from praxis import train_states
import tensorflow.compat.v2 as tf


class PyUtilsTest(test_utils.TestCase):

  def test_reshard_empty_array(self):
    batch_size = 128
    empty_inputs = tf.ones(shape=(batch_size, 0))
    sharded_inputs = py_utils.reshard(empty_inputs)
    # Check the shape of returned inputs.
    num_devices = jax.local_device_count()
    self.assertEqual(sharded_inputs.shape,
                     (num_devices, batch_size // num_devices, 0))

  def test_extract_prefixed_keys_from_state_specs(self):
    w_sepc = base_layer.var_partition_specs(
        {'w': base_layer.WeightHParams(shape=(4, 8))},
        mesh_shape=[1, 1],
        device_axis_names=['a', 'b'])
    train_state_partition_specs = train_states.TrainState(
        step=pjit.PartitionSpec(), mdl_vars=w_sepc, opt_states={})
    nested_names = py_utils.extract_prefixed_keys_from_nested_map(
        train_state_partition_specs)
    flattened_names, _ = jax.tree_flatten(nested_names)
    self.assertListEqual(['step', 'mdl_vars/w'], flattened_names)

  def test_extract_prefixed_keys_from_nested_map(self):
    Point = collections.namedtuple('Point', ['x', 'y'])

    inputs = {'a': [1, 2, Point(x=3, y=4), (5, 6)], 'b': ('c', 'd')}
    outputs = py_utils.extract_prefixed_keys_from_nested_map(inputs)
    self.assertEqual(
        {
            'a': [
                'a[0]', 'a[1]',
                Point(x='a[2]/x', y='a[2]/y'), ('a[3][0]', 'a[3][1]')
            ],
            'b': ('b[0]', 'b[1]')
        }, outputs)

  def test_extract_prefixed_keys_from_dataclass(self):

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

    nested_data = py_utils.NestedMap(stats0=stats0, stats1=stats1)
    nested_names = py_utils.extract_prefixed_keys_from_nested_map(nested_data)
    flattened_nested_names, _ = jax.tree_flatten(nested_names)

    self.assertListEqual([
        'stats0/statistics', 'stats0/preconditioners', 'stats0/exponents',
        'stats1/statistics', 'stats1/preconditioners', 'stats1/exponents'
    ], flattened_nested_names)

  def test_sync_global_devices(self):
    py_utils.sync_global_devices('sync')

  def test_select_nodes_by_indices(self):
    result = py_utils.select_nodes_by_indices(
        (0, 1, 2), ('a', 'b', 'c'), ('A', 'B', 'C'), ('alpha', 'beta', 'gamma'))
    self.assertEqual(result, ('a', 'B', 'gamma'))

  def test_match_variable_names(self):
    tree = py_utils.NestedMap(
        a=py_utils.NestedMap(x=0, y=1, zz=2),
        b=py_utils.NestedMap(z=1),
    )
    expected = py_utils.NestedMap(
        a=py_utils.NestedMap(x=True, y=True, zz=False),
        b=py_utils.NestedMap(z=False),
    )
    result = py_utils.match_variable_names(tree, r'a/.')
    self.assertEqual(result, expected)
    expected.a.zz = True
    result = py_utils.match_variable_names(tree, [r'a/.', re.compile('.*zz')])
    self.assertEqual(result, expected)

  def test_update_matched_variables(self):
    old_tree = py_utils.NestedMap(
        a=py_utils.NestedMap(x=0, y=0, zz=0),
        b=py_utils.NestedMap(z=0),
    )
    new_tree = jax.tree_map(lambda x: x + 1, old_tree)
    result = py_utils.update_matched_variables(old_tree, new_tree,
                                               re.compile('.*z'))
    expected = py_utils.NestedMap(
        a=py_utils.NestedMap(x=0, y=0, zz=1),
        b=py_utils.NestedMap(z=1),
    )
    self.assertEqual(result, expected)
    result = py_utils.update_matched_variables(
        old_tree, new_tree, re.compile('.*z'), invert=True)
    expected_inv = py_utils.NestedMap(
        a=py_utils.NestedMap(x=1, y=1, zz=0),
        b=py_utils.NestedMap(z=0),
    )
    self.assertEqual(result, expected_inv)

  @parameterized.parameters(jnp.int32, jnp.float32, jnp.int64, jnp.float64)
  def test_get_large_negative_number(self, dtype):
    jax_number = py_utils.get_large_negative_number(dtype)
    self.assertDtypesMatch(jax_number, dtype)

  @parameterized.parameters(jnp.int32, jnp.float32, jnp.bool_)
  def test_sequence_mask(self, dtype):
    lengths = np.array([0, 1, 2, 3])
    mask = py_utils.sequence_mask(lengths, maxlen=4, dtype=dtype)
    expected = np.tri(4, k=-1, dtype=dtype)
    self.assertAllClose(mask, expected)

  @parameterized.parameters(jnp.int32, jnp.float32, jnp.bool_)
  def test_sequence_paddings(self, dtype):
    lengths = np.array([0, 1, 2, 3])
    paddings = py_utils.sequence_paddings(lengths, maxlen=4, dtype=dtype)
    expected = (1 - np.tri(4, k=-1)).astype(dtype)
    self.assertAllClose(paddings, expected)

  @parameterized.named_parameters(
      ('_numpy', np),
      ('_jax_numpy', jnp),
  )
  def test_tree_unstack(self, np_module):
    batch_axis, batch_size = 0, 8
    tree = py_utils.NestedMap(
        a=np_module.reshape(np_module.arange(batch_size), (batch_size, 1)),
        b=py_utils.NestedMap(
            c=np_module.reshape(
                np_module.arange(batch_size * 2 * 3), (batch_size, 2, 3)),
        ),
    )

    flat_trees = py_utils.tree_unstack(tree, batch_axis)
    self.assertLen(flat_trees, batch_size)

    # Merge tree back
    merged_tree = jax.tree_map(
        lambda x: np_module.expand_dims(x, batch_axis), flat_trees[0])

    def _concat_tree_with_batch(x_batch, y):
      y_batch = np_module.expand_dims(y, batch_axis)
      return np_module.concatenate((x_batch, y_batch), axis=batch_axis)

    for other_tree in flat_trees[1:]:
      merged_tree = jax.tree_map(
          _concat_tree_with_batch, merged_tree, other_tree)

    # Check all leaves are element-wise equal
    for l1, l2 in zip(jax.tree_leaves(tree), jax.tree_leaves(merged_tree)):
      self.assertArraysEqual(l1, l2)


if __name__ == '__main__':
  absltest.main()
