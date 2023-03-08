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

"""Flax-related utils."""

from typing import Optional

from flax import traverse_util
from flax.core import frozen_dict
from flax.linen import partitioning as flax_partitioning
import jax
import jax.numpy as jnp
from praxis import base_layer
from praxis import pytypes


# Internal unpacking comparison."],
def maybe_unpack_summary(tree: pytypes.PyTree, unpack_summaries: bool,
                         x_times: int) -> pytypes.PyTree:
  """Unpacks the summary when `unpack_summaries` is set."""
  if not unpack_summaries:
    return tree

  def unpack(value):
    # If unpacked, callers will get a list of split summary values.
    # e.g., (4, ) -> [(1,), (1,), (1,), (1,)]
    assert value.shape[0] == x_times
    return jnp.split(value, x_times)

  return jax.tree_map(unpack, tree)


def maybe_repack_summary(tree: pytypes.PyTree, unpack_summaries: bool,
                         x_times: int) -> pytypes.PyTree:
  """Repacks the summary when `unpack_summaries` is set."""
  if not unpack_summaries:
    return tree

  def maybe_repack(value):
    if not isinstance(value, list):
      return value
    # If unpacked, callers will get a list of repacked summary values.
    # e.g., [(1,), (1,), (1,), (1,)] -> (4, )
    assert len(value) == x_times
    return jnp.stack(value)

  return jax.tree_map(maybe_repack, tree, is_leaf=lambda x: isinstance(x, list))


def convert_to_boxed_params(
    var_tree: pytypes.PyTree,
    logical_axes_rules: Optional[pytypes.LogicalAxisRules] = None,
    mesh_shape=None,
) -> pytypes.PyTree:
  """Converts raw params into BoxedParams."""
  if logical_axes_rules is not None:
    assert mesh_shape is not None

  var_tree = frozen_dict.unfreeze(var_tree)

  # Converts "_axes" collections from pass-through metadata objects to
  # jax.sharding.PartitionSpec tuples, which can be tree-mapped later; also
  # removes extraneous "_axes" names.
  axes_tree = {
      key: flax_partitioning.get_axis_names(var_tree.pop(key))
      for key in list(var_tree)
      if key.endswith('_axes')
  }

  def to_boxed(x_param, var_collection: str,
               logical_axes: Optional[jax.sharding.PartitionSpec]):
    if isinstance(x_param, base_layer.BoxedParam):
      # The param might already be boxed (if the flax module contain praxis
      # submodules). We should not box it again.
      return x_param
    if logical_axes is None:
      tensor_split_dims_mapping = None
    else:
      tensor_split_dims_mapping = list(
          flax_partitioning.logical_to_mesh_axes(
              tuple(logical_axes), logical_axes_rules))
    if var_collection == base_layer.PARAMS:
      collections = []
    else:
      # Here, we simply assume those vars are non-learnable and require mean
      # sync during training.
      collections = [
          base_layer.WeightHParamsCollection.NON_TRAINABLE,
          base_layer.WeightHParamsCollection.REQUIRES_MEAN_SYNC
      ]

    x_meta = base_layer.WeightHParams(
        x_param.shape,
        None,
        x_param.dtype,
        collections=collections,
        mesh_shape=mesh_shape,
        tensor_split_dims_mapping=tensor_split_dims_mapping)
    return base_layer.BoxedParam(x_param, x_meta)

  boxed_params = {}

  for key in var_tree:
    # pylint: disable=cell-var-from-loop
    if f'{key}_axes' in axes_tree:
      logical_axes_tree = frozen_dict.unfreeze(axes_tree[f'{key}_axes'])
      # If users mix raw Flax modules with Flaxformer modules, logical_axes_tree
      # would only have annotations for a Flaxformer layers.
      # Ensure logical_axes_tree has the same pytree structure as var_tree[key]
      # by filling missing annotations with PartitionSpec(None).
      flat_logical_axes_tree = traverse_util.flatten_dict(
          logical_axes_tree, sep='/')
      flat_full_logical_axes_tree = traverse_util.flatten_dict(
          var_tree[key], sep='/')
      flat_full_logical_axes_tree = {
          k: jax.sharding.PartitionSpec(None)
          for k in flat_full_logical_axes_tree
      }
      for k in flat_logical_axes_tree:
        flat_full_logical_axes_tree[k] = flat_logical_axes_tree[k]
      full_logical_axes_tree = traverse_util.unflatten_dict(
          flat_full_logical_axes_tree, sep='/')
      boxed_params[key] = jax.tree_map(
          lambda x, y: to_boxed(x, var_collection=key, logical_axes=y),
          var_tree[key],
          full_logical_axes_tree,
          # Consider BoxedParam as leaf to prevent boxing it again.
          is_leaf=lambda x: isinstance(x, base_layer.BoxedParam))
    else:
      boxed_params[key] = jax.tree_map(
          lambda x: to_boxed(x, var_collection=key, logical_axes=None),
          var_tree[key],
          is_leaf=lambda x: isinstance(x, base_layer.BoxedParam))
    # pylint: enable=cell-var-from-loop

  return boxed_params
