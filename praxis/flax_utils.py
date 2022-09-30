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

# ==============================================================================
"""Flax-related utils."""

from typing import Optional

from flax import traverse_util
from flax.core import frozen_dict
from flax.linen import partitioning as flax_partitioning
import jax
from jax.experimental import pjit
import jax.numpy as jnp
from praxis import base_layer
from praxis import pytypes


def add_axis_to_metadata(tree, sub_weight_split_dims_mapping, x_times):
  """Adds an axis to the metadata."""
  wp_sub = sub_weight_split_dims_mapping

  def _tree_map_boxed(fn, tree):
    """Only map over Boxed leaves in pytree - identity for other leaves."""
    safe_fn = lambda x: fn(x) if isinstance(x, base_layer.BoxedParam) else x
    return jax.tree_map(
        safe_fn, tree, is_leaf=lambda x: isinstance(x, base_layer.BoxedParam))

  def update(boxed):
    if boxed.meta.repeat_prefix:
      assert isinstance(boxed.meta.repeat_prefix, list)
      repeat_prefix = [x_times] + boxed.meta.repeat_prefix
    else:
      repeat_prefix = [x_times]

    if boxed.meta.repeat_prefix_split_dims_mapping:
      assert isinstance(boxed.meta.repeat_prefix_split_dims_mapping, tuple)
      repeat_prefix_split_dims_mapping = wp_sub + tuple(
          boxed.meta.repeat_prefix_split_dims_mapping)
    else:
      repeat_prefix_split_dims_mapping = wp_sub

    boxed.meta.repeat_prefix = repeat_prefix
    boxed.meta.repeat_prefix_split_dims_mapping = (
        repeat_prefix_split_dims_mapping)
    return base_layer.BoxedParam(value=boxed.value, meta=boxed.meta)

  return _tree_map_boxed(update, tree)


def remove_axis_to_metadata(tree, sub_weight_split_dims_mapping, x_times):
  """Remove an axis to the metadata."""
  wp_sub = sub_weight_split_dims_mapping

  def _tree_map_boxed(fn, tree):
    """Only map over Boxed leaves in pytree - identity for other leaves."""
    safe_fn = lambda x: fn(x) if isinstance(x, base_layer.BoxedParam) else x
    return jax.tree_map(
        safe_fn, tree, is_leaf=lambda x: isinstance(x, base_layer.BoxedParam))

  def update(boxed):

    if boxed.meta.repeat_prefix:
      assert isinstance(boxed.meta.repeat_prefix, list)
      removed_axis = boxed.meta.repeat_prefix.pop(0)
      assert removed_axis == x_times

    if boxed.meta.repeat_prefix_split_dims_mapping:
      assert isinstance(boxed.meta.repeat_prefix_split_dims_mapping, tuple)
      updated_dims_mapping = list(boxed.meta.repeat_prefix_split_dims_mapping)
      removed = updated_dims_mapping.pop(0)
      assert (removed,) == tuple(wp_sub)
      boxed.meta.repeat_prefix_split_dims_mapping = updated_dims_mapping

    return base_layer.BoxedParam(value=boxed.value, meta=boxed.meta)

  return _tree_map_boxed(update, tree)


# Internal unpacking comparison."],
def maybe_unpack_summary(tree: pytypes.PyTreeDef, unpack_summaries: bool,
                         x_times: int) -> pytypes.PyTreeDef:
  """Unpacks the summary when `unpack_summaries` is set."""
  if not unpack_summaries:
    return tree

  def unpack(value):
    # If unpacked, callers will get a list of split summary values.
    # e.g., (4, ) -> [(1,), (1,), (1,), (1,)]
    assert value.shape[0] == x_times
    return jnp.split(value, x_times)

  return jax.tree_map(unpack, tree)


def maybe_repack_summary(tree: pytypes.PyTreeDef, unpack_summaries: bool,
                         x_times: int) -> pytypes.PyTreeDef:
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
    var_tree: pytypes.PyTreeDef,
    logical_axes_rules: Optional[pytypes.LogicalAxisRules] = None,
    mesh_shape=None,
) -> pytypes.PyTreeDef:
  """Converts raw params into BoxedParams."""
  if logical_axes_rules is not None:
    assert mesh_shape is not None

  var_tree = frozen_dict.unfreeze(var_tree)

  # Converts "_axes" collections from pass-through metadata objects to
  # pjit.PartitionSpec tuples, which can be tree-mapped later; also removes
  # extraneous "_axes" names.
  axes_tree = {
      key: flax_partitioning.get_axis_names(var_tree.pop(key))
      for key in list(var_tree)
      if key.endswith('_axes')
  }

  def to_boxed(x_param, var_collection: str,
               logical_axes: Optional[pjit.PartitionSpec]):
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
          k: pjit.PartitionSpec(None) for k in flat_full_logical_axes_tree
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
