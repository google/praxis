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

"""Python utility functions for JAX which contains minimal TF lingvo deps."""

import contextlib
import dataclasses
import functools
import inspect
import re
import threading
import time
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, NamedTuple, Sequence

from absl import flags
from absl import logging
import flax
import jax
from jax import lax
from jax.experimental import mesh_utils
from jax.experimental import multihost_utils
from jax.interpreters import pxla
import jax.numpy as jnp
import numpy as np
import optax
from praxis import lingvo_lib
from praxis import pytypes
from praxis import trees

flags.DEFINE_bool(
    'pmap_use_tensorstore', False,
    'Temporary flag to allow pmap users to fall back to flax checkpointing.')

# SeqIOInput enumeration provenance keys
PROVENANCE_PREFIX = '_seqio_provenance'
INDEX_WITHIN_SHARD_KEY = f'{PROVENANCE_PREFIX}/index_within_shard'
SHARD_INDEX_KEY = f'{PROVENANCE_PREFIX}/shard_index'
NUM_SHARDS_KEY = f'{PROVENANCE_PREFIX}/num_shards'
ENUM_PROVENANCE_KEYS = (INDEX_WITHIN_SHARD_KEY, SHARD_INDEX_KEY, NUM_SHARDS_KEY)
_CPU = 'cpu'


def pmap_use_tensorstore():
  return flags.FLAGS.pmap_use_tensorstore


# No other imports from lingvo should be accessed by core JAX library.
current_cluster = lingvo_lib.current_cluster
infeed_context_scope = lingvo_lib.infeed_context_scope
InstantiableParams = pytypes.InstantiableParams
NestedMap = pytypes.NestedMap
HParams = pytypes.HParams
Nested = pytypes.Nested

JTensor = jax.Array


def merge_dict(dict1, dict2):
  """Merges two dictionaries and asserts keys in both have identical values."""
  for key in set(dict1) & set(dict2):
    # The values must be the same object
    if dict1[key] is not dict2[key]:
      raise ValueError(
          f'The same key {key} corresponds to different values '
          f'in the dictionaries: {dict1[key]} vs {dict2[key]}.'
      )
  return {**dict1, **dict2}


class ThreadLocalStack(threading.local):
  """Stack of thread-local data."""

  def __init__(self):
    """Constructor."""
    super().__init__()
    self.stack = []


def sharded_file_pattern_to_glob(file_pattern: str) -> str:
  """Converts a file pattern path@shards in to path-?????-of-shards."""
  if ',' in file_pattern:
    raise ValueError(
        'sharded_file_pattern_to_glob does not support multiple file patterns.'
    )
  if '@' not in file_pattern:
    return file_pattern
  path, shards = file_pattern.split('@')
  if shards == '*':
    return f'{path}-?????-of-*'
  return f'{path}-?????-of-{int(shards):05}'


def _nested_map_to_state_dict(xs: NestedMap) -> dict[str, Any]:
  return flax.serialization.to_state_dict(dict(xs))


def _nested_map_from_state_dict(
    xs: NestedMap, states: dict[str, Any]
) -> NestedMap:
  return NestedMap(flax.serialization.from_state_dict(dict(xs), states))

try:
  flax.serialization.register_serialization_state(
      NestedMap, _nested_map_to_state_dict, _nested_map_from_state_dict
  )
except ValueError:
  logging.error(
      'ValueError: a serialization handler for "NestedMap" is already'
      ' registered'
  )


@functools.partial(functools.partial, jax.tree.map)
def assert_same_shape_and_dtype(x, y):
  assert x.shape == y.shape and x.dtype == y.dtype, f'x={x}, y={y}'


def reshard(array: jnp.ndarray) -> np.ndarray:
  """Reshards an input tensor according to the number of local devices."""
  num_devices = jax.local_device_count()
  batch_size = array.shape[0]
  return np.reshape(array,
                    (num_devices, batch_size // num_devices) + array.shape[1:])


def unshard(array: jnp.ndarray) -> np.ndarray:
  """Undo the resharding to reshape away the local device count leading dim."""
  return np.reshape(array, (-1,) + array.shape[2:])


def reshape_with_outer_batch_size(array: jnp.ndarray,
                                  outer_bs: int) -> np.ndarray:
  """Reshapes an input tensor according to an outer batch size."""
  batch_size = array.shape[0]
  if batch_size // outer_bs < 1 or batch_size % outer_bs != 0:
    raise ValueError('outer_bs should be a factor of batch_size.')
  return np.reshape(array, (outer_bs, batch_size // outer_bs) + array.shape[1:])


def combine_inner_and_outer_batches(array: jnp.ndarray) -> np.ndarray:
  """Combines the first two dimensions of the array."""
  return np.reshape(array, (-1,) + array.shape[2:])


def _unreplicate(x):
  """Helper to unreplicated the data based on its type."""
  if isinstance(x, jax.Array):
    y = x.addressable_data(0)
    # if jax_pmap_no_rank_reduction is set, we need to perform rank reduction
    # manually assuming that we're sharded along the first axis.
    if (
        not x.sharding.is_fully_replicated
        and len(y.shape) == len(x.shape)
        and y.shape[0] == 1
    ):
      return np.array(y)[0]
    return y
  else:
    return x


def maybe_unreplicate_for_fully_replicated(data):
  """Fully replicated data.

  Data may contain multiple shards, but here we assume data is fully replicated,
  and we unreplicate 'data' by taking just the first shard. In
  the following cases, 'data' are fully replicated:
  1. All metrics are fully replicated. In pmap training, we explicitly
     synchronize metrics across different data replicas, and as a result
     metrics are fully replicated (identical across different replicas).
     In spmd training, there is one single model replica. metric output are
     specifically marked as replicated.
  2. Similarly most summaries are replicated.
  3. In pmap training, model weights are fully replicated.

  Args:
    data: An array containing data.

  Returns:
    First shard of data.
  """
  return jax.tree.map(_unreplicate, data)


def maybe_unreplicate_for_first_shard(data):
  """Unreplicate data for first shard.

  'data' may not be fully replicated.

  `data` may contain multiple shards (as device buffers in multiple devices),
  but here we simply return the first shard.

  Args:
    data: An array containing data.

  Returns:
    First shard of data.
  """
  return jax.tree.map(_unreplicate, data)


def extract_keys(n, p, key_separator, left_separator, right_separator, is_leaf):
  """Alias long function call with fixed separators."""
  return extract_prefixed_keys_from_nested_map(
      n,
      p,
      key_separator=key_separator,
      left_separator=left_separator,
      right_separator=right_separator,
      is_leaf=is_leaf)


def _handle_dict(
    node,
    prefix,
    key_separator,
    left_separator,
    right_separator,
    node_type=None,
    is_leaf=None,
):
  """Handles dictionaries."""
  result = {}
  for key, value in node.items():
    if prefix:
      path = f'{prefix}{key_separator}{key}'
    else:
      path = key
    result[key] = extract_keys(
        value,
        path,
        key_separator,
        left_separator,
        right_separator,
        is_leaf=is_leaf)
  if node_type is not None:
    return node_type(**result)
  else:
    return type(node)(result)


def extract_prefixed_keys_from_nested_map(
    node: Any,
    prefix: str = '',
    key_separator: str = '/',
    left_separator: str = '[',
    right_separator: str = ']',
    is_leaf: Callable[[Any], bool] | None = None,
) -> Any:
  """Extracts a NestedMap with the nested prefix keys from its NestedMap node.
  """
  if is_leaf is not None and is_leaf(node):
    return None
  elif isinstance(node, dict):  # NestedMap inherits from dict.
    return _handle_dict(
        node,
        prefix,
        key_separator,
        left_separator,
        right_separator,
        is_leaf=is_leaf)
  # PartitionSpec is subclass of tuple.
  elif isinstance(node, jax.sharding.PartitionSpec):
    return prefix
  elif isinstance(node, (list, tuple)):
    # Check if it is a NamedTuple.
    if hasattr(node, '_fields'):
      if prefix:
        prefix += f'{key_separator}'
      out = {}
      for field in node._fields:
        out[field] = extract_keys(
            getattr(node, field),
            f'{prefix}{field}',
            key_separator,
            left_separator,
            right_separator,
            is_leaf=is_leaf)
      return type(node)(**out)
    # Convert back to list or tuple.
    out = []
    for i, v in enumerate(node):
      out.append(
          extract_keys(
              v,
              f'{prefix}{left_separator}{i}{right_separator}',
              key_separator,
              left_separator,
              right_separator,
              is_leaf=is_leaf))
    return type(node)(out)
  elif (dataclasses.is_dataclass(node) and
        node.__class__ in flax.serialization._STATE_DICT_REGISTRY):  # pylint: disable=protected-access
    if hasattr(node, '__dict__'):
      node_dict = node.__dict__
    else:
      node_dict = flax.serialization.to_state_dict(node)
    return _handle_dict(
        node_dict,
        prefix,
        key_separator,
        left_separator,
        right_separator,
        node_type=type(node),
        is_leaf=is_leaf,
    )
  if not prefix:
    return None
  return prefix


def is_mock_tpu_backend() -> bool:
  """Checks if a mock TPU backend is detected.

  Returns:
    True if Mock TPU backend detected.
  """
  # Internal mock TPU checking implementation
  return False


def sync_global_devices(name: str) -> None:
  """Sync across all hosts/devices."""
  if is_mock_tpu_backend():
    return

  global_device_count = jax.device_count()
  logging.info('Starting sync_global_devices %s across %s devices globally',
               name, global_device_count)
  multihost_utils.sync_global_devices(name)
  logging.info('Finished sync_global_devices %s across %s devices globally',
               name, global_device_count)


def put_to_devices(
    host_array: np.ndarray, local_devices: Sequence[Any]
) -> list[Any]:
  """Transfers a host array to the local devices."""
  local_device_count = len(local_devices)
  try:
    per_device_arrays = np.split(host_array, local_device_count, axis=0)
  except ValueError as array_split_error:
    raise ValueError(
        f'Unable to put to devices shape {host_array.shape} with '
        f'local device count {local_device_count}') from array_split_error
  device_buffers = jax.device_put(per_device_arrays, local_devices)
  return device_buffers


# We use Any types to allow nested data structures. They are defined in pytypes
# which would cause a circular dependency.
# TODO(pax-dev): Rename globally into e.g. create_jax_array()
def make_array(
    host_arrays: np.ndarray | Any,
    global_shapes: jax.ShapeDtypeStruct | Any,
    global_mesh: jax.sharding.Mesh,
    pspecs: Any | None = None,
    sharding: Mapping[str, jax.sharding.Sharding] | None = None,
) -> Any:
  """Makes a Jax Array from host array.

  Evenly partitioning x along axis 0 and device_put shards to local devices.

  Args:
    host_arrays: host-local arrays.
    global_shapes: global shapes of the resultant Array.
    global_mesh: global mesh of the resultant Array.
    pspecs: Optional partition specs of the resultant Array.
    sharding: Optional sharding of the resultant Array. Either pspecs or
      sharding must be provided.

  Returns:
    A Jax Array with x as the host-local data.
  """
  assert (pspecs is not None) != (
      sharding is not None
  ), 'Either pspecs or sharding must be provided.'
  local_devices = global_mesh.local_devices

  def _put_to_devices(x):
    return put_to_devices(x, local_devices)

  device_buffers = jax.tree.map(_put_to_devices, host_arrays)

  def _jax_array(global_shape, dbs, sharding):
    return jax.make_array_from_single_device_arrays(
        global_shape.shape, sharding, dbs
    )

  # If sharding not provided, create it from the partition spec.
  sharding = sharding or jax.tree.map(
      lambda x: jax.sharding.NamedSharding(global_mesh, x), pspecs
  )
  return jax.tree.map(_jax_array, global_shapes, device_buffers, sharding)


def convert_fully_replicated_array_to_pmap_array(arr):
  """Converts a fully replicated Array to Array with PmapSharding.

  Args:
    arr: Fully replicated jax.Array.

  Returns:
    Fully replicated jax.Array with PmapSharding. This is suitable as an
    input to pmap.
  """
  assert isinstance(arr, jax.Array)
  with jax.transfer_guard('disallow'):
    local_shape = (jax.local_device_count(),) + arr.shape
    device_buffers = [shard.data for shard in arr.addressable_shards]
    devices = np.array([shard.device for shard in arr.addressable_shards])

    s = jax.sharding.PmapSharding.default(
        local_shape, sharded_dim=0, devices=devices
    )
    return jax.make_array_from_single_device_arrays(local_shape, s,
                                                    device_buffers)


def convert_host_local_array_to_global_array(arr):
  """Converts a host local array from pmap to global jax.Array.

  Args:
    arr: Input host local array produced by pmap.

  Returns:
    A global array similar to GDA.
  """
  # input `arr` is fully replicated, so it's shape is the global shape.
  global_shape = arr.addressable_data(0).shape
  # Create a 1D mesh to create fully replicated global jax.Array.
  mesh = jax.sharding.Mesh(np.array(jax.devices()), axis_names=('x',))
  partition_spec = (
      jax.sharding.PartitionSpec(None)
      if global_shape
      else jax.sharding.PartitionSpec()
  )
  # pmap-produced Array has a "scrambled" device order.
  dbs = sorted(
      [shard.data for shard in arr.addressable_shards],
      key=lambda x: list(x.devices())[0].id,
  )
  return jax.make_array_from_single_device_arrays(
      global_shape, jax.sharding.NamedSharding(mesh, partition_spec), dbs
  )


def get_global_input_shape_dtype(x: jnp.ndarray) -> jax.ShapeDtypeStruct:
  """Get global input shape/dtype assuming fully sharded batch dim."""
  assert len(x.shape) >= 1
  # Assume fully sharded batch dim.
  x_shape = (x.shape[0] * jax.process_count(),) + tuple(x.shape[1:])
  return jax.ShapeDtypeStruct(x_shape, x.dtype)


def set_globally_use_rbg_prng_key() -> None:
  """Must call this before any JAX computation to set RBG PRNGKey globally."""
  jax.config.update('jax_default_prng_impl', 'rbg')


def total_num_vars(variables) -> int:
  """Returns the total number of variables of the given variable collections."""
  param_shape_counts = jax.tree.map(lambda x: np.prod(x.shape), variables)
  flattened_counts, _ = jax.tree_util.tree_flatten(param_shape_counts)
  return np.sum(flattened_counts)


def global_mesh_defined() -> bool:
  """Checks if global xmap/pjit mesh resource environment is defined."""
  maps_env = pxla.thread_resources.env
  return maps_env.physical_mesh.devices.shape != ()  # pylint: disable=g-explicit-bool-comparison


# This wrapped with_sharding_constraint will not throw error for eval_shape
# outside pjit. It is also used in p5x.
def with_sharding_constraint(
    x: JTensor, axis_resources: jax.sharding.PartitionSpec | None
) -> JTensor:
  """Wrapper for lax.with_sharding_constraint, no-op outside pjit."""
  if not global_mesh_defined():
    return x
  else:
    return lax.with_sharding_constraint(x, axis_resources)


def get_uneven_sharding_paddings(
    partition_spec: jax.sharding.PartitionSpec, shape: Sequence[int],
    mesh_shape: Sequence[int], mesh_axis_names: Sequence[str]) -> Sequence[int]:
  """Returns the padding size on each dimension due to uneven sharding."""
  axes_sizes = {}
  for size, name in zip(mesh_shape, mesh_axis_names):
    axes_sizes[name] = size
  paddings = []
  for axes, dim_size in zip(partition_spec, shape):
    if isinstance(axes, str):
      axes = [axes]
    partitions = int(np.prod([axes_sizes[axis] for axis in (axes or ())]))
    padding = (partitions - dim_size % partitions) % partitions
    paddings.append(padding)
  return paddings


def is_optax_masked_node(x: Any) -> bool:
  """Check whether the input is an instance of optax MaskedNode."""
  return isinstance(x, optax.MaskedNode)


def maybe_pad_uneven_sharding(
    xs: pytypes.NestedJTensor,
    partition_specs: jax.sharding.PartitionSpec | flax.struct.PyTreeNode,
    unpadded_shapes: Sequence[int],
    mesh_shape: Sequence[int],
    mesh_axis_names: Sequence[str],
) -> pytypes.NestedJTensor:
  """Pads xs to make them evenly shardable, if needed."""

  def _maybe_pad(x, pspec, shape):
    if is_optax_masked_node(x):
      return x
    paddings = get_uneven_sharding_paddings(pspec, shape, mesh_shape,
                                            mesh_axis_names)
    if all(p == 0 for p in paddings):
      return x
    # Annotate before pad to make sure they have the same sharding.
    # (Pad does not have the highest sharding propagation priority.)
    x = with_sharding_constraint(x, pspec)
    return jnp.pad(x, [[0, p] for p in paddings])

  return jax.tree.map(
      _maybe_pad,
      xs,
      partition_specs,
      unpadded_shapes,
      is_leaf=is_optax_masked_node,
  )


def maybe_slice_uneven_sharding(
    xs: pytypes.NestedJTensor,
    partition_spec: jax.sharding.PartitionSpec,
    unpadded_shapes: Sequence[int],
    is_leaf: Any = None,
) -> pytypes.NestedJTensor:
  """Slices xs to remove padding due to uneven sharding, if needed."""

  def _maybe_slice(x, pspec, shape):
    if is_optax_masked_node(x):
      return x
    if list(shape) == list(x.shape):
      return x
    if x.shape == (0,):
      return x
    x = jax.lax.slice(x, [0] * x.ndim, shape)
    # Annotate after slice to make sure they have the same sharding. (Slice does
    # not have the highest sharding propagation priority.)
    return with_sharding_constraint(x, pspec)

  return jax.tree.map(
      _maybe_slice, xs, partition_spec, unpadded_shapes, is_leaf=is_leaf
  )


@contextlib.contextmanager
def logging_verbosity_level(level: str):
  prev_level = logging.get_verbosity()
  try:
    logging.set_verbosity(level)
    yield
  finally:
    logging.set_verbosity(prev_level)


def select_nodes_by_indices(indices, *trees):
  """Selects PyTree nodes from multiple trees and constructs new tree.

  Args:
    indices: PyTree with the same structure as other `trees`. The leaf nodes are
      indices to select nodes from `trees`
    *trees: PyTree with the same structure as `indices`.

  Returns:
    PyTree with the same structure with the arguments. For example, if tree
    nodes are accessible as `tree[key]`, each node in the return value is
    defined as `ret[key] = trees[indices[key]][key]`.
  """
  return jax.tree.map(lambda idx, *arrays: arrays[idx], indices, *trees)


Patterns = str | re.Pattern | Iterable[re.Pattern | str]


def match_variable_names(
    tree: NestedMap,
    patterns: Patterns,
    is_leaf: Callable[..., bool] | None = None,
) -> NestedMap:
  """Checks if a prefix key of each variable is matching to one of `patterns`.

  Args:
    tree: NestedMap to be matched against `patterns`.
    patterns: `re.Pattern`, `str` that can be compiled into `re.Pattern`, or an
      iterator of those.
    is_leaf: an optional Callable returning a boolean. When it is true, the
      prefix is replaced by None.

  Returns:
    A nested map with the same structure as `tree`. Each node of the tree is
    a boolean flag denoting whether the prefix name of the variable is matching
    to one of `patterns`.
  """
  return trees.fullmatch_path(tree, patterns, is_leaf=is_leaf)


def update_matched_variables(old_tree: NestedMap,
                             new_tree: NestedMap,
                             patterns: Patterns,
                             invert: bool = False) -> NestedMap:
  """Partially updates `old_tree` by `new_tree`.

  depending on patterns.

  This function tests whether variable names are matching to the given
  regexp patterns, and if so, replace the variable with the corresponding
  variable in `new_tree`.

  Args:
    old_tree: A nested map to be updated.
    new_tree: A nested map with the same structure as `old_tree` containing the
      updated values.
    patterns: Regular expression patterns (`str`, `re.Patterns`, or an iterator
      of those) that are used to determine whether the variable should be
      updated.
    invert: If True, condition on the variable names is inverted. i.e. only the
      variables that are not matching to `patterns` will be updated.

  Returns:
    An updated NestedMap
  """
  mask = match_variable_names(old_tree, patterns)  # True for update
  if invert:
    mask = jax.tree.map(lambda x: not x, mask)
  flat_var_prefix = jax.tree.flatten(
      extract_prefixed_keys_from_nested_map(old_tree))[0]
  flat_mask = jax.tree.flatten(mask)[0]
  assert len(flat_var_prefix) == len(flat_mask)
  for prefix, match in zip(flat_var_prefix, flat_mask):
    if match:
      logging.info('Bprop included var: %s', prefix)
  for prefix, match in zip(flat_var_prefix, flat_mask):
    if not match:
      logging.info('Bprop excluded var: %s', prefix)

  indices = jax.tree.map(lambda x: 1 if x else 0, mask)
  return select_nodes_by_indices(indices, old_tree, new_tree)


def l2_normalize(
    x: JTensor, axis: int | Sequence[int] = -1, epsilon: float = 1e-12
) -> JTensor:
  """L2-normalize a Jax tensor along certain dimension."""
  norm = jnp.sqrt(jnp.sum(x * x, axis=axis, keepdims=True) + epsilon)
  return x / norm


def create_device_mesh(
    ici_mesh_shape: Sequence[int],
    dcn_mesh_shape: Sequence[int] | None = None,
    contiguous_submeshes: bool | None = False,
):
  """Creates a single- or multi-slice device mesh from mesh shapes.

  Args:
    ici_mesh_shape: The mesh shape for a single slice, or for each slice in a
      multi-slice setting.
    dcn_mesh_shape: The mesh shape to use for between-slice parallelism. If
      None, creates a single-slice mesh.
    contiguous_submeshes: If True, the mesh_utils.create_device_mesh() call will
      attempt to create a mesh where each process's local devices form a
      contiguous submesh. This is unused when `dcn_mesh_shape` is not None.

  Returns:
    An ndarray of JAX devices.
  """
  contiguous_submeshes = bool(contiguous_submeshes)
  if dcn_mesh_shape is not None and any(s > 1 for s in dcn_mesh_shape):
    devices = jax.devices()
    device_kind = devices[-1].device_kind
    if device_kind == _CPU:
      target_shape = np.array(ici_mesh_shape) * np.array(dcn_mesh_shape)
      device_mesh = np.array(devices).reshape(target_shape)
    else:
      try:
        device_mesh = mesh_utils.create_hybrid_device_mesh(
            ici_mesh_shape, dcn_mesh_shape, devices=devices)
      except AssertionError as e:
        raise ValueError(
            'Setting a nontrivial dcn_mesh_shape requires multiple slices. '
            f'[{ici_mesh_shape=}, {dcn_mesh_shape=}, {devices=}]'
        ) from e
  else:
    device_mesh = mesh_utils.create_device_mesh(
        ici_mesh_shape, contiguous_submeshes=contiguous_submeshes
    )
  logging.info('device_mesh: %s', device_mesh)
  return device_mesh


def get_large_negative_number(dtype: jnp.dtype | np.dtype) -> JTensor:
  """Returns a large negative value for the given dtype."""
  # -0.7 is a float64 in Jax. Explicit cast output to target dtype.
  if jnp.issubdtype(dtype, jnp.inexact):
    dtype_max = jnp.finfo(dtype).max
  elif jnp.issubdtype(dtype, jnp.integer):
    dtype_max = jnp.iinfo(dtype).max
  else:
    raise ValueError('Unsupported dtype for inputs.')
  return jnp.asarray(-0.7 * dtype_max, dtype=dtype)


def apply_mask_to_logits(logits: JTensor, mask: JTensor) -> JTensor:
  """Applies a floating-point mask to a set of logits.

  The mask is represented as a float32 tensor where 0 represents true and values
  below a large negative number (here set to
  get_large_negative_number(jnp.float32) / 2) represent false. Applying the mask
  leaves the logits alone in the true case and replaces them by
  get_large_negative_number(jnp.float32) in the false case. Previously, this was
  done by adding the logits to the mask; however, this leads to a bad fusion
  decision in the compiler that saves the float32 values in memory rather than
  just the predicate. This implementation avoids that problem.

  Args:
    logits: A JTensor of logit values.
    mask: A JTensor (float32) of mask values with the encoding described in the
      function documentation.

  Returns:
    Masked logits.
  """

  min_value = get_large_negative_number(logits.dtype)
  return jnp.where((mask >= min_value * 0.5), logits, min_value)


def sequence_mask(
    lengths: JTensor | Sequence[int], maxlen: int, dtype=jnp.bool_
) -> JTensor:
  """Creates a sequence mask where 1s are valid positions and 0s are padded.

  Args:
    lengths: A JTensor or Python list of integers.
    maxlen: A Python int.
    dtype: Output data type.

  Returns:
    [..., maxlen] of 0/1 JTensor where 1s are valid positions.
  """
  lengths = jnp.array(lengths)
  return (jnp.arange(maxlen)[jnp.newaxis, ...] <
          lengths[..., jnp.newaxis]).astype(dtype)


def sequence_paddings(
    lengths: JTensor | Sequence[int], maxlen: int, dtype=jnp.float32
) -> JTensor:
  """Creates sequence paddings based on the lengths.

  Args:
    lengths: A JTensor or Python list of integers.
    maxlen: A Python int.
    dtype: Output data type.

  Returns:
    A 0/1 JTensor of shape [..., maxlen], in which 1 indicates paddings.
  """
  lengths = jnp.array(lengths)
  return (jnp.arange(maxlen)[jnp.newaxis, ...] >=
          lengths[..., jnp.newaxis]).astype(dtype)


@jax.vmap
def flip_sequence(inputs: JTensor, lengths: JTensor):
  max_length = inputs.shape[0]
  return jnp.flip(jnp.roll(inputs, max_length - lengths, axis=0), axis=0)


def concat_sequences_with_padding(
    input0: JTensor, paddings0: JTensor, input1: JTensor, paddings1: JTensor
) -> tuple[JTensor, JTensor]:
  """Concatenates input sequences with varying lengths as defined by paddings.

  This is a helper function for concatenating 2 batches of input sequences,
  where each example in the batch can have different lengths, as defined by
  the corresponding paddings.

  NOTE: We assume that the tensors have no leading paddings.

  Args:
    input0: A tensor of size [batch, max_length0, ...]
    paddings0: A Tensor of size [batch, max_length1]
    input1:  A tensor of size [batch, max_length0, ...]
    paddings1: A Tensor of size [batch, max_length1]

  Returns:
    The concatenation of input0 and input1, and the corresponding padding.
  """
  assert (
      input0.shape[0] == input1.shape[0] and input0.shape[2] == input1.shape[2]
  ), (
      f'dim0 and dim2 should match. input0 shape: {input0.shape}, '
      f'input1 shape: {input1.shape}'
  )
  assert input0.shape[:2] == paddings0.shape[:2], (
      f'dim0 and dim1 should match. input0 shape: {input0.shape}, '
      f'paddings0 shape: {paddings0.shape}'
  )
  assert input1.shape[:2] == paddings1.shape[:2], (
      f'dim0 and dim1 should match. input1 shape: {input1.shape}, '
      f'paddings1 shape: {paddings1.shape}'
  )

  batch_size = input0.shape[0]

  seq_length0 = (1 - paddings0).sum(-1).astype(jnp.int32)
  seq_length1 = (1 - paddings1).sum(-1).astype(jnp.int32)

  # Concatenate input sequences.
  input0_seq_dim = (
      jnp.ones([
          batch_size,
      ])
      * paddings0.shape[1]
  )
  input1_seq_dim = (
      jnp.ones([
          batch_size,
      ])
      * paddings1.shape[1]
  )
  reversed_input0 = flip_sequence(input0, seq_length0)
  reversed_input1 = flip_sequence(input1, input1_seq_dim)
  reversed_concat = jnp.concatenate([reversed_input1, reversed_input0], axis=1)
  concat_inputs = flip_sequence(reversed_concat, seq_length0 + input1_seq_dim)
  # Concatenate paddings. Note that paddings are always a Tensor of 0s and 1s,
  # so, unlike the inputs, we don't have to reverse padding1, we can simply
  # concatenate reversed padding0 and padding1.
  reversed_padding0 = flip_sequence(paddings0, input0_seq_dim)
  reversed_concat_padding = jnp.concatenate(
      [reversed_padding0, paddings1], axis=1
  )
  concat_paddings = flip_sequence(
      reversed_concat_padding, input0_seq_dim + seq_length1
  )
  return concat_inputs, concat_paddings


def tree_unstack(tree: Any, axis: int) -> Sequence[Any]:
  """Extracts an axis' dimension to the list dimension of the output.

  Args:
    tree: PyTree which must have the above axis dimension with same size for all
      leaf nodes. All leafs must be one of (np.ndarray, jnp.ndarray) types.
    axis: int, the axis to extract into the list dimension. All leafs in the
      pytree must have this dimension and must have the same shape.

  Returns:
    A list of PyTrees with the `axis` dimension extracted. I.e., if
      tree_leaf.shape[axis] == N, then len(returned_list) == N.
  """
  leaves = jax.tree_util.tree_leaves(tree)
  if not leaves:
    return []

  if not all(isinstance(leaf, (jnp.ndarray, np.ndarray)) for leaf in leaves):
    raise ValueError('leaves must be either a pure numpy or jax ndarray')

  axis_size = leaves[0].shape[axis]
  if not all(
      leaf.ndim > axis and leaf.shape[axis] == axis_size for leaf in leaves):
    raise ValueError(f'all leaves must have x.ndim > {axis}'
                     f' and x.shape[{axis}] == {axis_size}')

  flat_pytrees = []
  for i in range(axis_size):
    flat_pytrees.append(jax.tree.map(lambda x: x.take(i, axis), tree))  # pylint: disable=cell-var-from-loop"

  return flat_pytrees


def apply_padding(
    inputs: JTensor,
    padding: JTensor,
    pad_value: JTensor | None = None,
    use_select: bool = True,
    axis: int | None = None,
) -> JTensor:
  """Applies padding to a tensor.

  `inputs` and `padding` should be broadcast compatible.

  `axis` defines the leading dimensions along which to broadcast. Specifically,
  `padding` is reshaped from [head|tail] -> [head|new_tail] such that the new
  tail has the same rank as inputs' tail.

  Args:
    inputs: JTensor to apply padding to.
    padding: JTensor of padding values where 0 == keep and 1 == pad.
    pad_value: Values to include for padded elements. Defaults to zeros. Must
      have a shape broadcastable to 'x' if specified.
    use_select: Controls whether padding is applied with a select-mask
      (True/default) or arithmetically (False). Some platforms have a
      sensitivity to one or the other and this is used to work around such
      issues.
   axis: Optional axis from where broadcasting starts.

  Returns:
    A tensor with the same shape as x with padded values masked.
  """
  if axis is not None:
    head, tail = list(padding.shape[:axis]), list(padding.shape[axis:])
    in_tail_len = len(inputs.shape[axis:])
    ones = [1] * max(in_tail_len - len(tail), 0)
    padding = jnp.reshape(padding, head + tail[:in_tail_len] + ones)
  if use_select:
    if pad_value is None:
      pad_value = jnp.zeros([], inputs.dtype)
    if padding.dtype != jnp.bool_:
      padding = padding > jnp.zeros([], padding.dtype)
    result = jnp.where(padding, pad_value, inputs)
  else:
    result = inputs * (1.0 - padding.astype(jnp.float32)).astype(inputs.dtype)
    if pad_value is not None:
      result += pad_value * padding.astype(pad_value.dtype)
  return result


def is_tpu() -> bool:
  """Whether the process runs on TPU."""
  return jax.local_devices()[0].platform == 'tpu'


@dataclasses.dataclass
class RunningPeriod:
  """Information about a running period."""

  start: float
  end: float | None = None
  min_elapsed: float = 0

  @property
  def elapsed(self) -> float:
    """Returns the elapsed time in second."""
    right_boundary = time.time() if self.end is None else self.end
    return max(right_boundary - self.start, self.min_elapsed)


@contextlib.contextmanager
def timeit(min_elapsed: float = 1e-6) -> Iterator[RunningPeriod]:
  """A context manager that times an interval of execution.

  Usage:
    with py_utils.timeit() as period:
      run_logic()
    print(period.elapsed)

  Args:
    min_elapsed: the smallest time interval (seconds) period.elapsed will yield.

  Yields:
    A `RunningPeriod` object that contains time information for execution
      under the context.
  """
  period = RunningPeriod(start=time.time(), min_elapsed=min_elapsed)
  try:
    yield period
  finally:
    period.end = time.time()


def _contract_path(s: str):
  """Contract a slash-deliminted path to just the last two parts."""
  parts = s.split('/')
  if len(parts) <= 2:
    return s
  return f'.../{"/".join(s.split("/")[-2:])}'


def benchmark(prefix: str = '', first_n: int | None = None):
  """Log walltime elapsed around the decorated function.

  Args:
    prefix: Optional. If provided, prefixes the standard message with the
      supplied string.
    first_n: Optional. If provided and positive, only meausre and print timings
      for the first n invocations of the wrapped function.

  Returns:
    The decorator.
  """

  def decorator(func: Callable[..., Any]):
    call_count = 0

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      nonlocal call_count
      call_count += 1

      if first_n is not None and call_count > first_n:
        return func(*args, **kwargs)

      # Get stack frame information about the wrapped function for reporting.
      child_frame = inspect.currentframe()
      frame = child_frame.f_back if child_frame else None
      filename = _contract_path(frame.f_code.co_filename) if frame else ''
      lineno = frame.f_lineno if frame else -1

      logging.info(
          '%sStarting timer for <%s> @ <%s:%d>',
          prefix,
          func.__name__,
          filename,
          lineno,
      )
      start = time.time()
      result = func(*args, **kwargs)
      end = time.time()
      logging.info(
          '%sElapsed time for <%s>: %.02f seconds  (@ <%s:%d>)',
          prefix,
          func.__name__,
          end - start,
          filename,
          lineno,
      )
      return result

    return wrapper

  return decorator


def filter_by_matching_keys(
    batch: dict[str, Nested], prefixes: Sequence[str]
) -> tuple[dict[str, Nested], dict[str, Nested]]:
  """Filter a map into one that matches any prefix and one that doesn't."""

  def _matching_fn(k: str) -> bool:
    return any(k.startswith(prefix) for prefix in prefixes)

  matching = NestedMap()
  non_matching = NestedMap()

  for k in batch:
    if _matching_fn(k):
      matching[k] = batch[k]
    else:
      non_matching[k] = batch[k]

  return matching, non_matching


def get_enumeration_id(
    example: dict[str, Any], pop: bool = False
) -> str | None:
  """Build enumeration ID string from example map's enumeration fields.

  Args:
    example: a mapping between field names and an item, which is typically an
      array.
    pop: whether or not to modify the input 'example' in-place and pop
      enumeration related fields.

  Returns:
    a string representing the enumeration ID which should be globally unique
      within a given dataset. If enum fields DNE in example, returns None.
  """
  if not all(
      k in example
      for k in (INDEX_WITHIN_SHARD_KEY, SHARD_INDEX_KEY, NUM_SHARDS_KEY)):
    return

  if pop:
    get_fn = lambda ex, key: int(ex.pop(key))
  else:
    get_fn = lambda ex, key: int(ex[key])

  return (
      f'{INDEX_WITHIN_SHARD_KEY}={get_fn(example, INDEX_WITHIN_SHARD_KEY)}/'
      f'{SHARD_INDEX_KEY}={get_fn(example, SHARD_INDEX_KEY)}/'
      f'{NUM_SHARDS_KEY}={get_fn(example, NUM_SHARDS_KEY)}'
  )


def pad_or_trim_to(
    x: JTensor | None, shape: Sequence[int], pad_val=0
) -> JTensor | None:
  """Pad and slice x to the given shape.

  Args:
    x: A tensor.
    shape: The shape of the returned tensor.
    pad_val: An int or float used to pad x.

  Returns:
    'x' is padded with pad_val and sliced so that the result has the given
    shape.
  """
  if x is None:
    return x

  expected_rank = len(shape)
  assert len(x.shape) == expected_rank, (x.shape, expected_rank)
  padings = [
      (0, pad_shape - min(orig_shape, pad_shape))
      for orig_shape, pad_shape in zip(x.shape, shape)
  ]
  x = jnp.pad(x, padings, constant_values=pad_val)
  x = jax.lax.slice(x, [0] * expected_rank, shape)
  return jnp.reshape(x, shape)


def append_eos(
    x: JTensor, paddings: JTensor, eos_id: int, extend_if_overflow: bool = True
) -> tuple[JTensor, JTensor]:
  """Ensure each sequence ends with eos by padding.

  Args:
    x: [b, t], input sequence
    paddings: [b, t], input paddings
    eos_id: id of eos for padding
    extend_if_overflow: if True, extend the output time dimension to t+1 in
      case of overflow, i.e., when input sequence does not end with eos and
      has no padded position. If false, the output will have shape [b, t]
      and eos is not appended in case of overflow.

  Returns:
    Updated x and paddings. Output paddings include appended eos as valid
    (non-padded). Padded positions of output sequences are filled with eos_id.
  """
  b, t = x.shape
  input_lens = (1 - paddings).astype(jnp.int32).sum(-1, keepdims=False)
  reversed_x = flip_sequence(x, input_lens)
  appended_reversed_x = jnp.concatenate(
      [jnp.ones([b, 1], x.dtype) * eos_id, reversed_x], axis=1
  )
  # shape [b, t+1]
  output_x = flip_sequence(appended_reversed_x, input_lens + 1)

  append_mask = jnp.logical_or(input_lens == 0, reversed_x[:, 0] != eos_id)
  output_lens = input_lens + append_mask.astype(jnp.int32)

  # shape [b, t+1]
  output_paddings = sequence_paddings(output_lens, t + 1, x.dtype)
  output_x = output_x * (1 - output_paddings) + eos_id * output_paddings

  # If no overflow or if extend_if_overflow=False, truncate time dimension to t.
  truncate = jnp.logical_or(
      jnp.all(output_lens <= t), not extend_if_overflow,
  ).astype(jnp.int32)
  return (
      output_x[:, : t + 1 - truncate],
      output_paddings.astype(paddings.dtype)[:, : t + 1 - truncate],
  )


class BpropMaskedNode(NamedTuple):
  """A node used to mask out vars excluded from bprop.

  We use this class help FLAX checkpointer to detect these nodes to be backward-
  compatible with legacy checkpoints where such nodes have placeholder tensors,
  while newer checkpoints don't have them.
  """


def is_bprop_masked_node(x: Any) -> bool:
  """Returns if x is an instance of BpropMaskedNode."""
  return isinstance(x, BpropMaskedNode)


def concat_nested_maps(map_list: list[NestedMap], axis: int) -> NestedMap:
  """Recursively concat tensors in a list of NestMaps.

  If a key exists in map_list[0] and in map_list[1] then it assumed to exist in
  all map_list[:]. Keys in map_list[0] that aren't JTensors, Maps, don't
  exist in map_list[1], or have a ndims <= axis will be copied directly.

  Args:
    map_list: A list of NestedMaps.
    axis: The axis to concat tensors along.

  Returns:
    A single NestedMap with all JTensors in all maps concatenated along axis.
  """
  if len(map_list) < 2:
    return map_list[0]
  results = NestedMap()
  for k in map_list[0].keys():
    results[k] = map_list[0][k]
    if (
        isinstance(map_list[0][k], JTensor)
        and k in map_list[1]
        and len(map_list[0][k].shape) > axis
    ):
      tensor_list = [output[k] for output in map_list]
      results[k] = jnp.concatenate(tensor_list, axis=axis)
    elif isinstance(map_list[0][k], Dict) and k in map_list[1]:
      next_map_list = [output[k] for output in map_list]
      results[k] = concat_nested_maps(next_map_list, axis=axis)
  return results
