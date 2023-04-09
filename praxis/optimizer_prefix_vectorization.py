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

"""Utilities to vectorize optimizers based on variable shape prefixes.

In WeightHParams, repeat_prefix represents stacking variables from multiple
layers of the type. Optimizers should treat such a variable as a collection of
separate per-layer variables.

get_transformations_with_vectorized_repeat_prefix creates a wrapper of existing
optimizers to vectorize them on prefix dimensions, such that using stacked
variables does not affect optimizer behavior.

If there are variables that use repeat_prefix, the optimizer state will be a
NestedMap with fields like ('#' being the repeat prefix separator):
{
  'no_prefix': state for vars without prefixes,
  'p#2.3#i0.i-1': state for vars w/ shape prefix [2,3], sharding prefix [0,-1],
  'p#4#i1': state for vars with shape prefix [4], sharding prefix [1],
   ...
}
The 'i' prefix of each sharding dim indicates it's an integer.

If the sharding prefix dims are strings are tuples of strings, the prefix keys
are encoded as:
{
  'p#2.3#sdata.smdl': sharding prefix ['data','mdl'],
  'p#2.3#tsdata,smdl.': sharding prefix [('data','mdl'), None],
  ...
}
The 's' prefix of each sharding dim indicates it's a string, and the 't' prefix
indicates it's a tuple of elements separated by ','.

Stacking variables helps reduce the number of individual variables, which can be
beneficial for compilation time and the current GDA-based checkpointing.
"""

import functools
from typing import Callable, Optional, Sequence, Tuple, Union

import jax
import optax
from praxis import base_layer
from praxis import optimizers
from praxis import py_utils
from praxis import pytypes

NestedMap = py_utils.NestedMap
NestedJTensor = pytypes.NestedJTensor
NestedHParams = pytypes.NestedHParams

SplitDimsMapping = pytypes.SplitDimsMapping
GeneralGradientTransformation = optimizers.GeneralGradientTransformation
ShardedGradientTransformation = optimizers.ShardedGradientTransformation

NO_PREFIX_KEY = 'no_prefix'
_REPEAT_PREFIX_SEP = '#'


def has_no_prefix(dct: NestedMap) -> bool:
  return dct.keys() == set([NO_PREFIX_KEY])


def _vectorize_on_prefix_dims(
    fn: Callable[..., NestedJTensor],
    num_dim: int,
    optimizer_prefix: SplitDimsMapping = None,
) -> Callable[..., NestedJTensor]:
  """Vectorize fn on multiple dimensions."""
  if num_dim == 0:
    return fn

  if optimizer_prefix is not None and len(optimizer_prefix) != num_dim:
    raise ValueError('`optimizer_prefix` must be the same length as the '
                     f'number of dimensions being vmapped across. Got '
                     f'num_dim={num_dim} vs {len(optimizer_prefix)}')

  v_fns = [fn]
  for i in range(num_dim):
    inner_fn = v_fns[-1]
    if optimizer_prefix is not None and optimizer_prefix[i] != -1:
      spmd_axis = optimizer_prefix[i]
    else:
      spmd_axis = None
    v_fns.append(jax.vmap(inner_fn, spmd_axis_name=spmd_axis))
  return v_fns[-1]


def _encode_sharding_dim(d: Optional[Union[str, Sequence[str], int]],
                         repeat_prefix_sep: str) -> str:
  """Encodes the sharding annotation into a string for one dimension."""
  if d is None:
    return ''
  if isinstance(d, int):
    return 'i%d' % d
  if isinstance(d, (list, tuple)):
    return 't' + ','.join(_encode_sharding_dim(e, repeat_prefix_sep) for e in d)

  assert isinstance(d, str)
  # The original string should not contain separators.
  assert '.' not in d
  assert ',' not in d
  assert repeat_prefix_sep not in d
  return 's' + d


def _decode_sharding_dim(d: str) -> Optional[Union[str, Sequence[str], int]]:
  """Decodes the sharding annotation from a string for one dimension."""
  if not d:
    return None
  if d.startswith('i'):
    return int(d[1:])
  if d.startswith('s'):
    return d[1:]

  assert d.startswith('t')
  if len(d) == 1:
    return ()
  tuple_elements = [_decode_sharding_dim(e) for e in d[1:].split(',')]
  return tuple(tuple_elements)  # pytype: disable=bad-return-type  # always-use-return-annotations


def _get_var_param_repeat_prefix_key(var_param: base_layer.WeightHParams,
                                     repeat_prefix_sep: str) -> str:
  """Returns string keys that uniquely identify shape and sharding prefixes."""
  if not var_param.repeat_prefix:
    return NO_PREFIX_KEY

  sharding_prefix = var_param.repeat_prefix_split_dims_mapping
  if sharding_prefix is None:
    sharding_prefix = [-1] * len(var_param.repeat_prefix)
  assert len(sharding_prefix) == len(var_param.repeat_prefix)

  optimizer_prefix = var_param.repeat_optimizer_dims_mapping
  if optimizer_prefix is not None:
    assert len(optimizer_prefix) == len(var_param.repeat_prefix)

  shape_str = '.'.join(str(d) for d in var_param.repeat_prefix)
  sharding_str = '.'.join(
      _encode_sharding_dim(d, repeat_prefix_sep) for d in sharding_prefix)

  # If optimizer_prefix is None or all `-1`'s then we omit the prefix string in
  # order to maintain backward compatibility.
  if optimizer_prefix is None or all(d == -1 for d in optimizer_prefix):
    return f'p{repeat_prefix_sep}{shape_str}{repeat_prefix_sep}{sharding_str}'

  optimizer_str = '.'.join(
      _encode_sharding_dim(d, repeat_prefix_sep) for d in optimizer_prefix)
  return (
      f'p{repeat_prefix_sep}{shape_str}{repeat_prefix_sep}{sharding_str}' +
      f'{repeat_prefix_sep}{optimizer_str}')


def _parse_var_param_repeat_prefix_key(
    prefix: str, repeat_prefix_sep: str
) -> Tuple[SplitDimsMapping, SplitDimsMapping, SplitDimsMapping]:
  """Parses shape and sharding prefixes from string keys."""
  if prefix == NO_PREFIX_KEY:
    return [], [], []

  # optimizer_prefix might have been omitted in the sharding string.
  if prefix.count(repeat_prefix_sep) == 2:
    _, shape_str, sharding_str = prefix.split(repeat_prefix_sep)
    optimizer_prefix = None
  else:
    _, shape_str, sharding_str, optimizer_str = prefix.split(repeat_prefix_sep)
    optimizer_prefix = [
        _decode_sharding_dim(d) for d in optimizer_str.split('.')]

  shape_prefix = [int(d) for d in shape_str.split('.')]
  sharding_prefix = [_decode_sharding_dim(d) for d in sharding_str.split('.')]
  return shape_prefix, sharding_prefix, optimizer_prefix


def _group_by_repeat_prefix(variables: NestedMap, var_hparams: NestedHParams,
                            repeat_prefix_sep: str) -> NestedMap:
  """Groups variables based on prefix keys."""
  var_hparams_flat, _ = jax.tree_util.tree_flatten(var_hparams)
  key_set = set()
  for p in var_hparams_flat:
    key = _get_var_param_repeat_prefix_key(p, repeat_prefix_sep)
    key_set.add(key)

  def _filter_key(key):

    def _filter_one(v, p):
      if key == _get_var_param_repeat_prefix_key(p, repeat_prefix_sep):
        return v
      return optax.MaskedNode()

    return jax.tree_map(_filter_one, variables, var_hparams)

  # Iterate `key_set` in a stable order to ensure determinism. Otherwise, the
  # produced HLO may differ across runs.
  groups = NestedMap()
  for key in sorted(key_set):
    groups[key] = _filter_key(key)
  # Make sure we still have NO_PREFIX_KEY, so that we can easily tell if it's
  # vectorized.
  if NO_PREFIX_KEY not in groups:
    groups[NO_PREFIX_KEY] = jax.tree_map(
        lambda _: optax.MaskedNode(), variables
    )

  return groups


def _ungroup_by_repeat_prefix(groups: NestedMap, var_hparams: NestedHParams,
                              repeat_prefix_sep: str) -> NestedMap:
  """Converts grouped values to the original structure of var_hparams."""

  group_list = []
  group_index = {}
  for key, group in groups.items():
    group_index[key] = len(group_list)
    group_list.append(group)

  def _get_item(p, *group_vals):
    key = _get_var_param_repeat_prefix_key(p, repeat_prefix_sep)
    return group_vals[group_index[key]]

  return jax.tree_map(_get_item, var_hparams, *group_list)


def _init_with_vectorized_repeat_prefix(
    tx: GeneralGradientTransformation, var_vals: NestedJTensor,
    var_hparams: NestedHParams, repeat_prefix_sep: str,
    force_prefix_structure: bool = False) -> optax.OptState:
  """init function for vectorized optimizers based on var_hparams."""
  vmap_groups = _group_by_repeat_prefix(var_vals, var_hparams,
                                        repeat_prefix_sep)
  results = NestedMap()
  for prefix, group in vmap_groups.items():
    shape_prefix, _, optimizer_prefix = (
        _parse_var_param_repeat_prefix_key(prefix, repeat_prefix_sep))
    results[prefix] = _vectorize_on_prefix_dims(
        tx.init, num_dim=len(shape_prefix), optimizer_prefix=optimizer_prefix)(
            group)

  if has_no_prefix(results) and not force_prefix_structure:
    # Do not change the structure if no prefix exists.
    results = results[NO_PREFIX_KEY]
  return results


def _update_with_vectorized_repeat_prefix(
    tx: GeneralGradientTransformation, updates: NestedJTensor,
    state: optax.OptState, old_vars: NestedJTensor, var_hparams: NestedHParams,
    repeat_prefix_sep: str,
    force_prefix_structure: bool = False
    ) -> Tuple[NestedJTensor, optax.OptState]:
  """update function for vectorized optimizers based on var_hparams."""
  grouped_updates = _group_by_repeat_prefix(updates, var_hparams,
                                            repeat_prefix_sep)
  grouped_old_vars = _group_by_repeat_prefix(old_vars, var_hparams,
                                             repeat_prefix_sep)
  update_results = NestedMap()
  state_results = NestedMap()
  grouped_state = state
  if has_no_prefix(grouped_updates) and not force_prefix_structure:
    # state structure did not change if no prefix exists.
    grouped_state = NestedMap()
    grouped_state[NO_PREFIX_KEY] = state

  def pure_tx_update(tx_update_fn, *args):
    with base_layer.JaxContext.new_context():
      new_updates, new_state = tx_update_fn(*args)
      summaries = base_layer.all_global_summaries()
      return new_updates, new_state, summaries

  for prefix, group in grouped_updates.items():
    shape_prefix, _, optimizer_prefix = _parse_var_param_repeat_prefix_key(
        prefix, repeat_prefix_sep)
    new_updates, new_state, bwd_summaries = _vectorize_on_prefix_dims(
        functools.partial(pure_tx_update, tx.update),
        num_dim=len(shape_prefix), optimizer_prefix=optimizer_prefix)(
            group, grouped_state[prefix], grouped_old_vars[prefix])
    # re-dispatch the summaries to the out-context
    assert isinstance(bwd_summaries, dict), repr(bwd_summaries)
    for k, v in bwd_summaries.items():
      summ_type = base_layer.get_summary_type_from_key(k)
      k_prefix = base_layer.trim_summary_type_from_key(k)
      # TODO(b/233664844): Maybe unpack summaries to individual values.
      base_layer.add_global_summary(f'{k_prefix}.{shape_prefix}', v, summ_type)

    update_results[prefix] = new_updates
    state_results[prefix] = new_state
  if has_no_prefix(state_results) and not force_prefix_structure:
    # Do not change the structure if no prefix exists.
    state_results = state_results[NO_PREFIX_KEY]
  update_results = _ungroup_by_repeat_prefix(update_results, var_hparams,
                                             repeat_prefix_sep)
  return update_results, state_results


def _init_partition_spec_with_vectorized_repeat_prefix(
    tx: ShardedGradientTransformation,
    var_hparams: NestedHParams,
    repeat_prefix_sep: str,
    force_prefix_structure: bool = False,
) -> NestedHParams:
  """init_partition_spec for vectorized optimizers based on var_hparams."""

  def call_inner_on_group(group, shape_prefix, sharding_prefix):

    def _remove_prefix(p):
      p = p.clone()
      p.repeat_prefix = None
      p.repeat_prefix_split_dims_mapping = None
      return p

    group = jax.tree_map(_remove_prefix, group)
    result = tx.init_partition_spec(group)

    def _add_prefix(p):
      p.repeat_prefix = shape_prefix
      p.repeat_prefix_split_dims_mapping = sharding_prefix
      return p

    return jax.tree_map(_add_prefix, result)

  # Use the same grouping as _init_with_vectorized_repeat_prefix, in order to
  # produce compatible tree structures.
  vmap_groups = _group_by_repeat_prefix(var_hparams, var_hparams,
                                        repeat_prefix_sep)
  results = NestedMap()
  for prefix, group in vmap_groups.items():
    shape_prefix, sharding_prefix, _ = _parse_var_param_repeat_prefix_key(
        prefix, repeat_prefix_sep)
    results[prefix] = call_inner_on_group(group, shape_prefix, sharding_prefix)
  if has_no_prefix(results) and not force_prefix_structure:
    # Do not change the structure if no prefix exists.
    results = results[NO_PREFIX_KEY]
  return results


def get_transformations_with_vectorized_repeat_prefix(
    tx: GeneralGradientTransformation,
    var_hparams: NestedHParams,
    repeat_prefix_sep: str = _REPEAT_PREFIX_SEP,
    force_prefix_structure: bool = False
) -> GeneralGradientTransformation:
  """Vectorizes a transformation on shape/sharding prefixes."""

  def _init(variables):
    return _init_with_vectorized_repeat_prefix(
        tx,
        variables,
        var_hparams,
        repeat_prefix_sep,
        force_prefix_structure=force_prefix_structure,
    )

  def _update(updates, state, params=None):
    return _update_with_vectorized_repeat_prefix(
        tx,
        updates,
        state,
        params,
        var_hparams,
        repeat_prefix_sep,
        force_prefix_structure=force_prefix_structure,
    )

  def _init_partition_spec(var_param_args):
    assert isinstance(tx, ShardedGradientTransformation)
    return _init_partition_spec_with_vectorized_repeat_prefix(
        tx,
        var_param_args,
        repeat_prefix_sep,
        force_prefix_structure=force_prefix_structure,
    )

  if isinstance(tx, ShardedGradientTransformation):
    return ShardedGradientTransformation(
        init=_init, update=_update, init_partition_spec=_init_partition_spec)
  else:
    assert isinstance(tx, optax.GradientTransformation)
    return optax.GradientTransformation(init=_init, update=_update)
