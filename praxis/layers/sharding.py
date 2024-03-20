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

"""Sharding utilities."""

import math
import re
from typing import Sequence

import jax
from jax.interpreters import pxla
from praxis import base_layer
from praxis import py_utils

DimSharding = str | Sequence[str] | None
Sharding = Sequence[DimSharding] | None


def derive(s: Sharding, eqn: str) -> Sharding:
  """Derives a sharding based on an equation `original->derived`.

  Each letter in original and derived represents a named dimension, and the
  derivation is done by matching dimension names. E.g., with s=('x', 'y') and
  eqn="ab->cbda", the result will be (None, 'y', None, 'x').

  Args:
    s: Source sharding.
    eqn: Derivation equation with named dimensions.

  Returns:
    The derived sharding.
  """
  if s is None:
    return None
  pieces = eqn.split('->')
  assert len(pieces) == 2, eqn
  original, derived = pieces

  return tuple(s[original.index(d)] if d in original else None for d in derived)


def shard(x: jax.Array, s: Sharding, eqn: str | None = None) -> jax.Array:
  """Annotates x with a sharding based on an optional equation.

  If equation is not specified, apply just jax.lax.with_sharding_constraint.

  In equation `original->derived`, each letter in original and derived
  represents a named dimension, and the derivation is done by matching
  dimension names.

  Each dim in `derived` can also be ?, which means unconstrained.

  E.g., with s=('x', 'y') and eqn="ab->cb?a", the derived sharding will be
  (None, 'y', unconstrained, 'x').

  In `derived`, there can also be a group of consecutive dims marked optional,
  which are represented as dims inside `[]`. The tensor x can have either all of
  these dims, or none of them.

  E.g., with s=('x', 'y', 'z') and eqn="abc->[ab]c", the derived sharding will
  be ('x', 'y', 'z') for a 3D tensor x, and ('z',) of a 1D tensor x.

  Args:
    x: The tensor to annotate.
    s: Source sharding.
    eqn: Derivation equation with named dimensions.

  Returns:
    The derived sharding.
  """
  if s is None or not py_utils.global_mesh_defined():
    return x

  if eqn is not None:
    original, derived = eqn.split('->')
    if '[' in derived:
      l, optional, r = re.split(r'\[|\]', derived)

      if x.ndim == len(l) + len(r):
        derived = l + r
      elif x.ndim == len(l) + len(optional) + len(r):
        derived = l + optional + r
      else:
        raise ValueError(f'Given {derived=} is incompatible with {x=}')

    s = derive(s, f'{original}->{derived}')
    assert s is not None
    s = list(s)
    for i, p in enumerate(derived):
      if p == '?':
        s[i] = jax.sharding.PartitionSpec.UNCONSTRAINED

  partition_spec = jax.sharding.PartitionSpec(*s)

  # If mesh_axes_transpose exists in the current context, device axes will be
  # remapped according to the transpose rules.
  partition_spec = base_layer.maybe_transpose_mesh_axes(partition_spec)

  return jax.lax.with_sharding_constraint(x, partition_spec)


def get_dim_sharding(s: Sharding, dim: int) -> DimSharding:
  """Returns the sharding on one dimension."""
  if s is None:
    return None
  return s[dim]


def shard_one_dim(x: jax.Array, s: DimSharding, dim: int) -> jax.Array:
  """Annotates x on one dim while other dims are unconstrained."""
  perm = '?' * dim + 'd' + '?' * (x.ndim - dim - 1)
  return shard(x, (s,), 'd->' + perm)


def num_shards_on_dim(dim_sharding: DimSharding) -> int:
  """Returns the number of shards on one dimension in a sharding."""
  mesh = pxla.thread_resources.env.physical_mesh
  axis_sizes = dict(zip(mesh.axis_names, mesh.devices.shape))

  mapping = None
  if base_layer.JaxContext.has_context():
    mapping = base_layer.cur_jax_context().hparams.mesh_axes_transpose

  match dim_sharding:
    case None:
      return 1
    case str():
      return axis_sizes.get(
          base_layer.transpose_one_axis(dim_sharding, mapping), 1
      )
    case _:
      assert isinstance(dim_sharding, Sequence)
      return math.prod(
          axis_sizes.get(base_layer.transpose_one_axis(axis, mapping), 1)
          for axis in dim_sharding
      )
