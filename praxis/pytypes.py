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

"""Common pytype definitions."""

from typing import Any, Callable, Sequence, TypeVar

from absl import logging
import clu.metrics as clu_metrics
import jax
from jax import core
from jax import numpy as jnp
from jax import tree_util as jtu
import jaxtyping
from jaxtyping import AbstractDtype, Bool, Float, Float32, Int, Int32, PyTree, Shaped  # pylint: disable=g-multiple-import, g-importing-member, unused-import
import numpy as np
from praxis import lingvo_lib
import typeguard

# No other imports from lingvo should be accessed by core JAX library.
InstantiableParams = lingvo_lib.InstantiableParams
HParams = lingvo_lib.HParams
HParamsT = HParams
NestedMap = lingvo_lib.NestedMap


def _transpose_pair_of_seqs(xys):
  """Pivots/transposes a sequence of pairs to a pair of sequences."""
  # While this function could be generalized to pivot any seq of seqs along its
  # first two axes, doing so yields some XLA-level errors in some tests, where
  # the compiler requires this function return a pair.
  if not xys:
    return ((), ())
  l, r = zip(*xys)
  return tuple(l), tuple(r)


def _flatten(xs):
  """Flattens a NestedMap into a values and keys tuple."""
  keys, values = _transpose_pair_of_seqs(sorted(xs.items()))
  return values, keys  # Note the order is flipped


def _to_keys_handler(xs):
  return tuple(jtu.DictKey(k) for k in sorted(xs))


def _flatten_with_keys(xs):
  children, treedef = _flatten(xs)
  return list(zip(_to_keys_handler(xs), children)), treedef


def _unflatten(keys, values):
  return NestedMap(zip(keys, values, strict=True))


try:
  # Register NestedMap the same way `dict` and its sibling types are registered,
  # so that when jtu..*_with_path functions yield paths with DictKey types.
  jtu.register_pytree_with_keys(
      NestedMap,
      _flatten_with_keys,
      _unflatten,
      _flatten,
  )
except ValueError:
  logging.error(
      'NestedMap is already registered as JAX PyTree node. This should not'
      ' happen - this is the canonical implementation'
  )

JTensor = jax.Array
PRNGKey = JTensor
JTensorOrPartitionSpec = JTensor | jax.sharding.PartitionSpec
NpTensor = np.ndarray
SummaryDict = NestedMap | dict[str, JTensor]
PyTreeDef = jax.tree_util.PyTreeDef


T = TypeVar('T')
Nested = T | tuple[Any, ...] | list[Any] | dict[str, Any] | NestedMap
NestedJTensor = Nested[JTensor]
NestedNpTensor = Nested[NpTensor]
NestedBool = Nested[bool]
NestedInt = Nested[int]
NestedHParams = Nested[HParamsT]
NestedPartitionSpec = Nested[jax.sharding.PartitionSpec]
NestedJTensorOrPartitionSpec = Nested[JTensorOrPartitionSpec]
NestedShapeDtypeStruct = Nested[jax.ShapeDtypeStruct]
NestedShapedArray = Nested[core.ShapedArray]
NestedShapeDtypeLike = (
    NestedJTensor | NestedNpTensor | NestedShapeDtypeStruct | NestedShapedArray
)


# Sharding annotation for a dim can be a single int, or a str, or a sequence of
# (int, str), or None. For example "1", "-1", "None", "data", "(data, replica)"
# are all valid sharding annotations for a particular tensor axis.
DimShardingAnnotation = Sequence[int | str] | int | str | None
SplitDimsMapping = Sequence[DimShardingAnnotation] | None

# Note(b/238657605): pytypes Metrics were renamed to WeightedScalars
# and Metrics are now true metric objects using clu.metrics
WeightedScalar = tuple[JTensor, JTensor]
WeightedScalars = dict[str, WeightedScalar] | NestedMap
WeightedScalarsList = dict[str, Sequence[WeightedScalar]] | NestedMap
Metrics = NestedMap | dict[str, clu_metrics.Metric]

LogicalAxisRules = Sequence[tuple[str, str | None]]

DotGeneralT = Callable[..., jnp.ndarray]


# jaxtyping utils.
class _MetaArrayT(type):
  types = ()

  def __instancecheck__(cls, obj):
    return isinstance(obj, cls.types)


class JaxArrayT(metaclass=_MetaArrayT):
  types = (jax.Array, jax.ShapeDtypeStruct)


class ArrayT(metaclass=_MetaArrayT):
  types = (JaxArrayT, np.ndarray)


AnyJaxArray = Shaped[ArrayT, '...']
AnyNPArray = Shaped[np.ndarray, '...']
AnyArray = Shaped[ArrayT, '...']
AnyPyTreeArray = jaxtyping.PyTree[AnyArray]

AnyFloatArray = Float[ArrayT, '...']
"""Float Jax array of any shape and precision."""
AnyIntArray = Int[ArrayT, '...']
"""Integer Jax array of any shape and precision."""

Scalar = Shaped[ArrayT, ''] | Shaped[np.generic, ''] | Shaped[jnp.generic, '']
"""A Jax scalar type. Note this does not include the Python `int` or `float` types."""
ScalarInt = Int[ArrayT, ''] | Int[np.generic, ''] | Int[jnp.generic, '']
"""A Jax int scalar type. Note this does not include the Python `int` type."""
ScalarFloat = Float[ArrayT, ''] | Float[np.generic, ''] | Float[jnp.generic, '']
"""A Jax float scalar type. Note this does not include the Python `float` type."""

typed = lambda fn: jaxtyping.jaxtyped(typeguard.typechecked(fn))
