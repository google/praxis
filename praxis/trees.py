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

"""Utility functions involving trees.

Longer-term, we ought to aim to move all tree-related utility functions here.
"""
# Even longer term, we could consider defining the Nested union type in a more
# object-oriented way here, rename the module nested.py, and write a full suite
# of utility functions for it. E.g. the way `Set` has `issubset`/`<=`, we could
# have for all Nesteds in the Praxis context.

import jax
import jaxtyping as jt
from praxis import pytypes

Nested = pytypes.Nested
NestedShapeDtypeLike = pytypes.NestedShapeDtypeLike
NestedShapeDtypeStruct = pytypes.NestedShapeDtypeStruct


def is_subset(subset: Nested, superset: Nested) -> bool:
  """Returns True if subset is completely contained in superset (from root).

  Note that for Sequences a and b, a is not considered a subset of b unless all
  elements match exactly - subset logic is used only for sets, dicts, and
  NestedMaps (e.g. [1, 3] is not a subset of [1, 2, 3]).

  Args:
    subset: a nested object which might be contained in superset
    superset: a nested object which may contain subset

  Returns:
    True if subset is a subset of superset (i.e. share the same root, etc), in
    that they have identical structure and values, but sets, dictionaries, and
    NestedMaps may have fields missing and sequences can have suffixes missing.
  """
  if isinstance(subset, dict) and isinstance(superset, dict):
    if set(subset) <= set(superset):
      return all(is_subset(subset[k], superset[k]) for k in subset)
    return False

  elif type(subset) != type(superset):  # pylint:disable=unidiomatic-typecheck
    # This conditional is here insetad of above to permit nestedmaps and dicts
    # to be comperable, i.e. those can differ by type if they're both dicts.
    return False

  elif type(subset) in (tuple, list):  # pylint:disable=unidiomatic-typecheck
    if len(subset) <= len(superset):
      return all(is_subset(sub, sup) for sub, sup in zip(subset, superset))
    return False

  return subset == superset


def get_shape_dtype(nested_obj: NestedShapeDtypeLike) -> NestedShapeDtypeStruct:
  """Returns the shape/dtype information for the given nested input."""
  fn = lambda x: jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)
  return jax.tree_map(fn, nested_obj)


def copy(pytree: jt.PyTree) -> jt.PyTree:
  """Produces a by-reference copy of the original nested object.

  (Could this just be replaced by copy.deepcopy?)

  Args:
    pytree: the object to deepcopy.

  Returns:
    a copy of the original object.
  """
  return jax.tree_map(lambda x: x, pytree)
