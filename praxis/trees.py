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


from jax import tree_util
from praxis import pytypes

Nested = pytypes.Nested


def is_subset(superset: Nested, subset: Nested) -> bool:
  """Returns True if subset is completely contained in superset (from root).

  Note that for Sequences a and b, a is not considered a subset of b unless all
  elements match exactly - subset logic is used only for sets, dicts, and
  NestedMaps (e.g. [1, 3] is not a subset of [1, 2, 3]).

  Args:
    superset: a nested object which may contain subset
    subset: a nested object which might be contained in superset

  Returns:
    True if subset is a subset of superset (i.e. share the same root, etc), in
    that they have identical structure and values, but sets, dictionaries, and
    NestedMaps may have fields missing and sequences can have suffixes missing.
  """
  subset_paths_and_values = tree_util.tree_flatten_with_path(subset)[0]
  superset_paths_and_values = tree_util.tree_flatten_with_path(superset)[0]
  return set(subset_paths_and_values) <= set(superset_paths_and_values)
