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
import re
from typing import Callable, Iterable, Union
import jax
from jax import tree_util as jtu
import jaxtyping as jt
from praxis import pytypes

KeyType = jtu.DictKey | jtu.SequenceKey | jtu.GetAttrKey | jtu.FlattenedIndexKey
Patterns = Union[str, re.Pattern, Iterable[Union[re.Pattern, str]]]


def _tree_key_to_str(path: KeyType) -> str:
  """Returns a string representation of a tree key used by to_path_tree."""
  match path:
    case jtu.DictKey(key=key):
      return f'{key}'
    case jtu.SequenceKey(idx=idx):
      return f'[{idx}]'
    case jtu.GetAttrKey(name=name):
      return name
    case jtu.FlattenedIndexKey(key=key):
      return f'{key}'
    case _:
      raise ValueError(f'Unexpected key type: {path=} {type(path)=}')


def to_path_tree(
    tree: jt.PyTree, is_leaf: None | Callable[..., bool] = None
) -> jt.PyTree:
  """For a given tree node, returns the pax-style prefix.

  This meant to be a simpler implementation of
  `extract_prefixed_keys_from_nested_map`.

  Example:
      {'a': [1, 2, Point(x=3, y=4), (5, 6, Masked())],
      'b': ('c', 'd'),
      'e': Masked()}

  Becomes
      {'a': ['a[0]', 'a[1]', Point(x='a[2]/x', y='a[2]/y'), ('a[3][0]',
                                                             'a[3][1]', None)],
      'b': ('b[0]', 'b[1]'),
      'e': None}

  When is_leaf returns true, that path is replaced by None.

  Args:
    tree: the pytree whose prefixes you wish to extract
    is_leaf: an optional Callable returning a boolean. When it is true, the
      prefix is replaced by None.

  Returns:
    A pytree of the same shape as tree, but having just string values
  """

  def process_node(path):
    return re.sub(r'/\[', '[', '/'.join(map(_tree_key_to_str, path)))

  return jtu.tree_map_with_path(
      lambda p, v: None if (is_leaf and is_leaf(v)) else process_node(p), tree
  )


def fullmatch_path(
    tree: jt.PyTree,
    patterns: Patterns,
    is_leaf: Callable[..., bool] | None = None,
) -> jt.PyTree[bool]:
  """Produces a PyTree[bool] indicating which nodes have paths that match.

  This function uses to_path_tree to generate string paths for each node,
  which are then matched against the given regex patterns, yielding a
  PyTree[bool] indicating where matches succeeded.

  Args:
    tree: The structure to process.
    patterns: a string, re.Pattern, or sequence thereof.
    is_leaf: Optional function which can limit the behavior of the traversal.

  Returns:
    A PyTree[bool] indicating whether each node's stringified path matches at
    least one of the given regexes.
  """
  patterns = [patterns] if isinstance(patterns, (str, re.Pattern)) else patterns
  compiled_patterns = [re.compile(pattern) for pattern in patterns]

  return jtu.tree_map(
      lambda x: any(p.fullmatch(x) for p in compiled_patterns),
      to_path_tree(tree, is_leaf=is_leaf),
      is_leaf=is_leaf,
  )


def is_subset(subset: pytypes.Nested, superset: pytypes.Nested) -> bool:
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
    # This conditional is here instead of above to permit nestedmaps and dicts
    # to be comperable, i.e. those can differ by type if they're both dicts.
    return False

  elif type(subset) in (tuple, list):  # pylint:disable=unidiomatic-typecheck
    if len(subset) <= len(superset):
      return all(is_subset(sub, sup) for sub, sup in zip(subset, superset))
    return False

  return subset == superset


def extract_elements_matching_subset_structure(
    subset: pytypes.Nested, superset: pytypes.Nested
) -> pytypes.Nested:
  """Returns elements from superset matching structure of subset.

  Note that for Sequences a and b, a is not considered a subset of b unless all
  elements match exactly - subset logic is used only for sets, dicts, and
  NestedMaps (e.g. [1, 3] is not a subset of [1, 2, 3]).

  Args:
    subset: a nested object which might be contained in superset
    superset: a nested object which may contain subset

  Returns:
    Elements from superset which match the structure of subset.

  Raises:
    ValueError if subset is not a subset of superset.
  """

  if isinstance(subset, dict) and isinstance(superset, dict):
    if not set(subset) <= set(superset):
      raise ValueError(
          'Expected extract_elements_matching_subset_structure to be called '
          'with subset and superset arguments; found elements '
          f'{subset} of subset which are not contained in '
          f'{superset}.'
      )
    return type(subset)(**{
        k: extract_elements_matching_subset_structure(subset[k], superset[k])
        for k in subset
    })

  elif type(subset) != type(superset):  # pylint:disable=unidiomatic-typecheck
    # If we encounter leaves of different types, no problem; but we should
    # ensure that these are indeed leaves. We only need to check the subset for
    # being a leaf, since if the subset is not a leaf, the supserset should not
    # be either and the failure below is appropriate.

    def _is_strict_leaf(x):
      treedef = jax.tree_util.tree_structure(x)
      return treedef.num_nodes == 1 and treedef.num_leaves == 1

    if _is_strict_leaf(subset):
      return superset

    raise ValueError(
        'Expected extract_elements_matching_subset_structure to be called '
        'with subset and superset arguments of matching type; '
        f'found subset type {type(subset)} with superset type '
        f'{type(superset)}.'
    )

  elif type(subset) in (tuple, list):  # pylint:disable=unidiomatic-typecheck
    if len(subset) > len(superset):
      raise ValueError(
          'Expected extract_elements_matching_subset_structure to be called '
          'with subset and superset arguments; found elements '
          f'{subset} of subset which are not contained in '
          f'{superset}.'
      )
    return type(subset)(
        extract_elements_matching_subset_structure(sub, sup)
        for sub, sup in zip(subset, superset)
    )

  return superset


def get_shape_dtype(
    nested_obj: pytypes.NestedShapeDtypeLike,
) -> pytypes.NestedShapeDtypeStruct:
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
