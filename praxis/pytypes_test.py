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

"""Praxis pytype tests."""

from absl.testing import absltest
from jax import tree_util as jtu
from praxis import pytypes
from praxis import test_utils


class TreesTest(test_utils.TestCase):

  def test_nestedmap_paths(self):
    """Ensure NestedMap is registered as a pytree_node correctly."""
    tree = pytypes.NestedMap(
        a=pytypes.NestedMap(x=0, y=1, zz=2),
        b=pytypes.NestedMap(z=1),
    )
    dict_tree = {'a': {'x': 0, 'y': 1, 'zz': 2}, 'b': {'z': 1}}
    self.assertSequenceEqual(
        jtu.tree_leaves_with_path(tree),
        [
            ((jtu.DictKey(key='a'), jtu.DictKey(key='x')), 0),
            ((jtu.DictKey(key='a'), jtu.DictKey(key='y')), 1),
            ((jtu.DictKey(key='a'), jtu.DictKey(key='zz')), 2),
            ((jtu.DictKey(key='b'), jtu.DictKey(key='z')), 1),
        ],
    )
    self.assertSequenceEqual(
        jtu.tree_leaves_with_path(dict_tree), jtu.tree_leaves_with_path(tree)
    )


if __name__ == '__main__':
  absltest.main()
