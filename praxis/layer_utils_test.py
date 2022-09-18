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

"""Tests for layer_utils."""
import os

from absl.testing import absltest

from praxis import layer_utils
from praxis import test_utils


class LayerUtilsTest(test_utils.TestCase):

  class Foo:

    def setup(self):
      layer_utils.LayerRegistry().add_layer('foolayer', self)

  class Bar:

    def setup(self):
      layer_utils.LayerRegistry().add_layer('barlayer', self, True)

  def test_add_layer(self):
    f = self.Foo()
    f.setup()

    b = self.Bar()
    b.setup()

    for key, layer_info in layer_utils.LayerRegistry().get_registry().items():
      if 'barlayer' in key:
        self.assertTrue(layer_info.conflict)
        expected = 'Bar\t' + os.path.basename(__file__) + ':27'
        self.assertEqual(expected, layer_info.to_text())
      else:
        self.assertFalse(layer_info.conflict)
        expected = 'Foo\t'
        self.assertEqual(expected, layer_info.to_text())

    self.assertSameElements(
        list(layer_utils.LayerRegistry().get_registry().keys()),
        ['barlayer : __main__', 'foolayer'])


if __name__ == '__main__':
  absltest.main()
