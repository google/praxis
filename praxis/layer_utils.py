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

"""Provide a global list of child/children/variables created in a pax system.

To keep track of name collisions for migrating to Fiddle configuration system, a
global list of layer names etc. is being maintained.
"""

import dataclasses
import inspect
import re
from typing import Any

from absl import logging

pax_layer_registry = {}


@dataclasses.dataclass
class LayerInfo:
  """Layer info and conflict indication.

  Attributes:
    layer: PAX layer.
    conflict: Boolean to indicate an hparam conflict.
    file_info: File/linenumber information for where layer exists.
  """
  layer: Any
  conflict: bool = False
  file_info: str = ''

  def to_text(self, filename_normalization_regex: str = r'^.*/(.*)$') -> str:
    file = re.sub(filename_normalization_regex, r'\1', self.file_info)
    items = [self.layer.__class__.__name__, file]
    return '\t'.join(items)


class LayerRegistry:
  """A dict holding information about layer creation."""

  def add_layer(self, name: str, layer: Any, conflict: bool = False) -> None:
    """Adds layer information for name to registry.

    Args:
      name: name of the layer.
      layer: the layer being created.
      conflict: the layer name has a conflict with an HParam attribute.
    """
    key = name
    file_info = ''
    if conflict:
      key += ' : ' + inspect.getmodule(layer).__name__
      location = inspect.getframeinfo(inspect.stack()[2][0])
      file_info = f'{location.filename}:{location.lineno}'
      hierarchy = []
      for base in self.__class__.mro():
        hierarchy.append(str(base))
      logging.warning('FiddleMigration: layer info:\t%s\t%s', file_info,
                      '\t'.join(hierarchy))
    layer_info = LayerInfo(layer=layer, conflict=conflict, file_info=file_info)
    pax_layer_registry[key] = layer_info

  def get_registry(self):
    return pax_layer_registry
