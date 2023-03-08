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

"""A sequential stack of layers for Pax."""
from typing import Any, Callable, Dict, Optional, Sequence

from praxis import base_layer


class Sequential(base_layer.BaseLayer):
  """Applies a linear chain of Modules."""
  layers: Optional[Sequence[Callable[..., Any]]] = None

  def __call__(self, *args, **kwargs):
    if not self.layers:
      raise ValueError(f'Empty Sequential module {self.name}.')

    outputs = self.layers[0](*args, **kwargs)
    for layer in self.layers[1:]:
      if isinstance(outputs, tuple):
        outputs = layer(*outputs)
      elif isinstance(outputs, Dict):
        outputs = layer(**outputs)
      else:
        outputs = layer(outputs)
    return outputs

