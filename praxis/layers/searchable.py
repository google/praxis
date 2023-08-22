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

"""Layers for Platform-aware AutoML Search."""

from typing import Any, Sequence

from flax import linen as nn
import jax.numpy as jnp
from praxis import base_layer
from praxis import pax_fiddle
from praxis import pytypes

JTensor = pytypes.JTensor
LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]
NestedJTensor = pytypes.NestedJTensor
template_field = pax_fiddle.template_field
WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams

template_field = pax_fiddle.template_field


class AutoMLSelect(base_layer.BaseLayer):
  """AutoMLSelect layer to switch branches according to AutoML decisions.

  Attributes:
    search_options_tpl: a sequence of layers, with each represents a branch.
      They layers must have the same shapes of input and output.
  """

  search_options_tpl: Sequence[LayerTpl] | None = template_field(None)

  def setup(self) -> None:
    if not self.search_options_tpl:
      raise AttributeError('Must set at least one search option.')
    decision = WeightHParams(
        shape=[],
        init=WeightInit.Constant(0),
        dtype=jnp.uint8,
        mesh_shape=self.mesh_shape
    )
    self.create_children('search_options', self.search_options_tpl)
    self.create_variable('decision', decision, trainable=False)

  def __call__(
      self,
      x: JTensor,
      *args: Any,
      **kwargs: Any,
  ) -> NestedJTensor:
    def branch_fn(i):
      return lambda mdl, x: mdl.search_options[i](x)

    branches = [branch_fn(i) for i in range(len(self.search_options))]

    if self.is_mutable_collection('params'):
      for branch in branches:
        _ = branch(self, x)

    return nn.switch(self.get_var('decision'), branches, self, x)
