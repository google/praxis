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

"""Praxis AutoMLSelect layer to switch branches according to AutoML decisions."""

from typing import Optional, Sequence, Union, Tuple

from flax import linen as nn
import jax.numpy as jnp
from praxis import base_layer
from praxis import pax_fiddle
from praxis import pytypes

JTensor = pytypes.JTensor
LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]
template_field = pax_fiddle.template_field
WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams


# TODO(yunn): Move BaseAutoMLSelect to efficient NAS package for Praxis
class BaseAutoMLSelect(base_layer.BaseLayer):
  """Praxis AutoMLSelect layer to switch branches according to AutoML decisions.

  Attributes:
    search_options_tpl: a sequence of layers, with each represents a branch.
      They layers must have the same shapes of input and output.
  """

  search_options_tpl: Optional[Sequence[LayerTpl]] = template_field(None)

  def setup(self) -> None:
    if not self.search_options_tpl:
      raise AttributeError('Must set at least one search option.')
    decision = WeightHParams(
        shape=[],
        init=WeightInit.Constant(len(self.search_options_tpl) - 1),
        dtype=jnp.uint8,
    )
    self.create_children('search_options', self.search_options_tpl)
    self.create_variable('decision', decision, trainable=False)


class AutoMLSelect(BaseAutoMLSelect):
  """An AutoMLSelect that switches between different quantizer."""

  def __call__(
      self,
      x: JTensor,
      contract_dims: Union[int, Sequence[int]],
      squeeze_scale=True,
      quantized_dtype: Union[jnp.dtype, None] = None,
  ) -> Tuple[JTensor, JTensor, Optional[JTensor]]:
    def branch_fn(i):
      def quantize_fn(mdl, inputs):
        return mdl.search_options[i].quantize(
            inputs, contract_dims, squeeze_scale, quantized_dtype
        )

      return quantize_fn

    branches = [branch_fn(i) for i in range(len(self.search_options))]
    return nn.switch(self.get_var('decision'), branches, self, x)

  def quantize(
      self,
      x: JTensor,
      contract_dims: Union[int, Sequence[int]],
      squeeze_scale=True,
      quantized_dtype: Union[jnp.dtype, None] = None,
  ) -> Tuple[JTensor, JTensor, Optional[JTensor]]:
    return self.__call__(x, contract_dims, squeeze_scale, quantized_dtype)
