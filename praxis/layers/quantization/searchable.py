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

"""Layers for Mixed-Precision Quantization Search."""
import dataclasses
from typing import Any, Dict, Sequence

from praxis import base_layer
from praxis import pax_fiddle
from praxis.layers.quantization import attentions
from praxis.layers.quantization import automl_select
from praxis.layers.quantization import linears
from praxis.layers.quantization import quantization_hparams
from praxis.layers.quantization import quantizer

QuantizationHParams = quantization_hparams.QuantizationHParams
template_field = pax_fiddle.template_field

_SUPPORTED_PRECISIONS = [2, 4, 8]


def _build_quantization_templates(
    precisions: Sequence[int], base_qhparams: QuantizationHParams
):
  """Build templates for precision search from the base quantization hparams."""
  act_qtzr_tmpls = []
  weight_qtzr_tmpls = []

  if not precisions:
    raise AttributeError('Must specify at least 1 precisions.')

  for precision in precisions:  # pylint: disable=not-an-iterable
    if precision not in _SUPPORTED_PRECISIONS:
      raise AttributeError('Precision %d is not supported.' % precision)
    act_qtzr_tmpl = quantizer.create_tensor_quantizer(
        f'act_quantizer_int{str(precision)}', base_qhparams.act_params
    )
    # if not quantizing activations, then don't set precision to indicate.
    if base_qhparams.act_params.precision:
      act_qtzr_tmpl.precision = precision
    act_qtzr_tmpls.append(act_qtzr_tmpl)
    weight_qtzr_tmpl = quantizer.create_tensor_quantizer(
        f'weight_quantizer_int{str(precision)}', base_qhparams.weight_params
    )
    weight_qtzr_tmpl.precision = precision
    weight_qtzr_tmpls.append(weight_qtzr_tmpl)
  return act_qtzr_tmpls, weight_qtzr_tmpls


class SearchableQuantizedLayer(base_layer.BaseLayer):
  """A template to make the precision of quantizers searchable."""
  precisions: Sequence[int] = dataclasses.field(default_factory=list)

  def create_tensor_quantizers(self):
    act_qtzr_tmpls, weight_qtzr_tmpls = _build_quantization_templates(
        self.precisions, self.quantization
    )
    # Use AutoMLSelect as quantizer
    self.create_child(
        'act_quantizer',
        pax_fiddle.Config(
            automl_select.AutoMLSelect,
            search_options_tpl=act_qtzr_tmpls,
        ),
    )
    self.create_child(
        'weight_quantizer',
        pax_fiddle.Config(
            automl_select.AutoMLSelect,
            search_options_tpl=weight_qtzr_tmpls,
        ),
    )

  # This is for fixing the return type mismatch during multiple inheritance.
  def replace(self, **kwargs: Dict[Any, Any]):
    return self.replace(**kwargs)


class SearchableLinear(SearchableQuantizedLayer, linears.Linear):
  """Quantized Linear layer with searchable precision."""

  pass


class SearchableAttentionProjection(
    SearchableQuantizedLayer, attentions.AttentionProjection
):
  """Layer that computes multi-head projection with searchable precision."""

  pass


class SearchableCombinedQKVProjectionLayer(
    SearchableQuantizedLayer, attentions.CombinedQKVProjectionLayer
):
  """Layer that computes QKV projection with searchable precision."""
  pass
