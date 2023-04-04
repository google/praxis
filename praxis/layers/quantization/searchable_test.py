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

"""Tests for praxis.layers.quantization.searchable."""

from absl.testing import absltest

import jax
import jax.numpy as jnp

from praxis import base_layer
from praxis import base_hyperparams
from praxis import pax_fiddle
from praxis import test_utils
from praxis.layers.quantization import quantization_hparams
from praxis.layers.quantization import searchable

instantiate = base_hyperparams.instantiate
QuantizationType = quantization_hparams.QuantizationType


def _run_option(model, model_vars, inputs, act_decision, weight_decision):
  model_vars['non_trainable']['act_quantizer']['decision'] = act_decision
  model_vars['non_trainable']['weight_quantizer']['decision'] = weight_decision
  return model.apply(model_vars, inputs)


class SearchableTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    self.quantization_tpl = quantization_hparams.QuantizationHParams(
        quantization_type=quantization_hparams.QuantizationType.AQT,
        mode=quantization_hparams.QuantizationMode.TRAINING,
        act_params=quantization_hparams.ActQuantizationParams,
        weight_params=quantization_hparams.WeightQuantizationParams,
    )

  def _test_common(self, p, x):
    m = instantiate(p)

    with base_layer.JaxContext.new_context():
      m_vars = m.init(jax.random.PRNGKey(0), x)
      a4w4 = _run_option(m, m_vars, x, 0, 0)
      a4w8 = _run_option(m, m_vars, x, 0, 1)
      a8w4 = _run_option(m, m_vars, x, 1, 0)
      a8w8 = _run_option(m, m_vars, x, 1, 1)

    self.assertAllClose(a4w4, a4w8, rtol=0.05, atol=0.05)
    self.assertAllClose(a4w4, a8w4, rtol=0.05, atol=0.05)
    self.assertAllClose(a4w8, a8w8, rtol=0.05, atol=0.05)

  def test_searchable_linear(self):
    p = pax_fiddle.Config(
        searchable.SearchableLinear,
        quantization=self.quantization_tpl,
        input_dims=3,
        output_dims=1,
        precisions=[4, 8],
    )
    self._test_common(p, jnp.ones((1, 3), dtype=p.dtype))

  def test_attention_projections(self):
    p = pax_fiddle.Config(
        searchable.SearchableAttentionProjection,
        quantization=self.quantization_tpl,
        input_dim=8,
        num_heads=2,
        dim_per_head=3,
        is_output_projection=True,
        precisions=[4, 8],
    )
    self._test_common(p, jnp.ones((4, 2, 3), dtype=p.dtype))

  def test_combined_qkv_projections(self):
    p = pax_fiddle.Config(
        searchable.SearchableCombinedQKVProjectionLayer,
        quantization=self.quantization_tpl,
        input_dim=8,
        num_heads=3,
        dim_per_head=2,
        precisions=[4, 8],
    )
    self._test_common(p, jnp.ones((4, 8), dtype=p.dtype))


if __name__ == '__main__':
  absltest.main()
