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

"""Tests for quantization methods."""

from absl.testing import absltest
from absl.testing import parameterized
from praxis import layers
from praxis import pax_fiddle
from praxis import test_utils
from praxis.layers import quantization as qlayer
from praxis.layers.quantization import quantization_hparams
from praxis.layers.quantization import quantize


class QuantizationTest(test_utils.TestCase):

  @parameterized.named_parameters(
      ('combine_qkv', True, qlayer.attentions.CombinedQKVProjectionLayer),
      ('separate_qkv', False, layers.attentions.CombinedQKVProjectionLayer),
  )
  def test_update_transformer(self, use_combine_qkv, qkv_cls):
    p = pax_fiddle.Config(
        layers.transformers.Transformer,
        name='jax_transformer_layer',
        input_dims=12,
        hidden_dims=4,
        num_heads=8,
    )
    p.tr_atten_tpl.combine_qkv = use_combine_qkv
    quantize.quantize_transformer_layer_weights(
        p,
        quantization_hparams.QuantizationType.PTQ,
        quantization_hparams.QuantizationMode.MATERIALIZE,
    )
    self.assertEqual(p.tr_fflayer_tpl.fflayer_tpl.linear_tpl.cls,
                     qlayer.linears.Linear)
    self.assertEqual(p.tr_atten_tpl.proj_tpl.cls,
                     qlayer.attentions.AttentionProjection)
    self.assertEqual(p.tr_atten_tpl.combined_qkv_proj_tpl.cls, qkv_cls)


if __name__ == '__main__':
  absltest.main()
