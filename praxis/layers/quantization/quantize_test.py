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

"""Tests for quantization methods."""

from absl.testing import absltest
from absl.testing import parameterized
from praxis import layers
from praxis import pax_fiddle
from praxis import test_utils
from praxis.layers import embedding_softmax
from praxis.layers import quantization as qlayer
from praxis.layers.quantization import quantization_hparams
from praxis.layers.quantization import quantize


class QuantizationTest(test_utils.TestCase):

  @parameterized.named_parameters(
      ('combine_qkv', True),
      ('separate_qkv', False),
  )
  def test_update_transformer(self, use_combine_qkv):
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
        quantization_hparams.QuantizationMode.TRAINING,
        quantization_hparams.WeightQuantizationParams(precision=8),
    )
    self.assertEqual(
        p.tr_fflayer_tpl.fflayer_tpl.linear_tpl.cls, qlayer.linears.Linear
    )
    self.assertEqual(
        p.tr_atten_tpl.proj_tpl.cls, qlayer.attentions.AttentionProjection
    )
    self.assertEqual(
        p.tr_atten_tpl.combined_qkv_proj_tpl.cls,
        qlayer.attentions.CombinedQKVProjectionLayer,
    )

  @parameterized.named_parameters(
      ('4bits', 4),
      ('8bits', 8),
  )
  def test_number_of_bits(self, num_bits):
    p = pax_fiddle.Config(
        layers.transformers.Transformer,
        name='jax_transformer_layer',
        input_dims=12,
        hidden_dims=4,
        num_heads=8,
    )
    quantize.quantize_transformer_layer_weights(
        p,
        quantization_hparams.QuantizationType.PTQ,
        quantization_hparams.QuantizationMode.TRAINING,
        quantization_hparams.WeightQuantizationParams(precision=num_bits),
    )
    self.assertEqual(
        p.tr_fflayer_tpl.fflayer_tpl.linear_tpl.quantization.weight_params.precision,
        num_bits,
    )
    self.assertEqual(
        p.tr_atten_tpl.proj_tpl.quantization.weight_params.precision, num_bits
    )

  def test_update_transformer_mq(self):
    p = pax_fiddle.Config(
        layers.transformers.Transformer,
        name='jax_transformer_layer',
        input_dims=12,
        hidden_dims=4,
        num_heads=8,
        tr_atten_tpl=pax_fiddle.Config(
            layers.multi_query_attention.MultiQueryDotProductAttention
        ),
    )
    quantize.quantize_transformer_layer_weights(
        p,
        quantization_hparams.QuantizationType.PTQ,
        quantization_hparams.QuantizationMode.TRAINING,
        quantization_hparams.WeightQuantizationParams(precision=8),
    )
    self.assertEqual(
        p.tr_fflayer_tpl.fflayer_tpl.linear_tpl.cls, qlayer.linears.Linear
    )
    self.assertEqual(
        p.tr_atten_tpl.proj_tpl.cls, qlayer.attentions.AttentionProjection
    )
    self.assertEqual(
        p.tr_atten_tpl.headless_proj_tpl.cls,
        qlayer.multi_query_attention.OneHeadedAttentionProjection,
    )

  @parameterized.named_parameters(
      ('combine_qkv', True),
      ('separate_qkv', False),
  )
  def test_update_transformer_linear_only(self, use_combine_qkv):
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
        quantization_hparams.QuantizationMode.TRAINING,
        quantization_hparams.WeightQuantizationParams(precision=8),
        linear_only=True,
    )

    # Expect only linear to be quantized.
    self.assertEqual(
        p.tr_fflayer_tpl.fflayer_tpl.linear_tpl.cls, qlayer.linears.Linear
    )
    self.assertEqual(
        p.tr_atten_tpl.proj_tpl.cls, layers.attentions.AttentionProjection
    )
    if use_combine_qkv:
      self.assertEqual(
          p.tr_atten_tpl.combined_qkv_proj_tpl.cls,
          layers.attentions.CombinedQKVProjectionLayer,
      )

  @parameterized.named_parameters(
      ('embedding_transposed', True),
      ('embedding_not_transposed', False),
  )
  def test_set_quantization_quantize_embedding(
      self, transposed_embedding_softmax
  ):
    class DummyNClassMajorSharedEmbeddingSoftmax(
        embedding_softmax.SharedEmbeddingSoftmax
    ):
      """Dummy class for transposed_embedding_softmax=True case."""

      pass

    lm_p = pax_fiddle.Config(
        layers.TransformerLm, packed_input=True, model_dims=16, vocab_size=8
    )
    tr_p = lm_p.stacked_transformer_tpl.transformer_layer_params_tpl
    tr_p.tr_atten_tpl.combine_qkv = True
    if transposed_embedding_softmax:
      embedding_softmax_p = lm_p.softmax_tpl
      new_embedding_softmax_p = pax_fiddle.Config(
          DummyNClassMajorSharedEmbeddingSoftmax
      )
      new_embedding_softmax_p.copy_fields_from(embedding_softmax_p)
      lm_p.softmax_tpl = new_embedding_softmax_p

    quantize.set_quantization(
        lm_p,
        quantize_embedding_softmax=True,
        transposed_embedding_softmax=transposed_embedding_softmax,
    )

    # Expect linears and attentions are all quantized.
    self.assertEqual(
        tr_p.tr_fflayer_tpl.fflayer_tpl.linear_tpl.cls, qlayer.linears.Linear
    )
    self.assertEqual(
        tr_p.tr_atten_tpl.proj_tpl.cls, qlayer.attentions.AttentionProjection
    )
    self.assertEqual(
        tr_p.tr_atten_tpl.combined_qkv_proj_tpl.cls,
        qlayer.attentions.CombinedQKVProjectionLayer,
    )

    if transposed_embedding_softmax:
      self.assertEqual(
          lm_p.softmax_tpl.cls, qlayer.NClassMajorSharedEmbeddingSoftmax
      )
    else:
      self.assertEqual(lm_p.softmax_tpl.cls, qlayer.SharedEmbeddingSoftmax)


if __name__ == '__main__':
  absltest.main()
