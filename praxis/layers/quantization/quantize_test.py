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
import itertools

from absl.testing import absltest
from absl.testing import parameterized
from praxis import base_layer
from praxis import layers
from praxis import pax_fiddle
from praxis import test_utils
from praxis.layers import embedding_softmax
from praxis.layers import quantization as qlayer
from praxis.layers.quantization import quantization_hparams
from praxis.layers.quantization import quantize


class DummyNClassMajorSharedEmbeddingSoftmax(
    embedding_softmax.SharedEmbeddingSoftmax
):
  """Dummy class for testing transposed_embedding_softmax=True case."""


def expected_attention_proj_cls(is_quantized):
  if is_quantized:
    return qlayer.attentions.AttentionProjection
  else:
    return layers.attentions.AttentionProjection


def expected_combined_qkv_proj_cls(is_quantized):
  if is_quantized:
    return qlayer.attentions.CombinedQKVProjectionLayer
  else:
    return layers.attentions.CombinedQKVProjectionLayer


def expected_embedding_softmax_cls(is_quantized, is_transposed):
  if is_quantized:
    if is_transposed:
      return qlayer.NClassMajorSharedEmbeddingSoftmax
    else:
      return qlayer.SharedEmbeddingSoftmax
  else:
    if is_transposed:
      return DummyNClassMajorSharedEmbeddingSoftmax
    else:
      return layers.SharedEmbeddingSoftmax


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

  @parameterized.parameters(
      itertools.product(
          (True, False),  # quantize_self_attention
          (True, False),  # quantize_cross_attnetion
      )
  )
  def test_update_transformer_with_cross_attention(
      self, quantize_self_attention, quantize_cross_attention
  ):
    p = pax_fiddle.Config(
        layers.transformers.Transformer,
        name='jax_transformer_layer',
        input_dims=12,
        hidden_dims=4,
        num_heads=8,
        use_cross_attention=True,
    )
    quantize.quantize_transformer_layer_weights(
        p,
        quantization_hparams.QuantizationType.PTQ,
        quantization_hparams.QuantizationMode.TRAINING,
        quantization_hparams.WeightQuantizationParams(precision=8),
        quantize_self_attention=quantize_self_attention,
        quantize_cross_attention=quantize_cross_attention,
    )
    self.assertEqual(
        p.tr_fflayer_tpl.fflayer_tpl.linear_tpl.cls, qlayer.linears.Linear
    )
    self.assertEqual(
        p.tr_atten_tpl.proj_tpl.cls,
        expected_attention_proj_cls(quantize_self_attention),
    )
    self.assertEqual(
        p.tr_atten_tpl.combined_qkv_proj_tpl.cls,
        expected_combined_qkv_proj_cls(quantize_self_attention),
    )
    if quantize_cross_attention != quantize_self_attention:
      self.assertEqual(
          p.cross_atten_tpl.proj_tpl.cls,
          expected_attention_proj_cls(quantize_cross_attention),
      )
      self.assertEqual(
          p.cross_atten_tpl.combined_qkv_proj_tpl.cls,
          expected_combined_qkv_proj_cls(quantize_cross_attention),
      )
    else:
      # Cross attention share the same config as self attention.
      self.assertIsNone(p.cross_atten_tpl)

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

  @parameterized.named_parameters(
      ('64int4', 64, 4),
      ('128int8', 128, 8),
  )
  def test_low_rank_factorize(self, rank, num_bits):
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
        rank=rank,
    )
    self.assertEqual(
        p.tr_fflayer_tpl.fflayer_tpl.linear_tpl.quantization.weight_params.precision,
        num_bits,
    )
    self.assertEqual(
        p.tr_atten_tpl.proj_tpl.quantization.weight_params.precision, num_bits
    )
    self.assertEqual(p.tr_fflayer_tpl.fflayer_tpl.linear_tpl.rank, rank)

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

  @parameterized.parameters(
      itertools.product(
          (True, False),  # transposed_embedding_softmax
          (True, False),  # separate_embedding
          (True, False),  # softmax_only
      )
  )
  def test_set_quantization_quantize_embedding_transformer_lm(
      self, transposed_embedding_softmax, separate_embedding, softmax_only
  ):
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

    if separate_embedding:
      lm_p.separate_embedding_tpl = lm_p.softmax_tpl.clone()

    quantize.set_transformer_quantization(
        lm_p,
        quantize_embedding_softmax=True,
        transposed_embedding_softmax=transposed_embedding_softmax,
        softmax_only=softmax_only,
    )

    # Expect linears and attentions are all quantized.
    self.assertEqual(
        tr_p.tr_fflayer_tpl.fflayer_tpl.linear_tpl.cls, qlayer.linears.Linear
    )
    self.assertEqual(
        tr_p.tr_atten_tpl.proj_tpl.cls,
        expected_attention_proj_cls(is_quantized=True),
    )
    self.assertEqual(
        tr_p.tr_atten_tpl.combined_qkv_proj_tpl.cls,
        expected_combined_qkv_proj_cls(is_quantized=True),
    )

    # softmax is always quantized.
    self.assertEqual(
        lm_p.softmax_tpl.cls,
        expected_embedding_softmax_cls(
            is_quantized=True, is_transposed=transposed_embedding_softmax
        ),
    )

    if separate_embedding:
      if softmax_only:
        self.assertEqual(
            lm_p.separate_embedding_tpl.cls,
            expected_embedding_softmax_cls(
                is_quantized=False, is_transposed=transposed_embedding_softmax
            ),
        )
      else:
        self.assertEqual(
            lm_p.separate_embedding_tpl.cls,
            expected_embedding_softmax_cls(
                is_quantized=True, is_transposed=transposed_embedding_softmax
            ),
        )
    else:
      # Both softmax and embedding are quantized regardless of softmax_only
      # setting since they are the same weights.
      self.assertIsNone(lm_p.separate_embedding_tpl)

  @parameterized.named_parameters(
      ('ngrammer', False),
      ('vq_ngrammer', True),
  )
  def test_set_quantization_ngramer_transformer_lm(self, use_vq_ngrammer):
    lm_p = pax_fiddle.Config(
        layers.TransformerLm, packed_input=True, model_dims=16, vocab_size=8
    )
    lm_p.ngrammer_tpl = pax_fiddle.Config(
        layers.ngrammer.VQNgrammer
        if use_vq_ngrammer
        else layers.ngrammer.Ngrammer
    )

    quantize.set_transformer_quantization(
        lm_p,
        quantize_ngrammer_embedding=True,
    )

    if use_vq_ngrammer:
      self.assertEqual(lm_p.ngrammer_tpl.cls, qlayer.ngrammer.VQNgrammer)
    else:
      self.assertEqual(lm_p.ngrammer_tpl.cls, qlayer.ngrammer.Ngrammer)

  @parameterized.parameters(
      itertools.product(
          (True, False),  # quantize_self_attention
          (True, False),  # quantize_cross_attention
          (True, False),  # quantize_embedding_softmax
          (True, False),  # transposed_embedding_softmax
          (True, False),  # separate_embedding
          (True, False),  # softmax_only
      )
  )
  def test_set_quantization_transformer_encdec(
      self,
      quantize_self_attention,
      quantize_cross_attention,
      quantize_embedding_softmax,
      transposed_embedding_softmax,
      separate_embedding,
      softmax_only,
  ):
    encdec_p = pax_fiddle.Config(
        layers.TransformerEncoderDecoder,
        packed_input=True,
        model_dims=16,
    )
    encdec_p.encoder_stacked_transformer_tpl = pax_fiddle.Config(
        layers.StackedTransformer,
        num_layers=2,
    )
    encdec_p.decoder_stacked_transformer_tpl = pax_fiddle.Config(
        layers.StackedTransformer,
        num_layers=2,
        use_cross_attention=True,
    )
    enc_tr_p = (
        encdec_p.encoder_stacked_transformer_tpl.transformer_layer_params_tpl
    )
    dec_tr_p = (
        encdec_p.decoder_stacked_transformer_tpl.transformer_layer_params_tpl
    )
    if transposed_embedding_softmax:
      embedding_softmax_p = encdec_p.softmax_tpl
      new_embedding_softmax_p = pax_fiddle.Config(
          DummyNClassMajorSharedEmbeddingSoftmax
      )
      new_embedding_softmax_p.copy_fields_from(embedding_softmax_p)
      encdec_p.softmax_tpl = new_embedding_softmax_p
    if separate_embedding:
      encdec_p.encoder_embedding_tpl = encdec_p.softmax_tpl.clone()
      encdec_p.decoder_embedding_tpl = encdec_p.softmax_tpl.clone()

    quantize.set_transformer_quantization(
        encdec_p,
        linear_only=False,
        quantize_self_attention=quantize_self_attention,
        quantize_cross_attention=quantize_cross_attention,
        quantize_embedding_softmax=quantize_embedding_softmax,
        transposed_embedding_softmax=transposed_embedding_softmax,
        softmax_only=softmax_only,
    )
    # Expect linears are quantized.
    self.assertEqual(
        enc_tr_p.tr_fflayer_tpl.fflayer_tpl.linear_tpl.cls,
        qlayer.linears.Linear,
    )
    self.assertEqual(
        dec_tr_p.tr_fflayer_tpl.fflayer_tpl.linear_tpl.cls,
        qlayer.linears.Linear,
    )

    # self attention
    self.assertEqual(
        enc_tr_p.tr_atten_tpl.proj_tpl.cls,
        expected_attention_proj_cls(is_quantized=quantize_self_attention),
    )
    self.assertEqual(
        dec_tr_p.tr_atten_tpl.proj_tpl.cls,
        expected_attention_proj_cls(is_quantized=quantize_self_attention),
    )
    # combined qkv projection is quantized but may not be used depending
    # on the combine_qkv setting.
    self.assertEqual(
        enc_tr_p.tr_atten_tpl.combined_qkv_proj_tpl.cls,
        expected_combined_qkv_proj_cls(is_quantized=quantize_self_attention),
    )
    self.assertEqual(
        dec_tr_p.tr_atten_tpl.combined_qkv_proj_tpl.cls,
        expected_combined_qkv_proj_cls(is_quantized=quantize_self_attention),
    )
    # cross attention
    if quantize_cross_attention != quantize_self_attention:
      self.assertEqual(
          dec_tr_p.cross_atten_tpl.proj_tpl.cls,
          expected_attention_proj_cls(is_quantized=quantize_cross_attention),
      )
      self.assertEqual(
          dec_tr_p.cross_atten_tpl.combined_qkv_proj_tpl.cls,
          expected_combined_qkv_proj_cls(is_quantized=quantize_cross_attention),
      )
    else:
      # Cross attention and self attention share the same config.
      self.assertIsNone(dec_tr_p.cross_atten_tpl)

    # softmax and embedding
    self.assertEqual(
        encdec_p.softmax_tpl.cls,
        expected_embedding_softmax_cls(
            is_quantized=quantize_embedding_softmax,
            is_transposed=transposed_embedding_softmax,
        ),
    )
    if separate_embedding:
      self.assertEqual(
          encdec_p.encoder_embedding_tpl.cls,
          expected_embedding_softmax_cls(
              is_quantized=(quantize_embedding_softmax and not softmax_only),
              is_transposed=transposed_embedding_softmax,
          ),
      )
      self.assertEqual(
          encdec_p.decoder_embedding_tpl.cls,
          expected_embedding_softmax_cls(
              is_quantized=(quantize_embedding_softmax and not softmax_only),
              is_transposed=transposed_embedding_softmax,
          ),
      )
    else:
      # Both softmax and embedding are either quantized or not quantized
      # together since they share weights, regardless of softmax_only setting.
      self.assertIsNone(encdec_p.encoder_embedding_tpl)
      self.assertIsNone(encdec_p.decoder_embedding_tpl)

  @parameterized.named_parameters(
      ('ngrammer', False),
      ('vq_ngrammer', True),
  )
  def test_set_quantization_ngramer_transformer_encdec(self, use_vq_ngrammer):
    encdec_p = pax_fiddle.Config(
        layers.TransformerEncoderDecoder,
        packed_input=True,
        model_dims=16,
    )
    encdec_p.decoder_ngrammer_tpl = pax_fiddle.Config(
        layers.ngrammer.VQNgrammer
        if use_vq_ngrammer
        else layers.ngrammer.Ngrammer
    )

    quantize.set_transformer_quantization(
        encdec_p,
        quantize_ngrammer_embedding=True,
    )

    self.assertIsNone(encdec_p.encoder_ngrammer_tpl)
    if use_vq_ngrammer:
      self.assertEqual(
          encdec_p.decoder_ngrammer_tpl.cls, qlayer.ngrammer.VQNgrammer
      )
    else:
      self.assertEqual(
          encdec_p.decoder_ngrammer_tpl.cls, qlayer.ngrammer.Ngrammer
      )

  def test_skip_transformer(self):
    tr_a_p = pax_fiddle.Config(
        layers.transformers.Transformer,
        input_dims=12,
        hidden_dims=4,
        num_heads=8,
        name='A',
    )
    tr_b_p = pax_fiddle.Config(
        layers.transformers.Transformer,
        input_dims=12,
        hidden_dims=4,
        num_heads=8,
        name='B',
    )
    tr_stacked = pax_fiddle.Config(
        layers.transformers.StackedTransformer,
        num_layers=2,
        transformer_layer_params_tpl=[tr_a_p, tr_b_p],
    )
    quantize.set_transformer_quantization(tr_stacked, skip_transformers=['B'])
    self.assertEqual(
        tr_a_p.tr_fflayer_tpl.fflayer_tpl.linear_tpl.cls, qlayer.linears.Linear
    )
    self.assertEqual(
        tr_b_p.tr_fflayer_tpl.fflayer_tpl.linear_tpl.cls, layers.Linear
    )

  def test_diffusion(self):
    class DummyDiffusion(base_layer.BaseLayer):
      conv_tpl: base_layer.BaseLayer = base_layer.template_field(layers.Conv2D)

    class DummyModel(base_layer.BaseLayer):
      diffusion_tpl: base_layer.BaseLayer = base_layer.template_field(
          DummyDiffusion
      )

    model_p = pax_fiddle.Config(DummyModel)

    quantize.set_diffusion_quantization(model_p, DummyDiffusion)

    # Expect conv2d quantized.
    self.assertEqual(model_p.diffusion_tpl.conv_tpl.cls, qlayer.Conv2D)


if __name__ == '__main__':
  absltest.main()
