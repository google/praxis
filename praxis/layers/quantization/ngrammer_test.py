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

"""Test for quantized ngrammer layers."""

import copy

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from praxis import base_layer
from praxis import pax_fiddle
from praxis import test_utils
from praxis.layers import embedding_softmax
from praxis.layers import quantization
from praxis.layers.quantization import quantization_hparams
from praxis.layers import ngrammer

instantiate = base_layer.instantiate
WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams
QuantizationHParams = quantization_hparams.QuantizationHParams
QuantizationMode = quantization_hparams.QuantizationMode
QuantizationType = quantization_hparams.QuantizationType
WeightQuantizationParams = quantization_hparams.WeightQuantizationParams


class NgrammerTest(test_utils.TestCase):
  UNIGRAM_VOCAB_SIZE = 2
  NGRAM_EMB_DIM = 2
  NUM_HEADS = 2
  DIM_PER_HEAD = 4
  BATCH_SIZE = 2
  SEQ_LEN = 2

  EMB_VAR_0 = [
      [-7, -6],
      [-5, -4],
      [-3, -2],
      [-1, 0],
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
  ]
  EMB_VAR_1 = [
      [7, 6],
      [5, 4],
      [3, 2],
      [1, 0],
      [-1, -2],
      [-3, -4],
      [-5, -6],
      [-7, -8],
  ]

  EMB_VAR_0_QUANTIZED_SYMMETRIC = [
      [-127, -109],
      [-127, -102],
      [-127, -85],
      [-127, 0],
      [64, 127],
      [95, 127],
      [106, 127],
      [111, 127],
  ]
  EMB_VAR_1_QUANTIZED_SYMMETRIC = [
      [127, 109],
      [127, 102],
      [127, 85],
      [127, 0],
      [-64, -127],
      [-95, -127],
      [-106, -127],
      [-111, -127],
  ]
  EMB_VAR_0_SCALES_SYMMETRIC = [
      0.05511811,
      0.03937008,
      0.02362205,
      0.00787402,
      0.01574803,
      0.03149606,
      0.04724409,
      0.06299213,
  ]
  EMB_VAR_1_SCALES_SYMMETRIC = [
      0.05511811,
      0.03937008,
      0.02362205,
      0.00787402,
      0.01574803,
      0.03149606,
      0.04724409,
      0.06299213,
  ]

  EMB_VAR_0_QUANTIZED_ASYMMETRIC = [
      [-128, 127],
      [-128, 127],
      [-128, 127],
      [-128, 127],
      [-128, 127],
      [-128, 127],
      [-128, 127],
      [-128, 127],
  ]
  EMB_VAR_1_QUANTIZED_ASYMMETRIC = [
      [127, -128],
      [127, -128],
      [127, -128],
      [127, -128],
      [127, -128],
      [127, -128],
      [127, -128],
      [127, -128],
  ]
  EMB_VAR_0_SCALES_ASYMMETRIC = [
      0.00392157,
      0.00392157,
      0.00392157,
      0.00392157,
      0.00392157,
      0.00392157,
      0.00392157,
      0.00392157,
  ]
  EMB_VAR_1_SCALES_ASYMMETRIC = [
      0.00392157,
      0.00392157,
      0.00392157,
      0.00392157,
      0.00392157,
      0.00392157,
      0.00392157,
      0.00392157,
  ]
  EMB_VAR_0_ZP = [
      6.4980392,
      4.498039,
      2.498039,
      0.4980392,
      -1.5019609,
      -3.5019608,
      -5.5019608,
      -7.5019608,
  ]
  EMB_VAR_1_ZP = [
      -6.5019608,
      -4.501961,
      -2.501961,
      -0.5019608,
      1.4980391,
      3.4980392,
      5.4980392,
      7.4980392,
  ]

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  @parameterized.named_parameters(('symmetric', True), ('asymmetric', False))
  def test_ptq_quantize(self, use_symmetric):
    quantization_option = QuantizationHParams(
        quantization_type=QuantizationType.PTQ,
        mode=QuantizationMode.INFERENCE,
        weight_params=WeightQuantizationParams(use_symmetric=use_symmetric),
    )

    f_p = pax_fiddle.Config(
        ngrammer.Ngrammer,
        name='_ngrammer',
        unigram_vocab_size=self.UNIGRAM_VOCAB_SIZE,
        ngram_vocab_size=self.NUM_HEADS * self.UNIGRAM_VOCAB_SIZE**2,
        ngram_emb_dim=self.NGRAM_EMB_DIM,
        num_heads=self.NUM_HEADS,
        dim_per_head=self.DIM_PER_HEAD,
        concat_ngrams=True,
        mesh_axis_names=['replica', 'mdl', 'data'],
        weight_split_dims_mapping=base_layer.BaseLayer.WeightSharding(
            wt=['mdl', 'data']
        ),
    )
    q_p = pax_fiddle.Config(
        quantization.Ngrammer,
        name='_ngrammer_q',
        unigram_vocab_size=self.UNIGRAM_VOCAB_SIZE,
        ngram_vocab_size=self.NUM_HEADS * self.UNIGRAM_VOCAB_SIZE**2,
        ngram_emb_dim=self.NGRAM_EMB_DIM,
        num_heads=self.NUM_HEADS,
        dim_per_head=self.DIM_PER_HEAD,
        concat_ngrams=True,
        mesh_axis_names=['replica', 'mdl', 'data'],
        weight_split_dims_mapping=base_layer.BaseLayer.WeightSharding(
            wt=['mdl', 'data']
        ),
        quantization=quantization_option,
    )
    f_layer = instantiate(f_p)
    q_layer = instantiate(q_p)

    inputs = np.random.randint(
        self.UNIGRAM_VOCAB_SIZE,
        size=[self.BATCH_SIZE, self.SEQ_LEN, self.NUM_HEADS],
        dtype=np.int32,
    )
    paddings = np.random.randint(1, size=[self.BATCH_SIZE, self.SEQ_LEN])
    input_embs = np.random.normal(
        1.5,
        2.0,
        (self.BATCH_SIZE, self.SEQ_LEN, self.NUM_HEADS * self.DIM_PER_HEAD),
    )
    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)
    with base_layer.JaxContext.new_context():
      f_initial_vars = f_layer.init(init_key, inputs, input_embs, paddings)
      f_initial_vars[base_layer.PARAMS]['ngram_table_0']['emb_var'] = np.array(
          self.EMB_VAR_0, dtype=f_p.dtype
      )
      f_initial_vars[base_layer.PARAMS]['ngram_table_1']['emb_var'] = np.array(
          self.EMB_VAR_1, dtype=f_p.dtype
      )
      f_layer = f_layer.bind(f_initial_vars, mutable=[base_layer.NON_TRAINABLE])
      f_embs = f_layer(inputs, input_embs, paddings)

      q_initial_vars = q_layer.init(init_key, inputs, input_embs, paddings)
      q_initial_vars[base_layer.PARAMS]['ngram_table_0']['emb_var'] = np.array(
          self.EMB_VAR_0_QUANTIZED_SYMMETRIC
          if use_symmetric
          else self.EMB_VAR_0_QUANTIZED_ASYMMETRIC,
          dtype=np.int8,
      )
      q_initial_vars[base_layer.PARAMS]['ngram_table_1']['emb_var'] = np.array(
          self.EMB_VAR_1_QUANTIZED_SYMMETRIC
          if use_symmetric
          else self.EMB_VAR_1_QUANTIZED_ASYMMETRIC,
          dtype=np.int8,
      )
      q_initial_vars[base_layer.PARAMS]['ngram_table_0'][
          'emb_var_quantized_scale'
      ] = np.array(
          self.EMB_VAR_0_SCALES_SYMMETRIC
          if use_symmetric
          else self.EMB_VAR_0_SCALES_ASYMMETRIC,
          dtype=q_p.dtype,
      )
      q_initial_vars[base_layer.PARAMS]['ngram_table_1'][
          'emb_var_quantized_scale'
      ] = np.array(
          self.EMB_VAR_1_SCALES_SYMMETRIC
          if use_symmetric
          else self.EMB_VAR_1_SCALES_ASYMMETRIC,
          dtype=q_p.dtype,
      )
      if not use_symmetric:
        q_initial_vars[base_layer.PARAMS]['ngram_table_0'][
            'emb_var_quantized_zp'
        ] = np.array(
            self.EMB_VAR_0_ZP,
            dtype=q_p.dtype,
        )
        q_initial_vars[base_layer.PARAMS]['ngram_table_1'][
            'emb_var_quantized_zp'
        ] = np.array(
            self.EMB_VAR_1_ZP,
            dtype=q_p.dtype,
        )
      q_layer = q_layer.bind(q_initial_vars, mutable=[base_layer.NON_TRAINABLE])
      q_embs = q_layer(inputs, input_embs, paddings)

    f_embs = np.reshape(
        f_embs,
        [self.BATCH_SIZE, self.SEQ_LEN, self.NUM_HEADS, self.DIM_PER_HEAD],
    )
    q_embs = np.reshape(
        q_embs,
        [self.BATCH_SIZE, self.SEQ_LEN, self.NUM_HEADS, self.DIM_PER_HEAD],
    )
    for i in range(self.NUM_HEADS):
      q_embs_slice = q_embs[:, :, i, -self.NGRAM_EMB_DIM :]
      f_embs_slice = f_embs[:, :, i, -self.NGRAM_EMB_DIM :]
      self.assertAllClose(q_embs_slice, f_embs_slice)


class VQNgrammerTest(test_utils.TestCase):
  NGRAM_EMB_DIM = 2
  NUM_HEADS = 2
  NUM_CLUSTERS = 2
  DIM_PER_HEAD = 4
  BATCH_SIZE = 2
  SEQ_LEN = 2

  EMB_VAR_0 = [
      [-7, -6],
      [-5, -4],
      [-3, -2],
      [-1, 0],
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
      [9, 10],
  ]
  EMB_VAR_1 = [
      [7, 6],
      [5, 4],
      [3, 2],
      [1, 0],
      [-1, -2],
      [-3, -4],
      [-5, -6],
      [-7, -8],
      [-9, -10],
  ]

  EMB_VAR_0_QUANTIZED_SYMMETRIC = [
      [-127, -109],
      [-127, -102],
      [-127, -85],
      [-127, 0],
      [64, 127],
      [95, 127],
      [106, 127],
      [111, 127],
      [114, 127],
  ]
  EMB_VAR_1_QUANTIZED_SYMMETRIC = [
      [127, 109],
      [127, 102],
      [127, 85],
      [127, 0],
      [-64, -127],
      [-95, -127],
      [-106, -127],
      [-111, -127],
      [-114, -127],
  ]
  EMB_VAR_0_SCALES_SYMMETRIC = [
      0.05511811,
      0.03937008,
      0.02362205,
      0.00787402,
      0.01574803,
      0.03149606,
      0.04724409,
      0.06299213,
      0.07874016,
  ]
  EMB_VAR_1_SCALES_SYMMETRIC = [
      0.05511811,
      0.03937008,
      0.02362205,
      0.00787402,
      0.01574803,
      0.03149606,
      0.04724409,
      0.06299213,
      0.07874016,
  ]

  EMB_VAR_0_QUANTIZED_ASYMMETRIC = [
      [-128, 127],
      [-128, 127],
      [-128, 127],
      [-128, 127],
      [-128, 127],
      [-128, 127],
      [-128, 127],
      [-128, 127],
      [-128, 127],
  ]
  EMB_VAR_1_QUANTIZED_ASYMMETRIC = [
      [127, -128],
      [127, -128],
      [127, -128],
      [127, -128],
      [127, -128],
      [127, -128],
      [127, -128],
      [127, -128],
      [127, -128],
  ]
  EMB_VAR_0_SCALES_ASYMMETRIC = [
      0.00392157,
      0.00392157,
      0.00392157,
      0.00392157,
      0.00392157,
      0.00392157,
      0.00392157,
      0.00392157,
      0.00392157,
  ]
  EMB_VAR_1_SCALES_ASYMMETRIC = [
      0.00392157,
      0.00392157,
      0.00392157,
      0.00392157,
      0.00392157,
      0.00392157,
      0.00392157,
      0.00392157,
      0.00392157,
  ]
  EMB_VAR_0_ZP = [
      6.4980392,
      4.498039,
      2.498039,
      0.4980392,
      -1.5019609,
      -3.5019608,
      -5.5019608,
      -7.5019608,
      -9.5019608,
  ]
  EMB_VAR_1_ZP = [
      -6.5019608,
      -4.501961,
      -2.501961,
      -0.5019608,
      1.4980391,
      3.4980392,
      5.4980392,
      7.4980392,
      9.4980392,
  ]

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  @parameterized.named_parameters(('symmetric', True), ('asymmetric', False))
  def test_ptq_quantize(self, use_symmetric):
    quantization_option = QuantizationHParams(
        quantization_type=QuantizationType.PTQ,
        mode=QuantizationMode.INFERENCE,
        weight_params=WeightQuantizationParams(use_symmetric=use_symmetric),
    )

    f_p = pax_fiddle.Config(
        ngrammer.VQNgrammer,
        name='_vq_ngrammer',
        ngram_vocab_size=self.NUM_HEADS * self.NUM_CLUSTERS**2 + 1,
        ngram_emb_dim=self.NGRAM_EMB_DIM,
        num_heads=self.NUM_HEADS,
        num_clusters=self.NUM_CLUSTERS,
        dim_per_head=self.DIM_PER_HEAD,
        concat_ngrams=True,
        ngram_using_attention_scores=False,
        causal_attention=False,
        mesh_axis_names=['replica', 'mdl', 'data'],
        weight_split_dims_mapping=base_layer.BaseLayer.WeightSharding(
            wt=['mdl', 'data']
        ),
    )
    q_p = pax_fiddle.Config(
        quantization.VQNgrammer,
        name='_vq_ngrammer_q',
        ngram_vocab_size=self.NUM_HEADS * self.NUM_CLUSTERS**2 + 1,
        ngram_emb_dim=self.NGRAM_EMB_DIM,
        num_heads=self.NUM_HEADS,
        num_clusters=self.NUM_CLUSTERS,
        dim_per_head=self.DIM_PER_HEAD,
        concat_ngrams=True,
        ngram_using_attention_scores=False,
        causal_attention=False,
        mesh_axis_names=['replica', 'mdl', 'data'],
        weight_split_dims_mapping=base_layer.BaseLayer.WeightSharding(
            wt=['mdl', 'data']
        ),
        quantization=quantization_option,
    )
    f_layer = instantiate(f_p)
    q_layer = instantiate(q_p)
    paddings = np.random.randint(1, size=[self.BATCH_SIZE, self.SEQ_LEN])
    input_embs = np.random.normal(
        1.5,
        2.0,
        (self.BATCH_SIZE, self.SEQ_LEN, self.NUM_HEADS * self.DIM_PER_HEAD),
    )
    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)
    with base_layer.JaxContext.new_context():
      f_initial_vars = f_layer.init(
          init_key, None, input_embs, paddings, attention_scores=None
      )
      f_initial_vars[base_layer.PARAMS]['ngram_layer']['ngram_table_0'][
          'emb_var'
      ] = np.array(self.EMB_VAR_0, dtype=f_p.dtype)
      f_initial_vars[base_layer.PARAMS]['ngram_layer']['ngram_table_1'][
          'emb_var'
      ] = np.array(self.EMB_VAR_1, dtype=f_p.dtype)
      f_layer = f_layer.bind(f_initial_vars, mutable=[base_layer.NON_TRAINABLE])
      f_embs = f_layer(None, input_embs, paddings, attention_scores=None)

      q_initial_vars = q_layer.init(
          init_key, None, input_embs, paddings, attention_scores=None
      )
      q_initial_vars[base_layer.PARAMS]['ngram_layer']['ngram_table_0'][
          'emb_var'
      ] = np.array(
          self.EMB_VAR_0_QUANTIZED_SYMMETRIC
          if use_symmetric
          else self.EMB_VAR_0_QUANTIZED_ASYMMETRIC,
          dtype=np.int8,
      )
      q_initial_vars[base_layer.PARAMS]['ngram_layer']['ngram_table_1'][
          'emb_var'
      ] = np.array(
          self.EMB_VAR_1_QUANTIZED_SYMMETRIC
          if use_symmetric
          else self.EMB_VAR_1_QUANTIZED_ASYMMETRIC,
          dtype=np.int8,
      )
      q_initial_vars[base_layer.PARAMS]['ngram_layer']['ngram_table_0'][
          'emb_var_quantized_scale'
      ] = np.array(
          self.EMB_VAR_0_SCALES_SYMMETRIC
          if use_symmetric
          else self.EMB_VAR_0_SCALES_ASYMMETRIC,
          dtype=q_p.dtype,
      )
      q_initial_vars[base_layer.PARAMS]['ngram_layer']['ngram_table_1'][
          'emb_var_quantized_scale'
      ] = np.array(
          self.EMB_VAR_1_SCALES_SYMMETRIC
          if use_symmetric
          else self.EMB_VAR_1_SCALES_ASYMMETRIC,
          dtype=q_p.dtype,
      )
      if not use_symmetric:
        q_initial_vars[base_layer.PARAMS]['ngram_layer']['ngram_table_0'][
            'emb_var_quantized_zp'
        ] = np.array(
            self.EMB_VAR_0_ZP,
            dtype=q_p.dtype,
        )
        q_initial_vars[base_layer.PARAMS]['ngram_layer']['ngram_table_1'][
            'emb_var_quantized_zp'
        ] = np.array(
            self.EMB_VAR_1_ZP,
            dtype=q_p.dtype,
        )
      q_layer = q_layer.bind(q_initial_vars, mutable=[base_layer.NON_TRAINABLE])
      q_embs = q_layer(None, input_embs, paddings, attention_scores=None)

    f_embs = np.reshape(
        f_embs,
        [self.BATCH_SIZE, self.SEQ_LEN, self.NUM_HEADS, self.DIM_PER_HEAD],
    )
    q_embs = np.reshape(
        q_embs,
        [self.BATCH_SIZE, self.SEQ_LEN, self.NUM_HEADS, self.DIM_PER_HEAD],
    )
    for i in range(self.NUM_HEADS):
      q_embs_slice = q_embs[:, :, i, -self.NGRAM_EMB_DIM :]
      f_embs_slice = f_embs[:, :, i, -self.NGRAM_EMB_DIM :]
      self.assertAllClose(q_embs_slice, f_embs_slice)

  def test_quantize(self):
    quantization_option = QuantizationHParams(
        quantization_type=QuantizationType.PTQ,
        mode=QuantizationMode.MATERIALIZE,
        weight_params=WeightQuantizationParams(use_symmetric=True),
    )

    q_p = pax_fiddle.Config(
        quantization.VQNgrammer,
        name='_vq_ngrammer_q',
        ngram_vocab_size=self.NUM_HEADS * self.NUM_CLUSTERS**2 + 1,
        ngram_emb_dim=self.NGRAM_EMB_DIM,
        num_heads=self.NUM_HEADS,
        num_clusters=self.NUM_CLUSTERS,
        dim_per_head=self.DIM_PER_HEAD,
        concat_ngrams=True,
        ngram_using_attention_scores=False,
        causal_attention=False,
        mesh_axis_names=['replica', 'mdl', 'data'],
        weight_split_dims_mapping=base_layer.BaseLayer.WeightSharding(
            wt=['mdl', 'data']
        ),
        quantization=quantization_option,
    )

    q_layer = instantiate(q_p)
    paddings = np.random.randint(1, size=[self.BATCH_SIZE, self.SEQ_LEN])
    input_embs = np.random.normal(
        1.5,
        2.0,
        (self.BATCH_SIZE, self.SEQ_LEN, self.NUM_HEADS * self.DIM_PER_HEAD),
    )
    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)
    with base_layer.JaxContext.new_context():
      q_initial_vars = q_layer.init(
          init_key, None, input_embs, paddings, attention_scores=None
      )
      q_initial_vars[base_layer.PARAMS]['ngram_layer']['ngram_table_0'][
          'emb_var'
      ] = np.array(
          self.EMB_VAR_0,
          dtype=q_layer.dtype,
      )
      q_initial_vars[base_layer.PARAMS]['ngram_layer']['ngram_table_1'][
          'emb_var'
      ] = np.array(
          self.EMB_VAR_1,
          dtype=q_layer.dtype,
      )

      res, _ = q_layer.apply(
          q_initial_vars, mutable=[], method=q_layer.quantize_weight
      )
      print(res[base_layer.PARAMS]['ngram_layer'])


if __name__ == '__main__':
  absltest.main()
