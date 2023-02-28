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

"""Tests for quantized attentions."""

from absl.testing import absltest
import jax
import numpy as np
from praxis import base_layer
from praxis import pax_fiddle
from praxis import test_utils
from praxis.layers import attentions
from praxis.layers import conformers
from praxis.layers.quantization import conformers as qconformers
from praxis.layers.quantization import quantization_hparams

QuantizationHParams = quantization_hparams.QuantizationHParams
QuantizationMode = quantization_hparams.QuantizationMode
QuantizationType = quantization_hparams.QuantizationType
instantiate = base_layer.instantiate


class DotProductAttentionWithContexSyncTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  def test_dot_product_attention_with_context(self):
    batch_size = 2
    source_max_length = 16
    input_dim = 32

    atten_f_p = pax_fiddle.Config(
        conformers.DotProductAttentionWithContext, name='atten_f'
    )
    atten_q_p = pax_fiddle.Config(
        qconformers.DotProductAttentionWithContext,
        name='atten_q',
        quantization=QuantizationHParams(
            quantization_type=QuantizationType.AQT,
            mode=QuantizationMode.TRAINING,
            # Test using 23 bits to minimize the quantization error and test
            # for numerical correctness.
            act_params=quantization_hparams.ActQuantizationParams(precision=23),
            weight_params=None,
        ),
    )
    for p in [atten_f_p, atten_q_p]:
      p.input_dim = input_dim
      p.hidden_dim = 16
      p.left_context = 3
      p.right_context = 5

    atten_f = instantiate(atten_f_p)
    atten_q = instantiate(atten_q_p)

    query_vec = np.random.normal(
        size=[batch_size, source_max_length, input_dim]
    ).astype(np.float32)
    key_vec = query_vec
    value_vec = query_vec
    atten_mask = attentions.causal_mask(query_vec)

    with base_layer.JaxContext.new_context():
      initial_vars = atten_f.init(
          jax.random.PRNGKey(0),
          query_vec,
          key_vec,
          value_vec,
          atten_mask,
      )
      fprop_out_f, _ = atten_f.apply(
          initial_vars,
          query_vec,
          key_vec,
          value_vec,
          atten_mask,
          method=atten_f.__call__,
      )
      fprop_out_q, _ = atten_q.apply(
          initial_vars,
          query_vec,
          key_vec,
          value_vec,
          atten_mask,
          method=atten_q.__call__,
      )
      self.assertAllClose(fprop_out_f, fprop_out_q)


if __name__ == '__main__':
  absltest.main()
