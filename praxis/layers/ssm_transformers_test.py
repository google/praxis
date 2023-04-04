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

"""Tests for Praxis ssm_transformer layers."""

import copy
import itertools

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_hyperparams
from praxis import base_layer
from praxis import pax_fiddle
from praxis import py_utils
from praxis import test_utils
from praxis.layers import attentions
from praxis.layers import ssm_transformers
import tensorflow.compat.v2 as tf

PARAMS = base_layer.PARAMS
RANDOM = base_layer.RANDOM
DECODE_CACHE = base_layer.DECODE_CACHE
instantiate = base_hyperparams.instantiate


class SSMTransformersTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)
    tf.random.set_seed(123)

  @parameterized.parameters(*list(itertools.product([True, False], repeat=2)))
  def test_ssm_transformer_layer(self, cross_attention, use_gss):
    p = pax_fiddle.Config(
        (ssm_transformers.SSMGated
         if use_gss else ssm_transformers.SSMTransformer),
        name='ssm_transformer_layer',
        input_dims=8,
        hidden_dims=32,
        num_heads=4,
        use_cross_attention=cross_attention,
        ssm_nheads=4, 
        ssm_dim=8, 
        ssm_l_max=4,
        decode_num_samples=1, 
        ssm_hippo_type='ss4d-1d-legs',
        ssm_step_size=1.0
    )
    if cross_attention:
      p.cross_atten_tpl = copy.deepcopy(p.tr_atten_tpl)

    p.tr_atten_tpl.dconv_kernel_size = 2
    seq_len = 4
    batch_size = 4
    transformer_layer = instantiate(p)
    prng_key = jax.random.PRNGKey(seed=123)
    npy_inputs = np.random.normal(
        1.0, 0.5, [batch_size, seq_len, p.input_dims]).astype('float32')
    inputs = jnp.asarray(npy_inputs)
    npy_paddings = np.random.randint(0, 1,
                                     [batch_size, seq_len]).astype('float32')
    # npy_paddings = np.zeros([batch_size, seq_len])
    paddings = jnp.asarray(npy_paddings)
    attention_mask = attentions.convert_paddings_to_mask(paddings)
    causal_mask = attentions.causal_mask(inputs)
    attention_mask = jnp.minimum(causal_mask, attention_mask)
    cross_inputs = None
    cross_paddings = None
    cross_attention_mask = None
    if cross_attention:
      cross_seq_len = np.random.randint(10, 32)
      npy_cross_inputs = np.random.normal(
          1.0, 0.5, [batch_size, cross_seq_len, p.input_dims]).astype('float32')
      cross_inputs = jnp.asarray(npy_cross_inputs)
      npy_cross_paddings = np.random.randint(
          0, 1, [batch_size, cross_seq_len]).astype('float32')
      cross_paddings = jnp.asarray(npy_cross_paddings)
      cross_attention_mask = attentions.convert_paddings_to_mask(cross_paddings)

    with base_layer.JaxContext.new_context():
      initial_vars = transformer_layer.init(
          prng_key,
          jnp.zeros_like(inputs),
          jnp.ones_like(paddings),
          attention_mask=attention_mask,
          cross_inputs=cross_inputs,
          cross_attention_mask=cross_attention_mask,
          method=transformer_layer.__call__)
      _, decoder_state = transformer_layer.apply(
          initial_vars,
          jnp.zeros_like(inputs),
          jnp.ones_like(paddings),
          attention_mask=attention_mask,
          cross_inputs=cross_inputs,
          cross_attention_mask=cross_attention_mask,
          method=transformer_layer.__call__,
          mutable=[DECODE_CACHE])
      fprop_outputs, _ = transformer_layer.apply(
          initial_vars,
          inputs,
          paddings,
          attention_mask=attention_mask,
          cross_inputs=cross_inputs,
          cross_attention_mask=cross_attention_mask,
          method=transformer_layer.__call__)
      decoder_outputs = jnp.zeros(shape=[seq_len, batch_size, p.input_dims])
      updated_vars = py_utils.merge_dict(decoder_state, initial_vars)
      for t in range(seq_len):
        attention_mask_t = attention_mask[:, :, t, :]
        cross_attention_mask_t = cross_attention_mask
        if cross_attention:
          cross_attention_mask_t = cross_attention_mask[:, :, t, :]
          cross_attention_mask_t = np.expand_dims(
              cross_attention_mask_t, axis=2)
        encoded, decoder_state = transformer_layer.apply(
            updated_vars,
            inputs=inputs[:, t, :],
            time_step=t,
            attention_mask=attention_mask_t,
            cross_attention_mask=cross_attention_mask_t,
            method=transformer_layer.extend_step,
            mutable=[DECODE_CACHE])
        updated_vars = py_utils.merge_dict(decoder_state, initial_vars)
        decoder_outputs = decoder_outputs.at[t].set(encoded)

    decoder_out_transposed = jnp.transpose(decoder_outputs, [1, 0, 2])
    logging.info('initial_vars in transformer layer = %s',
                 jax.tree_map(lambda x: x.shape, initial_vars))
    np_fprop_outputs = test_utils.to_np(fprop_outputs)
    np_decoder_outputs = test_utils.to_np(decoder_out_transposed)
    self.assertAllClose(np_fprop_outputs, np_decoder_outputs, atol=1e-5)


if __name__ == '__main__':
  absltest.main()
