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

"""Tests for conformers."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from lingvo.core import cluster_factory
from lingvo.core import conformer_layer
import numpy as np
from praxis import base_layer
from praxis import py_utils
from praxis import test_utils
from praxis.layers import conformers
import tensorflow as tf

instantiate = base_layer.instantiate
to_np = test_utils.to_np
NestedMap = py_utils.NestedMap


class ConformerTest(test_utils.TestCase):

  @parameterized.parameters(
      (2, 10, 3, 8, 8, 4, 0.0),
      (3, 12, 5, 16, 16, 2, 0.1),
      (5, 7, 2, 8, 8, 8, 0.25),
      (7, 8, 4, 16, 16, 4, 0.5),
  )
  def test_conformer_layer(self, batch_size, seq_len, kernel_size, input_dims,
                           model_dims, atten_num_heads, dropout_prob):
    # Lingvo TF layers only use dropout on FF and Attention layers
    p = conformers.Conformer.HParams(
        name='jax_conformer_layer',
        input_dims=input_dims,
        conv_residual_dropout=0.0,
        atten_residual_dropout=dropout_prob,
        ffn_residual_dropout=dropout_prob,
        atten_dropout=dropout_prob,
        ffn_relu_dropout=dropout_prob,
        kernel_size=kernel_size,
        model_dims=model_dims,
        atten_num_heads=atten_num_heads)
    conformer = instantiate(p)
    npy_inputs = np.random.normal(
        1.0, 0.5, [batch_size, seq_len, input_dims]).astype('float32')
    inputs = jnp.asarray(npy_inputs)

    def get_padding_from_length(length):
      idx = np.tile(np.arange(seq_len), [batch_size, 1])
      return (idx >= np.expand_dims(length, -1)).astype('float32')

    length = np.random.randint(seq_len // 2, seq_len, (batch_size,))
    npy_paddings = get_padding_from_length(length)
    paddings = jnp.asarray(npy_paddings)

    context_p = base_layer.JaxContext.HParams(do_eval=True)

    with base_layer.JaxContext.new_context(hparams=context_p):
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = conformer.init(prng_key, inputs, paddings)
      output = conformer.apply(initial_vars, inputs, paddings)
    # Test whether tf Conformer layer returns the same output
    # Modify initial_vars to use TF compatible params
    initial_vars_trainable = py_utils.NestedMap.FromNestedDict(
        initial_vars['params'])
    initial_vars_non_trainable = py_utils.NestedMap.FromNestedDict(
        initial_vars['non_trainable'])
    tf_initial_vars = test_utils.replace_jax_conformer_layer_vars_to_tf(
        initial_vars_trainable, initial_vars_non_trainable)

    tf_p = conformer_layer.ConformerLayer.CommonParams(
        input_dim=input_dims,
        dropout_prob=dropout_prob,
        atten_num_heads=atten_num_heads,
        kernel_size=kernel_size,
        fflayer_hidden_dim=model_dims * p.ffn_dim_multiplier,
        use_relative_atten=False,
        fflayer_residual_weight=0.5).Set(name='tf_conformer')
    tf_p.trans_atten_tpl = tf_p.trans_atten_tpl.Set(hidden_dim=model_dims)

    tf_conformer = tf_p.Instantiate()
    with cluster_factory.SetEval(True):
      tf_output = tf_conformer.FProp(
          tf_initial_vars,
          py_utils.NestedMap(
              features=tf.constant(inputs, dtype=tf.float32),
              paddings=tf.constant(npy_paddings, dtype=tf.float32)))
    np_output = to_np(output)
    tf_np_output = to_np(tf_output.features)
    self.assertAllClose(tf_np_output, np_output, atol=1e-5)


if __name__ == '__main__':
  absltest.main()
