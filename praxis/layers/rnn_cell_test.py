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

"""Tests for Praxis rnn_cell layers."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from lingvo.core import py_utils as tf_py_utils
from lingvo.core import rnn_cell
import numpy as np
from praxis import base_layer
from praxis import test_utils
from praxis.layers import rnn_cell as jax_rnn_cell
import tensorflow.compat.v2 as tf

NestedMap = tf_py_utils.NestedMap
instantiate = base_layer.instantiate

_INIT_RANDOM_SEED = 429891685
_NUMPY_RANDOM_SEED = 12345
PARAMS = base_layer.PARAMS


class RnnCellTest(test_utils.TestCase):

  @parameterized.parameters(
      (jax_rnn_cell.LstmCellSimple, False, False),
      (jax_rnn_cell.LstmCellSimple, False, True),
      (jax_rnn_cell.CifgLstmCellSimple, True, False),
      (jax_rnn_cell.CifgLstmCellSimple, True, True),
  )
  def test_LstmSimple(self, jax_cell_class, cifg, output_nonlinearity):
    np.random.seed(_NUMPY_RANDOM_SEED)
    inputs = tf_py_utils.NestedMap(
        act=[np.random.uniform(size=(3, 2))], padding=jnp.zeros([3, 1]))
    state0 = tf_py_utils.NestedMap(
        c=np.random.uniform(size=(3, 2)), m=np.random.uniform(size=(3, 2)))
    tf_inputs = tf_py_utils.NestedMap(
        act=[tf.constant(inputs.act[0], tf.float32)], padding=tf.zeros([3, 1]))
    tf_state0 = tf_py_utils.NestedMap(
        c=tf.constant(state0.c, tf.float32),
        m=tf.constant(state0.m, tf.float32))

    params = rnn_cell.LSTMCellSimple.Params().Set(
        name='lstm',
        params_init=tf_py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED),
        bias_init=tf_py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED),
        num_input_nodes=2,
        num_output_nodes=2,
        couple_input_forget_gates=cifg,
        enable_lstm_bias=True,
        output_nonlinearity=output_nonlinearity)
    lstm = rnn_cell.LSTMCellSimple(params)
    res, _ = lstm.FPropDefaultTheta(tf_state0, tf_inputs)
    m_expected = res.m.numpy()
    c_expected = res.c.numpy()

    p = jax_cell_class.HParams(
        num_input_nodes=2,
        num_output_nodes=2,
        name='lstm',
        output_nonlinearity=output_nonlinearity,
    )
    model = instantiate(p)

    with base_layer.JaxContext.new_context():
      initial_vars = model.init(jax.random.PRNGKey(5678), state0, inputs)
      initial_vars[PARAMS]['wm'] = lstm.vars['wm'].numpy()
      initial_vars[PARAMS]['b'] = lstm.vars['b'].numpy()
      output = model.apply(initial_vars, state0, inputs)
    self.assertAllClose(m_expected, output.m)
    self.assertAllClose(c_expected, output.c)


if __name__ == '__main__':
  absltest.main()
