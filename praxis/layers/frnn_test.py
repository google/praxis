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
import numpy as np
from praxis import base_layer
from praxis import py_utils
from praxis import test_utils
from praxis.layers import frnn
from praxis.layers import rnn_cell as jax_rnn_cell

instantiate = base_layer.instantiate
NestedMap = py_utils.NestedMap
WeightInit = base_layer.WeightInit


class FRNNTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  def _get_cell_params(self,
                       jax_cell_class,
                       output_nonlinearity,
                       direct_config=False):
    params = jax_cell_class.HParams(name='rnn')
    params.params_init = WeightInit.Uniform(1.24)
    params.num_input_nodes = 7
    params.num_output_nodes = 9
    params.zo_prob = 0.0
    params.output_nonlinearity = output_nonlinearity
    return params

  def _sequence_mask(self, lengths: jnp.ndarray, maxlen, dtype):
    batch_size = lengths.shape[0]
    a = jnp.ones([batch_size, maxlen])
    b = jnp.cumsum(a, axis=-1)
    c = jnp.less_equal(b, lengths[:, jnp.newaxis]).astype(dtype)
    return c

  def _get_test_inputs(self):
    seqlen, batch, input_dim, output_dim = 4, 5, 7, 9
    inputs = jnp.array(
        np.random.rand(batch, seqlen, input_dim).astype(np.float32))
    input_lens = np.random.randint(1, seqlen + 1, size=batch)
    padding = 1. - self._sequence_mask(
        input_lens, maxlen=seqlen, dtype=jnp.float32)
    padding = padding[:, :, None]

    m = jnp.array(np.random.rand(batch, output_dim).astype(np.float32))
    c = jnp.array(np.random.rand(batch, output_dim).astype(np.float32))
    return inputs, padding, m, c

  @parameterized.parameters(
      (jax_rnn_cell.LstmCellSimple, False),
      (jax_rnn_cell.LstmCellSimple, True),
      (jax_rnn_cell.CifgLstmCellSimple, False),
      (jax_rnn_cell.CifgLstmCellSimple, True),
  )
  def test_frnn_lstm_cell(self, jax_cell_class, output_nonlinearity):
    cell_p = self._get_cell_params(jax_cell_class, output_nonlinearity)
    frnn_p = frnn.LstmFrnn.HParams(name='frnn', cell_tpl=cell_p)

    act_in, padding, m0, c0 = self._get_test_inputs()
    cell = instantiate(cell_p)
    frnn_model = instantiate(frnn_p)

    state0 = NestedMap(m=m0, c=c0)
    inputs = py_utils.NestedMap(act=act_in, padding=padding)

    with base_layer.JaxContext.new_context():
      theta = frnn_model.init(jax.random.PRNGKey(5678), inputs, state0=state0)
      frnn_act, frnn_state = frnn_model.apply(theta, inputs, state0=state0)

    rnn_theta = {'params': theta['params']['cell']}
    ys = []
    cell_state = jax.tree_map(lambda x: x, state0)
    for t in range(act_in.shape[1]):
      with base_layer.JaxContext.new_context():
        inputs_t = NestedMap(act=act_in[:, t], padding=padding[:, t])
        cell_state = cell.apply(rnn_theta, cell_state, inputs_t)
        y = cell.get_output(cell_state)
      ys.append(y)
    np.testing.assert_allclose(frnn_state.m, cell_state.m, atol=1E-5, rtol=1E-5)
    np.testing.assert_allclose(frnn_state.c, cell_state.c, atol=1E-5, rtol=1E-5)
    np.testing.assert_allclose(frnn_act, jnp.stack(ys, 1), atol=1E-5, rtol=1E-5)

  @parameterized.parameters(
      (jax_rnn_cell.LstmCellSimple, False, 1),
      (jax_rnn_cell.LstmCellSimple, True, 2),
      (jax_rnn_cell.CifgLstmCellSimple, False, 3),
      (jax_rnn_cell.CifgLstmCellSimple, True, 4),
  )
  def test_stackfrnn_lstm(self, jax_cell_class, output_nonlinearity,
                          num_layers):
    input_dim, output_dim = 7, 9
    cell_p = self._get_cell_params(jax_cell_class, output_nonlinearity)
    frnn_p = frnn.LstmFrnn.HParams(name='frnn', cell_tpl=cell_p)
    stack_frnn_p = frnn.StackFrnn.HParams(
        name='stackfrnn',
        frnn_tpl=frnn_p,
        num_input_nodes=input_dim,
        num_output_nodes=output_dim,
        num_layers=num_layers)

    act_in, padding, m0, c0 = self._get_test_inputs()
    stack_frnn_model = instantiate(stack_frnn_p)

    state0 = [NestedMap(m=jnp.copy(m0), c=jnp.copy(c0)) for _ in range(num_layers)]
    inputs = NestedMap(act=act_in, padding=padding)
    with base_layer.JaxContext.new_context():
      theta = stack_frnn_model.init(
          jax.random.PRNGKey(5678), inputs, state0=state0)
      stack_frnn_act, stack_frnn_state = stack_frnn_model.apply(
          theta, inputs, state0=state0)

    num_input_nodes = input_dim
    inputs = NestedMap(act=act_in, padding=padding)
    for ii in range(num_layers):
      cell_p.num_input_nodes = num_input_nodes
      cell_p.num_output_nodes = output_dim
      frnn_p = frnn.LstmFrnn.HParams(name='frnn', cell_tpl=cell_p)
      frnn_model = instantiate(frnn_p)
      state0 = NestedMap(m=jnp.copy(m0), c=jnp.copy(c0))

      rnn_theta = {'params': {'cell': theta['params']['frnn_%d' % ii]['cell']}}
      with base_layer.JaxContext.new_context():
        frnn_act, frnn_state = frnn_model.apply(
            rnn_theta, inputs, state0=state0)
      inputs.act = frnn_act
      self.assertAllClose(frnn_state.m, stack_frnn_state[ii].m)
      self.assertAllClose(frnn_state.c, stack_frnn_state[ii].c)
      num_input_nodes = output_dim
    self.assertAllClose(stack_frnn_act, frnn_act)

  @parameterized.parameters(
      (jax_rnn_cell.LstmCellSimple, False, 1),
      (jax_rnn_cell.LstmCellSimple, True, 2),
      (jax_rnn_cell.CifgLstmCellSimple, False, 3),
      (jax_rnn_cell.CifgLstmCellSimple, True, 4),
  )
  def test_frnn_vs_lstm(self, jax_cell_class, output_nonlinearity, num_layers):
    input_dim, output_dim = 7, 9
    cell_p = self._get_cell_params(jax_cell_class, output_nonlinearity)
    lstm_p = frnn.LstmFrnn.HParams(cell_tpl=cell_p)
    frnn_p = frnn.FRnn.HParams(cell_tpl=cell_p)
    stack_frnn_p = frnn.StackFrnn.HParams(
        name='stackfrnn',
        frnn_tpl=frnn_p,
        num_input_nodes=input_dim,
        num_output_nodes=output_dim,
        num_layers=num_layers)
    stack_lstm_p = frnn.StackFrnn.HParams(
        name='stackfrnn',
        frnn_tpl=lstm_p,
        num_input_nodes=input_dim,
        num_output_nodes=output_dim,
        num_layers=num_layers)

    act_in, padding, m0, c0 = self._get_test_inputs()
    stack_frnn_model = instantiate(stack_frnn_p)
    stack_lstm_model = instantiate(stack_lstm_p)

    state0 = [NestedMap(m=jnp.copy(m0), c=jnp.copy(c0)) for _ in range(num_layers)]
    inputs = NestedMap(act=act_in, padding=padding)
    with base_layer.JaxContext.new_context():
      theta = stack_frnn_model.init(
          jax.random.PRNGKey(5678), inputs, state0=state0)
      stack_frnn_act, stack_frnn_state = stack_frnn_model.apply(
          theta, inputs, state0=state0)
      stack_lstm_act, stack_lstm_state = stack_lstm_model.apply(
          theta, inputs, state0=state0)

    for ii in range(num_layers):
      self.assertAllClose(stack_frnn_state[ii].m, stack_lstm_state[ii].m)
      self.assertAllClose(stack_frnn_state[ii].c, stack_lstm_state[ii].c)
    self.assertAllClose(stack_frnn_act, stack_lstm_act)


if __name__ == '__main__':
  absltest.main()
