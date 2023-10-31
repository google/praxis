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

"""Tests for Praxis rnn_cell layers."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import pax_fiddle
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

  def _get_cell_params(
      self, jax_cell_class, reset_cell_state, output_nonlinearity
  ):
    params = pax_fiddle.Config(jax_cell_class, name='rnn')
    params.params_init = WeightInit.Uniform(1.24)
    params.num_input_nodes = 7
    params.num_output_nodes = 9
    params.reset_cell_state = reset_cell_state
    params.zo_prob = 0.0
    params.output_nonlinearity = output_nonlinearity
    return params

  def _sequence_mask(self, lengths: jnp.ndarray, maxlen, dtype):
    batch_size = lengths.shape[0]
    a = jnp.ones([batch_size, maxlen])
    b = jnp.cumsum(a, axis=-1)
    c = jnp.less_equal(b, lengths[:, jnp.newaxis]).astype(dtype)
    return c

  def _get_test_inputs(self, packed_input: bool = False):
    seqlen, batch, input_dim, output_dim = 4, 5, 7, 9
    inputs = jnp.array(
        np.random.rand(batch, seqlen, input_dim).astype(np.float32)
    )
    input_lens = np.random.randint(1, seqlen + 1, size=batch)
    sequence_mask = self._sequence_mask(
        input_lens, maxlen=seqlen, dtype=jnp.float32
    )
    padding = 1.0 - sequence_mask
    padding = padding[:, :, None]

    if packed_input:
      segment_start_indicator = (
          jnp.concatenate(
              [
                  jnp.ones([batch, 1], dtype=jnp.int32),
                  jnp.array(np.random.randint(2, size=(batch, seqlen - 1))),
              ],
              axis=-1,
          )
          * sequence_mask
      ).astype(jnp.int32)
      segment_ids = (
          jnp.cumsum(segment_start_indicator, axis=-1) * sequence_mask
      ).astype(jnp.int32)
      segment_ids = jnp.expand_dims(segment_ids, axis=-1)
      reset_mask = 1 - segment_start_indicator
      reset_mask = jnp.expand_dims(reset_mask, axis=-1)
    else:
      segment_ids = (
          jnp.ones([batch, seqlen], dtype=jnp.int32) * sequence_mask
      ).astype(jnp.int32)
      segment_ids = jnp.expand_dims(segment_ids, axis=-1)
      reset_mask = jnp.ones_like(segment_ids)

    m = jnp.array(np.random.rand(batch, output_dim).astype(np.float32))
    c = jnp.array(np.random.rand(batch, output_dim).astype(np.float32))

    return inputs, padding, segment_ids, m, c, reset_mask

  @parameterized.parameters(
      *list(itertools.product((jnp.int32, jnp.float32, jnp.bfloat16), repeat=2))
  )
  def test_reset_mask_without_padding(
      self, segment_ids_dtype, reset_mask_dtype
  ):
    segment_ids = jnp.array(
        [
            [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4],
        ],
        dtype=segment_ids_dtype,
    )

    expected_value = jnp.array(
        [
            [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
        ],
        dtype=reset_mask_dtype,
    )
    with self.subTest(name='2D_input'):
      reset_mask = frnn.reset_mask(segment_ids, dtype=reset_mask_dtype)
      self.assertAllClose(reset_mask, expected_value, check_dtypes=True)
    with self.subTest(name='3D_input'):
      reset_mask = frnn.reset_mask(
          jnp.expand_dims(segment_ids, axis=-1), dtype=reset_mask_dtype
      )
      self.assertAllClose(
          reset_mask,
          jnp.expand_dims(expected_value, axis=-1),
          check_dtypes=True,
      )

  @parameterized.parameters(
      *list(itertools.product((jnp.int32, jnp.float32, jnp.bfloat16), repeat=3))
  )
  def test_reset_mask_with_padding(
      self, segment_ids_dtype, paddings_dtype, reset_mask_dtype
  ):
    segment_ids = jnp.array(
        [
            [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4],
        ],
        dtype=segment_ids_dtype,
    )
    paddings = jnp.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=paddings_dtype,
    )
    expected_value = jnp.array(
        [
            [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
        ],
        dtype=reset_mask_dtype,
    )
    with self.subTest(name='2D_input'):
      reset_mask = frnn.reset_mask(
          segment_ids, paddings, dtype=reset_mask_dtype
      )
      self.assertAllClose(reset_mask, expected_value, check_dtypes=True)
    with self.subTest(name='3D_input'):
      reset_mask = frnn.reset_mask(
          jnp.expand_dims(segment_ids, axis=-1),
          jnp.expand_dims(paddings, axis=-1),
          dtype=reset_mask_dtype,
      )
      self.assertAllClose(
          reset_mask,
          jnp.expand_dims(expected_value, axis=-1),
          check_dtypes=True,
      )

  @parameterized.parameters(
      (jax_rnn_cell.LstmCellSimple, False),
      (jax_rnn_cell.LstmCellSimple, True),
      (jax_rnn_cell.CifgLstmCellSimple, False),
      (jax_rnn_cell.CifgLstmCellSimple, True),
  )
  def test_frnn_lstm_cell(self, jax_cell_class, output_nonlinearity):
    cell_p = self._get_cell_params(jax_cell_class, False, output_nonlinearity)
    frnn_p = pax_fiddle.Config(frnn.LstmFrnn, name='frnn', cell_tpl=cell_p)

    act_in, padding, _, m0, c0, _ = self._get_test_inputs()
    cell = instantiate(cell_p)
    frnn_model = instantiate(frnn_p)

    state0 = NestedMap(m=m0, c=c0)
    inputs = NestedMap(act=act_in, padding=padding)

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
    np.testing.assert_allclose(frnn_state.m, cell_state.m, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(frnn_state.c, cell_state.c, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(frnn_act, jnp.stack(ys, 1), atol=1e-5, rtol=1e-5)

  @parameterized.parameters(
      (jax_rnn_cell.LstmCellSimple, False, 1),
      (jax_rnn_cell.LstmCellSimple, True, 2),
      (jax_rnn_cell.CifgLstmCellSimple, False, 3),
      (jax_rnn_cell.CifgLstmCellSimple, True, 4),
  )
  def test_stackfrnn_lstm(
      self, jax_cell_class, output_nonlinearity, num_layers
  ):
    input_dim, output_dim = 7, 9
    cell_p = self._get_cell_params(jax_cell_class, False, output_nonlinearity)
    frnn_p = pax_fiddle.Config(frnn.LstmFrnn, name='frnn', cell_tpl=cell_p)
    stack_frnn_p = pax_fiddle.Config(
        frnn.StackFrnn,
        name='stackfrnn',
        frnn_tpl=frnn_p,
        num_input_nodes=input_dim,
        num_output_nodes=output_dim,
        num_layers=num_layers,
    )

    act_in, padding, _, m0, c0, _ = self._get_test_inputs()
    stack_frnn_model = instantiate(stack_frnn_p)

    state0 = [
        NestedMap(m=jnp.copy(m0), c=jnp.copy(c0)) for _ in range(num_layers)
    ]
    inputs = NestedMap(act=act_in, padding=padding)
    with base_layer.JaxContext.new_context():
      theta = stack_frnn_model.init(
          jax.random.PRNGKey(5678), inputs, state0=state0
      )
      stack_frnn_act, stack_frnn_state = stack_frnn_model.apply(
          theta, inputs, state0=state0
      )

    num_input_nodes = input_dim
    inputs = NestedMap(act=act_in, padding=padding)
    for ii in range(num_layers):
      cell_p.num_input_nodes = num_input_nodes
      cell_p.num_output_nodes = output_dim
      frnn_p = pax_fiddle.Config(frnn.LstmFrnn, name='frnn', cell_tpl=cell_p)
      frnn_model = instantiate(frnn_p)
      state0 = NestedMap(m=jnp.copy(m0), c=jnp.copy(c0))

      rnn_theta = {'params': {'cell': theta['params']['frnn_%d' % ii]['cell']}}
      with base_layer.JaxContext.new_context():
        frnn_act, frnn_state = frnn_model.apply(
            rnn_theta, inputs, state0=state0
        )
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
  def test_stackbifrnn_lstm(
      self, jax_cell_class, output_nonlinearity, num_layers
  ):
    seqlen, batch, input_dim, output_dim = 4, 5, 7, 18
    cell_p = self._get_cell_params(jax_cell_class, False, output_nonlinearity)
    frnn_p = pax_fiddle.Config(frnn.LstmFrnn, name='frnn', cell_tpl=cell_p)
    stack_bifrnn_p = pax_fiddle.Config(
        frnn.StackBiFrnn,
        name='StackBiFrnn',
        frnn_tpl=frnn_p,
        num_input_nodes=input_dim,
        num_output_nodes=output_dim,
        num_layers=num_layers,
    )

    act_in, padding, _, m0, c0, _ = self._get_test_inputs()
    stack_bifrnn_model = instantiate(stack_bifrnn_p)

    inputs = NestedMap(act=act_in, padding=padding)
    with base_layer.JaxContext.new_context():
      theta = stack_bifrnn_model.init(jax.random.PRNGKey(5678), inputs)

      # Test init_state.
      state0 = stack_bifrnn_model.apply(
          theta, batch_size=batch, method=stack_bifrnn_model.init_states
      )
      # We have an init_satae for each layer and for fwd and bwd.
      self.assertLen(state0.fwd, num_layers)
      self.assertLen(state0.bwd, num_layers)

      # Test the shapes of those states.
      self.assertTupleEqual(state0.fwd[-1].m.shape, m0.shape)
      self.assertTupleEqual(state0.bwd[0].m.shape, m0.shape)
      self.assertTupleEqual(state0.fwd[-1].c.shape, c0.shape)
      self.assertTupleEqual(state0.bwd[0].c.shape, c0.shape)

      stack_frnn_act, stack_frnn_state = stack_bifrnn_model.apply(
          theta, inputs, state0=state0
      )

      self.assertTupleEqual(stack_frnn_act.shape, (batch, seqlen, output_dim))
      self.assertLen(stack_frnn_state, num_layers)

      last_state = stack_frnn_state[-1]
      self.assertTupleEqual(last_state.fwd.m.shape, (batch, output_dim // 2))
      self.assertTupleEqual(last_state.fwd.c.shape, (batch, output_dim // 2))
      self.assertTupleEqual(last_state.bwd.m.shape, (batch, output_dim // 2))
      self.assertTupleEqual(last_state.bwd.c.shape, (batch, output_dim // 2))

  @parameterized.parameters(
      (jax_rnn_cell.LstmCellSimple, False, 1),
      (jax_rnn_cell.LstmCellSimple, True, 2),
      (jax_rnn_cell.CifgLstmCellSimple, False, 3),
      (jax_rnn_cell.CifgLstmCellSimple, True, 4),
  )
  def test_frnn_vs_lstm(self, jax_cell_class, output_nonlinearity, num_layers):
    input_dim, output_dim = 7, 9
    cell_p = self._get_cell_params(jax_cell_class, False, output_nonlinearity)
    lstm_p = pax_fiddle.Config(frnn.LstmFrnn, cell_tpl=cell_p)
    frnn_p = pax_fiddle.Config(frnn.FRnn, cell_tpl=cell_p)
    stack_frnn_p = pax_fiddle.Config(
        frnn.StackFrnn,
        name='stackfrnn',
        frnn_tpl=frnn_p,
        num_input_nodes=input_dim,
        num_output_nodes=output_dim,
        num_layers=num_layers,
    )
    stack_lstm_p = pax_fiddle.Config(
        frnn.StackFrnn,
        name='stackfrnn',
        frnn_tpl=lstm_p,
        num_input_nodes=input_dim,
        num_output_nodes=output_dim,
        num_layers=num_layers,
    )

    act_in, padding, _, m0, c0, _ = self._get_test_inputs()
    stack_frnn_model = instantiate(stack_frnn_p)
    stack_lstm_model = instantiate(stack_lstm_p)

    state0 = [
        NestedMap(m=jnp.copy(m0), c=jnp.copy(c0)) for _ in range(num_layers)
    ]
    inputs = NestedMap(act=act_in, padding=padding)
    with base_layer.JaxContext.new_context():
      theta = stack_frnn_model.init(
          jax.random.PRNGKey(5678), inputs, state0=state0
      )
      stack_frnn_act, stack_frnn_state = stack_frnn_model.apply(
          theta, inputs, state0=state0
      )
      stack_lstm_act, stack_lstm_state = stack_lstm_model.apply(
          theta, inputs, state0=state0
      )

    for ii in range(num_layers):
      self.assertAllClose(stack_frnn_state[ii].m, stack_lstm_state[ii].m)
      self.assertAllClose(stack_frnn_state[ii].c, stack_lstm_state[ii].c)
    self.assertAllClose(stack_frnn_act, stack_lstm_act)

  @parameterized.parameters(
      *list(
          itertools.product(
              (
                  jax_rnn_cell.LstmCellSimple,
                  jax_rnn_cell.CifgLstmCellSimple,
                  jax_rnn_cell.LayerNormalizedLstmCellSimple,
              ),
              (frnn.FRnn, frnn.LstmFrnn),
              (True, False),
          )
      )
  )
  def test_frnn_reset_cell_state(
      self, jax_cell_class, frnn_class, output_nonlinearity
  ):
    cell_p = self._get_cell_params(jax_cell_class, True, output_nonlinearity)
    frnn_p = pax_fiddle.Config(frnn_class, cell_tpl=cell_p)

    act_in, padding, segment_ids, m0, c0, reset_mask = self._get_test_inputs(
        packed_input=True
    )
    cell = instantiate(cell_p)
    frnn_model = instantiate(frnn_p)

    state0 = NestedMap(m=m0, c=c0)
    inputs = NestedMap(act=act_in, padding=padding, segment_ids=segment_ids)

    with base_layer.JaxContext.new_context():
      theta = frnn_model.init(jax.random.PRNGKey(5678), inputs, state0=state0)
      frnn_act, frnn_state = frnn_model.apply(theta, inputs, state0=state0)

    rnn_theta = {'params': theta['params']['cell']}
    ys = []
    cell_state = jax.tree_map(lambda x: x, state0)
    for t in range(act_in.shape[1]):
      with base_layer.JaxContext.new_context():
        inputs_t = NestedMap(
            act=act_in[:, t], padding=padding[:, t], reset_mask=reset_mask[:, t]
        )
        cell_state = cell.apply(rnn_theta, cell_state, inputs_t)
        y = cell.get_output(cell_state)
      ys.append(y)
    np.testing.assert_allclose(frnn_state.m, cell_state.m, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(frnn_state.c, cell_state.c, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(frnn_act, jnp.stack(ys, 1), atol=1e-5, rtol=1e-5)

  @parameterized.parameters(
      *list(
          itertools.product(
              (
                  jax_rnn_cell.LstmCellSimple,
                  jax_rnn_cell.CifgLstmCellSimple,
                  jax_rnn_cell.LayerNormalizedLstmCellSimple,
              ),
              (frnn.FRnn, frnn.LstmFrnn),
              (True, False),
          )
      )
  )
  def test_lstm_reset_cell_state_extend_step(
      self, jax_cell_class, frnn_class, output_nonlinearity
  ):
    cell_p = self._get_cell_params(jax_cell_class, True, output_nonlinearity)
    frnn_p = pax_fiddle.Config(frnn_class, cell_tpl=cell_p)

    act_in, padding, _, m0, c0, _ = self._get_test_inputs(packed_input=False)
    frnn_model = instantiate(frnn_p)

    state0 = NestedMap(m=m0, c=c0)
    inputs = NestedMap(act=act_in, padding=padding)
    seq_len = inputs.act.shape[1]

    # Verify the output of '__call__' is equivalent to the output of sequential
    # calls of 'extend_step'.
    with base_layer.JaxContext.new_context():
      theta = frnn_model.init(jax.random.PRNGKey(5678), inputs, state0=state0)
      # Outpuf of '__call__'.
      frnn_act, frnn_state = frnn_model.apply(theta, inputs, state0=state0)
      # Sequential calls of 'extend_step'.
      state_i = state0
      for i in range(seq_len):
        inputs_i = NestedMap(act=inputs.act[:, i], padding=padding[:, i])
        step_state, step_act = frnn_model.apply(
            theta, inputs_i, state0=state_i, method=frnn_model.extend_step
        )
        state_i = step_state
        np.testing.assert_allclose(
            frnn_act[:, i], step_act, atol=1e-5, rtol=1e-5
        )
      np.testing.assert_allclose(frnn_state.m, state_i.m, atol=1e-5, rtol=1e-5)
      np.testing.assert_allclose(frnn_state.c, state_i.c, atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
  absltest.main()
