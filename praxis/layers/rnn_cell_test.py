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
from lingvo.core import py_utils as tf_py_utils
from lingvo.core import rnn_cell
import numpy as np
from praxis import base_layer
from praxis import pax_fiddle
from praxis import test_utils
from praxis.layers import rnn_cell as jax_rnn_cell
import tensorflow.compat.v2 as tf

NestedMap = tf_py_utils.NestedMap
instantiate = base_layer.instantiate

_INIT_RANDOM_SEED = 429891685
_NUMPY_RANDOM_SEED = 12345
PARAMS = base_layer.PARAMS


def _jax_to_tf_dtype(dtype):
  if dtype is None:
    return None
  return tf.as_dtype(dtype.dtype.name)


class RnnCellTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(_NUMPY_RANDOM_SEED)

  @parameterized.parameters(
      *list(
          itertools.product(
              (jax_rnn_cell.LstmCellSimple, jax_rnn_cell.CifgLstmCellSimple),
              (True, False),  # output_nonlinearity
              (jnp.int32, jnp.float32),  # inputs_nonact_dtype
              (jnp.float32, jnp.bfloat16),  # dtype
              (jnp.float32, jnp.bfloat16, jnp.float16),  # fprop_dtype
          )
      )
  )
  def test_LstmSimple(
      self,
      jax_cell_class,
      output_nonlinearity,
      inputs_nonact_dtype,
      dtype,
      fprop_dtype,
  ):
    """Test Jax vs TF implementation equivalence of fprop.

    Args:
      jax_cell_class: Class of jax rnn cell.
      output_nonlinearity: Whether or not to apply tanh non-linearity on lstm
        output.
      inputs_nonact_dtype: The dtype of non-activation inputs to the module.
        Non-activation inputs includes, ids, paddings, segment_ids, segment_pos.
      dtype: Default dtype for all variables.
      fprop_dtype: Activations datatype to use. This applies to both module
        internal and output activations, also external activation inputs.
    """
    act = np.random.uniform(size=(3, 2)).astype(fprop_dtype)
    c = np.random.uniform(size=(3, 2)).astype(fprop_dtype)
    m = np.random.uniform(size=(3, 2)).astype(fprop_dtype)
    inputs = tf_py_utils.NestedMap(
        act=[jnp.array(act, dtype=fprop_dtype)],
        padding=jnp.zeros([3, 1], dtype=inputs_nonact_dtype),
    )
    state0 = tf_py_utils.NestedMap(
        c=jnp.array(c, dtype=fprop_dtype), m=jnp.array(m, dtype=fprop_dtype)
    )
    tf_inputs = tf_py_utils.NestedMap(
        act=[tf.constant(act, _jax_to_tf_dtype(fprop_dtype))],
        padding=tf.zeros([3, 1], _jax_to_tf_dtype(inputs_nonact_dtype)),
    )
    tf_state0 = tf_py_utils.NestedMap(
        c=tf.constant(c, _jax_to_tf_dtype(fprop_dtype)),
        m=tf.constant(m, _jax_to_tf_dtype(fprop_dtype)),
    )

    cifg = True if jax_cell_class == jax_rnn_cell.CifgLstmCellSimple else False
    params = rnn_cell.LSTMCellSimple.Params().Set(
        name='lstm',
        params_init=tf_py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED),
        bias_init=tf_py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED),
        num_input_nodes=2,
        num_output_nodes=2,
        couple_input_forget_gates=cifg,
        enable_lstm_bias=True,
        output_nonlinearity=output_nonlinearity,
        dtype=_jax_to_tf_dtype(dtype),
        fprop_dtype=_jax_to_tf_dtype(fprop_dtype),
    )
    lstm = rnn_cell.LSTMCellSimple(params)
    res, _ = lstm.FPropDefaultTheta(tf_state0, tf_inputs)
    m_expected = res.m.numpy()
    c_expected = res.c.numpy()

    p = pax_fiddle.Config(
        jax_cell_class,
        num_input_nodes=2,
        num_output_nodes=2,
        name='lstm',
        output_nonlinearity=output_nonlinearity,
        dtype=dtype,
        fprop_dtype=fprop_dtype,
    )
    model = instantiate(p)

    with base_layer.JaxContext.new_context():
      initial_vars = model.init(jax.random.PRNGKey(5678), state0, inputs)
      jax.tree_map(lambda x: self.assertDtypesMatch(x, dtype), initial_vars)
      initial_vars[PARAMS]['wm'] = lstm.vars['wm'].numpy()
      initial_vars[PARAMS]['b'] = lstm.vars['b'].numpy()
      jax.tree_map(lambda x: self.assertDtypesMatch(x, dtype), initial_vars)
      output = model.apply(initial_vars, state0, inputs)

    self.assertDtypesMatch(m_expected, fprop_dtype)
    self.assertDtypesMatch(c_expected, fprop_dtype)
    self.assertDtypesMatch(output.m, fprop_dtype)
    self.assertDtypesMatch(output.c, fprop_dtype)
    # Jax and tf matmul and sigmoid, etc do not produce identical result
    # when lower precision floating point format is used.
    # For test stability we only assert equivalence for fp32.
    if fprop_dtype == jnp.float32:
      self.assertAllClose(output.m, m_expected)
      self.assertAllClose(output.c, c_expected)

  @parameterized.parameters(
      *list(
          itertools.product(
              (True, False),  # output_nonlinearity
              (jnp.int32, jnp.float32),  # inputs_nonact_dtype
              (jnp.float32, jnp.bfloat16),  # dtype
              (jnp.float32, jnp.bfloat16, jnp.float16),  # fprop_dtype
          )
      )
  )
  def test_LayerNormedLstmSimple(
      self,
      output_nonlinearity,
      inputs_nonact_dtype,
      dtype,
      fprop_dtype,
  ):
    """Test Jax vs TF implementation equivalence.

    Args:
      output_nonlinearity: Whether or not to apply tanh non-linearity on lstm
        output.
      inputs_nonact_dtype: The dtype of non-activation inputs to the module.
        Non-activation inputs includes, ids, paddings, segment_ids, segment_pos.
      dtype: Default dtype for all variables.
      fprop_dtype: Activations datatype to use. This applies to both module
        internal and output activations, also external activation inputs.
    """
    act = np.random.uniform(size=(3, 2)).astype(fprop_dtype)
    c = np.random.uniform(size=(3, 2)).astype(fprop_dtype)
    m = np.random.uniform(size=(3, 2)).astype(fprop_dtype)
    inputs = tf_py_utils.NestedMap(
        act=[jnp.array(act, dtype=fprop_dtype)],
        padding=jnp.zeros([3, 1], dtype=inputs_nonact_dtype),
    )
    state0 = tf_py_utils.NestedMap(
        c=jnp.array(c, dtype=fprop_dtype), m=jnp.array(m, dtype=fprop_dtype)
    )
    tf_inputs = tf_py_utils.NestedMap(
        act=[tf.constant(act, _jax_to_tf_dtype(fprop_dtype))],
        padding=tf.zeros([3, 1], _jax_to_tf_dtype(inputs_nonact_dtype)),
    )
    tf_state0 = tf_py_utils.NestedMap(
        c=tf.constant(c, _jax_to_tf_dtype(fprop_dtype)),
        m=tf.constant(m, _jax_to_tf_dtype(fprop_dtype)),
    )
    params = rnn_cell.LayerNormalizedLSTMCellSimple.Params().Set(
        name='lstm',
        params_init=tf_py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED),
        bias_init=tf_py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED),
        num_input_nodes=2,
        num_output_nodes=2,
        couple_input_forget_gates=False,
        enable_lstm_bias=True,
        output_nonlinearity=output_nonlinearity,
        dtype=_jax_to_tf_dtype(dtype),
        fprop_dtype=_jax_to_tf_dtype(fprop_dtype),
    )
    lstm = rnn_cell.LayerNormalizedLSTMCellSimple(params)
    res, _ = lstm.FPropDefaultTheta(tf_state0, tf_inputs)
    m_expected = res.m.numpy()
    c_expected = res.c.numpy()
    p = pax_fiddle.Config(
        jax_rnn_cell.LayerNormalizedLstmCellSimple,
        num_input_nodes=2,
        num_output_nodes=2,
        name='lstm',
        output_nonlinearity=output_nonlinearity,
        dtype=dtype,
        fprop_dtype=fprop_dtype,
    )
    model = instantiate(p)
    with base_layer.JaxContext.new_context():
      initial_vars = model.init(jax.random.PRNGKey(5678), state0, inputs)
      jax.tree_map(lambda x: self.assertDtypesMatch(x, dtype), initial_vars)
      initial_vars[PARAMS]['wm'] = lstm.vars['wm'].numpy()
      initial_vars[PARAMS]['b'] = lstm.vars['b'].numpy()
      initial_vars[PARAMS]['ln_scale'] = lstm.vars['ln_scale'].numpy()
      jax.tree_map(lambda x: self.assertDtypesMatch(x, dtype), initial_vars)
      output = model.apply(initial_vars, state0, inputs)
    self.assertDtypesMatch(m_expected, fprop_dtype)
    self.assertDtypesMatch(c_expected, fprop_dtype)
    self.assertDtypesMatch(output.m, fprop_dtype)
    self.assertDtypesMatch(output.c, fprop_dtype)
    # Jax and tf matmul and sigmoid, etc do not produce identical result
    # when lower precision floating point format is used.
    # For test stability we only assert equivalence for fp32.
    if fprop_dtype == jnp.float32:
      self.assertAllClose(output.m, m_expected)
      self.assertAllClose(output.c, c_expected)

  @parameterized.parameters(
      *list(
          itertools.product(
              (
                  (jax_rnn_cell.LstmCellSimple, rnn_cell.LSTMCellSimple),
                  (
                      jax_rnn_cell.LayerNormalizedLstmCellSimple,
                      rnn_cell.LayerNormalizedLSTMCellSimple,
                  ),
              ),
              (jnp.float32, jnp.bfloat16),  # dtype
              (jnp.float32, jnp.bfloat16, jnp.float16, None),  # fprop_dtype
          )
      )
  )
  def test_init_states(
      self,
      rnn_cell_cls_pair,
      dtype,
      fprop_dtype,
  ):
    """Test Jax and TF implementation for init_states are equivalent.

    Args:
      rnn_cell_cls_pair: Pair of (jax_rnn_cell_cls, tf_rnn_cell_cls) to test
        equivalency.
      dtype: Default dtype for all variables.
      fprop_dtype: Activations datatype to use. If None, default to dtype.
    """
    jax_rnn_cell_cls, tf_rnn_cell_cls = rnn_cell_cls_pair
    batch_size = 3
    params = tf_rnn_cell_cls.Params().Set(
        name='lstm',
        num_input_nodes=2,
        num_output_nodes=2,
        zero_state_init_params=tf_py_utils.RNNCellStateInit.Zeros(),
        dtype=_jax_to_tf_dtype(dtype),
        fprop_dtype=_jax_to_tf_dtype(fprop_dtype),
    )
    lstm = tf_rnn_cell_cls(params)
    tf_state0 = lstm.zero_state(lstm.theta, batch_size)
    expected_c = tf_state0.c.numpy()
    expected_m = tf_state0.m.numpy()

    p = pax_fiddle.Config(
        jax_rnn_cell_cls,
        num_input_nodes=2,
        num_output_nodes=2,
        name='lstm',
        dtype=dtype,
        fprop_dtype=fprop_dtype,
    )
    model = instantiate(p)
    state0 = model.init_states(batch_size)
    # When fprop_dtype = None, module should internally fallback to dtype.
    self.assertDtypesMatch(state0.c, fprop_dtype or dtype)
    self.assertDtypesMatch(state0.m, fprop_dtype or dtype)
    self.assertAllClose(state0.c, expected_c)
    self.assertAllClose(state0.m, expected_m)

  @parameterized.parameters(
      *list(
          itertools.product(
              (
                  jax_rnn_cell.LstmCellSimple,
                  jax_rnn_cell.LayerNormalizedLstmCellSimple,
                  jax_rnn_cell.CifgLstmCellSimple,
              ),
              (True, False),  # output_nonlinearity
              (jnp.int32, jnp.float32),  # inputs_nonact_dtype
              (jnp.float32, jnp.bfloat16),  # dtype
              (jnp.float32, jnp.bfloat16, jnp.float16),  # fprop_dtype
          )
      )
  )
  def test_fprop_with_projected_inputs(
      self,
      rnn_cell_cls,
      output_nonlinearity,
      inputs_nonact_dtype,
      dtype,
      fprop_dtype,
  ):
    p = pax_fiddle.Config(
        rnn_cell_cls,
        num_input_nodes=2,
        num_output_nodes=2,
        name='lstm',
        output_nonlinearity=output_nonlinearity,
        dtype=dtype,
        fprop_dtype=fprop_dtype,
    )
    model = instantiate(p)

    act = np.random.uniform(size=(3, 2)).astype(fprop_dtype)
    c = np.random.uniform(size=(3, 2)).astype(fprop_dtype)
    m = np.random.uniform(size=(3, 2)).astype(fprop_dtype)
    inputs = tf_py_utils.NestedMap(
        act=[jnp.array(act, dtype=fprop_dtype)],
        padding=jnp.zeros([3, 1], dtype=inputs_nonact_dtype),
    )
    state0 = tf_py_utils.NestedMap(
        c=jnp.array(c, dtype=fprop_dtype), m=jnp.array(m, dtype=fprop_dtype)
    )

    with base_layer.JaxContext.new_context():
      initial_vars = model.init(jax.random.PRNGKey(5678), state0, inputs)
      jax.tree_map(lambda x: self.assertDtypesMatch(x, dtype), initial_vars)
      state1_fprop = model.apply(initial_vars, state0, inputs)

      projected_inputs = model.apply(
          initial_vars,
          jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), inputs),
          method=model.project_input,
      )
      inputs_with_proj = jax.tree_map(lambda x: x, inputs)
      inputs_with_proj.proj_inputs = jnp.squeeze(projected_inputs, axis=0)
      state1_fprop_with_projected_inputs = model.apply(
          initial_vars,
          state0,
          inputs_with_proj,
          method=model.fprop_with_projected_inputs,
      )
      # With low precision (bfloat16 and float16), the results do not exactly
      # match.
      if fprop_dtype == jnp.float32:
        self.assertAllClose(
            state1_fprop.m, state1_fprop_with_projected_inputs.m
        )
        self.assertAllClose(
            state1_fprop.c, state1_fprop_with_projected_inputs.c
        )


if __name__ == '__main__':
  absltest.main()
