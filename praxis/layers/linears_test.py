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

"""Tests for Praxis linear layers."""

from absl import logging
from praxis import pax_fiddle
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from lingvo.core import layers as lingvo_layers
import numpy as np
from flax import linen as nn
from praxis import base_hyperparams
from praxis import base_layer
from praxis import py_utils
from praxis import pytypes
from praxis import test_utils
from praxis.layers import activations
from praxis.layers import linears
import tensorflow.compat.v2 as tf

to_np = test_utils.to_np
to_tf_nmap = test_utils.to_tf_nmap
instantiate = base_layer.instantiate

JTensor = pytypes.JTensor


class LinearsTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)
    tf.random.set_seed(123)

  @parameterized.named_parameters(
      {
          'testcase_name': 'ReLU',
          'activation_tpl': pax_fiddle.Config(activations.ReLU),
          'lingvo_activation_name': 'RELU',
      },
      {
          'testcase_name': 'Tanh',
          'activation_tpl': pax_fiddle.Config(activations.Tanh),
          'lingvo_activation_name': 'TANH',
      },
      {
          'testcase_name': 'ReLU6',
          'activation_tpl': pax_fiddle.Config(activations.ReLU6),
          'lingvo_activation_name': 'RELU6',
      },
      {
          'testcase_name': 'Sigmoid',
          'activation_tpl': pax_fiddle.Config(activations.Sigmoid),
          'lingvo_activation_name': 'SIGMOID',
      },
      {
          'testcase_name': 'Identity',
          'activation_tpl': pax_fiddle.Config(activations.Identity),
          'lingvo_activation_name': 'NONE',
      },
  )
  def test_feedforward_layer(self, activation_tpl, lingvo_activation_name):
    p = pax_fiddle.Config(
        linears.FeedForward,
        name='jax_ffn',
        input_dims=3,
        output_dims=20,
        activation_tpl=activation_tpl.clone(),
    )
    ffn = instantiate(p)
    npy_input = np.random.normal(1.0, 0.5, [10, 10, p.input_dims]).astype(
        'float32'
    )
    inputs = jnp.asarray(npy_input)
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = ffn.init(prng_key, inputs)
    outputs = ffn.apply(initial_vars, inputs)
    logging.info('initial_vars in ffn = %s', initial_vars)
    # Test whether tf projection layer returns same output
    # Modify initial_vars to use TF compatible params
    initial_vars = py_utils.NestedMap.FromNestedDict(initial_vars['params'])
    tf_initial_vars = py_utils.NestedMap()
    tf_initial_vars.w = initial_vars.linear.w
    tf_initial_vars.b = initial_vars.bias.b
    tf_initial_vars = to_tf_nmap(tf_initial_vars)
    tf_p = lingvo_layers.ProjectionLayer.Params().Set(
        name='tf_ffn',
        input_dim=p.input_dims,
        output_dim=p.output_dims,
        batch_norm=False,
        has_bias=True,
        activation=lingvo_activation_name,
    )
    tf_ffn = tf_p.Instantiate()
    tf_output = tf_ffn.FProp(
        tf_initial_vars, tf.constant(inputs, dtype=tf.float32)
    )
    np_outputs = to_np(outputs)
    tf_np_outputs = to_np(tf_output)
    self.assertAllClose(tf_np_outputs, np_outputs, atol=1e-6)

  def test_mlp_block_activate_final(self):
    ffn_p = pax_fiddle.Config(
        linears.FeedForward,
        name='jax_ffn',
        input_dims=3,
        output_dims=20,
        activation_tpl=pax_fiddle.Config(activations.Tanh),
    )
    mlp_p_activate = pax_fiddle.Config(
        linears.MLPBlock, name='jax_mlp', activate_final=True, ff_tpl=ffn_p
    )
    mlp_activate = instantiate(mlp_p_activate)
    mlp_p_no_activate = mlp_p_activate.clone().set(activate_final=False)
    mlp_no_activate = instantiate(mlp_p_no_activate)

    npy_input = np.random.normal(1.0, 0.5, [10, 10, ffn_p.input_dims]).astype(
        'float32'
    )
    inputs = jnp.asarray(npy_input)
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = mlp_activate.init(prng_key, inputs)
    outputs_activate = mlp_activate.apply(initial_vars, inputs)
    outputs_no_activate = mlp_no_activate.apply(initial_vars, inputs)
    self.assertAllClose(outputs_activate, np.tanh(outputs_no_activate))

  @parameterized.named_parameters(
      {
          'testcase_name': 'ReLU',
          'activation_tpl': pax_fiddle.Config(activations.ReLU),
          'lingvo_activation_name': 'RELU',
      },
      {
          'testcase_name': 'Tanh',
          'activation_tpl': pax_fiddle.Config(activations.Tanh),
          'lingvo_activation_name': 'TANH',
      },
      {
          'testcase_name': 'ReLU6',
          'activation_tpl': pax_fiddle.Config(activations.ReLU6),
          'lingvo_activation_name': 'RELU6',
      },
      {
          'testcase_name': 'Sigmoid',
          'activation_tpl': pax_fiddle.Config(activations.Sigmoid),
          'lingvo_activation_name': 'SIGMOID',
      },
      {
          'testcase_name': 'Identity',
          'activation_tpl': pax_fiddle.Config(activations.Identity),
          'lingvo_activation_name': 'NONE',
      },
  )
  def test_feedforward_layer_weight_init(
      self, activation_tpl, lingvo_activation_name
  ):
    p = pax_fiddle.Config(
        linears.FeedForward,
        name='jax_ffn',
        input_dims=3,
        output_dims=20,
        linear_tpl=pax_fiddle.Config(
            linears.Linear, weight_init=base_layer.WeightInit.Xavier(scale=1.0)
        ),
        activation_tpl=activation_tpl.clone(),
    )
    ffn = instantiate(p)
    npy_input = np.random.normal(1.0, 0.5,
                                 [10, 10, p.input_dims]).astype('float32')
    inputs = jnp.asarray(npy_input)
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = ffn.init(prng_key, inputs)
    outputs = ffn.apply(initial_vars, inputs)
    logging.info('initial_vars in ffn = %s', initial_vars)
    # Test whether tf projection layer returns same output
    # Modify initial_vars to use TF compatible params
    initial_vars = py_utils.NestedMap.FromNestedDict(initial_vars['params'])
    tf_initial_vars = py_utils.NestedMap()
    tf_initial_vars.w = initial_vars.linear.w
    tf_initial_vars.b = initial_vars.bias.b
    tf_initial_vars = to_tf_nmap(tf_initial_vars)
    tf_p = lingvo_layers.ProjectionLayer.Params().Set(
        name='tf_ffn',
        input_dim=p.input_dims,
        output_dim=p.output_dims,
        batch_norm=False,
        has_bias=True,
        activation=lingvo_activation_name)
    tf_ffn = tf_p.Instantiate()
    tf_output = tf_ffn.FProp(tf_initial_vars,
                             tf.constant(inputs, dtype=tf.float32))
    np_outputs = to_np(outputs)
    tf_np_outputs = to_np(tf_output)
    self.assertAllClose(tf_np_outputs, np_outputs, atol=1e-6)

  @parameterized.named_parameters(
      {
          'testcase_name': 'BiasZero',
          'activation_tpl': pax_fiddle.Config(activations.Identity),
          'lingvo_activation_name': 'NONE',
          'bias_init': 0.,
      },
      {
          'testcase_name': 'BiasOne',
          'activation_tpl': pax_fiddle.Config(activations.Identity),
          'lingvo_activation_name': 'NONE',
          'bias_init': 1.,
      },
  )
  def test_feedforward_layer_bias_init(
      self, activation_tpl, lingvo_activation_name, bias_init
  ):
    p = pax_fiddle.Config(
        linears.FeedForward,
        name='jax_ffn',
        input_dims=3,
        output_dims=20,
        linear_tpl=pax_fiddle.Config(
            linears.Linear, weight_init=base_layer.WeightInit.Xavier(scale=1.0)
        ),
        bias_tpl=pax_fiddle.Config(
            linears.Bias, bias_init=bias_init,
        ),
        activation_tpl=activation_tpl.clone(),
    )
    ffn = instantiate(p)
    npy_input = np.random.normal(1.0, 0.5,
                                 [10, 10, p.input_dims]).astype('float32')
    inputs = jnp.asarray(npy_input)
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = ffn.init(prng_key, inputs)
    outputs = ffn.apply(initial_vars, inputs)
    logging.info('initial_vars in ffn = %s', initial_vars)
    # Test whether tf projection layer returns same output
    # Modify initial_vars to use TF compatible params
    initial_vars = py_utils.NestedMap.FromNestedDict(initial_vars['params'])
    tf_initial_vars = py_utils.NestedMap()
    tf_initial_vars.w = initial_vars.linear.w
    tf_initial_vars.b = initial_vars.bias.b
    tf_initial_vars = to_tf_nmap(tf_initial_vars)
    tf_p = lingvo_layers.ProjectionLayer.Params().Set(
        name='tf_ffn',
        input_dim=p.input_dims,
        output_dim=p.output_dims,
        batch_norm=False,
        has_bias=True,
        activation=lingvo_activation_name)
    tf_ffn = tf_p.Instantiate()
    tf_output = tf_ffn.FProp(tf_initial_vars,
                             tf.constant(inputs, dtype=tf.float32))
    np_outputs = to_np(outputs)
    tf_np_outputs = to_np(tf_output)
    self.assertAllClose(tf_np_outputs, np_outputs, atol=1e-6)

  @parameterized.named_parameters(
      {
          'testcase_name': 'ReLU',
          'activation_tpl': pax_fiddle.Config(activations.ReLU),
          'lingvo_activation_name': 'RELU',
      },
      {
          'testcase_name': 'Tanh',
          'activation_tpl': pax_fiddle.Config(activations.Tanh),
          'lingvo_activation_name': 'TANH',
      },
      {
          'testcase_name': 'ReLU6',
          'activation_tpl': pax_fiddle.Config(activations.ReLU6),
          'lingvo_activation_name': 'RELU6',
      },
      {
          'testcase_name': 'Sigmoid',
          'activation_tpl': pax_fiddle.Config(activations.Sigmoid),
          'lingvo_activation_name': 'SIGMOID',
      },
      {
          'testcase_name': 'Identity',
          'activation_tpl': pax_fiddle.Config(activations.Identity),
          'lingvo_activation_name': 'NONE',
      },
  )
  def test_feedforward_layer_no_bias(self, activation_tpl,
                                     lingvo_activation_name):
    p = pax_fiddle.Config(
        linears.FeedForward,
        name='jax_ffn',
        input_dims=3,
        output_dims=20,
        has_bias=False,
        activation_tpl=activation_tpl.clone(),
    )
    ffn = instantiate(p)
    npy_input = np.random.normal(1.0, 0.5,
                                 [10, 10, p.input_dims]).astype('float32')
    inputs = jnp.asarray(npy_input)
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = ffn.init(prng_key, inputs)
    outputs = ffn.apply(initial_vars, inputs)
    logging.info('initial_vars in ffn = %s', initial_vars)
    # Test whether tf projection layer returns same output
    # Modify initial_vars to use TF compatible params
    initial_vars = py_utils.NestedMap.FromNestedDict(initial_vars['params'])
    tf_initial_vars = py_utils.NestedMap()
    tf_initial_vars.w = initial_vars.linear.w
    tf_initial_vars = to_tf_nmap(tf_initial_vars)
    tf_p = lingvo_layers.ProjectionLayer.Params().Set(
        name='tf_ffn',
        input_dim=p.input_dims,
        output_dim=p.output_dims,
        batch_norm=False,
        has_bias=False,
        activation=lingvo_activation_name)
    tf_ffn = tf_p.Instantiate()
    tf_output = tf_ffn.FProp(tf_initial_vars,
                             tf.constant(inputs, dtype=tf.float32))
    np_outputs = to_np(outputs)
    tf_np_outputs = to_np(tf_output)
    self.assertAllClose(tf_np_outputs, np_outputs, atol=1e-6)

  def test_feedforward_post_init_hparams(self):
    p = pax_fiddle.Config(
        linears.FeedForward,
        name='jax_ffn',
        input_dims=3,
        output_dims=20,
        has_bias=True,
        activation_tpl=pax_fiddle.Config(activations.ReLU),
    )
    ffn = instantiate(p)

    hyper_params = ffn.abstract_init_with_mdl_config(jnp.zeros((1, 3)))
    # This is the actual value of input_dims and output_dims, not the default
    # values.
    self.assertEqual(3, hyper_params['linear']['_hparams'].input_dims)
    self.assertEqual(20, hyper_params['linear']['_hparams'].output_dims)

    logging.info('hyper_params: \n%s',
                 base_hyperparams.nested_struct_to_text(hyper_params))

    params_inits = ffn.abstract_init_with_metadata(jnp.zeros((1, 3)))
    logging.info('params_inits: \n%s',
                 base_hyperparams.nested_struct_to_text(params_inits))

  def test_einsum_injection(self):
    class CustomEinsum(base_layer.BaseLayer):

      @nn.compact
      def __call__(self, equation, lhs, rhs):
        self.create_variable(
            'mult',
            base_layer.WeightHParams(
                shape=[1],
                init=base_layer.WeightInit.Constant(2.0),
            ),
            trainable=False,
        )
        mult = self.get_var('mult')
        self.update_var('mult', mult * 2.0)

        lhs_stats = self.create_variable(
            'lhs_stats',
            var_hparams=base_layer.WeightHParams(
                shape=(lhs.shape),
                init=base_layer.WeightInit.Constant(0.0),
            ),
            trainable=False,
        )

        def dg(*args, **kwargs):
          return jax.lax.dot_general(*args, **kwargs) * mult

        return jnp.einsum(equation, lhs + lhs_stats, rhs, _dot_general=dg)

    def run(custom_einsum_tpl, expected_shapes):
      p = pax_fiddle.Config(
          linears.Linear,
          name='jax_ffn',
          input_dims=10,
          output_dims=20,
      )
      if custom_einsum_tpl:
        p.set(einsum_tpl=custom_einsum_tpl)

      ffn = instantiate(p)
      inputs = jnp.ones((4, 10))
      initial_vars = ffn.init(
          {
              'params': jax.random.PRNGKey(seed=123),
              'random': jax.random.PRNGKey(seed=123),
          },
          inputs,
      )
      vars_shapes = jax.tree_util.tree_map(jnp.shape, initial_vars)
      self.assertEqual(vars_shapes, expected_shapes)
      v1, new_vars = ffn.apply(
          initial_vars,
          inputs,
          rngs={'random': jax.random.PRNGKey(seed=123)},
          mutable=True,
      )
      v2, _ = ffn.apply(
          new_vars,
          inputs,
          rngs={'random': jax.random.PRNGKey(seed=123)},
          mutable=True,
      )
      return v1, v2

    expected_shapes_original = {
        'params': {'w': (10, 20)},
    }

    expected_shapes_new = {
        'non_trainable': {
            'einsum': {
                'mult': (1,),
                'lhs_stats': (4, 10),
            }
        },
        'params': {'w': (10, 20)},
    }

    output1a, output1b = run(None, expected_shapes_original)
    einsum_tpl = pax_fiddle.Config(CustomEinsum)
    output2a, output2b = run(einsum_tpl, expected_shapes_new)
    # We can use exact equality because in floats division by 2.0 does not
    # have a rounding error.
    self.assertAllClose(output1a, output1b, atol=0.0)
    self.assertAllClose(output1a, output2a / 2.0, atol=0.0)
    self.assertAllClose(output1a, output2b / 4.0, atol=0.0)


class StackingOverTimeLayerTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  @parameterized.named_parameters(
      {
          'testcase_name': 'pad_with_left_frame',
          'pad_with_left_frame': True
      },
      {
          'testcase_name': 'pad_with_zeros',
          'pad_with_left_frame': False
      },
  )
  def testStackingOverTimeFProp(self, pad_with_left_frame):
    p = pax_fiddle.Config(linears.StackingOverTime)
    p.name = 'stackingOverTime'
    p.left_context = 2
    p.right_context = 0
    p.stride = 2
    p.pad_with_left_frame = pad_with_left_frame

    stacker = instantiate(p)
    self.assertEqual(stacker.window_size, 3)

    inputs = jnp.array([[[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]],
                        [[7, 7], [8, 8], [0, 0], [0, 0], [0, 0], [0, 0]]],
                       dtype=jnp.float32)
    paddings = jnp.array(
        [[[0], [0], [0], [0], [0], [0]], [[0], [0], [1], [1], [1], [1]]],
        dtype=jnp.float32)

    stacker_vars = stacker.init(jax.random.PRNGKey(123), inputs, paddings)
    outputs, output_paddings = stacker.apply(stacker_vars, inputs, paddings)
    print(f'{outputs}')
    if pad_with_left_frame:
      expected_outputs = jnp.array([
          [[1, 1, 1, 1, 1, 1], [1, 1, 2, 2, 3, 3], [3, 3, 4, 4, 5, 5]],
          [[7, 7, 7, 7, 7, 7], [7, 7, 8, 8, 0, 0], [0, 0, 0, 0, 0, 0]],
      ],
                                   dtype=jnp.float32)
    else:
      expected_outputs = jnp.array([
          [[0, 0, 0, 0, 1, 1], [1, 1, 2, 2, 3, 3], [3, 3, 4, 4, 5, 5]],
          [[0, 0, 0, 0, 7, 7], [7, 7, 8, 8, 0, 0], [0, 0, 0, 0, 0, 0]],
      ],
                                   dtype=jnp.float32)
    self.assertAllClose(expected_outputs, outputs)

    expected_output_paddings = jnp.array([[[0], [0], [0]], [[0], [0], [1]]],
                                         dtype=jnp.float32)
    self.assertAllClose(expected_output_paddings, output_paddings)

  @parameterized.named_parameters(
      {
          'testcase_name': 'pad_with_right_frame',
          'pad_with_right_frame': True
      },
      {
          'testcase_name': 'pad_with_zeros',
          'pad_with_right_frame': False
      },
  )
  def testStackingOverTimePadWithRightFrameFProp(self, pad_with_right_frame):
    p = pax_fiddle.Config(linears.StackingOverTime)
    p.name = 'stackingOverTime'
    p.left_context = 0
    p.right_context = 1
    p.stride = 2
    p.pad_with_right_frame = pad_with_right_frame

    stacker = instantiate(p)
    self.assertEqual(stacker.window_size, 2)

    # input shape [2, 5, 2]
    inputs = jnp.array([[[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
                        [[7, 7], [8, 8], [0, 0], [0, 0], [0, 0]]],
                       dtype=jnp.float32)
    paddings = jnp.array([[[0], [0], [0], [0], [0]], [[0], [0], [1], [1], [1]]],
                         dtype=jnp.float32)
    prng_key = jax.random.PRNGKey(seed=123)
    stacker_vars = stacker.init(prng_key, inputs, paddings)
    outputs, output_paddings = stacker.apply(stacker_vars, inputs, paddings)
    print(f'{outputs}')

    if pad_with_right_frame:
      # output shape [2, 3, 4]
      # [5, 5] is duplication of the last input frame.
      expected_outputs = jnp.array([
          [[1, 1, 2, 2], [3, 3, 4, 4], [5, 5, 5, 5]],
          [[7, 7, 8, 8], [0, 0, 0, 0], [0, 0, 0, 0]],
      ],
                                   dtype=jnp.float32)
    else:
      expected_outputs = jnp.array([
          [[1, 1, 2, 2], [3, 3, 4, 4], [5, 5, 0, 0]],
          [[7, 7, 8, 8], [0, 0, 0, 0], [0, 0, 0, 0]],
      ],
                                   dtype=jnp.float32)

    self.assertAllClose(expected_outputs, outputs)

    expected_output_paddings = jnp.array([[[0], [0], [0]], [[0], [1], [1]]],
                                         dtype=jnp.float32)
    self.assertAllClose(expected_output_paddings, output_paddings)

  def testStackingOverTimeFPropReduceMaxPadding(self):
    p = pax_fiddle.Config(linears.StackingOverTime)
    p.name = 'stackingOverTime'
    p.left_context = 2
    p.right_context = 0
    p.stride = 2
    p.padding_reduce_option = 'reduce_max'

    stacker = instantiate(p)
    self.assertEqual(stacker.window_size, 3)

    inputs = jnp.array([[[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]],
                        [[7, 7], [8, 8], [0, 0], [0, 0], [0, 0], [0, 0]]],
                       dtype=jnp.float32)
    paddings = jnp.array(
        [[[0], [0], [0], [0], [0], [0]], [[0], [0], [1], [1], [1], [1]]],
        dtype=jnp.float32)

    prng_key = jax.random.PRNGKey(seed=123)
    stacker_vars = stacker.init(prng_key, inputs, paddings)
    outputs, output_paddings = stacker.apply(stacker_vars, inputs, paddings)
    print(f'{outputs}')
    expected_outputs = jnp.array([
        [[0, 0, 0, 0, 1, 1], [1, 1, 2, 2, 3, 3], [3, 3, 4, 4, 5, 5]],
        [[0, 0, 0, 0, 7, 7], [7, 7, 8, 8, 0, 0], [0, 0, 0, 0, 0, 0]],
    ],
                                 dtype=jnp.float32)

    self.assertAllClose(expected_outputs, outputs)

    expected_output_paddings = jnp.array([[[1], [0], [0]], [[1], [1], [1]]],
                                         dtype=jnp.float32)
    self.assertAllClose(expected_output_paddings, output_paddings)

  def testStackingOverTimeFProp2(self):
    p = pax_fiddle.Config(linears.StackingOverTime)
    p.name = 'stackingOverTime'
    p.left_context = 0
    p.right_context = 1
    p.stride = 2

    stacker = instantiate(p)
    self.assertEqual(stacker.window_size, 2)

    inputs = np.random.normal(size=[2, 21, 16])
    # poor man's tf.sequence_mask in np.
    mask = np.zeros([2, 21]).astype(np.float32)
    mask[0, :9] = 1.
    mask[1, :14] = 1.

    paddings = 1.0 - mask
    paddings = jnp.expand_dims(paddings, -1)
    prng_key = jax.random.PRNGKey(seed=123)
    stacker_vars = stacker.init(prng_key, inputs, paddings)
    outputs, output_paddings = stacker.apply(stacker_vars, inputs, paddings)

    # length
    self.assertAllClose(
        np.array([5, 7], dtype=np.float32), np.sum(1.0 - output_paddings,
                                                   (1, 2)))
    # input and output sums are equal
    self.assertAllClose(np.sum(inputs, (1, 2)), np.sum(outputs, (1, 2)))

  def testStackingOverTimeIdentityFProp(self):
    p = pax_fiddle.Config(linears.StackingOverTime)
    p.name = 'stackingOverTime'
    p.left_context = 0
    p.right_context = 0
    p.stride = 1

    stacker = instantiate(p)
    self.assertEqual(stacker.window_size, 1)
    inputs = jnp.array([[[1], [2], [3], [4], [5]]], dtype=jnp.float32)
    paddings = jnp.zeros([1, 5, 1], dtype=jnp.float32)

    prng_key = jax.random.PRNGKey(seed=123)
    stacker_vars = stacker.init(prng_key, inputs, paddings)
    outputs, output_paddings = stacker.apply(stacker_vars, inputs, paddings)
    print(f'{outputs}')
    expected_outputs = jnp.array([[[1], [2], [3], [4], [5]]], dtype=jnp.float32)
    self.assertAllClose(expected_outputs, outputs)
    expected_output_paddings = jnp.array([[[0], [0], [0], [0], [0]]],
                                         dtype=jnp.float32)
    self.assertAllClose(expected_output_paddings, output_paddings)

  def _testUnstack(self, inputs, **kwargs):
    p = pax_fiddle.Config(
        linears.StackingOverTime, name='stackingOverTime', **kwargs
    )

    stacker = instantiate(p)
    prng_key = jax.random.PRNGKey(seed=123)
    stacker_vars = stacker.init(prng_key, inputs)
    stacked, _ = stacker.apply(stacker_vars, inputs)
    unstacked = stacker.apply(stacker_vars, stacked, method=stacker.unstack)
    print(f'{unstacked}')

    batch, input_length, depth = inputs.shape
    stacked_length = stacked.shape[1]
    stride = stacker.stride
    right_context = stacker.right_context

    self.assertAllClose(
        unstacked.shape,
        [batch, (stacked_length - 1) * stride + right_context + 1, depth])
    if right_context + 1 >= stride:
      self.assertGreaterEqual(unstacked.shape[1], input_length)
      self.assertAllClose(inputs, unstacked[:, :input_length])
    else:
      self.assertLessEqual(unstacked.shape[1], input_length)
      # The final up to stride - right_context - 1 values are missing.
      self.assertLessEqual(input_length - unstacked.shape[1],
                           stride - right_context - 1)
      self.assertAllClose(inputs[:, :unstacked.shape[1]], unstacked)

  def testStackingOverTimeUnstack(self):
    batch_size = 2
    length = 7
    depth = 3
    inputs = jnp.reshape(
        jnp.arange(batch_size * length * depth, dtype=jnp.float32),
        [batch_size, length, depth])
    self._testUnstack(inputs, left_context=2, stride=1)
    with self.assertRaises(ValueError):
      self._testUnstack(inputs, stride=2)
    self._testUnstack(inputs, stride=2, right_context=3)
    self._testUnstack(inputs, left_context=2, stride=3)
    self._testUnstack(inputs, stride=4, right_context=3)
    self._testUnstack(inputs, stride=4, left_context=1, right_context=2)


class Linear(base_layer.BaseLayer):
  output_dims: int = 0

  @nn.compact
  def __call__(self, inputs: JTensor) -> JTensor:
    self.create_variable(
        'w',
        base_layer.WeightHParams(
            shape=[inputs.shape[-1], self.output_dims],
            init=self.params_init
        ),
    )
    return jnp.einsum('...y,yz->...z', inputs, self.theta.w)


class Bias(base_layer.BaseLayer):
  bias_init: float | None = 0.0

  @nn.compact
  def __call__(self, inputs: JTensor) -> JTensor:
    self.create_variable(
        'b',
        base_layer.WeightHParams(
            shape=[inputs.shape[-1]],
            init=base_layer.WeightInit.Constant(self.bias_init),
        ),
    )
    return inputs + self.theta.b


class FeedForward(base_layer.BaseLayer):
  output_dims: int = 0
  has_bias: bool = True
  activation: activations.BaseActivation = activations.ReLU()
  bias_init: float | None = 0.0

  @nn.compact
  def __call__(self, inputs: JTensor) -> JTensor:
    linear = Linear(name='linear', output_dims=self.output_dims)
    projected_inputs = linear(inputs)
    if self.has_bias:
      bias = Bias(name='bias', bias_init=self.bias_init)
      projected_inputs += bias(projected_inputs)
    output = self.activation(projected_inputs)
    return output


class PostInitParamsTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  def test_fetch_post_init_params(self):
    batch, time, dim = 1, 2, 3
    x = jnp.ones((batch, time, dim))
    ffwd = FeedForward(name='ffw', output_dims=4 * dim)
    v = ffwd.abstract_init_with_mdl_config(x)
    self.assertEqual(Bias, v['bias']['_hparams'].cls)

  def test_auto_param_inheritance(self):
    batch, time, dim = 1, 2, 3
    x = jnp.ones((batch, time, dim))
    ffwd = FeedForward(
        name='ffw',
        output_dims=4 * dim,
        dtype=jnp.float64,
        fprop_dtype=jnp.float64,
        dcn_mesh_shape=(2,),
        ici_mesh_shape=(1,),
        mesh_axis_names=('name',),
        params_init=base_layer.WeightInit.Gaussian(2.0),
    )
    v = ffwd.abstract_init_with_mdl_config(x)
    # default initialization is properly propagated through to the children
    # layers.
    self.assertEqual(2.0, v['linear']['_hparams'].params_init.scale)
    self.assertEqual(2.0, v['bias']['_hparams'].params_init.scale)
    self.assertEqual(jnp.float64, v['linear']['_hparams'].dtype)
    self.assertEqual(jnp.float64, v['linear']['_hparams'].fprop_dtype)

  def test_instance_field(self):
    class ChildLayer(base_layer.BaseLayer):
      pass

      def __call__(self):
        return 0.0

    class ParentLayer(base_layer.BaseLayer):
      # instance fields:
      a: base_layer.BaseLayer = base_layer.instance_field(ChildLayer)

      def __call__(self):
        self.a()
        return 0

    @pax_fiddle.auto_config
    def make_model():
      return ParentLayer(
          dtype=jnp.int64,
          fprop_dtype=jnp.bfloat16,
          ici_mesh_shape=(1,),
          dcn_mesh_shape=(2,),
          params_init=base_layer.WeightInit.Gaussian(2.0),
          a=ParentLayer(
              ici_mesh_shape=(3,),
          ),
      )

    mdl_config = make_model.as_buildable()
    print('mdl_config', mdl_config)
    model = mdl_config.Instantiate()
    configs = model.abstract_init_with_mdl_config()
    print('post_init_config', configs)

    self.assertEqual(2.0, configs['a']['a']['_hparams'].params_init.scale)
    self.assertEqual(jnp.int64, configs['a']['a']['_hparams'].dtype)
    self.assertEqual(jnp.bfloat16, configs['a']['a']['_hparams'].fprop_dtype)

  def test_inline_instantiation(self):
    class L0(base_layer.BaseLayer):

      def setup(self):
        self.create_variable('x', base_layer.WeightHParams(shape=[128, 1280]))

      def __call__(self):
        return (
            jnp.zeros([1]).astype(self.dtype),
            jnp.zeros([1]).astype(self.fprop_dtype),
            self.theta.x,
        )

    class L1(base_layer.BaseLayer):

      def setup(self):
        self.a = L0()

      @nn.compact
      def __call__(self):
        b = L0(name='b')
        b_o = b()
        a_o = self.a()
        return [
            a_o[0] + b_o[0],
            a_o[1] + b_o[1],
            a_o[2] + b_o[2],
        ]

    class L2(base_layer.BaseLayer):

      def setup(self):
        self.c = L1()

      def __call__(self):
        return self.c()

    l2 = L2(
        dtype=jnp.bfloat16,
        fprop_dtype=jnp.float16,
        params_init=base_layer.WeightInit.Gaussian(3.0),
    )

    configs = l2.abstract_init_with_mdl_config()
    print('post_init_config', configs)

    # configs are properly propagated down.
    self.assertEqual(3.0, configs['c']['a']['_hparams'].params_init.scale)
    self.assertEqual(3.0, configs['c']['b']['_hparams'].params_init.scale)
    self.assertEqual(jnp.bfloat16, configs['c']['a']['_hparams'].dtype)
    self.assertEqual(jnp.float16, configs['c']['a']['_hparams'].fprop_dtype)
    self.assertEqual(jnp.bfloat16, configs['c']['b']['_hparams'].dtype)
    self.assertEqual(jnp.float16, configs['c']['b']['_hparams'].fprop_dtype)

    l2_init_vars = l2.init({'params': jax.random.PRNGKey(123456)})
    out = l2.apply(l2_init_vars)
    self.assertEqual(jnp.bfloat16, l2_init_vars['params']['c']['a']['x'].dtype)
    self.assertLess(
        abs(np.std(l2_init_vars['params']['c']['a']['x']) - 3.0), 0.01
    )
    self.assertEqual(jnp.bfloat16, out[0].dtype)
    self.assertEqual(jnp.float16, out[1].dtype)


if __name__ == '__main__':
  absltest.main()
