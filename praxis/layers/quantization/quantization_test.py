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

"""Tests for quantize_weight() method for more complicated cases."""

from absl.testing import absltest
import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import pax_fiddle
from praxis import test_utils
from praxis.layers import repeats
from praxis.layers.quantization import linears as qlinears
from praxis.layers.quantization import operations
from praxis.layers.quantization import quantization_hparams

instantiate = base_layer.instantiate
LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]
WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams


class RepeatsLinearQuantizeTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  def test_quantize_repeats(self):

    sub_p = pax_fiddle.Config(
        qlinears.Linear,
        name='_linear_q',
        input_dims=2,
        output_dims=2,
        quantization=quantization_hparams.QuantizationHParams(
            mode=quantization_hparams.QuantizationMode.TRAINING
        ),
    )
    p = pax_fiddle.Config(repeats.Repeat, name='ffn', sub_tpl=sub_p, x_times=3)
    ffn = instantiate(p)

    inputs = np.random.normal(1.0, 1.5, [2, 2]).astype(np.float32)
    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      init_vars = ffn.init(prng_key, inputs)
      shapes = jax.tree_map(lambda x: x.shape, init_vars)
      res, _ = ffn.apply(init_vars, mutable=[], method=ffn.quantize_weight)
    shapes = jax.tree_map(lambda x: x.shape, res)
    expected_shapes = {
        base_layer.PARAMS: {
            'sub': {
                'w': (3, 2, 2),
                'w_quantized_scale': (3, 2)
            }
        }
    }
    self.assertEqual(shapes, expected_shapes)
    rescaled = jnp.multiply(
        res[base_layer.PARAMS]['sub']['w'],
        jnp.expand_dims(res[base_layer.PARAMS]['sub']['w_quantized_scale'],
                        1).astype(jnp.float32))
    self.assertAllClose(
        rescaled,
        init_vars[base_layer.PARAMS]['sub']['w'],
        rtol=0.02,
        atol=0.02)


class FeedForwardQuant(base_layer.BaseLayer):
  """Feedforward layer with quantize_weight() method."""
  input_dim: int = 0
  output_dim: int = 0

  def setup(self):
    self.create_variable(
        'w',
        WeightHParams(
            shape=[self.input_dim, self.output_dim],
            init=WeightInit.Gaussian(1.0),
        ),
    )

  def __call__(self, inputs):
    return jnp.einsum('...y,yz->...z', inputs, self.theta.w)

  def quantize_weight(self):
    theta = self.theta
    eqn = 'xy,yz->xz'
    q_w, q_s, _ = operations.reduce_einsum_weight_precision(eqn, theta.w)
    scale_name = 'w' + base_layer.QUANTIZED_SCALE_NAME_POSTFIX
    return {base_layer.PARAMS: {'w': q_w, scale_name: q_s}}


class FeedForward(base_layer.BaseLayer):
  """Feedforward layer with default method."""
  input_dim: int = 0
  output_dim: int = 0

  def setup(self):
    self.create_variable(
        'w',
        WeightHParams(
            shape=[self.input_dim, self.output_dim],
            init=WeightInit.Gaussian(1.0),
        ),
    )
    self.create_variable(
        'b',
        WeightHParams(shape=[self.output_dim], init=WeightInit.Gaussian(1.0)),
    )

  def __call__(self, inputs):
    res = jnp.einsum('...y,yz->...z', inputs, self.theta.w)
    return jnp.add(res, self.theta.b)


class ParentLayer(base_layer.BaseLayer):
  ff1_tpl: LayerTpl = base_layer.template_field(FeedForwardQuant)
  ff2_tpl: LayerTpl = base_layer.template_field(FeedForward)

  def setup(self):
    self.create_child(
        'ff1', self.ff1_tpl.clone().set(name='ff1', input_dim=3, output_dim=2)
    )
    self.create_child(
        'ff2', self.ff2_tpl.clone().set(name='ff2', input_dim=2, output_dim=3)
    )

  def __call__(self, inputs):
    return self.ff2(self.ff1(inputs))


class ChildrenQuantizeTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  def test_quantize_parent(self):
    p = pax_fiddle.Config(ParentLayer, name='_parent_q')
    layer = instantiate(p)

    inputs = np.random.normal(1.5, 2.0, [5, 3]).astype(np.float32)
    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = layer.init(prng_key, inputs)
      res, _ = layer.apply(
          initial_vars, mutable=[], method=layer.quantize_weight
      )

    shapes = jax.tree_map(lambda x: x.shape, res)
    types = jax.tree_map(lambda x: x.dtype, res)
    self.assertEqual(
        shapes,
        {
            base_layer.PARAMS: {
                'ff1': {'w': (3, 2), 'w_quantized_scale': (2,)},
                'ff2': {'b': (3,), 'w': (2, 3)},
            }
        },
    )
    self.assertEqual(
        types,
        {
            base_layer.PARAMS: {
                'ff1': {'w': jnp.int8, 'w_quantized_scale': jnp.bfloat16},
                'ff2': {'b': jnp.float32, 'w': jnp.float32},
            }
        },
    )


if __name__ == '__main__':
  absltest.main()
