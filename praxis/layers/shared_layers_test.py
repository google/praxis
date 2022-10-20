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

"""Tests for layers with shared sub-layers."""

from typing import Optional

from absl import logging
from absl.testing import absltest
import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_hyperparams
from praxis import base_layer
from praxis import test_utils
from praxis.layers import linears

instantiate = base_layer.instantiate
BaseHParams = base_layer.BaseLayer.HParams
sub_config_field = base_layer.sub_config_field


class FooShared(base_layer.BaseLayer):
  linear1: Optional[linears.Linear] = None
  linear2: Optional[linears.Linear] = None

  class HParams(BaseHParams):
    linear_private_tpl: BaseHParams = sub_config_field(linears.Linear.HParams)

  def setup(self):
    p = self.hparams
    # Note submodule name must be unique.
    self.create_child('linear_private', p.linear_private_tpl)

  def __call__(self, x):
    x = self.linear1(x)
    x = self.linear2(x)
    x = self.linear_private(x)
    return x


class SharedLayersTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  def test_shared_sublayers(self):
    # Shared layers should be passed `config`.
    linear_cfg = linears.Linear.config(
        name='linear', input_dims=2, output_dims=2)
    # Private layers should be passed `Params`.
    linear_p = linears.Linear.HParams(
        name='linear', input_dims=2, output_dims=2)
    foo_shared_p = FooShared.config(
        name='foo_shared',
        linear1=linear_cfg,  # shared
        linear2=linear_cfg,  # shared
        linear_private_tpl=linear_p  # private
    )
    foo_shared = instantiate(foo_shared_p)

    inputs = jnp.ones((3, 2))

    prng_key = jax.random.PRNGKey(1)
    initial_vars = foo_shared.init(prng_key, inputs)
    logging.info('initial_vars=%s', initial_vars)

    # Note: foo_shared.linear1 has the name 'linear' before the Fiddle
    # migration, but the name changes to 'linear1' after the migration.
    # TODO(b/249483164): Replace linear_name w/ 'linear1' after migration.
    linear_name = 'linear' if 'linear' in initial_vars['params'] else 'linear1'

    expected_shape = {
        'params': {
            linear_name: {
                'w': (2, 2)
            },
            'linear_private': {
                'w': (2, 2)
            }
        }
    }
    self.assertEqual(
        jax.tree_map(lambda x: x.shape, initial_vars), expected_shape)

    output = foo_shared.apply(initial_vars, inputs)
    logging.info('output=%s', output)

    shared_sublayer_initial_vars = {
        'params': {
            'w': initial_vars['params'][linear_name]['w']
        }
    }
    private_sublayer_initial_vars = {
        'params': {
            'w': initial_vars['params']['linear_private']['w']
        }
    }
    linear_layer = instantiate(linear_p)
    expected_output = linear_layer.apply(shared_sublayer_initial_vars, inputs)
    expected_output = linear_layer.apply(shared_sublayer_initial_vars,
                                         expected_output)
    expected_output = linear_layer.apply(private_sublayer_initial_vars,
                                         expected_output)
    self.assertAllClose(output, expected_output)

    # Dump post_init_hparams
    def gen_post_init_hparams(prng_key):
      return foo_shared.apply({},
                              rngs={base_layer.PARAMS: prng_key},
                              method=foo_shared.post_init_hparams,
                              mutable=True)[1]

    variables_abstract = jax.eval_shape(gen_post_init_hparams, prng_key)
    assert base_layer.HYPER_PARAMS in variables_abstract

    hyper_params = jax.tree_map(
        lambda x: x.meta,
        variables_abstract[base_layer.HYPER_PARAMS],
        is_leaf=lambda x: isinstance(x, base_layer.WrappedHParams))

    # This is the actual value of input_dims and output_dims, not the default
    # values.
    self.assertEqual(2, hyper_params[linear_name]['_hparams'].input_dims)
    self.assertEqual(2,
                     hyper_params[linear_name]['_hparams'].output_dims)

    logging.info('hyper_params: \n%s',
                 base_hyperparams.nested_struct_to_text(hyper_params))


class SimpleShared01(base_layer.BaseLayer):
  """A layer to test weight sharing."""

  class HParams(BaseHParams):
    sub1_tpl: BaseHParams = sub_config_field(None)
    sub2_tpl: BaseHParams = sub_config_field(None)

  def setup(self) -> None:
    p = self.hparams
    self.create_child('sub1', p.sub1_tpl)
    self.create_child('sub2', p.sub2_tpl)

  def __call__(self, x_in):
    return self.sub2(self.sub1(x_in))


class SharedLayerTest(test_utils.TestCase):

  def testSharedLayer(self):
    sub_params = linears.FeedForward.HParams(input_dims=8, output_dims=8)
    # Share the entire FeedForward layer.
    sub_params.shared_weight_layer_id = 'shared_layer'
    test_layer_p = SimpleShared01.HParams(
        name='test', sub1_tpl=sub_params.clone(), sub2_tpl=sub_params.clone())
    x_in = jnp.ones([2, 8])
    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(1234)
      layer = base_layer.instantiate(test_layer_p)
      init_vars = layer.init(prng_key, x_in)
      dummy = jnp.ones([1])
      logging.info('SimpleShared01 initial_vars %s',
                   jax.tree_util.tree_map(lambda x: dummy, init_vars))
      # 'sub2' share the same weights as 'sub1'
      expected_vars_struct = {
          'params': {
              'shared_layer': {
                  'bias': {
                      'b': dummy
                  },
                  'linear': {
                      'w': dummy
                  }
              }
          }
      }
      self.assertEqual(
          jax.tree_util.tree_structure(expected_vars_struct),
          jax.tree_util.tree_structure(init_vars))
      # We share the linear and bias layer.
      # TODO(yonghui): check the shape of the shared vars.
      out1 = layer.apply(init_vars, x_in)
      logging.info('out1: %s', out1)
      # We can apply again.
      out2 = layer.apply(init_vars, out1)
      logging.info('out2: %s', out2)

  def testSharedTemplateLayer(self):
    sub_params = linears.FeedForward.HParams(input_dims=8, output_dims=8)
    # Only share the linear projection, not the entire FeedForward layer.
    sub_params.linear_tpl.shared_weight_layer_id = 'shared_weight'
    test_layer_p = SimpleShared01.HParams(
        name='test', sub1_tpl=sub_params.clone(), sub2_tpl=sub_params.clone())
    x_in = jnp.ones([2, 8])
    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(1234)
      layer = base_layer.instantiate(test_layer_p)
      init_vars = layer.init(prng_key, x_in)
      dummy = jnp.ones([1])
      logging.info('SimpleShared01 initial_vars %s',
                   jax.tree_util.tree_map(lambda x: dummy, init_vars))
      # 'sub2' share the same linear weights as 'sub1', but has its own bias var
      expected_vars_struct = {
          'params': {
              'shared_weight': {
                  'w': dummy
              },
              'sub1': {
                  'bias': {
                      'b': dummy
                  }
              },
              'sub2': {
                  'bias': {
                      'b': dummy
                  }
              }
          }
      }
      self.assertEqual(
          jax.tree_util.tree_structure(expected_vars_struct),
          jax.tree_util.tree_structure(init_vars))
      # We share the linear and bias layer.
      # TODO(yonghui): check the shape of the shared vars.
      out1 = layer.apply(init_vars, x_in)
      logging.info('out1: %s', out1)
      # We can apply again.
      out2 = layer.apply(init_vars, out1)
      logging.info('out2: %s', out2)

  def testRecursiveSharing(self):
    sub_params = linears.FeedForward.HParams(input_dims=8, output_dims=8)
    # Share the linear projection.
    sub_params.linear_tpl.shared_weight_layer_id = 'shared_linear'
    parent_p = SimpleShared01.HParams(
        sub1_tpl=sub_params.clone(), sub2_tpl=sub_params.clone())
    # Share parent nodes.
    parent_p.shared_weight_layer_id = 'shared'
    root_layer_p = SimpleShared01.HParams(
        name='root', sub1_tpl=parent_p.clone(), sub2_tpl=parent_p.clone())
    x_in = jnp.ones([2, 8])
    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(1234)
      layer = base_layer.instantiate(root_layer_p)
      init_vars = layer.init(prng_key, x_in)
      dummy = jnp.ones([1])
      logging.info('SimpleShared01 initial_vars %s',
                   jax.tree_util.tree_map(lambda x: dummy, init_vars))
      # 'sub2' share the same linear weights as 'sub1', but has its own bias var
      expected_vars_struct = {
          'params': {
              'shared': {
                  'sub1': {
                      'bias': {
                          'b': dummy
                      }
                  },
                  'sub2': {
                      'bias': {
                          'b': dummy
                      }
                  }
              },
              'shared_linear': {
                  'w': dummy
              }
          }
      }
      self.assertEqual(
          jax.tree_util.tree_structure(expected_vars_struct),
          jax.tree_util.tree_structure(init_vars))
      # We share the linear and bias layer.
      # TODO(yonghui): check the shape of the shared vars.
      out1 = layer.apply(init_vars, x_in)
      logging.info('out1: %s', out1)
      # We can apply again.
      out2 = layer.apply(init_vars, out1)
      logging.info('out2: %s', out2)


if __name__ == '__main__':
  absltest.main()
