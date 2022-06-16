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

    prng_key = jax.random.PRNGKey(1)
    initial_vars = foo_shared.init(prng_key)
    logging.info('initial_vars=%s', initial_vars)

    expected_shape = {
        'params': {
            'linear': {
                'w': (2, 2)
            },
            'linear_private': {
                'w': (2, 2)
            }
        }
    }
    self.assertEqual(
        jax.tree_map(lambda x: x.shape, initial_vars), expected_shape)

    inputs = jnp.ones((3, 2))
    output = foo_shared.apply(initial_vars, inputs)
    logging.info('output=%s', output)

    shared_sublayer_initial_vars = {
        'params': {
            'w': initial_vars['params']['linear']['w']
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
    self.assertEqual(2, hyper_params['linear']['_hparams'].input_dims)
    self.assertEqual(2, hyper_params['linear']['_hparams'].output_dims)

    logging.info('hyper_params: \n%s',
                 base_hyperparams.nested_struct_to_text(hyper_params))


if __name__ == '__main__':
  absltest.main()
