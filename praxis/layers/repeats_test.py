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

"""Tests for Praxis repeats layers."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import py_utils
from praxis import test_utils
from praxis.layers import repeats

NestedMap = py_utils.NestedMap
WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams

BaseHParams = base_layer.BaseLayer.HParams
SplitDimsMapping = base_layer.SplitDimsMapping

instantiate = base_layer.instantiate


class FeedForward(base_layer.BaseLayer):
  """Feedforward layer."""

  class HParams(BaseHParams):
    """Associated hyperparams for this layer class.

    Attributes:
      input_dim: Input dimension size.
      output_dim: Output dimension size.
    """
    input_dim: int = 0
    output_dim: int = 0

  def setup(self):
    p = self.hparams
    assert p.name
    assert p.input_dim > 0
    assert p.output_dim > 0

    self.create_variable(
        'w',
        WeightHParams(
            shape=[p.input_dim, p.output_dim], init=WeightInit.Gaussian(1.0)))
    self.create_variable(
        'step',
        WeightHParams(shape=[], dtype=jnp.int32, init=WeightInit.Constant(0)),
        trainable=False)

  def __call__(self, inputs):
    self.add_summary('inputs_mean', jnp.mean(inputs))
    self.add_aux_loss('z_loss', 1, 0.5)
    self.update_var('step', self.get_var('step') + 1)
    out = jnp.einsum('...y,yz->...z', inputs, self.theta.w)
    out = jax.nn.sigmoid(out)
    return out


class RepeatsTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  @parameterized.parameters((False,), (True,))
  def test_repeats(self, unpack_summaries):

    sub_p = FeedForward.HParams(input_dim=2, output_dim=2)
    p = repeats.Repeat.HParams(
        name='repeated_ffn',
        sub=sub_p,
        x_times=5,
        unpack_summaries=unpack_summaries)
    repeated_ffn = instantiate(p)

    k = jax.random.PRNGKey(123)
    k, init_key = jax.random.split(k)
    init_vars = repeated_ffn.init(init_key)
    init_vars_shape = jax.tree_map(lambda x: x.shape, init_vars)
    self.assertEqual(set(init_vars_shape), {'params', 'non_trainable'})

    k, input_random_key = jax.random.split(k)
    x = jax.random.uniform(input_random_key, shape=(4, 2))

    _, updated_vars = repeated_ffn.apply(init_vars, x, mutable=True)
    self.assertArraysEqual(updated_vars['non_trainable']['sub']['step'],
                           jnp.ones((5,), dtype=jnp.int32))

    # Ensure top level variables all exist with the right shape.
    updated_vars_shape = jax.tree_map(lambda x: x.shape, updated_vars)
    self.assertEqual(
        set(updated_vars_shape),
        {'params', 'non_trainable', 'summaries', 'aux_loss'})
    self.assertEqual(updated_vars_shape['params']['sub']['w'], (5, 2, 2))
    self.assertEqual(updated_vars_shape['non_trainable']['sub']['step'], (5,))
    self.assertEqual(updated_vars_shape['aux_loss']['sub']['z_loss'].value, ())
    self.assertEqual(updated_vars_shape['aux_loss']['sub']['z_loss'].weight, ())
    self.assertEqual(updated_vars['aux_loss']['sub']['z_loss'].value, 5.0)
    self.assertEqual(updated_vars['aux_loss']['sub']['z_loss'].weight, 2.5)

    if unpack_summaries:
      self.assertEqual(
          updated_vars_shape['summaries']['sub']['inputs_mean_scalar'], [(1,),
                                                                         (1,),
                                                                         (1,),
                                                                         (1,),
                                                                         (1,)])
    else:
      self.assertEqual(
          updated_vars_shape['summaries']['sub']['inputs_mean_scalar'], (5,))


    print(jax.tree_map(lambda x: x.shape, updated_vars))


if __name__ == '__main__':
  absltest.main()
