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
from jax.experimental import pjit
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
PARAMS = base_layer.PARAMS
NON_TRAINABLE = base_layer.NON_TRAINABLE
SUMMARIES = base_layer.SUMMARIES
AUX_LOSS = base_layer.AUX_LOSS
DECODE_CACHE = base_layer.DECODE_CACHE


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


class RepeatCalledTwice(base_layer.BaseLayer):

  def setup(self):
    sub_p = FeedForward.HParams(input_dim=2, output_dim=2)
    p = repeats.Repeat.HParams(
        name='repeated_ffn',
        sub_tpl=sub_p,
        x_times=3)

    self.create_child('repeated_ffn', p)

  def __call__(self, x):
    x = self.repeated_ffn(x)
    x = self.repeated_ffn(x)
    return x


class Decoder(base_layer.BaseLayer):
  """Decoder layer."""

  class HParams(BaseHParams):
    """Associated hyperparams for this layer class.

    Attributes:
      model_dim: Model dimension size.
    """
    model_dim: int = 0

  def setup(self):
    p = self.hparams
    assert p.name
    assert p.model_dim > 0

    self.create_variable(
        'w',
        WeightHParams(
            shape=[p.model_dim, p.model_dim], init=WeightInit.Gaussian(1.0)))

  def __call__(self, inputs):
    x = jnp.einsum('bty,yz->btz', inputs, self.theta.w)
    self.update_decode_state('x', x)
    x = jnp.cumsum(x, axis=1)
    out = jax.nn.sigmoid(x)
    return out

  def extend_step(self, step_inputs, t, seqlen):
    x = self.get_decode_state('x')
    x_t = jnp.einsum('by,yz->bz', step_inputs, self.theta.w)
    x.at[:, t].set(x_t)
    self.update_decode_state('x', x)
    x_t = jnp.sum(jnp.where(jnp.arange(seqlen)[:, None] > t, 0.0, x), axis=1)
    x_t = jax.nn.sigmoid(x_t)
    return x_t


class RepeatsTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  @parameterized.parameters((False,), (True,))
  def test_repeats(self, unpack_summaries):

    sub_p = FeedForward.HParams(input_dim=2, output_dim=2)
    p = repeats.Repeat.HParams(
        name='repeated_ffn',
        sub_tpl=sub_p,
        x_times=5,
        unpack_summaries=unpack_summaries)
    repeated_ffn = instantiate(p)

    k = jax.random.PRNGKey(123)
    k, input_random_key = jax.random.split(k)
    x = jax.random.uniform(input_random_key, shape=(4, 2))

    k, init_key = jax.random.split(k)
    weight_hparams = repeated_ffn.abstract_init_with_metadata(x)
    self.assertEqual(set(weight_hparams), {PARAMS, NON_TRAINABLE})
    self.assertEqual(weight_hparams[PARAMS]['sub']['w'].shape, [2, 2])
    self.assertEqual(weight_hparams[PARAMS]['sub']['w'].repeat_prefix, [5])
    self.assertEqual(
        weight_hparams[PARAMS]['sub']['w'].repeat_prefix_split_dims_mapping,
        (-1,))

    init_vars = repeated_ffn.init(init_key, x)
    init_vars_shape = jax.tree_map(lambda x: x.shape, init_vars)
    self.assertEqual(set(init_vars_shape), {PARAMS, NON_TRAINABLE})
    self.assertEqual(init_vars_shape[PARAMS]['sub']['w'], (5, 2, 2))
    self.assertEqual(init_vars_shape[NON_TRAINABLE]['sub']['step'], (5,))
    self.assertArraysEqual(init_vars[NON_TRAINABLE]['sub']['step'],
                           jnp.zeros((5,), dtype=jnp.int32))

    _, updated_vars = repeated_ffn.apply(
        init_vars, x, mutable=[NON_TRAINABLE, SUMMARIES, AUX_LOSS])
    self.assertArraysEqual(updated_vars[NON_TRAINABLE]['sub']['step'],
                           jnp.ones((5,), dtype=jnp.int32))

    # Ensure top level variables all exist with the right shape.
    updated_vars_shape = jax.tree_map(lambda x: x.shape, updated_vars)
    self.assertEqual(
        set(updated_vars_shape), {NON_TRAINABLE, SUMMARIES, AUX_LOSS})
    self.assertEqual(updated_vars_shape[NON_TRAINABLE]['sub']['step'], (5,))
    self.assertEqual(updated_vars_shape[AUX_LOSS]['sub']['z_loss'].value, ())
    self.assertEqual(updated_vars_shape[AUX_LOSS]['sub']['z_loss'].weight, ())
    self.assertEqual(updated_vars[AUX_LOSS]['sub']['z_loss'].value, 5.0)
    self.assertEqual(updated_vars[AUX_LOSS]['sub']['z_loss'].weight, 2.5)

    if unpack_summaries:
      self.assertEqual(
          updated_vars_shape[SUMMARIES]['sub']['inputs_mean_scalar'],
          [(1,), (1,), (1,), (1,), (1,)])
    else:
      self.assertEqual(
          updated_vars_shape[SUMMARIES]['sub']['inputs_mean_scalar'], (5,))

    print(jax.tree_map(lambda x: x.shape, updated_vars))

  @parameterized.parameters((False,), (True,))
  def test_extend_step(self, unroll):

    sub_p = Decoder.HParams(model_dim=4)
    p = repeats.Repeat.HParams(
        name='repeated_decoder',
        sub_tpl=sub_p,
        x_times=5,
        unroll_in_decode=unroll)
    repeated_decoder = instantiate(p)

    k = jax.random.PRNGKey(123)
    k, input_random_key = jax.random.split(k)
    x = jax.random.uniform(input_random_key, shape=(2, 6, 4))

    k, init_key = jax.random.split(k)
    init_vars = repeated_decoder.init(init_key, x)
    fprop_outs, updated_vars = repeated_decoder.apply(
        init_vars, x, mutable=[PARAMS, DECODE_CACHE])

    for t in range(6):
      step_out, updated_vars = repeated_decoder.apply(
          updated_vars,
          x[:, t],
          t,
          6,
          mutable=[PARAMS, DECODE_CACHE],
          method=repeated_decoder.extend_step)
      self.assertAllClose(step_out, fprop_outs[:, t])

  def test_repeat_called_twice(self):
    p = RepeatCalledTwice.HParams(name='repeat_called_twice')
    repeated_layer = instantiate(p)

    k = jax.random.PRNGKey(123)
    k, input_random_key = jax.random.split(k)
    x = jax.random.uniform(input_random_key, shape=(4, 2))
    k, init_key = jax.random.split(k)
    weight_hparams = repeated_layer.abstract_init_with_metadata(x)
    print('weight_hparams = ', weight_hparams)


class RepeatsQuantizeTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  def test_quantize_repeats(self):

    sub_p = FeedForward.HParams(input_dim=2, output_dim=2)
    p = repeats.Repeat.HParams(name='ffn', sub_tpl=sub_p, x_times=5)
    ffn = instantiate(p)

    inputs = np.random.normal(1.0, 1.5, [2, 2]).astype(np.float32)
    prng_key = jax.random.PRNGKey(seed=123)
    init_vars = ffn.init(prng_key, inputs)

    res, _ = ffn.apply(init_vars, mutable=[], method=ffn.quantize_weight)
    shapes = jax.tree_map(lambda x: x.shape, res)
    expected_shapes = {
        'non_trainable': {
            'sub': {
                'step': (5,)
            }
        },
        'params': {
            'sub': {
                'w': (5, 2, 2)
            }
        }
    }
    self.assertEqual(shapes, expected_shapes)

    pspecs, _ = ffn.apply(
        init_vars, mutable=[], method=ffn.quantized_partitioned_specs)
    expected_pspecs = {
        'non_trainable': {
            'sub': {
                'step':
                    base_layer.BoxedPartitionSpec(
                        meta=pjit.PartitionSpec(None,))
            }
        },
        'params': {
            'sub': {
                'w':
                    base_layer.BoxedPartitionSpec(
                        meta=pjit.PartitionSpec(None, None, None))
            }
        }
    }
    self.assertEqual(pspecs, expected_pspecs)

if __name__ == '__main__':
  absltest.main()
