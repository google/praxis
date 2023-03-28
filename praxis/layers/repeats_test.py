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

"""Tests for Praxis repeats layers."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import pax_fiddle
from praxis import py_utils
from praxis import test_utils
from praxis.layers import repeats
from praxis.layers.linears import FeedForward as ff
from praxis.layers.sparsity import linears as slinears
from praxis.layers.sparsity import sparsity_hparams


template_field = base_layer.template_field

NestedMap = py_utils.NestedMap
WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams
SplitDimsMapping = base_layer.SplitDimsMapping

instantiate = base_layer.instantiate
PARAMS = base_layer.PARAMS
NON_TRAINABLE = base_layer.NON_TRAINABLE
SUMMARIES = base_layer.SUMMARIES
AUX_LOSS = base_layer.AUX_LOSS
DECODE_CACHE = base_layer.DECODE_CACHE
INTERMEDIATES = base_layer.INTERMEDIATES

SparsityHParams = sparsity_hparams.SparsityHParams
WeightSparsityParams = sparsity_hparams.WeightSparsityParams
SparsityMode = sparsity_hparams.SparsityMode
SparsityType = sparsity_hparams.SparsityType


class FeedForward(base_layer.BaseLayer):
  """Feedforward layer.

  Attributes:
    input_dim: Input dimension size.
    output_dim: Output dimension size.
  """
  input_dim: int = 0
  output_dim: int = 0

  def setup(self):
    assert self.name
    assert self.input_dim > 0
    assert self.output_dim > 0

    self.create_variable(
        'w',
        WeightHParams(
            shape=[self.input_dim, self.output_dim],
            init=WeightInit.Gaussian(1.0),
        ),
    )
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


class FeedForwardWithPadding(FeedForward):
  """Feedforward layer with paddings in input."""

  def __call__(self, inputs, paddings):
    self.add_summary('inputs_mean', jnp.mean(inputs))
    self.add_aux_loss('z_loss', 1, 0.5)
    self.update_var('step', self.get_var('step') + 1)
    out = jnp.einsum('...y,yz->...z', inputs, self.theta.w)
    out = jax.nn.sigmoid(out)
    out = py_utils.apply_padding(out, paddings)
    return out, paddings


class RepeatCalledTwice(base_layer.BaseLayer):

  def setup(self):
    sub_p = pax_fiddle.Config(FeedForward, input_dim=2, output_dim=2)
    p = pax_fiddle.Config(
        repeats.Repeat, name='repeated_ffn', sub_tpl=sub_p, x_times=3
    )

    self.create_child('repeated_ffn', p)

  def __call__(self, x):
    x = self.repeated_ffn(x)
    x = self.repeated_ffn(x)
    return x


class Decoder(base_layer.BaseLayer):
  """Decoder layer.

  Attributes:
    model_dim: Model dimension size.
  """
  model_dim: int = 0

  def setup(self):
    assert self.name
    assert self.model_dim > 0

    self.create_variable(
        'w',
        WeightHParams(
            shape=[self.model_dim, self.model_dim],
            init=WeightInit.Gaussian(1.0),
        ),
    )

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
    sub_p = pax_fiddle.Config(FeedForwardWithPadding, input_dim=2, output_dim=2)
    p = pax_fiddle.Config(
        repeats.Repeat,
        name='repeated_ffn',
        positional_args_as_scan_carry=True,
        sub_tpl=sub_p,
        x_times=5,
        unpack_summaries=unpack_summaries,
    )
    repeated_ffn = instantiate(p)

    k = jax.random.PRNGKey(123)
    k, input_random_key = jax.random.split(k)
    x = jax.random.uniform(input_random_key, shape=(4, 2))
    paddings = jnp.ones((4, 2))

    k, init_key = jax.random.split(k)
    weight_hparams = repeated_ffn.abstract_init_with_metadata(x, paddings)
    self.assertEqual(set(weight_hparams), {PARAMS, NON_TRAINABLE})
    self.assertEqual(weight_hparams[PARAMS]['sub']['w'].shape, [2, 2])
    self.assertEqual(weight_hparams[PARAMS]['sub']['w'].repeat_prefix, [5])
    self.assertEqual(
        weight_hparams[PARAMS]['sub']['w'].repeat_prefix_split_dims_mapping,
        (-1,))

    init_vars = repeated_ffn.init(init_key, x, paddings)
    init_vars_shape = jax.tree_map(lambda x: x.shape, init_vars)
    self.assertEqual(set(init_vars_shape), {PARAMS, NON_TRAINABLE})
    self.assertEqual(init_vars_shape[PARAMS]['sub']['w'], (5, 2, 2))
    self.assertEqual(init_vars_shape[NON_TRAINABLE]['sub']['step'], (5,))
    self.assertArraysEqual(init_vars[NON_TRAINABLE]['sub']['step'],
                           jnp.zeros((5,), dtype=jnp.int32))

    _, updated_vars = repeated_ffn.apply(
        init_vars, x, paddings, mutable=[NON_TRAINABLE, SUMMARIES, AUX_LOSS]
    )
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
  def test_repeats_nd(self, unpack_summaries):

    sub_p = pax_fiddle.Config(FeedForward, input_dim=2, output_dim=2)
    p = pax_fiddle.Config(
        repeats.Repeat,
        name='repeated_ffn',
        sub_tpl=sub_p,
        x_times=6,
        nd_prefix_shape=(2, 3),
        unpack_summaries=unpack_summaries,
    )
    repeated_ffn = instantiate(p)

    k = jax.random.PRNGKey(123)
    k, input_random_key = jax.random.split(k)
    x = jax.random.uniform(input_random_key, shape=(4, 2))

    k, init_key = jax.random.split(k)
    weight_hparams = repeated_ffn.abstract_init_with_metadata(x)
    self.assertEqual(set(weight_hparams), {PARAMS, NON_TRAINABLE})
    self.assertEqual(weight_hparams[PARAMS]['sub']['w'].shape, [2, 2])
    self.assertEqual(weight_hparams[PARAMS]['sub']['w'].repeat_prefix, [2, 3])
    self.assertEqual(
        weight_hparams[PARAMS]['sub']['w'].repeat_prefix_split_dims_mapping,
        (-1, -1))

    init_vars = repeated_ffn.init(init_key, x)
    init_vars_shape = jax.tree_map(lambda x: x.shape, init_vars)
    self.assertEqual(set(init_vars_shape), {PARAMS, NON_TRAINABLE})
    self.assertEqual(init_vars_shape[PARAMS]['sub']['w'], (2, 3, 2, 2))
    self.assertEqual(init_vars_shape[NON_TRAINABLE]['sub']['step'], (2, 3,))
    self.assertArraysEqual(init_vars[NON_TRAINABLE]['sub']['step'],
                           jnp.zeros((2, 3,), dtype=jnp.int32))

    _, updated_vars = repeated_ffn.apply(
        init_vars, x, mutable=[NON_TRAINABLE, SUMMARIES, AUX_LOSS])
    self.assertArraysEqual(updated_vars[NON_TRAINABLE]['sub']['step'],
                           jnp.ones((2, 3,), dtype=jnp.int32))

    # Ensure top level variables all exist with the right shape.
    updated_vars_shape = jax.tree_map(lambda x: x.shape, updated_vars)
    self.assertEqual(
        set(updated_vars_shape), {NON_TRAINABLE, SUMMARIES, AUX_LOSS})
    self.assertEqual(updated_vars_shape[NON_TRAINABLE]['sub']['step'], (2, 3,))
    self.assertEqual(updated_vars_shape[AUX_LOSS]['sub']['z_loss'].value, ())
    self.assertEqual(updated_vars_shape[AUX_LOSS]['sub']['z_loss'].weight, ())
    self.assertEqual(updated_vars[AUX_LOSS]['sub']['z_loss'].value, 6.0)
    self.assertEqual(updated_vars[AUX_LOSS]['sub']['z_loss'].weight, 3.0)

    if unpack_summaries:
      self.assertEqual(
          updated_vars_shape[SUMMARIES]['sub']['inputs_mean_scalar'],
          [(1,), (1,), (1,), (1,), (1,), (1,)])
    else:
      self.assertEqual(
          updated_vars_shape[SUMMARIES]['sub']['inputs_mean_scalar'], (6,))

    print(jax.tree_map(lambda x: x.shape, updated_vars))

  @parameterized.parameters((False,), (True,))
  def test_extend_step(self, unroll):

    sub_p = pax_fiddle.Config(Decoder, model_dim=4)
    p = pax_fiddle.Config(
        repeats.Repeat,
        name='repeated_decoder',
        sub_tpl=sub_p,
        x_times=5,
        unroll_in_decode=unroll,
    )
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
    p = pax_fiddle.Config(RepeatCalledTwice, name='repeat_called_twice')
    repeated_layer = instantiate(p)

    k = jax.random.PRNGKey(123)
    k, input_random_key = jax.random.split(k)
    x = jax.random.uniform(input_random_key, shape=(4, 2))
    k, _ = jax.random.split(k)
    weight_hparams = repeated_layer.abstract_init_with_metadata(x)
    print('weight_hparams = ', weight_hparams)

  def test_capture_intermediate(self):
    sub_p = pax_fiddle.Config(FeedForward, input_dim=2, output_dim=2)
    p = pax_fiddle.Config(
        repeats.Repeat,
        name='repeated_ffn',
        sub_tpl=sub_p,
        x_times=3,
        collect_intermediate_outputs=True,
    )
    repeated_ffn = instantiate(p)

    k = jax.random.PRNGKey(123)
    k, input_random_key = jax.random.split(k)
    x = jax.random.uniform(input_random_key, shape=(4, 2))

    k, init_key = jax.random.split(k)
    init_vars = repeated_ffn.init(init_key, x)
    outputs, updated_vars = repeated_ffn.apply(
        init_vars, x, mutable=[INTERMEDIATES], capture_intermediates=True
    )
    intermediates = updated_vars[INTERMEDIATES]['repeat_intermediates'][0]
    self.assertEqual(intermediates.shape, (3, 4, 2))
    self.assertAllClose(outputs, intermediates[-1])


class RepeatsSparsityTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  def test_sparsity_repeats(self):
    sparse_linear = pax_fiddle.Config(
        slinears.Linear,
        name='_sparse',
        sparsity=SparsityHParams(
            sparsity_type=SparsityType.STRUCTURED_NM,
            weight_params=WeightSparsityParams(prune_rate=(2, 4)),
            mode=SparsityMode.TRAINING,
        ),
    )
    sub_p = pax_fiddle.Config(
        ff,
        input_dims=4,
        output_dims=4,
        linear_tpl=sparse_linear,
    )
    p = pax_fiddle.Config(repeats.Repeat, name='ffn', sub_tpl=sub_p, x_times=2)
    ffn = instantiate(p)

    inputs = np.random.normal(1.0, 1.5, [4, 4]).astype(np.float32)
    prng_key = jax.random.PRNGKey(seed=123)
    init_vars = ffn.init(prng_key, inputs)
    init_vars['params']['sub']['linear']['w'] = jnp.array([
        [
            [1.0, 2.0, 3.0, 4.0],
            [-3.0, -4.0, 1.0, 2.0],
            [3.0, 1.0, -4.0, 2.0],
            [-3.0, 1.0, 2.0, -4.0],
        ],
        [
            [-3.0, 1.0, 2.0, -4.0],
            [3.0, 1.0, -4.0, 2.0],
            [-3.0, -4.0, 1.0, 2.0],
            [1.0, 2.0, 3.0, 4.0],
        ],
    ])
    res, state = ffn.apply(init_vars, inputs, mutable=True)
    self.assertArraysEqual(
        state['non_trainable']['sub']['linear'][
            'w' + base_layer.SPARSITY_NAME_POSTFIX
        ],
        jnp.array([
            [
                [False, False, True, True],
                [True, True, False, False],
                [True, False, True, False],
                [True, False, False, True],
            ],
            [
                [True, False, False, True],
                [True, False, True, False],
                [True, True, False, False],
                [False, False, True, True],
            ],
        ]),
    )
    self.assertEqual(res.shape, (4, 4))
    shapes = jax.tree_map(lambda x: x.shape, state)
    expected_shapes = {
        'non_trainable': {
            'sub': {
                'linear': {'w' + base_layer.SPARSITY_NAME_POSTFIX: (2, 4, 4)}
            }
        },
        'params': {'sub': {'linear': {'w': (2, 4, 4)}, 'bias': {'b': (2, 4)}}},
    }
    self.assertEqual(shapes, expected_shapes)


class RepeatsQuantizeTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  def test_quantize_repeats(self):

    sub_p = pax_fiddle.Config(FeedForward, input_dim=2, output_dim=2)
    p = pax_fiddle.Config(repeats.Repeat, name='ffn', sub_tpl=sub_p, x_times=5)
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
        init_vars, mutable=[], method=ffn.quantized_partition_specs
    )
    expected_pspecs = {
        'non_trainable': {
            'sub': {
                'step':
                    base_layer.BoxedPartitionSpec(
                        meta=jax.sharding.PartitionSpec(None,))
            }
        },
        'params': {
            'sub': {
                'w':
                    base_layer.BoxedPartitionSpec(
                        meta=jax.sharding.PartitionSpec(None, None, None))
            }
        }
    }
    self.assertEqual(pspecs, expected_pspecs)

if __name__ == '__main__':
  absltest.main()
