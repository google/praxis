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

"""Tests for Praxis normalization layers."""

from absl import logging
from praxis import pax_fiddle
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from lingvo.core import bn_layers
from lingvo.core import layers as lingvo_layers
from lingvo.core import conv_layers_with_time_padding as clwp
import numpy as np
from praxis import base_layer
from praxis import py_utils
from praxis import test_utils
from praxis.layers import convolutions
from praxis.layers import normalizations
import tensorflow.compat.v2 as tf

instantiate = base_layer.instantiate
to_np = test_utils.to_np

PARAMS = base_layer.PARAMS
NON_TRAINABLE = base_layer.NON_TRAINABLE
SUMMARIES = base_layer.SUMMARIES


def _JaxToTfDtype(jax_dtype):
  if jax_dtype == jnp.bfloat16:
    return tf.bfloat16
  elif jax_dtype == jax.dtypes.float0:
    return tf.float32
  else:
    return tf.dtypes.as_dtype(jax_dtype)


class NormalizationsTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)
    tf.random.set_seed(123)

  def test_momentum(self):
    inputs = np.random.normal(1.5, 2.0, [2, 200, 8])
    paddings = np.zeros([2, 200, 1])
    paddings[1, 1, 0] = 1.0
    reduce_over_dims = [0, 1]
    keepdims = True

    jax_mean, jax_variance = normalizations.compute_moments(
        inputs, paddings, reduce_over_dims=reduce_over_dims, keepdims=keepdims)

    tf_mean, tf_variance = bn_layers.ComputeMoments(
        inputs, paddings, reduce_over_dims=reduce_over_dims, keepdims=keepdims)

    logging.info('jax_mean: %s', jax_mean)
    logging.info('jax_variance: %s', jax_variance)
    logging.info('tf_mean: %s', tf_mean)
    logging.info('tf_variance: %s', tf_variance)

    self.assertAllClose(to_np(jax_mean), to_np(tf_mean))
    self.assertAllClose(to_np(jax_variance), to_np(tf_variance))

  def test_bn01(self):
    test_layer_p = pax_fiddle.Config(
        normalizations.BatchNorm, name='bn', decay=0.8, dim=8
    )
    layer = instantiate(test_layer_p)

    inputs = np.random.normal(1.5, 2.0, [2, 200, 8])
    paddings = np.zeros([2, 200])
    paddings[1, 1] = 1.0

    # JaxContext needed for `do_eval`.
    context_p = base_layer.JaxContext.HParams(summary_verbosity=4)
    with base_layer.JaxContext.new_context(hparams=context_p):
      prng_key = jax.random.PRNGKey(seed=1234)
      prng_key, init_key = jax.random.split(prng_key)
      # initial_vars[NON_TRAINABLE] has initialized moving_mean/variance.
      initial_vars = layer.init(init_key, inputs, paddings)
      logging.info('initial_vars: %s', initial_vars)
      self.assertAllClose(
          np.sum(initial_vars[NON_TRAINABLE]['moving_mean']), 0.0)
      output1, updated_variables = layer.apply(
          initial_vars, inputs, paddings, mutable=[NON_TRAINABLE, SUMMARIES])
    new_vars = updated_variables[NON_TRAINABLE]
    summaries = updated_variables[SUMMARIES]

    logging.info('summaries: %s', summaries)
    tf.nest.assert_same_structure(
        summaries, {
            'moving_mean_scalar': None,
            'variance_scalar': None,
            'mean_scalar': None,
            'moving_variance_scalar': None
        })

    logging.info('new_vars: %s', new_vars)
    logging.info('output1: %s', output1)

    expected_moving_mean = (
        initial_vars[NON_TRAINABLE]['moving_mean'] * 0.8 +
        0.2 * summaries['mean_scalar'])
    expected_moving_variance = (
        initial_vars[NON_TRAINABLE]['moving_variance'] * 0.8 +
        0.2 * summaries['variance_scalar'])

    self.assertAllClose(
        to_np(expected_moving_mean), to_np(new_vars['moving_mean']))
    self.assertAllClose(
        to_np(expected_moving_variance), to_np(new_vars['moving_variance']))
    self.assertEqual(expected_moving_mean.shape, (8,))
    self.assertEqual(expected_moving_variance.shape, (8,))

  def test_bn02(self):
    test_layer_p = pax_fiddle.Config(
        normalizations.BatchNorm, name='bn', decay=0.8, dim=1
    )
    layer = instantiate(test_layer_p)

    inputs = np.random.normal(1.5, 2.0, [2, 200, 1])
    paddings = np.zeros([2, 200])
    paddings[1, 1] = 1.0

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123456)
      prng_key, init_key = jax.random.split(prng_key)
      initial_vars = layer.init(init_key, inputs, paddings)
      initial_vars[PARAMS]['beta'] = jnp.array([0.7])
      initial_vars[PARAMS]['gamma'] = jnp.array([1.8])
      logging.info('initial_vars: %s', initial_vars)
      jax_output, updated_variables = layer.apply(
          initial_vars, inputs, paddings, mutable=[NON_TRAINABLE])
      del updated_variables

    tf_initial_vars = py_utils.NestedMap.FromNestedDict(initial_vars[PARAMS])
    tf_initial_vars = tf_initial_vars.Transform(to_np)

    # Now run TF based computation.
    tf_layer_p = bn_layers.BatchNormLayer.Params().Set(
        name='bn', dim=1, decay=8)
    tf_layer = tf_layer_p.Instantiate()
    tf_output = tf_layer.FProp(tf_initial_vars, inputs,
                               np.expand_dims(paddings, axis=-1))
    logging.info('tf_output: %s', tf_output)
    self.assertAllClose(to_np(jax_output), to_np(tf_output))

  @parameterized.parameters(
      (0.0, 0.0, False),
      (0.5, 0.0, False),
      (0.0, 0.5, False),
      (0.5, 0.5, False),
      (0.5, 1.0, False),
      (0.0, 0.0, True),
      (0.5, 0.0, True),
      (0.0, 0.5, True),
      (0.5, 0.5, True),
      (0.5, 1.0, True),
  )
  def test_layer_norm(self, scale, bias, direct_scale):
    dim = 3
    p = pax_fiddle.Config(
        normalizations.LayerNorm,
        name='jax_ln',
        dim=dim,
        direct_scale=direct_scale,
    )
    layer_norm = instantiate(p)
    npy_input = np.random.normal(1.0, 0.5, [10, 10, 10, p.dim]).astype(
        'float32'
    )
    inputs = jnp.asarray(npy_input)
    prng_key = jax.random.PRNGKey(seed=123456)
    prng_key, init_key = jax.random.split(prng_key)
    initial_vars = layer_norm.init(init_key, inputs)
    logging.info('LN initial_vars: %s', initial_vars)
    initial_vars[PARAMS]['scale'] = jnp.array([scale] * dim, dtype=jnp.float32)
    initial_vars[PARAMS]['bias'] = jnp.array([bias] * dim, dtype=jnp.float32)
    outputs = layer_norm.apply(initial_vars, inputs)
    # Now test whether tf layer norm returns same output
    tf_initial_vars = py_utils.NestedMap.FromNestedDict(initial_vars[PARAMS])
    tf_initial_vars = tf_initial_vars.Transform(to_np)
    tf_p = lingvo_layers.LayerNorm.Params().Set(
        name='tf_ln', input_dim=p.dim, direct_scale=direct_scale
    )
    tf_layer_norm = tf_p.Instantiate()
    tf_output = tf_layer_norm.FProp(
        tf_initial_vars, tf.constant(inputs, dtype=tf.float32)
    )
    np_outputs = to_np(outputs)
    tf_np_outputs = to_np(tf_output)
    self.assertAllClose(bias, np_outputs.mean(), atol=1e-3)
    scale = scale if direct_scale else (1.0 + scale)
    self.assertAllClose(scale**2, np.var(np_outputs), atol=5e-3)
    self.assertAllClose(tf_np_outputs, np_outputs, atol=6e-5)

  @parameterized.parameters((0.0,), (0.5,))
  def test_rms_norm(self, scale):
    dim = 3
    p = pax_fiddle.Config(
        normalizations.RmsNorm, name='jax_rmsn', dim=dim, direct_scale=False
    )
    rms_norm = instantiate(p)
    npy_input = np.random.normal(1.0, 0.5,
                                 [10, 10, 10, p.dim]).astype('float32')
    inputs = jnp.asarray(npy_input)
    prng_key = jax.random.PRNGKey(seed=123456)
    prng_key, init_key = jax.random.split(prng_key)
    initial_vars = rms_norm.init(init_key, inputs)
    initial_vars[PARAMS]['scale'] = jnp.array([scale] * dim, dtype=jnp.float32)
    outputs = rms_norm.apply(initial_vars, inputs)
    # Now test whether tf RMS norm returns same output.
    tf_initial_vars = py_utils.NestedMap.FromNestedDict(initial_vars[PARAMS])
    tf_p = lingvo_layers.LayerNorm.Params().Set(
        name='tf_rmsn', input_dim=p.dim, bias=False, center=False)
    tf_layer_norm = tf_p.Instantiate()
    tf_output = tf_layer_norm.FProp(tf_initial_vars,
                                    tf.constant(inputs, dtype=tf.float32))
    np_outputs = to_np(outputs)
    tf_np_outputs = to_np(tf_output)
    np_norms = np.linalg.norm(np_outputs / np.sqrt(float(dim)), axis=-1)
    self.assertAllClose(
        (1.0 + scale) * np.ones_like(np_norms), np_norms, atol=5e-3)
    self.assertAllClose(tf_np_outputs, np_outputs, atol=6e-5)

  @parameterized.named_parameters(
      ('_epsilon_1e-3', 4, 2, False, 4, 1e-3, [2, 2, 2, 4], jnp.float32, None,
       jnp.float32),
      ('_epsilon_1e-6', 4, 2, False, 4, 1e-6, [2, 2, 2, 4], jnp.float32, None,
       jnp.float32),
      ('_f32_input_f32_fprop', 4, 2, False, 4, 1e-3, [2, 2, 2, 4], jnp.float32,
       [[0, 0], [0, 1]], jnp.float32),
      ('_bf16_input_f32_fprop', 4, 2, False, 4, 1e-3, [2, 2, 2, 4],
       jnp.bfloat16, [[0, 0], [0, 1]], jnp.float32),
      ('_f32_input_bf16_fprop', 4, 2, False, 4, 1e-3, [2, 2, 2, 4], jnp.float32,
       [[0, 0], [0, 1]], jnp.bfloat16),
      ('_bf16_input_bf16_fprop', 4, 2, False, 4, 1e-3, [2, 2, 2, 4],
       jnp.bfloat16, [[0, 0], [0, 1]], jnp.bfloat16),
      ('_3d_input', 4, 2, False, 3, 1e-3, [2, 2, 4], jnp.float32,
       [[0, 0], [0, 1]], jnp.float32),
      ('_4d_input_cumulative_mode', 2, 2, True, 4, 1e-3, [2, 4, 1, 2],
       jnp.float32, [[0, 0, 0, 0], [0, 0, 0, 0]], jnp.float32),
      ('_3d_input_cumulative_mode', 2, 2, True, 3, 1e-3, [2, 4, 2], jnp.float32,
       [[0, 0, 0, 0], [0, 0, 0, 0]], jnp.float32),
  )
  def test_group_norm(self, dim, num_groups, cumulative, input_rank, epsilon,
                      input_shape, input_dtype, paddings, fprop_dtype):
    p = pax_fiddle.Config(
        normalizations.GroupNorm,
        name='jax_gn',
        dim=dim,
        num_groups=num_groups,
        cumulative=cumulative,
        input_rank=input_rank,
        epsilon=epsilon,
        fprop_dtype=fprop_dtype,
    )
    group_norm = instantiate(p)
    npy_input = np.random.normal(1.0, 0.5, input_shape).astype(np.float32)
    inputs = jnp.asarray(npy_input, dtype=input_dtype)
    jax_paddings = paddings
    if jax_paddings is not None:
      jax_paddings = jnp.asarray(jax_paddings, dtype=input_dtype)
    prng_key = jax.random.PRNGKey(seed=123456)
    prng_key, init_key = jax.random.split(prng_key)
    initial_vars = group_norm.init(init_key, inputs, paddings=jax_paddings)
    output = group_norm.apply(initial_vars, inputs, paddings=jax_paddings)

    # Now test whether tf layer norm returns same output.
    tf_initial_vars = py_utils.NestedMap.FromNestedDict(initial_vars[PARAMS])
    tf_p = bn_layers.GroupNormLayer.Params().Set(
        name='tf_gn',
        dim=dim,
        num_groups=num_groups,
        cumulative=cumulative,
        input_rank=input_rank,
        epsilon=epsilon,
        fprop_dtype=_JaxToTfDtype(fprop_dtype))
    tf_group_norm = tf_p.Instantiate()
    tf_inputs = tf.constant(inputs, dtype=_JaxToTfDtype(input_dtype))
    if paddings is None:
      tf_output = tf_group_norm.FProp(tf_initial_vars, tf_inputs, paddings=None)
    else:
      tf_output, _ = tf_group_norm.FProp(
          tf_initial_vars,
          tf_inputs,
          paddings=tf.convert_to_tensor(
              paddings, dtype=_JaxToTfDtype(input_dtype)))
      # TF doesn't apply padding so we apply manually
      paddings = np.array(paddings)
      expanded_shape = list(paddings.shape) + [
          1,
      ] * (
          len(tf_output.shape) - 2)
      expanded_padding = np.reshape(paddings, expanded_shape)
      tf_output *= (1 - expanded_padding)

    self.assertAllClose(to_np(tf_output), to_np(output))

  @parameterized.named_parameters(
      (
          '_epsilon_1e-3',
          4,
          2,
          False,
          5,
          1e-3,
          [2, 2, 2, 2, 4],
          jnp.float32,
          None,
          jnp.float32,
      ),
      (
          '_epsilon_1e-6',
          4,
          2,
          False,
          5,
          1e-6,
          [2, 2, 2, 2, 4],
          jnp.float32,
          None,
          jnp.float32,
      ),
      (
          '_f32_input_f32_fprop',
          4,
          2,
          False,
          5,
          1e-3,
          [2, 2, 2, 2, 4],
          jnp.float32,
          [[0, 0], [0, 1]],
          jnp.float32,
      ),
      (
          '_bf16_input_f32_fprop',
          4,
          2,
          False,
          5,
          1e-3,
          [2, 2, 2, 2, 4],
          jnp.bfloat16,
          [[0, 0], [0, 1]],
          jnp.float32,
      ),
      (
          '_f32_input_bf16_fprop',
          4,
          2,
          False,
          5,
          1e-3,
          [2, 2, 2, 2, 4],
          jnp.float32,
          [[0, 0], [0, 1]],
          jnp.bfloat16,
      ),
      (
          '_bf16_input_bf16_fprop',
          4,
          2,
          False,
          5,
          1e-3,
          [2, 2, 2, 2, 4],
          jnp.bfloat16,
          [[0, 0], [0, 1]],
          jnp.bfloat16,
      ),
      (
          '_5d_input',
          4,
          2,
          False,
          5,
          1e-3,
          [2, 5, 2, 3, 4],
          jnp.float32,
          [[0, 0, 0, 0, 1], [1, 0, 0, 1, 0]],
          jnp.float32,
      ),
      (
          '_5d_input_cumulative_mode',
          4,
          2,
          True,
          5,
          1e-3,
          [2, 5, 2, 3, 4],
          jnp.float32,
          [[0, 0, 0, 0, 1], [1, 0, 0, 1, 0]],
          jnp.float32,
      ),
  )
  def test_group_norm_3d(
      self,
      dim,
      num_groups,
      cumulative,
      input_rank,
      epsilon,
      input_shape,
      input_dtype,
      paddings,
      fprop_dtype,
  ):
    p = pax_fiddle.Config(
        normalizations.GroupNorm,
        name='jax_gn',
        dim=dim,
        num_groups=num_groups,
        cumulative=cumulative,
        input_rank=input_rank,
        epsilon=epsilon,
        fprop_dtype=fprop_dtype,
    )
    group_norm = instantiate(p)
    npy_input = np.random.normal(1.0, 0.5, input_shape).astype(np.float32)
    inputs = jnp.asarray(npy_input, dtype=input_dtype)
    jax_paddings = paddings
    if jax_paddings is not None:
      jax_paddings = jnp.asarray(jax_paddings, dtype=input_dtype)
    prng_key = jax.random.PRNGKey(seed=123456)
    _, init_key = jax.random.split(prng_key)
    initial_vars = group_norm.init(init_key, inputs, paddings=jax_paddings)
    output = group_norm.apply(initial_vars, inputs, paddings=jax_paddings)

    # Now test whether reshaping input into NHWC returns same output.
    p_2d = pax_fiddle.Config(
        normalizations.GroupNorm,
        name='jax_gn_2d',
        dim=dim,
        num_groups=num_groups,
        cumulative=cumulative,
        input_rank=4,
        epsilon=epsilon,
        fprop_dtype=fprop_dtype,
    )
    group_norm_2d = instantiate(p_2d)
    new_shape = tuple(input_shape[:2]) + (
        input_shape[2] * input_shape[3],
        input_shape[4],
    )
    inputs_2d = jnp.reshape(inputs, new_shape)
    initial_vars_2d = group_norm_2d.init(
        init_key, inputs_2d, paddings=jax_paddings
    )
    output_2d = group_norm_2d.apply(
        initial_vars_2d, inputs_2d, paddings=jax_paddings
    )
    output_2d = jnp.reshape(output_2d, input_shape)
    self.assertAllClose(to_np(output_2d), to_np(output))

  @parameterized.parameters(
      ((5, 4, 24, 36), (1, 1), [2, 16, 36, 72]),
      ((2, 4, 16, 8), (2, 2), [2, 16, 32, 128]),
      ((4, 8, 16, 32), (1, 1), [2, 16, 32, 64]),
  )
  def test_weight_norm_conv(self, filter_shape, filter_stride, input_shape):
    inputs = np.random.normal(1.0, 0.5, input_shape).astype('float32')

    p = pax_fiddle.Config(
        convolutions.Conv2D,
        name='jax_conv2d',
        filter_shape=filter_shape,
        filter_stride=filter_stride,
        weight_norm_tpl=pax_fiddle.Config(normalizations.WeightNormL2),
    )
    conv_layer = instantiate(p)
    initial_vars = conv_layer.init(jax.random.PRNGKey(seed=123), inputs)
    output = conv_layer.apply(initial_vars, inputs)

    tf_p = clwp.Conv2DLayerWithPadding.Params().Set(
        name='tf_conv2d',
        filter_shape=filter_shape,
        filter_stride=filter_stride,
        weight_norm=True,
    )
    tf_conv_layer = tf_p.Instantiate()
    pax_g = initial_vars[PARAMS]['weight_norm']['g']
    self.assertAllClose(tf_conv_layer.theta.g, pax_g)

    tf_theta = py_utils.NestedMap.FromNestedDict(initial_vars[PARAMS])
    tf_theta.Set('g', pax_g)
    tf_output, unused_padding = tf_conv_layer.FProp(
        theta=tf_theta,
        inputs=inputs,
        paddings=tf.zeros(input_shape[:2]),
    )
    self.assertAllClose(tf_output, output)

  def test_higher_intermediate_precision(self):
    rms_norm = instantiate(
        pax_fiddle.Config(
            normalizations.RmsNorm,
            fprop_dtype=jnp.bfloat16,
            name='jax_rmsn',
            dim=10,
            epsilon=1e-6,
            direct_scale=False,
            intermediate_dtype=jnp.float32,
        )
    )
    inputs = jnp.asarray(
        jnp.arange(100).reshape((10, 10)) * 1e-10, jnp.bfloat16
    )
    float32_inputs = jnp.asarray(inputs, jnp.float32)

    # The scale is all zeros, which with direct_scale=False means the scaling
    # is a no-op, as it is multiplication by 1.
    output = rms_norm.apply({'params': {'scale': jnp.zeros((10,))}}, inputs)

    # The expected output is the bfloat16-rounded version of the correct,
    # float32 answer.
    expected_output = jnp.asarray(
        float32_inputs * jax.lax.rsqrt(jnp.square(float32_inputs) + 1e-6),
        jnp.bfloat16,
    )

    # The bfloat16 version would have too much rounding error.
    self.assertAllClose(expected_output, output, atol=1e-12)

  @parameterized.product(kernel_size=[2, 3, 5], do_eval=[True, False])
  def test_spectral_norm(self, kernel_size: int, do_eval: bool):
    # This test is based on keras SpectralNormalizationTest.test_normalization,
    # which checks that SN normalizes weights by the maximum eigen value.
    dim = 1
    w = np.random.rand(kernel_size, kernel_size).astype(np.float32)
    w = w @ w.T
    eigen_val, _ = jnp.linalg.eig(w)
    expected = w / jnp.max(jnp.abs(eigen_val))

    layer = normalizations.SpectralNorm(dim=dim, n_power_iteration=10)
    prng_key = jax.random.PRNGKey(seed=1234)
    context_p = base_layer.JaxContext.HParams(do_eval=do_eval)
    with base_layer.JaxContext.new_context(hparams=context_p):
      init = layer.init(prng_key, w[:, :, None, None])
      actual, updated = layer.apply(
          init, w[:, :, None, None], mutable=[NON_TRAINABLE]
      )

    self.assertAllClose(actual[:, :, 0, 0], expected, atol=1e-2)
    if do_eval:
      self.assertAllClose(updated[NON_TRAINABLE]['u'], init[NON_TRAINABLE]['u'])
    else:
      self.assertNotAllClose(
          updated[NON_TRAINABLE]['u'], init[NON_TRAINABLE]['u']
      )

  @parameterized.parameters(
      ((5, 4, 24, 36), (1, 1), [2, 16, 36, 72], jnp.bfloat16),
      ((2, 4, 16, 8), (2, 2), [2, 16, 32, 128], jnp.bfloat16),
      ((4, 8, 16, 32), (1, 1), [2, 16, 32, 64], jnp.float32),
  )
  def test_spectral_norm_conv_fprop_dtype(
      self, filter_shape, filter_stride, input_shape, fprop_dtype
  ):
    inputs = np.random.normal(1.0, 0.5, input_shape).astype('float32')

    p = pax_fiddle.Config(
        convolutions.Conv2D,
        name='jax_conv2d',
        filter_shape=filter_shape,
        filter_stride=filter_stride,
        weight_norm_tpl=pax_fiddle.Config(normalizations.SpectralNorm),
        fprop_dtype=fprop_dtype,
    )
    conv_layer = instantiate(p)
    initial_vars = conv_layer.init(jax.random.PRNGKey(seed=123), inputs)
    output = conv_layer.apply(initial_vars, inputs)
    self.assertEqual(output.dtype, fprop_dtype)


if __name__ == '__main__':
  absltest.main()
