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

"""Tests for quantized linears."""

import itertools
from typing import Any, Dict, Sequence

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import pax_fiddle
from praxis import test_utils
from praxis.layers import linears
from praxis.layers.quantization import linears as qlinears
from praxis.layers.quantization import quantization_hparams

instantiate = base_layer.instantiate
WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams
QuantizationHParams = quantization_hparams.QuantizationHParams
QuantizationMode = quantization_hparams.QuantizationMode
QuantizationType = quantization_hparams.QuantizationType

PARAMS = base_layer.PARAMS
NON_TRAINABLE = base_layer.NON_TRAINABLE
SUMMARIES = base_layer.SUMMARIES


def _generate_quantization_types_modes() -> Sequence[Dict[str, Any]]:
  keys = ['testcase_name', 'quantization_type', 'mode', 'dtype', 'precision']
  types = [QuantizationType.PTQ, QuantizationType.FQ, QuantizationType.AQT]
  modes = [QuantizationMode.INFERENCE, QuantizationMode.TRAINING]
  dtypes = [jnp.int8, jnp.uint8]
  precisions = [8, 4]

  cases = []
  for case in itertools.product(types, modes, dtypes, precisions):
    name = (
        case[0].value
        + '_'
        + case[1].value
        + '_'
        + str(case[2])
        + '_'
        + str(case[3])
    )
    cases.append([name] + list(case))

  return [dict(zip(keys, case)) for case in cases]


def _generate_quantization_types_symmetric() -> Sequence[Dict[str, Any]]:
  keys = ['testcase_name', 'quantization_type', 'use_symmetric']
  types = [QuantizationType.PTQ, QuantizationType.FQ, QuantizationType.AQT]
  use_symmetric = [True, False]

  cases = []
  for case in itertools.product(types, use_symmetric):
    is_symmetric = 'symmetric' if case[1] else 'asymmetric'
    name = case[0].value + '_' + is_symmetric
    cases.append([name] + list(case))

  return [dict(zip(keys, case)) for case in cases]


class QuantizedLinearTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  @parameterized.named_parameters(_generate_quantization_types_modes())
  def test_linear_quantized(self, quantization_type, mode, dtype, precision):
    p = pax_fiddle.Config(
        qlinears.Linear,
        name='_linear',
        input_dims=8,
        output_dims=4,
        quantization=QuantizationHParams(
            quantization_type=quantization_type,
            mode=mode,
            weight_params=quantization_hparams.WeightQuantizationParams(
                dtype=dtype,
                precision=precision,
            ),
        ),
    )
    linear = instantiate(p)
    inputs = jnp.array(
        [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]],
        dtype=p.dtype,
    )
    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = linear.init(prng_key, inputs)
      outputs = linear.apply(initial_vars, inputs)
    self.assertEqual(outputs.shape, (2, 4))
    if mode == QuantizationMode.INFERENCE:
      self.assertAllClose(jnp.full((2, 4), 0.0), outputs)
    else:
      self.assertRaises(AssertionError, self.assertAllClose,
                        jnp.full((2, 4), 0.0, dtype=p.dtype), outputs)

  def test_linear_aqt_quantized(self):
    p_q = pax_fiddle.Config(
        qlinears.Linear,
        name='_linear_q',
        quantization=QuantizationHParams(
            quantization_type=QuantizationType.AQT,
            mode=QuantizationMode.TRAINING,
            act_params=quantization_hparams.ActQuantizationParams(precision=3),
            weight_params=quantization_hparams.WeightQuantizationParams(
                precision=2,
                add_scale_eps=False,
            ),
        ),
    )
    p_q.input_dims = 3
    p_q.output_dims = 2

    inputs = np.array(
        [
            [-7.0, 4.01, 4.01],
            [-7.0, 0.01, -4.01],
        ],)

    weight = np.array(
        [
            [-1.5, 0.99],
            [-0.99, 0],
            [-0.01, 1.5]
        ],)
    expected_output = np.array(
        [
            [3.5, -3.5],
            [10.5, -17.5]
        ])

    linear_q = instantiate(p_q)
    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars_q = linear_q.init(prng_key, inputs)
      initial_vars_q['params']['w'] = weight
      outputs_q = linear_q.apply(initial_vars_q, inputs)
    self.assertAllClose(expected_output, outputs_q)

  def _get_layer_and_inputs(self):
    p_q = pax_fiddle.Config(
        qlinears.Linear,
        name='_linear_q',
        quantization=QuantizationHParams(
            quantization_type=QuantizationType.AQT,
            mode=QuantizationMode.TRAINING,
            act_params=quantization_hparams.ActQuantizationParams(precision=3),
            weight_params=quantization_hparams.WeightQuantizationParams(
                precision=2,
                use_step_count=True,
            ),
        ),
    )
    p_q.input_dims = 4
    p_q.output_dims = 4

    inputs = np.random.normal(size=[4, 4, p_q.input_dims]).astype(np.float32)

    linear_q = instantiate(p_q)
    return linear_q, inputs

  def test_linear_step_count_in_eval(self):
    linear_q, inputs = self._get_layer_and_inputs()
    context_params = base_layer.JaxContext.HParams(do_eval=True)
    with base_layer.JaxContext.new_context(hparams=context_params):
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars_q = linear_q.init(prng_key, inputs)
      _, updated_variables = linear_q.apply(
          initial_vars_q, inputs, mutable=[NON_TRAINABLE])
    # In eval mode step_count is not changed
    self.assertArraysEqual(updated_variables[NON_TRAINABLE]['step_count'],
                           np.array([0]))

  def test_linear_step_count_in_train(self):
    linear_q, inputs = self._get_layer_and_inputs()
    context_params = base_layer.JaxContext.HParams(do_eval=False)
    with base_layer.JaxContext.new_context(hparams=context_params):
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars_q = linear_q.init(prng_key, inputs)
      _, updated_vars = linear_q.apply(
          initial_vars_q, inputs, mutable=[PARAMS, NON_TRAINABLE, SUMMARIES]
      )
    self.assertArraysEqual(
        updated_vars[NON_TRAINABLE]['step_count'], np.array([1])
    )
    self.assertArraysEqual(
        updated_vars[SUMMARIES]['step_count_scalar'], np.array([0])
    )

    with base_layer.JaxContext.new_context(hparams=context_params):
      _, updated_vars = linear_q.apply(
          updated_vars, inputs, mutable=[PARAMS, NON_TRAINABLE, SUMMARIES]
      )
    self.assertArraysEqual(
        updated_vars[NON_TRAINABLE]['step_count'], np.array([2])
    )
    self.assertArraysEqual(
        updated_vars[SUMMARIES]['step_count_scalar'], np.array([1])
    )


class QuantizedLinearsSyncTest(test_utils.TestCase):
  """Sync tests between quantized Linear and regular Linear.

  Quantized Linear is expected to be identical to regular linear when running
  with mode=TRAINING.
  """

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  def run_and_compare(self, p_f, p_q, inputs):
    linear_f = instantiate(p_f)
    linear_q = instantiate(p_q)
    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars_f = linear_f.init(prng_key, inputs)
      initial_vars_q = linear_q.init(prng_key, inputs)
      outputs_f = linear_f.apply(initial_vars_f, inputs)
      outputs_q = linear_q.apply(initial_vars_q, inputs)
    self.assertAllClose(outputs_f, outputs_q)

  def test_linear_ptq_quantized(self):
    p_f = pax_fiddle.Config(linears.Linear, name='_linear_f')
    p_q = pax_fiddle.Config(
        qlinears.Linear,
        name='_linear_q',
        quantization=QuantizationHParams(mode=QuantizationMode.TRAINING),
    )
    for p in [p_f, p_q]:
      p.input_dims = 16
      p.output_dims = 24

    inputs = np.random.normal(1.5, 2.0, [5, 16]).astype(np.float32)
    self.run_and_compare(p_f, p_q, inputs)

  def test_linear_quantized_in_inference_mode(self):
    p_f = pax_fiddle.Config(linears.Linear, name='_linear_f')
    p_q = pax_fiddle.Config(
        qlinears.Linear,
        name='_linear_q',
        quantization=QuantizationHParams(mode=QuantizationMode.INFERENCE),
    )
    for p in [p_f, p_q]:
      p.input_dims = 4
      p.output_dims = 2

    inputs = jax.random.normal(jax.random.PRNGKey(0), (3, 4)).astype(
        jnp.float32
    )
    quantized_weight = jax.random.randint(
        jax.random.PRNGKey(0), (4, 2), minval=-128, maxval=127, dtype=jnp.int8
    )
    w_scale = jnp.array([0.5, 2.0], dtype=jnp.float32)
    weight_rescaled = quantized_weight * w_scale

    linear_f = instantiate(p_f)
    linear_q = instantiate(p_q)
    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars_f = linear_f.init(prng_key, inputs)
      initial_vars_q = linear_q.init(prng_key, inputs)
      initial_vars_f['params']['w'] = weight_rescaled
      initial_vars_q['params']['w'] = quantized_weight
      initial_vars_q['params']['w_quantized_scale'] = w_scale
      outputs_f = linear_f.apply(initial_vars_f, inputs)
      outputs_q = linear_q.apply(initial_vars_q, inputs)
    self.assertAllClose(outputs_f, outputs_q)


class QuantizeLinearTest(test_utils.TestCase):
  """Quantize Linear."""

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  @parameterized.named_parameters(_generate_quantization_types_symmetric())
  def test_quantize_linear(self, quantization_type, use_symmetric):
    p = pax_fiddle.Config(
        qlinears.Linear,
        name='_linear_q',
        mesh_axis_names=['replica', 'mdl', 'data'],
        weight_split_dims_mapping=base_layer.BaseLayer.WeightSharding(
            wt=['mdl', 'data']
        ),
        quantization=QuantizationHParams(
            quantization_type=quantization_type,
            mode=QuantizationMode.TRAINING,
            weight_params=quantization_hparams.WeightQuantizationParams(
                use_symmetric=use_symmetric
            ),
        ),
    )
    p.input_dims = 6
    p.output_dims = 4
    layer = instantiate(p)

    inputs = np.random.normal(1.5, 2.0, [5, 6]).astype(np.float32)

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = layer.init(prng_key, inputs)

      res, _ = layer.apply(
          initial_vars, mutable=[], method=layer.quantize_weight)
      # Check ParititionSpecs.
      pspec, _ = layer.apply(
          initial_vars, mutable=[], method=layer.quantized_partition_specs
      )

    shapes = jax.tree_map(lambda x: x.shape, res)
    types = jax.tree_map(lambda x: x.dtype, res)

    expected_shape = {
        base_layer.PARAMS: {'w': (6, 4), 'w_quantized_scale': (4,)}
    }
    expected_types = {
        base_layer.PARAMS: {'w': jnp.int8, 'w_quantized_scale': p.dtype}
    }
    expected_pspec = {
        'params': {
            'w': base_layer.BoxedPartitionSpec(
                meta=jax.sharding.PartitionSpec('mdl', 'data')
            ),
            'w_quantized_scale': base_layer.BoxedPartitionSpec(
                meta=jax.sharding.PartitionSpec('data')
            ),
        }
    }

    if not use_symmetric:
      expected_shape[base_layer.PARAMS]['w_quantized_zp'] = (4,)
      expected_types[base_layer.PARAMS]['w_quantized_zp'] = p.dtype
      expected_pspec['params']['w_quantized_zp'] = (
          base_layer.BoxedPartitionSpec(meta=jax.sharding.PartitionSpec('data'))
      )

    self.assertEqual(shapes, expected_shape)
    self.assertEqual(types, expected_types)
    self.assertEqual(pspec, expected_pspec)

  def test_aqt_quantize_weight(self):
    p = pax_fiddle.Config(
        qlinears.Linear,
        name='_linear_q',
        quantization=QuantizationHParams(
            quantization_type=QuantizationType.AQT,
            mode=QuantizationMode.TRAINING,
            act_params=None,
            weight_params=quantization_hparams.WeightQuantizationParams(
                precision=3,
                stop_scale_gradient=True,
                add_scale_eps=False,
            ),
        ),
    )

    p.input_dims = 3
    p.output_dims = 3
    layer = instantiate(p)

    inputs = np.random.normal(1.5, 2.0, [2, 3]).astype(np.float32)
    weight = np.array(
        [
            [-7.0, -1.01, 1.01],
            [-4.01, 3.50, 0.99],
            [-1.01, 1.99, -1.75],
        ],)
    q_weight = np.array(
        [
            [-3, -1, 2],
            [-2, 3, 2],
            [0, 2, -3]
        ], dtype=np.int8)
    expected_scale = jnp.array([2.333333, 1.166667, 0.583333], dtype=p.dtype)

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = layer.init(prng_key, inputs)
      initial_vars['params']['w'] = weight

      res, _ = layer.apply(
          initial_vars, mutable=[], method=layer.quantize_weight
      )

    self.assertArraysEqual(res['params']['w'], q_weight)
    self.assertAllClose(
        res['params']['w_quantized_scale'], expected_scale, atol=1e-6
    )


if __name__ == '__main__':
  absltest.main()
