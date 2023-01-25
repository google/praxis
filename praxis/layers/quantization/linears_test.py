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

"""Tests for quantized linears."""

import itertools
from typing import Any, Dict, Sequence

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from jax.experimental import pjit
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


def _generate_quantization_types_modes() -> Sequence[Dict[str, Any]]:
  keys = ['testcase_name', 'quantization_type', 'mode']
  types = [QuantizationType.PTQ, QuantizationType.FQ, QuantizationType.AQT]
  modes = [QuantizationMode.INFERENCE, QuantizationMode.TRAINING]

  cases = []
  for case in itertools.product(types, modes):
    name = case[0].value + '_' + case[1].value
    cases.append([name] + list(case))

  return [dict(zip(keys, case)) for case in cases]


class QuantizedLinearTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  @parameterized.named_parameters(_generate_quantization_types_modes())
  def test_linear_quantized(self, quantization_type, mode):
    p = pax_fiddle.Config(
        qlinears.Linear,
        name='_linear',
        input_dims=5,
        output_dims=4,
        quantization=QuantizationHParams(
            quantization_type=quantization_type, mode=mode
        ),
    )
    linear = instantiate(p)
    inputs = jnp.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=p.dtype)
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

  def test_linear_aqt_quantized(self):
    p_f = pax_fiddle.Config(linears.Linear, name='_linear_f')
    p_q = pax_fiddle.Config(
        qlinears.Linear,
        name='_linear_q',
        quantization=QuantizationHParams(
            quantization_type=QuantizationType.AQT,
            mode=QuantizationMode.TRAINING,
            act_params=quantization_hparams.ActQuantizationParams(precision=3),
            weight_params=quantization_hparams.WeightQuantizationParams(
                precision=2
            ),
        ),
    )
    for p in [p_f, p_q]:
      p.input_dims = 3
      p.output_dims = 2

    inputs = np.array(
        [
            [-7.0, 4.01, 4.01],
            [-7.0, 0.01, -4.01],
        ],)
    q_inputs = np.array(
        [
            [-6, 4, 4],
            [-6, 0, -4]
        ],)

    weight = np.array(
        [
            [-1.5, 0.99],
            [-0.99, 0],
            [-0.01, 1.5]
        ],)
    q_weight = np.array(
        [
            [-1, 1],
            [-1, 0],
            [0, 1]
        ],)

    linear_f = instantiate(p_f)
    linear_q = instantiate(p_q)
    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars_f = linear_f.init(prng_key, q_inputs)
      initial_vars_q = linear_q.init(prng_key, inputs)
      initial_vars_f['params']['w'] = q_weight
      initial_vars_q['params']['w'] = weight
      outputs_f = linear_f.apply(initial_vars_f, q_inputs)
      outputs_q = linear_q.apply(initial_vars_q, inputs)
    self.assertAllClose(outputs_f.astype(outputs_q.dtype), outputs_q)

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

  @parameterized.named_parameters(
      dict(testcase_name='PTQ', quantization_type=QuantizationType.PTQ),
      dict(testcase_name='FQ', quantization_type=QuantizationType.FQ),
      dict(testcase_name='AQT', quantization_type=QuantizationType.AQT)
  )
  def test_quantize_linear(self, quantization_type):
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
    shapes = jax.tree_map(lambda x: x.shape, res)
    types = jax.tree_map(lambda x: x.dtype, res)
    self.assertEqual(
        shapes, {base_layer.PARAMS: {
            'w': (6, 4),
            'w_quantized_scale': (4,)
        }})
    self.assertEqual(
        types,
        {base_layer.PARAMS: {
            'w': jnp.int8,
            'w_quantized_scale': p.dtype
        }})

    # Check ParititionSpecs.
    pspec, _ = layer.apply(
        initial_vars, mutable=[], method=layer.quantized_partition_specs
    )
    expected_pspec = {
        'params': {
            'w': base_layer.BoxedPartitionSpec(
                meta=pjit.PartitionSpec('mdl', 'data')
            ),
            'w_quantized_scale': base_layer.BoxedPartitionSpec(
                meta=pjit.PartitionSpec('data')
            ),
        }
    }
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
            [-1, 2, -3]
        ], dtype=np.int8)
    expected_scale = jnp.array([2, 1, 0.5], dtype=p.dtype)

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = layer.init(prng_key, inputs)
      initial_vars['params']['w'] = weight

      res, _ = layer.apply(
          initial_vars, mutable=[], method=layer.quantize_weight)

    self.assertArraysEqual(res['params']['w'], q_weight)
    self.assertArraysEqual(res['params']['w_quantized_scale'], expected_scale)


if __name__ == '__main__':
  absltest.main()
