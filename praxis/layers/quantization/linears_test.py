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

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from jax.experimental import pjit
import numpy as np
from praxis import base_layer
from praxis import test_utils
from praxis.layers import linears
from praxis.layers.quantization import linears as qlinears

instantiate = base_layer.instantiate
BaseHParams = base_layer.BaseLayer.HParams
WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams


class QuantizedAttentionTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  @parameterized.named_parameters(
      ('inference', base_layer.QuantizationMode.INFERENCE),
      ('quantize', base_layer.QuantizationMode.QUANTIZE),
  )
  def test_linear_quantized(self, mode):
    p = qlinears.Linear.HParams(
        name='_linear',
        input_dims=5,
        output_dims=4,
        quantization=base_layer.QuantizationHParams(mode=mode))
    linear = instantiate(p)
    inputs = jnp.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=jnp.bfloat16)
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = linear.init(prng_key, inputs)
    outputs = linear.apply(initial_vars, inputs)
    self.assertEqual(outputs.shape, (2, 4))
    if mode == base_layer.QuantizationMode.INFERENCE:
      self.assertAllClose(jnp.full((2, 4), 0.0), outputs)


class QuantizedLinearsSyncTest(test_utils.TestCase):
  """Sync tests between quantized Linear and regular Linear.

  Quantized Linear is expected to be identical to regular linear when
  running with mode = QUANTIZE.
  """

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  def run_and_compare(self, p_f, p_q, inputs):
    attn_f = instantiate(p_f)
    attn_q = instantiate(p_q)
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars_f = attn_f.init(prng_key, inputs)
    initial_vars_q = attn_q.init(prng_key, inputs)
    outputs_f = attn_f.apply(initial_vars_f, inputs)
    outputs_q = attn_q.apply(initial_vars_q, inputs)
    self.assertAllClose(outputs_f, outputs_q)

  def test_linear_quantized(self):
    p_f = linears.Linear.HParams(name='_linear_f')
    p_q = qlinears.Linear.HParams(
        name='_linear_q',
        quantization=base_layer.QuantizationHParams(
            mode=base_layer.QuantizationMode.QUANTIZE))
    for p in [p_f, p_q]:
      p.input_dims = 16
      p.output_dims = 24

    inputs = np.random.normal(1.5, 2.0, [5, 16]).astype(np.float32)
    self.run_and_compare(p_f, p_q, inputs)


class QuantizeLinearTest(test_utils.TestCase):
  """Quantize Linear."""

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  def test_quantize_linear(self):
    p = qlinears.Linear.HParams(
        name='_linear_q',
        mesh_axis_names=['replica', 'mdl', 'data'],
        weight_split_dims_mapping=base_layer.BaseLayer.WeightShardingHParams(
            wt=['mdl', 'data']),
        quantization=base_layer.QuantizationHParams(
            mode=base_layer.QuantizationMode.QUANTIZE))
    p.input_dims = 6
    p.output_dims = 4
    layer = instantiate(p)

    inputs = np.random.normal(1.5, 2.0, [5, 6]).astype(np.float32)
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = layer.init(prng_key, inputs)

    res, _ = layer.apply(initial_vars, mutable=[], method=layer.quantize_weight)
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
            'w_quantized_scale': jnp.bfloat16
        }})

    # Check ParititionSpecs.
    pspec, _ = layer.apply(
        initial_vars, mutable=[], method=layer.quantized_partitioned_specs)
    exepected_pspec = {
        'params': {
            'w':
                base_layer.BoxedPartitionSpec(
                    meta=pjit.PartitionSpec('mdl', 'data')),
            'w_quantized_scale':
                base_layer.BoxedPartitionSpec(meta=pjit.PartitionSpec('data'))
        }
    }
    self.assertEqual(pspec, exepected_pspec)


if __name__ == '__main__':
  absltest.main()
