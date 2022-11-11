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

"""Tests for quantized attentions."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from jax.experimental import pjit
import numpy as np
from praxis import base_layer
from praxis import py_utils
from praxis import test_utils
from praxis.layers import attentions
from praxis.layers.quantization import attentions as qattentions

instantiate = base_layer.instantiate


class QuantizedAttentionTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  @parameterized.named_parameters(
      ('inference', base_layer.QuantizationMode.INFERENCE),
      ('quantize', base_layer.QuantizationMode.QUANTIZE),
  )
  def test_attention_projection_quantized(self, mode):
    p = qattentions.AttentionProjection.HParams(
        name='_attn_proj',
        input_dim=5,
        num_heads=2,
        dim_per_head=3,
        is_output_projection=True,
        quantization=base_layer.QuantizationHParams(mode=mode))
    attn = instantiate(p)
    inputs = jnp.ones((4, 2, 3), dtype=jnp.bfloat16)
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = attn.init(prng_key, inputs)
    outputs = attn.apply(initial_vars, inputs)
    self.assertEqual(outputs.shape, (4, 5))
    if mode == base_layer.QuantizationMode.INFERENCE:
      self.assertAllClose(jnp.full((4, 5), 0.0), outputs)

  @parameterized.named_parameters(
      ('inference', base_layer.QuantizationMode.INFERENCE),
      ('quantize', base_layer.QuantizationMode.QUANTIZE),
  )
  def test_attention_projection_no_output_proj_quantized(self, mode):
    p = qattentions.AttentionProjection.HParams(
        name='_attn_proj',
        input_dim=5,
        num_heads=2,
        dim_per_head=3,
        is_output_projection=False,
        quantization=base_layer.QuantizationHParams(mode=mode))
    attn = instantiate(p)
    inputs = jnp.ones((4, 3, 5), dtype=jnp.bfloat16)
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = attn.init(prng_key, inputs)
    outputs = attn.apply(initial_vars, inputs)
    self.assertEqual(outputs.shape, (4, 3, 2, 3))
    if mode == base_layer.QuantizationMode.INFERENCE:
      self.assertAllClose(jnp.full((4, 3, 2, 3), 0.0), outputs)

  @parameterized.named_parameters(
      ('inference', base_layer.QuantizationMode.INFERENCE),
      ('quantize', base_layer.QuantizationMode.QUANTIZE),
  )
  def test_combined_projection_quantized(self, mode):
    p = qattentions.CombinedQKVProjectionLayer.HParams(
        name='_combined_qkv',
        input_dim=5,
        num_heads=3,
        dim_per_head=2,
        quantization=base_layer.QuantizationHParams(mode=mode))
    attn = instantiate(p)
    inputs = jnp.ones((4, 5), dtype=jnp.bfloat16)
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = attn.init(prng_key, inputs)
    out_q, out_k, out_v = attn.apply(initial_vars, inputs)
    expected_shape = (4, 3, 2)
    self.assertEqual(out_q.shape, expected_shape)
    self.assertEqual(out_k.shape, expected_shape)
    self.assertEqual(out_v.shape, expected_shape)
    if mode == base_layer.QuantizationMode.INFERENCE:
      self.assertAllClose(jnp.full(expected_shape, 0.0), out_q)
      self.assertAllClose(jnp.full(expected_shape, 0.0), out_k)
      self.assertAllClose(jnp.full(expected_shape, 0.0), out_v)


def assert_var_stats_close(map01, map02, test_case):

  def var_stats(x):
    return np.mean(x), np.std(x)

  map01_items = map01.FlattenItems()
  map02_items = map02.FlattenItems()

  def have_similar_stats(x, y):
    mean1, std1 = var_stats(test_utils.to_np(x))
    mean2, std2 = var_stats(test_utils.to_np(y))
    delta_mean = np.abs(mean1 - mean2)
    delta_std = np.abs(std1 - std2)
    test_case.assertLess(delta_mean, 0.0002)
    test_case.assertLess(delta_std, 0.0002)

  for x, y in zip(map01_items, map02_items):
    assert x[0] == y[0]
    have_similar_stats(x[1], y[1])


class QuantizedAttentionSyncTest(test_utils.TestCase):
  """Sync tests between quantized attention and regular attention.

  Quantized attention is expected to be identical to regular attention when
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

  # test case copied from test_mhd_projection_01.
  def test_mhd_projection_01_quantized(self):
    p_f = attentions.AttentionProjection.HParams(name='_attn_proj_f')
    p_q = qattentions.AttentionProjection.HParams(
        name='_attn_proj_q',
        quantization=base_layer.QuantizationHParams(
            mode=base_layer.QuantizationMode.QUANTIZE))
    for p in [p_f, p_q]:
      p.input_dim = 16
      p.num_heads = 2
      p.dim_per_head = 5
      p.is_output_projection = False

    inputs = np.random.normal(1.5, 2.0, [5, 16]).astype(np.float32)
    self.run_and_compare(p_f, p_q, inputs)

  # test case copied from test_mhd_projection_02.
  @parameterized.parameters([False, True])
  def test_mhd_projection_02_quantized(self, use_nhd_shape):
    p_f = attentions.AttentionProjection.HParams(name='_attn_proj_f')
    p_q = qattentions.AttentionProjection.HParams(
        name='_attn_proj_q',
        quantization=base_layer.QuantizationHParams(
            mode=base_layer.QuantizationMode.QUANTIZE))
    for p in [p_f, p_q]:
      p.input_dim = 16
      p.num_heads = 2
      p.dim_per_head = 5
      p.is_output_projection = True
      p.use_nhd_shape = use_nhd_shape

    inputs = np.random.normal(1.5, 2.0, [5, 2, 5]).astype(np.float32)
    self.run_and_compare(p_f, p_q, inputs)

  # test case copied from test_mhd_projection_var_stats.
  def test_mhd_projection_var_stats_quantized(self):
    p_f = attentions.AttentionProjection.HParams(name='_attn_proj_f')
    p_q = qattentions.AttentionProjection.HParams(
        name='_attn_proj_q',
        quantization=base_layer.QuantizationHParams(
            mode=base_layer.QuantizationMode.QUANTIZE))
    for p in [p_f, p_q]:
      p.input_dim = 256
      p.num_heads = 16
      p.dim_per_head = 16
      p.is_output_projection = True

    attn_f = instantiate(p_f)
    attn_q = instantiate(p_q)
    inputs = np.random.normal(1.5, 2.0, [2, 16, 16]).astype(np.float32)
    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)
    initial_vars_f = attn_f.init(init_key, inputs)
    initial_vars_q = attn_q.init(init_key, inputs)
    assert_var_stats_close(
        py_utils.NestedMap.FromNestedDict(initial_vars_f['params']),
        py_utils.NestedMap.FromNestedDict(initial_vars_q['params']), self)

  # test case copied from test_combine_qkv_with_attention_combine_dims.
  def test_combine_qkv_with_attention_combine_dims_quantized(self):
    p_f = attentions.CombinedQKVProjectionLayer.HParams(name='_attn_qkv_f')
    p_q = qattentions.CombinedQKVProjectionLayer.HParams(
        name='_attn_qkv_q',
        quantization=base_layer.QuantizationHParams(
            mode=base_layer.QuantizationMode.QUANTIZE))
    for p in [p_f, p_q]:
      p.input_dim = 64
      p.num_heads = 8
      p.dim_per_head = 8
      p.attention_combine_dims = True

    inputs = np.random.normal(size=[3, 64]).astype(np.float32)
    self.run_and_compare(p_f, p_q, inputs)


class QuantizeAttentionTest(test_utils.TestCase):
  """Quantize attention."""

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  def test_quantize_attention_projection(self):
    p = qattentions.AttentionProjection.HParams(
        name='_attn_proj_q',
        mesh_axis_names=['replica', 'mdl', 'data'],
        weight_split_dims_mapping=base_layer.BaseLayer.WeightShardingHParams(
            wt=['mdl', 'data']),
        quantization=base_layer.QuantizationHParams(
            mode=base_layer.QuantizationMode.QUANTIZE))
    p.input_dim = 16
    p.num_heads = 2
    p.dim_per_head = 5
    p.is_output_projection = True
    p.use_nhd_shape = True
    layer = instantiate(p)

    inputs = np.random.normal(1.5, 2.0, [5, 2, 5]).astype(np.float32)
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = layer.init(prng_key, inputs)

    res, _ = layer.apply(initial_vars, mutable=[], method=layer.quantize_weight)
    self.assertEqual(len(res), 1)
    self.assertEqual(len(res[base_layer.PARAMS]), 2)
    self.assertEqual(res[base_layer.PARAMS]['w'].shape, (2, 5, 16))
    self.assertEqual(res[base_layer.PARAMS]['w_quantized_scale'].shape, (16,))

    pspec, _ = layer.apply(
        initial_vars, mutable=[], method=layer.quantized_partitioned_specs)
    exepected_pspec = {
        'params': {
            'w':
                base_layer.BoxedPartitionSpec(
                    meta=pjit.PartitionSpec('mdl', 'data')),
            'w_quantized_scale':
                base_layer.BoxedPartitionSpec(meta=pjit.PartitionSpec('mdl'))
        }
    }
    self.assertEqual(pspec, exepected_pspec)

  def test_quantize_attention_qkv(self):
    p = qattentions.CombinedQKVProjectionLayer.HParams(
        name='_combined_qkv',
        input_dim=5,
        num_heads=6,
        dim_per_head=2,
        ici_mesh_shape=[0, 1, 2],
        mesh_axis_names=['replica', 'mdl', 'data'],
        weight_split_dims_mapping=base_layer.BaseLayer.WeightShardingHParams(
            wt=['replica', 'mdl', 'data']),
        quantization=base_layer.QuantizationHParams(
            mode=base_layer.QuantizationMode.QUANTIZE))
    layer = instantiate(p)
    inputs = jnp.ones((4, 5), dtype=jnp.bfloat16)
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = layer.init(prng_key, inputs)

    res, _ = layer.apply(initial_vars, mutable=[], method=layer.quantize_weight)
    self.assertEqual(len(res), 1)
    self.assertEqual(len(res[base_layer.PARAMS]), 2)
    self.assertEqual(res[base_layer.PARAMS]['w'].shape, (3, 5, 6, 2))
    self.assertEqual(res[base_layer.PARAMS]['w_quantized_scale'].shape,
                     (3, 6, 2))

    pspec, _ = layer.apply(
        initial_vars, mutable=[], method=layer.quantized_partitioned_specs)
    exepected_pspec = {
        'params': {
            'w':
                base_layer.BoxedPartitionSpec(
                    meta=pjit.PartitionSpec(None, 'replica', 'mdl', 'data')),
            'w_quantized_scale':
                base_layer.BoxedPartitionSpec(
                    meta=pjit.PartitionSpec(None, 'mdl', 'data'))
        }
    }
    self.assertEqual(pspec, exepected_pspec)

if __name__ == '__main__':
  absltest.main()
