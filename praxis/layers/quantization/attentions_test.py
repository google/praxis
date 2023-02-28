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

"""Tests for quantized attentions."""

import itertools
from typing import Any, Dict, Sequence

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import pax_fiddle
from praxis import py_utils
from praxis import test_utils
from praxis.layers import attentions
from praxis.layers.quantization import attentions as qattentions
from praxis.layers.quantization import quantization_hparams

QuantizationHParams = quantization_hparams.QuantizationHParams
QuantizationMode = quantization_hparams.QuantizationMode
QuantizationType = quantization_hparams.QuantizationType
instantiate = base_layer.instantiate


def _generate_quantization_types_modes() -> Sequence[Dict[str, Any]]:
  keys = ['testcase_name', 'quantization_type', 'mode', 'precision']
  types = [QuantizationType.PTQ, QuantizationType.AQT]
  modes = [QuantizationMode.INFERENCE, QuantizationMode.TRAINING]
  precisions = [8, 4]

  cases = []
  for case in itertools.product(types, modes, precisions):
    name = case[0].value + '_' + case[1].value + '_' + str(case[2])
    cases.append([name] + list(case))
  return [dict(zip(keys, case)) for case in cases]


class QuantizedAttentionTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  @parameterized.named_parameters(_generate_quantization_types_modes())
  def test_attention_projection_quantized(
      self, quantization_type, mode, precision
  ):
    p = pax_fiddle.Config(
        qattentions.AttentionProjection,
        name='_attn_proj',
        input_dim=8,
        num_heads=2,
        dim_per_head=3,
        is_output_projection=True,
        quantization=QuantizationHParams(
            quantization_type=quantization_type,
            mode=mode,
            weight_params=quantization_hparams.WeightQuantizationParams(
                precision=precision,
            ),
        ),
    )
    attn = instantiate(p)
    inputs = jnp.ones((4, 2, 3), dtype=p.dtype)
    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = attn.init(prng_key, inputs)
      outputs = attn.apply(initial_vars, inputs)
    self.assertEqual(outputs.shape, (4, 8))
    if mode == QuantizationMode.INFERENCE:
      self.assertAllClose(jnp.full((4, 8), 0.0), outputs)
    else:
      self.assertRaises(AssertionError, self.assertAllClose,
                        jnp.full((4, 8), 0.0), outputs)

  @parameterized.named_parameters(_generate_quantization_types_modes())
  def test_attention_projection_no_output_proj_quantized(
      self, quantization_type, mode, precision
  ):
    p = pax_fiddle.Config(
        qattentions.AttentionProjection,
        name='_attn_proj',
        input_dim=8,
        num_heads=2,
        dim_per_head=3,
        is_output_projection=False,
        quantization=QuantizationHParams(
            quantization_type=quantization_type,
            mode=mode,
            weight_params=quantization_hparams.WeightQuantizationParams(
                precision=precision,
            ),
        ),
    )
    attn = instantiate(p)
    inputs = jnp.ones((4, 3, 8), dtype=p.dtype)
    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = attn.init(prng_key, inputs)
      outputs = attn.apply(initial_vars, inputs)
    self.assertEqual(outputs.shape, (4, 3, 2, 3))
    if mode == QuantizationMode.INFERENCE:
      self.assertAllClose(jnp.full((4, 3, 2, 3), 0.0), outputs)
    else:
      self.assertRaises(AssertionError, self.assertAllClose,
                        jnp.full((4, 3, 2, 3), 0.0), outputs)

  @parameterized.named_parameters(_generate_quantization_types_modes())
  def test_combined_projection_quantized(
      self, quantization_type, mode, precision
  ):
    p = pax_fiddle.Config(
        qattentions.CombinedQKVProjectionLayer,
        name='_combined_qkv',
        input_dim=8,
        num_heads=3,
        dim_per_head=2,
        quantization=QuantizationHParams(
            quantization_type=quantization_type,
            mode=mode,
            weight_params=quantization_hparams.WeightQuantizationParams(
                precision=precision,
            ),
        ),
    )
    attn = instantiate(p)
    inputs = jnp.ones((4, 8), dtype=p.dtype)
    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = attn.init(prng_key, inputs)
      out_q, out_k, out_v = attn.apply(initial_vars, inputs)
    expected_shape = (4, 3, 2)
    self.assertEqual(out_q.shape, expected_shape)
    self.assertEqual(out_k.shape, expected_shape)
    self.assertEqual(out_v.shape, expected_shape)
    if mode == QuantizationMode.INFERENCE:
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
    p_f = pax_fiddle.Config(attentions.AttentionProjection, name='_attn_proj_f')
    p_q = pax_fiddle.Config(
        qattentions.AttentionProjection,
        name='_attn_proj_q',
        quantization=QuantizationHParams(mode=QuantizationMode.TRAINING),
    )
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
    p_f = pax_fiddle.Config(attentions.AttentionProjection, name='_attn_proj_f')
    p_q = pax_fiddle.Config(
        qattentions.AttentionProjection,
        name='_attn_proj_q',
        quantization=QuantizationHParams(mode=QuantizationMode.TRAINING),
    )
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
    p_f = pax_fiddle.Config(attentions.AttentionProjection, name='_attn_proj_f')
    p_q = pax_fiddle.Config(
        qattentions.AttentionProjection,
        name='_attn_proj_q',
        quantization=QuantizationHParams(mode=QuantizationMode.TRAINING),
    )
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
    p_f = pax_fiddle.Config(
        attentions.CombinedQKVProjectionLayer, name='_attn_qkv_f'
    )
    p_q = pax_fiddle.Config(
        qattentions.CombinedQKVProjectionLayer,
        name='_attn_qkv_q',
        quantization=QuantizationHParams(mode=QuantizationMode.TRAINING),
    )
    for p in [p_f, p_q]:
      p.input_dim = 64
      p.num_heads = 8
      p.dim_per_head = 8
      p.attention_combine_dims = True

    inputs = np.random.normal(size=[3, 64]).astype(np.float32)
    self.run_and_compare(p_f, p_q, inputs)

  @parameterized.parameters([
      (False, True, 3, True, True, True),
      (True, True, 3, True, True, False),
      (False, True, 3, True, False, False),
      (True, True, 3, True, False, True),
      (False, True, 4, False, False, True),
      (True, True, 4, True, False, True),
      (False, False, 1, False, False, False),
      (True, False, 1, True, False, False),
      (False, False, 1, True, False, False),
      (True, False, 1, True, False, True),
  ])
  def test_mha_01_quantized(
      self,
      combine_qkv,
      dconv_qkv,
      dconv_kernel_size,
      use_rotary_position_emb,
      simulate_packed,
      zero_fully_masked,
  ):
    # Test case copied and modified from test_mha_01.
    mdl_dim = 16
    hidden_dim = 32
    num_heads = 4
    atten_f_p = pax_fiddle.Config(
        attentions.DotProductAttention,
        name='mh',
    )
    atten_q_p = pax_fiddle.Config(
        qattentions.DotProductAttention,
        name='mh_quant',
        quantization=QuantizationHParams(
            quantization_type=QuantizationType.AQT,
            mode=QuantizationMode.TRAINING,
            # Test using 23 bits to minimize the quantization error and test
            # for numerical correctness.
            act_params=quantization_hparams.ActQuantizationParams(precision=23),
            weight_params=None,
        ),
    )
    for p in [atten_f_p, atten_q_p]:
      p.input_dim = mdl_dim
      p.hidden_dim = hidden_dim
      p.num_heads = num_heads
      p.dim_per_head = 16 if use_rotary_position_emb else None
      p.atten_logit_cap = 20.0
      p.combine_qkv = combine_qkv
      p.dconv_qkv = dconv_qkv
      p.dconv_kernel_size = dconv_kernel_size
      p.use_rotary_position_emb = use_rotary_position_emb
      p.zero_fully_masked = zero_fully_masked
    atten_f = instantiate(atten_f_p)
    atten_q = instantiate(atten_q_p)

    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)
    target_batch_size = 3
    source_max_length = 16
    target_max_length = 16
    query_vec = np.random.normal(
        size=[target_batch_size, source_max_length, mdl_dim]
    ).astype(np.float32)
    key_vec = query_vec
    value_vec = query_vec
    fake_query_vec = jnp.zeros_like(query_vec)
    atten_mask = attentions.causal_mask(query_vec)
    segment_pos = np.tile(np.arange(source_max_length), (target_batch_size, 1))

    starting_index = 0
    if simulate_packed:
      starting_index = dconv_kernel_size
      atten_mask = atten_mask.at[:, :, :, :starting_index].set(-2.3819763e38)
      segment_pos = jnp.maximum(segment_pos - starting_index, 0)

    with base_layer.JaxContext.new_context():
      initial_vars = atten_f.init(
          init_key,
          fake_query_vec,
          fake_query_vec,
          fake_query_vec,
          atten_mask,
          query_segment_pos=segment_pos,
          key_segment_pos=segment_pos,
      )
      logging.info('initial_vars: %s', initial_vars)
      fprop_out_f, _ = atten_f.apply(
          initial_vars,
          query_vec,
          key_vec,
          value_vec,
          atten_mask,
          query_segment_pos=segment_pos,
          key_segment_pos=segment_pos,
          method=atten_f.__call__,
      )
      fprop_out_q, _ = atten_q.apply(
          initial_vars,
          query_vec,
          key_vec,
          value_vec,
          atten_mask,
          query_segment_pos=segment_pos,
          key_segment_pos=segment_pos,
          method=atten_q.__call__,
      )
      self.assertAllClose(fprop_out_q, fprop_out_f)

      # Compute the quantized extend_step result to compare against the standard
      # floating point implementation.
      _, attention_states_q = atten_q.apply(
          initial_vars,
          fake_query_vec,
          fake_query_vec,
          fake_query_vec,
          atten_mask,
          query_segment_pos=segment_pos,
          key_segment_pos=segment_pos,
          method=atten_q.__call__,
          mutable=[base_layer.DECODE_CACHE],
      )
      decoder_output_q = jnp.zeros(
          shape=[target_max_length, target_batch_size, mdl_dim]
      )
      updated_vars = py_utils.merge_dict(attention_states_q, initial_vars)
      for t in range(starting_index, target_max_length):
        encoded, attention_states = atten_q.apply(
            updated_vars,
            query_vec=query_vec[:, t, :],
            atten_mask=atten_mask[:, :, t, :],
            time_step=t,
            segment_pos=None,
            method=atten_q.extend_step,
            mutable=[base_layer.DECODE_CACHE],
        )
        updated_vars = py_utils.merge_dict(attention_states, initial_vars)
        decoder_output_q = decoder_output_q.at[t].set(encoded)

      decoder_output_q = decoder_output_q[starting_index:]
      decoder_out_transposed_q = jnp.transpose(decoder_output_q, [1, 0, 2])
      fprop_out_f = fprop_out_f[:, starting_index:]

      logging.info('fprop_out: %s', fprop_out_f)
      logging.info('decoder_out: %s', decoder_output_q)
      self.assertAllClose(fprop_out_f, decoder_out_transposed_q)


class QuantizeAttentionTest(test_utils.TestCase):
  """Quantize attention."""

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  @parameterized.named_parameters(
      ('PTQ', QuantizationType.PTQ),
      ('AQT', QuantizationType.AQT),
  )
  def test_quantize_attention_projection(self, quantization_type):
    p = pax_fiddle.Config(
        qattentions.AttentionProjection,
        name='_attn_proj_q',
        mesh_axis_names=['replica', 'mdl', 'data'],
        weight_split_dims_mapping=base_layer.BaseLayer.WeightSharding(
            wt=['mdl', 'data']
        ),
        quantization=QuantizationHParams(
            quantization_type=quantization_type,
            mode=QuantizationMode.TRAINING,
        ),
    )
    p.input_dim = 16
    p.num_heads = 2
    p.dim_per_head = 5
    p.is_output_projection = True
    p.use_nhd_shape = True
    layer = instantiate(p)
    inputs = np.random.normal(1.5, 2.0, [5, 2, 5]).astype(np.float32)

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = layer.init(prng_key, inputs)
      res, _ = layer.apply(
          initial_vars, mutable=[], method=layer.quantize_weight)

    self.assertEqual(len(res), 1)
    self.assertEqual(len(res[base_layer.PARAMS]), 2)
    self.assertEqual(res[base_layer.PARAMS]['w'].shape, (2, 5, 16))
    self.assertEqual(res[base_layer.PARAMS]['w_quantized_scale'].shape, (16,))

    pspec, _ = layer.apply(
        initial_vars, mutable=[], method=layer.quantized_partition_specs
    )
    expected_pspec = {
        'params': {
            'w': base_layer.BoxedPartitionSpec(
                meta=jax.sharding.PartitionSpec('mdl', 'data')
            ),
            'w_quantized_scale': base_layer.BoxedPartitionSpec(
                meta=jax.sharding.PartitionSpec('mdl')
            ),
        }
    }
    self.assertEqual(pspec, expected_pspec)

  @parameterized.named_parameters(
      ('PTQ_symmetric', QuantizationType.PTQ, True),
      ('AQT_symmetric', QuantizationType.AQT, True),
      ('PTQ_asymmetric', QuantizationType.PTQ, False),
      ('AQT_asymmetric', QuantizationType.AQT, False),
  )
  def test_quantize_attention_qkv(self, quantization_type, use_symmetric):
    p = pax_fiddle.Config(
        qattentions.CombinedQKVProjectionLayer,
        name='_combined_qkv',
        input_dim=5,
        num_heads=6,
        dim_per_head=2,
        ici_mesh_shape=[0, 1, 2],
        mesh_axis_names=['replica', 'mdl', 'data'],
        weight_split_dims_mapping=base_layer.BaseLayer.WeightSharding(
            wt=['replica', 'mdl', 'data']
        ),
        quantization=QuantizationHParams(
            quantization_type=quantization_type,
            mode=QuantizationMode.TRAINING,
            weight_params=quantization_hparams.WeightQuantizationParams(
                use_symmetric=use_symmetric
            ),
        ),
    )
    layer = instantiate(p)
    inputs = jnp.ones((4, 5), dtype=p.dtype)

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = layer.init(prng_key, inputs)
      res, _ = layer.apply(
          initial_vars, mutable=[], method=layer.quantize_weight)
      pspec, _ = layer.apply(
          initial_vars, mutable=[], method=layer.quantized_partition_specs
      )

    self.assertEqual(len(res), 1)

    shapes = jax.tree_map(lambda x: x.shape, res)
    types = jax.tree_map(lambda x: x.dtype, res)

    expected_shape = {
        base_layer.PARAMS: {'w': (3, 5, 6, 2), 'w_quantized_scale': (3, 6, 2)}
    }
    expected_types = {
        base_layer.PARAMS: {'w': jnp.int8, 'w_quantized_scale': p.dtype}
    }
    expected_pspec = {
        'params': {
            'w': base_layer.BoxedPartitionSpec(
                meta=jax.sharding.PartitionSpec(None, 'replica', 'mdl', 'data')
            ),
            'w_quantized_scale': base_layer.BoxedPartitionSpec(
                meta=jax.sharding.PartitionSpec(None, 'mdl', 'data')
            ),
        }
    }

    if not use_symmetric:
      expected_shape[base_layer.PARAMS]['w_quantized_zp'] = (3, 6, 2)
      expected_types[base_layer.PARAMS]['w_quantized_zp'] = p.dtype
      expected_pspec['params']['w_quantized_zp'] = (
          base_layer.BoxedPartitionSpec(
              meta=jax.sharding.PartitionSpec(None, 'mdl', 'data')
          )
      )

    self.assertEqual(shapes, expected_shape)
    self.assertEqual(types, expected_types)
    self.assertEqual(pspec, expected_pspec)


if __name__ == '__main__':
  absltest.main()
