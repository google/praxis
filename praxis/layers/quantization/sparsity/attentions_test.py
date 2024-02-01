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

"""Tests for sparse linears."""

import itertools
from typing import Any, Sequence

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
from praxis.layers.quantization import attentions as sattentions
from praxis.layers.quantization.sparsity import sparsity_hparams
from praxis.layers.quantization.sparsity import sparsity_modes

NON_TRAINABLE = base_layer.NON_TRAINABLE
SPARSITY_NAME_POSTFIX = base_layer.SPARSITY_NAME_POSTFIX
WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams
SparsityHParams = sparsity_hparams.SparsityHParams
WeightSparsityParams = sparsity_hparams.WeightSparsityParams
SparsityType = sparsity_hparams.SparsityType
InferenceMode = sparsity_modes.InferenceMode
MaterializeMode = sparsity_modes.MaterializeMode
TrainingMode = sparsity_modes.TrainingMode
FewShotMode = sparsity_modes.FewShotMode

instantiate = base_layer.instantiate


def _generate_sparsity_types_modes() -> Sequence[dict[str, Any]]:
  keys = ['testcase_name', 'sparsity_type', 'mode_name', 'mode']
  types = [SparsityType.STRUCTURED_NM]
  modes = [
      ('inference_mode', pax_fiddle.Config(InferenceMode)),
      ('materialize_mode', pax_fiddle.Config(MaterializeMode)),
      ('training_mode', pax_fiddle.Config(TrainingMode)),
  ]
  cases = []
  for case in itertools.product(types, modes):
    name = case[0].value + '_' + case[1][0]
    cases.append([name, case[0], case[1][0], case[1][1]])

  return [dict(zip(keys, case)) for case in cases]


class SparseAttentionTest(test_utils.TestCase):
  """Check the functionality of structured sparsity."""

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  @parameterized.named_parameters(_generate_sparsity_types_modes())
  def test_attention_projection_sparse(self, sparsity_type, mode_name, mode):
    p = pax_fiddle.Config(
        sattentions.AttentionProjection,
        name='_attn_proj',
        input_dim=2,
        num_heads=2,
        dim_per_head=4,
        quantization=None,
        sparsity=pax_fiddle.Config(
            SparsityHParams,
            sparsity_type=sparsity_type,
            weight_params=WeightSparsityParams(prune_rate=(2, 4)),
            mode=mode,
            order='R',
        ),
    )
    attn = instantiate(p)
    inputs = jnp.ones((1, 1, 2), dtype=p.dtype)
    weights = jnp.array([
        [
            [1, 2, 3, 4],
            [-3, -4, 1, 2],
        ],
        [
            [3, 1, -4, 2],
            [-3, 1, 2, -4],
        ],
    ])
    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = attn.init(prng_key, inputs)
      initial_vars['params']['w'] = weights
      # Materialize mode do not making pruning or mask updating.
      if mode_name == 'materialize_mode':
        initial_vars[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX] = jnp.array([
            [
                [False, False, True, True],
                [True, True, False, False],
            ],
            [
                [True, False, True, False],
                [True, False, False, True],
            ],
        ])
      outputs, state = attn.apply(initial_vars, inputs, mutable=True)
    self.assertEqual(outputs.shape, (1, 1, 2, 4))
    if mode_name != 'inference_mode':
      self.assertArraysEqual(
          state[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
          jnp.array([
              [
                  [False, False, True, True],
                  [True, True, False, False],
              ],
              [
                  [True, False, True, False],
                  [True, False, False, True],
              ],
          ]),
      )
      self.assertArraysEqual(
          outputs,
          jnp.array([[[[3.0, 0.0, -1.0, 4.0], [-6.0, -4.0, 0.0, -4.0]]]]),
      )
    else:
      self.assertArraysEqual(
          outputs,
          jnp.array([[[[4.0, 3.0, -1.0, 6.0], [-6.0, -3.0, 3.0, -2.0]]]]),
      )

  def test_few_shot_with_mask_update_interval(self):
    p = pax_fiddle.Config(
        sattentions.AttentionProjection,
        name='_attn_proj',
        input_dim=2,
        num_heads=2,
        dim_per_head=4,
        quantization=None,
        sparsity=pax_fiddle.Config(
            SparsityHParams,
            sparsity_type=SparsityType.STRUCTURED_NM,
            weight_params=WeightSparsityParams(prune_rate=(2, 4)),
            mode=pax_fiddle.Config(
                FewShotMode, num_shots=2, mask_update_interval=2
            ),  # Update mask every 2 steps
            order='R',
        ),
    )
    attn = instantiate(p)
    inputs = jnp.ones((1, 1, 2), dtype=p.dtype)
    weights = jnp.array([
        [
            [1, 2, 3, 4],
            [-3, -4, 1, 2],
        ],
        [
            [3, 1, -4, 2],
            [-3, 1, 2, -4],
        ],
    ])
    # Init and Step 0
    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = attn.init(prng_key, inputs)
      initial_vars['params']['w'] = weights
      self.assertEqual(initial_vars[NON_TRAINABLE]['step'], 0)
      outputs, state = attn.apply(initial_vars, inputs, mutable=True)
    self.assertEqual(outputs.shape, (1, 1, 2, 4))
    self.assertArraysEqual(
        state[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
        jnp.array([
            [
                [False, False, True, True],
                [True, True, False, False],
            ],
            [
                [True, False, True, False],
                [True, False, False, True],
            ],
        ]),
    )
    # Step 1, mask remains unchanged, even if we update weight matrix
    weights = jnp.array([
        [
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ],
        [
            [3, 1, -4, 2],
            [-3, 1, 2, -4],
        ],
    ])
    with base_layer.JaxContext.new_context():
      state['params']['w'] = weights
      self.assertEqual(state[NON_TRAINABLE]['step'], 1)
      outputs, state = attn.apply(state, inputs, mutable=True)
    self.assertArraysEqual(
        state[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
        jnp.array([
            [
                [False, False, True, True],
                [True, True, False, False],
            ],
            [
                [True, False, True, False],
                [True, False, False, True],
            ],
        ]),
    )
    # Step 2, mask changes
    with base_layer.JaxContext.new_context():
      self.assertEqual(state[NON_TRAINABLE]['step'], 2)
      outputs, state = attn.apply(state, inputs, mutable=True)
    self.assertArraysEqual(
        state[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
        jnp.array([
            [
                [False, False, True, True],
                [False, False, True, True],
            ],
            [
                [True, False, True, False],
                [True, False, False, True],
            ],
        ]),
    )

  @parameterized.named_parameters(_generate_sparsity_types_modes())
  def test_combine_qkv_with_attention_combine_dims(
      self, sparsity_type, mode_name, mode
  ):
    input_dim = 2
    dim_per_head = 4
    num_heads = 2
    # Reference combine qkv projection layer.
    ref_proj_p = pax_fiddle.Config(
        sattentions.CombinedQKVProjectionLayer,
        name='ref',
        input_dim=input_dim,
        dim_per_head=dim_per_head,
        num_heads=num_heads,
        quantization=None,
        sparsity=pax_fiddle.Config(
            SparsityHParams,
            sparsity_type=sparsity_type,
            weight_params=WeightSparsityParams(prune_rate=(2, 4)),
            mode=mode,
        ),
    )
    proj = instantiate(ref_proj_p)

    # Combine attention dim combine qkv projection layer.
    combine_proj_p = pax_fiddle.Config(
        sattentions.CombinedQKVProjectionLayer,
        name='ref',
        input_dim=input_dim,
        dim_per_head=dim_per_head,
        num_heads=num_heads,
        attention_combine_dims=True,
        quantization=None,
        sparsity=pax_fiddle.Config(
            SparsityHParams,
            sparsity_type=sparsity_type,
            weight_params=WeightSparsityParams(prune_rate=(2, 4)),
            mode=mode,
            order='R',
        ),
    )
    combine_proj = instantiate(combine_proj_p)

    batch_size = 3
    inputs = np.random.normal(size=[batch_size, input_dim]).astype(np.float32)
    with base_layer.JaxContext.new_context():
      # Set up initial vars for combine attention dim projection.
      prng_key = jax.random.PRNGKey(seed=123)
      prng_key, init_key = jax.random.split(prng_key)
      initial_vars = proj.init(init_key, inputs)

      combine_initial_vars = combine_proj.init(init_key, inputs)
      combine_initial_vars['params']['w'] = np.reshape(
          initial_vars['params']['w'], (3, input_dim, num_heads * dim_per_head))
      combine_initial_vars['params']['b'] = np.reshape(
          initial_vars['params']['b'], (3, num_heads * dim_per_head))

      q_proj_ref, k_proj_ref, v_proj_ref = proj.apply(initial_vars, inputs)
      q_proj_combine, k_proj_combine, v_proj_combine = combine_proj.apply(
          combine_initial_vars, inputs)

    self.assertAllClose(q_proj_ref, q_proj_combine)
    self.assertAllClose(k_proj_ref, k_proj_combine)
    self.assertAllClose(v_proj_ref, v_proj_combine)


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


class AttentionLayersConsistencyTest(test_utils.TestCase):
  """Consistency check fo sparse Attention and base Praxis Attention layers.

  The weights in both layers must be identical when running in
    mode={INFERENCE} when no sparsification applied.
  """

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  def run_and_compare(self, p_f, p_s, inputs):
    atten_proj_f = instantiate(p_f)
    atten_proj_s = instantiate(p_s)

    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars_f = atten_proj_f.init(prng_key, inputs)
    initial_vars_s = atten_proj_s.init(prng_key, inputs)
    # initial_vars_f['params']['w'] = weights
    # initial_vars_s['params']['w'] = weights

    outputs_f = atten_proj_f.apply(initial_vars_f, inputs)
    outputs_s = atten_proj_s.apply(initial_vars_s, inputs)
    self.assertAllClose(outputs_f, outputs_s)

  # test case copied from test_mhd_projection_01.
  def test_mhd_projection_01_sparsified(self):
    p_f = pax_fiddle.Config(attentions.AttentionProjection, name='_attn_proj_f')
    p_s = pax_fiddle.Config(
        sattentions.AttentionProjection,
        name='_attn_proj_s',
        quantization=None,
        sparsity=pax_fiddle.Config(
            SparsityHParams,
            sparsity_type=SparsityType.STRUCTURED_NM,
            weight_params=WeightSparsityParams(prune_rate=(2, 4)),
            mode=pax_fiddle.Config(InferenceMode),
        ),
    )
    for p in [p_f, p_s]:
      p.input_dim = 16
      p.num_heads = 2
      p.dim_per_head = 5
      p.is_output_projection = False

    inputs = np.random.normal(1.5, 2.0, [5, 16]).astype(np.float32)
    self.run_and_compare(p_f, p_s, inputs)

  # test case copied from test_mhd_projection_02.
  @parameterized.parameters([False, True])
  def test_mhd_projection_02_sparsified(self, use_nhd_shape):
    p_f = pax_fiddle.Config(attentions.AttentionProjection, name='_attn_proj_f')
    p_s = pax_fiddle.Config(
        sattentions.AttentionProjection,
        name='_attn_proj_s',
        quantization=None,
        sparsity=pax_fiddle.Config(
            SparsityHParams,
            sparsity_type=SparsityType.STRUCTURED_NM,
            weight_params=WeightSparsityParams(prune_rate=(2, 4)),
            mode=pax_fiddle.Config(InferenceMode),
        ),
    )
    for p in [p_f, p_s]:
      p.input_dim = 16
      p.num_heads = 2
      p.dim_per_head = 5
      p.is_output_projection = True
      p.use_nhd_shape = use_nhd_shape

    inputs = np.random.normal(1.5, 2.0, [5, 2, 5]).astype(np.float32)
    self.run_and_compare(p_f, p_s, inputs)

  # test case copied from test_mhd_projection_var_stats.
  def test_mhd_projection_var_stats_sparsified(self):
    p_f = pax_fiddle.Config(attentions.AttentionProjection, name='_attn_proj_f')
    p_s = pax_fiddle.Config(
        sattentions.AttentionProjection,
        name='_attn_proj_s',
        quantization=None,
        sparsity=pax_fiddle.Config(
            SparsityHParams,
            sparsity_type=SparsityType.STRUCTURED_NM,
            weight_params=WeightSparsityParams(prune_rate=(2, 4)),
            mode=pax_fiddle.Config(InferenceMode),
        ),
    )
    for p in [p_f, p_s]:
      p.input_dim = 256
      p.num_heads = 16
      p.dim_per_head = 16
      p.is_output_projection = True

    attn_f = instantiate(p_f)
    attn_s = instantiate(p_s)
    inputs = np.random.normal(1.5, 2.0, [2, 16, 16]).astype(np.float32)
    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)
    initial_vars_f = attn_f.init(init_key, inputs)
    initial_vars_s = attn_s.init(init_key, inputs)
    assert_var_stats_close(
        py_utils.NestedMap.FromNestedDict(initial_vars_f['params']),
        py_utils.NestedMap.FromNestedDict(initial_vars_s['params']), self)

  # test case copied from test_combine_qkv_with_attention_combine_dims.
  def test_combine_qkv_with_attention_combine_dims_quantized(self):
    p_f = pax_fiddle.Config(
        attentions.CombinedQKVProjectionLayer, name='_attn_qkv_f'
    )
    p_s = pax_fiddle.Config(
        sattentions.CombinedQKVProjectionLayer,
        name='_attn_qkv_s',
        quantization=None,
        sparsity=pax_fiddle.Config(
            SparsityHParams,
            sparsity_type=SparsityType.STRUCTURED_NM,
            weight_params=WeightSparsityParams(prune_rate=(2, 4)),
            mode=pax_fiddle.Config(InferenceMode),
        ),
    )
    for p in [p_f, p_s]:
      p.input_dim = 64
      p.num_heads = 8
      p.dim_per_head = 8
      p.attention_combine_dims = True

    inputs = np.random.normal(size=[3, 64]).astype(np.float32)
    self.run_and_compare(p_f, p_s, inputs)


if __name__ == '__main__':
  absltest.main()
