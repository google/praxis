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

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import pax_fiddle
from praxis import test_utils
from praxis.layers import linears
from praxis.layers.quantization import linears as slinears
from praxis.layers.quantization.sparsity import sparsity_hparams
from praxis.layers.quantization.sparsity import sparsity_modes

instantiate = base_layer.instantiate
NON_TRAINABLE = base_layer.NON_TRAINABLE
PARAMS = base_layer.PARAMS
SPARSITY_NAME_POSTFIX = base_layer.SPARSITY_NAME_POSTFIX
SUMMARIES = base_layer.SUMMARIES
WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams
WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams
SparsityHParams = sparsity_hparams.SparsityHParams
WeightSparsityParams = sparsity_hparams.WeightSparsityParams
SparsityType = sparsity_hparams.SparsityType
InferenceMode = sparsity_modes.InferenceMode
MaterializeMode = sparsity_modes.MaterializeMode
TrainingMode = sparsity_modes.TrainingMode
FewShotMode = sparsity_modes.FewShotMode


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


class SparseLinearTest(test_utils.TestCase):
  """Check the functionality of structured sparsity."""

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  @parameterized.named_parameters(_generate_sparsity_types_modes())
  def test_linear_sparse(self, sparsity_type, mode_name, mode):
    p = pax_fiddle.Config(
        slinears.Linear,
        name='_linear',
        input_dims=4,
        output_dims=4,
        quantization=None,
        sparsity=pax_fiddle.Config(
            SparsityHParams,
            sparsity_type=sparsity_type,
            weight_params=WeightSparsityParams(prune_rate=(2, 4)),
            mode=mode,
            order='R',
        ),
    )
    linear = instantiate(p)
    inputs = jnp.array([[1, 2, 3, 4], [6, 7, 8, 9]], dtype=p.dtype)
    weights = jnp.array([
        [1, 2, 3, 4],
        [-3, -4, 1, 2],
        [3, 1, -4, 2],
        [-3, 1, 2, -4],
    ])
    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = linear.init(prng_key, inputs)
      initial_vars[PARAMS]['w'] = weights

      if mode_name != 'inference_mode':
        self.assertArraysEqual(
            initial_vars[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
            jnp.array([
                [True, True, True, True],
                [True, True, True, True],
                [True, True, True, True],
                [True, True, True, True],
            ]),
        )

      # Materialize mode do not making pruning or mask updating.
      if mode_name == 'materialize_mode':
        initial_vars[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX] = jnp.array([
            [False, False, True, True],
            [True, True, False, False],
            [True, False, True, False],
            [True, False, False, True],
        ])
      outputs, state = linear.apply(initial_vars, inputs, mutable=True)
    self.assertEqual(outputs.shape, (2, 4))
    if mode_name != 'inference_mode':
      self.assertArraysEqual(
          state[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
          jnp.array([
              [False, False, True, True],
              [True, True, False, False],
              [True, False, True, False],
              [True, False, False, True],
          ]),
      )
      self.assertArraysEqual(
          outputs,
          jnp.array([[-9.0, -8.0, -9.0, -12.0], [-24.0, -28.0, -14.0, -12.0]]),
      )
    else:
      self.assertArraysEqual(
          outputs, jnp.array([[-8.0, 1.0, 1.0, -2.0], [-18.0, 1.0, 11.0, 18.0]])
      )

  def test_few_shot_with_mask_update_interval(self):
    p = pax_fiddle.Config(
        slinears.Linear,
        name='_linear',
        input_dims=4,
        output_dims=4,
        quantization=None,
        sparsity=pax_fiddle.Config(
            SparsityHParams,
            sparsity_type=SparsityType.STRUCTURED_NM,
            order='R',
            weight_params=WeightSparsityParams(prune_rate=(2, 4)),
            mode=pax_fiddle.Config(
                FewShotMode, num_shots=2, mask_update_interval=2, target_step=0
            ),  # Update mask every 2 steps
        ),
    )
    linear = instantiate(p)
    inputs = jnp.array([[1, 2, 3, 4], [6, 7, 8, 9]], dtype=p.dtype)
    weights = jnp.array([
        [1, 2, 3, 4],
        [-3, -4, 1, 2],
        [3, 1, -4, 2],
        [-3, 1, 2, -4],
    ])

    context_p = base_layer.JaxContext.HParams(summary_verbosity=4)
    # Init and Step 0
    with base_layer.JaxContext.new_context(hparams=context_p):
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = linear.init(prng_key, inputs)
      initial_vars[PARAMS]['w'] = weights
      self.assertEqual(initial_vars[NON_TRAINABLE]['step'], 0)
      outputs, state = linear.apply(initial_vars, inputs, mutable=True)
    self.assertEqual(outputs.shape, (2, 4))
    self.assertArraysEqual(
        state[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
        jnp.array([
            [False, False, True, True],
            [True, True, False, False],
            [True, False, True, False],
            [True, False, False, True],
        ]),
    )
    self.assertArraysEqual(
        state[SUMMARIES]['mask_update_count_scalar'],
        jnp.array(1, jnp.int32))

    # Step 1, mask remains unchanged, even if we update weight matrix
    weights = jnp.array([
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [3, 1, -4, 2],
        [-3, 1, 2, -4],
    ])
    with base_layer.JaxContext.new_context(hparams=context_p):
      state[PARAMS]['w'] = weights
      self.assertEqual(state[NON_TRAINABLE]['step'], 1)
      outputs, state = linear.apply(state, inputs, mutable=True)
    self.assertArraysEqual(
        state[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
        jnp.array([
            [False, False, True, True],
            [True, True, False, False],
            [True, False, True, False],
            [True, False, False, True],
        ]),
    )
    self.assertArraysEqual(
        state[SUMMARIES]['mask_update_count_scalar'],
        jnp.array(1, jnp.int32))

    # Step 2, mask changes
    with base_layer.JaxContext.new_context(hparams=context_p):
      self.assertEqual(state[NON_TRAINABLE]['step'], 2)
      outputs, state = linear.apply(state, inputs, mutable=True)
    self.assertArraysEqual(
        state[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
        jnp.array([
            [False, False, True, True],
            [False, False, True, True],
            [True, False, True, False],
            [True, False, False, True],
        ]),
    )
    self.assertArraysEqual(
        state[SUMMARIES]['mask_update_count_scalar'],
        jnp.array(2, jnp.int32))

  def test_sparsity_hparams_asserts(self):
    with self.assertRaises(AssertionError):
      instantiate(
          pax_fiddle.Config(
              SparsityHParams,
              sparsity_type=SparsityType.STRUCTURED_NM,
              weight_params=WeightSparsityParams(prune_rate=0.2),
              mode=pax_fiddle.Config(InferenceMode),
              order='R',
          )
      )

    with self.assertRaises(AssertionError):
      instantiate(
          pax_fiddle.Config(
              SparsityHParams,
              sparsity_type=SparsityType.UNSTRUCTURED,
              weight_params=WeightSparsityParams(prune_rate=(2, 4)),
              mode=pax_fiddle.Config(InferenceMode),
              order='R',
          )
      )


class LinearLayersConsistencyTest(test_utils.TestCase):
  """Consistency check fo sparse linear and base Praxis linear layers.

  The weights in both layers must be identical when running in
    mode={INFERENCE} when no sparsification applied.
  """

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  def _run_and_compare(self, p_f, p_s, inputs):
    linear_f = instantiate(p_f)
    linear_s = instantiate(p_s)

    weights = jnp.array([
        [1, 2, 3, 4],
        [-3, -4, 1, 2],
        [3, 1, -4, 2],
        [-3, 1, 2, -4],
    ])
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars_f = linear_f.init(prng_key, inputs)
    initial_vars_s = linear_s.init(prng_key, inputs)
    initial_vars_f[PARAMS]['w'] = weights
    initial_vars_s[PARAMS]['w'] = weights

    outputs_f = linear_f.apply(initial_vars_f, inputs)
    outputs_s = linear_s.apply(initial_vars_s, inputs)
    self.assertAllClose(outputs_f, outputs_s)

  def test_linear_inference_before_sparsification(self):
    p_f = pax_fiddle.Config(linears.Linear, name='_linear_f')
    p_s = pax_fiddle.Config(
        slinears.Linear,
        name='_linear',
        quantization=None,
        sparsity=pax_fiddle.Config(
            SparsityHParams,
            sparsity_type=SparsityType.STRUCTURED_NM,
            weight_params=WeightSparsityParams(prune_rate=(2, 4)),
            mode=pax_fiddle.Config(InferenceMode),
            order='R',
        ),
    )
    for p in [p_f, p_s]:
      p.input_dims = 4
      p.output_dims = 4
    inputs = np.random.normal(1.5, 2.0, [5, 4]).astype(np.float32)
    self._run_and_compare(p_f, p_s, inputs)


if __name__ == '__main__':
  absltest.main()
