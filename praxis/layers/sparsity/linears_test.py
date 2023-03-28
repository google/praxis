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
from praxis.layers.sparsity import linears as slinears
from praxis.layers.sparsity import sparsity_hparams

instantiate = base_layer.instantiate
WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams
SparsityHParams = sparsity_hparams.SparsityHParams
WeightSparsityParams = sparsity_hparams.WeightSparsityParams
SparsityMode = sparsity_hparams.SparsityMode
SparsityType = sparsity_hparams.SparsityType


def _generate_sparsity_types_modes() -> Sequence[Dict[str, Any]]:
  keys = ['testcase_name', 'sparsity_type', 'mode']
  types = [SparsityType.STRUCTURED_NM]
  modes = [
      SparsityMode.INFERENCE,
      SparsityMode.MATERIALIZE,
      SparsityMode.TRAINING,
  ]
  cases = []
  for case in itertools.product(types, modes):
    name = case[0].value + '_' + case[1].value
    cases.append([name] + list(case))

  return [dict(zip(keys, case)) for case in cases]


class SparseLinearTest(test_utils.TestCase):
  """Check the functionality of structured sparsity."""

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  @parameterized.named_parameters(_generate_sparsity_types_modes())
  def test_linear_sparse(self, sparsity_type, mode):
    p = pax_fiddle.Config(
        slinears.Linear,
        name='_linear',
        input_dims=4,
        output_dims=4,
        sparsity=SparsityHParams(
            sparsity_type=sparsity_type,
            weight_params=WeightSparsityParams(prune_rate=(2, 4)),
            mode=mode,
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
      initial_vars['params']['w'] = weights
      outputs, state = linear.apply(initial_vars, inputs, mutable=True)
    self.assertEqual(outputs.shape, (2, 4))
    if mode != SparsityMode.INFERENCE:
      self.assertArraysEqual(
          state['non_trainable']['w' + base_layer.SPARSITY_NAME_POSTFIX],
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

  def test_sparsity_hparams_asserts(self):
    with self.assertRaises(AssertionError):
      SparsityHParams(
          sparsity_type=SparsityType.STRUCTURED_NM,
          weight_params=WeightSparsityParams(prune_rate=0.2),
          mode=SparsityMode.INFERENCE,
      )

    with self.assertRaises(AssertionError):
      SparsityHParams(
          sparsity_type=SparsityType.UNSTRUCTURED,
          weight_params=WeightSparsityParams(prune_rate=(2, 4)),
          mode=SparsityMode.INFERENCE,
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
    initial_vars_f['params']['w'] = weights
    initial_vars_s['params']['w'] = weights

    outputs_f = linear_f.apply(initial_vars_f, inputs)
    outputs_s = linear_s.apply(initial_vars_s, inputs)
    self.assertAllClose(outputs_f, outputs_s)

  def test_linear_inference_before_sparsification(self):
    p_f = pax_fiddle.Config(linears.Linear, name='_linear_f')
    p_s = pax_fiddle.Config(
        slinears.Linear,
        name='_linear',
        sparsity=SparsityHParams(
            sparsity_type=SparsityType.STRUCTURED_NM,
            weight_params=WeightSparsityParams(prune_rate=(2, 4)),
            mode=SparsityMode.INFERENCE,
        ),
    )
    for p in [p_f, p_s]:
      p.input_dims = 4
      p.output_dims = 4
    inputs = np.random.normal(1.5, 2.0, [5, 4]).astype(np.float32)
    self._run_and_compare(p_f, p_s, inputs)


if __name__ == '__main__':
  absltest.main()
