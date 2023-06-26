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

"""Tests for sparsity."""

import dataclasses
from absl.testing import absltest
from absl.testing import parameterized
from jax import numpy as jnp
import numpy as np
from praxis.layers.sparsity import sparsity
from praxis.layers.sparsity import sparsity_hparams


dataclass = dataclasses.dataclass


class PruningParamsTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(sparse_type='structured_nm', prune_rate=0.1),
      dict(sparse_type='unstructured', prune_rate=(1, 4)),
  )
  def test_invalid_params(self, sparse_type, prune_rate):
    with self.assertRaisesRegex(
        AssertionError, 'Prune rate must be either None'
    ):
      weight_params = sparsity_hparams.WeightSparsityParams(
          prune_rate=prune_rate
      )
      sparsity_hparams.SparsityHParams(
          sparsity_type=sparse_type, weight_params=weight_params
      )

  @parameterized.parameters(
      dict(prune_rate=(1, 4), mask_decay_weight=-0.1),
      dict(prune_rate=0.2, mask_decay_weight=-0.1),
  )
  def test_invalid_mask_decay_weight(
      self, prune_rate, mask_decay_weight
  ):
    with self.assertRaisesRegex(
        AssertionError, '.* `mask_decay_weight` must be positive.'
    ):
      sparsity_hparams.WeightSparsityParams(
          prune_rate=prune_rate,
          mask_decay_weight=mask_decay_weight
      )

  @parameterized.parameters(
      dict(
          prune_rate=(1, 4),
          sparse_ste=True,
          mask_decay_weight=0.1,
      ),
      dict(
          prune_rate=0.2,
          sparse_ste=True,
          mask_decay_weight=0.1,
      ),
  )
  def test_invalid_sparse_ste_with_non_zero_mask_decay_weight(
      self, prune_rate, sparse_ste, mask_decay_weight
  ):
    with self.assertRaisesRegex(
        ValueError, 'SR-STE only works with non-decaying mask.'
    ):
      sparsity_hparams.WeightSparsityParams(
          prune_rate=prune_rate,
          sparse_ste=sparse_ste,
          mask_decay_weight=mask_decay_weight,
      )

  @parameterized.parameters(
      dict(
          prune_rate=(1, 4),
          sparse_ste=True,
          structure_decay=True,
      ),
      dict(
          prune_rate=0.2,
          sparse_ste=True,
          structure_decay=True,
      ),
  )
  def test_invalid_sparse_ste_with_structure_decay(
      self, prune_rate, sparse_ste, structure_decay
  ):
    with self.assertRaisesRegex(
        ValueError, 'SR-STE only works with non-decaying sparse structure.'
    ):
      sparsity_hparams.WeightSparsityParams(
          prune_rate=prune_rate,
          sparse_ste=sparse_ste,
          structure_decay=structure_decay,
      )

  @parameterized.parameters(
      dict(sparse_type='unstructured', prune_rate=0.2, sparse_ste=True)
  )
  def test_invalid_sparse_ste_with_unstructured_sparsity(
      self, sparse_type, prune_rate, sparse_ste
  ):
    with self.assertRaisesRegex(
        ValueError, 'SR-STE only works with structured sparsity.'
    ):
      weight_params = sparsity_hparams.WeightSparsityParams(
          prune_rate=prune_rate, sparse_ste=sparse_ste
      )
      sparsity_hparams.SparsityHParams(
          sparsity_type=sparse_type, weight_params=weight_params
      )

  def test_invalid_prune_rate(self):
    error_msg = 'N must be lower than M'
    inputs = jnp.arange(12)
    with self.assertRaisesRegex(AssertionError, error_msg):
      sparsity.get_sparsity_mask(inputs, n_sparsity=4, m_sparsity=1)


class PruningFunctionalityTest(parameterized.TestCase):

  def test_prune_inputs_n_m(self):
    inputs = jnp.array(np.random.rand(10, 2, 4))
    prune_rate = (1, 4)

    out = sparsity.prune_inputs_n_m(inputs, n=prune_rate[0], m=prune_rate[1])
    self.assertEqual(out.shape[0], inputs.shape[0])
    self.assertEqual(out.shape[1], inputs.shape[1])
    self.assertEqual(out.shape[2], inputs.shape[2])

    # Only 20 non-zero elements must exist after pruning.
    num_non_zero_elems = 0.25 * inputs.size
    self.assertEqual(out[out != 0].shape[0], num_non_zero_elems)
    self.assertEqual(
        list(np.argmax(inputs, axis=2).flatten()),
        list(np.argmax(out != 0, axis=2).flatten()),
    )

  def test_n_m_pruning_mask(self):
    inputs = jnp.array(np.random.rand(10, 2, 4))
    prune_rate = (1, 4)
    mask = sparsity.get_pruning_n_m_mask(
        inputs, n=prune_rate[0], m=prune_rate[1]
    )
    self.assertEqual(
        list(np.argmax(inputs, axis=2).flatten()),
        list(np.argmax(mask == 1, axis=2).flatten()),
    )


class PruningScoreTest(parameterized.TestCase):

  def test_score_activation_weighted(self):
    weight = jnp.array([[1.0, 2.0, 3.0, 4.0], [-4.0, -3.0, -2.0, -1.0]])
    activation = jnp.array([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
    expected_score = jnp.array([[4.0, 8.0, 12.0, 16.0], [0.0, 0.0, 0.0, 0.0]])
    score = sparsity.compute_score(
        weight,
        inputs=activation,
        score_func=sparsity_hparams.SparsityScore.ACTIVATION_WEIGHTED,
    )
    self.assertTrue((score == expected_score).all())


if __name__ == '__main__':
  absltest.main()
