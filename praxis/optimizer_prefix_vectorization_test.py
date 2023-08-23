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

"""Unit tests for prefix vectorization in optimizers."""

from absl import logging
from absl.testing import absltest
import jax
from jax import numpy as jnp
import optax
from praxis import base_layer
from praxis import optimizer_prefix_vectorization as opt_vec
from praxis import test_utils


class OptimizerPrefixVectorizationTest(test_utils.TestCase):

  def test_vectorized_prefix_with_tree_map_params(self):
    def _opt_init(params):
      # Reduction over each variable. Behavior will depend on vectorization.
      logging.info(f'Init called with params {params}')
      return jax.tree_map(jnp.sum, params)

    def _opt_update(updates, state, params):
      del params
      return jax.tree_map(lambda u, s: u + s, updates, state), state

    grad_tx = optax.GradientTransformationExtraArgs(
        init=_opt_init, update=_opt_update
    )

    grads = base_layer.NestedMap(
        a=jnp.array([1, 2], dtype=jnp.float32),
        b=jnp.array([1, 2], dtype=jnp.float32),
        c=jnp.array([[1, 2], [3, 4]], dtype=jnp.float32),
    )
    variables = grads.copy()
    a_var_param = base_layer.WeightHParams(())
    a_var_param.repeat_prefix = [2]
    a_var_param.repeat_prefix_split_dims_mapping = [-1]
    b_var_param = base_layer.WeightHParams((2,))
    c_var_param = base_layer.WeightHParams(())
    c_var_param.repeat_prefix = [2, 2]
    c_var_param.repeat_prefix_split_dims_mapping = [('data', 'mdl'), None]
    var_hparams = base_layer.NestedMap(
        a=a_var_param, b=b_var_param, c=c_var_param
    )

    grad_tx = opt_vec.get_transformations_with_vectorized_repeat_prefix(
        grad_tx, var_hparams
    )

    state = grad_tx.init(variables)
    logging.info(state)
    opt_states_pspec = opt_vec.partition_params(grad_tx, var_hparams, state)
    logging.info('opt_states_pspec=%s', opt_states_pspec)
    # Computed update is 0 + state, and state is sum of each variable.
    update, _ = grad_tx.update(
        jax.tree_map(jnp.zeros_like, variables), state, variables
    )
    # Variables a and c are scalars excluding the prefix, so the update must be
    # equal to the initial variable values.
    self.assertAllClose(update.a, variables.a)
    self.assertAllClose(update.c, variables.c)
    # b is not vectorized, so the update equals the sum reduction of the initial
    # variable value.
    self.assertAllClose(
        update.b, jnp.zeros_like(variables.b) + jnp.sum(variables.b)
    )


if __name__ == '__main__':
  absltest.main()
