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

"""Tests for sparse_base_layer."""

import copy
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import pax_fiddle
from praxis import test_utils
from praxis.layers import linears
from praxis.layers.sparsity import sparse_base_layer
from praxis.layers.sparsity import sparsity_hparams

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
SparsityMode = sparsity_hparams.SparsityMode
SparsityType = sparsity_hparams.SparsityType


class SparseLinearTestLayer(
    sparse_base_layer.SparsityBaseLayer, linears.Linear
):
  input_dims: int = 3
  output_dims: int = 4

  def setup(self):
    weight_hp = base_layer.WeightHParams(
        shape=[self.input_dims, self.output_dims],
        init=self.params_init,
        dtype=self.dtype,
    )
    name = 'w'
    self.create_variable(name, weight_hp)
    self.create_child('einsum', self.einsum_tpl.clone())
    self.create_aux_variables(name, weight_hp)

  def __call__(self, inputs):
    w = self.sparsifiy(
        self.theta.w, inputs=inputs, name='w'
    )  # sparsify weight.
    out = self.einsum('...y,yz->...z', inputs, w)
    return out


class SparseBaseLayerCorrectnessTest(test_utils.TestCase):
  """Checks the expected behaviors of sparse_base_layer."""

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  @parameterized.named_parameters(
      ('training_mode', 0, 1, 0, SparsityMode.TRAINING),
      ('inference_mode', 0, 1, 0, SparsityMode.INFERENCE),
      ('materialize_mode', 0, 1, 0, SparsityMode.MATERIALIZE),
      ('oneshot_mode', 1, 1, 10, SparsityMode.ONESHOT),
      ('fewshot_mode', 10, 2, 20, SparsityMode.FEWSHOT),
  )
  def test_create_aux_variables(
      self, num_shots, mask_update_interval, target_step, mode
  ):
    sparsity_p = sparsity_hparams.SparsityHParams(
        sparsity_type=SparsityType.STRUCTURED_NM,
        mode=mode,
        weight_params=WeightSparsityParams(prune_rate=(2, 4)),
        num_shots=num_shots,
        mask_update_interval=mask_update_interval,
        target_step=target_step,
    )

    p = pax_fiddle.Config(SparseLinearTestLayer, sparsity=sparsity_p)
    test_layer = instantiate(p)
    prng_key = jax.random.PRNGKey(seed=123)
    inputs = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=p.dtype)
    initial_var = test_layer.init(prng_key, inputs)
    self.assertEqual(test_layer.sparsity.num_shots, num_shots)
    self.assertEqual(
        test_layer.sparsity.mask_update_interval, mask_update_interval
    )
    self.assertEqual(test_layer.sparsity.target_step, target_step)
    if mode == SparsityMode.TRAINING or mode == SparsityMode.MATERIALIZE:
      self.assertEqual(initial_var[NON_TRAINABLE]['num_shots'], -1)
    elif mode != SparsityMode.INFERENCE:
      self.assertEqual(initial_var[NON_TRAINABLE]['num_shots'], num_shots)

  def test_masked_weight_gradient(self):
    sparsity_p = sparsity_hparams.SparsityHParams(
        sparsity_type=SparsityType.STRUCTURED_NM,
        mode=SparsityMode.MATERIALIZE,
        weight_params=WeightSparsityParams(prune_rate=(2, 4)),
    )

    p = pax_fiddle.Config(SparseLinearTestLayer, sparsity=sparsity_p)

    test_layer = instantiate(p)
    prng_key = jax.random.PRNGKey(seed=123)
    inputs = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=p.dtype)
    weights = jnp.array(
        [
            [1, 2, 3, 4],
            [-3, 1, 4, -2],
            [2, 4, -1, 3],
        ],
        dtype=p.dtype,
    )
    step_size = 0.01

    def update(test_layer, params, inputs, targets):
      def value_and_loss(params, inputs, targets):
        outputs = test_layer.apply(params, inputs)
        return -jnp.mean(jnp.abs(outputs - targets))

      grads = jax.grad(value_and_loss, allow_int=True)(params, inputs, targets)

      out_params = copy.deepcopy(params)
      w_grad = grads[PARAMS]['w']
      out_params[PARAMS]['w'] = params[PARAMS]['w'] - step_size * w_grad

      return grads, out_params

    with base_layer.JaxContext.new_context():
      initial_vars = test_layer.init(prng_key, inputs)
      initial_vars[PARAMS]['w'] = weights

      self.assertArraysEqual(
          initial_vars[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
          jnp.array([
              [True, True, True, True],
              [True, True, True, True],
              [True, True, True, True],
          ]),
      )

      fixed_mask = jnp.array([
          [False, False, True, True],
          [True, True, False, False],
          [True, False, True, False],
      ])
      initial_vars[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX] = fixed_mask

      targets = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=p.dtype)
      grads, _ = update(test_layer, initial_vars, inputs, targets)
      self.assertArraysEqual(grads[PARAMS]['w'] != 0, fixed_mask)

  def test_training_mode(self):
    sparsity_p = sparsity_hparams.SparsityHParams(
        sparsity_type=SparsityType.STRUCTURED_NM,
        mode=SparsityMode.TRAINING,
        weight_params=WeightSparsityParams(prune_rate=(2, 4)),
        target_step=2,
    )

    p = pax_fiddle.Config(SparseLinearTestLayer, sparsity=sparsity_p)

    test_layer = instantiate(p)
    prng_key = jax.random.PRNGKey(seed=123)
    inputs = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=p.dtype)
    weights = jnp.array(
        [
            [1, 2, 3, 4],
            [-3, 1, 4, -2],
            [2, 4, -1, 3],
        ],
        dtype=p.dtype,
    )

    def update(test_layer, params, inputs, updated_weights):
      outputs, updated_params = test_layer.apply(params, inputs, mutable=True)
      updated_params[PARAMS]['w'] = updated_weights

      return outputs, updated_params

    with base_layer.JaxContext.new_context():
      params = test_layer.init(prng_key, inputs)
      params[PARAMS]['w'] = weights

      self.assertArraysEqual(
          params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
          jnp.array([
              [True, True, True, True],
              [True, True, True, True],
              [True, True, True, True],
          ]),
      )

      # step 0, no update, no apply
      outputs, updated_params = update(
          test_layer,
          params,
          inputs,
          jnp.array(
              [
                  [10, 20, 3, 4],
                  [-3, 10, 4, -2],
                  [2, 4, -1, 3],
              ],
              dtype=p.dtype,
          ),
      )
      self.assertArraysEqual(
          params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
          updated_params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
      )
      self.assertArraysEqual(
          outputs, jnp.einsum('...y,yz->...z', inputs, params[PARAMS]['w'])
      )

      # step 1, no update, no apply
      params = copy.deepcopy(updated_params)
      outputs, updated_params = update(
          test_layer,
          params,
          inputs,
          jnp.array(
              [
                  [1, 20, 30, 4],
                  [-3, 1, 4, -20],
                  [2, 4, -1, 3],
              ],
              dtype=p.dtype,
          ),
      )
      self.assertArraysEqual(
          params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
          updated_params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
      )
      self.assertArraysEqual(
          outputs, jnp.einsum('...y,yz->...z', inputs, params[PARAMS]['w'])
      )

      # step 2, update mask, apply mask
      params = copy.deepcopy(updated_params)
      outputs, updated_params = update(
          test_layer,
          params,
          inputs,
          jnp.array(
              [
                  [1, 20, 3, 4],
                  [-3, 1, 4, -20],
                  [2, 4, -1, 30],
              ],
              dtype=p.dtype,
          ),
      )
      self.assertFalse(
          jnp.array_equal(
              params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
              updated_params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
          )
      )
      self.assertEqual(updated_params[NON_TRAINABLE]['mask_update_count'], 1)
      self.assertArraysEqual(
          outputs,
          jnp.einsum(
              '...y,yz->...z',
              inputs,
              jnp.multiply(
                  params[PARAMS]['w'],
                  updated_params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
              ),
          ),
      )

      # step 3, update mask, apply mask
      params = copy.deepcopy(updated_params)
      outputs, updated_params = update(
          test_layer,
          params,
          inputs,
          jnp.array(
              [
                  [1, 2, 30, 40],
                  [-3, 10, 4, -2],
                  [2, 4, -1, 3],
              ],
              dtype=p.dtype,
          ),
      )
      self.assertFalse(
          jnp.array_equal(
              params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
              updated_params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
          )
      )
      self.assertEqual(updated_params[NON_TRAINABLE]['mask_update_count'], 2)
      self.assertArraysEqual(
          outputs,
          jnp.einsum(
              '...y,yz->...z',
              inputs,
              jnp.multiply(
                  params[PARAMS]['w'],
                  updated_params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
              ),
          ),
      )

  @parameterized.named_parameters(
      ('materialize_mode_test1', 1, 1, 3),
      ('materialize_mode_test2', 5, 1, 5),
  )
  def test_materialize_mode(self, num_shots, mask_update_interval, target_step):
    sparsity_p = sparsity_hparams.SparsityHParams(
        sparsity_type=SparsityType.STRUCTURED_NM,
        mode=SparsityMode.MATERIALIZE,
        weight_params=WeightSparsityParams(prune_rate=(2, 4)),
        num_shots=num_shots,
        mask_update_interval=mask_update_interval,
        target_step=target_step,
    )

    p = pax_fiddle.Config(SparseLinearTestLayer, sparsity=sparsity_p)

    test_layer = instantiate(p)
    prng_key = jax.random.PRNGKey(seed=123)
    inputs = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=p.dtype)
    weights = jnp.array(
        [
            [1, 2, 3, 4],
            [-3, 1, 4, -2],
            [2, 4, -1, 3],
        ],
        dtype=p.dtype,
    )

    with base_layer.JaxContext.new_context():
      params = test_layer.init(prng_key, inputs)
      params[PARAMS]['w'] = weights

      self.assertArraysEqual(
          params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
          jnp.array([
              [True, True, True, True],
              [True, True, True, True],
              [True, True, True, True],
          ]),
      )

      fixed_mask = jnp.array([
          [False, False, True, True],
          [True, True, False, False],
          [True, False, True, False],
      ])

      current_step = 0
      params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX] = fixed_mask
      self.assertArraysEqual(
          params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX], fixed_mask
      )
      self.assertArraysEqual(params[PARAMS]['w'], weights)
      self.assertArraysEqual(params[NON_TRAINABLE]['num_shots'], -1)
      self.assertArraysEqual(params[NON_TRAINABLE]['mask_update_count'], 0)
      self.assertArraysEqual(params[NON_TRAINABLE]['step'], current_step)

      while current_step < 10:
        outputs, updated_params = test_layer.apply(params, inputs, mutable=True)
        current_step += 1
        sparsified_weights = jnp.multiply(weights, fixed_mask)
        self.assertArraysEqual(
            outputs, jnp.einsum('...y,yz->...z', inputs, sparsified_weights)
        )  # apply mask
        self.assertArraysEqual(
            params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX], fixed_mask
        )
        self.assertArraysEqual(updated_params[PARAMS]['w'], weights)
        self.assertArraysEqual(updated_params[NON_TRAINABLE]['num_shots'], -1)
        self.assertArraysEqual(
            updated_params[NON_TRAINABLE]['mask_update_count'], 0
        )
        self.assertArraysEqual(
            updated_params[NON_TRAINABLE]['step'], current_step
        )
        params = copy.deepcopy(updated_params)

  @parameterized.named_parameters(
      ('inference_mode_test1', 1, 1, 3), ('inference_mode_test2', 5, 1, 5)
  )
  def test_inference_mode(self, num_shots, mask_update_interval, target_step):
    sparsity_p = sparsity_hparams.SparsityHParams(
        sparsity_type=SparsityType.STRUCTURED_NM,
        mode=SparsityMode.INFERENCE,
        weight_params=WeightSparsityParams(prune_rate=(2, 4)),
        num_shots=num_shots,
        mask_update_interval=mask_update_interval,
        target_step=target_step,
    )

    p = pax_fiddle.Config(SparseLinearTestLayer, sparsity=sparsity_p)

    test_layer = instantiate(p)
    prng_key = jax.random.PRNGKey(seed=123)
    inputs = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=p.dtype)
    weights = jnp.array(
        [
            [1, 2, 3, 4],
            [-3, 1, 4, -2],
            [2, 4, -1, 3],
        ],
        dtype=p.dtype,
    )

    with base_layer.JaxContext.new_context():
      params = test_layer.init(prng_key, inputs)
      params[PARAMS]['w'] = weights

      self.assertArraysEqual(params[PARAMS]['w'], weights)

      current_step = 0
      while current_step < 10:
        outputs, updated_params = test_layer.apply(params, inputs, mutable=True)
        current_step += 1
        self.assertArraysEqual(
            outputs, jnp.einsum('...y,yz->...z', inputs, weights)
        )  # no apply
        params = copy.deepcopy(updated_params)

  def test_one_shot_mode(self):
    sparsity_p = sparsity_hparams.SparsityHParams(
        sparsity_type=SparsityType.STRUCTURED_NM,
        mode=SparsityMode.ONESHOT,
        weight_params=WeightSparsityParams(prune_rate=(2, 4)),
        num_shots=1,
        target_step=2,
    )

    p = pax_fiddle.Config(SparseLinearTestLayer, sparsity=sparsity_p)

    test_layer = instantiate(p)
    prng_key = jax.random.PRNGKey(seed=123)
    inputs = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=p.dtype)
    weights = jnp.array(
        [
            [1, 2, 3, 4],
            [-3, 1, 4, -2],
            [2, 4, -1, 3],
        ],
        dtype=p.dtype,
    )

    def update(test_layer, params, inputs, updated_weights):
      outputs, updated_params = test_layer.apply(params, inputs, mutable=True)
      updated_params[PARAMS]['w'] = updated_weights

      return outputs, updated_params

    with base_layer.JaxContext.new_context():
      params = test_layer.init(prng_key, inputs)
      params[PARAMS]['w'] = weights

      self.assertArraysEqual(
          params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
          jnp.array([
              [True, True, True, True],
              [True, True, True, True],
              [True, True, True, True],
          ]),
      )

      # step 0, no update, no apply
      outputs, updated_params = update(
          test_layer,
          params,
          inputs,
          jnp.array(
              [
                  [10, 20, 3, 4],
                  [-3, 10, 4, -2],
                  [2, 4, -1, 3],
              ],
              dtype=p.dtype,
          ),
      )
      self.assertArraysEqual(
          params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
          updated_params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
      )
      self.assertArraysEqual(
          outputs, jnp.einsum('...y,yz->...z', inputs, params[PARAMS]['w'])
      )

      # step 1, no update, no apply
      params = copy.deepcopy(updated_params)
      outputs, updated_params = update(
          test_layer,
          params,
          inputs,
          jnp.array(
              [
                  [1, 20, 30, 4],
                  [-3, 1, 4, -20],
                  [2, 4, -1, 3],
              ],
              dtype=p.dtype,
          ),
      )
      self.assertArraysEqual(
          params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
          updated_params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
      )
      self.assertArraysEqual(
          outputs, jnp.einsum('...y,yz->...z', inputs, params[PARAMS]['w'])
      )

      # step 2, update mask, apply mask
      params = copy.deepcopy(updated_params)
      outputs, updated_params = update(
          test_layer,
          params,
          inputs,
          jnp.array(
              [
                  [1, 20, 3, 4],
                  [-3, 1, 4, -20],
                  [2, 4, -1, 30],
              ],
              dtype=p.dtype,
          ),
      )
      self.assertFalse(
          jnp.array_equal(
              params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
              updated_params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
          )
      )
      self.assertEqual(updated_params[NON_TRAINABLE]['mask_update_count'], 1)
      self.assertArraysEqual(
          outputs,
          jnp.einsum(
              '...y,yz->...z',
              inputs,
              jnp.multiply(
                  params[PARAMS]['w'],
                  updated_params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
              ),
          ),
      )

      # step 3, no update, apply mask
      params = copy.deepcopy(updated_params)
      outputs, updated_params = update(
          test_layer,
          params,
          inputs,
          jnp.array(
              [
                  [1, 2, 30, 40],
                  [-3, 10, 4, -2],
                  [2, 4, -10, 3],
              ],
              dtype=p.dtype,
          ),
      )
      self.assertArraysEqual(
          params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
          updated_params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
      )

      self.assertEqual(updated_params[NON_TRAINABLE]['mask_update_count'], 1)
      self.assertArraysEqual(
          outputs,
          jnp.einsum(
              '...y,yz->...z',
              inputs,
              jnp.multiply(
                  params[PARAMS]['w'],
                  updated_params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
              ),
          ),
      )

      # step 4, no update, apply mask
      params = copy.deepcopy(updated_params)
      outputs, updated_params = update(
          test_layer,
          params,
          inputs,
          jnp.array(
              [
                  [10, 2, 30, 4],
                  [-3, 1, 4, -20],
                  [2, 4, -1, 3],
              ],
              dtype=p.dtype,
          ),
      )
      self.assertArraysEqual(
          params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
          updated_params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
      )

      self.assertEqual(updated_params[NON_TRAINABLE]['mask_update_count'], 1)
      self.assertArraysEqual(
          outputs,
          jnp.einsum(
              '...y,yz->...z',
              inputs,
              jnp.multiply(
                  params[PARAMS]['w'],
                  updated_params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
              ),
          ),
      )

  def test_few_shot_mode(self):
    sparsity_p = sparsity_hparams.SparsityHParams(
        sparsity_type=SparsityType.STRUCTURED_NM,
        mode=SparsityMode.FEWSHOT,
        weight_params=WeightSparsityParams(prune_rate=(2, 4)),
        num_shots=2,
        mask_update_interval=2,
        target_step=2,
    )

    p = pax_fiddle.Config(SparseLinearTestLayer, sparsity=sparsity_p)

    test_layer = instantiate(p)
    prng_key = jax.random.PRNGKey(seed=123)
    inputs = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=p.dtype)
    weights = jnp.array(
        [
            [1, 2, 3, 4],
            [-3, 1, 4, -2],
            [2, 4, -1, 3],
        ],
        dtype=p.dtype,
    )

    def update(test_layer, params, inputs, updated_weights):
      outputs, updated_params = test_layer.apply(params, inputs, mutable=True)
      updated_params[PARAMS]['w'] = updated_weights

      return outputs, updated_params

    with base_layer.JaxContext.new_context():
      params = test_layer.init(prng_key, inputs)
      params[PARAMS]['w'] = weights

      self.assertArraysEqual(
          params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
          jnp.array([
              [True, True, True, True],
              [True, True, True, True],
              [True, True, True, True],
          ]),
      )

      # step 0, no update, no apply
      outputs, updated_params = update(
          test_layer,
          params,
          inputs,
          jnp.array(
              [
                  [10, 20, 3, 4],
                  [-3, 10, 4, -2],
                  [2, 4, -1, 3],
              ],
              dtype=p.dtype,
          ),
      )
      self.assertArraysEqual(
          params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
          updated_params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
      )
      self.assertArraysEqual(
          outputs, jnp.einsum('...y,yz->...z', inputs, params[PARAMS]['w'])
      )

      # step 1, no update, no apply
      params = copy.deepcopy(updated_params)
      outputs, updated_params = update(
          test_layer,
          params,
          inputs,
          jnp.array(
              [
                  [1, 20, 30, 4],
                  [-3, 1, 4, -20],
                  [2, 4, -1, 3],
              ],
              dtype=p.dtype,
          ),
      )
      self.assertArraysEqual(
          params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
          updated_params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
      )
      self.assertArraysEqual(
          outputs, jnp.einsum('...y,yz->...z', inputs, params[PARAMS]['w'])
      )

      # step 2, update mask, apply mask
      params = copy.deepcopy(updated_params)
      outputs, updated_params = update(
          test_layer,
          params,
          inputs,
          jnp.array(
              [
                  [1, 20, 3, 4],
                  [-3, 1, 4, -20],
                  [2, 4, -1, 30],
              ],
              dtype=p.dtype,
          ),
      )
      self.assertFalse(
          jnp.array_equal(
              params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
              updated_params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
          )
      )
      self.assertEqual(updated_params[NON_TRAINABLE]['mask_update_count'], 1)
      self.assertArraysEqual(
          outputs,
          jnp.einsum(
              '...y,yz->...z',
              inputs,
              jnp.multiply(
                  params[PARAMS]['w'],
                  updated_params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
              ),
          ),
      )

      # step 3, no update, apply mask
      params = copy.deepcopy(updated_params)
      outputs, updated_params = update(
          test_layer,
          params,
          inputs,
          jnp.array(
              [
                  [1, 2, 30, 40],
                  [-3, 10, 4, -2],
                  [2, 4, -10, 3],
              ],
              dtype=p.dtype,
          ),
      )
      self.assertArraysEqual(
          params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
          updated_params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
      )

      self.assertEqual(updated_params[NON_TRAINABLE]['mask_update_count'], 1)
      self.assertArraysEqual(
          outputs,
          jnp.einsum(
              '...y,yz->...z',
              inputs,
              jnp.multiply(
                  params[PARAMS]['w'],
                  updated_params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
              ),
          ),
      )

      # step 4, update mask, apply mask
      params = copy.deepcopy(updated_params)
      outputs, updated_params = update(
          test_layer,
          params,
          inputs,
          jnp.array(
              [
                  [10, 2, 30, 4],
                  [-3, 1, 4, -20],
                  [2, 4, -1, 3],
              ],
              dtype=p.dtype,
          ),
      )
      self.assertFalse(
          jnp.array_equal(
              params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
              updated_params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
          )
      )

      self.assertEqual(updated_params[NON_TRAINABLE]['mask_update_count'], 2)
      self.assertArraysEqual(
          outputs,
          jnp.einsum(
              '...y,yz->...z',
              inputs,
              jnp.multiply(
                  params[PARAMS]['w'],
                  updated_params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
              ),
          ),
      )

      # step 5, no update, apply mask
      params = copy.deepcopy(updated_params)
      outputs, updated_params = update(
          test_layer,
          params,
          inputs,
          jnp.array(
              [
                  [10, 2, 3, 4],
                  [-3, 10, 4, -2],
                  [2, 4, -10, 3],
              ],
              dtype=p.dtype,
          ),
      )
      self.assertArraysEqual(
          params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
          updated_params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
      )

      self.assertEqual(updated_params[NON_TRAINABLE]['mask_update_count'], 2)
      self.assertArraysEqual(
          outputs,
          jnp.einsum(
              '...y,yz->...z',
              inputs,
              jnp.multiply(
                  params[PARAMS]['w'],
                  updated_params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
              ),
          ),
      )

      # step 6, no update, apply mask
      params = copy.deepcopy(updated_params)
      outputs, updated_params = update(
          test_layer,
          params,
          inputs,
          jnp.array(
              [
                  [10, 2, 30, 4],
                  [-30, 10, 4, -2],
                  [20, 4, -10, 3],
              ],
              dtype=p.dtype,
          ),
      )
      self.assertArraysEqual(
          params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
          updated_params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
      )

      self.assertEqual(updated_params[NON_TRAINABLE]['mask_update_count'], 2)
      self.assertArraysEqual(
          outputs,
          jnp.einsum(
              '...y,yz->...z',
              inputs,
              jnp.multiply(
                  params[PARAMS]['w'],
                  updated_params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
              ),
          ),
      )

      # step 7, no update, apply mask
      params = copy.deepcopy(updated_params)
      outputs, updated_params = update(
          test_layer,
          params,
          inputs,
          jnp.array(
              [
                  [1, 2, 30, 4],
                  [-30, 10, 40, -2],
                  [20, 4, -10, 3],
              ],
              dtype=p.dtype,
          ),
      )
      self.assertArraysEqual(
          params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
          updated_params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
      )

      self.assertEqual(updated_params[NON_TRAINABLE]['mask_update_count'], 2)
      self.assertArraysEqual(
          outputs,
          jnp.einsum(
              '...y,yz->...z',
              inputs,
              jnp.multiply(
                  params[PARAMS]['w'],
                  updated_params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
              ),
          ),
      )


if __name__ == '__main__':
  absltest.main()
