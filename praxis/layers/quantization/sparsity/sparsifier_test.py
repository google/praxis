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
from praxis.layers.quantization.sparsity import sparsifier
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
FewShotMode = sparsity_modes.FewShotMode
OneShotMode = sparsity_modes.OneShotMode
MaterializeMode = sparsity_modes.MaterializeMode
TrainingMode = sparsity_modes.TrainingMode


class SparseLinearTestLayer(sparsifier.SparsityBaseLayer, linears.Linear):

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
      ('training_mode', 'training_mode', pax_fiddle.Config(TrainingMode)),
      ('inference_mode', 'inference_mode', pax_fiddle.Config(InferenceMode)),
      (
          'materialize_mode',
          'materialize_mode',
          pax_fiddle.Config(MaterializeMode),
      ),
      (
          'oneshot_mode',
          'oneshot_mode',
          pax_fiddle.Config(OneShotMode, target_step=10),
      ),
      (
          'fewshot_mode',
          'fewshot_mode',
          pax_fiddle.Config(
              FewShotMode, num_shots=10, mask_update_interval=2, target_step=20
          ),
      ),
  )
  def test_create_aux_variables(self, mode_name, mode):
    sparsity_p = pax_fiddle.Config(
        SparsityHParams,
        sparsity_type=SparsityType.STRUCTURED_NM,
        mode=mode,
        weight_params=WeightSparsityParams(prune_rate=(2, 4)),
    )

    p = pax_fiddle.Config(
        SparseLinearTestLayer, sparsity=sparsity_p, input_dims=3, output_dims=4
    )
    test_layer = instantiate(p)
    prng_key = jax.random.PRNGKey(seed=123)
    inputs = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=p.dtype)
    initial_var = test_layer.init(prng_key, inputs)
    if mode_name == 'training_mode' or mode_name == 'materialize_mode':
      self.assertEqual(initial_var[NON_TRAINABLE]['num_shots'], -1)
    elif mode_name == 'oneshot_mode':
      self.assertEqual(initial_var[NON_TRAINABLE]['num_shots'], 1)
    elif mode_name == 'fewshot_mode':
      self.assertEqual(initial_var[NON_TRAINABLE]['num_shots'], 10)

  def test_masked_weight_gradient(self):
    sparsity_p = pax_fiddle.Config(
        SparsityHParams,
        sparsity_type=SparsityType.STRUCTURED_NM,
        mode=pax_fiddle.Config(MaterializeMode),
        weight_params=WeightSparsityParams(prune_rate=(2, 4)),
    )

    p = pax_fiddle.Config(
        SparseLinearTestLayer, sparsity=sparsity_p, input_dims=3, output_dims=4
    )

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
    sparsity_p = pax_fiddle.Config(
        SparsityHParams,
        sparsity_type=SparsityType.STRUCTURED_NM,
        mode=pax_fiddle.Config(TrainingMode, target_step=2),
        weight_params=WeightSparsityParams(prune_rate=(2, 4)),
    )

    p = pax_fiddle.Config(
        SparseLinearTestLayer, sparsity=sparsity_p, input_dims=3, output_dims=4
    )

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

  def test_materialize_mode(self):
    sparsity_p = pax_fiddle.Config(
        SparsityHParams,
        sparsity_type=SparsityType.STRUCTURED_NM,
        mode=pax_fiddle.Config(MaterializeMode),
        weight_params=WeightSparsityParams(prune_rate=(2, 4)),
    )

    p = pax_fiddle.Config(
        SparseLinearTestLayer, sparsity=sparsity_p, input_dims=3, output_dims=4
    )

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

  def test_inference_mode(self):
    sparsity_p = pax_fiddle.Config(
        SparsityHParams,
        sparsity_type=SparsityType.STRUCTURED_NM,
        mode=pax_fiddle.Config(InferenceMode),
        weight_params=WeightSparsityParams(prune_rate=(2, 4)),
    )

    p = pax_fiddle.Config(
        SparseLinearTestLayer, sparsity=sparsity_p, input_dims=3, output_dims=4
    )

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
    sparsity_p = pax_fiddle.Config(
        SparsityHParams,
        sparsity_type=SparsityType.STRUCTURED_NM,
        mode=pax_fiddle.Config(OneShotMode, target_step=2),
        weight_params=WeightSparsityParams(prune_rate=(2, 4)),
    )

    p = pax_fiddle.Config(
        SparseLinearTestLayer, sparsity=sparsity_p, input_dims=3, output_dims=4
    )

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
    sparsity_p = pax_fiddle.Config(
        SparsityHParams,
        sparsity_type=SparsityType.STRUCTURED_NM,
        mode=pax_fiddle.Config(
            FewShotMode, num_shots=2, mask_update_interval=2, target_step=2
        ),
        weight_params=WeightSparsityParams(prune_rate=(2, 4)),
    )

    p = pax_fiddle.Config(
        SparseLinearTestLayer, sparsity=sparsity_p, input_dims=3, output_dims=4
    )

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

  @parameterized.named_parameters(
      ('row_wise', 'R'),
      ('column_wise', 'C'),
  )
  def test_sparsity_order(self, sparsity_order):
    sparsity_p = pax_fiddle.Config(
        SparsityHParams,
        sparsity_type=SparsityType.STRUCTURED_NM,
        mode=pax_fiddle.Config(TrainingMode, target_step=0),
        weight_params=WeightSparsityParams(prune_rate=(2, 4)),
        order=sparsity_order,
    )

    p = pax_fiddle.Config(
        SparseLinearTestLayer, sparsity=sparsity_p, input_dims=4, output_dims=4
    )

    test_layer = instantiate(p)
    prng_key = jax.random.PRNGKey(seed=123)
    inputs = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=p.dtype)
    weights = jnp.array(
        [
            [10, 20, 3, 4],
            [-3, 10, 4, 2],
            [2, 4, -1, 30],
            [40, -1, 2, -3],
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
              [True, True, True, True],
          ]),
      )

      # step 0, update mask, apply mask
      outputs, updated_params = update(
          test_layer,
          params,
          inputs,
          jnp.array(
              [
                  [10, 20, 3, 4],
                  [-3, 1, 4, 2],
                  [20, 4, -10, 3],
                  [4, -1, 2, -30],
              ],
              dtype=p.dtype,
          ),
      )
      if sparsity_order == 'C':
        self.assertArraysEqual(
            updated_params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
            jnp.array([
                [True, True, True, True],
                [False, True, True, False],
                [False, False, False, True],
                [True, False, False, False],
            ]),
        )
      else:
        self.assertArraysEqual(
            updated_params[NON_TRAINABLE]['w' + SPARSITY_NAME_POSTFIX],
            jnp.array([
                [True, True, False, False],
                [False, True, True, False],
                [False, True, False, True],
                [True, False, False, True],
            ]),
        )
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
