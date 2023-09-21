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

"""Unit tests for optimizers."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
import optax
from praxis import base_layer
from praxis import optimizers
from praxis import pax_fiddle
from praxis import py_utils
from praxis import schedules
from praxis import test_utils


def _run_transformation(
    num_steps=1,
    initial_value=2.0,
    l2_regularizer_weight=0.0,
    l1_regularizer_weight=0.0,
    decoupled_weight_decay=0.0,
    create_regularizer_mask=False,
    use_native_optax_gradient_transformations=False,
):
  """Applies a gradient transformation with Adam for some steps.

  Args:
    num_steps: Number of steps to apply transformation for.
    initial_value: Initial value of the model parameter.
    l2_regularizer_weight: Optional L2 regularization weight.
    l1_regularizer_weight: Optional L2 regularization weight.
    decoupled_weight_decay: Optional decoupled weight decay.
    create_regularizer_mask: Whether to create mask for regularization.
    use_native_optax_gradient_transformations: Whether to use OptaxOptimizer.

  Returns:
    The updated states and the final update.
  """
  param_name = 'var'
  mdl_vars = {param_name: jnp.array(initial_value, dtype=jnp.float32)}
  if use_native_optax_gradient_transformations:
    opt_tpl = pax_fiddle.Config(
        optimizers.OptaxOptimizer,
        lr_schedule=pax_fiddle.Config(schedules.Constant, value=1.0),
        learning_rate=1.0,
        l2_regularizer_weight=l2_regularizer_weight,
        l1_regularizer_weight=l1_regularizer_weight,
        decoupled_weight_decay=decoupled_weight_decay,
        grad_tx=optax.sgd(1.0),
    )
  else:
    opt_tpl = pax_fiddle.Config(
        optimizers.ShardedSgd,
        lr_schedule=pax_fiddle.Config(schedules.Constant, value=1.0),
        learning_rate=1.0,
        l2_regularizer_weight=l2_regularizer_weight,
        l1_regularizer_weight=l1_regularizer_weight,
        decoupled_weight_decay=decoupled_weight_decay,
    )
  sgd = optimizers.instantiate(opt_tpl)
  var_weight_hparams = None
  if create_regularizer_mask:
    var_weight_hparams = {
        param_name: base_layer.WeightHParams(
            shape=[1],
            collections=[
                base_layer.WeightHParamsCollection.SKIP_LP_REGULARIZATION
            ]
        )
    }
  tx = sgd.get_grad_transformation(var_weight_hparams)
  opt_state = tx.init(mdl_vars)
  for t in range(num_steps):
    fake_update = float(t + 1)
    updates, opt_state = tx.update(
        {param_name: jnp.array(fake_update)}, opt_state, mdl_vars
    )
    mdl_vars = optax.apply_updates(mdl_vars, updates)
  return mdl_vars[param_name]


class OptimizersTest(test_utils.TestCase, parameterized.TestCase):

  def test_adafactor(self):
    opt_tpl = optimizers.ShardedAdafactor.HParamsAdamB()
    opt_tpl.lr_schedule = pax_fiddle.Config(schedules.Constant, value=1.0)
    opt_tpl.learning_rate = 1.0
    opt = optimizers.instantiate(opt_tpl)
    tx = opt.get_grad_transformation()
    mdl_vars = {'var': jnp.array(0.0, dtype=jnp.float32)}
    opt_state = tx.init(mdl_vars)
    num_steps = 3
    for t in range(num_steps):
      fake_update = float(t + 1)
      updates, opt_state = tx.update(
          {'var': jnp.array(fake_update)}, opt_state, mdl_vars
      )
      mdl_vars = optax.apply_updates(mdl_vars, updates)
    expected_var_value = -0.561
    self.assertEqual(
        mdl_vars['var'], jnp.array(expected_var_value, dtype=jnp.float32)
    )

  def test_static_accumulator(self):
    base_opt_tpl = pax_fiddle.Config(optimizers.ShardedSgd)
    num_sub_batches = 3
    acc_opt_tpl = pax_fiddle.Config(
        optimizers.ShardedStaticAccumulator,
        optimizer_tpl=base_opt_tpl,
        num_sub_batches=num_sub_batches,
        lr_schedule=pax_fiddle.Config(schedules.Constant, value=1.0),
        learning_rate=1.0,
    )
    acc_opt = optimizers.instantiate(acc_opt_tpl)
    tx = acc_opt.get_grad_transformation()
    mdl_vars = {'var': jnp.array(0.0, dtype=jnp.float32)}
    opt_state = tx.init(mdl_vars)
    num_steps = 12
    for t in range(12):
      # Apply updates of value 1, 2, 3, 1, 2, 3, ... so that it averages out to
      # an accumulated update value of 2.0.
      fake_update = float((t % num_sub_batches) + 1)
      updates, opt_state = tx.update({'var': jnp.array(fake_update)}, opt_state)
      mdl_vars = optax.apply_updates(mdl_vars, updates)
    # Each accumulated update averages to 2.0.
    expected_var_value = -1.0 * num_steps // num_sub_batches * 2.0
    self.assertEqual(
        mdl_vars['var'], jnp.array(expected_var_value, dtype=jnp.float32))

  def test_ewc_regularization(self):
    opt_tpl = pax_fiddle.Config(
        optimizers.ShardedSgd,
        lr_schedule=pax_fiddle.Config(schedules.Constant, value=1.0),
        learning_rate=1.0,
        ewc_regularizer_weight=1.0,
        ewc_weight_per_var={'var': 1.0},
    )
    acc_opt = optimizers.instantiate(opt_tpl)
    tx = acc_opt.get_grad_transformation()
    mdl_vars = {'var': jnp.array(0, dtype=jnp.float32)}
    opt_state = tx.init(mdl_vars)
    num_steps = 3
    for t in range(num_steps):
      # t=0: v=0, g=1, p=0 -> v = v - g - 0.5  * (v - p) = -1
      # t=1: v=-1, g=2, p=0 -> v = v - g - 0.5  * (v - p) = -2.5
      # t=2: v=-3.5, g=3, p=0 -> v = v - g - 0.5 * (v - p) = -4.25
      fake_update = float(t + 1)
      updates, opt_state = tx.update(
          {'var': jnp.array(fake_update)}, opt_state, mdl_vars)
      mdl_vars = optax.apply_updates(mdl_vars, updates)
    expected_var_value = -4.25
    self.assertEqual(
        mdl_vars['var'], jnp.array(expected_var_value, dtype=jnp.float32))

  def test_ewc_regularization_extra_args(self):
    opt_tpl = pax_fiddle.Config(
        optimizers.OptaxOptimizer,
        lr_schedule=pax_fiddle.Config(schedules.Constant, value=1.0),
        learning_rate=1.0,
        ewc_regularizer_weight=1.0,
        ewc_weight_per_var={'var': 1.0},
        grad_tx=optax.sgd(1.0),
    )
    acc_opt = optimizers.instantiate(opt_tpl)
    tx = acc_opt.get_grad_transformation()
    mdl_vars = {'var': jnp.array(0, dtype=jnp.float32)}
    opt_state = tx.init(mdl_vars)
    num_steps = 3
    for t in range(num_steps):
      # t=0: v=0, g=1, p=0 -> v = v - g - 0.5  * (v - p) = -1
      # t=1: v=-1, g=2, p=0 -> v = v - g - 0.5  * (v - p) = -2.5
      # t=2: v=-3.5, g=3, p=0 -> v = v - g - 0.5 * (v - p) = -4.25
      fake_update = float(t + 1)
      updates, opt_state = tx.update(
          {'var': jnp.array(fake_update)}, opt_state, mdl_vars
      )
      mdl_vars = optax.apply_updates(mdl_vars, updates)
    expected_var_value = -4.25
    self.assertEqual(
        mdl_vars['var'], jnp.array(expected_var_value, dtype=jnp.float32)
    )

  @parameterized.named_parameters(
      ('ewc_regularization_with_per_var_weight', True),
      ('ewc_regularization_without_per_var_weight', False),
  )
  def test_ewc_regularization_with_partitioning(
      self, ewc_regularization_with_per_var_weight
  ):
    mesh_shape = [1, 2, 1]
    num_devices = np.prod(mesh_shape)
    mesh_shape = np.arange(num_devices).reshape(mesh_shape)

    mdl_vars = py_utils.NestedMap()
    mdl_vars.lm = py_utils.NestedMap(w=jnp.array([[0, 0]]).astype('float32'))
    mdl_vars.ffn = jnp.array([[0, 0]]).astype('float32')

    opt_tpl = pax_fiddle.Config(
        optimizers.OptaxOptimizer,
        lr_schedule=pax_fiddle.Config(schedules.Constant, value=1.0),
        learning_rate=1.0,
        ewc_regularizer_weight=1.0,
        grad_tx=optax.sgd(1.0),
    )
    if ewc_regularization_with_per_var_weight:
      ewc_weight_per_var = py_utils.NestedMap()
      ewc_weight_per_var.lm = py_utils.NestedMap(
          w=jnp.array([[2.0, 2.0]]).astype('float32')
      )
      ewc_weight_per_var.ffn = jnp.array([[2.0, 2.0]]).astype('float32')

      opt_tpl.ewc_weight_per_var = ewc_weight_per_var
    else:
      opt_tpl.ewc_weight_per_var = None
    acc_opt = optimizers.instantiate(opt_tpl)
    tx = acc_opt.get_grad_transformation()

    var_weight_hparams = jax.tree_map(
        lambda v: base_layer.WeightHParams(
            v.shape, mesh_shape=mesh_shape, tensor_split_dims_mapping=[-1, 1]
        ),
        mdl_vars,
    )

    opt_state = tx.init(mdl_vars)
    opt_states_pspec = optimizers.partition_params(
        tx, var_weight_hparams, opt_state
    )
    jax.tree_map(
        lambda x, y: True,
        opt_states_pspec,
        opt_state,
        is_leaf=lambda x: isinstance(x, base_layer.WeightHParams),
    )
    num_steps = 3
    for t in range(num_steps):
      update_val = float(t + 1)
      fake_update = jax.tree_map(
          lambda x: jnp.array([[update_val, update_val]]), mdl_vars
      )
      updates, opt_state = tx.update(
          fake_update,
          opt_state,
          mdl_vars,
      )
      mdl_vars = optax.apply_updates(mdl_vars, updates)

    if ewc_regularization_with_per_var_weight:
      # with ewc_weight_per_var=2.0 for each var and ewc_regularizer_weight=1.0:
      # t=0: v=0, g=1, p=0 -> v = v - g - 1  * (v - p) = -2
      # t=1: v=-2, g=2, p=0 -> v = v - g - 1  * (v - p) = -2 -2 + 2 = -2
      # t=2: v=-2, g=3, p=0 -> v = v - g - 1 * (v - p) = -2 -3 +2 = -3
      expected_var_value = -3.0
    else:
      # with no ewc_weight_per_var and ewc_regularizer_weight= 1.0:
      # t=0: v=0, g=1, p=0 -> v = v - g - 0.5  * (v - p) = -1
      # t=1: v=-1, g=2, p=0 -> v = v - g - 0.5  * (v - p) = -2.5
      # t=2: v=-3.5, g=3, p=0 -> v = v - g - 0.5 * (v - p) = -4.25
      expected_var_value = -4.25

    self.assertAllClose(mdl_vars['lm']['w'], expected_var_value)
    self.assertAllClose(mdl_vars['ffn'], expected_var_value)


class OptimizersRegularizationTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('l2_regularizer', 2.0, 0.0, 0.0, False, -3.0, False),
      ('l2_regularizer_mask', 2.0, 0.0, 0.0, True, 1.0, False),
      ('l1_regularizer', 0.0, 2.0, 0.0, False, -1.0, False),
      ('l1_regularizer_mask', 0.0, 2.0, 0.0, True, 1.0, False),
      ('decoupled_weight_decay', 0.0, 0.0, 2.0, False, -3.0, False),
      ('decoupled_weight_decay_mask', 0.0, 0.0, 2.0, True, 1.0, False),
      ('l2_regularizer_with_opt_wrapper', 2.0, 0.0, 0.0, False, -3.0, True),
      ('l2_regularizer_mask_with_opt_wrapper', 2.0, 0.0, 0.0, True, 1.0, True),
      ('l1_regularizer_with_opt_wrapper', 0.0, 2.0, 0.0, False, -1.0, True),
      ('l1_regularizer_mask_with_opt_wrapper', 0.0, 2.0, 0.0, True, 1.0, True),
      (
          'decoupled_weight_decay_with_opt_wrapper',
          0.0,
          0.0,
          2.0,
          False,
          -3.0,
          True,
      ),
      (
          'decoupled_weight_decay_mask_with_opt_wrapper',
          0.0,
          0.0,
          2.0,
          True,
          1.0,
          True,
      ),
  )
  def test_regularizer(
      self,
      l2_regularizer_weight,
      l1_regularizer_weight,
      decoupled_weight_decay,
      create_regularizer_mask,
      expected_value,
      with_opt_wrapper,
  ):
    output = _run_transformation(
        l2_regularizer_weight=l2_regularizer_weight,
        l1_regularizer_weight=l1_regularizer_weight,
        decoupled_weight_decay=decoupled_weight_decay,
        create_regularizer_mask=create_regularizer_mask,
        use_native_optax_gradient_transformations=with_opt_wrapper,
    )
    self.assertEqual(output, jnp.array(expected_value, dtype=jnp.float32))


class CustomMaskedTest(test_utils.TestCase, parameterized.TestCase):
  """Tests for the masked wrapper."""

  def test_masked(self):
    mask = {'a': True, 'b': False, 'c': {'d': False, 'e': True}}
    params = {
        'a': 1.0,
        'b': {'f': 2.0},
        'c': {'d': 3.0, 'e': ([4.0, 5.0], 6.0)},
    }
    params = jax.tree_util.tree_map(jnp.asarray, params)

    opt_tpl = pax_fiddle.Config(
        optimizers.OptaxOptimizer,
        lr_schedule=pax_fiddle.Config(schedules.Constant, value=1.0),
        learning_rate=1.0,
        grad_tx=optax.sgd(1.0),
    )
    var_weight_hparams = jax.tree_map(
        lambda x: base_layer.WeightHParams(x.shape), params
    )

    sgd = optimizers.instantiate(opt_tpl)
    masked_opt = optimizers.sharded_masked(
        sgd.get_grad_transformation(var_weight_hparams, include_ema=False),
        mask,
    )

    opt_state = masked_opt.init(var_weight_hparams)
    # Should succeed
    _ = optimizers.partition_params(masked_opt, var_weight_hparams, opt_state)

    num_steps = 3
    for t in range(num_steps):
      update_val = float(t + 1)
      fake_update = jax.tree_map(lambda x: jnp.array(update_val), params)
      updates, opt_state = masked_opt.update(
          fake_update,
          opt_state,
          params,
      )

      def _negate_updates_for_masked(m, upd):
        return jax.tree_map(lambda x: -x, upd) if m else upd

      expected_updates = jax.tree_map(
          _negate_updates_for_masked,
          mask,
          fake_update,
      )

      jax.tree_map(self.assertAllClose, updates, expected_updates)
      expected_params = jax.tree_util.tree_map(
          lambda p, u: p + u, params, expected_updates
      )
      params = optax.apply_updates(params, updates)
      jax.tree_map(self.assertAllClose, params, expected_params)


if __name__ == '__main__':
  absltest.main()
