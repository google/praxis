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
from jax import numpy as jnp
import optax
from praxis import base_layer
from praxis import optimizers
from praxis import pax_fiddle
from praxis import schedules
from praxis import test_utils


def _run_transformation(num_steps=1,
                        initial_value=2.0,
                        l2_regularizer_weight=0.0,
                        l1_regularizer_weight=0.0,
                        decoupled_weight_decay=0.0,
                        create_regularizer_mask=False):
  """Applies a gradient transformation with Adam for some steps.

  Args:
    num_steps: Number of steps to apply transformation for.
    initial_value: Initial value of the model parameter.
    l2_regularizer_weight: Optional L2 regularization weight.
    l1_regularizer_weight: Optional L2 regularization weight.
    decoupled_weight_decay: Optional decoupled weight decay.
    create_regularizer_mask: Whether to create mask for regularization.
  Returns:
    The updated states and the final update.
  """
  param_name = 'var'
  mdl_vars = {param_name: jnp.array(initial_value, dtype=jnp.float32)}
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


class OptimizersTest(test_utils.TestCase):

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


class OptimizersRegularizationTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('l2_regularizer', 2.0, 0.0, 0.0, False, -3.0),
      ('l2_regularizer_mask', 2.0, 0.0, 0.0, True, 1.0),
      ('l1_regularizer', 0.0, 2.0, 0.0, False, -1.0),
      ('l1_regularizer_mask', 0.0, 2.0, 0.0, True, 1.0),
      ('decoupled_weight_decay', 0.0, 0.0, 2.0, False, -3.0),
      ('decoupled_weight_decay_mask', 0.0, 0.0, 2.0, True, 1.0))
  def test_regularizer(
      self, l2_regularizer_weight, l1_regularizer_weight,
      decoupled_weight_decay, create_regularizer_mask, expected_value):
    output = _run_transformation(
        l2_regularizer_weight=l2_regularizer_weight,
        l1_regularizer_weight=l1_regularizer_weight,
        decoupled_weight_decay=decoupled_weight_decay,
        create_regularizer_mask=create_regularizer_mask)
    self.assertEqual(output, jnp.array(expected_value, dtype=jnp.float32))


if __name__ == '__main__':
  absltest.main()
