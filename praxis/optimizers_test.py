# coding=utf-8
# Copyright 2022 Google LLC.
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
from jax import numpy as jnp
import optax
from praxis import optimizers
from praxis import schedules
from praxis import test_utils


class OptimizersTest(test_utils.TestCase):

  def test_static_accumulator(self):
    base_opt_tpl = optimizers.ShardedSgd.HParams()
    num_sub_batches = 3
    acc_opt_tpl = optimizers.ShardedStaticAccumulator.HParams(
        optimizer_tpl=base_opt_tpl,
        num_sub_batches=num_sub_batches,
        lr_schedule=schedules.Constant.HParams(value=1.0),
        learning_rate=1.0)
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


if __name__ == '__main__':
  absltest.main()
