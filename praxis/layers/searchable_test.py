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

"""Tests for praxis.layers.searchable."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
from praxis import base_hyperparams
from praxis import base_layer
from praxis import pax_fiddle
from praxis import test_utils
from praxis.layers import linears
from praxis.layers import searchable

instantiate = base_hyperparams.instantiate
weight_init = base_layer.WeightInit


class AutomlSelectTest(test_utils.TestCase):

  def test_automl_select(self):
    p = pax_fiddle.Config(searchable.AutoMLSelect)
    p.search_options_tpl = [
        pax_fiddle.Config(
        linears.Linear,
        name='jax_ffn0',
        weight_init=weight_init.Constant(0.0),
        input_dims=1,
        output_dims=1),
        pax_fiddle.Config(
        linears.Linear,
        name='jax_ffn1',
        weight_init=weight_init.Constant(1.0),
        input_dims=1,
        output_dims=1),
    ]

    m = instantiate(p)
    x = jnp.ones(1, dtype=jnp.float32)
    m_vars = m.init(jax.random.PRNGKey(0), x)
    m_vars['non_trainable']['decision'] = 0
    x1 = m.apply(m_vars, x)
    m_vars['non_trainable']['decision'] = 1
    x2 = m.apply(m_vars, x)
    self.assertAllClose(x1, jnp.zeros(1, dtype=jnp.float32))
    self.assertAllClose(x2, jnp.ones(1, dtype=jnp.float32))


if __name__ == '__main__':
  absltest.main()
