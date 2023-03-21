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

"""Tests for praxis.layers.quantization.automl_select."""

from absl.testing import absltest

import jax
import jax.numpy as jnp

from praxis import base_hyperparams
from praxis import pax_fiddle
from praxis import test_utils
from praxis.layers.quantization import quantizer
from praxis.layers.quantization import automl_select
from praxis.layers.quantization import quantization_hparams


ActQuantizationParams = quantization_hparams.ActQuantizationParams
instantiate = base_hyperparams.instantiate


class AutomlSelectTest(test_utils.TestCase):

  def test_automl_select(self):
    p = pax_fiddle.Config(automl_select.AutoMLSelect)
    p.search_options_tpl = [
        quantizer.create_tensor_quantizer('int4', ActQuantizationParams(precision=4)),
        quantizer.create_tensor_quantizer('int8', ActQuantizationParams(precision=8)),
    ]

    m = instantiate(p)
    x = jnp.ones((1, 3))
    m_vars = m.init(jax.random.PRNGKey(0), x, 0)
    m_vars['non_trainable']['decision'] = 0
    q_x1, q_s1, _ = m.apply(m_vars, x, 0)
    m_vars['non_trainable']['decision'] = 1
    q_x2, q_s2, _ = m.apply(m_vars, x, 0)
    self.assertNotAllClose(q_x1, q_x2)
    self.assertNotAllClose(q_s1, q_s2)


if __name__ == '__main__':
  absltest.main()
