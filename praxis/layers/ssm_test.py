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

"""Tests for Praxis SSM layers."""

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import pax_fiddle
from praxis import py_utils
from praxis import test_utils
from praxis.layers import ssm
import tensorflow.compat.v2 as tf

to_np = test_utils.to_np
to_tf_nmap = test_utils.to_tf_nmap
instantiate = base_layer.instantiate


class SSMTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)
    tf.random.set_seed(123)

  @parameterized.named_parameters(
      {
          'testcase_name': 'ss4d-1d-legs',
          'hippo_type': 'ss4d-1d-legs',
      },
      {
          'testcase_name': 'ss4d-1d-lagt',
          'hippo_type': 'ss4d-1d-lagt',
      },
      {
          'testcase_name': 'ss4d-1d',
          'hippo_type': 'ss4d-1d',
      },
  )
  def test_s4d_layer(self, hippo_type):
    p = pax_fiddle.Config(
        ssm.SSM,
        name='ssm',
        nheads=5,
        dim=1,
        l_max=2,
        decode_num_samples=1,
        step_size=1.,
        hippo_type=hippo_type
    )
    s4d = instantiate(p)
    npy_input = np.random.normal(1.0, 0.5,
                                 [2, p.l_max, p.dim]).astype('float32')
    inputs = jnp.asarray(npy_input)
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = s4d.init(prng_key, inputs)

    # Test convolution/fft.
    outputs, ssm_state = s4d.apply(
        initial_vars, inputs, mutable=[base_layer.DECODE_CACHE])
    logging.info('outputs = %s', outputs)

    # Test extend_step.
    out_step = []
    updated_vars = py_utils.merge_dict(ssm_state, initial_vars)
    logging.info('init_vars w state = %s', updated_vars)
    for i in range(p.l_max):
      out, ssm_state = s4d.apply(
          updated_vars, inputs[:, i, :], method=s4d.extend_step,
          mutable=[base_layer.DECODE_CACHE])
      logging.info('outputs = %s', out)
      logging.info('ssm_states = %s', ssm_state)
      updated_vars['decoder_cache'] = ssm_state['decoder_cache']
      out_step.append(out)

    out_step = jnp.stack(out_step, axis=1)

    # Make sure the convolution/fft gets the same results as extend_step.
    self.assertAllClose(to_np(outputs), to_np(out_step), atol=1e-6)

if __name__ == '__main__':
  absltest.main()
