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

"""Tests for Fp8 injection to Praxis linear layers."""

from functools import partial

from absl.testing import absltest, parameterized
from flax.linen.fp8_ops import quantize_dequantize
import jax
from jax import numpy as jnp
from praxis import base_layer
from praxis import pax_fiddle
from praxis import test_utils
from praxis.layers import linears
from praxis.layers.injection import fp8_nvidia_gpu as fp8_ops


class Fp8LinearsTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.parameters([True, False])
  def test_fp8_einsum_injection(self, use_direct_quant):
    # Used to cast the inputs to be representable in FP8, so that the difference
    # of the results from the original gemm and fp8 gemm is small.
    cast_to_representable = partial(
        quantize_dequantize, scale=jnp.ones((1,)), compute_dtype=jnp.float32
    )

    def run(custom_einsum_tpl, expected_shapes):
      p = pax_fiddle.Config(
          linears.Linear,
          name='jax_ffn',
          input_dims=32,
          output_dims=64,
      )
      if custom_einsum_tpl:
        p.set(einsum_tpl=custom_einsum_tpl)

      ffn = base_layer.instantiate(p)
      prng_key = jax.random.PRNGKey(seed=123)
      prng_key, init_key, random_key = jax.random.split(prng_key, 3)
      inputs = jax.random.uniform(random_key, (16, 32))
      inputs = cast_to_representable(inputs, jnp.float8_e4m3fn)
      dy = jax.random.uniform(random_key, (16, 64))
      dy = cast_to_representable(dy, jnp.float8_e5m2)
      initial_vars = ffn.init(
          {
              'params': init_key,
              'random': init_key,
          },
          inputs,
      )
      initial_vars['params']['w'] = cast_to_representable(
          initial_vars['params']['w'], jnp.float8_e4m3fn
      )
      vars_shapes = jax.tree_util.tree_map(jnp.shape, initial_vars)
      self.assertEqual(vars_shapes, expected_shapes)

      def _train(variables, x):
        y = ffn.apply(variables, x)
        loss = y * dy
        return jnp.mean(loss)

      train_fn = jax.jit(jax.value_and_grad(_train, argnums=[0, 1]))
      outputs, grads = train_fn(initial_vars, inputs)

      return outputs, grads

    expected_shapes_original = {
        'params': {'w': (32, 64)},
    }

    expected_shapes_new = {
        'params': {
            'w': (32, 64),
            'einsum': {
                'input_amax_history': (1024,),
                'input_scale': (1,),
                'kernel_amax_history': (1024,),
                'kernel_scale': (1,),
                'output_grad_amax_history': (1024,),
                'output_grad_scale': (1,),
            },
        }
    }

    output1a, output1b = run(None, expected_shapes_original)
    einsum_tpl = pax_fiddle.Config(
        fp8_ops.Fp8EinsumOp, use_direct_quant=use_direct_quant
    )
    output2a, output2b = run(einsum_tpl, expected_shapes_new)
    dw1, dw2 = output1b[0]['params']['w'], output2b[0]['params']['w']
    dx1, dx2 = output1b[1], output2b[1]
    self.assertAllClose(output1a, output2a)
    self.assertAllClose(dx1, dx1)
    self.assertAllClose(dw1, dw2, rtol=1e-04, atol=1e-04)


if __name__ == '__main__':
  absltest.main()
