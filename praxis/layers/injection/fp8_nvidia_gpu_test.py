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

from absl.testing import absltest
from flax.linen.fp8_ops import quantize_dequantize
import jax
from jax import numpy as jnp
from praxis import base_layer
from praxis import pax_fiddle
from praxis import test_utils
from praxis.layers import linears
from praxis.layers import pipeline
from praxis.layers.injection import fp8_nvidia_gpu as fp8_ops

PARAMS = base_layer.PARAMS

class Fp8LinearsTest(test_utils.TestCase):

  def test_fp8_einsum_injection(self):
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

      init_fn = jax.jit(ffn.init)
      initial_vars = init_fn(
          {
              PARAMS: init_key,
              'random': init_key,
          },
          inputs,
      )
      initial_vars[PARAMS]['w'] = cast_to_representable(
          initial_vars[PARAMS]['w'], jnp.float8_e4m3fn
      )
      vars_shapes = jax.tree.map(jnp.shape, initial_vars)
      self.assertEqual(vars_shapes, expected_shapes)

      def _train(variables, x):
        y = ffn.apply(variables, x)
        loss = y * dy
        return jnp.mean(loss)

      train_fn = jax.jit(jax.value_and_grad(_train, argnums=[0, 1]))
      outputs, grads = train_fn(initial_vars, inputs)

      return outputs, grads

    expected_shapes_original = {
        PARAMS: {'w': (32, 64)},
    }

    expected_shapes_new = {
        PARAMS: {
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
    einsum_tpl = pax_fiddle.Config(fp8_ops.Fp8EinsumOp)
    output2a, output2b = run(einsum_tpl, expected_shapes_new)
    dw1, dw2 = output1b[0][PARAMS]['w'], output2b[0][PARAMS]['w']
    dx1, dx2 = output1b[1], output2b[1]
    self.assertAllClose(output1a, output2a)
    self.assertAllClose(dx1, dx1)
    self.assertAllClose(dw1, dw2, rtol=1e-04, atol=1e-04)

  def test_pipeline(self):
    NUM_MB = 3
    einsum_p = pax_fiddle.Config(
        fp8_ops.Fp8EinsumOp,
        name='op',
        amax_history_length=3,
    )
    layer_p = pax_fiddle.Config(
        linears.Linear,
        name='layer',
        einsum_tpl=einsum_p,
        input_dims=1,
        output_dims=1,
    )
    p = pax_fiddle.Config(
        pipeline.LayerwiseShardablePipelined,
        name='pipelined',
        num_stages=1,
        single_stage_body=layer_p,
        num_microbatches=NUM_MB,
        microbatch_size=None,
    )
    model = base_layer.instantiate(p)
    init_fn = jax.jit(model.init)

    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key, random_key = jax.random.split(prng_key, 3)
    in_shape = (NUM_MB, 1)
    variables = init_fn(init_key, jnp.ones(in_shape))

    def loss(vars_f32, arr_x):
      y = model.apply(vars_f32, arr_x)
      l = jnp.max(y)
      return l

    jitted_loss_grads = jax.jit(jax.grad(loss, (0, 1)))

    AH = 'input_amax_history'
    SF = 'input_scale'
    # 1st iteration
    new_vars, grads = jitted_loss_grads(variables, jnp.full(in_shape, 2.0))
    self.assertAllClose(
        new_vars[PARAMS]['body']['einsum'][AH], [[2.0, 0.0, 0.0]]
    )
    self.assertAllClose(new_vars[PARAMS]['body']['einsum'][SF], [[1.0]])
    # 2nd iteration
    new_vars, grads = jitted_loss_grads(new_vars, jnp.full(in_shape, 3.0))
    self.assertAllClose(
        new_vars[PARAMS]['body']['einsum'][AH], [[3.0, 0.0, 2.0]]
    )
    self.assertAllClose(new_vars[PARAMS]['body']['einsum'][SF], [[2.0 / 448]])
    # 3rd iteration
    new_vars, grads = jitted_loss_grads(new_vars, jnp.full(in_shape, 4.0))
    self.assertAllClose(
        new_vars[PARAMS]['body']['einsum'][AH], [[4.0, 2.0, 3.0]]
    )
    self.assertAllClose(new_vars[PARAMS]['body']['einsum'][SF], [[3.0 / 448]])


if __name__ == '__main__':
  absltest.main()
