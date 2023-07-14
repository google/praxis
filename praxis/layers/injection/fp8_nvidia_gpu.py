# coding=utf-8
# Copyright 2023 The Pax Authors.
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

"""Op wrappers to support FP8 GEMMs."""

from functools import partial

from jax import custom_vjp
from jax import lax
from jax import numpy as jnp

from praxis import base_layer
from praxis import layers
from praxis import pax_fiddle
from praxis import pytypes

WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams
JTensor = pytypes.JTensor
OVERWRITE_WITH_GRADIENT = \
    base_layer.WeightHParamsCollection.OVERWRITE_WITH_GRADIENT


def get_fp8_max(fp8_dtype, out_dtype):
  assert fp8_dtype in (jnp.float8_e4m3fn, jnp.float8_e5m2)
  return jnp.finfo(fp8_dtype).max.astype(out_dtype)

def quantize(x, q_dtype, scale, compute_dtype):
  # We need to explicitly cast the max value to compute_dtype, otherwise the jax
  # dtype promotion will cast the scaled_x to fp32 in the following ops, which
  # would violate the fp8-matmul pattern matching.
  dtype_max = get_fp8_max(q_dtype, compute_dtype)

  scaled_x = x / scale.astype(compute_dtype)

  clipped_x = jnp.clip(scaled_x, -dtype_max, dtype_max)

  return clipped_x.astype(q_dtype)

def dequantize(x, dq_dtype, scale):
  return x.astype(dq_dtype) * scale.astype(dq_dtype)

def quantize_dequantize(x, q_dtype, scale, compute_dtype):
  qx = quantize(x, q_dtype, scale, compute_dtype)
  return dequantize(qx, x.dtype, scale)

def compute_scale(amax, scale, fp8_max, margin=0):
  """Default function to convert amax to scaling factor."""
  # This function copied from the TransformerEngine is used to compute its
  # `scale`. However, our scale matches its `scale_inv` concept. So, we apply
  # the reciprocal operation at the entry and exit of the function.
  scale = 1.0 / scale
  exp = jnp.floor(jnp.log2(fp8_max / amax)) - margin
  sf = jnp.round(lax.pow(2., jnp.abs(exp)))
  sf = jnp.where(amax > 0.0, sf, scale)
  sf = jnp.where(lax.is_finite(amax), sf, scale)
  sf = jnp.where(exp < 0, 1.0 / sf, sf)
  return 1.0 / sf

def compute_scale_and_amax_history(x, q_dtype, scale, amax_history):
  dtype_max = get_fp8_max(q_dtype, jnp.float32)

  amax_update = jnp.max(jnp.abs(x)).astype(scale.dtype)
  new_amax_history = \
      jnp.roll(amax_history, shift=-1, axis=0).at[0].set(amax_update)

  amax_from_history = jnp.max(new_amax_history, axis=0)
  new_scale = compute_scale(amax_from_history, scale, dtype_max)
  return new_scale, new_amax_history

def qdq_and_return(x, q_dtype, scale, amax_history, compute_dtype):
  qx = quantize_dequantize(x, q_dtype, scale, compute_dtype)
  new_scale, new_amax_history = compute_scale_and_amax_history(
      x, q_dtype, scale, amax_history)
  return qx, new_scale, new_amax_history

# For variables in the collection of OVERWRITE_WITH_GRADIENT, their gradients
# will be used as the new variables in the next step. Here, the scale and
# amax_history are in such collection.
@partial(custom_vjp, nondiff_argnums=(0,))
def in_qdq(compute_dtype, inp, scale, amax_history):
  qin, _, _ = qdq_and_return(
      inp, jnp.float8_e4m3fn, scale, amax_history, compute_dtype)
  return qin

def in_qdq_fwd(compute_dtype, inp, scale, amax_history):
  qin, new_scale, new_amax_history = qdq_and_return(
      inp, jnp.float8_e4m3fn, scale, amax_history, compute_dtype)
  return qin, (new_scale, new_amax_history)

def in_qdq_bwd(compute_dtype, res, g):
  new_scale, new_amax_history = res
  q_g = g
  return q_g, new_scale, new_amax_history

in_qdq.defvjp(in_qdq_fwd, in_qdq_bwd)


@partial(custom_vjp, nondiff_argnums=(0,))
def out_qdq(compute_dtype, out, scale, amax_history):
  return out

def out_qdq_fwd(compute_dtype, out, scale, amax_history):
  return out, (scale, amax_history)

def out_qdq_bwd(compute_dtype, res, g):
  scale, amax_history = res
  q_g, new_scale, new_amax_history = qdq_and_return(
      g, jnp.float8_e5m2, scale, amax_history, compute_dtype)
  return q_g, new_scale, new_amax_history
  
out_qdq.defvjp(out_qdq_fwd, out_qdq_bwd)


def fp8_einsum(eqn, x, k, compute_dtype, x_scale, x_amax_history, k_scale,
               k_amax_history, dy_scale, dy_amax_history):
  """Perform any einsum formula.  """

  x_qdq = in_qdq(compute_dtype, x, x_scale, x_amax_history)

  k_qdq = in_qdq(compute_dtype, k, k_scale, k_amax_history)

  y_qdq = jnp.einsum(eqn, x_qdq, k_qdq)

  y = out_qdq(compute_dtype, y_qdq, dy_scale, dy_amax_history)

  return y


class Fp8EinsumOp(base_layer.BaseLayer):
  """Wrapper around jnp.einsum used in standard Pax layers."""
  amax_history_length: int = 1024

  def setup(self) -> None:
    scale_args = {
        'shape': [1],
        'init': WeightInit.Constant(1.0),
        'dtype': jnp.float32,
        'mesh_shape': self.mesh_shape,
        'tensor_split_dims_mapping': None,
        'collections': [OVERWRITE_WITH_GRADIENT],
    }
    amax_history_args = {
        'shape': [self.amax_history_length],
        'init': WeightInit.Constant(0.0),
        'dtype': jnp.float32,
        'mesh_shape': self.mesh_shape,
        'tensor_split_dims_mapping': None,
        'collections': [OVERWRITE_WITH_GRADIENT],
    }
    self.create_variable(
        'input_amax_history', WeightHParams(**amax_history_args))
    self.create_variable(
        'kernel_amax_history', WeightHParams(**amax_history_args))
    self.create_variable(
        'output_grad_amax_history', WeightHParams(**amax_history_args))

    self.create_variable('input_scale', WeightHParams(**scale_args))
    self.create_variable('kernel_scale', WeightHParams(**scale_args))
    self.create_variable(
         'output_grad_scale', WeightHParams(**scale_args))

  def __call__(self, equation: str, *args: JTensor) -> JTensor:

    assert len(args) == 2
    x = args[0]
    k = args[1]

    comp_dtype = self.fprop_dtype
    assert (
        k.dtype == comp_dtype
    ), f'k dtype has to be {comp_dtype}, but got {k.dtype}'
    x = jnp.asarray(x, comp_dtype)

    theta = self.theta
    out = fp8_einsum(equation, x, k, comp_dtype, theta.input_scale,
                     theta.input_amax_history, theta.kernel_scale,
                     theta.kernel_amax_history, theta.output_grad_scale,
                     theta.output_grad_amax_history)
    return out


def tr_set_fp8_quantization(
    transformer_layer_p: pax_fiddle.Config[layers.transformers.Transformer]
):
  """Inject Fp8EinsumOp to desired layers in transformer."""
  transformer_layer_p.tr_atten_tpl.proj_tpl.einsum_tpl = \
      pax_fiddle.Config(Fp8EinsumOp)
  transformer_layer_p.tr_atten_tpl.combined_qkv_proj_tpl.einsum_tpl = \
      pax_fiddle.Config(Fp8EinsumOp)
  transformer_layer_p.tr_fflayer_tpl.fflayer_tpl.linear_tpl.einsum_tpl = \
      pax_fiddle.Config(Fp8EinsumOp)
