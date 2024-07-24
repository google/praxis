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
from typing import Union

from flax.linen import fp8_ops
from jax import custom_vjp
from jax import lax
from jax import numpy as jnp
from praxis import base_layer
from praxis import layers
from praxis import pax_fiddle
from praxis import pytypes

JTensor = pytypes.JTensor


def _get_fp8_args(amax_history_length, mesh_shape):
  OVERWRITE_WITH_GRADIENT = (
      base_layer.WeightHParamsCollection.OVERWRITE_WITH_GRADIENT
  )
  DISALLOW_BFLOAT16_CONVERSION = (
      base_layer.WeightHParamsCollection.DISALLOW_BFLOAT16_CONVERSION
  )
  scale_args = {
      'shape': [1],
      'init': base_layer.WeightInit.Constant(1.0),
      'dtype': jnp.float32,
      'mesh_shape': mesh_shape,
      'tensor_split_dims_mapping': None,
      'collections': [OVERWRITE_WITH_GRADIENT, DISALLOW_BFLOAT16_CONVERSION],
  }
  amax_history_args = {
      'shape': [amax_history_length],
      'init': base_layer.WeightInit.Constant(0.0),
      'dtype': jnp.float32,
      'mesh_shape': mesh_shape,
      'tensor_split_dims_mapping': None,
      'collections': [OVERWRITE_WITH_GRADIENT, DISALLOW_BFLOAT16_CONVERSION],
  }
  return scale_args, amax_history_args


class Fp8EinsumOp(base_layer.BaseLayer):
  """Wrapper around jnp.einsum used in standard Pax layers."""

  amax_history_length: int = 1024

  def setup(self) -> None:
    scale_args, amax_history_args = _get_fp8_args(
        self.amax_history_length, self.mesh_shape
    )

    self.create_variable(
        'input_amax_history', base_layer.WeightHParams(**amax_history_args)
    )
    self.create_variable(
        'kernel_amax_history', base_layer.WeightHParams(**amax_history_args)
    )
    self.create_variable(
        'output_grad_amax_history',
        base_layer.WeightHParams(**amax_history_args),
    )

    self.create_variable('input_scale', base_layer.WeightHParams(**scale_args))
    self.create_variable('kernel_scale', base_layer.WeightHParams(**scale_args))
    self.create_variable(
        'output_grad_scale', base_layer.WeightHParams(**scale_args)
    )

  def quantized_einsum(
      self, equation: str, x: JTensor, k: JTensor, return_quantized_x: bool
  ) -> JTensor | tuple[JTensor, JTensor]:
    theta = self.theta
    comp_dtype = self.fprop_dtype

    x_qdq = fp8_ops.in_qdq(
        comp_dtype,
        jnp.float8_e4m3fn,
        x,
        theta.input_scale,
        theta.input_amax_history,
    )
    k_qdq = fp8_ops.in_qdq(
        comp_dtype,
        jnp.float8_e4m3fn,
        k,
        theta.kernel_scale,
        theta.kernel_amax_history,
    )
    y_qdq = jnp.einsum(
        equation, x_qdq, k_qdq, _dot_general=fp8_ops.dot_general_with_precision
    )
    y = fp8_ops.out_qdq(
        comp_dtype,
        jnp.float8_e5m2,
        y_qdq,
        theta.output_grad_scale,
        theta.output_grad_amax_history,
    )

    if return_quantized_x:
      return y, x_qdq
    return y

  def __call__(
      self, equation: str, *args: JTensor
  ) -> Union[JTensor, tuple[JTensor, JTensor]]:
    assert len(args) == 2
    x = args[0]
    k = args[1]

    comp_dtype = self.fprop_dtype
    assert (
        k.dtype == comp_dtype
    ), f'k dtype has to be {comp_dtype}, but got {k.dtype}'
    x = jnp.asarray(x, comp_dtype)

    y = self.quantized_einsum(equation, x, k, return_quantized_x=False)

    return y


class Fp8EinsumGatedOp(Fp8EinsumOp):
  """Wrapper around two jnp.einsum for gated FFN."""

  def setup(self) -> None:
    super().setup()
    scale_args, amax_history_args = _get_fp8_args(
        self.amax_history_length, self.mesh_shape
    )

    self.create_variable(
        'kernel_amax_history_gated',
        base_layer.WeightHParams(**amax_history_args),
    )
    self.create_variable(
        'output_grad_amax_history_gated',
        base_layer.WeightHParams(**amax_history_args),
    )

    self.create_variable(
        'kernel_scale_gated', base_layer.WeightHParams(**scale_args)
    )
    self.create_variable(
        'output_grad_scale_gated', base_layer.WeightHParams(**scale_args)
    )

  def __call__(self, equation: str, *args: JTensor) -> tuple[JTensor, JTensor]:
    assert len(args) == 3
    x, k, k_gated = args

    comp_dtype = self.fprop_dtype
    assert (
        k.dtype == k_gated.dtype == comp_dtype
    ), f'k dtype has to be {comp_dtype}, but got {k.dtype} and {k_gated.dtype}'
    x = jnp.asarray(x, comp_dtype)

    y, x_qdq = self.quantized_einsum(equation, x, k, return_quantized_x=True)

    theta = self.theta

    k_gated_qdq = fp8_ops.in_qdq(
        comp_dtype,
        jnp.float8_e4m3fn,
        k_gated,
        theta.kernel_scale_gated,
        theta.kernel_amax_history_gated,
    )
    y_gated_qdq = jnp.einsum(
        equation,
        x_qdq,
        k_gated_qdq,
        _dot_general=fp8_ops.dot_general_with_precision,
    )
    y_gated = fp8_ops.out_qdq(
        comp_dtype,
        jnp.float8_e5m2,
        y_gated_qdq,
        theta.output_grad_scale_gated,
        theta.output_grad_amax_history_gated,
    )

    return y, y_gated
