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

from typing import Union

from flax.linen import fp8_ops
from jax import numpy as jnp
from praxis import base_layer
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
  use_direct_quant: bool = True

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

  def __call__(self, equation: str, *args: JTensor) -> JTensor:
    assert len(args) == 2
    x = args[0]
    k = args[1]

    comp_dtype = self.fprop_dtype
    assert (
        k.dtype == comp_dtype
    ), f'k dtype has to be {comp_dtype}, but got {k.dtype}'
    x = jnp.asarray(x, comp_dtype)

    if self.use_direct_quant:
      def _quantized_dot_general(
          lhs, rhs, dimension_numbers, precision=None,
          preferred_element_type=None
      ):
        theta = self.theta
        return fp8_ops.q_dot_dq(
            lhs,
            rhs,
            lhs_scale=theta.input_scale,
            rhs_scale=theta.kernel_scale,
            out_grad_scale=theta.output_grad_scale,
            lhs_amax_history=theta.input_amax_history,
            rhs_amax_history=theta.kernel_amax_history,
            out_grad_amax_history=theta.output_grad_amax_history,
            compute_dtype=comp_dtype,
            dimension_numbers=dimension_numbers,
            precision=precision,
            preferred_element_type=preferred_element_type,
        )
      y = jnp.einsum(equation, x, k, _dot_general=_quantized_dot_general)
    else:
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

    theta = self.theta

    if self.use_direct_quant:
      q_x, new_input_scale = fp8_ops.in_q(comp_dtype, jnp.float8_e4m3fn, x, theta.input_scale, theta.input_amax_history)
    #   def create_one_sided_q_dot_dq(comp_dtype, q_x, new_input_scale, kernel_scale, out_grad_scale, kernel_amax_history, out_grad_amax_history):
    #     def _quantized_one_sided_dot_general(
    #         lhs, rhs, dimension_numbers, precision=None,
    #         preferred_element_type=None
    #     ):
    #       return fp8_ops.one_sided_q_dot_dq(
    #           lhs=lhs,
    #           q_lhs=q_x,
    #           lhs_scale=new_input_scale,
    #           rhs=rhs,
    #           rhs_scale=kernel_scale,
    #           out_grad_scale=out_grad_scale,
    #           rhs_amax_history=kernel_amax_history,
    #           out_grad_amax_history=out_grad_amax_history,
    #           compute_dtype=comp_dtype,
    #           dimension_numbers=dimension_numbers,
    #           precision=precision,
    #           preferred_element_type=preferred_element_type
    #       )
    #     return _quantized_one_sided_dot_general

    #   _one_sided_quantized_dot_general = create_one_sided_q_dot_dq(
    #       comp_dtype, q_x, new_input_scale, 
    #       theta.kernel_scale, theta.out_grad_scale, 
    #       theta.kernel_amax_history, theta.out_grad_amax_history
    #   )
  
    #   _one_sided_quantized_dot_general_gated = create_one_sided_q_dot_dq(
    #       comp_dtype, q_x, new_input_scale, 
    #       theta.kernel_scale_gated, theta.out_grad_scale_gated, 
    #       theta.kernel_amax_history_gated, theta.out_grad_amax_history_gated
    #   )
      def one_sided_q_dot_dq(comp_dtype, q_x, new_input_scale, kernel_scale, out_grad_scale, kernel_amax_history, out_grad_amax_history):
        def decorator(func):
          @wraps(func)
          def wrapper(lhs, rhs, dimension_numbers, precision=None, preferred_element_type=None):
            return fp8_ops.one_sided_q_dot_dq(
                lhs=lhs,
                q_lhs=q_x,
                lhs_scale=new_input_scale,
                rhs=rhs,
                rhs_scale=kernel_scale,
                out_grad_scale=out_grad_scale,
                rhs_amax_history=kernel_amax_history,
                out_grad_amax_history=out_grad_amax_history,
                compute_dtype=comp_dtype,
                dimension_numbers=dimension_numbers,
                precision=precision,
                preferred_element_type=preferred_element_type
            )
          return wrapper
        return decorator
      common_args = (comp_dtype, q_x, new_input_scale)
      main_fp8_metas = (
          theta.kernel_scale, theta.out_grad_scale, 
          theta.kernel_amax_history, theta.out_grad_amax_history
      )
      gated_fp8_metas = (
          theta.kernel_scale_gated, theta.out_grad_scale_gated, 
          theta.kernel_amax_history_gated, theta.out_grad_amax_history_gated
      )
      _dot_general_main = one_sided_q_dot_dq(*common_args, *main_fp8_metas)
      _dot_general_gated = one_sided_q_dot_dq(*common_args, *gated_fp8_metas)

      y = jnp.einsum(equation, x, k, _dot_general=_dot_general_main)
      y_gated = jnp.einsum(equation, x, k_gated, _dot_general=_dot_general_gated)
    else:
      y, x_qdq = self.quantized_einsum(
          equation, x, k, return_quantized_x=True
      )
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
