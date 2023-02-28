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

"""Quantization Aware Training ops."""

from __future__ import annotations
from typing import Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from praxis import base_layer
from praxis import pax_fiddle
from praxis import pytypes
from praxis.layers.quantization import quantization_hparams


WeightHParams = base_layer.WeightHParams
JTensor = pytypes.JTensor
ActQuantizationParams = quantization_hparams.ActQuantizationParams
WeightQuantizationParams = quantization_hparams.WeightQuantizationParams


def _pass_through(x: JTensor, fn) -> JTensor:
  # Create an exactly-zero expression with Sterbenz lemma that has an
  # exactly-one gradient.
  return x - jax.lax.stop_gradient(x) + jax.lax.stop_gradient(fn(x))


def create_tensor_quantizer(
    name: str,
    quant_params: Optional[
        Union[ActQuantizationParams, WeightQuantizationParams]
    ],
) -> pax_fiddle.Config[TensorQuantizer]:
  """Creates tensor quantizer.

  Args:
    name: Tensor quantizer name.
    quant_params: Quantization parameters for weights or activation.

  Returns:
    Tensor quantizer.
  """
  tq_params = pax_fiddle.Config(TensorQuantizer, name=name)
  if quant_params is not None:
    tq_params.precision = quant_params.precision
    tq_params.stop_scale_gradient = quant_params.stop_scale_gradient
    tq_params.unsigned_int_bounds = quant_params.unsigned_int_bounds

  if isinstance(quant_params, WeightQuantizationParams):
    tq_params.min_clipping = quant_params.min_clipping
    tq_params.num_optimize_clipping = quant_params.num_optimize_clipping
    tq_params.add_scale_eps = quant_params.add_scale_eps
    tq_params.use_symmetric = quant_params.use_symmetric
    tq_params.quant_loss_weight = quant_params.quant_loss_weight

  return tq_params


class TensorQuantizer(base_layer.BaseLayer):
  """Maintains state associated with the quantization of an input tensor.

  Attributes:
    precision: Number of bits to quantize to (e.g 4 for int4). Must be positive.
    stop_scale_gradient: Stop the gradient of the quantization scale for
      numerical stability. Note: this is numerically incorrect.
    min_clipping: Clipping value which will be used for clipping optimization
      in range [min_clipping ... 1].
    num_optimize_clipping: Number of optimization steps used for
      scale estimation with search over clipping values in
      range [min_clipping ... 1].
    add_scale_eps: If True add epsilon to scale to avoid division by zero,
      else it will replace zero scale by 1.
    unsigned_int_bounds: If True, use [0, 2**precision-1] clip bound for better
      quantization bucket utilization in case where the input has a positive
      distribution (e.g., ReLU).
    use_symmetric: Do symmetric quantization for weights.
    quant_loss_weight: Weight for quantization loss.
  """
  precision: Optional[int] = None
  stop_scale_gradient: bool = False
  min_clipping: Optional[float] = None
  num_optimize_clipping: Optional[int] = None
  add_scale_eps: Optional[bool] = True
  unsigned_int_bounds: bool = False
  use_symmetric: bool = True
  quant_loss_weight: Optional[float] = None

  def setup(self):
    del self.dtype  # not used

    assert (
        self.precision is None or self.precision <= 23
    ), 'Too many bits, float32 has less precision.'

    if self.min_clipping is not None or self.num_optimize_clipping is not None:
      assert (
          self.min_clipping is not None
          and self.num_optimize_clipping is not None
      ), (
          'Both min_clipping and num_optimize_clipping must be defined '
          'or both must be None.'
      )
      assert self.min_clipping > 0.0 and self.min_clipping < 1.0
      assert self.num_optimize_clipping > 0

  def __call__(self):
    # Since TensorQuantizer does nothing when initialized, __call__ is a no-op.
    pass

  def _get_clip_bound(self) -> Tuple[float, float]:

    # For unsigned 8 bits precision it is [0, 255]
    if self.unsigned_int_bounds:
      return 0, 2.0**self.precision - 1
    else:
      # For signed 8 bits precision it is [-128, 127]
      return (
          -(2.0 ** (self.precision - 1)),
          2.0 ** (self.precision - 1) - 1,
      )

  def _get_scale(
      self,
      x: JTensor,
      contract_dims: Union[int, Sequence[int]],
      clipping: Optional[float] = None,
  ) -> JTensor:
    if self.precision is None:
      return jnp.ones(shape=(1,) * x.ndim, dtype=x.dtype)

    clip_bound_min, clip_bound_max = self._get_clip_bound()

    if self.use_symmetric:
      x_bound = jnp.max(jnp.abs(x), axis=contract_dims, keepdims=True)
      range_bound = clip_bound_max
    else:
      x_max = jnp.max(x, axis=contract_dims, keepdims=True)
      x_min = jnp.min(x, axis=contract_dims, keepdims=True)
      x_bound = x_max - x_min
      range_bound = clip_bound_max - clip_bound_min

    if clipping is not None:
      x_bound *= clipping

    scale = x_bound / range_bound
    if self.stop_scale_gradient:
      scale = jax.lax.stop_gradient(scale)

    if self.add_scale_eps:
      # Add epsilon to avoid NaN gradients for near-zero inputs during training.
      scale = scale + jnp.finfo(x.dtype).eps
    else:
      scale = jnp.where(scale == 0, jnp.ones_like(scale), scale)

    return scale

  def _get_optimal_scale(
      self,
      x: JTensor,
      contract_dims: Union[int, Sequence[int]],
  ) -> JTensor:
    def quantization_error_and_scale(clipping):
      q_scale = self._get_scale(x, contract_dims, clipping=clipping)
      x_scaled, zp_time_scale = self._scale(x, q_scale, contract_dims)
      x_quantized = self.to_quant(x_scaled)
      x_quantized_dequantized = self.dequantize(
          x_quantized, q_scale, contract_dims, zp_time_scale
      )
      sum_error = jnp.sum(jnp.abs(jnp.subtract(x, x_quantized_dequantized)))
      return sum_error, q_scale

    clipping = jnp.linspace(
        1.0, self.min_clipping, num=self.num_optimize_clipping, dtype=x.dtype
    )
    res = jax.vmap(quantization_error_and_scale)(clipping)
    best_ind = jnp.argmin(res[0])
    best_scale = res[1].at[best_ind].get()
    return best_scale

  def get_quant_scale(
      self,
      x: JTensor,
      contract_dims: Union[int, Sequence[int]],
  ) -> JTensor:
    """Computes scale for quantization.

    It can compute standard scale or scale optimized over different
    clipping values in range [min_clipping ... 1.0].

    Args:
      x: Input tensor.
      contract_dims: Axis along which to quantize acts (the non-feature axis).

    Returns:
      Scale tensor.
    """
    if self.min_clipping is not None and self.num_optimize_clipping is not None:
      return self._get_optimal_scale(x, contract_dims)
    else:
      return self._get_scale(x, contract_dims)

  def update(self, x: JTensor):
    # This function is no-op for now. Once static quantization is supported,
    # statistics update will be performed through this function.
    pass

  def to_quant(self, x: JTensor) -> JTensor:
    """Converts normalized float x to quantized value.

    Args:
      x: Input tensor. It has to be normalized: x / scale

    Returns:
      Quantized tensor.
    """

    if self.precision is None:
      return x

    x = _pass_through(x + 0.5, jnp.floor)
    clip_bound_min, clip_bound_max = self._get_clip_bound()
    x = jnp.clip(x, clip_bound_min, clip_bound_max)

    return x

  def _get_zero_point(self, x, contract_dims, scale) -> JTensor:
    x_min = jnp.min(x, axis=contract_dims, keepdims=True)
    clip_bound_min, _ = self._get_clip_bound()
    zp = clip_bound_min - x_min / scale
    return zp

  def dequantize(
      self,
      q_x: JTensor,
      q_scale: JTensor,
      contract_dims: Union[int, Sequence[int]],
      zp_time_scale: Optional[JTensor] = None) -> JTensor:
    """Dequantizes quantized q_x.

    Args:
      q_x: Input tensor.
      q_scale: Quantization scale.
      contract_dims: Contraction dims.
      zp_time_scale: Zero point times q_scale, for asymmetric quantization.

    Returns:
      Dequantized tensor.
    """
    if self.use_symmetric and zp_time_scale is not None:
      raise ValueError('Symmetric quantization can not be used '
                       'with zp_time_scale.')
    if not self.use_symmetric and zp_time_scale is None:
      raise ValueError('Asymmetric quantization need zp_time_scale.')

    if self.use_symmetric:
      deq_q_x = q_x * q_scale
    else:
      deq_q_x = q_x * q_scale - jnp.expand_dims(zp_time_scale, contract_dims)
    return deq_q_x

  def _scale(
      self,
      x: JTensor,
      q_scale: JTensor,
      contract_dims: Union[int, Sequence[int]],
  ) -> Tuple[JTensor, Optional[JTensor]]:
    """Rescales input x for quantization.

    Args:
      x: Input tensor.
      q_scale: Quantization scale.
      contract_dims: Contraction dims.

    Returns:
      Rescaled tensor.
    """

    if self.use_symmetric:
      x_scaled = jnp.divide(x, q_scale)
      zp_time_scale = None
    else:
      zp = self._get_zero_point(x, contract_dims, q_scale)
      x_scaled = jnp.divide(x, q_scale) + zp
      zp_time_scale = jnp.multiply(q_scale, zp).squeeze()

    return x_scaled, zp_time_scale

  def quantize(
      self,
      x: JTensor,
      contract_dims: Union[int, Sequence[int]],
      squeeze_scale=True,
      quantized_dtype: Union[jnp.dtype, None] = None,
  ) -> Tuple[JTensor, JTensor, Optional[JTensor]]:
    """Quantizes input x.

    Args:
      x: Input tensor.
      contract_dims: Contraction dims.
      squeeze_scale: If True it will squeeze output scale.
      quantized_dtype: Output type.

    Returns:
      Quantized tensor with scale (used for dequantization).
    """

    q_s = self.get_quant_scale(x, contract_dims)
    x_scaled, zp_time_scale = self._scale(x, q_s, contract_dims)
    q_x = self.to_quant(x_scaled)

    if (
        quantized_dtype != jnp.int8  # it is used for materialization
        and self.quant_loss_weight is not None
        and not self.do_eval
    ):
      q_deq_x = self.dequantize(q_x, q_s, contract_dims, zp_time_scale)
      quant_loss = (
          jnp.sum(jnp.square(jnp.subtract(x, q_deq_x))) * self.quant_loss_weight
      )

      self.add_summary('quant_loss', quant_loss)
      self.add_aux_loss('quant_loss', quant_loss)

    if squeeze_scale:
      q_s = jnp.squeeze(q_s)

    if quantized_dtype is not None:
      q_x = q_x.astype(quantized_dtype)

    return q_x, q_s, zp_time_scale
