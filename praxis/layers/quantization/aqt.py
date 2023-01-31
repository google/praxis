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

"""Quantization Aware Training ops."""

from __future__ import annotations
from typing import Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
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

  if isinstance(quant_params, WeightQuantizationParams):
    tq_params.min_clipping = quant_params.min_clipping
    tq_params.num_optimize_clipping = quant_params.num_optimize_clipping

  return tq_params


class TensorQuantizer(base_layer.BaseLayer):
  """Maintains state associated with the quantization of an input tensor.

  Attributes:
    precision: Number of bits to quantize to (e.g 4 for int4). Must be positive.
    stop_scale_gradient: stop the gradient of the quantization scale for
      numerical stability. Note: this is numerically incorrect.
    min_clipping: Clipping value which will be used for clipping optimization
      in range [min_clipping ... 1].
    num_optimize_clipping: Number of optimization steps used for
      scale estimation with search over clipping values in
      range [min_clipping ... 1].
  """
  precision: Optional[int] = None
  stop_scale_gradient: bool = False
  min_clipping: Optional[float] = None
  num_optimize_clipping: Optional[int] = None

  def setup(self):
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

  def _get_clip_bound(self) -> float:
    bucket_count = 2.0**self.precision
    bucket_count -= 1.0
    return bucket_count / 2.0

  def _safe_clip_bound(self) -> float:
    cb_unsafe = self._get_clip_bound()
    cb = cb_unsafe - 2.0 ** (-20 + self.precision)
    assert cb < cb_unsafe, 'Internal error, epsilon too small.'
    return cb

  def _get_scale(
      self,
      x: JTensor,
      contract_dims: Union[int, Sequence[int]],
      dtype: jnp.dtype = jnp.bfloat16,
      clipping: Optional[float] = None,
  ) -> JTensor:
    if self.precision is None:
      return jnp.ones(shape=(1,) * x.ndim, dtype=dtype)

    x_max = jnp.max(jnp.abs(x), axis=contract_dims, keepdims=True)
    if clipping is not None:
      x_max *= clipping

    clip_bound = self._get_clip_bound()
    scale = x_max / clip_bound
    if self.stop_scale_gradient:
      scale = jax.lax.stop_gradient(scale)
      scale = jnp.where(scale == 0, jnp.ones_like(scale), scale)
    else:
      # Add a small to avoid NaN gradients for near-zero inputs during training.
      scale = scale + jnp.finfo(dtype).eps
    return scale.astype(dtype)

  def _get_optimal_scale(
      self,
      x: JTensor,
      contract_dims: Union[int, Sequence[int]],
      dtype: jnp.dtype = jnp.bfloat16,):

    def quantization_error_and_scale(clipping):
      scale = self._get_scale(x, contract_dims, dtype, clipping=clipping)
      x_quantized = self.to_quant(jnp.divide(x, scale), jnp.int8)
      x_quantized_dequantized = jnp.multiply(scale, x_quantized)
      sum_error = jnp.sum(jnp.abs(jnp.subtract(x, x_quantized_dequantized)))
      return sum_error, scale

    clipping = np.linspace(
        1.0, self.min_clipping, num=self.num_optimize_clipping
    )
    res = jax.vmap(quantization_error_and_scale)(clipping)
    best_ind = jnp.argmin(res[0])
    best_scale = res[1].at[best_ind].get().astype(dtype)
    return best_scale

  def get_quant_scale(
      self,
      x: JTensor,
      contract_dims: Union[int, Sequence[int]],
      dtype: jnp.dtype = jnp.bfloat16,
  ) -> JTensor:
    """Computes scale for quantization.

    It can compute standard scale or scale optimized over different
    clipping values in range [min_clipping ... 1.0].

    Args:
      x: Input tensor.
      contract_dims: Axis along which to quantize acts (the non-feature axis).
      dtype: Output type.

    Returns:
      Scale tensor.
    """
    if self.min_clipping is not None and self.num_optimize_clipping is not None:
      return self._get_optimal_scale(x, contract_dims, dtype)
    else:
      return self._get_scale(x, contract_dims, dtype)

  def update(self, x: JTensor):
    # This function is no-op for now. Once static quantization is supported,
    # statistics update will be performed through this function.
    pass

  def to_quant(self, x: JTensor, dtype=jnp.bfloat16):
    if self.precision is None:
      return x.astype(dtype)

    clip_bound = self._safe_clip_bound()
    x = jnp.clip(x, -clip_bound, clip_bound)
    x = _pass_through(x + 0.5, jnp.floor)

    return x.astype(dtype)
