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

import itertools
from typing import Sequence

from absl import logging
import jax
import jax.numpy as jnp
from praxis import base_layer
from praxis import pax_fiddle
from praxis import pytypes
from praxis.layers.quantization import operations
from praxis.layers.quantization import quantization_hparams
from praxis.layers.quantization import utils


WeightHParams = base_layer.WeightHParams
JTensor = pytypes.JTensor
ActQuantizationParams = quantization_hparams.ActQuantizationParams
QuantizationMode = quantization_hparams.QuantizationMode
QuantizationType = quantization_hparams.QuantizationType
QuantizationParams = quantization_hparams.QuantizationParams
instance_field = base_layer.instance_field
WeightQuantizationParams = quantization_hparams.WeightQuantizationParams


class QuantizationLayer(base_layer.BaseLayer):
  """QuantizationLayer which quantizes weights and activations.

  QuantizationLayer sets up weights and optionally quantizes weights and
  activations, this layer acts as non-quantized layer if quantization param is
  set to None

  Attributes:
    quantization: QuantizationParams to specify quantization hyper params for a
      given layer. Defaults to default QuantizationParams.
  """

  quantization: QuantizationParams | None = instance_field(QuantizationParams)

  def set_up_weights(
      self,
      *,
      weight_name: str,
      weight_params: base_layer.WeightHParams,
      scale_shape: list[int],
      pack_dim: int,
  ):
    """Set up weights, quantizer, steps."""
    if not self.quantization:
      self.create_variable(weight_name, weight_params)
      return

    dtype = self.quantization.weight_params.dtype
    if self.quantization.mode == QuantizationMode.INFERENCE:
      if (
          self.quantization.weight_params.precision == 4
          and self.quantization.weight_params.use_int4_packed_weights
      ):
        # For 4bit pack/unpack.
        # TODO(jianlijianli): Replace this with proper 4bit type.
        # Type to store int4 values, int32 or int8 are supported.
        dtype = (
            self.quantization.weight_params.int4_packed_weights_container_dtype
        )
        packing_factor = 2 if dtype == jnp.int8 else 8
        weight_params.shape = utils.get_packed_shape(
            weight_params.shape, pack_dim, packing_factor=packing_factor
        )
      if do_static_activation_quantization(self.quantization.act_params):
        raise NotImplementedError(
            'Static activation quantization is not supported yet.'
        )
      if (
          jax.dtypes.scalar_type_of(dtype) == float
          and jnp.finfo(dtype).bits == 8
      ):
        dtype = jnp.int8
      self.create_quantized_variable(
          weight_name,
          weight_params,
          scale_shape,
          dtype=dtype,
          use_symmetric=self.quantization.weight_params.use_symmetric,
      )
    else:
      self.create_variable(weight_name, weight_params)

    if self.quantization.quantization_type == QuantizationType.AQT:
      self.create_tensor_quantizers()
      # Optionally create stateful quantizer.

    if self.quantization.mode == QuantizationMode.TRAINING:
      if do_static_activation_quantization(self.quantization.act_params):
        raise NotImplementedError(
            'Static activation quantization is not supported yet.'
        )

    # Optionally create step count.
    if self.quantization.weight_params.use_step_count:
      step_count_pc = base_layer.WeightHParams(
          shape=[],
          init=base_layer.WeightInit.Constant(0),
          dtype=jnp.int32,
      )
      self.create_variable('step_count', step_count_pc, trainable=False)

  def quantized_einsum(
      self,
      *,
      eqn: str,
      x: JTensor,
      w: JTensor,
      pack_dim: int,
      reshape: list[int],
      weight_name: str = 'w',
  ) -> JTensor:
    """Quantized Einsum for inference and training."""

    if not self.quantization:
      if reshape:
        w = jnp.reshape(w, reshape)
      return jnp.einsum(eqn, x, w)

    # Optionally create step count.
    step_count = None
    if self.quantization.weight_params.use_step_count:
      step_count = self.get_var('step_count')
      if not self.do_eval:
        self.update_var('step_count', step_count + 1)
      self.add_summary('step_count', step_count)

    if self.quantization.mode == QuantizationMode.INFERENCE:
      # PTQ, QAT has the same inference graph, only difference is on activation.
      # No matter which quantization type is used, the weight and scale
      # dimensions are the same for all types.
      # Note: lower-bit types are not reflected during inference for now due to
      # b/259306620.
      w, s, zp = self.get_quantized_weight(
          weight_name,
          use_symmetric=self.quantization.weight_params.use_symmetric,
      )
      scale_act = None
      if (
          self.quantization.weight_params.precision == 4
          and self.quantization.weight_params.use_int4_packed_weights
      ):
        w = utils.unpack_4bit(
            w, pack_dim, self.quantization.weight_params.dtype
        ).astype(jnp.int8)
      if do_static_activation_quantization(self.quantization.act_params):
        # This is for benchmarking only, to get the headroom.
        # TODO(jianlijianli): implement this properly.
        x = x.astype(jnp.int8)
        logging.info('Static activation quantization is not supported yet.')
      elif self.quantization.act_params is not None:
        act_params = self.quantization.act_params
        x, scale_act = operations.reduce_einsum_activation_precision(
            eqn,
            x,
            bits=act_params.precision,
            per_channel=act_params.per_channel,
        )
      dtype = self.quantization.weight_params.dtype
      if (
          jax.dtypes.scalar_type_of(dtype) == float
          and jnp.finfo(dtype).bits == 8
      ):
        w = jax.lax.bitcast_convert_type(w, dtype)
        # cast to bf16 since bf16 x fp8 is not supported.
        w = w.astype(jnp.bfloat16)
      out = operations.einsum(eqn, x, w, s, zp=zp, scale_act=scale_act)

      return out
    else:
      if reshape:
        w = jnp.reshape(w, reshape)

      if self.quantization.quantization_type == QuantizationType.AQT:
        out = operations.aqt_einsum(
            eqn,
            x,
            w,
            lhs_quantizer=self.act_quantizer,
            rhs_quantizer=self.weight_quantizer,
        )
        return out
      elif self.quantization.quantization_type == QuantizationType.FQ:
        if self.quantization.act_params is not None:
          act_params = self.quantization.act_params
          x = operations.fakequant_activation(x, bits=act_params.precision)
        w = operations.fakequant_einsum(
            eqn,
            w,
            bits=self.quantization.weight_params.precision,
            use_symmetric=self.quantization.weight_params.use_symmetric,
            calculation_type=self.quantization.weight_params.calculation_dtype,
            block_size=self.quantization.weight_params.block_size,
        )
        out = jnp.einsum(eqn, x, w)
        return out
      elif self.quantization.quantization_type == QuantizationType.FQ_VN:
        w = operations.fakequant_vn(
            eqn,
            w,
            self.next_prng_key(),
            self.quantization.weight_params,
            step_count,
            self.do_eval,
            bits=self.quantization.weight_params.precision,
            use_symmetric=self.quantization.weight_params.use_symmetric,
        )
        out = jnp.einsum(eqn, x, w)
        return out
      # Fall back to regular einsum.
      return jnp.einsum(eqn, x, w)


def quantized_conv(
    layer: base_layer.BaseLayer,
    x: JTensor,
    padding: str,
    dimension_numbers: tuple[str, ...],
    feature_group_count: int,
    pack_dim: int,
) -> JTensor:
  """Quantized Conv for inference and training.

  Static activation quantization is to be added.

  Args:
    layer: the layer that calls this function.
    x: input tensor.
    padding: the padding dims on input.
    dimension_numbers: the dimension numbers
    feature_group_count: feature groups
    pack_dim: pack dimension for int4 packing.

  Returns:
    the convolution output.
  """
  if not layer.quantization.weight_params.use_symmetric:
    raise ValueError('Conv supports only symmetric weight quantization.')
  if layer.quantization.mode == QuantizationMode.INFERENCE:
    w, s, _ = layer.get_quantized_weight('w')
    if (
        layer.quantization.weight_params.precision == 4
        and layer.quantization.weight_params.use_int4_packed_weights
    ):
      w = utils.unpack_4bit(
          w, pack_dim, layer.quantization.weight_params.dtype
      )
    if do_static_activation_quantization(layer.quantization.act_params):
      raise NotImplementedError(
          'Static activation quantization is not supported yet.'
      )
    elif layer.quantization.act_params is not None:
      x, act_scale = operations.reduce_precision_activation(x)
      s = jnp.multiply(jnp.squeeze(act_scale), s)
    dtype = layer.quantization.weight_params.dtype
    if (
        jax.dtypes.scalar_type_of(dtype) == float
        and jnp.finfo(dtype).bits == 8
    ):
      w = jax.lax.bitcast_convert_type(w, dtype)
      # cast to bf16 since bf16 x fp8 is not supported.
      w = w.astype(jnp.bfloat16)

    # lax.conv_general_dilated needs same type on x and w.
    w = w.astype(x.dtype)
    outputs = jax.lax.conv_general_dilated(
        lhs=x,
        rhs=w,
        window_strides=layer.filter_stride,
        padding=padding,
        rhs_dilation=layer.dilations,
        dimension_numbers=dimension_numbers,
        feature_group_count=feature_group_count,
    )
    outputs = jnp.multiply(outputs, s)
    return outputs
  else:
    raise NotImplementedError('QAT not supported for conv.')


def do_static_activation_quantization(act_params) -> bool:
  return act_params is not None and act_params.stats_config is not None


def create_tensor_quantizer(
    name: str,
    quant_params: ActQuantizationParams | WeightQuantizationParams | None,
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
    tq_params.kurt_loss_weight = quant_params.kurt_loss_weight
    tq_params.kurt = quant_params.kurt
    tq_params.optimize_clipping_per_channel = (
        quant_params.optimize_clipping_per_channel
    )
    tq_params.clipping_coeff = (
        None
        if quant_params.clipping_coeff == 1.0
        else quant_params.clipping_coeff
    )
    tq_params.sub_channels = quant_params.sub_channels

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
    clipping_coeff: The coefficient to shrink the hard range for weight
      quantization. None means using hard min/max.
    add_scale_eps: If True add epsilon to scale to avoid division by zero,
      else it will replace zero scale by 1.
    unsigned_int_bounds: If True, use [0, 2**precision-1] clip bound for better
      quantization bucket utilization in case where the input has a positive
      distribution (e.g., ReLU).
    use_symmetric: Do symmetric quantization for weights.
    quant_loss_weight: Weight for quantization loss.
    kurt_loss_weight: Weight for Kurtosis loss.
    kurt: Kurtosis target. By default it is 1.8 (uniform distribution).
      It is based on paper: "Robust Quantization: One Model to Rule Them All".
    optimize_clipping_per_channel: If True choose the best clipping value
      per channel, else per-tensor.
    sub_channels: Number of sub channels for splitting channelwise quantization.
  """
  precision: int | None = None
  stop_scale_gradient: bool = False
  min_clipping: float | None = None
  num_optimize_clipping: int | None = None
  clipping_coeff: float | None = None
  add_scale_eps: bool | None = True
  unsigned_int_bounds: bool = False
  use_symmetric: bool = True
  quant_loss_weight: float | None = None
  kurt_loss_weight: float | None = None
  kurt: float = 1.8
  optimize_clipping_per_channel: bool = False
  sub_channels: int | None = None

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

      if self.clipping_coeff is not None:
        raise ValueError(
            'clipping_coeff can not be used together with min_clipping '
            'and num_optimize_clipping'
        )
    elif self.optimize_clipping_per_channel:
      raise ValueError(
          'optimize_clipping_per_channel can not be True if '
          'min_clipping is None and num_optimize_clipping is None.'
      )

    if self.clipping_coeff is not None and (
        self.clipping_coeff < 0 or self.clipping_coeff > 1
    ):
      raise ValueError('clipping_coeff has to be in range 0...1')

  def __call__(self):
    # Since TensorQuantizer does nothing when initialized, __call__ is a no-op.
    pass

  def _get_scale_and_min(
      self,
      x: JTensor,
      contract_dims: int | Sequence[int],
      clipping_coeff: float | None = None,
  ) -> tuple[JTensor, JTensor | None]:
    if self.precision is None:
      return jnp.ones(shape=(1,) * x.ndim, dtype=x.dtype), None

    clip_bound_min, clip_bound_max = operations.get_min_max(
        self.precision, self.unsigned_int_bounds
    )

    if self.use_symmetric:
      x_bound = jnp.max(jnp.abs(x), axis=contract_dims, keepdims=True)
      x_min = None
      range_bound = clip_bound_max
    else:
      x_max = jnp.max(x, axis=contract_dims, keepdims=True)
      x_min = jnp.min(x, axis=contract_dims, keepdims=True)
      x_bound = x_max - x_min
      range_bound = clip_bound_max - clip_bound_min

    if clipping_coeff is not None:
      x_bound *= clipping_coeff
      if x_min is not None:
        x_min *= clipping_coeff

    scale = x_bound / range_bound
    if self.stop_scale_gradient:
      scale = jax.lax.stop_gradient(scale)

    if self.add_scale_eps:
      # Add epsilon to avoid NaN gradients for near-zero inputs during training.
      scale = scale + jnp.finfo(x.dtype).eps
    else:
      scale = jnp.where(scale == 0, jnp.ones_like(scale), scale)

    return scale, x_min

  def _get_optimal_scale_and_min(
      self,
      x: JTensor,
      contract_dims: int | Sequence[int],
  ) -> tuple[JTensor, JTensor | None]:
    def quantization_error_and_scale(clipping):
      q_scale, x_min = self._get_scale_and_min(
          x, contract_dims, clipping_coeff=clipping
      )
      x_scaled, zp_time_scale = self._scale(x, q_scale, x_min)
      x_quantized = self.to_quant(x_scaled)
      x_quantized_dequantized = self.dequantize(
          x_quantized, q_scale, contract_dims, zp_time_scale
      )
      quant_error = jnp.abs(jnp.subtract(x, x_quantized_dequantized))

      if self.optimize_clipping_per_channel:
        sum_error = jnp.sum(quant_error, axis=contract_dims, keepdims=True)
      else:
        sum_error = jnp.sum(quant_error)

      return sum_error, q_scale, x_min

    clipping = jnp.linspace(
        1.0, self.min_clipping, num=self.num_optimize_clipping, dtype=x.dtype
    )
    sum_errors, q_scales, x_mins = jax.vmap(
        quantization_error_and_scale
    )(clipping)

    reduced_shape = list(x.shape)
    if isinstance(contract_dims, int):
      reduced_shape[contract_dims] = 1
    elif contract_dims is not None:
      for i in contract_dims:
        reduced_shape[i] = 1
    else:
      reduced_shape = ()

    if not self.optimize_clipping_per_channel:
      best_ind = jnp.array([jnp.argmin(sum_errors)], dtype=jnp.int32)
    else:
      # clip_index is a flat tensor containing the indcies to the first
      # dimension in scales which provide the lowest quantization errors.
      clip_index = jnp.reshape(jnp.argmin(sum_errors, axis=0), [-1])
      # scale_index iteratively indexes the trailing dimensions in scales
      # so we know which scalars to pull out (in tf using tf.gather_nd).
      scale_index = itertools.product(*[range(d) for d in reduced_shape])
      scale_index = jnp.array(list(scale_index), dtype=jnp.int32)
      # optimal_index combines these two indices to specify which clipping value
      # to use for each feature in inputs.
      best_ind = jnp.concatenate([clip_index[:, None], scale_index], axis=1)

    # Returns a flat tensor with all of the optimal scale value
    # for each channel. It is unbatched version of tf.gather_nd:
    best_ind = tuple(jnp.moveaxis(best_ind, -1, 0))
    best_scale = q_scales[best_ind]

    if self.optimize_clipping_per_channel:
      best_scale = jnp.reshape(best_scale, reduced_shape)

    if self.use_symmetric:
      best_x_min = None
    else:
      best_x_min = x_mins[best_ind]
      if self.optimize_clipping_per_channel:
        best_x_min = jnp.reshape(best_x_min, reduced_shape)

    return best_scale, best_x_min

  def get_quant_scale(
      self,
      x: JTensor,
      contract_dims: int | Sequence[int],
  ) -> tuple[JTensor, JTensor | None]:
    """Computes scale for quantization.

    It can compute standard scale or scale optimized over different
    clipping values in range [min_clipping ... 1.0].

    Args:
      x: Input tensor.
      contract_dims: Axis along which to quantize acts (the non-feature axis).

    Returns:
      Scale and min value of a tensor x.
      If clipping is not None then it is applied on both scale and min value.
    """
    if self.min_clipping is not None and self.num_optimize_clipping is not None:
      return self._get_optimal_scale_and_min(x, contract_dims)
    else:
      return self._get_scale_and_min(
          x, contract_dims, clipping_coeff=self.clipping_coeff
      )

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

    x = operations.pass_through(x + 0.5, jnp.floor)
    clip_bound_min, clip_bound_max = operations.get_min_max(
        self.precision, self.unsigned_int_bounds
    )
    x = jnp.clip(x, clip_bound_min, clip_bound_max)

    return x

  def _get_zero_point(self, x, x_min, scale) -> JTensor:
    clip_bound_min, _ = operations.get_min_max(
        self.precision, self.unsigned_int_bounds
    )
    zp = clip_bound_min - x_min / scale
    return zp

  def dequantize(
      self,
      q_x: JTensor,
      q_scale: JTensor,
      contract_dims: int | Sequence[int],
      zp_time_scale: JTensor | None = None,
  ) -> JTensor:
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
      deq_q_x = q_x * q_scale - zp_time_scale
    return deq_q_x

  def _scale(
      self,
      x: JTensor,
      q_scale: JTensor,
      x_min: JTensor | None = None,
  ) -> tuple[JTensor, JTensor | None]:
    """Rescales input x for quantization.

    Args:
      x: Input tensor.
      q_scale: Quantization scale.
      x_min: Per channel minimum values of x. It is required only
        for asymmetric quantization.

    Returns:
      Rescaled tensor.
    """

    if self.use_symmetric:
      if x_min is not None:
        raise ValueError('x_min has to be None for symmetric quantization.')
      x_scaled = jnp.divide(x, q_scale)
      zp_time_scale = None
    else:
      if x_min is None:
        raise ValueError('x_min is required for asymmetric quantization.')
      zp = self._get_zero_point(x, x_min, q_scale)
      x_scaled = jnp.divide(x, q_scale) + zp
      zp_time_scale = jnp.multiply(q_scale, zp)

    return x_scaled, zp_time_scale

  def quantize(
      self,
      x: JTensor,
      contract_dims: int | Sequence[int],
      squeeze_scale=True,
      quantized_dtype: jnp.dtype | None = None,
  ) -> tuple[JTensor, JTensor, JTensor | None]:
    """Quantizes input x.

    Args:
      x: Input tensor.
      contract_dims: Contraction dims.
      squeeze_scale: If True it will squeeze output scale.
      quantized_dtype: Output type.

    Returns:
      Quantized tensor with scale (used for dequantization).
    """

    q_s, x_min = self.get_quant_scale(x, contract_dims)
    x_scaled, zp_time_scale = self._scale(x, q_s, x_min)
    q_x = self.to_quant(x_scaled)

    if (
        quantized_dtype != jnp.int8  # it is used for materialization
        and self.quant_loss_weight is not None
        and not self.do_eval
    ):
      q_deq_x = self.dequantize(q_x, q_s, contract_dims, zp_time_scale)
      quant_loss = (
          jnp.mean(jnp.square(jnp.subtract(x, q_deq_x)))
          * self.quant_loss_weight
      )

      self.add_summary('quant_loss', quant_loss)
      self.add_aux_loss('quant_loss', quant_loss)

    if (
        quantized_dtype != jnp.int8  # it is used for materialization
        and self.kurt_loss_weight is not None
        and not self.do_eval
    ):
      # Per-channel mean.
      x_mean = jnp.mean(x, axis=contract_dims, keepdims=True)

      # Per channel std.
      x_std = jnp.std(x, axis=contract_dims, keepdims=True)
      x_std = jnp.where(x_std == 0, jnp.ones_like(x_std), x_std)

      # Per channel Kurtosis is the fourth standardized moment:
      x_norm4 = jnp.mean(
          jnp.power((x - x_mean) / x_std, 4.0), axis=contract_dims
      )

      # Kurtosis loss.
      kurt_loss = (
          jnp.mean(jnp.square(x_norm4 - self.kurt)) * self.kurt_loss_weight
      )

      self.add_summary('kurt_loss', kurt_loss)
      self.add_aux_loss('kurt_loss', kurt_loss)

    if squeeze_scale:
      q_s = jnp.squeeze(q_s)
      if zp_time_scale is not None:
        zp_time_scale = jnp.squeeze(zp_time_scale)

    if quantized_dtype is not None:
      q_x = q_x.astype(quantized_dtype)

    return q_x, q_s, zp_time_scale
