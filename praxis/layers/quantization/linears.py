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

"""Quantized Linear Layers."""

from typing import Any

from jax import numpy as jnp
from praxis import base_layer
from praxis import pytypes
from praxis.layers import linears
from praxis.layers.quantization import aqt
from praxis.layers.quantization import operations
from praxis.layers.quantization import quantization_hparams

QuantizationMode = quantization_hparams.QuantizationMode
QuantizationType = quantization_hparams.QuantizationType
QuantizationHParams = quantization_hparams.QuantizationHParams
WeightHParams = base_layer.WeightHParams
sub_config_field = base_layer.sub_config_field
JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor


class Linear(linears.Linear):
  """Quantized Linear layer without bias.

  Attributes:
    quantization: Information related to the quantization applied to this layer,
      such as the mode for the quantization.
  """
  quantization: QuantizationHParams = sub_config_field(QuantizationHParams)

  def create_tensor_quantizers(self):
    self.create_child(
        'act_quantizer',
        aqt.create_tensor_quantizer(
            'aqt_quantizer', self.quantization.act_params
        ),
    )
    self.create_child(
        'weight_quantizer',
        aqt.create_tensor_quantizer(
            'weight_quantizer', self.quantization.weight_params
        ),
    )

  def _do_static_activation_quantization(self) -> bool:
    act_params = self.quantization.act_params
    return act_params is not None and act_params.stats_config is not None

  def setup(self) -> None:
    wp = self.weight_split_dims_mapping
    pc = WeightHParams(
        shape=[self.input_dims, self.output_dims],
        mesh_shape=self.mesh_shape,
        tensor_split_dims_mapping=wp.wt,
    )
    if self.quantization.mode == QuantizationMode.INFERENCE:
      if self._do_static_activation_quantization():
        raise NotImplementedError(
            'Static activation quantization is not supported yet.')
        # Additionally add activation scale.
      self.create_quantized_variable('w', pc, [self.output_dims])
    elif self.quantization.mode == QuantizationMode.TRAINING:
      # TODO(jihwanlee): Now, having many different branches and non-unified
      # quantization logic between PTQ, FQ, and AQT, the overall code is quite
      # complex. DO simplify.
      if self._do_static_activation_quantization():
        raise NotImplementedError(
            'Static activation quantization is not supported yet.')
        # Additionally add mutable tensor to record activation range.
      self.create_variable('w', pc)
    else:
      self.create_variable('w', pc)

    if self.quantization.quantization_type == QuantizationType.AQT:
      self.create_tensor_quantizers()

  def __call__(self, inputs: JTensor) -> JTensor:
    """Apply projection to inputs.

    Args:
      inputs: The inputs JTensor.  Shaped [..., input_dims].

    Returns:
      Projected inputs.
    """
    ap = self.activation_split_dims_mapping
    eqn = '...y,yz->...z'
    if self.quantization.mode == QuantizationMode.INFERENCE:
      # PTQ, QAT has the same inference graph, only difference is on activation.
      # No matter which quantization type is used, the weight and scale
      # dimensions are the same for all types.
      # Note: lower-bit types are not reflected during inference for now due to
      # b/259306620.
      w, s = self.get_quantized_weight('w')
      if self._do_static_activation_quantization():
        raise NotImplementedError(
            'Static activation quantization is not supported yet.'
        )
      elif self.quantization.act_params is not None:
        inputs, act_scale = operations.reduce_precision_activation(inputs)
        s = jnp.multiply(jnp.squeeze(act_scale), s)
      out = operations.einsum(eqn, inputs, w, s)
    else:
      w = self.theta.w
      if self.quantization.quantization_type == QuantizationType.AQT:
        dimension_numbers = (((len(inputs.shape) - 1,), (0,)), ((), ()))
        out = operations.dot_general(
            lhs=inputs,
            rhs=w,
            lhs_quantizer=self.act_quantizer,
            rhs_quantizer=self.weight_quantizer,
            dimension_numbers=dimension_numbers,
            is_eval=self.do_eval)
      elif self.quantization.quantization_type == QuantizationType.FQ:
        bits = self.quantization.weight_params.precision
        w = operations.fakequant_einsum(
            eqn, w, bits=bits, calculation_type=self.dtype
        )
        out = linears.project_last_dim(inputs, w)
      else:
        out = linears.project_last_dim(inputs, w)
    # Adjust sharding annotation during decoding.
    # TODO(pax): This logic should likely be lifted somewhere else.
    ap_out = ap.out
    if ap_out is not None and len(ap_out) == 3 and out.ndim == 2:
      ap_out = [ap_out[0], ap_out[2]]
    out = base_layer.maybe_shard(out, ap_out, self.mesh_axis_names)
    return out

  def quantized_partition_specs(self) -> Any:
    """Get quantized PartitionSpec.

    Returns:
      a map from names to partition spec.
    """
    scale_name = 'w' + base_layer.QUANTIZED_NAME_POSTFIX
    weight_pspec = base_layer._weight_hparam_to_pspec(
        self._weight_hparams['w'], self.mesh_axis_names
    )
    wp = self.weight_split_dims_mapping
    scale_split_dims_mapping = [wp.wt[1]]
    # scale_weight_hparam is unmaterialized so shape is irrelevant.
    scale_weight_hparam = WeightHParams(
        shape=(), tensor_split_dims_mapping=scale_split_dims_mapping)
    scale_pspec = base_layer._weight_hparam_to_pspec(
        scale_weight_hparam, self.mesh_axis_names
    )
    partitionspec = {'w': weight_pspec, scale_name: scale_pspec}

    # Activation variable partitioning is only needed for static quantization.
    if self._do_static_activation_quantization():
      raise NotImplementedError(
          'Static activation quantization is not supported yet.')

    return {base_layer.PARAMS: partitionspec}

  def quantize_weight(self) -> NestedJTensor:
    """Get quantized weight.

    Returns:
      a map from names to quantized weights.
    """
    theta = self.theta
    scale_name = 'w' + base_layer.QUANTIZED_NAME_POSTFIX
    eqn = 'xy,yz->xz'
    bits = self.quantization.weight_params.precision
    percentile = self.quantization.weight_params.clipping_coeff
    if self.quantization.quantization_type == QuantizationType.PTQ:
      if self._do_static_activation_quantization():
        raise NotImplementedError(
            'Static activation quantization is not supported yet.')
      else:
        q_w, q_s = operations.reduce_einsum_weight_precision(
            eqn,
            theta.w,
            calculation_type=self.dtype,
            bits=bits,
            percentile=percentile,
        )
        return {base_layer.PARAMS: {'w': q_w, scale_name: q_s}}
    elif self.quantization.quantization_type == QuantizationType.FQ:
      if self._do_static_activation_quantization():
        raise NotImplementedError(
            'Static activation quantization is not supported yet.')
      else:
        q_w, q_s = operations.reduce_einsum_weight_precision(
            eqn,
            theta.w,
            calculation_type=self.dtype,
            bits=bits,
            percentile=percentile,
        )
        return {base_layer.PARAMS: {'w': q_w, scale_name: q_s}}
    elif self.quantization.quantization_type == QuantizationType.AQT:
      if self._do_static_activation_quantization():
        raise NotImplementedError(
            'Static activation quantization is not supported yet.')
      else:
        q_s = self.weight_quantizer.get_quant_scale(
            theta.w, contract_dims=[0], dtype=self.dtype)
        q_s = jnp.squeeze(q_s)
        q_w = theta.w / q_s
        q_w = self.weight_quantizer.to_quant(q_w, dtype=jnp.int8)
        return {base_layer.PARAMS: {'w': q_w, scale_name: q_s}}
