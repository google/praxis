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

"""Quantized Linear Layers."""

import copy
from typing import Any

from jax import numpy as jnp
from praxis import base_layer
from praxis import pytypes
from praxis.layers import linears
from praxis.layers.quantization import operations
from praxis.layers.quantization import quantization_hparams
from praxis.layers.quantization import quantizer
from praxis.layers.quantization import utils

QuantizationMode = quantization_hparams.QuantizationMode
QuantizationType = quantization_hparams.QuantizationType
QuantizationHParams = quantization_hparams.QuantizationHParams
WeightHParams = base_layer.WeightHParams
sub_config_field = base_layer.sub_config_field
JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor
WeightInit = base_layer.WeightInit


class Linear(linears.Linear):
  """Quantized Linear layer without bias.

  Attributes:
    quantization: Information related to the quantization applied to this layer,
      such as the mode for the quantization.
  """
  quantization: QuantizationHParams = sub_config_field(QuantizationHParams)

  _PACK_4BIT_DIM = 0

  def create_tensor_quantizers(self):
    self.create_child(
        'act_quantizer',
        quantizer.create_tensor_quantizer(
            'quantizer_quantizer', self.quantization.act_params
        ),
    )
    self.create_child(
        'weight_quantizer',
        quantizer.create_tensor_quantizer(
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
    dtype = self.quantization.weight_params.dtype
    if self.quantization.mode == QuantizationMode.INFERENCE:
      if self.quantization.weight_params.precision == 4:
        pc.shape = utils.get_packed_shape(
            pc.shape, self._PACK_4BIT_DIM, packing_factor=8
        )
        pc.shape = [self.input_dims // 8, self.output_dims]
        dtype = jnp.int32
      if self._do_static_activation_quantization():
        raise NotImplementedError(
            'Static activation quantization is not supported yet.'
        )
        # Additionally add activation scale.
      self.create_quantized_variable(
          'w',
          pc,
          [self.output_dims],
          dtype=dtype,
          use_symmetric=self.quantization.weight_params.use_symmetric,
      )
    elif self.quantization.mode == QuantizationMode.TRAINING:
      # TODO(jihwanlee): Now, having many different branches and non-unified
      # quantization logic between PTQ, FQ, and AQT, the overall code is quite
      # complex. DO simplify.
      if self._do_static_activation_quantization():
        raise NotImplementedError(
            'Static activation quantization is not supported yet.'
        )
        # Additionally add mutable tensor to record activation range.
      self.create_variable('w', pc)
    else:
      self.create_variable('w', pc)

    if self.quantization.quantization_type == QuantizationType.AQT:
      self.create_tensor_quantizers()

    if self.quantization.weight_params.use_step_count:
      step_count_pc = WeightHParams(
          shape=[],
          init=WeightInit.Constant(0),
          dtype=jnp.int32,
      )
      self.create_variable('step_count', step_count_pc, trainable=False)

  def __call__(self, inputs: JTensor) -> JTensor:
    """Apply projection to inputs.

    Args:
      inputs: The inputs JTensor.  Shaped [..., input_dims].

    Returns:
      Projected inputs.
    """

    if self.quantization.weight_params.use_step_count:
      step_count = self.get_var('step_count')
      if not self.do_eval:
        self.update_var('step_count', step_count + 1)
      self.add_summary('step_count', step_count)

    ap = self.activation_split_dims_mapping
    eqn = '...y,yz->...z'
    if self.quantization.mode == QuantizationMode.INFERENCE:
      # PTQ, QAT has the same inference graph, only difference is on activation.
      # No matter which quantization type is used, the weight and scale
      # dimensions are the same for all types.
      # Note: lower-bit types are not reflected during inference for now due to
      # b/259306620.
      w, s, zp = self.get_quantized_weight(
          'w', use_symmetric=self.quantization.weight_params.use_symmetric
      )
      if self.quantization.weight_params.precision == 4:
        w = utils.unpack_4bit(
            w, self._PACK_4BIT_DIM, self.quantization.weight_params.dtype
        )
      if self._do_static_activation_quantization():
        raise NotImplementedError(
            'Static activation quantization is not supported yet.'
        )
      elif self.quantization.act_params is not None:
        inputs, act_scale = operations.reduce_precision_activation(inputs)
        s = jnp.multiply(jnp.squeeze(act_scale), s)
      if self.quantization.weight_params.use_symmetric:
        if self.quantization.weight_params.dequant_upfront:
          raise NotImplementedError('Dequantize upfront not supported.')
        else:
          out = operations.einsum(eqn, inputs, w, s)
      else:
        out = operations.einsum(eqn, inputs, w, s, zp)
    else:
      w = self.theta.w
      if self.quantization.quantization_type == QuantizationType.AQT:
        out = operations.aqt_einsum(
            eqn,
            inputs,
            w,
            lhs_quantizer=self.act_quantizer,
            rhs_quantizer=self.weight_quantizer,
        )
      elif self.quantization.quantization_type == QuantizationType.FQ:
        w = operations.fakequant_einsum(
            eqn,
            w,
            bits=self.quantization.weight_params.precision,
            use_symmetric=self.quantization.weight_params.use_symmetric,
            calculation_type=self.quantization.weight_params.calculation_dtype,
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
    scale_name = 'w' + base_layer.QUANTIZED_SCALE_NAME_POSTFIX
    weight_pspec = base_layer._weight_hparam_to_pspec(  # pylint: disable=protected-access
        self._weight_hparams['w'], self.mesh_axis_names
    )
    wp = self.weight_split_dims_mapping
    scale_split_dims_mapping = [wp.wt[1]]
    # scale_weight_hparam is unmaterialized so shape is irrelevant.
    scale_weight_hparam = WeightHParams(
        shape=(), tensor_split_dims_mapping=scale_split_dims_mapping)
    scale_pspec = base_layer._weight_hparam_to_pspec(  # pylint: disable=protected-access
        scale_weight_hparam, self.mesh_axis_names
    )
    partitionspec = {'w': weight_pspec, scale_name: scale_pspec}

    if not self.quantization.weight_params.use_symmetric:
      zp_name = 'w' + base_layer.QUANTIZED_ZP_NAME_POSTFIX
      partitionspec[zp_name] = copy.deepcopy(scale_pspec)

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
    scale_name = 'w' + base_layer.QUANTIZED_SCALE_NAME_POSTFIX
    eqn = 'xy,yz->xz'
    if (
        self.quantization.quantization_type == QuantizationType.PTQ
        or self.quantization.quantization_type == QuantizationType.FQ
    ):
      if self._do_static_activation_quantization():
        raise NotImplementedError(
            'Static activation quantization is not supported yet.'
        )
      else:
        q_w, q_s, zp = operations.reduce_einsum_weight_precision(
            eqn,
            theta.w,
            calculation_type=self.dtype,
            bits=self.quantization.weight_params.precision,
            percentile=self.quantization.weight_params.clipping_coeff,
            use_symmetric=self.quantization.weight_params.use_symmetric,
        )
        if self.quantization.weight_params.precision == 4:
          q_w = utils.pack_4bit(q_w, self._PACK_4BIT_DIM)
    elif self.quantization.quantization_type == QuantizationType.AQT:
      if self._do_static_activation_quantization():
        raise NotImplementedError(
            'Static activation quantization is not supported yet.'
        )
      else:
        q_w, q_s, zp = self.weight_quantizer.quantize(
            self.theta.w,
            [0],
            quantized_dtype=self.quantization.weight_params.dtype,
        )

    if self.quantization.weight_params.use_symmetric:
      return {base_layer.PARAMS: {'w': q_w, scale_name: q_s}}
    else:
      zp_name = 'w' + base_layer.QUANTIZED_ZP_NAME_POSTFIX
      return {base_layer.PARAMS: {'w': q_w, scale_name: q_s, zp_name: zp}}
