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

"""Quantized Multi-Query Attention layers."""

import copy
from typing import Any

from jax import numpy as jnp
from praxis import base_layer
from praxis import pax_fiddle
from praxis import pytypes
from praxis.layers import multi_query_attention
from praxis.layers.quantization import operations
from praxis.layers.quantization import quantization_hparams
from praxis.layers.quantization import quantizer
from praxis.layers.quantization import utils

WeightInit = base_layer.WeightInit
QuantizationMode = quantization_hparams.QuantizationMode
QuantizationType = quantization_hparams.QuantizationType
QuantizationParams = quantization_hparams.QuantizationParams
WeightHParams = base_layer.WeightHParams
JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor


class OneHeadedAttentionProjection(  # pytype: disable=signature-mismatch
    multi_query_attention.OneHeadedAttentionProjection,
    quantizer.QuantizationLayer,
):
  """Quantized OneHeadedAttentionProjection.

  Attributes:
    quantization: Information related to the quantization applied to this layer,
      such as the mode for the quantization.
  """

  quantization: QuantizationParams = pax_fiddle.instance_field(
      QuantizationParams
  )
  _PACK_4BIT_DIM = 0

  def create_tensor_quantizers(self):
    weight_params = (
        self.quantization.weight_params if self.quantization else None
    )
    act_params = self.quantization.act_params if self.quantization else None
    self.create_child(
        'act_quantizer',
        quantizer.create_tensor_quantizer('act_quantizer', act_params),
    )
    self.create_child(
        'weight_quantizer',
        quantizer.create_tensor_quantizer('weight_quantizer', weight_params),
    )

  def _do_static_activation_quantization(self) -> bool:
    """If activation need to be quantized."""
    act_params = self.quantization.act_params if self.quantization else None
    return act_params is not None and act_params.stats_config is not None

  def setup(self) -> None:
    wp = self.weight_split_dims_mapping
    if self.mesh_shape is not None:
      assert wp.wt is not None, ('Must provide sharding annotations for the '
                                 'weights if mesh shape is provided')
    wt = wp.wt
    pc_shape = [self.input_dim, self.output_dim]
    pc = WeightHParams(
        shape=pc_shape, mesh_shape=self.mesh_shape, tensor_split_dims_mapping=wt
    )
    self.set_up_weights(
        weight_name='w',
        weight_params=pc,
        scale_shape=[self.output_dim],
    )
    if self.use_bias:
      if self.mesh_shape is not None:
        bias_split_dims_mapping = [wp.wt[1]]
      else:
        bias_split_dims_mapping = None
      pc_bias = WeightHParams(
          shape=[self.output_dim],
          init=WeightInit.Constant(0.0),
          mesh_shape=self.mesh_shape,
          tensor_split_dims_mapping=bias_split_dims_mapping,
      )
      self.create_variable('b', pc_bias)

  def __call__(self, inputs: JTensor) -> JTensor:
    """Computes the multi headed projection for inputs.

    Args:
      inputs: A JTensor of shape [..., p.input_dim].

    Returns:
      The projected JTensor with shape [..., p.output_dim].
    """
    theta = self.theta

    shape = inputs.shape
    inputs = self._cast_to_fprop_dtype(inputs)

    assert (
        shape[-1] == self.input_dim
    ), f'Expecting shape[-1] == p.input_dim, {shape[-1]} != {self.input_dim}'
    eqn = '...D,DH->...H'
    ret = self.quantized_einsum(
        eqn=eqn,
        x=inputs,
        w=self.theta.w,
        reshape=[],
    )
    if self.use_bias:
      ret += theta.b
    return ret

  def quantized_partition_specs(self) -> Any:
    """Get quantized PartitionSpec.

    Returns:
      a map from names to partition spec.
    """
    assert self.quantization is not None, (
        'quantized_partition_specs is called during serving for quantized'
        ' model, please set quantized config for the model.'
    )
    scale_name = 'w' + base_layer.QUANTIZED_SCALE_NAME_POSTFIX
    weight_pspec = base_layer._weight_hparam_to_pspec(
        self._weight_hparams['w'], self.mesh_axis_names
    )
    wp = self.weight_split_dims_mapping
    scale_split_dims_mapping = [wp.wt[1]]
    # scale_weight_hparam is unmaterialized so shape is irrelevant.
    scale_weight_hparam = WeightHParams(
        shape=(), tensor_split_dims_mapping=scale_split_dims_mapping
    )
    scale_pspec = base_layer._weight_hparam_to_pspec(
        scale_weight_hparam, self.mesh_axis_names
    )
    partitionspec = {'w': weight_pspec, scale_name: scale_pspec}

    if not self.quantization.weight_params.use_symmetric:
      zp_name = 'w' + base_layer.QUANTIZED_ZP_NAME_POSTFIX
      partitionspec[zp_name] = copy.deepcopy(scale_pspec)

    # Activation variable partitioning is only needed for static quantization.
    if self._do_static_activation_quantization():
      raise NotImplementedError(
          'Static activation quantization is not supported yet.'
      )

    return {base_layer.PARAMS: partitionspec}

  def quantize_weight(self) -> NestedJTensor:
    """Get quantized weight.

    Returns:
      a map from names to quantized weights.
    """
    assert self.quantization is not None, (
        'quantize_weight is called during serving for quantized model, please'
        ' set quantized config for the model.'
    )
    theta = self.theta
    scale_name = 'w' + base_layer.QUANTIZED_SCALE_NAME_POSTFIX
    eqn = 'xy,yz->xz'
    if self.quantization.quantization_type == QuantizationType.PTQ:
      if self._do_static_activation_quantization():
        raise NotImplementedError(
            'Static activation quantization is not supported yet.'
        )
      else:
        q_w, q_s, zp = operations.reduce_einsum_weight_precision(
            eqn,
            theta.w,
            calculation_dtype=self.dtype,
            bits=self.quantization.weight_params.precision,
            percentile=self.quantization.weight_params.clipping_coeff,
            use_symmetric=self.quantization.weight_params.use_symmetric,
            quant_method=self.quantization.weight_params.quant_method,
        )
    elif self.quantization.quantization_type == QuantizationType.AQT:
      dimension_numbers, _ = utils.einsum_eqn_to_dimension_numbers(eqn)
      weight_contract_dims = dimension_numbers[0][1]
      q_w, q_s, zp = self.weight_quantizer.quantize(
          theta.w,
          weight_contract_dims,
          quantized_dtype=self.quantization.weight_params.dtype,
      )
    else:
      raise ValueError(
          'Unsupported quantization_type'
          f' {self.quantization.quantization_type} in quantized'
          ' multi_query_attention.'
      )

    if (
        self.quantization.weight_params.precision == 4
        and self.quantization.weight_params.use_int4_packed_weights
    ):
      q_w = utils.pack_4bit(q_w, self._PACK_4BIT_DIM)

    if self.quantization.weight_params.use_symmetric:
      return {base_layer.PARAMS: {'w': q_w, scale_name: q_s}}  # pytype: disable=bad-return-type  # jax-ndarray
    else:
      zp_name = 'w' + base_layer.QUANTIZED_ZP_NAME_POSTFIX
      return {base_layer.PARAMS: {'w': q_w, scale_name: q_s, zp_name: zp}}  # pytype: disable=bad-return-type  # jax-ndarray
