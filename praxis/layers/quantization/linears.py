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
QuantizationParams = quantization_hparams.QuantizationParams
WeightHParams = base_layer.WeightHParams
instance_field = base_layer.instance_field
JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor
WeightInit = base_layer.WeightInit


class Linear(linears.Linear, quantizer.QuantizationLayer):  # pytype: disable=signature-mismatch
  """Quantized and low-rank Linear layer without bias.

  Attributes:
    quantization: Information related to the quantization applied to this layer,
      such as the mode for the quantization.
    rank: Rank to factorize to low-weights. Set to -1 to disable low-rank
      factorization.
  """
  _PACK_4BIT_DIM = 0
  rank: int = -1

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
    act_params = self.quantization.act_params if self.quantization else None
    return act_params is not None and act_params.stats_config is not None

  def setup(self) -> None:
    wp = self.weight_split_dims_mapping
    if self.rank > 0:
      shape_a, shape_b = (
          [self.input_dims, self.rank],
          [self.rank, self.output_dims],
      )
      self.set_up_weights(
          weight_name='w_a',
          weight_params=WeightHParams(
              shape=shape_a,
              mesh_shape=self.mesh_shape,
              tensor_split_dims_mapping=wp.wt,
          ),
          scale_shape=[self.rank],
          pack_dim=self._PACK_4BIT_DIM,
      )
      self.set_up_weights(
          weight_name='w_b',
          weight_params=WeightHParams(
              shape=shape_b,
              mesh_shape=self.mesh_shape,
              tensor_split_dims_mapping=wp.wt,
          ),
          scale_shape=[self.output_dims],
          pack_dim=self._PACK_4BIT_DIM,
      )
    else:
      pc = WeightHParams(
          shape=[self.input_dims, self.output_dims],
          mesh_shape=self.mesh_shape,
          tensor_split_dims_mapping=wp.wt,
      )
      self.set_up_weights(
          weight_name='w',
          weight_params=pc,
          scale_shape=[self.output_dims],
          pack_dim=self._PACK_4BIT_DIM,
      )

  def __call__(self, inputs: JTensor) -> JTensor:
    """Apply projection to inputs.

    Args:
      inputs: The inputs JTensor.  Shaped [..., input_dims].

    Returns:
      Projected inputs.
    """

    ap = self.activation_split_dims_mapping
    eqn = '...y,yz->...z'

    if self.rank > 0:
      intermediate = self.quantized_einsum(
          eqn=eqn,
          x=inputs,
          w=self.theta.w_a,
          pack_dim=self._PACK_4BIT_DIM,
          reshape=[],
          weight_name='w_a',
      )
      out = self.quantized_einsum(
          eqn=eqn,
          x=intermediate,
          w=self.theta.w_b,
          pack_dim=self._PACK_4BIT_DIM,
          reshape=[],
          weight_name='w_b',
      )
    else:
      out = self.quantized_einsum(
          eqn=eqn,
          x=inputs,
          w=self.theta.w,
          pack_dim=self._PACK_4BIT_DIM,
          reshape=[],
      )
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
    assert self.quantization is not None, (
        'quantized_partition_specs is called during serving for quantized'
        ' model, please set quantized config for the model.'
    )
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
    assert self.quantization is not None, (
        'quantize_weight is called during serving for quantized model, please'
        ' set quantized config for the model.'
    )
    theta = self.theta
    scale_name = 'w' + base_layer.QUANTIZED_SCALE_NAME_POSTFIX
    eqn = 'xy,yz->xz'
    if self.quantization.quantization_type in [
        QuantizationType.PTQ,
        QuantizationType.FQ,
        QuantizationType.FQ_VN,
    ]:
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
    elif self.quantization.quantization_type == QuantizationType.AQT:
      if self._do_static_activation_quantization():
        raise NotImplementedError(
            'Static activation quantization is not supported yet.'
        )
      else:
        q_w, q_s, zp = self.weight_quantizer.quantize(
            self.theta.w,
            [0],
            squeeze_scale=True,
            quantized_dtype=self.quantization.weight_params.dtype,
        )

    if (
        self.quantization.weight_params.precision == 4
        and self.quantization.weight_params.use_int4_packed_weights
    ):
      q_w = utils.pack_4bit(
          q_w,
          self._PACK_4BIT_DIM,
          self.quantization.weight_params.int4_packed_weights_container_dtype,
      )

    if self.quantization.weight_params.use_symmetric:
      return {base_layer.PARAMS: {'w': q_w, scale_name: q_s}}
    else:
      zp_name = 'w' + base_layer.QUANTIZED_ZP_NAME_POSTFIX
      return {base_layer.PARAMS: {'w': q_w, scale_name: q_s, zp_name: zp}}
