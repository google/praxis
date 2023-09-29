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
from typing import Any, Sequence, Tuple

from jax import numpy as jnp
from praxis import base_layer
from praxis import pytypes
from praxis.layers import linears
from praxis.layers.quantization import operations
from praxis.layers.quantization import quantization_hparams
from praxis.layers.quantization import quantizer
from praxis.layers.quantization import utils
from praxis.layers.quantization.sparsity import sparsifier

QuantizationMode = quantization_hparams.QuantizationMode
QuantizationType = quantization_hparams.QuantizationType
QuantizationParams = quantization_hparams.QuantizationParams
WeightHParams = base_layer.WeightHParams
instance_field = base_layer.instance_field
JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor
WeightInit = base_layer.WeightInit


class Linear(  # pytype: disable=signature-mismatch
    linears.Linear, quantizer.QuantizationLayer, sparsifier.SparsityBaseLayer
):
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

  def _get_sub_channel_shape(
      self, shape: Sequence[int], block_size: int, contract_dim: int
  ) -> Sequence[int]:
    """Converts a shape's contract dim into sub-channel and block_size.

    For activation, by => bsz
    For weight, yz => scz
    """
    sub_channels, rem = divmod(shape[contract_dim], block_size)
    if rem != 0:
      raise ValueError(
          f'block_size {block_size} must fully divide shape[contract_dim]'
          f' {shape}[{contract_dim}]'
      )
    out_shape = list(shape)
    out_shape.insert(contract_dim, sub_channels)
    out_shape[contract_dim + 1] = block_size
    return out_shape

  def _sub_channel_block_size(self) -> int:
    """Determine sub-channels' block_size if it was given."""
    if (
        self.quantization is not None
        and self.quantization.weight_params is not None
        and self.quantization.weight_params.block_size > 0
    ):
      return self.quantization.weight_params.block_size
    return 0

  def _get_weight_hparams(
      self, using_sub_channel: bool
  ) -> Tuple[WeightHParams, WeightHParams]:
    """Determines shard-aware weight params.

    Without sub-channel, the weight is sharded along the contract_dim, and the
    scale is replicated.

    With Sub-channel, both the weight and the scale are sharded along the
    sub-channel dimension.
    """
    wp = self.weight_split_dims_mapping
    weight_shape = [self.input_dims, self.output_dims]
    scale_shape = [self.output_dims]
    block_size = self._sub_channel_block_size()
    if using_sub_channel:
      weight_shape = self._get_sub_channel_shape(weight_shape, block_size, 0)
      scale_shape = [weight_shape[0], weight_shape[2]]
      weight_sharding = wp.wt.copy()
      weight_sharding.insert(1, -1)
      scale_sharding = wp.wt.copy()
    else:
      weight_sharding = wp.wt
      scale_sharding = (
          [wp.wt[1]] if wp.wt is not None and len(wp.wt) > 1 else [-1]
      )
    weight_hparams = WeightHParams(
        shape=weight_shape,
        mesh_shape=self.mesh_shape,
        tensor_split_dims_mapping=weight_sharding,
    )
    scale_hparams = WeightHParams(
        shape=scale_shape,
        mesh_shape=self.mesh_shape,
        tensor_split_dims_mapping=scale_sharding,
    )
    return (weight_hparams, scale_hparams)

  def setup(self) -> None:
    wp = self.weight_split_dims_mapping
    if self.rank > 0:
      if self._sub_channel_block_size() > 0:
        raise NotImplementedError(
            'No low rank support for sub-channel quantization'
        )
      shape_a, shape_b = (
          [self.input_dims, self.rank],
          [self.rank, self.output_dims],
      )
      wp_a = WeightHParams(
          shape=shape_a,
          mesh_shape=self.mesh_shape,
          tensor_split_dims_mapping=wp.wt,
      )
      self.set_up_weights(
          weight_name='w_a',
          weight_params=wp_a,
          scale_shape=[self.rank],
          pack_dim=self._PACK_4BIT_DIM,
      )
      self.create_aux_variables('w_a', wp_a)
      wp_b = WeightHParams(
          shape=shape_b,
          mesh_shape=self.mesh_shape,
          tensor_split_dims_mapping=wp.wt,
      )
      self.set_up_weights(
          weight_name='w_b',
          weight_params=wp_b,
          scale_shape=[self.output_dims],
          pack_dim=self._PACK_4BIT_DIM,
      )
      self.create_aux_variables('w_b', wp_b)

    else:
      block_size = self._sub_channel_block_size()
      using_sub_channel = False
      if block_size > 0:
        self._PACK_4BIT_DIM = 1
        using_sub_channel = (
            self.quantization is not None
            and self.quantization.mode == QuantizationMode.INFERENCE
        )
      weight_hparams, scale_hparams = self._get_weight_hparams(
          using_sub_channel
      )
      self.set_up_weights(
          weight_name='w',
          weight_params=weight_hparams,
          pack_dim=self._PACK_4BIT_DIM,
          scale_hparams=scale_hparams,
      )
      self.create_aux_variables('w', weight_hparams)

  def __call__(self, inputs: JTensor) -> JTensor:
    """Apply projection to inputs.

    Args:
      inputs: The inputs JTensor.  Shaped [..., input_dims].

    Returns:
      Projected inputs.
    """

    # Adjust sharding annotation during decoding.
    # TODO(pax): This logic should likely be lifted somewhere else.
    ap_out = self.activation_split_dims_mapping.out

    q_einsum_params = {
        'eqn': '...y,yz->...z',
        'scale_eqn': None,
        'zp_eqn': None,
        'pack_dim': self._PACK_4BIT_DIM,
        'reshape': [],
    }

    if self.rank > 0:
      w_a = self.sparsifiy(self.theta.w_a, inputs=inputs, name='w_a')
      intermediate = self.quantized_einsum(
          x=inputs,
          w=w_a,
          weight_name='w_a',
          **q_einsum_params,
      )
      w_b = self.sparsifiy(self.theta.w_b, inputs=inputs, name='w_b')
      out = self.quantized_einsum(
          x=intermediate,
          w=w_b,
          weight_name='w_b',
          **q_einsum_params,
      )
    else:
      w = self.sparsifiy(self.theta.w, inputs=inputs, name='w')
      # Sub-channel
      block_size = self._sub_channel_block_size()
      if (
          self.quantization is not None
          and (
              self.quantization.mode == QuantizationMode.INFERENCE
              or self.quantization.quantization_type == QuantizationType.AQT
          )
          and block_size > 0
      ):
        # The contract dimension is split into s and c
        #   s := number of sub-channels,
        #   c := block size, the contract dim
        # Inputs must be reshaped from ...y into ...sc.
        # Weights are stored in scz form, scale and offset are stored as sz.
        inputs_shape = list(inputs.shape)
        inputs = jnp.reshape(
            inputs,
            self._get_sub_channel_shape(
                inputs_shape, block_size, len(inputs_shape) - 1
            ),
        )
        q_einsum_params['eqn'] = '...sc,scz->...sz'
        q_einsum_params['scale_eqn'] = '...sz,sz->...z'
        q_einsum_params['zp_eqn'] = '...sc,sz->...z'
        if len(w.shape) == 2:
          q_einsum_params['reshape'] = self._get_sub_channel_shape(
              list(w.shape), block_size, 0
          )
      out = self.quantized_einsum(
          x=inputs, w=w, weight_name='w', **q_einsum_params
      )
    if ap_out is not None and len(ap_out) == 3 and out.ndim == 2:
      ap_out = [ap_out[0], ap_out[2]]
    return base_layer.maybe_shard(out, ap_out, self.mesh_axis_names)

  def quantized_partition_specs(self) -> Any:
    """Get quantized PartitionSpec.

    Returns:
      a map from names to partition spec.
    """
    assert self.quantization is not None, (
        'quantized_partition_specs is called during serving for quantized'
        ' model, please set quantized config for the model.'
    )
    # Activation variable partitioning is only needed for static quantization.
    if self._do_static_activation_quantization():
      raise NotImplementedError(
          'Static activation quantization is not supported yet.'
      )

    scale_name = 'w' + base_layer.QUANTIZED_SCALE_NAME_POSTFIX
    zp_name = 'w' + base_layer.QUANTIZED_ZP_NAME_POSTFIX

    block_size = self._sub_channel_block_size()
    weight_hparams, scale_hparams = self._get_weight_hparams(block_size > 0)
    weight_pspec = base_layer._weight_hparam_to_pspec(  # pylint: disable=protected-access
        weight_hparams, self.mesh_axis_names
    )
    scale_pspec = base_layer._weight_hparam_to_pspec(  # pylint: disable=protected-access
        scale_hparams, self.mesh_axis_names
    )
    partitionspec = {'w': weight_pspec, scale_name: scale_pspec}

    if not self.quantization.weight_params.use_symmetric:
      partitionspec[zp_name] = copy.deepcopy(scale_pspec)

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
    # static_activation_quantization is not supported by any quantization mode.
    if self._do_static_activation_quantization():
      raise NotImplementedError(
          'Static activation quantization is not supported.'
      )
    scale_name = 'w' + base_layer.QUANTIZED_SCALE_NAME_POSTFIX
    zp_name = 'w' + base_layer.QUANTIZED_ZP_NAME_POSTFIX

    w = self.theta.w
    contract_dims = [0]
    block_size = self._sub_channel_block_size()
    if block_size > 0:
      w = jnp.reshape(
          w, self._get_sub_channel_shape(list(w.shape), block_size, 0)
      )
      contract_dims = [1]

    if self.quantization.quantization_type in [
        QuantizationType.PTQ,
        QuantizationType.FQ,
        QuantizationType.FQ_VN,
    ]:
      if w.dtype != self.dtype:
        w = w.astype(self.dtype)
      q_w, q_s, zp = operations.reduce_precision(
          w,
          contract_dims,
          bits=self.quantization.weight_params.precision,
          percentile=self.quantization.weight_params.clipping_coeff,
          use_symmetric=self.quantization.weight_params.use_symmetric,
      )
      q_s = jnp.squeeze(q_s)
      if zp is not None:
        zp = jnp.squeeze(zp)
    elif self.quantization.quantization_type == QuantizationType.AQT:
      q_w, q_s, zp = self.weight_quantizer.quantize(
          w,
          contract_dims,
          squeeze_scale=True,
          quantized_dtype=self.quantization.weight_params.dtype,
      )
    # Internal quantization type support.

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
      return {base_layer.PARAMS: {'w': q_w, scale_name: q_s, zp_name: zp}}
