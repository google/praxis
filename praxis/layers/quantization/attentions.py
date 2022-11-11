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

"""Quantized Attention Layers."""

import string
from typing import Tuple, Any

from jax import numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from praxis import base_layer
from praxis import pytypes
from praxis.layers import attentions
from praxis.layers.quantization import operations

QuantizationHParams = base_layer.QuantizationHParams
WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams
sub_config_field = base_layer.sub_config_field
JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor


class AttentionProjection(attentions.AttentionProjection):
  """Layer that computes quantized multi heads projection.

    This layer is expected to be used within DotProductAttention.
  """

  class HParams(attentions.AttentionProjection.HParams):
    """Associated hyper-params for this layer class.

    Attributes:
      quantization: Information related to the quantization applied to this
        layer, such as dtype for the quantized weight.
    """
    quantization: QuantizationHParams = sub_config_field(QuantizationHParams)

  def setup(self) -> None:
    p = self.hparams
    wp = p.weight_split_dims_mapping
    has_sharding = p.mesh_shape is not None and wp.wt is not None
    if p.attention_combine_dims:
      assert not p.use_bias
      hd_shape = [p.num_heads * p.dim_per_head]
    else:
      hd_shape = [p.num_heads, p.dim_per_head]

    if p.attention_combine_dims and has_sharding:
      if len(wp.wt) == 3:
        h_sharding = ()
        for axes in (wp.wt[0], wp.wt[1]):
          if isinstance(axes, (str, int)):
            h_sharding += (axes,)
          elif axes is not None:
            h_sharding += axes
        wt = [h_sharding, wp.wt[2]]
      assert len(wt) == 2
    else:
      wt = wp.wt
    pc_shape = [p.input_dim] + hd_shape
    if p.is_output_projection and p.use_nhd_shape:
      pc_shape = hd_shape + [p.input_dim]
    pc = WeightHParams(
        shape=pc_shape, mesh_shape=p.mesh_shape, tensor_split_dims_mapping=wt)
    if p.quantization.mode == base_layer.QuantizationMode.INFERENCE:
      if p.is_output_projection:
        self.create_quantized_variable('w', pc, [p.input_dim])
      else:
        self.create_quantized_variable('w', pc, hd_shape)
    else:
      self.create_variable('w', pc)
    if p.use_bias:
      if p.is_output_projection:
        if has_sharding:
          bias_split_dims_mapping = [wp.wt[0]]
        else:
          bias_split_dims_mapping = None
        pc_bias = WeightHParams(
            shape=[p.input_dim],
            init=WeightInit.Constant(0.0),
            mesh_shape=p.mesh_shape,
            tensor_split_dims_mapping=bias_split_dims_mapping)
      else:
        if has_sharding:
          bias_split_dims_mapping = [wp.wt[1], wp.wt[2]]
        else:
          bias_split_dims_mapping = None
        pc_bias = WeightHParams(
            shape=[p.num_heads, p.dim_per_head],
            init=WeightInit.Constant(0.0),
            mesh_shape=p.mesh_shape,
            tensor_split_dims_mapping=bias_split_dims_mapping)
      self.create_variable('b', pc_bias)

  def __call__(self, inputs: JTensor) -> JTensor:
    """Computes the multi headed projection for inputs.

    Args:
      inputs: A JTensor of shape [..., num_heads, dim_per_head] if
        p.is_output_projection is True or [..., p.input_dim] otherwise..

    Returns:
      The projected JTensor with shape [..., p.input_dim] if
      p.is_output_projection is True or [..., num_heads, dim_per_head]
      otherwise.
    """
    p = self.hparams
    theta = self.theta

    # Because tf.einsum is not fully optimized unless all the dimensions are
    # fully specified, we have to avoid using '...' for batch dimensions in the
    # equation in tf.einsum for optimized performance. This is only feasible
    # when the rank of the tensor is known.
    # Sort the available symbols to avoid nondeterminism.
    eqn_sym = ''.join(sorted(set(string.ascii_uppercase) - set('DHN')))
    shape = inputs.shape
    rank = len(shape)

    inputs = self._cast_to_fprop_dtype(inputs)
    if p.attention_combine_dims:
      pc_shape = [p.input_dim, p.num_heads, p.dim_per_head]
      if p.is_output_projection and p.use_nhd_shape:
        pc_shape = [p.num_heads, p.dim_per_head, p.input_dim]
      w = jnp.reshape(theta.w, pc_shape)
    else:
      w = theta.w

    if p.is_output_projection:
      assert shape[-2:] == (p.num_heads, p.dim_per_head)
      batch_eqn = eqn_sym[:(rank - 2)]
      if p.use_nhd_shape:
        eqn = f'{batch_eqn}NH,NHD->{batch_eqn}D'
      else:
        eqn = f'{batch_eqn}NH,DNH->{batch_eqn}D'
    else:
      assert shape[-1] == p.input_dim, (
          f'Expecting shape[-1] == p.input_dim, {shape[-1]} != {p.input_dim}')
      batch_eqn = eqn_sym[:(rank - 1)] if rank else '...'
      eqn = f'{batch_eqn}D,DNH->{batch_eqn}NH'

    if p.quantization.mode == base_layer.QuantizationMode.INFERENCE:
      w, s = self.get_quantized_weight('w')
      ret = operations.einsum(eqn, inputs, w, s)
    else:
      ret = jnp.einsum(eqn, inputs, w)
    if p.use_bias:
      ret += theta.b
    return ret

  def quantized_partitioned_specs(self) -> Any:
    """Get quantized PartitionSpec.

    Returns:
      a map from names to partition spec.
    """
    p = self.hparams
    scale_name = 'w' + base_layer.QUANTIZED_NAME_POSTFIX
    weight_pspec = base_layer._weight_hparam_to_pspec(
        self._weight_hparams['w'], self.hparams.mesh_axis_names)
    wp = p.weight_split_dims_mapping
    if p.is_output_projection:
      scale_split_dims_mapping = [wp.wt[0]]
    else:
      scale_split_dims_mapping = [wp.wt[1], wp.wt[2]]
    # scale_weight_hparam is unmaterialized so shape is irrelevant.
    scale_weight_hparam = WeightHParams(
        shape=(), tensor_split_dims_mapping=scale_split_dims_mapping)
    scale_pspec = base_layer._weight_hparam_to_pspec(
        scale_weight_hparam, self.hparams.mesh_axis_names)
    partitionspec = {'w': weight_pspec, scale_name: scale_pspec}
    return {base_layer.PARAMS: partitionspec}

  def quantize_weight(self) -> NestedJTensor:
    """Get quantized weight.

    Returns:
      a map from names to quantized weights.
    """
    p = self.hparams
    assert p.quantization.mode == base_layer.QuantizationMode.QUANTIZE
    eqn = ''
    # This matches the equantion logic in __call__ for weights.
    if p.is_output_projection:
      if p.use_nhd_shape:
        eqn = 'ANH,NHD->AD'
      else:
        eqn = 'ANH,DNH->AD'
    else:
      eqn = 'AD,DNH->ANH'
    q_w, q_s = operations.reduce_einsum_weight_precision(eqn, self.theta.w)
    scale_name = 'w' + base_layer.QUANTIZED_NAME_POSTFIX
    return {base_layer.PARAMS: {'w': q_w, scale_name: q_s}}


class CombinedQKVProjectionLayer(attentions.CombinedQKVProjectionLayer):
  """Layer that computes quantized QKV projection with a combined weight.

  This layer is expected to be used within DotProductAttention below.
  """

  class HParams(attentions.CombinedQKVProjectionLayer.HParams):
    """Associated hyper-params for this layer class.

    Attributes:
      quantization: Information related to the quantization applied to this
        layer, such as dtype for the quantized weight.
    """
    quantization: base_layer.QuantizationHParams = sub_config_field(
        base_layer.QuantizationHParams)

  def setup(self) -> None:
    p = self.hparams
    wp = p.weight_split_dims_mapping
    if p.mesh_shape is not None:
      assert wp.wt is not None, ('Must provide sharding annotations for the '
                                 'weights if mesh shape is provided')
      if (p.attention_combine_dims and isinstance(wp.wt, list) and
          len(wp.wt) == 3):
        wt = [axis for axis in wp.wt if axis is not None]
        assert len(wt) == 2, ('wp.wt only specifies the sharding for '
                              'the last two dims of the weight tensor.')
      else:
        wt = wp.wt
        # Replicate the concat axis.
        assert len(wt) == 3, ('wp.wt only specifies the sharding for '
                              'the last three dims of the weight tensor.')
      weight_split_dims_mapping = [None] + list(wt)
      if p.attention_combine_dims:
        bias_split_dims_mapping = [None, wt[1]]
      else:
        bias_split_dims_mapping = [None, wt[1], wt[2]]
    else:
      weight_split_dims_mapping = None
      bias_split_dims_mapping = None

    if p.attention_combine_dims:
      hd_shape = [p.num_heads * p.dim_per_head]
    else:
      hd_shape = [p.num_heads, p.dim_per_head]

    pc_shape = [3, p.input_dim] + hd_shape
    # Combined weight for q, k, v projections.
    pc = WeightHParams(
        shape=pc_shape,
        init=p.params_init,
        dtype=p.dtype,
        mesh_shape=p.mesh_shape,
        tensor_split_dims_mapping=weight_split_dims_mapping)
    if p.quantization.mode == base_layer.QuantizationMode.INFERENCE:
      self.create_quantized_variable('w', pc, [3] + hd_shape)
    else:
      self.create_variable('w', pc)
    if p.use_bias:
      # Combined bias weight for q, k, v projections.
      pc_bias = WeightHParams(
          shape=[3] + hd_shape,
          init=WeightInit.Constant(0.0),
          mesh_shape=p.mesh_shape,
          tensor_split_dims_mapping=bias_split_dims_mapping)
      self.create_variable('b', pc_bias)

  # TODO(zhangqiaorjc): Take query, key, value as inputs to support all
  # attentions.
  def __call__(self, inputs: JTensor) -> Tuple[JTensor, JTensor, JTensor]:
    """Computes the QKV projection for inputs.

    Args:
      inputs: A JTensor of shape [..., p.input_dim].

    Returns:
      The three projected JTensor with shape [..., num_heads, dim_per_head]
      in q_proj, k_proj and v_proj order.
    """
    p = self.hparams
    theta = self.theta

    # Because tf.einsum is not fully optimized unless all the dimensions are
    # fully specified, we have to avoid using '...' for batch dimensions in the
    # equation in tf.einsum for optimized performance. This is only feasible
    # when the rank of the tensor is known.
    # Sort the available symbols to avoid nondeterminism.
    eqn_sym = ''.join(sorted(set(string.ascii_uppercase) - set('KDHN')))
    shape = inputs.shape
    rank = len(shape)
    assert rank > 0

    assert shape[-1] == p.input_dim
    batch_dims_rank = rank - 1
    batch_eqn = eqn_sym[:batch_dims_rank] if rank else '...'
    if p.attention_combine_dims:
      pc_shape = [3, p.input_dim, p.num_heads, p.dim_per_head]
      w = jnp.reshape(theta.w, pc_shape)
      if p.use_bias:
        b_shape = [3, p.num_heads, p.dim_per_head]
        b = jnp.reshape(theta.b, b_shape)
    else:
      w = theta.w
      if p.use_bias:
        b = theta.b

    # K indexes qkv.
    eqn = f'{batch_eqn}D,KDNH->K{batch_eqn}NH'
    if p.quantization.mode == base_layer.QuantizationMode.INFERENCE:
      w, s = self.get_quantized_weight('w')
      ret = operations.einsum(eqn, inputs, w, s)
    else:
      ret = jnp.einsum(eqn, inputs, w)
    ret = checkpoint_name(ret, 'combined_qkv_proj')
    if p.use_bias:
      # Add newaxis to bias weight for each batch dim since ret is K...NH
      # and theta.b is KNH. Need to reshape theta.b to K...NH
      ret += jnp.expand_dims(b, list(range(1, batch_dims_rank + 1)))
    # Split into three projections.
    query_proj, key_proj, value_proj = ret
    query_proj = checkpoint_name(query_proj, 'query_proj')
    key_proj = checkpoint_name(key_proj, 'key_proj')
    value_proj = checkpoint_name(value_proj, 'value_proj')
    return query_proj, key_proj, value_proj

  def quantized_partitioned_specs(self) -> Any:
    """Get quantized PartitionSpec.

    Returns:
      a map from names to partition spec.
    """
    p = self.hparams
    scale_name = 'w' + base_layer.QUANTIZED_NAME_POSTFIX
    weight_pspec = base_layer._weight_hparam_to_pspec(
        self._weight_hparams['w'], self.hparams.mesh_axis_names)
    wp = p.weight_split_dims_mapping
    if p.attention_combine_dims:
      scale_split_dims_mapping = [None, wp.wt[1]]
    else:
      scale_split_dims_mapping = [None, wp.wt[1], wp.wt[2]]
    # scale_weight_hparam is unmaterialized so shape is irrelevant.
    scale_weight_hparam = WeightHParams(
        shape=(), tensor_split_dims_mapping=scale_split_dims_mapping)
    scale_pspec = base_layer._weight_hparam_to_pspec(
        scale_weight_hparam, self.hparams.mesh_axis_names)
    partitionspec = {'w': weight_pspec, scale_name: scale_pspec}
    return {base_layer.PARAMS: partitionspec}

  def quantize_weight(self) -> NestedJTensor:
    """Get quantized weight.

    Returns:
      a map from names to quantized weights.
    """
    theta = self.theta
    p = self.hparams
    assert p.quantization.mode == base_layer.QuantizationMode.QUANTIZE
    eqn = 'AD,KDNH->KANH'
    q_w, q_s = operations.reduce_einsum_weight_precision(eqn, theta.w)
    scale_name = 'w' + base_layer.QUANTIZED_NAME_POSTFIX
    return {base_layer.PARAMS: {'w': q_w, scale_name: q_s}}
