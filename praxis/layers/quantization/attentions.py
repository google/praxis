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
from praxis import pax_fiddle
from typing import Tuple, Any, Sequence

from jax import numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from praxis import base_layer
from praxis import pytypes
from praxis.layers import attentions
from praxis.layers.quantization import aqt
from praxis.layers.quantization import operations
from praxis.layers.quantization import quantization_hparams
from praxis.layers.quantization import utils

QuantizationHParams = quantization_hparams.QuantizationHParams
QuantizationMode = quantization_hparams.QuantizationMode
QuantizationType = quantization_hparams.QuantizationType
WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams
sub_config_field = base_layer.sub_config_field
JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor


class AttentionProjection(attentions.AttentionProjection):
  """Layer that computes quantized multi heads projection.

  This layer is expected to be used within DotProductAttention.

  Attributes:
    quantization: Information related to the quantization applied to this
      layer, such as dtype for the quantized weight.
  """
  quantization: QuantizationHParams = sub_config_field(QuantizationHParams)

  def create_tensor_quantizer(self):
    self.create_child(
        'act_quantizer',
        pax_fiddle.Config(
            aqt.TensorQuantizer,
            name='act_quantizer',
            precision=self.quantization.act_params.precision
            if self.quantization.act_params
            else None,
        ),
    )
    self.create_child(
        'weight_quantizer',
        pax_fiddle.Config(
            aqt.TensorQuantizer,
            name='weight_quantizer',
            precision=self.quantization.weight_params.precision,
        ),
    )

  def setup(self) -> None:
    wp = self.weight_split_dims_mapping
    has_sharding = self.mesh_shape is not None and wp.wt is not None
    if self.attention_combine_dims:
      assert not self.use_bias
      hd_shape = [self.num_heads * self.dim_per_head]
    else:
      hd_shape = [self.num_heads, self.dim_per_head]

    if self.attention_combine_dims and has_sharding:
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
    pc_shape = [self.input_dim] + hd_shape
    if self.is_output_projection and self.use_nhd_shape:
      pc_shape = hd_shape + [self.input_dim]
    pc = WeightHParams(
        shape=pc_shape, mesh_shape=self.mesh_shape, tensor_split_dims_mapping=wt
    )
    if self.quantization.mode == QuantizationMode.INFERENCE:
      if self.is_output_projection:
        self.create_quantized_variable('w', pc, [self.input_dim])
      else:
        self.create_quantized_variable('w', pc, hd_shape)
    else:
      self.create_variable('w', pc)
    if self.use_bias:
      if self.is_output_projection:
        if has_sharding:
          bias_split_dims_mapping = [wp.wt[0]]
        else:
          bias_split_dims_mapping = None
        pc_bias = WeightHParams(
            shape=[self.input_dim],
            init=WeightInit.Constant(0.0),
            mesh_shape=self.mesh_shape,
            tensor_split_dims_mapping=bias_split_dims_mapping,
        )
      else:
        if has_sharding:
          bias_split_dims_mapping = [wp.wt[1], wp.wt[2]]
        else:
          bias_split_dims_mapping = None
        pc_bias = WeightHParams(
            shape=[self.num_heads, self.dim_per_head],
            init=WeightInit.Constant(0.0),
            mesh_shape=self.mesh_shape,
            tensor_split_dims_mapping=bias_split_dims_mapping,
        )
      self.create_variable('b', pc_bias)

    if self.quantization.quantization_type == QuantizationType.AQT:
      self.create_tensor_quantizer()

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
    if self.attention_combine_dims:
      pc_shape = [self.input_dim, self.num_heads, self.dim_per_head]
      if self.is_output_projection and self.use_nhd_shape:
        pc_shape = [self.num_heads, self.dim_per_head, self.input_dim]
      w = jnp.reshape(theta.w, pc_shape)
    else:
      w = theta.w

    if self.is_output_projection:
      assert shape[-2:] == (self.num_heads, self.dim_per_head)
      batch_eqn = eqn_sym[:(rank - 2)]
      if self.use_nhd_shape:
        eqn = f'{batch_eqn}NH,NHD->{batch_eqn}D'
      else:
        eqn = f'{batch_eqn}NH,DNH->{batch_eqn}D'
    else:
      assert (
          shape[-1] == self.input_dim
      ), f'Expecting shape[-1] == p.input_dim, {shape[-1]} != {self.input_dim}'
      batch_eqn = eqn_sym[:(rank - 1)] if rank else '...'
      eqn = f'{batch_eqn}D,DNH->{batch_eqn}NH'

    if self.quantization.mode == QuantizationMode.INFERENCE:
      w, s = self.get_quantized_weight('w')
      # TODO(b/262309036): refactor logics under INFERNCE so there is no
      # difference in quantization_type and there is no need for
      # lhs_quantizer/rhs_quantizer.
      if self.quantization.quantization_type == QuantizationType.AQT:
        dimension_numbers, perm = utils.convert_einsum_eqn_to_dimension_numbers(
            eqn)
        ret = operations.dot_general(
            lhs=inputs,
            rhs=None,
            lhs_quantizer=self.act_quantizer,
            rhs_quantizer=self.weight_quantizer,
            dimension_numbers=dimension_numbers,
            is_eval=True,
            perm=perm,
            rhs_quantized=(w, s))
      else:
        if (
            self.quantization.act_params is not None
            and self.quantization.act_params.stats_config is not None
        ):
          raise NotImplementedError(
              'Static activation quantization is not supported yet.'
          )
        elif (
            self.quantization.act_params is not None
            and self.quantization.act_params.stats_config is None
        ):
          inputs, act_scale = operations.reduce_precision_activation(inputs)
          ret = operations.einsum(eqn, inputs, w, jnp.multiply(act_scale, s))
        elif self.quantization.act_params is None:
          ret = operations.einsum(eqn, inputs, w, s)
    elif (
        self.quantization.mode == QuantizationMode.TRAINING
        or self.quantization.mode == QuantizationMode.MATERIALIZE
    ):
      if self.quantization.quantization_type == QuantizationType.AQT:
        dimension_numbers, perm = utils.convert_einsum_eqn_to_dimension_numbers(
            eqn)
        ret = operations.dot_general(
            lhs=inputs,
            rhs=w,
            lhs_quantizer=self.act_quantizer,
            rhs_quantizer=self.weight_quantizer,
            dimension_numbers=dimension_numbers,
            is_eval=self.do_eval,
            perm=perm)
      elif self.quantization.quantization_type == QuantizationType.FQ:
        w = operations.fakequant_einsum(eqn, w)
        ret = jnp.einsum(eqn, inputs, w)
      elif self.quantization.quantization_type == QuantizationType.PTQ:
        ret = jnp.einsum(eqn, inputs, w)

    else:
      raise ValueError(
          f'Unsupported quantization_mode {self.quantization.mode}'
      )

    if self.use_bias:
      ret += theta.b
    return ret

  def quantized_partitioned_specs(self) -> Any:
    """Get quantized PartitionSpec.

    Returns:
      a map from names to partition spec.
    """
    scale_name = 'w' + base_layer.QUANTIZED_NAME_POSTFIX
    weight_pspec = base_layer._weight_hparam_to_pspec(
        self._weight_hparams['w'], self.mesh_axis_names
    )
    wp = self.weight_split_dims_mapping
    if self.is_output_projection:
      scale_split_dims_mapping = [wp.wt[0]]
    else:
      scale_split_dims_mapping = [wp.wt[1], wp.wt[2]]
    # scale_weight_hparam is unmaterialized so shape is irrelevant.
    scale_weight_hparam = WeightHParams(
        shape=(), tensor_split_dims_mapping=scale_split_dims_mapping)
    scale_pspec = base_layer._weight_hparam_to_pspec(
        scale_weight_hparam, self.mesh_axis_names
    )
    partitionspec = {'w': weight_pspec, scale_name: scale_pspec}
    return {base_layer.PARAMS: partitionspec}

  def quantize_weight(self) -> NestedJTensor:
    """Get quantized weight.

    Returns:
      a map from names to quantized weights.
    """
    eqn = ''
    # This matches the equantion logic in __call__ for weights.
    if self.is_output_projection:
      if self.use_nhd_shape:
        eqn = 'ANH,NHD->AD'
      else:
        eqn = 'ANH,DNH->AD'
    else:
      eqn = 'AD,DNH->ANH'

    # TODO(jihwanlee): Handle the cases for FQ and static quantization.
    if self.quantization.quantization_type == QuantizationType.PTQ:
      q_w, q_s = operations.reduce_einsum_weight_precision(
          eqn, self.theta.w, calculation_type=self.dtype)
    elif self.quantization.quantization_type == QuantizationType.AQT:
      dimension_numbers, _ = utils.convert_einsum_eqn_to_dimension_numbers(
          eqn)
      weight_contract_dims = dimension_numbers[0][1]
      q_s = self.weight_quantizer.get_quant_scale(
          self.theta.w, contract_dims=weight_contract_dims, dtype=self.dtype)
      q_w = q_s * self.theta.w
      q_w = self.weight_quantizer.to_quant(q_w, dtype=jnp.int8)
      q_s = jnp.squeeze(q_s)
    else:
      raise ValueError(
          f'Usupported quantization_type {self.quantization.quantization_type}'
      )

    scale_name = 'w' + base_layer.QUANTIZED_NAME_POSTFIX
    return {base_layer.PARAMS: {'w': q_w, scale_name: q_s}}


class CombinedQKVProjectionLayer(attentions.CombinedQKVProjectionLayer):
  """Layer that computes quantized QKV projection with a combined weight.

  This layer is expected to be used within DotProductAttention below.

  Attributes:
    quantization: Information related to the quantization applied to this
      layer, such as dtype for the quantized weight.
  """
  quantization: QuantizationHParams = sub_config_field(QuantizationHParams)

  def create_tensor_quantizer(self):
    self.create_child(
        'act_quantizer',
        pax_fiddle.Config(
            aqt.TensorQuantizer,
            name='act_quantizer',
            precision=self.quantization.act_params.precision
            if self.quantization.act_params
            else None,
        ),
    )
    self.create_child(
        'weight_quantizer',
        pax_fiddle.Config(
            aqt.TensorQuantizer,
            name='weight_quantizer',
            precision=self.quantization.weight_params.precision,
        ),
    )

  def setup(self) -> None:
    wp = self.weight_split_dims_mapping
    if self.mesh_shape is not None:
      assert wp.wt is not None, ('Must provide sharding annotations for the '
                                 'weights if mesh shape is provided')
      if (
          self.attention_combine_dims
          and isinstance(wp.wt, Sequence)
          and len(wp.wt) == 3
      ):
        wt = [axis for axis in wp.wt if axis is not None]
        assert len(wt) == 2, ('wp.wt only specifies the sharding for '
                              'the last two dims of the weight tensor.')
      else:
        wt = wp.wt
        # Replicate the concat axis.
        assert len(wt) == 3, ('wp.wt only specifies the sharding for '
                              'the last three dims of the weight tensor.')
      weight_split_dims_mapping = [None] + list(wt)
      if self.attention_combine_dims:
        bias_split_dims_mapping = [None, wt[1]]
      else:
        bias_split_dims_mapping = [None, wt[1], wt[2]]
    else:
      weight_split_dims_mapping = None
      bias_split_dims_mapping = None

    if self.attention_combine_dims:
      hd_shape = [self.num_heads * self.dim_per_head]
    else:
      hd_shape = [self.num_heads, self.dim_per_head]

    pc_shape = [3, self.input_dim] + hd_shape
    # Combined weight for q, k, v projections.
    pc = WeightHParams(
        shape=pc_shape,
        init=self.params_init,
        dtype=self.dtype,
        mesh_shape=self.mesh_shape,
        tensor_split_dims_mapping=weight_split_dims_mapping,
    )
    if self.quantization.mode == QuantizationMode.INFERENCE:
      self.create_quantized_variable('w', pc, [3] + hd_shape)
    else:
      self.create_variable('w', pc)
    if self.use_bias:
      # Combined bias weight for q, k, v projections.
      pc_bias = WeightHParams(
          shape=[3] + hd_shape,
          init=WeightInit.Constant(0.0),
          mesh_shape=self.mesh_shape,
          tensor_split_dims_mapping=bias_split_dims_mapping,
      )
      self.create_variable('b', pc_bias)

    if self.quantization.quantization_type == QuantizationType.AQT:
      self.create_tensor_quantizer()

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

    assert shape[-1] == self.input_dim
    batch_dims_rank = rank - 1
    batch_eqn = eqn_sym[:batch_dims_rank] if rank else '...'
    if self.attention_combine_dims:
      pc_shape = [3, self.input_dim, self.num_heads, self.dim_per_head]
      w = jnp.reshape(theta.w, pc_shape)
      if self.use_bias:
        b_shape = [3, self.num_heads, self.dim_per_head]
        b = jnp.reshape(theta.b, b_shape)
    else:
      w = theta.w
      if self.use_bias:
        b = theta.b

    # K indexes qkv.
    eqn = f'{batch_eqn}D,KDNH->K{batch_eqn}NH'
    # TOOD(jihwanlee): Implement the inference logic that can be shared between
    # different quantization types.
    if self.quantization.mode == QuantizationMode.INFERENCE:
      w, s = self.get_quantized_weight('w')
      # TODO(b/262309036): refactor logics under INFERNCE so there is no
      # difference in quantization_type and there is no need for
      # lhs_quantizer/rhs_quantizer.
      if self.quantization.quantization_type == QuantizationType.AQT:
        dimension_numbers, perm = utils.convert_einsum_eqn_to_dimension_numbers(
            eqn)
        ret = operations.dot_general(
            lhs=inputs,
            rhs=None,
            lhs_quantizer=self.act_quantizer,
            rhs_quantizer=self.weight_quantizer,
            dimension_numbers=dimension_numbers,
            is_eval=True,
            perm=perm,
            rhs_quantized=(w, s))
      else:
        if (
            self.quantization.act_params is not None
            and self.quantization.act_params.stats_config is not None
        ):
          raise NotImplementedError(
              'Static activation quantization is not supported yet.'
          )
        elif (
            self.quantization.act_params is not None
            and self.quantization.act_params.stats_config is None
        ):
          inputs, act_scale = operations.reduce_precision_activation(inputs)
          ret = operations.einsum(eqn, inputs, w, jnp.multiply(act_scale, s))
        elif self.quantization.act_params is None:
          ret = operations.einsum(eqn, inputs, w, s)
    else:
      if self.quantization.quantization_type == QuantizationType.AQT:
        dimension_numbers, perm = utils.convert_einsum_eqn_to_dimension_numbers(
            eqn)
        ret = operations.dot_general(
            lhs=inputs,
            rhs=w,
            lhs_quantizer=self.act_quantizer,
            rhs_quantizer=self.weight_quantizer,
            dimension_numbers=dimension_numbers,
            is_eval=self.do_eval,
            perm=perm)
      elif self.quantization.quantization_type == QuantizationType.FQ:
        w = operations.fakequant_einsum(eqn, w)
        ret = jnp.einsum(eqn, inputs, w)
      elif self.quantization.quantization_type == QuantizationType.PTQ:
        ret = jnp.einsum(eqn, inputs, w)

    ret = checkpoint_name(ret, 'combined_qkv_proj')
    if self.use_bias:
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
    scale_name = 'w' + base_layer.QUANTIZED_NAME_POSTFIX
    weight_pspec = base_layer._weight_hparam_to_pspec(
        self._weight_hparams['w'], self.mesh_axis_names
    )
    wp = self.weight_split_dims_mapping
    if self.attention_combine_dims:
      scale_split_dims_mapping = [None, wp.wt[1]]
    else:
      scale_split_dims_mapping = [None, wp.wt[1], wp.wt[2]]
    # scale_weight_hparam is unmaterialized so shape is irrelevant.
    scale_weight_hparam = WeightHParams(
        shape=(), tensor_split_dims_mapping=scale_split_dims_mapping)
    scale_pspec = base_layer._weight_hparam_to_pspec(
        scale_weight_hparam, self.mesh_axis_names
    )
    partitionspec = {'w': weight_pspec, scale_name: scale_pspec}
    return {base_layer.PARAMS: partitionspec}

  def quantize_weight(self) -> NestedJTensor:
    """Get quantized weight.

    Returns:
      a map from names to quantized weights.
    """
    theta = self.theta
    eqn = 'AD,KDNH->KANH'
    # TODO(jihwanlee): Handle the cases for FQ and static quantization.
    if self.quantization.quantization_type == QuantizationType.PTQ:
      q_w, q_s = operations.reduce_einsum_weight_precision(
          eqn, theta.w, calculation_type=self.dtype)
    elif self.quantization.quantization_type == QuantizationType.AQT:
      dimension_numbers, _ = utils.convert_einsum_eqn_to_dimension_numbers(
          eqn)
      weight_contract_dims = dimension_numbers[0][1]
      q_s = self.weight_quantizer.get_quant_scale(
          self.theta.w, contract_dims=weight_contract_dims, dtype=self.dtype)
      q_w = q_s * self.theta.w
      q_w = self.weight_quantizer.to_quant(q_w, dtype=jnp.int8)
      q_s = jnp.squeeze(q_s)
    else:
      raise ValueError(
          f'Usupported quantization_type {self.quantization.quantization_type}'
      )

    scale_name = 'w' + base_layer.QUANTIZED_NAME_POSTFIX
    return {base_layer.PARAMS: {'w': q_w, scale_name: q_s}}
