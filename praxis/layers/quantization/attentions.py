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

"""Quantized Attention Layers."""

import copy
import string
from typing import Any, Optional, Sequence, Tuple

import jax
from jax import numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from praxis import base_layer
from praxis import py_utils
from praxis import pytypes
from praxis.layers import attentions
from praxis.layers.quantization import quantizer
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

  _PACK_4BIT_DIM = 0

  def create_tensor_quantizers(self):
    self.create_child(
        'act_quantizer',
        quantizer.create_tensor_quantizer(
            'aqt_quantizer', self.quantization.act_params
        ),
    )
    self.create_child(
        'weight_quantizer',
        quantizer.create_tensor_quantizer(
            'weight_quantizer', self.quantization.weight_params
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
      dtype = self.quantization.weight_params.dtype
      if self.quantization.weight_params.precision == 4:
        dtype = jnp.int32
        pc.shape = utils.get_packed_shape(
            pc.shape, self._PACK_4BIT_DIM, packing_factor=8
        )
      scale_shape = [self.input_dim] if self.is_output_projection else hd_shape
      self.create_quantized_variable(
          'w',
          pc,
          scale_shape,
          dtype=dtype,
          use_symmetric=self.quantization.weight_params.use_symmetric,
      )
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
      self.create_tensor_quantizers()

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
        ret = operations.einsum(
            eqn, inputs, w, jnp.multiply(jnp.squeeze(act_scale), s)
        )
      elif self.quantization.act_params is None:
        if self.quantization.weight_params.use_symmetric:
          if self.quantization.weight_params.dequant_upfront:
            raise NotImplementedError('Dequantize upfront not supported.')
          else:
            ret = operations.einsum(eqn, inputs, w, s)
        else:
          assert zp is not None, 'zp cannot be None when use_symmetric=False.'
          ret = operations.einsum(eqn, inputs, w, s, zp)

    else:
      if self.quantization.quantization_type == QuantizationType.AQT:
        ret = operations.aqt_einsum(
            eqn=eqn,
            lhs=inputs,
            rhs=w,
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
        ret = jnp.einsum(eqn, inputs, w)
      elif self.quantization.quantization_type == QuantizationType.PTQ:
        ret = jnp.einsum(eqn, inputs, w)

    if self.use_bias:
      ret += theta.b
    return ret

  def quantized_partition_specs(self) -> Any:
    """Get quantized PartitionSpec.

    Returns:
      a map from names to partition spec.
    """
    scale_name = 'w' + base_layer.QUANTIZED_SCALE_NAME_POSTFIX
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

    if not self.quantization.weight_params.use_symmetric:
      zp_name = 'w' + base_layer.QUANTIZED_ZP_NAME_POSTFIX
      partitionspec[zp_name] = copy.deepcopy(scale_pspec)

    return {base_layer.PARAMS: partitionspec}

  def quantize_weight(self) -> NestedJTensor:
    """Get quantized weight.

    Returns:
      a map from names to quantized weights.
    """
    eqn = ''
    # This matches the equation logic in __call__ for weights.
    if self.is_output_projection:
      if self.use_nhd_shape:
        eqn = 'ANH,NHD->AD'
      else:
        eqn = 'ANH,DNH->AD'
    else:
      eqn = 'AD,DNH->ANH'

    percentile = self.quantization.weight_params.clipping_coeff
    # TODO(jihwanlee): Handle the cases for FQ and static quantization.
    if self.quantization.quantization_type == QuantizationType.PTQ:
      q_w, q_s, zp = operations.reduce_einsum_weight_precision(
          eqn,
          self.theta.w,
          calculation_type=self.dtype,
          need_gradient=False,
          bits=self.quantization.weight_params.precision,
          optimization_on_bound=False,
          percentile=percentile,
          use_symmetric=self.quantization.weight_params.use_symmetric,
      )
      if self.quantization.weight_params.precision == 4:
        q_w = utils.pack_4bit(q_w, self._PACK_4BIT_DIM)
    elif self.quantization.quantization_type == QuantizationType.AQT:
      dimension_numbers, _ = utils.einsum_eqn_to_dimension_numbers(eqn)
      weight_contract_dims = dimension_numbers[0][1]
      q_w, q_s, zp = self.weight_quantizer.quantize(
          self.theta.w,
          weight_contract_dims,
          quantized_dtype=self.quantization.weight_params.dtype,
      )
    else:
      raise ValueError(
          f'Unsupported quantization_type {self.quantization.quantization_type}'
      )

    scale_name = 'w' + base_layer.QUANTIZED_SCALE_NAME_POSTFIX
    if self.quantization.weight_params.use_symmetric:
      return {base_layer.PARAMS: {'w': q_w, scale_name: q_s}}
    else:
      zp_name = 'w' + base_layer.QUANTIZED_ZP_NAME_POSTFIX
      return {base_layer.PARAMS: {'w': q_w, scale_name: q_s, zp_name: zp}}


class CombinedQKVProjectionLayer(attentions.CombinedQKVProjectionLayer):
  """Layer that computes quantized QKV projection with a combined weight.

  This layer is expected to be used within DotProductAttention below.

  Attributes:
    quantization: Information related to the quantization applied to this
      layer, such as dtype for the quantized weight.
  """
  quantization: QuantizationHParams = sub_config_field(QuantizationHParams)

  _PACK_4BIT_DIM = 1

  def create_tensor_quantizers(self):
    self.create_child(
        'act_quantizer',
        quantizer.create_tensor_quantizer(
            'aqt_quantizer', self.quantization.act_params
        ),
    )
    self.create_child(
        'weight_quantizer',
        quantizer.create_tensor_quantizer(
            'weight_quantizer', self.quantization.weight_params
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
      dtype = self.quantization.weight_params.dtype
      if self.quantization.weight_params.precision == 4:
        dtype = jnp.int32
        pc.shape = utils.get_packed_shape(
            pc.shape, self._PACK_4BIT_DIM, packing_factor=8
        )
      self.create_quantized_variable(
          'w',
          pc,
          [3] + hd_shape,
          dtype=dtype,
          use_symmetric=self.quantization.weight_params.use_symmetric,
      )
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
      self.create_tensor_quantizers()

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
        ret = operations.einsum(
            eqn, inputs, w, jnp.multiply(jnp.squeeze(act_scale), s)
        )
      elif self.quantization.act_params is None:
        if self.quantization.weight_params.use_symmetric:
          if self.quantization.weight_params.dequant_upfront:
            raise NotImplementedError('Dequantize upfront not supported.')
          else:
            ret = operations.einsum(eqn, inputs, w, s)
        else:
          ret = operations.einsum(eqn, inputs, w, s, zp)
    else:
      if self.quantization.quantization_type == QuantizationType.AQT:
        ret = operations.aqt_einsum(
            eqn=eqn,
            lhs=inputs,
            rhs=w,
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
    if self.attention_combine_dims:
      scale_split_dims_mapping = [None, wp.wt[1]]
    else:
      scale_split_dims_mapping = [None, wp.wt[1], wp.wt[2]]
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
      q_w, q_s, zp = operations.reduce_einsum_weight_precision(
          eqn,
          theta.w,
          calculation_type=self.dtype,
          need_gradient=False,
          bits=self.quantization.weight_params.precision,
          optimization_on_bound=False,
          percentile=self.quantization.weight_params.clipping_coeff,
          use_symmetric=self.quantization.weight_params.use_symmetric,
      )
      if self.quantization.weight_params.precision == 4:
        q_w = utils.pack_4bit(q_w, self._PACK_4BIT_DIM)
    elif self.quantization.quantization_type == QuantizationType.AQT:
      dimension_numbers, _ = utils.einsum_eqn_to_dimension_numbers(eqn)
      weight_contract_dims = dimension_numbers[0][1]
      q_w, q_s, zp = self.weight_quantizer.quantize(
          self.theta.w,
          weight_contract_dims,
          quantized_dtype=self.quantization.weight_params.dtype,
      )
    else:
      raise ValueError(
          f'Unsupported quantization_type {self.quantization.quantization_type}'
      )

    scale_name = 'w' + base_layer.QUANTIZED_SCALE_NAME_POSTFIX
    if self.quantization.weight_params.use_symmetric:
      return {base_layer.PARAMS: {'w': q_w, scale_name: q_s}}
    else:
      zp_name = 'w' + base_layer.QUANTIZED_ZP_NAME_POSTFIX
      return {base_layer.PARAMS: {'w': q_w, scale_name: q_s, zp_name: zp}}


class DotProductAttention(attentions.DotProductAttention):
  """Dot-product attention with multiple attention heads.

  This implementation heavily uses einsum to be efficient on TPUs.  We use the
  following capital letters to denote certain JTensor parameters.

    B = batch size
    S = length of the key/value (source)
    T = length of the query (target)
    D = model dimension
    N = number of attention heads
    H = dimensions of each attention head.

  The algorithm is sketched as follows. Each intermediate JTensor or weight
  JTensor is annotated with its shape. E.g., Wq, the weight JTensor for query's
  projection, its shape is [D, N, H].

  Trainable weights:
    Wq, Wk, Wv: [D{q,k,v}, N, H]
    Wout: [Dq, N, H]

  Note it also allows k, v and q to have different input dimension by setting
  input_dim as a dict: {'key': key_dim, 'value': value_dim, 'query': query_dim}.

  Input q:[B, T, Dq]; k:[B, S, Dk]; v:[B, S, Dv]
  q_proj: [B, T, N, H] = einsum('BTD,DNH->BTNH', x, Wq)
  k_proj: [B, S, N, H] = einsum('BSD,DNH->BSNH', x, Wk)
  v_proj: [B, S, N, H] = einsum('BSD,DNH->BSNH', x, Wv)
  logits: [B, N, T, S] = einsum('BTNH,BSNH->BNTS', q_proj, k_proj) / sqrt(H)
  probs:  [B, N, T, S] = softmax(logits, axis=-1)
  context:[B, T, N, H] = einsum('BNTS,BSNH->BTNH', probs, v_proj)
  output: [B, T, Dq]   = einsum('BTNH,DNH>BTD', context, Wout)

  Attributes:
    quantization: Information related to the quantization applied to this layer,
      such as dtype for the quantized weight.
  """

  quantization: QuantizationHParams = sub_config_field(QuantizationHParams)

  def create_tensor_quantizers(self):
    self.create_child(
        'act_quantizer',
        quantizer.create_tensor_quantizer(
            'aqt_quantizer', self.quantization.act_params
        ),
    )

  def _do_static_activation_quantization(self) -> bool:
    act_params = self.quantization.act_params
    return act_params is not None and act_params.stats_config is not None

  def setup(self) -> None:
    super().setup()
    if self.quantization.quantization_type == QuantizationType.AQT:
      self.create_tensor_quantizers()
    else:
      raise NotImplementedError(
          'Only AQT-style quantization is implemented for DotProductAttention'
      )
    if self._do_static_activation_quantization():
      raise NotImplementedError(
          'Static activation quantization is not supported yet.'
      )

  def quantized_partition_specs(self) -> Any:
    """Get quantized PartitionSpec.

    Returns:
      A map from names to partition spec.
    """
    partitionspec = {}
    # Activation variable partitioning is only needed for static quantization.
    if self._do_static_activation_quantization():
      raise NotImplementedError(
          'Static activation quantization is not supported yet.'
      )
    return {base_layer.PARAMS: partitionspec}

  def _atten_logits(self, query: JTensor, key: JTensor) -> JTensor:
    """Compute logits from query and key."""
    logits = operations.aqt_einsum(
        eqn='BTNH,BSNH->BNTS',
        lhs=query,
        rhs=key,
        lhs_quantizer=self.act_quantizer,
        rhs_quantizer=self.act_quantizer,
    )
    return logits

  def _dot_atten(
      self,
      query: JTensor,
      key: JTensor,
      value: JTensor,
      atten_mask: JTensor,
      relative_bias: Optional[JTensor] = None,
  ) -> Tuple[JTensor, JTensor]:
    """Main attention function.

    Args:
      query: JTensor of shape [B, T, N, H].
      key: JTensor of shape [B, S, N, H].
      value: JTensor of shape [B, S, N, H].
      atten_mask: JTensor of shape [1|B, 1, 1|T, S] which is a mask that is
        applied to prevent attention between unwanted pairs. This has already
        been converted into large negative logits. Note that the first and third
        dimension allow size 1 if the mask is shared by every item in the batch
        or every token in the target sequence.
      relative_bias: Relative bias of shape [B, N, T, S].

    Returns:
      encoded: JTensor of shape [B, T, N, H].
      atten_probs: JTensor of shape [B, N, T, S].
    """
    query = self._shard_blnh(query)
    key = self._shard_blnh(key)
    value = self._shard_blnh(value)

    b, s, n, h = key.shape
    base_layer.assert_has_shape(value, [b, s, n, h])
    base_layer.assert_has_shape(query, [b, -1, n, h])
    t = query.shape[1]
    # If only padding bias is supplied, then atten_mask can be [B, 1, 1, S]
    # since each target token is prohibited from attending to the same set of
    # source tokens. In this case tiling is inefficient and unnecessary.
    # If there is no padding mask, and only causal mask then the shape can be
    # [1, 1, T, S]
    base_layer.assert_has_shape(atten_mask, [-1, 1, -1, s])
    assert atten_mask.shape[2] in [1, t]
    assert atten_mask.shape[0] in [b, 1]

    query = self._scale_query(query)
    logits = self._atten_logits(query, key)
    if relative_bias is not None:
      # The relative_bias has shape [1, n, t, s] or [b, n, t, s].
      base_layer.assert_has_shape(relative_bias, [-1, n, t, s])
      logits += relative_bias
    logits = checkpoint_name(logits, 'logits')
    self.add_summary(
        'max_logit_precap',
        jnp.max(logits + atten_mask.astype(jnp.float32)),
        verbosity=4,
    )
    logits = self._cap_logits(logits)
    # Attention softmax is always carried out in fp32.
    logits = logits.astype(jnp.float32)
    # Apply attention masking
    padded_logits = logits + atten_mask.astype(jnp.float32)
    if self.attention_mask_summary:
      self.add_summary('attention_mask', atten_mask)
    if self.attention_extra_logit is None:
      probs = jax.nn.softmax(padded_logits, axis=-1).astype(key.dtype)
    else:
      probs = jnp.exp(self._log_softmax_with_extra_logit(padded_logits)).astype(
          key.dtype
      )
    # Apply attention dropout.
    probs = self.atten_dropout(probs)
    # Compute the attention context.
    encoded = operations.aqt_einsum(
        eqn='BNTS,BSNH->BTNH',
        lhs=probs,
        rhs=value,
        lhs_quantizer=self.act_quantizer,
        rhs_quantizer=self.act_quantizer,
    )

    if self.zero_fully_masked:
      # Return zeros for tokens which don't attend anything.
      fully_masked = jnp.all(
          atten_mask < py_utils.get_large_negative_number(jnp.float32) / 2,
          axis=-1,
      )[:, 0, :, jnp.newaxis, jnp.newaxis]
      encoded *= 1 - fully_masked
    encoded = checkpoint_name(encoded, 'context')
    encoded = self._shard_blnh(encoded)
    return encoded, probs

  def _dot_atten_one_step(
      self,
      query: JTensor,
      key_state_name: str,
      value_state_name: str,
      atten_mask: JTensor,
      relative_bias: Optional[JTensor] = None,
      time_step: Optional[JTensor] = None,
  ) -> JTensor:
    """Dot attention function for queries with 1 time step.

    Args:
      query: JTensor of shape [B, N, H].
      key_state_name: Name of the decoding key state variable.
      value_state_name: Name of the decoding value state variable.
      atten_mask: JTensor of shape [1|B, 1, S] which is a mask that is applied
        to prevent attention between unwanted pairs. This has already been
        converted into large negative logits. The first dimension is allowed to
        be of size 1, if the mask is shared by all items in the batch (e.g.,
        only a causal mask).
      relative_bias: Relative bias of shape [1|B, N, 1, S].
      time_step: A scalar. The time step tensor.

    Returns:
      encoded: JTensor of shape [B, N, H].
      probs: JTensor of shape [B, N, S].
    """
    del time_step
    key = self._shard_blnh(self.get_decode_state(key_state_name))
    value = self._shard_blnh(self.get_decode_state(value_state_name))
    k_b = key.shape[0]
    q_b = query.shape[0]
    if q_b != k_b:
      if q_b % k_b != 0:
        raise ValueError(
            f'q batch size {q_b} is not divisible by state batch size {k_b}'
        )
      key = jnp.repeat(key, q_b // k_b, axis=0)
      value = jnp.repeat(value, q_b // k_b, axis=0)
    if atten_mask.shape[0] != q_b and atten_mask.shape[0] != 1:
      assert atten_mask.shape[0] == k_b, (atten_mask.shape, k_b)
      atten_mask = jnp.repeat(atten_mask, q_b // k_b, axis=0)
    # query is 3d.
    query = self._shard_bnh(query)

    b, s, n, h = key.shape
    base_layer.assert_has_shape(value, [b, s, n, h])
    base_layer.assert_has_shape(query, [b, n, h])
    base_layer.assert_has_shape(atten_mask, [-1, 1, s])
    assert atten_mask.shape[0] in [b, 1]
    query = self._scale_query(query)
    logits = operations.aqt_einsum(
        eqn='BNH,BSNH->BNS',
        lhs=query,
        rhs=key,
        lhs_quantizer=self.act_quantizer,
        rhs_quantizer=self.act_quantizer,
    )
    if relative_bias is not None:
      base_layer.assert_has_shape(relative_bias, [-1, n, 1, s])
      assert relative_bias.shape[0] in [b, 1]
      relative_bias = jnp.squeeze(relative_bias, axis=2)
      logits += relative_bias
    logits = self._cap_logits(logits)
    # Attention softmax is always carried out in fp32.
    logits = logits.astype(jnp.float32)
    # Apply attention masking
    padded_logits = logits + atten_mask.astype(jnp.float32)
    # Of shape [b, n, s]
    if self.attention_extra_logit is None:
      probs = jax.nn.softmax(padded_logits, axis=-1).astype(key.dtype)
    else:
      probs = jnp.exp(self._log_softmax_with_extra_logit(padded_logits)).astype(
          key.dtype
      )
    # Compute the attention context.
    encoded = operations.aqt_einsum(
        eqn='BNS,BSNH->BNH',
        lhs=probs,
        rhs=value,
        lhs_quantizer=self.act_quantizer,
        rhs_quantizer=self.act_quantizer,
    )
    encoded = self._shard_bnh(encoded)
    return encoded, probs  # pytype: disable=bad-return-type  # jax-ndarray
