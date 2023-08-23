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

"""Sparse Attention Layers."""
# pytype: disable=signature-mismatch

import string

from jax import numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from praxis import base_layer
from praxis import pytypes
from praxis.layers import attentions
from praxis.layers.sparsity import sparse_base_layer
from praxis.layers.sparsity import sparsity_hparams

SparsityHParams = sparsity_hparams.SparsityHParams
SparsityMode = sparsity_hparams.SparsityMode
SparsityType = sparsity_hparams.SparsityType
WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams
instance_field = base_layer.instance_field
JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor


class AttentionProjection(sparse_base_layer.SparsityBaseLayer,
                          attentions.AttentionProjection):
  """Layer that computes quantized multi heads projection.

  This layer is expected to be used within DotProductAttention.

  Attributes:
    sparsity: Information related to the sparsity applied to this
      layer.
  """
  sparsity: SparsityHParams = instance_field(SparsityHParams)

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
        if self.is_output_projection and self.use_nhd_shape:
          h_sharding = ()
          for axes in (wp.wt[0], wp.wt[1]):
            if isinstance(axes, (str, int)):
              h_sharding += (axes,)
            elif axes is not None:
              h_sharding += tuple(axes)
          wt = [h_sharding, wp.wt[2]]
        else:
          h_sharding = ()
          for axes in (wp.wt[1], wp.wt[2]):
            if isinstance(axes, (str, int)):
              h_sharding += (axes,)
            elif axes is not None:
              h_sharding += tuple(axes)
          wt = [wp.wt[0], h_sharding]
      assert len(wt) == 2
    else:
      wt = wp.wt

    if self.is_output_projection and self.use_nhd_shape:
      pc_shape = hd_shape + [self.input_dim]
      if self.attention_combine_dims:
        fan_in_axes, fan_out_axes = [-1], [-2]
      else:
        fan_in_axes, fan_out_axes = [-1], [-2, -3]
    else:
      pc_shape = [self.input_dim] + hd_shape
      if self.attention_combine_dims:
        fan_in_axes, fan_out_axes = [-2], [-1]
      else:
        fan_in_axes, fan_out_axes = [-3], [-1, -2]

    weight_hp = WeightHParams(
        shape=pc_shape,
        mesh_shape=self.mesh_shape,
        tensor_split_dims_mapping=wt,
        fan_in_axes=(fan_in_axes
                     if self.explicit_fan_in_fan_out_axes else None),
        fan_out_axes=(fan_out_axes
                      if self.explicit_fan_in_fan_out_axes else None),
    )
    name = 'w'
    self.create_variable(name, weight_hp)
    self.create_child('einsum', self.einsum_tpl.clone())
    self.create_aux_variables(name, weight_hp)

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

    w = self.sparsifiy(w, inputs=inputs, name='w')  # sparsify weight.
    ret = self.einsum(eqn, inputs, w)
    if self.use_bias:
      ret += theta.b
    return ret


class CombinedQKVProjectionLayer(sparse_base_layer.SparsityBaseLayer,
                                 attentions.CombinedQKVProjectionLayer):
  """Layer that computes quantized QKV projection with a combined weight.

  This layer is expected to be used within DotProductAttention below.

  Attributes:
    sparsity: Information related to the sparsity applied to this
      layer.
  """
  sparsity: SparsityHParams = instance_field(SparsityHParams)

  def setup(self) -> None:
    # Sharding has the same convention of AttentionProjection, which doesn't
    # contain the leading stacking dimension.
    wt = self.weight_split_dims_mapping.wt
    if wt is not None:
      assert isinstance(wt, (list, tuple))
      if self.attention_combine_dims:
        if len(wt) == 3:
          hd_sharding = ()
          for s in wt[1:]:
            if isinstance(s, (list, tuple)):
              hd_sharding += tuple(s)
            elif s is not None:
              hd_sharding += (s,)
          wt = [wt[0], hd_sharding]
        else:
          assert len(wt) == 2
      else:
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
      fan_in_axes, fan_out_axes = [-2], [-1]
    else:
      hd_shape = [self.num_heads, self.dim_per_head]
      fan_in_axes, fan_out_axes = [-3], [-1, -2]

    pc_shape = [3, self.input_dim] + hd_shape
    # Combined weight for q, k, v projections.
    weight_hp = WeightHParams(
        shape=pc_shape,
        init=self.params_init,
        dtype=self.dtype,
        mesh_shape=self.mesh_shape,
        tensor_split_dims_mapping=weight_split_dims_mapping,
        fan_in_axes=(fan_in_axes
                     if self.explicit_fan_in_fan_out_axes else None),
        fan_out_axes=(fan_out_axes
                      if self.explicit_fan_in_fan_out_axes else None),
    )
    name = 'w'
    self.create_variable(name, weight_hp)
    self.create_child('einsum', self.einsum_tpl.clone())
    self.create_aux_variables(name, weight_hp)
    if self.use_bias:
      # Combined bias weight for q, k, v projections.
      pc_bias = WeightHParams(
          shape=[3] + hd_shape,
          init=WeightInit.Constant(0.0),
          mesh_shape=self.mesh_shape,
          tensor_split_dims_mapping=bias_split_dims_mapping,
      )
      self.create_variable('b', pc_bias)

  # TODO(zhangqiaorjc): Take query, key, value as inputs to support all
  # attentions.
  def __call__(self, inputs: JTensor) -> tuple[JTensor, JTensor, JTensor]:
    """Computes the QKV projection for inputs.

    Args:
      inputs: A JTensor of shape [..., p.input_dim].

    Returns:
      The three projected JTensor with shape [..., num_heads, dim_per_head]
      in q_proj, k_proj and v_proj order.
    """
    theta = self.theta

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
    w = self.sparsifiy(w, inputs=inputs, name='w')  # sparsify weight.
    ret = self.einsum(eqn, inputs, w)
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
