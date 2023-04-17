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

"""Quantized Embedding and softmax layers."""

import copy
from typing import Any

import jax
from jax import numpy as jnp
from praxis import base_layer
from praxis import pax_fiddle
from praxis import pytypes
from praxis.layers import activations
from praxis.layers import embedding_softmax
from praxis.layers import linears
from praxis.layers.quantization import linears as quantized_linears
from praxis.layers.quantization import operations as quantized_operations
from praxis.layers.quantization import quantization_hparams

QuantizationMode = quantization_hparams.QuantizationMode
QuantizationType = quantization_hparams.QuantizationType
QuantizationHParams = quantization_hparams.QuantizationHParams

WeightHParams = base_layer.WeightHParams
JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor
SplitDimsMapping = pytypes.SplitDimsMapping
LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]

sub_config_field = base_layer.sub_config_field
template_field = base_layer.template_field


class Embedding(embedding_softmax.Embedding):
  """Quantized Embedding layer."""

  quantization: QuantizationHParams = sub_config_field(QuantizationHParams)

  def setup(self) -> None:
    wp = self.weight_split_dims_mapping
    pc = WeightHParams(
        shape=[self.num_classes, self.input_dims],
        mesh_shape=self.mesh_shape,
        tensor_split_dims_mapping=wp.wt,
    )
    if self.quantization.mode == QuantizationMode.INFERENCE:
      self.create_quantized_variable(
          'emb_var',
          pc,
          [self.num_classes],
          dtype=jnp.int8,
          use_symmetric=self.quantization.weight_params.use_symmetric,
      )
    else:
      self.create_variable('emb_var', pc)

  def emb_lookup(self, ids: JTensor) -> JTensor:
    ap = self.activation_split_dims_mapping

    if self.quantization.mode == QuantizationMode.INFERENCE:
      emb_var, scale_var, zp_var = self.get_quantized_weight(
          'emb_var', use_symmetric=self.quantization.weight_params.use_symmetric
      )
    else:
      emb_var = self.theta.emb_var

    if self.lookup_style == 'index':
      embs = jnp.asarray(emb_var)[(ids,)]
    elif self.lookup_style == 'matmul':
      # Explicit casting to fprop_dtype needed for bf16.
      one_hot_ids = jax.nn.one_hot(
          ids, self.num_classes, dtype=self.fprop_dtype
      )
      embs = linears.project_last_dim(one_hot_ids, emb_var)
    else:
      raise ValueError('Unknown lookup style.')

    if self.quantization.mode == QuantizationMode.INFERENCE:
      scale = jnp.expand_dims(scale_var[(ids,)], axis=2)
      embs = jnp.multiply(embs, scale)
      if not self.quantization.weight_params.use_symmetric:
        zp = jnp.expand_dims(zp_var[(ids,)], axis=2)
        embs = embs - zp

    # map out-of-boundary ids to nan for easier debug
    if self.set_nan_for_oob_id:
      embs = jnp.where(ids[..., jnp.newaxis] < self.num_classes, embs, jnp.nan)

    if self.scale_sqrt_depth:
      embs *= self.input_dims**0.5

    embs = base_layer.maybe_shard(
        embs, ap.emb_out_split_dims_mapping, self.mesh_axis_names
    )
    return embs

  def quantized_partition_specs(self) -> Any:
    """Get quantized PartitionSpec.

    Returns:
      a map from names to partition spec.
    """
    scale_name = 'emb_var' + base_layer.QUANTIZED_SCALE_NAME_POSTFIX
    wp = self.weight_split_dims_mapping
    weight_pspec = base_layer._weight_hparam_to_pspec(  # pylint: disable=protected-access
        self._weight_hparams['emb_var'], self.mesh_axis_names
    )
    scale_split_dims_mapping = [wp.wt[0]] if wp.wt is not None else None
    # scale_weight_hparam is unmaterialized so shape is irrelevant.
    scale_weight_hparam = WeightHParams(
        shape=(), tensor_split_dims_mapping=scale_split_dims_mapping
    )
    scale_pspec = base_layer._weight_hparam_to_pspec(  # pylint: disable=protected-access
        scale_weight_hparam, self.mesh_axis_names
    )
    partitionspec = {'emb_var': weight_pspec, scale_name: scale_pspec}

    if not self.quantization.weight_params.use_symmetric:
      zp_name = 'emb_var' + base_layer.QUANTIZED_ZP_NAME_POSTFIX
      partitionspec[zp_name] = copy.deepcopy(scale_pspec)

    res = {base_layer.PARAMS: partitionspec}
    return res

  def quantize_weight(self) -> NestedJTensor:
    """Get quantized weight, where w is transposed and class-dim (dim 0) wise quantized.

    Returns:
      a map from names to quantized weights.
    """
    scale_name = 'emb_var' + base_layer.QUANTIZED_SCALE_NAME_POSTFIX
    eqn = 'xy,zy->xz'
    bits = self.quantization.weight_params.precision
    percentile = self.quantization.weight_params.clipping_coeff

    q_w, q_s, zp = quantized_operations.reduce_einsum_weight_precision(
        eqn,
        self.theta.emb_var,
        calculation_type=self.dtype,
        bits=bits,
        percentile=percentile,
        use_symmetric=self.quantization.weight_params.use_symmetric,
    )

    if self.quantization.weight_params.use_symmetric:
      res = {base_layer.PARAMS: {'emb_var': q_w, scale_name: q_s}}
    else:
      zp_name = 'emb_var' + base_layer.QUANTIZED_ZP_NAME_POSTFIX
      res = {base_layer.PARAMS: {'emb_var': q_w, scale_name: q_s, zp_name: zp}}
    return res


class SharedEmbeddingSoftmax(embedding_softmax.SharedEmbeddingSoftmax):
  """A softmax layer that also supports embedding lookups.

  SharedEmbeddingSoftmax class which is extended by this class's get_logits()
  uses its feed forward layer. As this quantized SharedEmbeddingSoftmax requires
  its feed forward layer's Linear to be quantized Linear (See setup()), explicit
  overriding for quantized get_logits() in this extended class is not required.

  Attributes:
    lookup_style: Style of lookup, one of index or matmul.
    scale_sqrt_depth: If set True, activations are scaled with
      sqrt(embedding_dim) in emb_lookup.
    quantization: Information related to the quantization applied to this layer,
      such as dtype for the quantized weight.
  """

  quantization: QuantizationHParams = sub_config_field(QuantizationHParams)

  def setup(self) -> None:
    if self.feed_forward_tpl is not None:
      wp = self.weight_split_dims_mapping
      ap = self.activation_split_dims_mapping
      ff_p = self.feed_forward_tpl.clone().set(
          input_dims=self.input_dims,
          output_dims=self.num_classes,
          activation_tpl=pax_fiddle.Config(activations.Identity),
          bias_init=self.bias_init,
          weight_split_dims_mapping=wp.clone(),
          activation_split_dims_mapping=ap.clone(),
      )
      new_linear_tpl = pax_fiddle.Config(
          quantized_linears.Linear,
          quantization=copy.deepcopy(self.quantization),
      )
      new_linear_tpl.copy_fields_from(ff_p.linear_tpl)
      ff_p.linear_tpl = new_linear_tpl
      self.create_child('logits_ffn', ff_p)
    if self.bi_tempered_loss_tpl:
      self.create_child('bi_tempered_loss', self.bi_tempered_loss_tpl)

  def emb_lookup(self, ids: JTensor) -> JTensor:
    linear_layer = self.logits_ffn.linear

    ap = self.activation_split_dims_mapping

    if self.quantization.mode == QuantizationMode.INFERENCE:
      emb_var, scale_var, zp_var = linear_layer.get_quantized_weight(
          'w', use_symmetric=self.quantization.weight_params.use_symmetric
      )
    else:
      emb_var = linear_layer.theta.w

    emb_var = jnp.transpose(emb_var)
    if self.lookup_style == 'index':
      embs = jnp.asarray(emb_var)[(ids,)]
    elif self.lookup_style == 'matmul':
      # Explicit casting to fprop_dtype needed for bf16.
      one_hot_ids = jax.nn.one_hot(
          ids, self.num_classes, dtype=self.fprop_dtype
      )
      eqn = '...y,yz->...z'
      embs = jnp.einsum(eqn, one_hot_ids, emb_var)
    else:
      raise ValueError('Unknown lookup style.')

    if self.quantization.mode == QuantizationMode.INFERENCE:
      scale = jnp.expand_dims(scale_var[(ids,)], axis=2)
      embs = jnp.multiply(embs, scale)
      if not self.quantization.weight_params.use_symmetric:
        zp = jnp.expand_dims(zp_var[(ids,)], axis=2)
        embs = embs - zp

    # Scale with sqrt(embedding dims)
    if self.scale_sqrt_depth:
      embs *= self.input_dims**0.5

    embs = base_layer.maybe_shard(
        embs, ap.emb_out_split_dims_mapping, self.mesh_axis_names
    )
    return embs


class NClassMajorSharedEmbeddingSoftmax(
    embedding_softmax.SharedEmbeddingSoftmax
):
  """A softmax layer that also supports embedding lookups.

  The embedding table is constructed with num_classes as the major dimension.

  Attributes:
    activation_tpl: Sub configurable field for the activation layer.
    quantization: Information related to the quantization applied to this layer,
      such as dtype for the quantized weight.
  """

  activation_tpl: pax_fiddle.Config[activations.BaseActivation] = (
      template_field(activations.Identity)
  )
  quantization: QuantizationHParams = sub_config_field(QuantizationHParams)

  def setup(self) -> None:
    wp = self.weight_split_dims_mapping
    ap = self.activation_split_dims_mapping
    if self.quantization.mode == QuantizationMode.INFERENCE:
      pc = WeightHParams(
          shape=[self.num_classes, self.input_dims],
          mesh_shape=self.mesh_shape,
          tensor_split_dims_mapping=[wp.wt[1], wp.wt[0]]
          if wp.wt is not None
          else None,
      )
      self.create_quantized_variable(
          'w',
          pc,
          [self.num_classes],
          dtype=jnp.int8,
          use_symmetric=self.quantization.weight_params.use_symmetric,
      )
    elif self.quantization.mode == QuantizationMode.MATERIALIZE:
      pc = WeightHParams(
          shape=[self.input_dims, self.num_classes],
          mesh_shape=self.mesh_shape,
          tensor_split_dims_mapping=wp.wt,
      )
      self.create_variable('w', pc)
    else:
      raise ValueError(
          f'{self.quantization.mode} not supported in quantized'
          ' NClassMajorSharedEmbeddingSoftmax.'
      )
    bias_layer_p = pax_fiddle.Config(
        linears.Bias, dims=self.num_classes, bias_init=self.bias_init
    )
    if self.mesh_shape is not None and ap.out is not None:
      wp_bias = [ap.out[-1]]
      bias_layer_p.weight_split_dims_mapping.wt = wp_bias
    self.create_child('bias', bias_layer_p)
    self.create_child('activation', self.activation_tpl.clone())
    if self.bi_tempered_loss_tpl:
      self.create_child('bi_tempered_loss', self.bi_tempered_loss_tpl)

  def get_logits(self, inputs: JTensor) -> JTensor:
    """Returns logits given the inputs with an option to soft cap it.

    Args:
      inputs: a single JTensor with shape [..., input_dim].

    Returns:
      logits: with shape [..., num_classes]. Unnormalized softmax's logits.
    """
    ap = self.activation_split_dims_mapping
    if self.quantization.mode == QuantizationMode.INFERENCE:
      w, s, zp = self.get_quantized_weight(
          'w', use_symmetric=self.quantization.weight_params.use_symmetric
      )
      projected_inputs = quantized_operations.einsum(
          '...y,zy->...z', inputs, w, s, zp
      )
    elif self.quantization.mode == QuantizationMode.MATERIALIZE:
      w = self.theta.w
      projected_inputs = jnp.einsum('...y,yz->...z', inputs, w)
    else:
      raise ValueError(
          f'{self.quantization.mode} not supported in quantized'
          ' NClassMajorSharedEmbeddingSoftmax.'
      )
    ap_out = ap.out
    if ap_out is not None and len(ap_out) == 3 and projected_inputs.ndim == 2:
      ap_out = [ap_out[0], ap_out[2]]
    projected_inputs = base_layer.maybe_shard(
        projected_inputs, ap_out, self.mesh_axis_names
    )
    projected_inputs = self.bias(projected_inputs)
    logits = self.activation(projected_inputs)

    # Soft cap logits if applicable.
    if self.soft_cap_logits:
      logits = self.soft_cap_logits * jnp.tanh(logits / self.soft_cap_logits)
    return logits

  def emb_lookup(self, ids: JTensor) -> JTensor:
    ap = self.activation_split_dims_mapping

    if self.quantization.mode == QuantizationMode.INFERENCE:
      emb_var, scale_var, zp_var = self.get_quantized_weight(
          'w', use_symmetric=self.quantization.weight_params.use_symmetric
      )
    elif self.quantization.mode == QuantizationMode.MATERIALIZE:
      emb_var = jnp.transpose(self.theta.w)
    else:
      raise ValueError(
          f'{self.quantization.mode} not supported in quantized'
          ' NClassMajorSharedEmbeddingSoftmax.'
      )

    if self.lookup_style == 'index':
      embs = jnp.asarray(emb_var)[(ids,)]
    elif self.lookup_style == 'matmul':
      # Explicit casting to fprop_dtype needed for bf16.
      one_hot_ids = jax.nn.one_hot(
          ids, self.num_classes, dtype=self.fprop_dtype
      )
      embs = linears.project_last_dim(one_hot_ids, emb_var)
    else:
      raise ValueError('Unknown lookup style.')

    if self.quantization.mode == QuantizationMode.INFERENCE:
      scale = jnp.expand_dims(scale_var[(ids,)], axis=2)
      embs = jnp.multiply(embs, scale)
      if not self.quantization.weight_params.use_symmetric:
        zp = jnp.expand_dims(zp_var[(ids,)], axis=2)
        embs = embs - zp

    # Scale with sqrt(embedding dims)
    if self.scale_sqrt_depth:
      embs *= self.input_dims**0.5

    embs = base_layer.maybe_shard(
        embs, ap.emb_out_split_dims_mapping, self.mesh_axis_names
    )
    return embs

  def quantized_partition_specs(self) -> Any:
    """Get quantized PartitionSpec.

    Returns:
      a map from names to partition spec.
    """
    scale_name = 'w' + base_layer.QUANTIZED_SCALE_NAME_POSTFIX

    wp = self.weight_split_dims_mapping
    weight_hparams_transposed = copy.deepcopy(self._weight_hparams['w'])
    weight_hparams_transposed.tensor_split_dims_mapping = (
        [wp.wt[1], wp.wt[0]] if wp.wt is not None else None
    )
    weight_pspec = base_layer._weight_hparam_to_pspec(  # pylint: disable=protected-access
        weight_hparams_transposed, self.mesh_axis_names
    )
    # wp.wt is not transposed at this point.
    scale_split_dims_mapping = [wp.wt[1]] if wp.wt is not None else None
    # scale_weight_hparam is unmaterialized so shape is irrelevant.
    scale_weight_hparam = WeightHParams(
        shape=(), tensor_split_dims_mapping=scale_split_dims_mapping
    )
    scale_pspec = base_layer._weight_hparam_to_pspec(  # pylint: disable=protected-access
        scale_weight_hparam, self.mesh_axis_names
    )
    partitionspec = {'w': weight_pspec, scale_name: scale_pspec}

    if not self.quantization.weight_params.use_symmetric:
      zp_name = 'w' + base_layer.QUANTIZED_ZP_NAME_POSTFIX
      partitionspec[zp_name] = copy.deepcopy(scale_pspec)

    res = {base_layer.PARAMS: partitionspec}
    res = self._add_children_params(res, return_pspec=True)
    return res

  def quantize_weight(self) -> NestedJTensor:
    """Get quantized weight, where w is transposed and class-dim (dim 0) wise quantized.

    Returns:
      a map from names to quantized weights.
    """
    scale_name = 'w' + base_layer.QUANTIZED_SCALE_NAME_POSTFIX
    eqn = 'xy,zy->xz'
    bits = self.quantization.weight_params.precision
    percentile = self.quantization.weight_params.clipping_coeff

    w = jnp.transpose(self.theta.w)
    q_w, q_s, zp = quantized_operations.reduce_einsum_weight_precision(
        eqn,
        w,
        calculation_type=self.dtype,
        bits=bits,
        percentile=percentile,
        use_symmetric=self.quantization.weight_params.use_symmetric,
    )

    if self.quantization.weight_params.use_symmetric:
      res = {base_layer.PARAMS: {'w': q_w, scale_name: q_s}}
    else:
      zp_name = 'w' + base_layer.QUANTIZED_ZP_NAME_POSTFIX
      res = {base_layer.PARAMS: {'w': q_w, scale_name: q_s, zp_name: zp}}
    res = self._add_children_params(res, return_pspec=False)
    return res

  def _add_children_params(
      self, params_dict: NestedJTensor, return_pspec: bool
  ) -> NestedJTensor:
    for name, child in self._private_children.items():
      if return_pspec:
        child_res = child.quantized_partition_specs()
      else:
        child_res = child.quantize_weight()
      for child_target in child_res:
        if child_target not in params_dict:
          params_dict[child_target] = {}  # pytype: disable=unsupported-operands
        params_dict[child_target][name] = child_res[child_target]
    return params_dict
