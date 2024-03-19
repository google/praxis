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

"""Quantized and optionally sparsified Linear Layers."""

import copy
import math
from typing import Any, Sequence, Tuple

import fiddle as fdl
from jax import numpy as jnp
from praxis import base_layer
from praxis import pax_fiddle
from praxis import pytypes
from praxis.layers import linears
from praxis.layers import normalizations
from praxis.layers.quantization import operations
from praxis.layers.quantization import quantization_hparams
from praxis.layers.quantization import quantizer
from praxis.layers.quantization import utils
from praxis.layers.quantization.sparsity import sparsifier

QuantizationMode = quantization_hparams.QuantizationMode
QuantizationType = quantization_hparams.QuantizationType
QuantizationParams = quantization_hparams.QuantizationParams
WeightHParams = base_layer.WeightHParams
JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor
WeightInit = base_layer.WeightInit
LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]

instance_field = base_layer.instance_field
template_field = base_layer.template_field


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
  rank: int = -1
  _PACK_4BIT_DIM = 0

  def _get_sub_channel_shape(
      self, shape: Sequence[int], block_size: int, contract_dim: int
  ) -> Sequence[int]:
    """Converts a shape's contract dim into sub-channel and block_size.

    For activation, by => bsc
    For weight, yz => scz

    Args:
      shape: Tensor shape.
      block_size: Block size, it defines number of sub-channels.
      contract_dim: Contraction dim.

    Returns:
      New shape with sub channels and block_size.
    """
    weight_shape, _ = operations.get_sub_channel_shape(
        shape, block_size, [contract_dim]
    )
    return weight_shape

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
      self,
      using_sub_channel: bool,
  ) -> Tuple[WeightHParams, WeightHParams]:
    """Determines shard-aware weight params.

    Args:
      using_sub_channel: If False, the weight is sharded along the contract_dim,
        and the scale is replicated. If True, both the weight and the scale are
        sharded along the sub-channel dimension.

    Returns:
      Tuple with weight and scale params.
    """
    wp = self.weight_split_dims_mapping
    weight_shape = [self.input_dims, self.output_dims]
    scale_shape = [self.output_dims]
    block_size = self._sub_channel_block_size()
    if using_sub_channel:
      weight_shape = self._get_sub_channel_shape(weight_shape, block_size, 0)
      scale_shape = [weight_shape[0], weight_shape[2]]
      if wp.wt is not None:
        weight_sharding = wp.wt.copy()
        weight_sharding.insert(1, None)
        scale_sharding = wp.wt.copy()
      else:
        weight_sharding = None
        scale_sharding = None
    else:
      weight_sharding = wp.wt
      if wp.wt is not None and len(wp.wt) > 1:
        scale_sharding = [wp.wt[1]]
      else:
        scale_sharding = None
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
      )
      self.create_sparsity_variables(
          'w_a',
          wp_a,
          scale_shape=[self.rank],
      )
      wp_b = WeightHParams(
          shape=shape_b,
          mesh_shape=self.mesh_shape,
          tensor_split_dims_mapping=wp.wt,
      )
      self.set_up_weights(
          weight_name='w_b',
          weight_params=wp_b,
          scale_shape=[self.output_dims],
      )
      self.create_sparsity_variables(
          'w_b',
          wp_b,
          scale_shape=[self.output_dims],
      )

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
          scale_hparams=scale_hparams,
      )
      self.create_sparsity_variables(
          'w',
          weight_hparams,
          scale_shape=[self.output_dims],
      )

  def __call__(self, inputs: JTensor) -> JTensor:
    """Apply projection to inputs.

    Args:
      inputs: The inputs JTensor.  Shaped [..., input_dims].

    Returns:
      Projected inputs.
    """

    ap = self.activation_split_dims_mapping
    q_einsum_params = {
        'eqn': '...y,yz->...z',
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
        if self.quantization.act_params is not None:
          q_einsum_params['eqn'] = '...sc,scz->...sz'
          q_einsum_params['scale_eqn'] = '...sz,sz->...z'
          q_einsum_params['zp_eqn'] = '...sc,sz->...z'
          q_einsum_params['swap_xw'] = False
        else:
          q_einsum_params['eqn'] = 'scz,...sc->...sz'
          q_einsum_params['scale_eqn'] = '...sz,sz->...z'
          q_einsum_params['zp_eqn'] = '...sc,sz->...z'
          q_einsum_params['swap_xw'] = True
        if len(w.shape) == 2:
          q_einsum_params['reshape'] = self._get_sub_channel_shape(
              list(w.shape), block_size, 0
          )
      out = self.quantized_einsum(
          x=inputs,
          w=w,
          **q_einsum_params,
      )
    # Adjust sharding annotation during decoding.
    # TODO(pax): This logic should likely be lifted somewhere else.
    ap_out = ap.out
    if out.ndim == 2:
      if (
          hasattr(ap, 'extend_step_out')
          and ap.extend_step_out is not None
          and len(ap.extend_step_out) == 2
      ):
        ap_out = ap.extend_step_out
      elif ap_out is not None and len(ap_out) == 3:
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
    block_size = self._sub_channel_block_size()
    weight_hparams, scale_hparams = self._get_weight_hparams(block_size > 0)
    # TODO(pax): This is a weird way to enforce scale replication.
    # We should fix related tests to use replicated scale for large models.
    if block_size == 0:
      scale_hparams.shape = ()
      scale_hparams.mesh_shape = None
    weight_pspec = base_layer._weight_hparam_to_pspec(  # pylint: disable=protected-access
        weight_hparams, self.mesh_axis_names
    )
    scale_pspec = base_layer._weight_hparam_to_pspec(  # pylint: disable=protected-access
        scale_hparams, self.mesh_axis_names
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
    scale_name = 'w' + base_layer.QUANTIZED_SCALE_NAME_POSTFIX

    w = self.theta.w
    contract_dims = [0]
    block_size = self._sub_channel_block_size()
    if block_size > 0:
      w = jnp.reshape(
          w, self._get_sub_channel_shape(list(w.shape), block_size, 0)
      )
      contract_dims = [1]
    calculation_dtype = self.dtype

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
        if w.dtype != calculation_dtype:
          w = w.astype(calculation_dtype)
        q_w, q_s, zp = operations.reduce_precision(
            w,
            contract_dims,
            bits=self.quantization.weight_params.precision,
            percentile=self.quantization.weight_params.clipping_coeff,
            use_symmetric=self.quantization.weight_params.use_symmetric,
            quant_method=self.quantization.weight_params.quant_method,
        )
        q_s = jnp.squeeze(q_s)
        if zp is not None:
          zp = jnp.squeeze(zp)
    # Internal quantization type support.
    elif self.quantization.quantization_type == QuantizationType.AQT:
      if self._do_static_activation_quantization():
        raise NotImplementedError(
            'Static activation quantization is not supported yet.'
        )
      else:
        q_w, q_s, zp = self.weight_quantizer.quantize(
            w,
            contract_dims,
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


class LinearLoRA(Linear):
  """Linear layer with residual LoRA.

  Attributes:
    lora_rank: Rank of LoRA.
    init_method: LoRA weights initialization method.
    norm_tpl: Normalization layer type.
    norm_order: Where to apply normalization layer:
      * None: no normalization. * 'pre': normalization before LoRA projections.
        * 'mid': normalization between LoRA projections. * 'post': normalization
        after LoRA projections.
  """

  lora_rank: int = 0
  init_method: str = 'one_zero'
  norm_tpl: LayerTpl = template_field(normalizations.LayerNorm)
  norm_order: str | None = None

  def setup(self):
    super().setup()

    eqn = '...y,yz->...z'
    weight_shape = [self.input_dims, self.output_dims]
    total_size_right = self.output_dims
    total_size_left = self.lora_rank

    if self.init_method == 'one_zero':
      w_left_scale = 1.0
      w_right_scale = 0.0
    elif self.init_method == 'output_dim':
      w_left_scale = 1.0 / math.sqrt(total_size_left)
      w_right_scale = 1.0 / math.sqrt(total_size_right)
    else:
      raise ValueError(f'Unrecognized init_method: {self.init_method}')

    (
        self.eqn_left,
        self.eqn_right,
        left_shape,
        right_shape,
        eqn_left_ind,
        eqn_right_ind,
    ) = utils.get_lora_shape_and_eqn(weight_shape, self.lora_rank, eqn)

    wp = self.weight_split_dims_mapping
    self.create_variable(
        'w_left',
        WeightHParams(
            shape=left_shape,
            mesh_shape=self.mesh_shape,
            tensor_split_dims_mapping=utils.get_left_weight_split_dims_mapping(
                wp, eqn_left_ind
            ),
            init=WeightInit.Gaussian(w_left_scale),
        ),
    )
    self.create_variable(
        'w_right',
        WeightHParams(
            shape=right_shape,
            mesh_shape=self.mesh_shape,
            tensor_split_dims_mapping=utils.get_right_weight_split_dims_mapping(
                wp, eqn_right_ind
            ),
            init=WeightInit.Constant(w_right_scale)
            if w_right_scale == 0.0
            else WeightInit.Gaussian(w_left_scale),
        ),
    )

    if self.norm_order is not None:
      norm_tpl = self.norm_tpl.clone()
      if fdl.get_callable(norm_tpl) not in {
          normalizations.BatchNorm,
          normalizations.GroupNorm,
          normalizations.LayerNorm,
      }:
        raise NotImplementedError(
            '%s is not supported' % fdl.get_callable(norm_tpl)
        )
      if self.norm_order == 'pre':
        norm_tpl.dim = self.input_dims
      elif self.norm_order == 'mid':
        norm_tpl.dim = self.lora_rank
      elif self.norm_order == 'post':
        norm_tpl.dim = self.output_dims
      else:
        raise ValueError(f'Unrecognized norm_order: {self.norm_order}')

      self.create_child('norm', norm_tpl)

  def __call__(self, inputs: JTensor) -> JTensor:
    """Apply projection to inputs.

    Args:
      inputs: The inputs JTensor.  Shaped [..., input_dims].

    Returns:
      Projected inputs.
    """
    ap = self.activation_split_dims_mapping
    out = super().__call__(inputs)

    if self.lora_rank:
      lora_output = inputs
      if self.norm_order == 'pre':
        lora_output = self.norm(lora_output)
      lora_output = jnp.einsum(self.eqn_left, lora_output, self.theta.w_left)
      if self.norm_order == 'mid':
        lora_output = self.norm(lora_output)
      lora_output = jnp.einsum(self.eqn_right, lora_output, self.theta.w_right)
      if self.norm_order == 'post':
        lora_output = self.norm(lora_output)
      out += lora_output

    # Adjust sharding annotation during decoding.
    ap_out = ap.out
    if ap_out is not None and len(ap_out) == 3 and out.ndim == 2:
      ap_out = [ap_out[0], ap_out[2]]
    out = base_layer.maybe_shard(out, ap_out, self.mesh_axis_names)
    return out


class LinearActScaling(Linear):
  """Linear layer with extra Activation Scaling."""

  def setup(self):
    super().setup()
    self.create_variable(
        'w_per_channel_act_max',
        WeightHParams(
            shape=[self.input_dims],
            init=WeightInit.Constant(0.0),
        ),
    )

  def __call__(self, inputs: JTensor) -> JTensor:
    """Apply Activation Scaling to inputs.

    Args:
      inputs: The inputs JTensor.  Shaped [..., input_dims].

    Returns:
      Projected inputs.
    """
    ap = self.activation_split_dims_mapping
    inputs = inputs * self.theta.w_per_channel_act_max
    out = super().__call__(inputs)
    ap_out = ap.out
    if ap_out is not None and len(ap_out) == 3 and out.ndim == 2:
      ap_out = [ap_out[0], ap_out[2]]
    out = base_layer.maybe_shard(out, ap_out, self.mesh_axis_names)
    return out
