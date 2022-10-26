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

"""Convolutional layers."""

import math
from typing import Optional, Sequence, Tuple

import jax
from jax import numpy as jnp
from praxis import base_layer
from praxis import py_utils
from praxis import pytypes
from praxis.layers import activations
from praxis.layers import linears
from praxis.layers import normalizations
from praxis.layers import stochastics

BaseActShardingHParams = base_layer.BaseLayer.ActivationShardingHParams
BaseWtShardingHParams = base_layer.BaseLayer.WeightShardingHParams

NestedMap = py_utils.NestedMap
SplitDimsMapping = pytypes.SplitDimsMapping
sub_config_field = base_layer.sub_config_field
WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams


JTensor = pytypes.JTensor

BaseHParams = base_layer.BaseLayer.HParams


class Conv2D(base_layer.BaseLayer):
  """Conv2D with support of SAME/VALID paddings."""

  class HParams(BaseHParams):
    """Associated hyperparams for this layer class.

    Attributes:
    filter_shape: Filter shape. Must be a sequence of length 4. Elements are in
      the order of height (time), width (frequency), in_channel, out_channel.
      filter_stride: Filter stride to use. Must be a pair of ints. The first int
        specifies the stride on the height dimension. The second int specifies
        the stride on the width dimension.
      dilations: An optional list of ints. Defaults to (1, 1). 1-D tensor of
        length 2. The dilation factor for each dimension of input. If set to k >
        1, there will be k-1 skipped cells between each filter element on that
        dimension.
      bias: Whether or not to apply a bias before activation.
      bias_init: Bias initializer to use if bias is to be applied.
      padding: The type of padding to use.
      tf_equivalent_padding: Whether to make it equivalent to tf. By default we
        apply extra padding that is different than tf conv when stride > 1. This
        is mainly used for multimodal which leads to better accuracy.
      is_causal: Whether this is a causal convolution. This assumes the first
        dimension of filter is time and if is_causal=True, each position would
        not observe any positions in the right. This is achieved by adding
        extra padding in the left to shift the whole convolution.
    """
    filter_shape: Sequence[int] = (0, 0, 0, 0)
    filter_stride: Sequence[int] = (0, 0)
    dilations: Sequence[int] = (1, 1)
    bias: bool = False
    bias_init: WeightInit = WeightInit.Constant(0.0)
    padding: str = 'SAME'
    tf_equivalent_padding: bool = False
    is_causal: bool = False

  def setup(self) -> None:
    p = self.hparams
    assert p.name
    assert p.padding in ['SAME', 'VALID']
    assert len(p.filter_shape) == 4
    assert len(p.filter_stride) == 2
    assert len(p.dilations) == 2
    assert all(x > 0 for x in p.filter_stride)

    # error if is_causal but not tf_equivalent_padding
    if p.is_causal and (not p.tf_equivalent_padding):
      raise Exception(
          'Causal convolution is only supported for tf equivalent padding')

    # error if is_causal but padding == 'valid'
    if p.is_causal and p.padding == 'VALID':
      raise NotImplementedError(
          'Causal convlution doesn\'t support valid padding')

    wp = p.weight_split_dims_mapping
    self.create_variable(
        'w',
        WeightHParams(
            shape=p.filter_shape,
            mesh_shape=p.mesh_shape,
            tensor_split_dims_mapping=wp.wt))
    if p.bias:
      self.create_variable(
          'b', WeightHParams(shape=[p.filter_shape[-1]], init=p.bias_init))

  def _compute_padding(self, inputs_shape, pad_height_zero=False):
    p = self.hparams
    if not p.tf_equivalent_padding:
      if p.padding == 'SAME':
        pad_height_total = p.filter_shape[0] - 1
        pad_height_beg = pad_height_total // 2
        pad_height_end = pad_height_total - pad_height_beg
        pad_width_total = p.filter_shape[1] - 1
        pad_width_beg = pad_width_total // 2
        pad_width_end = pad_width_total - pad_width_beg
      else:
        assert p.padding == 'VALID', p.padding
        pad_height_beg = 0
        pad_height_end = 0
        pad_width_beg = 0
        pad_width_end = 0
      padding = [(pad_height_beg, pad_height_end),
                 (pad_width_beg, pad_width_end)]
    else:
      if not p.is_causal:
        padding = p.padding
      else:
        # Compute padding for causal convolution
        # Reference:
        # https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2
        filter_height = (p.filter_shape[0] - 1) * p.dilations[0] + 1
        filter_width = (p.filter_shape[1] - 1) * p.dilations[1] + 1
        if inputs_shape[1] % p.filter_stride[0] == 0:
          pad_height_total = max(filter_height - p.filter_stride[0], 0)
        else:
          pad_height_total = max(
              filter_height - (inputs_shape[1] % p.filter_stride[0]), 0)
        if inputs_shape[2] % p.filter_stride[1] == 0:
          pad_width_total = max(filter_width - p.filter_stride[1], 0)
        else:
          pad_width_total = max(
              filter_width - (inputs_shape[2] % p.filter_stride[1]), 0)

        # first dimension is causal
        pad_height_beg = 0 if pad_height_zero else pad_height_total
        pad_height_end = 0
        pad_width_beg = pad_width_total // 2
        pad_width_end = pad_width_total - pad_width_beg

        padding = [(pad_height_beg, pad_height_end),
                   (pad_width_beg, pad_width_end)]
    return padding

  def __call__(self, inputs: JTensor) -> JTensor:
    """FProp that supports strided, dilated convolution, depthwise convolution.

    Args:
      inputs: Input sequence of shape [B, H, W, D_in], also known more popularly
        as NHWC format.

    Returns:
      Output sequence after applying convolutions of shape [B, H', W', D_out].
      Note that if the padding is SAME and there is no dilation and striding,
      then H' = H and W' = W.
    """
    p = self.hparams
    # Check if the feature_group_count is compatible with the inputs and filter
    # For more information see XLA docs on ConvWithGeneralPadding below
    # https://www.tensorflow.org/xla/operation_semantics#convwithgeneralpadding_convolution
    if inputs.shape[3] % p.filter_shape[2] != 0:
      raise ValueError(f'Input features {inputs.shape[3]} must be a'
                       f'multiple of filter input dim {p.filter_shape[2]} '
                       f'(Input shape: {inputs.shape}, '
                       f'filter shape: {p.filter_shape}).')
    # feature group count is D_in // filter input dim
    feature_group_count = inputs.shape[3] // p.filter_shape[2]
    if p.filter_shape[3] % feature_group_count != 0:
      raise ValueError(f'Filter output dim {p.filter_shape[3]} must be a '
                       f'multiple of feature group count {feature_group_count} '
                       f'(Input shape: {inputs.shape}, '
                       f'filter shape: {p.filter_shape}).')
    padding = self._compute_padding(inputs.shape)

    # The `dimension_numbers=('NHWC', 'HWIO', 'NHWC')` is to be consistent
    # with tf.conv2d, see e.g., see
    # https://github.com/google/jax/blob/main/jax/_src/lax/lax.py#L622
    outputs = jax.lax.conv_general_dilated(
        lhs=inputs,
        rhs=self.theta.w,
        window_strides=p.filter_stride,
        padding=padding,
        rhs_dilation=p.dilations,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=feature_group_count)
    if p.bias:
      outputs += jnp.reshape(self.theta.b, (1,) * (outputs.ndim - 1) + (-1,))
    return outputs


class ConvBNAct(Conv2D):
  """A block of conv-bn-activation layers used for image encoders.

  By default, we use cross-replica sum on TPUs.
  """

  class HParams(Conv2D.HParams):
    """Associated hyper-params for this layer class.

    Attributes:
      batch_norm_tpl: The batchnorm layer template.
      activation_tpl: Activation function to use.
    """
    batch_norm_tpl: Optional[BaseHParams] = sub_config_field(
        normalizations.BatchNorm.HParams)
    activation_tpl: activations.BaseActivation = sub_config_field(
        activations.ReLU.HParams)

  def setup(self) -> None:
    super().setup()
    p = self.hparams

    if p.batch_norm_tpl is not None:
      bn = p.batch_norm_tpl.clone()
      bn.dim = p.filter_shape[3]
      bn.use_moving_avg_in_training = False
      self.create_child('bn', bn)
    self.create_child('activation', p.activation_tpl.clone())

  def __call__(self, inputs: JTensor) -> JTensor:
    """Forward prop which applies conv-bn-activation.

    Args:
      inputs: Input sequence of shape [B, H, W, D_in], also known more popularly
        as NHWC format.

    Returns:
      Output sequence after applying convolutions of shape [B, H', W', D_out].
      Note that if the padding is SAME and there is no dilation and striding,
      then H' = H and W' = W.
    """
    p = self.hparams
    outputs = super().__call__(inputs)
    if p.batch_norm_tpl is not None:
      outputs = self.bn(outputs)
    outputs = self.activation(outputs)
    return outputs


class ConvBNActWithPadding(ConvBNAct):
  """A block of conv-bn-activation layers with padding processing."""

  class HParams(ConvBNAct.HParams):
    """Associated hyper-params for this layer class.

    Attributes:
      compat_with_lingvo: If use lingvo-compatible logic.
    """
    compat_with_lingvo: bool = False

  def setup(self) -> None:
    super().setup()
    p = self.hparams
    if p.compat_with_lingvo:
      assert tuple(
          p.dilations) == (1, 1), ('compat_with_lingvo supports no dilation.')

  def __call__(
      self, inputs: JTensor, paddings: JTensor
  ) -> Tuple[JTensor, JTensor]:  # pytype:disable=signature-mismatch
    """Forward prop which applies conv-bn-activation with time paddings.

    Args:
      inputs: Input sequence of shape [B, H, W, D_in], also known more popularly
        as NHWC format.
      paddings: Input sequence of shape [B, H], where H is the time dimension.

    Returns:
      Output sequence after applying convolutions of shape [B, H', W', D_out].
      Note that if the padding is SAME and there is no dilation and striding,
      then H' = H and W' = W.
      Output padding after applying convolutions.
    """
    p = self.hparams

    # Applying padding.
    inputs *= (1 - paddings)[:, :, None, None]

    outputs = super().__call__(inputs)

    if p.filter_stride[0] == 1 and p.padding == 'SAME':
      return outputs, paddings
    if p.padding == 'SAME':
      input_length = paddings.shape[1]
      stride = p.filter_stride[0]

      if p.compat_with_lingvo:
        out_padding = paddings[:, stride - 1::stride]
        out_padding = jnp.pad(
            out_padding, [[0, 0], [0, outputs.shape[1] - out_padding.shape[1]]],
            constant_values=1)
      else:
        pad_len = (input_length + stride - 1) // stride * stride - input_length
        out_padding = jax.lax.conv_general_dilated(
            lhs=paddings[:, :, None],
            rhs=jnp.ones([1, 1, 1]),
            window_strides=p.filter_stride[:1],
            padding=[(0, pad_len)],
            rhs_dilation=p.dilations[:1],
            dimension_numbers=('NHC', 'HIO', 'NHC'))
        out_padding = jnp.squeeze(out_padding, axis=-1)
    else:
      def rolling_window(arr: JTensor, window: int, stride: int):
        idx = jnp.arange(0, arr.shape[1] - window + 1,
                         stride)[:, None] + jnp.arange(window)[None, :]
        return arr[:, idx]

      window = p.filter_shape[0]
      stride = p.filter_stride[0]
      out_padding = rolling_window(paddings, window, stride)
      out_padding = out_padding.min(axis=-1, keepdims=False)
    outputs = outputs * (1.0 -
                         jnp.expand_dims(jnp.expand_dims(out_padding, -1), -1))
    return outputs, out_padding


# TODO(nanxinchen): add Depthwise Conv2D support
class BaseDepthwiseConv1D(base_layer.BaseLayer):
  """Base class for Depthwise 1D convolution."""

  class HParams(BaseHParams):
    """Associated hyper-params for this layer class.

    Attributes:
      filter_shape: Filter shape. Must be a sequence of length 3. Elements are
        in the order of kernel_size, in_channels, channel_multipliers.
      bias:         Whether or not to apply a bias before activation.
      bias_init:    Bias initializer to use if bias is to be applied.
      is_causal:    Whether this is a causal layer.
      use_2d_conv_weight_shape: Whether to use 2d conv's weight shape. This is
        for checkpoint backwards-compatibility.
      rhs_dilation_rate: The dilation rate in atrous convolution.
    """
    filter_shape: Sequence[int] = (0, 0, 0)
    bias: bool = False
    bias_init: WeightInit = WeightInit.Constant(0.0)
    is_causal: bool = False
    use_2d_conv_weight_shape: bool = False
    rhs_dilation_rate: int = 1

  def __call__(self,
               inputs: JTensor,
               paddings: Optional[JTensor] = None) -> JTensor:
    """Depthwise convolution.

    Args:
      inputs: Input sequence JTensor of shape [B, T, H].
      paddings: Input paddings JTensor of shape [B, T].

    Returns:
      The depthwise conv output with shape [B, T, H * channel_multipliers]
        where channel_multipliers = filter_shape[-1].
    """
    raise NotImplementedError()


class DepthwiseConv1D(BaseDepthwiseConv1D):
  """Depthwise 1D convolution based on lax implementation."""

  # SPMD partition related params.
  # h - height
  # w - width
  # i - in_channels
  # m - channel_multiplier
  class WeightShardingHParams(BaseWtShardingHParams):
    """Represents how layer's learned parameters are partitioned across a mesh.

    Attributes:
      him:  Mesh split for weight. If use_2d_conv_weight_shape is set, the
        weight shape is actually him, and w dim is not sharded.
    """
    him: SplitDimsMapping = None

  def setup(self) -> None:
    p = self.hparams
    wp_him = p.weight_split_dims_mapping.clone().him

    assert len(p.filter_shape) == 3
    assert p.rhs_dilation_rate > 0

    w_shape = [p.filter_shape[0], 1, p.filter_shape[1] * p.filter_shape[2]]
    bias_shape = w_shape[-1]

    if p.use_2d_conv_weight_shape:
      w_shape = w_shape + [1]
      if wp_him:
        # [h, None, i, m]
        wp_him.insert(1, None)

    self.create_variable(
        'w',
        WeightHParams(
            shape=w_shape,
            mesh_shape=p.mesh_shape,
            tensor_split_dims_mapping=wp_him))
    if p.bias:
      self.create_variable(
          'b', WeightHParams(shape=[bias_shape], init=p.bias_init))

  def get_w(self) -> JTensor:
    p = self.hparams
    wp_him = p.weight_split_dims_mapping.him
    if p.use_2d_conv_weight_shape:
      w = jnp.squeeze(self.theta.w, -1)
      w = base_layer.maybe_shard(w, wp_him, p.mesh_axis_names)
      return w
    else:
      return self.theta.w

  def __call__(self,
               inputs: JTensor,
               paddings: Optional[JTensor] = None) -> JTensor:
    """Depthwise convolution layer.

    Args:
      inputs: Input sequence JTensor of shape [B, T, H].
      paddings: Input paddings JTensor of shape [B, T].

    Returns:
      The depthwise conv output with shape [B, T, H].
    """
    p = self.hparams

    # Applying padding.
    if paddings is not None:
      inputs = inputs * (1.0 - jnp.expand_dims(paddings, axis=-1))

    dn = jax.lax.conv_dimension_numbers(inputs.shape,
                                        self.get_w().shape,
                                        ('NHC', 'HIO', 'NHC'))

    if p.is_causal:
      causal_pad_size = p.rhs_dilation_rate * (p.filter_shape[0] - 1)
      padding = [(causal_pad_size, 0)]
    else:
      padding = 'SAME'

    out = jax.lax.conv_general_dilated(
        lhs=inputs,
        rhs=self.get_w(),
        window_strides=(1,),
        padding=padding,
        lhs_dilation=(1,),
        rhs_dilation=(p.rhs_dilation_rate,),
        dimension_numbers=dn,
        feature_group_count=p.filter_shape[1])
    if p.bias:
      out = out + self.theta.b
    return out


class LightConv1D(base_layer.BaseLayer):
  """Lightweight conv layer.

  architecture::

  input-ln()-ff()-glu()-depthwise_conv1d()-norm()-act()-ff()-dropout()-+-output
    |__________________________________________________________________|

  """

  # TODO(nanxinchen): add causal support
  # TODO(nanxinchen): add SPMD partitioning support

  class HParams(BaseHParams):
    """Associated hyper-params for this layer class.

    Attributes:
      input_dims:      Input and (in fact,) output dimension.
      kernel_size:     Kernel size of 1d deptwise conv.
      conv_activation_tpl: Activation after normalization.
      dropout_prob:    Dropout probability.
      ln_tpl:          Parameterization of input layer normalization.
      linear_start_tpl:     Parameterization of linear start layer.
      depthwise_conv_tpl:   Parameterization of depthwise conv layer.
      conv_norm_layer_tpl:  Parameterization of normalization layer after conv.
      linear_end_tpl:       Parameterization of linear end layer.
      dropout_tpl:          Parameterization of residual dropout layer.
      is_causal:            Whether this is a causal layer.
      use_2d_conv_norm:     Whether to expand the input to conv_norm to 2d. This
        is for compatibility with old models trained in TF lingvo.
    """
    input_dims: Optional[int] = None
    kernel_size: Optional[int] = None
    conv_activation_tpl: activations.BaseActivation.HParams = sub_config_field(
        activations.Swish.HParams)
    dropout_prob: float = 0.0
    ln_tpl: BaseHParams = sub_config_field(normalizations.LayerNorm.HParams)

    linear_start_tpl: BaseHParams = sub_config_field(
        linears.FeedForward.HParams)
    depthwise_conv_tpl: BaseHParams = sub_config_field(DepthwiseConv1D.HParams)
    conv_norm_layer_tpl: BaseHParams = sub_config_field(
        normalizations.BatchNorm.HParams)
    linear_end_tpl: BaseHParams = sub_config_field(linears.FeedForward.HParams)
    dropout_tpl: BaseHParams = sub_config_field(stochastics.Dropout.HParams)
    is_causal: bool = False
    use_2d_conv_norm: bool = False

  # SPMD partition related params.
  #
  # d - model_dim
  # f - ff_hidden_dim (here ff_hidden_dim has the same size as model_dim)
  # h - height
  # i - in_channels
  # m - channel_multiplier
  # b - batch_size
  # l - seq_len
  class WeightShardingHParams(BaseWtShardingHParams):
    """Represents how layer's learned parameters are partitioned across a mesh.

    Attributes:
      df:    Mesh split for lconv linear start weight.
      him:  Mesh split for lconv depthwise conv weight.
    """
    df: SplitDimsMapping = None
    him: SplitDimsMapping = None

  class ActivationShardingHParams(BaseActShardingHParams):
    """Represents how intermediate values should be partitioned across a mesh.

    Attributes:
      blf: Mesh split for lconv linear start act and lconv depthwise conv after
        normalization.
      bld: Mesh split for lconv linear end act.
    """
    blf: SplitDimsMapping = None
    bld: SplitDimsMapping = None

  def _create_conv(self):
    p = self.hparams
    wp = p.weight_split_dims_mapping
    depthwise_conv_p = p.depthwise_conv_tpl.clone().set(
        filter_shape=(p.kernel_size, p.input_dims, 1), is_causal=p.is_causal)
    depthwise_conv_p.weight_split_dims_mapping.him = wp.him
    self.create_child('depthwise_conv1d', depthwise_conv_p)

  def _create_conv_norm_layer(self):
    p = self.hparams
    norm_p = p.conv_norm_layer_tpl.clone().set(dim=p.input_dims)
    self.create_child('conv_norm', norm_p)

  def setup(self) -> None:
    p = self.hparams
    wp = p.weight_split_dims_mapping
    ap = p.activation_split_dims_mapping

    ln_p = p.ln_tpl.clone().set(name='ln', dim=p.input_dims)
    self.create_child('ln', ln_p)

    # The activation/gate matrix is a sub-matrix of large matrix so the scale
    # needs to be smaller.
    # More specifically, since input_dim == output_dim, the scale of xavier
    # should be \sqrt(6) / \sqrt(input_dim + input_dim * 2) instead of \sqrt(6)
    # / \sqrt(input_dim + input_dim).

    linear_start_act_p = p.linear_start_tpl.clone().set(
        input_dims=p.input_dims,
        output_dims=p.input_dims,
        activation_tpl=activations.Identity.HParams(),
        params_init=WeightInit.Xavier(math.sqrt(3 / 2)))
    linear_start_act_p.weight_split_dims_mapping.wt = wp.df
    linear_start_act_p.activation_split_dims_mapping.out = ap.blf
    self.create_child('linear_start_act', linear_start_act_p)

    linear_start_gated_p = p.linear_start_tpl.clone().set(
        input_dims=p.input_dims,
        output_dims=p.input_dims,
        activation_tpl=activations.Identity.HParams(),
        params_init=WeightInit.Xavier(math.sqrt(3 / 2)))
    linear_start_gated_p.weight_split_dims_mapping.wt = wp.df
    linear_start_gated_p.activation_split_dims_mapping.out = ap.blf
    self.create_child('linear_start_gated', linear_start_gated_p)

    # TODO(nanxinchen): the end layer doesn't split so it shouldn't use 3/2
    linear_end_p = p.linear_end_tpl.clone().set(
        input_dims=p.input_dims,
        output_dims=p.input_dims,
        activation_tpl=activations.Identity.HParams(),
        params_init=WeightInit.Xavier(math.sqrt(3 / 2)))
    if wp.df:
      linear_end_p.weight_split_dims_mapping.wt = list(reversed(wp.df))
    linear_end_p.activation_split_dims_mapping.out = ap.bld
    self.create_child('linear_end', linear_end_p)

    self._create_conv()
    self._create_conv_norm_layer()
    self.create_child('conv_activation', p.conv_activation_tpl.clone())

    dropout_p = p.dropout_tpl.clone().set(keep_prob=1. - p.dropout_prob)
    self.create_child('dropout', dropout_p)

  def _conv_norm(self, inputs: JTensor, paddings: JTensor) -> JTensor:
    p = self.hparams
    if p.use_2d_conv_norm:
      # BTH -> BT1H
      inputs = jnp.expand_dims(inputs, 2)
    inputs = self.conv_norm(inputs, paddings)
    if p.use_2d_conv_norm:
      # BT1H -> BTH
      inputs = jnp.squeeze(inputs, 2)
    return inputs

  def __call__(self, inputs: JTensor, paddings: JTensor) -> JTensor:
    """Lightweight conv layer.

    Args:
      inputs: Input sequence JTensor of shape [B, T, H].
      paddings: Input paddings JTensor of shape [B, T].

    Returns:
      The lconv output with shape [B, T, H].
    """
    p = self.hparams
    ap = p.activation_split_dims_mapping

    unnormalized_inputs = inputs

    inputs = self.ln(inputs)
    act_inputs = self.linear_start_act(inputs)
    gated_inputs = self.linear_start_gated(inputs)
    inputs = act_inputs * jax.nn.sigmoid(gated_inputs)

    inputs = self.depthwise_conv1d(inputs, paddings)
    inputs = base_layer.maybe_shard(inputs, ap.blf, p.mesh_axis_names)

    inputs = self._conv_norm(inputs, paddings)

    inputs = base_layer.maybe_shard(inputs, ap.blf, p.mesh_axis_names)
    inputs = self.conv_activation(inputs)

    inputs = self.linear_end(inputs)
    inputs = self.dropout(inputs)

    output = inputs + unnormalized_inputs
    return output
