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

"""Mobilenet-style depth-separable convolution classifiers."""

from typing import Optional, Sequence, Union

from praxis import base_layer
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis.layers import activations
from praxis.layers import convolutions
from praxis.layers import linears
from praxis.layers import normalizations
from praxis.layers import poolings
from praxis.layers import sequential

NestedMap = py_utils.NestedMap
WeightInit = base_layer.WeightInit
JTensor = pytypes.JTensor

LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]
template_field = base_layer.template_field


# Taken from the original tf repo:
# https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
def _make_divisible(
    v: Union[int, float], divisor: int, min_value: Optional[int] = None
) -> int:
  """Ensure that all layers have num channels divisible by, e.g., 8."""
  if min_value is None:
    min_value = divisor
  new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_v < 0.9 * v:
    new_v += divisor
  return new_v


class InvertedResidualBlock(base_layer.BaseLayer):
  """MobileNetv2 Inverted Residual Block as in https://arxiv.org/abs/1801.04381.

  Attributes:
    input_dim: Input dimension.
    output_dim: Output dimension.
    expansion: Expansion factor for inverted bottleneck.
    conv_params: Parameterization of the convolution layer.
    kernel_size: Kernel sizes of the block.
    stride: Stride.
    activation_tpl: Activation function to use.
  """

  input_dim: int = 0
  output_dim: int = 0
  expansion: int = 6
  conv_params: LayerTpl = template_field(convolutions.ConvBNAct)
  kernel_size: int = 3
  stride: int = 1
  activation_tpl: pax_fiddle.Config[activations.BaseActivation] = (
      template_field(activations.ReLU6)
  )

  def setup(self) -> None:

    body = []
    if self.expansion != 1:  # Some blocks (e.g., the first) have no expansion.
      # Expand with a pointwise 1x1 convolution.
      expanded_input_dim = self.expansion * self.input_dim
      body.append(
          self.conv_params.clone().set(
              name='expansion',
              filter_shape=(1, 1, self.input_dim, expanded_input_dim),
              filter_stride=(1, 1),
              activation_tpl=self.activation_tpl,
          )
      )
    else:
      expanded_input_dim = self.input_dim

    # Depthwise 3x3 convoluition.
    body.append(
        self.conv_params.clone().set(
            name='depthwise_conv',
            filter_shape=(
                self.kernel_size,
                self.kernel_size,
                1,  # Input channels of 1 means depthwise.
                expanded_input_dim,
            ),
            filter_stride=(self.stride, self.stride),
            activation_tpl=self.activation_tpl,
        )
    )

    # Project with a pointwise 1x1 convolution.
    body.append(
        self.conv_params.clone().set(
            name='project',
            filter_shape=(1, 1, expanded_input_dim, self.output_dim),
            filter_stride=(1, 1),
            activation_tpl=pax_fiddle.Config(activations.Identity),
        )
    )
    self.create_child(
        'body', pax_fiddle.Config(sequential.Sequential, layers=body)
    )

  def __call__(self, inputs: JTensor) -> JTensor:
    """Forward propagation of an InvertedResidualBlock.

    Args:
      inputs: A `.JTensor` as inputs of [B, H, W, D_in] also commonly known as
        NHWC format.

    Returns:
      A `.JTensor` as outputs of shape [B, H', W', D_out].
    """
    outputs = self.body(inputs)
    if self._in_out_same_shape:
      outputs += inputs
    return outputs

  @property
  def _in_out_same_shape(self):
    """Indicates whether the input/output have the same shape or not."""
    return self.input_dim == self.output_dim and self.stride == 1


@pax_fiddle.auto_config
def _mobile_net_conv_params_default():
  return pax_fiddle.Config(
      convolutions.ConvBNAct,
      activation_tpl=pax_fiddle.Config(activations.ReLU6),
      batch_norm_tpl=pax_fiddle.Config(normalizations.BatchNorm, decay=0.9),
      params_init=WeightInit.GaussianSqrtFanOut(1.4141),
  )


@pax_fiddle.auto_config
def _mobile_net_output_spatial_pooling_params_default():
  return pax_fiddle.Config(poolings.GlobalPooling, pooling_dims=(1, 2))


@pax_fiddle.auto_config
def _mobile_net_final_prediction_params_default():
  return pax_fiddle.Config(linears.FeedForward)


class MobileNet(base_layer.BaseLayer):
  """MobileNetV2 Inverted-Residual network.

  Raises:
    ValueError if length of `strides`, `channels`, and `expansions` do
    not match.

  Attributes:
    conv_params: A layer params template specifying Conv-BN-Activation template
      used by the MobileNet model.
    block_params: A layer params template specifying InvertedResidual block to
      use in the core.
    expansions: A list of integers specifying the expansion factor to apply in
      each block.
    strides: A list of integers specifying the stride for each block.
    channels: A list of integers specifying the number of channels at each
      block.
    output_channels: Dimension of final embedding layer.
    num_classes: If nonzero, add a final prediction layer to this many classes.
    entryflow_conv_kernel: A tuple of three integers giving the kernel size of
      entryflow convolution.  The third is the depth of the input data (e.g., 3
      for RGB images, 1 for spectrograms).
    entryflow_conv_stride: A tuple of two integers giving the stride of
      entryflow convolution.
    output_spatial_pooling_params: A layer params template specifying spatial
      pooling before output. If None, spatial pooling is not added.
    predict_params: A layer params template for the final dense prediction
      layer.
    multiplier: float scaling of layer sizes, for variably-sized models.
    divisible_by: After applying multiplier, ensure sizes are still multiples of
      this.
    min_depth: After applying multiplier, ensure no layer is shallower than
      this.
  """

  conv_params: LayerTpl = pax_fiddle.fdl_field(
      default_factory=_mobile_net_conv_params_default
  )
  block_params: LayerTpl = template_field(InvertedResidualBlock)
  # Parameters for the 17 internal blocks in 5 groups.
  # pyformat: disable
  expansions: Sequence[int] = (
      1,
      6, 6,
      6, 6, 6,
      6, 6, 6, 6, 6, 6, 6,
      6, 6, 6, 6,
  )
  strides: Sequence[int] = (
      1,
      2, 1,
      2, 1, 1,
      2, 1, 1, 1, 1, 1, 1,
      2, 1, 1, 1,
  )
  channels: Sequence[int] = (
      16,
      24, 24,
      32, 32, 32,
      64, 64, 64, 64, 96, 96, 96,
      160, 160, 160, 320,
  )
  # pyformat: enable
  output_channels: int = 1280
  num_classes: int = 0
  entryflow_conv_kernel: Sequence[int] = (3, 3, 3)
  entryflow_conv_stride: Sequence[int] = (2, 2)
  output_spatial_pooling_params: LayerTpl | None = pax_fiddle.fdl_field(
      default_factory=_mobile_net_output_spatial_pooling_params_default
  )
  predict_params: LayerTpl = pax_fiddle.fdl_field(
      default_factory=_mobile_net_final_prediction_params_default
  )
  multiplier: float = 1.0
  divisible_by: int = 8
  min_depth: int = 8

  def setup(self) -> None:
    body = []
    num_blocks = len(self.expansions)
    if num_blocks != len(self.strides):
      raise ValueError(
          f'num_blocks {num_blocks} != len strides {len(self.strides)}.'
      )
    if num_blocks != len(self.channels):
      raise ValueError(
          f'num_blocks {num_blocks} != len channels {len(self.channels)}.'
      )

    # Setup the convolution type used in the InvertedResidual block.
    block_params = self.block_params.clone()
    if hasattr(block_params, 'conv_params'):
      block_params.conv_params = self.conv_params.clone()

    # First conv layer: chans = 32, stride = 1.
    input_dim = _make_divisible(
        32 * self.multiplier, self.divisible_by, self.min_depth
    )
    entryflow_conv_params = self.conv_params.clone()
    entryflow_conv_params.filter_shape = (
        self.entryflow_conv_kernel[0],
        self.entryflow_conv_kernel[1],
        self.entryflow_conv_kernel[2],
        input_dim,
    )
    entryflow_conv_params.filter_stride = self.entryflow_conv_stride
    entryflow_conv_params.name = 'entryflow_conv'
    body.append(entryflow_conv_params)

    # Create the chain of InvertedResidual blocks.
    for block_id, (expansion, stride, channel) in enumerate(
        zip(self.expansions, self.strides, self.channels)
    ):
      name = f'block_{block_id}'
      output_dim = _make_divisible(channel * self.multiplier, self.divisible_by)
      block_p = block_params.clone().set(
          name=name,
          input_dim=input_dim,
          output_dim=output_dim,
          expansion=expansion,
          stride=stride,
      )
      body.append(block_p)
      input_dim = output_dim

    # Final conv layer.
    output_channels = self.output_channels
    if self.multiplier > 1.0:
      output_channels = _make_divisible(
          output_channels * self.multiplier, self.divisible_by, self.min_depth
      )
    final_conv_params = self.conv_params.clone()
    final_conv_params.filter_shape = (1, 1, input_dim, output_channels)
    final_conv_params.filter_stride = (1, 1)
    final_conv_params.name = 'final_conv'
    body.append(final_conv_params)

    # Add optional spatial global pooling.
    if self.output_spatial_pooling_params is not None:
      self.output_spatial_pooling_params.name = 'output_spatial_pooling'
      body.append(self.output_spatial_pooling_params)
    # Add optional fully-connected classification layer.
    if self.num_classes:
      predict_params = self.predict_params.clone()
      predict_params.input_dims = self.output_channels
      predict_params.output_dims = self.num_classes
      predict_params.name = 'predictions'
      body.append(predict_params)

    self.create_child(
        'body', pax_fiddle.Config(sequential.Sequential, layers=body)
    )

  def __call__(self, inputs: JTensor) -> JTensor:
    """Applies the MobileNet model to the inputs.

    Args:
      inputs: Input image tensor of shape [B, H, W, c].

    Returns:
      Output tensor of MobileNet of shape [B, H, W, D] where D is the
      output_channels. If `output_spatial_pooling_params` is not None, then the
      output will be of a shape that depends on which dims are pooled. For e.g.,
      if the pooling dims are [1, 2], then output shape will be [B, D].
    """
    return self.body(inputs)
