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

"""ResNet layers."""

import dataclasses
from typing import Optional, Sequence

from praxis import base_layer
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis.layers import activations
from praxis.layers import convolutions
from praxis.layers import normalizations
from praxis.layers import poolings
from praxis.layers import stochastics

NestedMap = py_utils.NestedMap
WeightInit = base_layer.WeightInit
JTensor = pytypes.JTensor

LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]
template_field = base_layer.template_field


class ResNetBlock(base_layer.BaseLayer):
  """ResNet Block as in https://arxiv.org/abs/1512.03385.

  Attributes:
    input_dim: Input dimension.
    output_dim: Output dimension.
    conv_params: Parameterization of the convolution layer.
    kernel_size: Kernel sizes of the block.
    stride: Stride.
    activation_tpl: Activation function to use.
    residual_droppath_prob: Probability for residual path.
    zero_init_residual: zero-initialize the gamma of the last BN in each
      residual branch. From https://arxiv.org/abs/2105.07576.
  """
  input_dim: int = 0
  output_dim: int = 0
  conv_params: LayerTpl = template_field(convolutions.ConvBNAct)
  kernel_size: int = 3
  stride: int = 1
  activation_tpl: pax_fiddle.Config[
      activations.BaseActivation
  ] = template_field(activations.ReLU)
  residual_droppath_prob: float = 0.0
  zero_init_residual: bool = False

  def setup(self) -> None:

    body = []
    # conv_in, reduce the hidden dims by 4
    body.append(
        self.conv_params.clone().set(
            name='conv_in',
            filter_shape=(1, 1, self.input_dim, self.output_dim // 4),
            filter_stride=(1, 1),
            activation_tpl=self.activation_tpl,
        )
    )

    # conv_mid using the kernel size and stride provided
    body.append(
        self.conv_params.clone().set(
            name='conv_mid',
            filter_shape=(
                self.kernel_size,
                self.kernel_size,
                self.output_dim // 4,
                self.output_dim // 4,
            ),
            filter_stride=(self.stride, self.stride),
            activation_tpl=self.activation_tpl,
        )
    )

    # conv_out, expand back to hidden dim
    last_bn_tpl = self.conv_params.batch_norm_tpl
    if self.zero_init_residual:
      if last_bn_tpl is None:
        # This can be implemented by zero-init the weights of conv layer.
        raise NotImplementedError(
            'zero_init_residual without BN is not implemented.')
      else:
        last_bn_tpl = last_bn_tpl.clone().set(
            gamma_init=WeightInit.Constant(
                -1.0 if self.zero_init_residual else 0.0
            )
        )
    body.append(
        self.conv_params.clone().set(
            name='conv_out',
            filter_shape=(1, 1, self.output_dim // 4, self.output_dim),
            filter_stride=(1, 1),
            activation_tpl=pax_fiddle.Config(activations.Identity),
            batch_norm_tpl=last_bn_tpl,
        )
    )
    self.create_children('body', body)

    # projection with 1x1 if input dim and output dim are not the same
    if not self._in_out_same_shape:
      shortcut = self.conv_params.clone().set(
          name='shortcut',
          filter_shape=(1, 1, self.input_dim, self.output_dim),
          filter_stride=(self.stride, self.stride),
          activation_tpl=pax_fiddle.Config(activations.Identity),
      )
      self.create_child('shortcut', shortcut)

    # Initialize droppath layer
    if self.residual_droppath_prob > 0:
      droppath_p = pax_fiddle.Config(
          stochastics.StochasticResidual,
          survival_prob=1.0 - self.residual_droppath_prob,
      )
      self.create_child('residual_droppath', droppath_p)

    post_activation = self.activation_tpl.clone().set(name='post_activation')
    self.create_child('postact', post_activation)

  def __call__(self, inputs: JTensor) -> JTensor:
    """Forward propagation of a ResNetBlock.

    Args:
      inputs: A `.JTensor` as inputs of [B, H, W, D_in] also commonly known as
        NHWC format.

    Returns:
      A `.JTensor` as outputs of shape [B, H', W', D_out].
    """
    outputs = inputs

    # body
    for i in range(len(self.body)):
      outputs = self.body[i](outputs)

    # projection
    if not self._in_out_same_shape:
      inputs = self.shortcut(inputs)

    # residual
    if self.residual_droppath_prob:
      outputs = self.residual_droppath(inputs, outputs)
    else:
      outputs += inputs

    # post activation
    outputs = self.postact(outputs)
    return outputs

  @property
  def _in_out_same_shape(self):
    """Indicates whether the input/output have the same shape or not."""
    return self.input_dim == self.output_dim and self.stride == 1


class ResNetBasicBlock(ResNetBlock):
  """ResNet Basic Block as in https://arxiv.org/abs/1512.03385."""

  def setup(self) -> None:

    body = []
    # first conv
    body.append(
        self.conv_params.clone().set(
            name='conv_in',
            filter_shape=(
                self.kernel_size,
                self.kernel_size,
                self.input_dim,
                self.output_dim,
            ),
            filter_stride=(self.stride, self.stride),
            activation_tpl=self.activation_tpl.clone(),
        )
    )

    # second conv
    body.append(
        self.conv_params.clone().set(
            name='conv_mid',
            filter_shape=(
                self.kernel_size,
                self.kernel_size,
                self.output_dim,
                self.output_dim,
            ),
            filter_stride=(1, 1),
            activation_tpl=pax_fiddle.Config(activations.Identity),
        )
    )
    self.create_children('body', body)

    # projection with 1x1 if input dim and output dim are not the same
    if not self._in_out_same_shape:
      shortcut = self.conv_params.clone().set(
          name='shortcut',
          filter_shape=(1, 1, self.input_dim, self.output_dim),
          filter_stride=(self.stride, self.stride),
          activation_tpl=pax_fiddle.Config(activations.Identity),
      )
      self.create_child('shortcut', shortcut)

    # Initialize droppath layer
    if self.residual_droppath_prob > 0:
      droppath_p = pax_fiddle.Config(
          stochastics.StochasticResidual,
          survival_prob=1.0 - self.residual_droppath_prob,
      )
      self.create_child('residual_droppath', droppath_p)

    post_activation = self.activation_tpl.clone().set(name='post_activation')
    self.create_child('postact', post_activation)

  def __call__(self, inputs: JTensor) -> JTensor:
    """Forward propagation of a ResNetBlock.

    Args:
      inputs: A `.JTensor` as inputs of [B, H, W, D_in] also commonly known as
        NHWC format.

    Returns:
      A `.JTensor` as outputs of shape [B, H', W', D_out].
    """
    outputs = inputs

    # body
    for i in range(len(self.body)):
      outputs = self.body[i](outputs)

    # projection
    if not self._in_out_same_shape:
      inputs = self.shortcut(inputs)

    # residual
    if self.residual_droppath_prob:
      outputs = self.residual_droppath(inputs, outputs)
    else:
      outputs += inputs

    # post activation
    outputs = self.postact(outputs)
    return outputs


@pax_fiddle.auto_config
def _res_net_conv_params_default():
  return pax_fiddle.Config(
      convolutions.ConvBNAct,
      batch_norm_tpl=pax_fiddle.Config(normalizations.BatchNorm, decay=0.9),
      params_init=WeightInit.GaussianSqrtFanOut(1.4141))


@pax_fiddle.auto_config
def _res_net_output_spatial_pooling_params_default():
  return pax_fiddle.Config(poolings.GlobalPooling, pooling_dims=(1, 2))


class ResNet(base_layer.BaseLayer):
  """Resnet model with default params matching Resnet-50.

  Additionally, params are also provided for Resnet-101 and Resnet-152.
  See https://arxiv.org/abs/1512.03385 for more details.

  Raises:
    ValueError if length of `strides`, `channels`, `blocks` and `kernels` do
    not match.

  Attributes:
  conv_params: A layer params template specifying Conv-BN-Activation template
    used by the ResNet model.
  block_params: A layer params template specifying Convolution Block used in
    each stage. We use the same ResNetBlock tpl in all stages (4 stages in
    total) in ResNet.
  strides: A list of integers specifying the stride for each stage. A stage is
    defined as a stack of Convolution Blocks that share same type, channels
    and kernels. The stride is always applied only at the beginning of each
    stage, while within that stage, all other strides are set to 1 (no
    stride).
  channels: A list of integers specifying the number of channels at different
    stages. The first channel is usually 4x the input dim.
  blocks: A list of integers specifying the number of blocks at different
    stages.
  kernels: A list of integers specifying the number of kernel sizes at
    different stages.
  entryflow_conv_kernel: A tuple of three integers as the kernel size of
    entryflow convolution.
  entryflow_conv_stride: A tuple of two integers as the stride of entryflow
    convolution.
  output_spatial_pooling_params: A layer params template specifying spatial
    pooling before output. If None, spatial pooling is not added.
  """
  # pylint: disable=g-long-lambda
  conv_params: LayerTpl = pax_fiddle.fdl_field(
      default_factory=_res_net_conv_params_default, tags=pax_fiddle.DoNotBuild)
  # pylint: enable=g-long-lambda
  block_params: LayerTpl = template_field(ResNetBlock)
  strides: Sequence[int] = (1, 2, 2, 2)
  channels: Sequence[int] = (256, 512, 1024, 2048)
  blocks: Sequence[int] = (3, 4, 6, 3)
  kernels: Sequence[int] = (3, 3, 3, 3)
  entryflow_conv_kernel: Sequence[int] = (7, 7, 3)
  entryflow_conv_stride: Sequence[int] = (2, 2)
  output_spatial_pooling_params: Optional[LayerTpl] = pax_fiddle.fdl_field(
      default_factory=_res_net_output_spatial_pooling_params_default,
      tags=pax_fiddle.DoNotBuild)
  return_block_features: bool = False

  @classmethod
  def HParamsResNet5(cls) -> LayerTpl:
    """Returns ResNet5 hyperparams for testing purposes."""
    return pax_fiddle.Config(
        cls, strides=[1], channels=[16], blocks=[1], kernels=[1]
    )

  @classmethod
  def HParamsResNet18(cls) -> LayerTpl:
    """Returns commonly used ResNet18 hyperparams."""
    return pax_fiddle.Config(
        cls,
        channels=(64, 128, 256, 512),
        block_params=pax_fiddle.Config(ResNetBasicBlock),
        blocks=[2, 2, 2, 2],
    )

  @classmethod
  def HParamsResNet34(cls) -> LayerTpl:
    """Returns commonly used ResNet18 hyperparams."""
    return pax_fiddle.Config(
        cls,
        channels=(64, 128, 256, 512),
        block_params=pax_fiddle.Config(ResNetBasicBlock),
        blocks=[3, 4, 6, 3],
    )

  @classmethod
  def HParamsResNet50(cls) -> LayerTpl:
    """Returns commonly used ResNet50 hyperparams."""
    return pax_fiddle.Config(
        cls,
    )

  @classmethod
  def HParamsResNet101(cls) -> LayerTpl:
    """Returns commonly used ResNet101 hyperparams."""
    return pax_fiddle.Config(cls, blocks=[3, 4, 23, 3])

  @classmethod
  def HParamsResNet152(cls) -> LayerTpl:
    """Returns commonly used ResNet152 hyperparams."""
    return pax_fiddle.Config(cls, blocks=[3, 8, 36, 3])

  def setup(self) -> None:
    num_stages = len(self.strides)
    if num_stages != len(self.channels):
      raise ValueError(
          f'num_stages {num_stages} != channels {len(self.channels)}.'
      )
    if num_stages != len(self.blocks):
      raise ValueError(f'num_stages {num_stages} != blocks {len(self.blocks)}.')
    if num_stages != len(self.kernels):
      raise ValueError(
          f'num_stages {num_stages} != kernels {len(self.kernels)}.'
      )

    # Set the convolution type used in the Resnet block.
    block_params = self.block_params.clone()
    if hasattr(block_params, 'conv_params'):
      block_params.conv_params = self.conv_params.clone()

    # Create the entryflow convolution layer.
    entryflow_contraction = 4 if self.block_params.cls == ResNetBlock else 1
    input_dim = self.channels[0] // entryflow_contraction
    entryflow_conv_params = self.conv_params.clone()
    entryflow_conv_params.filter_shape = (
        self.entryflow_conv_kernel[0],
        self.entryflow_conv_kernel[1],
        self.entryflow_conv_kernel[2],
        input_dim,
    )
    entryflow_conv_params.filter_stride = self.entryflow_conv_stride
    self.create_child('entryflow_conv', entryflow_conv_params)

    # Create the entryflow max pooling layer.
    maxpool_params = pax_fiddle.Config(
        poolings.Pooling,
        name='entryflow_maxpool',
        window_shape=(3, 3),
        window_stride=(2, 2),
        pooling_type='MAX',
    )
    self.create_child('entryflow_maxpool', maxpool_params)

    # Create the chain of ResNet blocks.
    for stage_id, (channel, num_blocks, kernel, stride) in enumerate(
        zip(self.channels, self.blocks, self.kernels, self.strides)
    ):
      for block_id in range(num_blocks):
        name = f'stage_{stage_id}_block_{block_id}'
        output_dim = channel
        block_p = block_params.clone().set(
            name=name,
            kernel_size=kernel,
            input_dim=input_dim,
            output_dim=output_dim,
            stride=1 if block_id != 0 else stride,
        )
        self.create_child(name, block_p)
        input_dim = output_dim

    # Add optional spatial global pooling.
    if self.output_spatial_pooling_params is not None:
      self.create_child(
          'output_spatial_pooling', self.output_spatial_pooling_params
      )

  def __call__(self, inputs: JTensor) -> JTensor:
    """Applies the ResNet model to the inputs.

    Args:
      inputs: Input image tensor of shape [B, H, W, 3].

    Returns:
      Output tensor of ResNet of shape [B, H, W, D] where D is the last channel
      dimension. If `output_spatial_pooling_params` is not None, then the
      output will be of a shape that depends on which dims are pooled. For e.g.,
      if the pooling dims are [1, 2], then output shape will be [B, D].
    """
    block_group_features = {}

    # Apply the entryflow conv.
    outputs = self.entryflow_conv(inputs)

    # Apply the entryflow maxpooling layer.
    outputs, _ = self.entryflow_maxpool(outputs)

    # Apply the ResNet blocks.
    for stage_id, num_blocks in enumerate(self.blocks):
      for block_id in range(num_blocks):
        block_name = f'stage_{stage_id}_block_{block_id}'
        instance = getattr(self, block_name)
        outputs = instance(outputs)

      if self.return_block_features:
        block_group_features[2 + stage_id] = outputs

    if self.return_block_features:
      return block_group_features  # pytype: disable=bad-return-type  # jax-ndarray

    # Apply optional spatial global pooling.
    if self.output_spatial_pooling_params is not None:
      outputs = self.output_spatial_pooling(outputs)
    return outputs
