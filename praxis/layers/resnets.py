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

"""ResNet layers."""

import dataclasses

from typing import Optional, Sequence
from praxis import base_layer
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

BaseHParams = base_layer.BaseLayer.HParams
sub_config_field = base_layer.sub_config_field


class ResNetBlock(base_layer.BaseLayer):
  """ResNet Block as in https://arxiv.org/abs/1512.03385."""

  class HParams(BaseHParams):
    """Associated hyper-params for this layer class.

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
    conv_params: BaseHParams = sub_config_field(convolutions.ConvBNAct.HParams)
    kernel_size: int = 3
    stride: int = 1
    activation_tpl: activations.BaseActivation.HParams = sub_config_field(
        activations.ReLU.HParams)
    residual_droppath_prob: float = 0.0
    zero_init_residual: bool = False

  def setup(self) -> None:
    p = self.hparams

    body = []
    # conv_in, reduce the hidden dims by 4
    body.append(p.conv_params.clone().set(
        name='conv_in',
        filter_shape=(1, 1, p.input_dim, p.output_dim // 4),
        filter_stride=(1, 1),
        activation_tpl=p.activation_tpl))

    # conv_mid using the kernel size and stride provided
    body.append(p.conv_params.clone().set(
        name='conv_mid',
        filter_shape=(p.kernel_size, p.kernel_size, p.output_dim // 4,
                      p.output_dim // 4),
        filter_stride=(p.stride, p.stride),
        activation_tpl=p.activation_tpl))

    # conv_out, expand back to hidden dim
    last_bn_tpl = p.conv_params.batch_norm_tpl
    if p.zero_init_residual:
      if last_bn_tpl is None:
        # This can be implemented by zero-init the weights of conv layer.
        raise NotImplementedError(
            'zero_init_residual without BN is not implemented.')
      else:
        last_bn_tpl = last_bn_tpl.clone().set(
            gamma_init=WeightInit.Constant(
                -1.0 if p.zero_init_residual else 0.0))
    body.append(p.conv_params.clone().set(
        name='conv_out',
        filter_shape=(1, 1, p.output_dim // 4, p.output_dim),
        filter_stride=(1, 1),
        activation_tpl=activations.Identity.HParams(),
        batch_norm_tpl=last_bn_tpl))
    self.create_children('body', body)

    # projection with 1x1 if input dim and output dim are not the same
    if not self._in_out_same_shape:
      shortcut = p.conv_params.clone().set(
          name='shortcut',
          filter_shape=(1, 1, p.input_dim, p.output_dim),
          filter_stride=(p.stride, p.stride),
          activation_tpl=activations.Identity.HParams())
      self.create_child('shortcut', shortcut)

    # Initialize droppath layer
    if p.residual_droppath_prob > 0:
      droppath_p = stochastics.StochasticResidual.HParams(
          survival_prob=1.0 - p.residual_droppath_prob)
      self.create_child('residual_droppath', droppath_p)

    post_activation = p.activation_tpl.clone().set(name='post_activation')
    self.create_child('postact', post_activation)

  def __call__(self, inputs: JTensor) -> JTensor:
    """Forward propagation of a ResNetBlock.

    Args:
      inputs: A `.JTensor` as inputs of [B, H, W, D_in] also commonly known as
        NHWC format.

    Returns:
      A `.JTensor` as outputs of shape [B, H', W', D_out].
    """
    p = self.hparams
    outputs = inputs

    # body
    for i in range(len(self.body)):
      outputs = self.body[i](outputs)

    # projection
    if not self._in_out_same_shape:
      inputs = self.shortcut(inputs)

    # residual
    if p.residual_droppath_prob:
      outputs = self.residual_droppath(inputs, outputs)
    else:
      outputs += inputs

    # post activation
    outputs = self.postact(outputs)
    return outputs

  @property
  def _in_out_same_shape(self):
    """Indicates whether the input/output have the same shape or not."""
    p = self.hparams
    return p.input_dim == p.output_dim and p.stride == 1


class ResNetBasicBlock(ResNetBlock):
  """ResNet Basic Block as in https://arxiv.org/abs/1512.03385."""

  def setup(self) -> None:
    p = self.hparams

    body = []
    # first conv
    body.append(p.conv_params.clone().set(
        name='conv_in',
        filter_shape=(p.kernel_size, p.kernel_size, p.input_dim, p.output_dim),
        filter_stride=(p.stride, p.stride),
        activation_tpl=p.activation_tpl.clone()))

    # second conv
    body.append(p.conv_params.clone().set(
        name='conv_mid',
        filter_shape=(p.kernel_size, p.kernel_size, p.output_dim, p.output_dim),
        filter_stride=(1, 1),
        activation_tpl=activations.Identity.HParams()))
    self.create_children('body', body)

    # projection with 1x1 if input dim and output dim are not the same
    if not self._in_out_same_shape:
      shortcut = p.conv_params.clone().set(
          name='shortcut',
          filter_shape=(1, 1, p.input_dim, p.output_dim),
          filter_stride=(p.stride, p.stride),
          activation_tpl=activations.Identity.HParams())
      self.create_child('shortcut', shortcut)

    # Initialize droppath layer
    if p.residual_droppath_prob > 0:
      droppath_p = stochastics.StochasticResidual.HParams(
          survival_prob=1.0 - p.residual_droppath_prob)
      self.create_child('residual_droppath', droppath_p)

    post_activation = p.activation_tpl.clone().set(name='post_activation')
    self.create_child('postact', post_activation)

  def __call__(self, inputs: JTensor) -> JTensor:
    """Forward propagation of a ResNetBlock.

    Args:
      inputs: A `.JTensor` as inputs of [B, H, W, D_in] also commonly known as
        NHWC format.

    Returns:
      A `.JTensor` as outputs of shape [B, H', W', D_out].
    """
    p = self.hparams
    outputs = inputs

    # body
    for i in range(len(self.body)):
      outputs = self.body[i](outputs)

    # projection
    if not self._in_out_same_shape:
      inputs = self.shortcut(inputs)

    # residual
    if p.residual_droppath_prob:
      outputs = self.residual_droppath(inputs, outputs)
    else:
      outputs += inputs

    # post activation
    outputs = self.postact(outputs)
    return outputs


class ResNet(base_layer.BaseLayer):
  """Resnet model with default params matching Resnet-50.

  Additionally, params are also provided for Resnet-101 and Resnet-152.
  See https://arxiv.org/abs/1512.03385 for more details.

  Raises:
    ValueError if length of `strides`, `channels`, `blocks` and `kernels` do
    not match.
  """

  class HParams(BaseHParams):
    """Associated hyper-params for this layer class.

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
    conv_params: BaseHParams = dataclasses.field(
        default_factory=lambda: convolutions.ConvBNAct.HParams(
            batch_norm_tpl=normalizations.BatchNorm.HParams(decay=0.9),
            params_init=WeightInit.GaussianSqrtFanOut(1.4141)))
    # pylint: enable=g-long-lambda
    block_params: BaseHParams = sub_config_field(ResNetBlock.HParams)
    strides: Sequence[int] = (1, 2, 2, 2)
    channels: Sequence[int] = (256, 512, 1024, 2048)
    blocks: Sequence[int] = (3, 4, 6, 3)
    kernels: Sequence[int] = (3, 3, 3, 3)
    entryflow_conv_kernel: Sequence[int] = (7, 7, 3)
    entryflow_conv_stride: Sequence[int] = (2, 2)
    output_spatial_pooling_params: Optional[
        BaseHParams] = poolings.GlobalPooling.HParams(pooling_dims=(1, 2))
    return_block_features: bool = False

  @classmethod
  def HParamsResNet5(cls) -> BaseHParams:
    """Returns ResNet5 hyperparams for testing purposes."""
    return cls.HParams(strides=[1], channels=[16], blocks=[1], kernels=[1])

  @classmethod
  def HParamsResNet18(cls) -> BaseHParams:
    """Returns commonly used ResNet18 hyperparams."""
    return cls.HParams(
        channels=(64, 128, 256, 512),
        block_params=ResNetBasicBlock.HParams(),
        blocks=[2, 2, 2, 2])

  @classmethod
  def HParamsResNet34(cls) -> BaseHParams:
    """Returns commonly used ResNet18 hyperparams."""
    return cls.HParams(
        channels=(64, 128, 256, 512),
        block_params=ResNetBasicBlock.HParams(),
        blocks=[3, 4, 6, 3])

  @classmethod
  def HParamsResNet50(cls) -> BaseHParams:
    """Returns commonly used ResNet50 hyperparams."""
    return cls.HParams()

  @classmethod
  def HParamsResNet101(cls) -> BaseHParams:
    """Returns commonly used ResNet101 hyperparams."""
    return cls.HParams(blocks=[3, 4, 23, 3])

  @classmethod
  def HParamsResNet152(cls) -> BaseHParams:
    """Returns commonly used ResNet152 hyperparams."""
    return cls.HParams(blocks=[3, 8, 36, 3])

  def setup(self) -> None:
    p = self.hparams
    num_stages = len(p.strides)
    if num_stages != len(p.channels):
      raise ValueError(
          f'num_stages {num_stages} != channels {len(p.channels)}.')
    if num_stages != len(p.blocks):
      raise ValueError(f'num_stages {num_stages} != blocks {len(p.blocks)}.')
    if num_stages != len(p.kernels):
      raise ValueError(f'num_stages {num_stages} != kernels {len(p.kernels)}.')

    # Set the convolution type used in the Resnet block.
    block_params = p.block_params.clone()
    if hasattr(block_params, 'conv_params'):
      block_params.conv_params = p.conv_params.clone()

    # Create the entryflow convolution layer.
    entryflow_contraction = 4 if p.block_params.cls == ResNetBlock else 1
    input_dim = p.channels[0] // entryflow_contraction
    entryflow_conv_params = p.conv_params.clone()
    entryflow_conv_params.filter_shape = (p.entryflow_conv_kernel[0],
                                          p.entryflow_conv_kernel[1],
                                          p.entryflow_conv_kernel[2], input_dim)
    entryflow_conv_params.filter_stride = p.entryflow_conv_stride
    self.create_child('entryflow_conv', entryflow_conv_params)

    # Create the entryflow max pooling layer.
    maxpool_params = poolings.Pooling.HParams(
        name='entryflow_maxpool',
        window_shape=(3, 3),
        window_stride=(2, 2),
        pooling_type='MAX')
    self.create_child('entryflow_maxpool', maxpool_params)

    # Create the chain of ResNet blocks.
    for stage_id, (channel, num_blocks, kernel, stride) in enumerate(
        zip(p.channels, p.blocks, p.kernels, p.strides)):
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
    if p.output_spatial_pooling_params is not None:
      self.create_child('output_spatial_pooling',
                        p.output_spatial_pooling_params)

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
    p = self.hparams
    block_group_features = {}

    # Apply the entryflow conv.
    outputs = self.entryflow_conv(inputs)

    # Apply the entryflow maxpooling layer.
    outputs, _ = self.entryflow_maxpool(outputs)

    # Apply the ResNet blocks.
    for stage_id, num_blocks in enumerate(p.blocks):
      for block_id in range(num_blocks):
        block_name = f'stage_{stage_id}_block_{block_id}'
        instance = getattr(self, block_name)
        outputs = instance(outputs)

      if p.return_block_features:
        block_group_features[2 + stage_id] = outputs

    if p.return_block_features:
      return block_group_features

    # Apply optional spatial global pooling.
    if p.output_spatial_pooling_params is not None:
      outputs = self.output_spatial_pooling(outputs)
    return outputs
