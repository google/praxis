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

"""Vanilla (Skip-free, Batch-norm free) layers."""

from __future__ import annotations

import math
from typing import Optional, Sequence

from jax import nn
from praxis import base_layer
from praxis import py_utils
from praxis import pytypes
from praxis.layers import convolutions
from praxis.layers import poolings

NestedMap = py_utils.NestedMap
JTensor = pytypes.JTensor

BaseHParams = base_layer.BaseLayer.HParams
sub_config_field = base_layer.sub_config_field


def tailored_lrelu(negative_slope, x):
  return math.sqrt(2.0 / (1 + negative_slope**2)) * nn.leaky_relu(
      x, negative_slope=negative_slope)


class VanillaBlock(base_layer.BaseLayer):
  """Vanilla Block."""

  class HParams(BaseHParams):
    """Associated hyper-params for this layer class.

    Attributes:
      input_dim: Input dimension.
      output_dim: Output dimension.
      conv_params: Which Conv block to use.
      kernel_size: Kernel sizes of the block.
      stride: Stride.
      negative_slope: Negative slope for leaky relu.
    """
    input_dim: int = 0
    output_dim: int = 0
    conv_params: BaseHParams = convolutions.Conv2D.HParams(
        # TODO(b/225770692): Migrate weight initializer into dataclasses.
        bias=True,
        params_init=base_layer.WeightInit.ScaledDeltaOrthogonal(1.0))
    kernel_size: int = 3
    stride: int = 1
    negative_slope: float = 0.4

  def setup(self) -> None:
    p = self.hparams

    body = []
    # conv_in, reduce the hidden dims by 4
    body.append(p.conv_params.clone().set(
        name='conv_in',
        filter_shape=(1, 1, p.input_dim, p.output_dim // 4),
        filter_stride=(1, 1)))

    # conv_mid using the kernel size and stride provided
    body.append(p.conv_params.clone().set(
        name='conv_mid',
        filter_shape=(p.kernel_size, p.kernel_size, p.output_dim // 4,
                      p.output_dim // 4),
        filter_stride=(p.stride, p.stride)))

    # conv_out, expand back to hidden dim
    body.append(p.conv_params.clone().set(
        name='conv_out',
        filter_shape=(1, 1, p.output_dim // 4, p.output_dim),
        filter_stride=(1, 1)))
    self.create_children('body', body)

  def __call__(self, inputs: JTensor) -> JTensor:
    """Forward propagation of a VanillaBlock.

    Args:
      inputs: A `.JTensor` as inputs of [B, H, W, D_in] also commonly known as
        NHWC format.

    Returns:
      A `.JTensor` as outputs of shape [B, H', W', D_out].
    """
    p = self.hparams
    outputs = inputs

    for i in range(len(self.body)):
      outputs = tailored_lrelu(p.negative_slope, self.body[i](outputs))
    return outputs


class VanillaNet(base_layer.BaseLayer):
  """VanillaNet model without skip-connection or batch-norm mirroring ResNets.

  https://openreview.net/forum?id=U0k7XNTiFEq

  Raises:
    ValueError if length of `strides`, `channels`, `blocks` and `kernels` do
    not match.
  """

  class HParams(BaseHParams):
    """Associated hyper-params for this layer class.

    Attributes:
      conv_params: A layer params template specifying Conv-BN-Activation
        template used by the VanillaNet model.
      block_params: A layer params template specifying Convolution Block used in
        each stage. We use the same VanillaNetBlock tpl in all stages (4 stages
        in total) in VanillaNet.
      strides: A list of integers specifying the stride for each stage. A stage
        is defined as a stack of Convolution Blocks that share same type,
        channels and kernels. The stride is always applied only at the beginning
        of each stage, while within that stage, all other strides are set to 1
        (no stride).
      channels: A list of integers specifying the number of channels at
        different stages. The first channel is usually 4x the input dim.
      blocks: A list of integers specifying the number of blocks at different
        stages.
      kernels: A list of integers specifying the number of kernel sizes at
        different stages.
      entryflow_conv_kernel: A tuple of two integers as the kernel size of
        entryflow convolution.
      entryflow_conv_stride: A tuple of two integers as the stride of entryflow
        convolution.
      output_spatial_pooling_params: A layer params template specifying spatial
        pooling before output. If None, spatial pooling is not added.
      negative_slope: Negative slope for leaky relu.
    """
    conv_params: BaseHParams = convolutions.Conv2D.HParams(
        bias=True, params_init=base_layer.WeightInit.ScaledDeltaOrthogonal(1.0))
    block_params: BaseHParams = sub_config_field(VanillaBlock.HParams)
    strides: Sequence[int] = (1, 2, 2, 2)
    channels: Sequence[int] = (256, 512, 1024, 2048)
    blocks: Sequence[int] = (3, 4, 6, 3)
    kernels: Sequence[int] = (3, 3, 3, 3)
    entryflow_conv_kernel: Sequence[int] = (7, 7)
    entryflow_conv_stride: Sequence[int] = (2, 2)
    output_spatial_pooling_params: Optional[BaseHParams] = sub_config_field(
        poolings.GlobalPooling.HParams)
    negative_slope: float = 0.4

  @classmethod
  def HParamsVanillaNet5(cls) -> VanillaNet.HParams:
    """Returns VanillaNet5 hyperparams for testing purposes."""
    return cls.HParams(strides=[1], channels=[16], blocks=[1], kernels=[1])

  @classmethod
  def HParamsVanillaNet50(cls) -> VanillaNet.HParams:
    """Returns commonly used VanillaNet50 hyperparams."""
    return cls.HParams()

  @classmethod
  def HParamsVanillaNet101(cls) -> VanillaNet.HParams:
    """Returns commonly used VanillaNet101 hyperparams."""
    return cls.HParams(blocks=[3, 4, 23, 3])

  @classmethod
  def HParamsVanillaNet152(cls) -> VanillaNet.HParams:
    """Returns commonly used VanillaNet152 hyperparams."""
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

    block_p_tpl = p.block_params.clone().set(negative_slope=p.negative_slope)
    # Set the convolution type used in the Resnet block.
    if hasattr(block_p_tpl, 'conv_params'):
      block_p_tpl.conv_params = p.conv_params

    # Create the entryflow convolution layer.
    input_dim = p.channels[0] // 4
    entryflow_conv_params = p.conv_params.clone()
    entryflow_conv_params.filter_shape = (p.entryflow_conv_kernel[0],
                                          p.entryflow_conv_kernel[1], 3,
                                          input_dim)
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
        block_p = block_p_tpl.clone().set(
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
    """Applies the VanillaNet model to the inputs.

    Args:
      inputs: Input image tensor of shape [B, H, W, 3].

    Returns:
      Output tensor of VanillaNet of shape [B, H, W, D] where D is the last
      channel
      dimension. If `output_spatial_pooling_params` is not None, then the
      output will be of a shape that depends on which dims are pooled. For e.g.,
      if the pooling dims are [1, 2], then output shape will be [B, D].
    """
    p = self.hparams

    # Apply the entryflow conv.
    outputs = tailored_lrelu(p.negative_slope, self.entryflow_conv(inputs))

    # Apply the entryflow maxpooling layer.
    outputs, _ = self.entryflow_maxpool(outputs)

    # Apply the VanillaNet blocks.
    for stage_id, num_blocks in enumerate(p.blocks):
      for block_id in range(num_blocks):
        block_name = f'stage_{stage_id}_block_{block_id}'
        outputs = getattr(self, block_name)(outputs)

    # Apply optional spatial global pooling.
    if p.output_spatial_pooling_params is not None:
      outputs = self.output_spatial_pooling(outputs)
    return outputs
