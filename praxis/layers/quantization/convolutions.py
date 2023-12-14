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

"""Convolutional layers."""

from jax import numpy as jnp
from praxis import base_layer
from praxis import pytypes
from praxis.layers import convolutions
from praxis.layers import normalizations
from praxis.layers.quantization import quantization_hparams
from praxis.layers.quantization import quantizer

WeightHParams = base_layer.WeightHParams
QuantizationParams = quantization_hparams.QuantizationParams
instance_field = base_layer.instance_field
JTensor = pytypes.JTensor


class Conv2D(convolutions.Conv2D, quantizer.QuantizationLayer):  # pytype: disable=signature-mismatch
  """Conv2D with support of SAME/VALID paddings."""

  _PACK_4BIT_DIM = 0

  def setup(self) -> None:
    self.check_dimensions()
    wp = self.weight_split_dims_mapping
    pc = WeightHParams(
        shape=self.filter_shape,
        mesh_shape=self.mesh_shape,
        tensor_split_dims_mapping=wp.wt,
        dtype=self.dtype,
        init=self.kernel_init,
    )
    self.set_up_weights(
        weight_name='w',
        weight_params=pc,
        scale_shape=[self.filter_shape[-1]],
    )

    if self.bias:
      self.create_variable(
          'b',
          WeightHParams(
              shape=[self.filter_shape[-1]],
              init=self.bias_init,
              dtype=self.dtype,
          ),
      )

    wn = self.weight_norm_tpl.clone().set(dim=self.filter_shape[-1])
    self.weight_norm: normalizations.BaseNormalization
    self.create_child('weight_norm', wn)

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
    # Check if the feature_group_count is compatible with the inputs and filter
    # For more information see XLA docs on ConvWithGeneralPadding below
    # https://www.tensorflow.org/xla/operation_semantics#convwithgeneralpadding_convolution
    if inputs.shape[3] % self.filter_shape[2] != 0:
      raise ValueError(
          f'Input features {inputs.shape[3]} must be a'
          f'multiple of filter input dim {self.filter_shape[2]} '
          f'(Input shape: {inputs.shape}, '
          f'filter shape: {self.filter_shape}).'
      )
    # feature group count is D_in // filter input dim
    feature_group_count = inputs.shape[3] // self.filter_shape[2]
    if self.filter_shape[3] % feature_group_count != 0:
      raise ValueError(
          f'Filter output dim {self.filter_shape[3]} must be a '
          f'multiple of feature group count {feature_group_count} '
          f'(Input shape: {inputs.shape}, '
          f'filter shape: {self.filter_shape}).'
      )
    padding = self._compute_padding(inputs.shape)
    inputs = self._shard_bhwc(inputs.astype(self.fprop_dtype))

    # The `dimension_numbers=('NHWC', 'HWIO', 'NHWC')` is to be consistent
    # with tf.conv2d, see e.g., see
    # https://github.com/google/jax/blob/main/jax/_src/lax/lax.py#L622
    dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
    outputs = quantizer.quantized_conv(
        self, inputs, padding, dimension_numbers, feature_group_count
    )
    outputs = self._shard_bhwc(outputs)
    if self.bias:
      outputs += jnp.reshape(self.theta.b, (1,) * (outputs.ndim - 1) + (-1,))
    return outputs
