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

"""Activation layers."""

import jax
import jax.numpy as jnp
from praxis import base_layer
from praxis import py_utils
from praxis import pytypes

NestedMap = py_utils.NestedMap
JTensor = pytypes.JTensor

BaseHParams = base_layer.BaseLayer.HParams


class Activation(base_layer.BaseLayer):
  """Activation layer that wraps popular activation functions."""

  class HParams(BaseHParams):
    """Associated hyperparams for this layer class.

    Attributes:
      activation: Activation function to use. Options are RELU, RELU6, RELU^2,
        RELU^3, LEAKY_RELU, SIGMOID, TANH, GELU, EXACT_GELU, SILU, SWISH, NONE.
      negative_slope: Negative slope of LEAKY_RELU.
    """
    activation: str = 'RELU'
    negative_slope: float = 0.01

  def fprop(self, inputs: JTensor) -> JTensor:
    p = self.hparams
    if p.activation == 'RELU':
      outputs = jax.nn.relu(inputs)
    elif p.activation == 'RELU6':
      outputs = jax.nn.relu6(inputs)
    elif p.activation == 'RELU^2':
      outputs = jax.nn.relu(inputs)
      outputs = jnp.square(outputs)
    elif p.activation == 'RELU^3':
      outputs = jax.nn.relu(inputs)
      outputs *= jnp.square(outputs)
    elif p.activation == 'LEAKY_RELU':
      outputs = jax.nn.leaky_relu(inputs, negative_slope=p.negative_slope)
    elif p.activation == 'SIGMOID':
      outputs = jax.nn.sigmoid(inputs)
    elif p.activation == 'TANH':
      outputs = jax.nn.tanh(inputs)
    elif p.activation == 'GELU':
      outputs = jax.nn.gelu(inputs)
    elif p.activation == 'EXACT_GELU':
      # By default `tf.nn.gelu` is exact.
      outputs = jax.nn.gelu(inputs, approximate=False)
    elif p.activation == 'SILU':
      outputs = jax.nn.silu(inputs)
    elif p.activation == 'SWISH':
      outputs = jax.nn.swish(inputs)
    else:  # 'NONE'
      outputs = inputs
    return outputs
