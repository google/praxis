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
from praxis import pytypes

JTensor = pytypes.JTensor


class BaseActivation(base_layer.BaseLayer):
  """Base class for activation functions."""

  def __call__(self, inputs: JTensor) -> JTensor:
    """Applies the activation function."""
    raise NotImplementedError(
        'Activation layers are expected to implement __call__().')


class ReLU(BaseActivation):
  """Rectified Linear Unit (ReLU) activation layer."""

  def __call__(self, inputs: JTensor) -> JTensor:
    """Applies the activation function."""
    return jax.nn.relu(inputs)


class ReLU6(BaseActivation):
  """ReLU6 activation layer."""

  def __call__(self, inputs: JTensor) -> JTensor:
    """Applies the activation function."""
    return jax.nn.relu6(inputs)


class SquaredReLU(BaseActivation):
  """Squared ReLU activation layer."""

  def __call__(self, inputs: JTensor) -> JTensor:
    """Applies the activation function."""
    outputs = jax.nn.relu(inputs)
    return jnp.square(outputs)


class CubedReLU(BaseActivation):
  """Cubed ReLU activation layer."""

  def __call__(self, inputs: JTensor) -> JTensor:
    """Applies the activation function."""
    outputs = jax.nn.relu(inputs)
    outputs *= jnp.square(outputs)
    return outputs


class LeakyReLU(BaseActivation):
  """Leaky ReLU activation layer."""

  class HParams(base_layer.BaseLayer.HParams):
    """Associated hyperparams for this layer class.

    Attributes:
      negative_slope: Negative slope of LEAKY_RELU.
    """
    negative_slope: float = 0.01

  def __call__(self, inputs: JTensor) -> JTensor:
    """Applies the activation function."""
    p = self.hparams
    return jax.nn.leaky_relu(inputs, negative_slope=p.negative_slope)


class Sigmoid(BaseActivation):
  """Sigmoid activation layer."""

  def __call__(self, inputs: JTensor) -> JTensor:
    """Applies the activation function."""
    return jax.nn.sigmoid(inputs)


class Tanh(BaseActivation):
  """Tanh activation layer."""

  def __call__(self, inputs: JTensor) -> JTensor:
    """Applies the activation function."""
    return jax.nn.tanh(inputs)


class GELU(BaseActivation):
  """Gaussian Error Linear Unit (GELU) activation layer."""

  class HParams(base_layer.BaseLayer.HParams):
    """Associated hyperparams for this layer class.

    Attributes:
      approximate: Whtether to use the approximate or exact formulation.
    """
    # By default `tf.nn.gelu` is exact.
    approximate: bool = True

  def __call__(self, inputs: JTensor) -> JTensor:
    """Applies the activation function."""
    p = self.hparams
    return jax.nn.gelu(inputs, approximate=p.approximate)


class SiLU(BaseActivation):
  """Sigmoid Linear Unit (SiLU) activation layer."""

  def __call__(self, inputs: JTensor) -> JTensor:
    """Applies the activation function."""
    return jax.nn.silu(inputs)


class Swish(BaseActivation):
  """Swish activation layer."""

  def __call__(self, inputs: JTensor) -> JTensor:
    """Applies the activation function."""
    return jax.nn.swish(inputs)


class Softplus(BaseActivation):
  """Softplus activation layer."""

  def __call__(self, inputs: JTensor) -> JTensor:
    """Applies the activation function."""
    return jax.nn.softplus(inputs)


class Exp(BaseActivation):
  """Exp activation layer."""

  def __call__(self, inputs: JTensor) -> JTensor:
    """Applies the activation function."""
    return jnp.exp(inputs)


class Identity(BaseActivation):
  """Identity (or lack of non-linear) activation layer."""

  def __call__(self, inputs: JTensor) -> JTensor:
    """Applies the activation function."""
    return inputs
