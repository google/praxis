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

"""Activation layers."""
from __future__ import annotations
from typing import Type
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

  @classmethod
  def get_subclass_by_name(cls, name: str) -> Type[BaseActivation]:
    """Returns a subclass with the given name.

    Matching is case insensitive and ignores underscores.

    Args:
      name: Name of subclass.

    Returns:
      A subclass.

    Raises:
      KeyError: if cannot find a unique subclass that matches the name.
    """
    candidates = []
    normalized_name = name.replace('_', '').lower()
    for subcls in cls.__subclasses__():
      if subcls.__name__.lower() == normalized_name:
        candidates.append(subcls)
    if len(candidates) > 1:
      full_names = [k.__module__ + ':' + k.__qualname__ for k in candidates]
      raise KeyError(f'Found >1 activation layers named {name}: {full_names}')
    if not candidates:
      raise KeyError(f'Cannot found activation layer named {name}.')
    return candidates[0]


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
  """Leaky ReLU activation layer.

  Attributes:
    negative_slope: Negative slope of LEAKY_RELU.
  """
  negative_slope: float = 0.01

  def __call__(self, inputs: JTensor) -> JTensor:
    """Applies the activation function."""
    return jax.nn.leaky_relu(inputs, negative_slope=self.negative_slope)


class ELU(BaseActivation):
  """Exponential Linear Unit (ELU) activation layer."""

  def __call__(self, inputs: JTensor) -> JTensor:
    """Applies the activation function."""
    return jax.nn.elu(inputs)


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
  """Gaussian Error Linear Unit (GELU) activation layer.

  Attributes:
    approximate: Whether to use the approximate or exact formulation.
  """
  # By default, use approximate.
  approximate: bool = True

  def __call__(self, inputs: JTensor) -> JTensor:
    """Applies the activation function."""
    return jax.nn.gelu(inputs, approximate=self.approximate)


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
