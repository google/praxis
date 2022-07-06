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

"""Bregman PCA layer."""

import enum
import functools
from typing import Optional, Sequence, Tuple, Union

import jax
from jax import lax
from jax import nn
from jax import numpy as jnp
import jax.scipy.special as jsp
from praxis import base_layer
from praxis import pytypes

WeightHParams = base_layer.WeightHParams
WeightInit = base_layer.WeightInit
JTensor = pytypes.JTensor

BaseHParams = base_layer.BaseLayer.HParams
sub_config_field = base_layer.sub_config_field
PARAMS = base_layer.PARAMS


def _leaky_relu_loss(coefficients: JTensor, components: JTensor, mean: JTensor,
                     labels: JTensor, negative_slope: float) -> JTensor:
  """Calculates the leaky ReLU loss between the labels and reconstructions.

  Args:
    coefficients: PCA coefficients for reconstrcuting the representations.
    components: Principal components.
    mean: PCA mean.
    labels: Target labels.
    negative_slope: Negative slope of the leaky ReLU activation function.

  Returns:
    Average loss.
  """
  labels = lax.stop_gradient(jnp.reshape(labels, [labels.shape[0], -1]))
  labels_pre = nn.leaky_relu(labels, negative_slope=1. / negative_slope)
  preds_pre = jnp.tensordot(coefficients, components, 1) + mean
  preds_pre = jnp.reshape(preds_pre, [preds_pre.shape[0], -1])
  preds = nn.leaky_relu(preds_pre, negative_slope=negative_slope)
  loss = 0.5 * (preds_pre * preds - labels_pre * labels) - labels * (
      preds_pre - labels_pre)
  loss = jnp.mean(jnp.sum(loss, axis=1))
  return loss


def _squared_loss(coefficients: JTensor, components: JTensor, mean: JTensor,
                  labels: JTensor) -> JTensor:
  """Calculates the squared loss between the labels and reconstructions.

  Args:
    coefficients: PCA coefficients for reconstrcuting the representations.
    components: Principal components.
    mean: PCA mean.
    labels: Target labels.

  Returns:
    Average loss.
  """
  preds = jnp.tensordot(coefficients, components, 1) + mean
  labels = lax.stop_gradient(jnp.reshape(labels, [labels.shape[0], -1]))
  preds = jnp.reshape(preds, [preds.shape[0], -1])
  loss = 0.5 * jnp.square(preds - labels)
  loss = jnp.mean(jnp.sum(loss, axis=1))
  return loss


def _inverse_softmax(inputs: JTensor) -> JTensor:
  outputs = jnp.log(jnp.maximum(inputs, 1e-6))
  return outputs - jnp.mean(outputs, axis=-1, keepdims=True)


def _softmax_loss(coefficients: JTensor, components: JTensor, mean: JTensor,
                  labels: JTensor) -> JTensor:
  """Calculates the softmax loss between the labels and reconstructions.

  Args:
    coefficients: PCA coefficients for reconstrcuting the representations.
    components: Principal components.
    mean: PCA mean.
    labels: Target labels.

  Returns:
    Average loss.
  """
  logits = jnp.tensordot(coefficients, components, 1) + mean
  labels = lax.stop_gradient(jnp.reshape(labels, [-1, labels.shape[-1]]))
  logits = jnp.reshape(logits, [-1, logits.shape[-1]])
  loss = jsp.logsumexp(
      logits, axis=1) + jnp.sum(
          labels * (jnp.log(jnp.maximum(labels, 1e-6)) - logits), axis=1)
  loss = jnp.mean(loss)
  return loss


@enum.unique
class ActivationType(str, enum.Enum):
  """Enumeration with the activation function names supported by this module."""
  IDENTITY = 'IDENTITY'
  LEAKY_RELU = 'LEAKY_RELU'
  SOFTMAX = 'SOFTMAX'


class BregmanPCA(base_layer.BaseLayer):
  """Implements an online Bregman PCA layer.

  Bregman PCA is a generalized form of standard PCA that replaces the squared
  loss with a Bregman divergence. The Bregman PCA layer is used to learn the
  representation of a layer in an online fashion. The layer behaves as an
  identity layer during training and returns the inputs. It also returns the
  compression coefficients of the inputs using the current stored mean and
  principal components. The Bregman loss for learning the representation is
  tailored to the layer's activation function. The shape of the inputs (without
  the batch size) must be specified as input_dims.

  Reference: https://cseweb.ucsd.edu/~dasgupta/papers/pca.pdf

  We use the following capital letters to denote shape parameters:
    B = batch size
    K = number of components
  """

  class HParams(BaseHParams):
    """Associated hyperparams for this layer class.

    Attributes:
      num_components: Number of PCA components.
      input_dims: Dimensions of input
      activation_type: The type of the activation function to use. See the
        supported activation functions in the ActivationType enum above.
      negative_slope: Negative slope for leaky ReLU.
      mean_beta: EMA constant for updating the mean.
      coefficients_lr: Learning rate for the coefficients.
      coefficients_beta: EMA constant for the coefficients updates.
      coefficients_steps: Number of steps for solving the coefficients.
      components_lr: Learning rate for the PCA components.
      components_beta: EMA constant for the PCA components updates.
      start_step: Step number to start updating the components.
      end_step: Step number to end updating the components.
      constant_lr_schedule: Whether to use a constant learning rate schedule for
        the components. Applies a linearly decaying schedule if False.
    """
    num_components: int = 0
    input_dims: Union[int, Sequence[int]] = 0
    activation_type: ActivationType = ActivationType.IDENTITY
    negative_slope: float = 0.0
    mean_beta: float = 0.99
    coefficients_lr: float = 0.01
    coefficients_beta: float = 0.9
    coefficients_steps: int = 20
    components_lr: float = 0.01
    components_beta: float = 0.9
    start_step: int = 0
    end_step: int = 0
    constant_lr_schedule: bool = True

  def setup(self) -> None:
    """Constructs an instance with a mean and K principal components."""
    p = self.hparams
    assert p.num_components
    assert p.input_dims
    assert p.end_step > p.start_step
    input_dims = p.input_dims
    if not isinstance(input_dims, (list, tuple)):
      input_dims = [input_dims]
    elif isinstance(input_dims, tuple):
      input_dims = list(input_dims)

    step = WeightHParams(
        shape=[],
        # TODO(eamid): switch to an int32 step counter.
        init=WeightInit.Constant(0.0),
        dtype=jnp.float32,
        collections=[base_layer.WeightHParamsCollection.REQUIRES_MEAN_SYNC])
    mean = WeightHParams(
        shape=[1] + input_dims,
        init=WeightInit.Constant(0.0),
        collections=[base_layer.WeightHParamsCollection.REQUIRES_MEAN_SYNC])
    components = WeightHParams(
        shape=[p.num_components] + input_dims,
        collections=[base_layer.WeightHParamsCollection.REQUIRES_MEAN_SYNC])
    components_momentum = WeightHParams(
        shape=[p.num_components] + input_dims,
        init=WeightInit.Constant(0.0),
        collections=[base_layer.WeightHParamsCollection.REQUIRES_MEAN_SYNC])
    if p.activation_type == ActivationType.IDENTITY:
      self.activation_fn = lambda x: x
      self.inv_activation_fn = lambda x: x
      self.bregman_loss_fn = _squared_loss
    elif p.activation_type == ActivationType.LEAKY_RELU:
      self.activation_fn = functools.partial(
          jax.nn.leaky_relu, negative_slope=p.negative_slope)
      self.inv_activation_fn = functools.partial(
          jax.nn.leaky_relu, negative_slope=1. / p.negative_slope)
      self.bregman_loss_fn = functools.partial(
          _leaky_relu_loss, negative_slope=p.negative_slope)
    elif p.activation_type == ActivationType.SOFTMAX:
      self.activation_fn = nn.softmax
      self.inv_activation_fn = _inverse_softmax
      self.bregman_loss_fn = _softmax_loss
    else:
      raise ValueError('Unknown activation type {}'.format(p.activation_type))
    self.coefficients_grad_fn = jax.grad(self.bregman_loss_fn, argnums=0)
    self.components_grad_fn = jax.grad(self.bregman_loss_fn, argnums=1)
    self.create_variable('step', step, trainable=False)
    self.create_variable('mean', mean, trainable=False)
    self.create_variable('components', components, trainable=False)
    self.create_variable(
        'components_momentum', components_momentum, trainable=False)

  def base_learning_rate(self, step: JTensor) -> Tuple[JTensor, JTensor]:
    p = self.hparams
    constant_lr_schedule = p.constant_lr_schedule
    apply_update = jnp.where(
        jnp.logical_and(step >= p.start_step, step < p.end_step), 1.0, 0.0)
    base_lr = jnp.maximum(
        1. - (step - p.start_step) / (p.end_step - p.start_step), 0.0)
    base_lr = jnp.where(constant_lr_schedule, 1.0, base_lr)
    base_lr = base_lr * apply_update
    return base_lr, apply_update

  def reconstruct(self, coefficients: JTensor) -> JTensor:
    """Reconstructs the representation using the input coefficients.

    Args:
      coefficients: Input coefficients for the PCA components of shape [B, K].

    Returns:
      representations: Reconstructed representations of shape [B, ...].
    """
    mean = self.get_var('mean')
    components = self.get_var('components')
    representations = jnp.tensordot(coefficients, components, 1) + mean
    representations = self.activation_fn(representations)
    return representations

  def __call__(self,
               inputs: JTensor,
               mask: Optional[JTensor] = None) -> Tuple[JTensor, JTensor]:
    """Updates the PCA parameters.

    Args:
      inputs: Input tensor of shape [B, ...].
      mask: Mask tensor of shape [B, ...]. The zero entries are masked out when
        updating the components.

    Returns:
      inputs: Passes the inputs through. The inputs may be casted based on
      fprop_dtype.
      coefficients: PCA coefficients of inputs.
    """
    p = self.hparams
    mean = self.get_var('mean')
    components = self.get_var('components')
    components_momentum = self.get_var('components_momentum')
    inputs = self._cast_to_fprop_dtype(inputs)

    def _iter_condition(carry):
      count, _, _, _, _ = carry
      return count < p.coefficients_steps

    def _iter_body(carry):
      count, coeffs, coeffs_mom, components, mean = carry
      base_lr = 1. - count / p.coefficients_steps
      coeffs_grad = self.coefficients_grad_fn(coeffs, components, mean, inputs)
      coeffs_grad_norm = jnp.maximum(jnp.linalg.norm(coeffs_grad), 1e-6)
      coeffs_grad = coeffs_grad / coeffs_grad_norm * jnp.sqrt(
          jnp.size(coeffs_grad))
      coeffs_mom = (
          p.coefficients_beta * coeffs_mom +
          (1. - p.coefficients_beta) * coeffs_grad)
      coeffs -= base_lr * p.coefficients_lr * coeffs_mom
      return count + 1, coeffs, coeffs_mom, components, mean

    if not self.do_eval:
      step = self.get_var('step')
      base_lr, apply_update = self.base_learning_rate(step)
      self.update_var('step', self.get_var('step') + 1)
      mean_beta = 1. - apply_update * (1. - p.mean_beta)
      mean = mean_beta * mean + (1. - mean_beta) * self.inv_activation_fn(
          jnp.mean(inputs, axis=0, keepdims=True))
    coefficients = jnp.zeros((inputs.shape[0], p.num_components),
                             dtype=inputs.dtype)
    coefficients_momentum = jnp.zeros_like(coefficients)
    _, coefficients, _, _, _ = lax.while_loop(
        _iter_condition, _iter_body,
        (0, coefficients, coefficients_momentum, components, mean))
    coefficients = lax.stop_gradient(coefficients)
    masked_coefficients = coefficients
    if mask is not None:
      masked_coefficients *= mask
    if not self.do_eval:
      components_grad = self.components_grad_fn(masked_coefficients, components,
                                                mean, inputs)
      components_grad_norm = jnp.maximum(jnp.linalg.norm(components_grad), 1e-6)
      components_grad = components_grad / components_grad_norm * jnp.sqrt(
          jnp.size(components_grad))
      components_momentum = (p.components_beta * components_momentum) + (
          1. - p.components_beta) * components_grad * apply_update
      components -= base_lr * p.components_lr * components_momentum
      self.update_var('mean', mean)
      self.update_var('components', components)
    bregman_loss = self.bregman_loss_fn(coefficients, components, mean, inputs)
    self.add_summary('bregman_loss', bregman_loss)
    return inputs, coefficients
