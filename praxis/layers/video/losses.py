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

"""Loss functions for vqvae/vqgan models."""

from collections.abc import Callable
import jax
import jax.numpy as jnp
from praxis import base_layer
from praxis import base_model
from praxis import py_utils
from praxis import pytypes

JTensor = pytypes.JTensor


def r1_gradient_penalty(
    inputs: JTensor,
    logits_fn: Callable[[JTensor], JTensor],
    grad_penalty_cost: float = 10.0,
) -> tuple[JTensor, JTensor]:
  """Calculates gradients penalty loss to regularize the discriminator.

  Args:
    inputs: A tensor of image inputs.
    logits_fn: A function that takes inputs and returns logits.
    grad_penalty_cost: scalar weight for the gradient penalty loss.

  Returns:
    A tuple of logits and the gradient penalty.
  """
  out, vjp_fn = jax.vjp(logits_fn, inputs, has_aux=False)
  # Check if jax.value_and_grad is more efficient than jax.vjp at scale.
  grad = vjp_fn(jnp.ones_like(out))[0]
  flattened_grad = jnp.asarray(grad.reshape((inputs.shape[0], -1)), jnp.float32)
  penalty = (
      jnp.mean(jnp.sum(jnp.square(flattened_grad), axis=-1)) * grad_penalty_cost
  )
  return out, penalty


def _discriminator_loss(logits_real: JTensor, logits_fake: JTensor) -> JTensor:
  """Calculates non-saturating discriminator loss."""
  d_loss_real = jax.nn.softplus(-logits_real)
  d_loss_fake = jax.nn.softplus(logits_fake)
  return jnp.mean(d_loss_real) + jnp.mean(d_loss_fake)


def _generator_loss(logits_fake):
  """Calculates non-saturating generator loss."""
  return jnp.mean(jax.nn.softplus(-logits_fake))


class VQGANLoss(base_layer.BaseLayer):
  """Loss layer for VQGAN."""

  g_adversarial_loss_weight: float = 0.1
  reconstruction_loss_weight: float = 5.0
  polyak_decay: float = 0.999
  lecam_weight: float = 0.001

  def lecam_loss(self, real_pred: JTensor, fake_pred: JTensor) -> JTensor:
    """Calculates lecam loss.

    Described in https://arxiv.org/abs/2104.03310

    Args:
      real_pred: scalar, predictions for the real samples.
      fake_pred: scalar, prdictions for the reconstructed (fake) samples.

    Returns:
      Lecam regularization loss (scalar).
    """
    ema_fake_pred = self.get_var('ema_fake_pred')
    ema_real_pred = self.get_var('ema_real_pred')
    return jnp.mean(
        jnp.power(jax.nn.relu(real_pred - ema_fake_pred), 2)
    ) + jnp.mean(jnp.power(jax.nn.relu(ema_real_pred - fake_pred), 2))

  def setup(self):
    """Constructs this jax module and registers variables."""
    decay_factor_hparams = base_layer.WeightHParams(
        shape=[],
        init=base_layer.WeightInit.Constant(0.0),
        dtype=jnp.float32,
        collections=[base_layer.WeightHParamsCollection.REQUIRES_MEAN_SYNC],
    )

    self.create_variable('ema_real_pred', decay_factor_hparams, trainable=False)
    self.create_variable('ema_fake_pred', decay_factor_hparams, trainable=False)

  def __call__(
      self, predictions: base_model.Predictions, input_batch: py_utils.NestedMap
  ) -> py_utils.NestedMap:
    original_video = input_batch.video
    reconstructed = predictions['reconstructed']
    logits_real = predictions['logits_real']
    logits_fake = predictions['logits_fake']
    real_pred = jnp.mean(logits_real)
    fake_pred = jnp.mean(logits_fake)

    ema_fake_pred = self.get_var('ema_fake_pred')
    ema_real_pred = self.get_var('ema_real_pred')
    ema_fake_pred = (
        fake_pred * (1 - self.polyak_decay) + ema_fake_pred * self.polyak_decay
    )
    ema_real_pred = (
        real_pred * (1 - self.polyak_decay) + ema_real_pred * self.polyak_decay
    )
    self.update_var('ema_fake_pred', ema_fake_pred)
    self.update_var('ema_real_pred', ema_real_pred)

    losses = py_utils.NestedMap()
    losses.grad_penalty = predictions['r1_gradient_penalty']
    losses.lecam_loss = (
        self.lecam_loss(logits_real, logits_fake) * self.lecam_weight
    )

    losses.d_adversarial_loss = _discriminator_loss(logits_real, logits_fake)
    losses.g_adversarial_loss = (
        _generator_loss(logits_fake) * self.g_adversarial_loss_weight
    )

    diff = jnp.asarray(original_video - reconstructed, jnp.float32)

    losses.reconstruction_loss = (
        jnp.mean(jnp.square(diff)) * self.reconstruction_loss_weight
    )
    losses.perceptual_loss = jnp.array(0.0, dtype=jnp.float32)
    if self.do_eval:
      losses.quantizer_loss = jnp.zeros_like(losses.reconstruction_loss)
    else:
      losses.quantizer_loss = predictions['quantizer_loss']
    losses.d_loss = (
        losses.d_adversarial_loss + losses.grad_penalty + losses.lecam_loss
    )
    losses.g_loss = (
        losses.reconstruction_loss
        + losses.g_adversarial_loss
        + losses.perceptual_loss
        + losses.quantizer_loss
    )
    return losses
