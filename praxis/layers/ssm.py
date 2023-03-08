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

"""Implementation of the SSM-S4D layer.

S4 means Structured State Space for Sequence Modeling. See:
https://srush.github.io/annotated-s4/

S4D is its diagonal version which is simpler and performs similarly, see:
https://arxiv.org/abs/2206.11893.
"""

import jax
from jax import numpy as jnp
from jax.numpy.linalg import eigh

from praxis import base_layer
from praxis import py_utils
from praxis import pytypes

NestedMap = py_utils.NestedMap
WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams
JTensor = pytypes.JTensor

PARAMS = base_layer.PARAMS
AUX_LOSS = base_layer.AUX_LOSS
SUMMARIES = base_layer.SUMMARIES
NON_TRAINABLE = base_layer.NON_TRAINABLE
RANDOM = base_layer.RANDOM

# RNN share weights across time dimension, so PARAMS are never split.
SCAN_SPLIT_RNGS = {PARAMS: False, RANDOM: True}


################## Hippo matrix and its normal version. #################
def make_hippo(n, hippo_type):
  """Make Hippo matrix according to https://arxiv.org/abs/2111.00396."""
  if hippo_type == "LegS":
    pmat = jnp.sqrt(1 + 2 * jnp.arange(n))
    amat = pmat[:, jnp.newaxis] * pmat[jnp.newaxis, :]
    amat = jnp.tril(amat) - jnp.diag(jnp.arange(n))
  elif hippo_type == "LagT":
    amat = jnp.ones([n, n])
    amat = jnp.tril(amat)
  return -amat


def make_nplr_hippo(n, hippo_type):
  """Make a normalized Hippo."""
  nhippo = make_hippo(n, hippo_type)  # A

  if hippo_type == "LegS":
    # Add in a rank 1 term. Makes it Normal.
    pmat = jnp.sqrt(jnp.arange(n) + 0.5)
    # HiPPO also specifies the B matrix
    bmat = jnp.sqrt(2 * jnp.arange(n) + 1.0)
  elif hippo_type == "LagT":
    # Add in a rank 1 term. Makes it Normal.
    pmat = jnp.ones(n) * jnp.sqrt(0.5)
    # HiPPO also specifies the B matrix
    bmat = jnp.ones(n)
  return nhippo, pmat, bmat


def make_dplr_hippo(n, hippo_type):
  """Diagonalized NPLR representation for S4D."""
  amat, pmat, bmat = make_nplr_hippo(n, hippo_type)  # A_diag = {-1, ..., -N}

  smat = amat + pmat[:, jnp.newaxis] * pmat[jnp.newaxis, :]  # A^N

  # Check skew symmetry
  s_diag = jnp.diagonal(smat)  # S_diag = {-0.5, ..., -0.5}, real part only
  lambda_real = jnp.mean(s_diag) * jnp.ones_like(s_diag)
  # assert jnp.allclose(Lambda_real, S_diag, atol=1e-3)

  # Diagonalize S to V \Lambda V^*
  lambda_imag, vmat = eigh(smat * -1j)

  if hippo_type == "LagT":
    lambda_imag *= float(n)  # Scaled LagT

  pmat = vmat.conj().T @ pmat
  bmat = vmat.conj().T @ bmat

  # Lambda is the same as eig(S), but eig is not supported in TPU.
  return lambda_real + 1j * lambda_imag, pmat, bmat, vmat


############### SSM kernel and layer #####################
def causal_convolution(u, ker):
  """y = conv(ker, u)."""
  assert ker.shape[-2] >= u.shape[-2], "%d, %d" % (ker.shape[-2], u.shape[-2])
  u_len = u.shape[-2]
  k_trun = ker[:u_len, :]
  ud = jnp.fft.rfft(
      jnp.float32(u), n=2*u_len, axis=-2)
  kd = jnp.fft.rfft(
      jnp.float32(k_trun), n=2*u_len, axis=-2)
  out = ud * jnp.expand_dims(kd, 0)
  return jnp.fft.irfft(out, axis=-2)[:, :u_len, :].real


def s4d_step(ab, bb, cb, u_k, x_k_1):
  """x_k = Ab x_k_1 + Bb u_k , y_k = Cb x_k."""
  # A = [nh, d], B = [nh, d], u_k = [B, d]
  x_k = ab[None, :, :] * x_k_1 + bb[None, :, :] * u_k[:, None, :]  # [B, nh, d]
  # C = [nh, d]
  y_k = jnp.sum(cb[None, :, :] * x_k, axis=1)  # [B, d]
  return x_k, y_k.real

def s4d_kernel_zoh(c, a, l, step):
  """A version of the kernel specialized to B=1 and ZOH."""
  kernel_l = c[None, :, :] * (
      jnp.exp(step[None, :, :] * a[None, :, :]) - 1) / a[None, :, :] * jnp.exp(
          l[:, None, None] * step[None, :, :] * a[None, :, :])
  return jnp.sum(kernel_l, axis=1).real  # The sum means * B where B is all-one.


def s4d_discretize(a, b, step):
  """A [nh, d] where nh represents a diagonal matrix for an input channel."""
  # A[:, i] is an nh-dim diagonal, B[:, i] is an (nh, 1) vector.
  abar = jnp.exp(step * a)
  return abar, (abar - 1) / a * b


class SSM(base_layer.BaseLayer):
  """A generic SSM layer for 1D input.

     Attributes:
       nheads: number of heads per channel.
       dim: input dimension/channel size.
       l_max: longest seq length.
       decode_num_samples: How many decoding samples for each example
       step_size: the step size for SSM discretization.
       hippo_type: which type of hippo to use.
  """
  nheads: int = 0
  dim: int = 0
  l_max: int = 0
  decode_num_samples: int = 0
  step_size: float = 0.01
  hippo_type: str = "ss4d-1d"

  def init_s4d_ac(self, nh, suffix=""):
    wp = self.weight_split_dims_mapping

    # We freeze the A matrix without training, because it behaves similarly.
    if self.hippo_type.endswith("legs"):
      amat, _, _, _ = make_dplr_hippo(nh, "LegS")
      amat = jnp.transpose(jnp.tile(amat, [self.dim, 1]), [1, 0])
    elif self.hippo_type.endswith("lagt"):
      amat, _, _, _ = make_dplr_hippo(nh, "LagT")
      amat = jnp.transpose(jnp.tile(amat, [self.dim, 1]), [1, 0])
    else:
      a_im = jnp.transpose(jnp.tile(jnp.arange(
          -jnp.pi / 2. * (nh - 1),
          jnp.pi / 2. / nh * (nh - 1) ** 2,
          jnp.pi / nh * (nh - 1)), [self.dim, 1]), [1, 0])
      a_re = jnp.ones([nh, self.dim]) * -0.5
      amat = a_re + 1j * a_im

    c_re = self.create_variable(
        "C_re" + suffix,
        WeightHParams(
            shape=[nh, self.dim],
            init=WeightInit.Gaussian(0.5 ** 0.5),
            mesh_shape=self.mesh_shape,
            tensor_split_dims_mapping=wp.wt))
    c_im = self.create_variable(
        "C_im" + suffix,
        WeightHParams(
            shape=[nh, self.dim],
            init=WeightInit.Gaussian(0.5 ** 0.5),
            mesh_shape=self.mesh_shape,
            tensor_split_dims_mapping=wp.wt))
    cmat = c_re + 1j * c_im
    return amat, cmat

  def setup(self) -> None:
    wp = self.weight_split_dims_mapping

    # We freeze the step_size without training, because it behaves better.
    self.ss = jnp.ones([1, self.dim]) * self.step_size
    self.dmat = self.create_variable(
        "D",
        WeightHParams(
            shape=[1, self.dim],
            init=WeightInit.Constant(1.),
            mesh_shape=self.mesh_shape,
            tensor_split_dims_mapping=wp.wt))

    l = jnp.arange(self.l_max)
    self.amat, self.cmat = self.init_s4d_ac(self.nheads)  # [nh, d]
    self.ssm_k = s4d_kernel_zoh(self.cmat, self.amat, l, self.ss)  # [L, d]
    self.abar, self.bbar = s4d_discretize(
        self.amat,
        jnp.ones((self.nheads, self.dim)), self.ss)  # B is just ones.

  def init_states(self, batch_size: int) -> None:
    """Initialize the ssm state to all zeros."""
    state = jnp.zeros((batch_size, self.nheads, self.dim), dtype=jnp.complex64)
    self.update_decode_state("ssm_state", state)

  def extend_step(self,
                  inputs: JTensor) -> JTensor:
    """Extends SSM for one step on 'inputs' from the last 'ssm_state'.

    Args:
      inputs: A JTensor of inputs.

    Returns:
      outputs: A JTensor
    """
    batch_size = inputs.shape[0]
    if not self.has_variable(base_layer.DECODE_CACHE, "ssm_state"):
      self.init_states(batch_size)

    x = self.get_decode_state("ssm_state")
    assert x.shape[0] == batch_size, (x.shape[0], batch_size)

    x, y = s4d_step(self.abar, self.bbar, self.cmat, inputs, x)
    self.update_decode_state("ssm_state", x)
    return y + self.dmat * inputs

  def __call__(self, inputs: JTensor) -> JTensor:
    """Use convolution to compute outputs in parallel.

    Args:
      inputs: A JTensor of inputs.

    Returns:
      outputs: A JTensor
    """
    batch_size = inputs.shape[0]
    # the batch is x decode_num_samples in extend_step decoding.
    self.init_states(batch_size * self.decode_num_samples)

    y = causal_convolution(inputs, self.ssm_k)
    return y + self.dmat[None, :, :] * inputs

