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

"""Vector Quantization layers."""

import dataclasses
from typing import Optional

import jax
import jax.numpy as jnp
from praxis import base_layer
from praxis import py_utils
from praxis import pytypes
from praxis.layers import linears
from praxis.layers import quantizer_objectives as objectives

JTensor = jnp.ndarray
NestedMap = py_utils.NestedMap
WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams
JTensor = pytypes.JTensor

BaseHParams = base_layer.BaseLayer.HParams


# TODO(nanxinchen): merge this with ngrammer.VectorQuantization
def quantize_vector(latent: JTensor, codebook: JTensor):
  """Vector quantization.

  (port from TF impl of ... speech/quantizer/layers.py)

  Symbols in comments:
  B: batch_size.
  D: latent_dim.
  C: num_latent_classes per group
  G: num of codebook groups.

  Args:
    latent:   [B, D]
    codebook: [C, G, D // G]

  Returns:
    (quantized, codes, onehot).
    - quantized: [B, D]
    - codes:     [B, G]
    - onehot:    [B, G, C]
  """
  # For lower HBM footprint.
  assert len(codebook.shape) == 3
  b, d = latent.shape
  c, g = codebook.shape[:2]
  assert d % g == 0

  latent = jnp.reshape(latent, [b, g, d // g])

  # [B, G, C]
  distance = (
      # [b, g, 1]
      jnp.sum(latent**2, -1, keepdims=True) -
      # [b, g, c]
      2 * jnp.einsum('bgd,cgd->bgc', latent, codebook) +
      # [1, g, c]
      jnp.sum(jnp.transpose(codebook, [2, 1, 0])**2, 0, keepdims=True))
  # distance = py_utils.check_numerics(distance, 'quantization NaN')

  # [B, G]
  codes = jnp.argmin(distance, axis=-1)

  # [B, G, C]
  one_hot = jax.nn.one_hot(codes, c, axis=-1, dtype=jnp.float32)
  quantized = jnp.einsum('bgc,cgd->bgd', one_hot, codebook)
  quantized = jnp.reshape(quantized, [b, d])
  return quantized, codes, one_hot


class RandomVectorQuantizer(base_layer.BaseLayer):
  """Random quantization for BEST-RQ: https://arxiv.org/pdf/2202.01855.pdf."""

  class HParams(BaseHParams):
    """Associated hyper-params for this layer class.

    Attributes:
      latent_dim:         Input dimension.
      projection_dim:     Projection dimension.
      num_latent_classes: Number of random quantized classes.
      num_groups:         Number of quantized projections
      stack_ratio:        Stacking layer ratio.
      normalize_latent_vector: Normalize the L2 norm of each latent input vector
        to 1.
      normalize_codebook: Normalize the L2 norm of each codebook vector to 1.
      codebook_init:      Initialization for codebook.
      low_rank_codebook:  If true, when num_groups is 1, the shape of the
        codebook weight is [num_latent_classes, projection_dim] instead of
        [num_latent_classes, 1, projection_dim]. This is for checkpoint
        compatibility with old models.
      plot_codebook:      Whether to plot the codebook as an image summary.
    """
    latent_dim: Optional[int] = None
    projection_dim: int = 16
    num_latent_classes: Optional[int] = None
    num_groups: int = 1
    stack_ratio: int = 1
    normalize_latent_vector: bool = False
    normalize_codebook: bool = False
    codebook_init: WeightInit = dataclasses.field(
        default_factory=WeightInit.Gaussian)
    low_rank_codebook: bool = False
    plot_codebook: bool = False

  def _l2_normalize(self, x, axis, epsilon=1e-12):
    dis = jnp.sum(x * x, axis=axis, keepdims=True) + epsilon
    return x * jax.lax.rsqrt(dis)

  def setup(self) -> None:
    p = self.hparams
    assert p.stack_ratio >= 1

    if p.stack_ratio != 1:
      self.create_child(
          'stack',
          linears.StackingOverTime.HParams(
              left_context=0,
              right_context=p.stack_ratio - 1,
              stride=p.stack_ratio,
              padding_reduce_option='reduce_max'))

    self.create_variable(
        'random_proj',
        WeightHParams(
            shape=[p.latent_dim * p.stack_ratio, p.projection_dim],
            init=p.params_init,
            dtype=jnp.float32,
            collections=[
                base_layer.WeightHParamsCollection.SKIP_LP_REGULARIZATION
            ]))
    self.create_variable(
        'random_bias',
        WeightHParams(
            shape=[p.projection_dim],
            init=WeightInit.Constant(0.0),
            dtype=jnp.float32,
            collections=[
                base_layer.WeightHParamsCollection.SKIP_LP_REGULARIZATION
            ]))

    if p.num_groups == 1 and p.low_rank_codebook:
      codebook_shape = [p.num_latent_classes, p.projection_dim]
    else:
      codebook_shape = [
          p.num_latent_classes, p.num_groups, p.projection_dim // p.num_groups
      ]
    self.create_variable(
        'random_codebook',
        WeightHParams(
            shape=codebook_shape,
            init=p.codebook_init,
            dtype=jnp.float32,
            collections=[
                base_layer.WeightHParamsCollection.SKIP_LP_REGULARIZATION
            ]))

  def _get_codebook(self):
    """Gets the latent embedding."""
    p = self.hparams

    # Recovers codebook to 3d.
    if p.low_rank_codebook and p.num_groups == 1:
      codebook = self.theta.random_codebook[:, jnp.newaxis, :]
    else:
      codebook = self.theta.random_codebook

    if p.normalize_codebook:
      codebook = self._l2_normalize(codebook, -1)

    return codebook

  def __call__(self, z: JTensor, paddings: JTensor) -> NestedMap:
    p = self.hparams

    # Stacking.
    # [b, t // s, s * input_dim]
    if p.stack_ratio > 1:
      z, paddings = self.stack(z, paddings[:, :, jnp.newaxis])
      paddings = jnp.squeeze(paddings, -1)

    proj_vec = jnp.einsum('dh,btd->bth', self.theta.random_proj, z)
    proj_vec = proj_vec + self.theta.random_bias

    batch_size, time_steps, dim = proj_vec.shape
    proj_vec = jnp.reshape(proj_vec, [batch_size * time_steps, dim])

    if p.normalize_latent_vector:
      proj_vec = self._l2_normalize(proj_vec, -1)

    codebook = self._get_codebook()

    if p.plot_codebook:
      # Considered as [B, H, W, C] in summaries.
      codebook_plots = jnp.einsum('cgd->gcd', codebook)
      codebook_plots = jnp.tile(codebook_plots[..., jnp.newaxis], [1, 1, 1, 3])
      self.add_summary(
          'codebook',
          codebook_plots,
          summary_type=base_layer.SummaryType.IMAGE)

    q, c, onehot = quantize_vector(proj_vec, codebook)
    q = jnp.reshape(q, [batch_size, time_steps, dim])
    c = jnp.reshape(c, [batch_size, time_steps, p.num_groups])
    onehot = jnp.reshape(
        onehot, [batch_size, time_steps, p.num_groups, p.num_latent_classes])

    # Add number of groups
    if base_layer.is_running_under_pmap():
      pplx, entropy, _ = objectives.batch_pplx_entropy_from_codes(
          c,
          p.num_latent_classes,
          paddings=paddings,
          data_parallel_axis=base_layer.PMAP_PARALLEL_AXIS_NAME)
      codebook_coverage = objectives.batch_codebook_coverage(
          c,
          p.num_latent_classes,
          paddings=paddings,
          data_parallel_axis=base_layer.PMAP_PARALLEL_AXIS_NAME)
    else:
      pplx, entropy, _ = objectives.batch_pplx_entropy_from_codes(
          c, p.num_latent_classes, paddings=paddings)
      codebook_coverage = objectives.batch_codebook_coverage(
          c, p.num_latent_classes, paddings=paddings)

    codebook_num_covered_codes = codebook_coverage * p.num_latent_classes
    return NestedMap(
        z_q=jax.lax.stop_gradient(q),
        z_codes=jax.lax.stop_gradient(c),
        z_onehot=jax.lax.stop_gradient(onehot),
        paddings=paddings,
        codebook_coverage=codebook_coverage,
        codebook_num_covered_codes=codebook_num_covered_codes,
        pplx=pplx,
        entropy=entropy)

  def look_up(self, z_codes):
    """Looks up latent vectors [B, T, D] by z_codes [B, T, G]."""
    p = self.hparams
    b, t = z_codes.shape[:2]
    latent = jnp.einsum('btgc,cgd->btgd',
                        jax.nn.one_hot(z_codes, p.num_latent_classes),
                        self._get_codebook())
    # Stops the gradient to keep the codebook frozen.
    return jax.lax.stop_gradient(jnp.reshape(latent, [b, t, -1]))


class VectorQuantizer(base_layer.BaseLayer):
  """The VQ-VAE sequence vector quantizer.

  https://arxiv.org/abs/1711.00937

  Symbols in comments:
  B: batch_size.
  T: sequence length.
  D: latent_dim.
  C: num_latent_classes.
  G: num of codebook groups.
  """

  class HParams(BaseHParams):
    """Associated hyper-params for this layer class.

    Attributes:
      num_latent_classes: Number of latent classes.
      latent_dim: Latent vector dimension.
      beta: Scale of the commitment loss.
      normalize_latent_vector: Normalize the L2 norm of each latent input vector
        to 1.
      normalize_codebook: Normalize the L2 norm of each codebook vector to 1.
      num_groups: Num of codebook groups.
    """
    _attribute_overrides = ('params_init',)

    num_latent_classes: Optional[int] = None
    latent_dim: Optional[int] = None
    beta: Optional[float] = None
    normalize_latent_vector: bool = True
    normalize_codebook: bool = True
    num_groups: int = 1
    params_init: WeightInit = dataclasses.field(
        default_factory=WeightInit.UniformSqrtDim)

  def setup(self) -> None:
    p = self.hparams
    assert p.num_latent_classes
    assert p.latent_dim
    assert p.beta is not None
    assert p.beta >= 0
    assert p.latent_dim % p.num_groups == 0
    wp = base_layer.WeightHParams(
        shape=[
            p.num_latent_classes, p.num_groups, p.latent_dim // p.num_groups
        ],
        dtype=jnp.float32)

    # [C, D]
    self.create_variable('w', wp)

  def _l2_normalize(self, x, axis, epsilon=1e-12):
    norm = jnp.sqrt(jnp.sum(x * x, axis=axis, keepdims=True) + epsilon)
    return x / norm

  def _get_latent_embedding(self):
    """Gets the latent embedding."""
    p = self.hparams
    w = self.theta.w
    if p.normalize_codebook:
      w = self._l2_normalize(w, -1)
    return w

  def _apply_mask(self, x, mask):
    x_rank = len(x.shape)
    mask_rank = len(mask.shape)
    mask = jnp.reshape(mask, mask.shape + tuple([1] * (x_rank - mask_rank)))
    return x * mask.astype(x.dtype)

  def __call__(self, z: JTensor, paddings: JTensor) -> NestedMap:
    """Quantizes 'z' of shape [B, T, D].

    The z_codes of padded locations are 0.

    Args:
      z:        [B, T, D].
      paddings: [B, T].

    Returns:
      A NestedMap of
        - z_q:               [B, T, D].
        - z_codes:           [B, T, G].
        - z_onehot:          [B, T, G, C].
        - loss:              [], weighted sum of quantization loss and
          commitment loss.
        - codebook_coverage: [], a float scalar tensor between [0, 1].
        - pplx:              [], pplx of quantized distribution over the
          codebook.
        - entropy:           [], exp(pplx).
    """
    p = self.hparams
    b, t, d = z.shape
    g, c = p.num_groups, p.num_latent_classes

    mask = 1.0 - paddings
    num_frames = jnp.sum(mask)
    z = self._apply_mask(z, mask)

    if p.normalize_latent_vector:
      z = self._l2_normalize(z, axis=-1)

    # [b * t, d], [b * t, g], [b * t, g, c]
    z_q, z_codes, z_onehot = quantize_vector(
        jnp.reshape(z, [b * t, d]), self._get_latent_embedding())

    z_q = jnp.reshape(z_q, [b, t, d])
    z_codes = jnp.reshape(z_codes, [b, t, g])
    z_onehot = jnp.reshape(z_onehot, [b, t, g, c])

    # Padded locations are all 0s without any 1.
    z_q = self._apply_mask(z_q, mask)
    # [b, t, g]
    z_codes = self._apply_mask(z_codes, mask)
    # [b, t, g, c]
    z_onehot = self._apply_mask(z_onehot, mask)

    # Move z towards z_q.
    normalizer = 1e-7 + num_frames
    # [b, t, d]
    loss_c = (z - jax.lax.stop_gradient(z_q))**2
    # [b, t, d] -> [b, t] -> []
    loss_c = jnp.sum(jnp.mean(loss_c, -1)) / normalizer
    # loss_c = py_utils.check_numerics(loss_c, 'loss_c has NaN.')

    # Move z_q towards z.
    loss_z = (z_q - jax.lax.stop_gradient(z))**2
    loss_z = jnp.sum(jnp.mean(loss_z, -1)) / normalizer
    # loss_z = py_utils.check_numerics(loss_z, 'loss_z has NaN.')
    loss = loss_z + p.beta * loss_c

    # Straight-through estimator.
    # Doesn't look like this line does anyhing besides stopping gradient ??
    z_q = z + jax.lax.stop_gradient(z_q - z)

    # [], []
    if base_layer.is_running_under_pmap():
      pplx, entropy, _ = objectives.batch_pplx_entropy_from_codes(
          z_codes,
          c,
          paddings=paddings,
          data_parallel_axis=base_layer.PMAP_PARALLEL_AXIS_NAME)
      codebook_coverage = objectives.batch_codebook_coverage(
          z_codes,
          c,
          paddings=paddings,
          data_parallel_axis=base_layer.PMAP_PARALLEL_AXIS_NAME)
    else:
      pplx, entropy, _ = objectives.batch_pplx_entropy_from_codes(
          z_codes, c, paddings=paddings)
      codebook_coverage = objectives.batch_codebook_coverage(
          z_codes, c, paddings=paddings)
    codebook_num_covered_words = codebook_coverage * c**g

    return py_utils.NestedMap(
        z_q=z_q,
        z_codes=z_codes,
        z_onehot=z_onehot,
        loss=loss,
        codebook_coverage=codebook_coverage,
        codebook_num_covered_words=codebook_num_covered_words,
        pplx=pplx,
        entropy=entropy)

  def look_up(self, z_codes):
    """Looks up latent vectors [B, T, D] by z_codes [B, T, G]."""
    p = self.hparams
    b, t = z_codes.shape[:2]
    latent = jnp.einsum('btgc,cgd->btgd',
                        jax.nn.one_hot(z_codes, p.num_latent_classes),
                        self._get_latent_embedding())
    return jnp.reshape(latent, [b, t, -1])
