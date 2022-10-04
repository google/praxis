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

"""Normalization layers."""

from typing import List, Optional, Tuple

import jax
from jax import numpy as jnp
from praxis import asserts
from praxis import base_layer
from praxis import py_utils
from praxis import pytypes

NestedMap = py_utils.NestedMap
WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams
PMAP_PARALLEL_AXIS_NAME = base_layer.PMAP_PARALLEL_AXIS_NAME
JTensor = pytypes.JTensor
PARAMS = base_layer.PARAMS

BaseHParams = base_layer.BaseLayer.HParams


def compute_moments(
    inputs: JTensor,
    padding: JTensor,
    reduce_over_dims: List[int],
    cumulative_axis: Optional[int] = None,
    keepdims: bool = False,
) -> Tuple[JTensor, JTensor]:
  """Computes mean and variance over the valid data points in inputs.

  Args:
    inputs: The inputs JTensor.
    padding: The paddings JTensor.
    reduce_over_dims: A sequence of ints for dimensions to reduce `inputs` over.
    cumulative_axis: An optional int for axis to compute a cumulative sum. If
      none, there will be no cumulative sum applied.
    keepdims: A boolean indicating whether summations reduction axes should be
      left in the result as dimensions with size one.

  Returns:
    Tuple of (mean, variance).
  """
  asserts.eq(inputs.ndim, padding.ndim)
  rank = inputs.ndim
  for dim in reduce_over_dims:
    asserts.between(dim, 0, rank, left_strict=False, right_strict=True)
  mask = 1.0 - padding
  sum_v = jnp.sum(inputs * mask, axis=reduce_over_dims, keepdims=keepdims)
  count_v = jnp.sum(
      jnp.ones_like(inputs) * mask, axis=reduce_over_dims, keepdims=keepdims)
  if cumulative_axis is not None:
    sum_v = jnp.cumsum(sum_v, axis=cumulative_axis)
    count_v = jnp.cumsum(count_v, axis=cumulative_axis)

  if base_layer.is_running_under_pmap():
    # Here the aggregated mean & variance is the true mean & variance of all
    # workers' samples even when they have different batch sizes. This
    # property is like PyTorch's SyncBatchNorm, but unlike Flax's BatchNorm.
    sum_v = jax.lax.psum(sum_v, axis_name=PMAP_PARALLEL_AXIS_NAME)
    count_v = jax.lax.psum(count_v, axis_name=PMAP_PARALLEL_AXIS_NAME)

  count_v = jnp.maximum(count_v, 1.0)
  mean = sum_v / count_v
  sum_vv = jnp.sum(
      (inputs - mean) * (inputs - mean) * mask,
      axis=reduce_over_dims,
      keepdims=keepdims)
  if cumulative_axis is not None:
    sum_vv = jnp.cumsum(sum_vv, axis=cumulative_axis)

  if base_layer.is_running_under_pmap():
    sum_vv = jax.lax.psum(sum_vv, axis_name=PMAP_PARALLEL_AXIS_NAME)

  variance = sum_vv / count_v
  return mean, variance


class BaseNormalization(base_layer.BaseLayer):
  """Base class for normalization layers."""

  class HParams(base_layer.BaseLayer.HParams):
    """Associated hyper-params for this layer class.

    Attributes:
      dim: Depth of the input and output.
    """
    dim: int = 0

  def __call__(self,
               inputs: JTensor,
               paddings: Optional[JTensor] = None) -> JTensor:
    """Applies the normalization."""
    raise NotImplementedError(
        'Normalization layers are expected to implement fprop().')


class BatchNorm(BaseNormalization):
  """Batch normalization layer.

  Note that statistics are aggregated across
  all workers (a.k.a. Sync BatchNorm) in both pmap and pjit mode.

  Note that gamma in this layer is reparameterized: the gamma variable is
  0-initialized and input is scaled by (1 + gamma). This is different from
  Flax, tf.layers, PyTorch etc., where gamma is by default 1-initialized
  and used to scale the input. The difference is that in our version,
  weight decay encourages gamma to move towards 1, instead of 0.
  """

  class HParams(BaseNormalization.HParams):
    """Associated hyper-params for this layer class.

    Attributes:
      decay: Decay in updating the mean and variance moving average used in
        batch normalization.
      use_moving_avg_in_training: If True, use global moving avg (mean,
        variance) during training to avoid mismatch between train and eval,
        which then essentially acts as an adaptive normalization step. When this
        is set to True, it also disables the use of beta and gamma variables.
      set_padded_output_to_zero: If True, sets the padded outputs to zero.
      force_eval_mode: If True, puts the layer in eval mode even if
        self.do_eval is False. This does not disable training of beta/gamma,
        which has to be done separately (e.g. via bprop_variable_exclusion).
        This is commonly used in object detection when using pretrained
        backbones.
      gamma_init: Initializer for gamma. It defaults to zero (which scales the
        input by 1.0) due to the reparameterization.
    """
    decay: float = 0.999
    use_moving_avg_in_training: bool = False
    set_padded_output_to_zero: bool = True
    force_eval_mode: bool = False
    gamma_init: WeightInit = WeightInit.Constant(0.0)

  def _get_weight_shape(self) -> JTensor:
    return [self.hparams.dim]

  def setup(self) -> None:
    """Creates batch normalization layer variables."""
    p = self.hparams
    self._epsilon = 0.001
    self._decay = p.decay

    beta_pc = WeightHParams(
        shape=self._get_weight_shape(), init=WeightInit.Constant(0.0))
    self.create_variable('beta', beta_pc)

    # gamma = self.theta.gamma + 1.0
    gamma_pc = WeightHParams(shape=self._get_weight_shape(), init=p.gamma_init)
    self.create_variable('gamma', gamma_pc)

    mva = WeightHParams(
        shape=[p.dim],
        init=WeightInit.Constant(0.0),
        collections=[base_layer.WeightHParamsCollection.REQUIRES_MEAN_SYNC])
    self.create_variable('moving_mean', mva, trainable=False)

    mvv = WeightHParams(
        shape=[p.dim],
        init=WeightInit.Constant(1.0),
        collections=[base_layer.WeightHParamsCollection.REQUIRES_MEAN_SYNC])
    self.create_variable('moving_variance', mvv, trainable=False)

  def _get_default_paddings(self, inputs: JTensor) -> JTensor:
    """Gets the default paddings for an input."""
    in_shape = list(inputs.shape)
    asserts.gt(len(in_shape), 1)
    in_shape[-1] = 1
    return jnp.zeros(in_shape, dtype=inputs.dtype)

  def _get_beta_gamma(self) -> Tuple[JTensor, JTensor]:
    p = self.hparams
    if p.use_moving_avg_in_training:
      beta = 0.0
      gamma = 1.0
    else:
      beta = self.theta.beta
      gamma = self.theta.gamma + 1.0
    return beta, gamma

  def compute_and_update_moments(
      self, inputs: JTensor,
      paddings: JTensor) -> Tuple[JTensor, JTensor, JTensor, JTensor]:
    """Computes moments and updates state.

    Args:
      inputs: The inputs JTensor. Shaped [..., dim].
      paddings: The paddings JTensor. Shaped [..., 1], with the same rank as the
        input JTensor.

    Returns:
      Tuple of (mean, variance, beta, gamma).
    """
    p = self.hparams
    if self.do_eval or p.force_eval_mode:
      # The mean and variance used for normalization.
      norm_mean = self.get_var('moving_mean')
      norm_variance = self.get_var('moving_variance')
      self.add_summary('moving_mean', norm_mean, verbosity=4)
      self.add_summary('moving_variance', norm_variance, verbosity=4)
    else:
      rank = inputs.ndim
      reduce_over_dims = list(range(0, rank - 1))
      mean, variance = compute_moments(
          inputs,
          paddings,
          reduce_over_dims,
          keepdims=False,  # Reduce to [p.dim] the same as moving mean/var.
      )

      new_moving_mean = (
          self.get_var('moving_mean') * p.decay + mean * (1.0 - p.decay))
      self.update_var('moving_mean', new_moving_mean)
      new_moving_variance = (
          self.get_var('moving_variance') * p.decay + variance *
          (1.0 - p.decay))
      self.update_var('moving_variance', new_moving_variance)

      # Add some summaries for visualization.
      self.add_summary('mean', mean, verbosity=4)
      self.add_summary('variance', variance, verbosity=4)
      self.add_summary('moving_mean', self.get_var('moving_mean'), verbosity=4)
      self.add_summary(
          'moving_variance', self.get_var('moving_variance'), verbosity=4)
      if p.use_moving_avg_in_training:
        # Use the global statistics for normalization.
        norm_mean = self.get_var('moving_mean')
        norm_variance = self.get_var('moving_variance')
      else:
        # Use the batch statistics for normalization.
        norm_mean = mean
        norm_variance = variance

    beta, gamma = self._get_beta_gamma()
    return norm_mean, norm_variance, beta, gamma

  def __call__(self,
               inputs: JTensor,
               paddings: Optional[JTensor] = None) -> JTensor:
    """Applies batch normalization.

    Args:
      inputs: The inputs JTensor. Shaped [..., dim].
      paddings: The paddings JTensor. Shaped [...].

    Returns:
      Output after applying batch normalization, with the same shape as
      'inputs'.
    """
    p = self.hparams
    inputs, paddings = self._cast_to_fprop_dtype((inputs, paddings))
    if paddings is None:
      paddings = self._get_default_paddings(inputs)
    else:
      # Make padding of same rank as `inputs`.
      paddings = jnp.expand_dims(paddings, -1)

    asserts.eq(inputs.ndim, paddings.ndim)
    asserts.eq(paddings.shape[-1], 1)

    norm_mean, norm_variance, beta, gamma = self.compute_and_update_moments(
        inputs, paddings)

    inv = gamma / jnp.sqrt(norm_variance + self._epsilon)
    bn_output = (inputs - norm_mean) * inv + beta

    if p.set_padded_output_to_zero:
      bn_output *= 1.0 - paddings

    return bn_output


class LayerNorm(BaseNormalization):
  """Layer normalization."""

  class HParams(BaseNormalization.HParams):
    """Associated hyper-params for this layer class.

    Attributes:
      epsilon: Tiny value to guard rsqrt.
      scale: Whether to use a learned scaling.
      bias: Whether to use bias.
    """
    epsilon: float = 1e-6
    use_scale: bool = True
    use_bias: bool = True

  def setup(self) -> None:
    """Creates layer normalization variables."""
    p = self.hparams
    wp = p.weight_split_dims_mapping
    wp_scale = wp.wt
    if p.mesh_shape is not None and wp.wt is None:
      # Simply replicate the weights.
      wp_scale = [-1]
    if p.use_scale:
      self.create_variable(
          'scale',
          WeightHParams(
              shape=[p.dim],
              init=WeightInit.Constant(0.0),
              mesh_shape=p.mesh_shape,
              tensor_split_dims_mapping=wp_scale,
              collections=[
                  base_layer.WeightHParamsCollection.SKIP_LP_REGULARIZATION
              ]))
    if p.use_bias:
      wp_bias = wp_scale  # bias should use the same sharding as scale.
      self.create_variable(
          'bias',
          WeightHParams(
              shape=[p.dim],
              init=WeightInit.Constant(0.0),
              mesh_shape=p.mesh_shape,
              tensor_split_dims_mapping=wp_bias,
              collections=[
                  base_layer.WeightHParamsCollection.SKIP_LP_REGULARIZATION
              ]))

  def __call__(self,
               inputs: JTensor,
               paddings: Optional[JTensor] = None) -> JTensor:
    """Applies layer norm to inputs.

    Args:
      inputs: The inputs JTensor. Shaped [..., dim].
      paddings: unused.

    Returns:
      Output after applying layer normalization, with the same shape as
      'inputs'.
    """
    del paddings  # Unused.
    p = self.hparams
    mean = jnp.mean(inputs, axis=[-1], keepdims=True)
    var = jnp.mean(jnp.square(inputs - mean), axis=[-1], keepdims=True)
    normed_inputs = (inputs - mean) * jax.lax.rsqrt(var + self.hparams.epsilon)
    if p.use_scale:
      normed_inputs *= (1 + self.theta.scale)
    if p.use_bias:
      normed_inputs += self.theta.bias
    return normed_inputs


class RmsNorm(BaseNormalization):
  """RMS normalization: https://arxiv.org/abs/1910.07467."""

  class HParams(BaseNormalization.HParams):
    """Associated hyper-params for this layer class.

    Attributes:
      epsilon: Tiny value to guard rsqrt.
      direct_scale: Whether to apply scale directly without a +1.0. Var is
        initialized to 1.0 instead when true. This makes the weight compatible
        with the implementation in gshard/glam.
    """
    epsilon: float = 1e-6
    direct_scale: bool = True

  def setup(self) -> None:
    """Creates RMS normalization variables."""
    p = self.hparams
    wp = p.weight_split_dims_mapping
    wp_scale = wp.wt
    if p.mesh_shape is not None and wp.wt is None:
      # Simply replicate the weights.
      wp_scale = [-1]
    # Scale variable that scales the RMS norm output by (1 + scale).
    init_value = 1.0 if p.direct_scale else 0.0
    self.create_variable(
        'scale',
        WeightHParams(
            shape=[p.dim],
            init=WeightInit.Constant(init_value),
            mesh_shape=p.mesh_shape,
            tensor_split_dims_mapping=wp_scale))

  def __call__(self,
               inputs: JTensor,
               paddings: Optional[JTensor] = None) -> JTensor:
    """Applies RMS norm to inputs.

    Args:
      inputs: The inputs JTensor. Shaped [..., dim].
      paddings: unused.

    Returns:
      Output after applying RMS normalization, with the same shape as 'inputs'.
    """
    del paddings  # Unused.
    var = jnp.mean(jnp.square(inputs), axis=[-1], keepdims=True)
    normed_inputs = inputs * jax.lax.rsqrt(var + self.hparams.epsilon)
    scale = (
        self.theta.scale if self.hparams.direct_scale else 1 + self.theta.scale)
    normed_inputs *= scale
    return normed_inputs


class RmsNormNoScale(BaseNormalization):
  """RMS normalization: https://arxiv.org/abs/1910.07467 without scale."""

  class HParams(BaseNormalization.HParams):
    """Associated hyper-params for this layer class.

    Attributes:
      epsilon: Tiny value to guard rsqrt.
    """
    epsilon: float = 1e-6

  def __call__(self,
               inputs: JTensor,
               paddings: Optional[JTensor] = None) -> JTensor:
    """Applies RMS norm to inputs.

    Args:
      inputs: The inputs JTensor. Shaped [..., dim].
      paddings: unused.

    Returns:
      Output after applying RMS normalization without scaling w/trainable
      weight. With the same shape as 'inputs'.
    """
    del paddings  # Unused.
    var = jnp.mean(
        jnp.square(inputs), axis=[-1], keepdims=True, dtype=jnp.float32)
    normed_inputs = (inputs * jax.lax.rsqrt(var + self.hparams.epsilon)).astype(
        inputs.dtype)
    return normed_inputs


class GroupNorm(BaseNormalization):
  """Group normalization layer (https://arxiv.org/abs/1803.08494)."""

  class HParams(BaseNormalization.HParams):
    """Associated hyper-params for this layer class.

    Attributes:
      num_groups: Number of groups for GroupNorm.
      min_group_size: Minimum group size for GroupNorm.
      cumulative: If true, only normalize by current and previous time steps.
      input_rank: Rank of input. Only 3(BTD) and 4(NHWC) are supported.
      epsilon: Epsilon added when computing the rsqrt.
      set_padded_output_to_zero: bool. whether to pad padding part to zero.
    """
    num_groups: int = 32
    min_group_size: int = 1
    cumulative: bool = False
    input_rank: Optional[int] = None
    epsilon: float = 0.001
    set_padded_output_to_zero: bool = True

  def setup(self) -> None:
    """Initializes GroupNorm layer and checks parameters."""
    p = self.hparams
    asserts.not_none(p.name)
    asserts.gt(p.num_groups, 0)
    asserts.gt(p.min_group_size, 0)
    asserts.le(p.min_group_size, p.dim)
    asserts.eq(p.dim % p.min_group_size, 0)

    if p.dim >= p.num_groups:
      asserts.eq(
          p.dim % p.num_groups,
          0,
          msg='p.dim({0}) is not dividable by p.num_groups({1})'.format(
              p.dim, p.num_groups))

    asserts.in_set(p.input_rank, (3, 4))

    shape = [1, 1, 1, p.dim] if p.input_rank == 4 else [1, 1, p.dim]
    pc = WeightHParams(
        shape,
        init=WeightInit.Constant(0.0),
        collections=[base_layer.WeightHParamsCollection.SKIP_LP_REGULARIZATION])

    self.create_variable('beta', pc)
    self.create_variable('gamma', pc)

  @property
  def _group_size(self) -> int:
    p = self.hparams
    return max(p.dim // p.num_groups, p.min_group_size)

  @property
  def _num_groups(self) -> int:
    p = self.hparams
    return p.dim // self._group_size

  def _normalize(self, grouped_inputs: JTensor, group_mean: JTensor,
                 group_variance: JTensor) -> JTensor:
    p = self.hparams
    moment_shape = list(grouped_inputs.shape)
    if p.input_rank == 4:
      moment_shape[2] = 1
    moment_shape[-1] = 1

    if not p.cumulative:
      # If not cumulative, the seqlen dimension is also reduced.
      moment_shape[1] = 1

    group_stddev_inv = jax.lax.rsqrt(group_variance + p.epsilon)

    grouped_inputs = (grouped_inputs - group_mean) * group_stddev_inv
    # Merges the last two dims.
    grouped_inputs = jnp.reshape(grouped_inputs,
                                 list(grouped_inputs.shape[:-2]) + [-1])

    # Note, The real gamma to use is 1 + gamma.
    outputs = grouped_inputs * (1 + self.theta.gamma) + self.theta.beta
    return outputs

  def __call__(self,
               inputs: JTensor,
               paddings: Optional[JTensor] = None) -> JTensor:
    """Applies group normalization.

    Args:
      inputs: The inputs JTensor. Shaped [batch_size, height, width, channel] if
        p.rank == 4, else [batch, height, channel].
      paddings: The paddings JTensor. Shaped [batch_size, height]. Intended to
        be used for sequence processing where `height` is `time`.

    Returns:
      Output after applying group normalization, with the same shape as
      'inputs'.
    """
    p = self.hparams
    inputs, paddings = self._cast_to_fprop_dtype((inputs, paddings))
    asserts.eq(inputs.ndim, p.input_rank)

    x = jnp.reshape(
        inputs,
        list(inputs.shape[:-1]) + [self._num_groups, self._group_size])
    expanded_rank = p.input_rank + 1
    all_dims = list(range(expanded_rank))
    if paddings is None or not p.cumulative:
      # Skips batch and num_groups.
      reduce_over_dims = all_dims[1:-2] + all_dims[-1:]
    else:
      # Skips batch, seqlen and num_groups.
      reduce_over_dims = all_dims[2:-2] + all_dims[-1:]

    if paddings is None and not p.cumulative:
      group_mean = jnp.mean(x, axis=reduce_over_dims, keepdims=True)
      group_variance = jnp.mean(
          jnp.square(x - jax.lax.stop_gradient(group_mean)),
          axis=reduce_over_dims,
          keepdims=True)
    else:
      expanded_paddings = jnp.reshape(
          paddings,
          list(inputs.shape[:2]) + [1] * (expanded_rank - 2))
      group_mean, group_variance = compute_moments(
          x,
          expanded_paddings,
          reduce_over_dims,
          cumulative_axis=1,
          keepdims=True)

    gn_output = self._normalize(x, group_mean, group_variance)
    if p.set_padded_output_to_zero and paddings is not None:
      expanded_paddings = jnp.reshape(
          paddings,
          list(inputs.shape[:2]) + [1] * (expanded_rank - 3))
      gn_output *= 1.0 - expanded_paddings
    return gn_output
