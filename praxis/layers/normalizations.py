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

"""Normalization layers."""

import dataclasses

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


def compute_moments(
    inputs: JTensor,
    padding: JTensor,
    reduce_over_dims: list[int],
    cumulative_axis: int | None = None,
    keepdims: bool = False,
) -> tuple[JTensor, JTensor]:
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
  """Base class for normalization layers.

  Attributes:
    dim: Depth of the input and output.
  """
  dim: int = 0

  def __call__(
      self, inputs: JTensor, paddings: JTensor | None = None
  ) -> JTensor:
    """Applies the normalization."""
    raise NotImplementedError(
        'Normalization layers are expected to implement fprop().')


class IdentityNorm(BaseNormalization):
  """Return the input as-is with BaseNormalization-compatible HParams."""

  def __call__(
      self, inputs: JTensor, paddings: JTensor | None = None
  ) -> JTensor:
    """Returns inputs.

    Args:
      inputs: The inputs JTensor. Shaped [..., dim].
      paddings: unused.

    Returns:
      Output as inputs.
    """
    del paddings  # Unused.
    return inputs


class BatchNorm(BaseNormalization):
  """Batch normalization layer.

  Note that statistics are aggregated across
  all workers (a.k.a. Sync BatchNorm) in both pmap and pjit mode.

  Note that gamma in this layer is reparameterized: the gamma variable is
  0-initialized and input is scaled by (1 + gamma). This is different from
  Flax, Tensorflow, PyTorch etc., where gamma is by default 1-initialized
  and used to scale the input. The difference is that in our version,
  weight decay encourages gamma to move towards 1, instead of 0.

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
  gamma_init: WeightInit = dataclasses.field(
      default_factory=lambda: WeightInit.Constant(0.0)
  )

  def _get_weight_shape(self) -> list[int]:
    return [self.dim]

  def setup(self) -> None:
    """Creates batch normalization layer variables."""
    self._epsilon = 0.001
    self._decay = self.decay

    beta_pc = WeightHParams(
        shape=self._get_weight_shape(), init=WeightInit.Constant(0.0))
    self.create_variable('beta', beta_pc)

    # gamma = self.theta.gamma + 1.0
    gamma_pc = WeightHParams(
        shape=self._get_weight_shape(), init=self.gamma_init
    )
    self.create_variable('gamma', gamma_pc)

    mva = WeightHParams(
        shape=[self.dim],
        init=WeightInit.Constant(0.0),
        collections=[base_layer.WeightHParamsCollection.REQUIRES_MEAN_SYNC],
    )
    self.create_variable('moving_mean', mva, trainable=False)

    mvv = WeightHParams(
        shape=[self.dim],
        init=WeightInit.Constant(1.0),
        collections=[base_layer.WeightHParamsCollection.REQUIRES_MEAN_SYNC],
    )
    self.create_variable('moving_variance', mvv, trainable=False)

  def _get_default_paddings(self, inputs: JTensor) -> JTensor:
    """Gets the default paddings for an input."""
    in_shape = list(inputs.shape)
    asserts.gt(len(in_shape), 1)
    in_shape[-1] = 1
    return jnp.zeros(in_shape, dtype=inputs.dtype)

  def _get_beta_gamma(self) -> tuple[JTensor, JTensor]:
    if self.use_moving_avg_in_training:
      beta = 0.0
      gamma = 1.0
    else:
      beta = self.theta.beta
      gamma = self.theta.gamma + 1.0
    return beta, gamma  # pytype: disable=bad-return-type  # jax-ndarray

  def compute_and_update_moments(
      self, inputs: JTensor, paddings: JTensor
  ) -> tuple[JTensor, JTensor, JTensor, JTensor]:
    """Computes moments and updates state.

    Args:
      inputs: The inputs JTensor. Shaped [..., dim].
      paddings: The paddings JTensor. Shaped [..., 1], with the same rank as the
        input JTensor.

    Returns:
      Tuple of (mean, variance, beta, gamma).
    """
    if self.do_eval or self.force_eval_mode:
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

      new_moving_mean = self.get_var('moving_mean') * self.decay + mean * (
          1.0 - self.decay
      )
      self.update_var('moving_mean', new_moving_mean)
      new_moving_variance = self.get_var(
          'moving_variance'
      ) * self.decay + variance * (1.0 - self.decay)
      self.update_var('moving_variance', new_moving_variance)

      # Add some summaries for visualization.
      self.add_summary('mean', mean, verbosity=4)
      self.add_summary('variance', variance, verbosity=4)
      self.add_summary('moving_mean', self.get_var('moving_mean'), verbosity=4)
      self.add_summary(
          'moving_variance', self.get_var('moving_variance'), verbosity=4)
      if self.use_moving_avg_in_training:
        # Use the global statistics for normalization.
        norm_mean = self.get_var('moving_mean')
        norm_variance = self.get_var('moving_variance')
      else:
        # Use the batch statistics for normalization.
        norm_mean = mean
        norm_variance = variance

    beta, gamma = self._get_beta_gamma()
    return norm_mean, norm_variance, beta, gamma

  def __call__(
      self, inputs: JTensor, paddings: JTensor | None = None
  ) -> JTensor:
    """Applies batch normalization.

    Args:
      inputs: The inputs JTensor. Shaped [..., dim].
      paddings: The paddings JTensor. Shaped [...].

    Returns:
      Output after applying batch normalization, with the same shape as
      'inputs'.
    """
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

    if self.set_padded_output_to_zero:
      bn_output *= 1.0 - paddings

    return bn_output


class LayerNorm(BaseNormalization):
  """Layer normalization.

  Attributes:
    direct_scale: Whether to apply scale directly without a +1.0. Var is
      initialized to 1.0 instead when true.
    epsilon: Tiny value to guard rsqrt.
    use_scale: Whether to use a learned scaling.
    use_bias: Whether to use bias.
    reductions_in_fp32: Whether to compute mean and variance in fp32.
      Recommended for stable training on GPUs.
  """
  direct_scale: bool = False
  epsilon: float = 1e-6
  use_scale: bool = True
  use_bias: bool = True
  reductions_in_fp32: bool = False

  def setup(self) -> None:
    """Creates layer normalization variables."""
    wp = self.weight_split_dims_mapping
    wp_scale = wp.wt
    if self.mesh_shape is not None and wp.wt is None:
      # Simply replicate the weights.
      wp_scale = [-1]
    if self.use_scale:
      init_value = 1.0 if self.direct_scale else 0.0
      self.create_variable(
          'scale',
          WeightHParams(
              shape=[self.dim],
              init=WeightInit.Constant(init_value),
              mesh_shape=self.mesh_shape,
              tensor_split_dims_mapping=wp_scale,
              collections=[
                  base_layer.WeightHParamsCollection.SKIP_LP_REGULARIZATION
              ],
          ),
      )
    if self.use_bias:
      wp_bias = wp_scale  # bias should use the same sharding as scale.
      self.create_variable(
          'bias',
          WeightHParams(
              shape=[self.dim],
              init=WeightInit.Constant(0.0),
              mesh_shape=self.mesh_shape,
              tensor_split_dims_mapping=wp_bias,
              collections=[
                  base_layer.WeightHParamsCollection.SKIP_LP_REGULARIZATION
              ],
          ),
      )

  def __call__(
      self, inputs: JTensor, paddings: JTensor | None = None
  ) -> JTensor:
    """Applies layer norm to inputs.

    Args:
      inputs: The inputs JTensor. Shaped [..., dim].
      paddings: unused.

    Returns:
      Output after applying layer normalization, with the same shape as
      'inputs'.
    """
    del paddings  # Unused.
    if self.reductions_in_fp32:
      inputs_dtype = inputs.dtype
      inputs = inputs.astype(jnp.float32)
    mean = jnp.mean(inputs, axis=[-1], keepdims=True)
    var = jnp.mean(jnp.square(inputs - mean), axis=[-1], keepdims=True)
    normed_inputs = (inputs - mean) * jax.lax.rsqrt(var + self.epsilon)
    if self.reductions_in_fp32:
      normed_inputs = normed_inputs.astype(inputs_dtype)
    if self.use_scale:
      scale = self.theta.scale if self.direct_scale else (1 + self.theta.scale)
      normed_inputs *= scale
    if self.use_bias:
      normed_inputs += self.theta.bias
    return normed_inputs


class RmsNorm(BaseNormalization):
  """RMS normalization: https://arxiv.org/abs/1910.07467.

  Attributes:
    epsilon: Tiny value to guard rsqrt.
    direct_scale: Whether to apply scale directly without a +1.0. Var is
      initialized to 1.0 instead when true. This makes the weight compatible
      with the implementation in gshard/glam.
    intermediate_dtype: If not None, use this datatype in the intermediate
      computations.
  """
  epsilon: float = 1e-6
  direct_scale: bool = True
  intermediate_dtype: jnp.dtype | None = None

  def setup(self) -> None:
    """Creates RMS normalization variables."""
    wp = self.weight_split_dims_mapping
    wp_scale = wp.wt
    if self.mesh_shape is not None and wp.wt is None:
      # Simply replicate the weights.
      wp_scale = [-1]
    # Scale variable that scales the RMS norm output by (1 + scale).
    init_value = 1.0 if self.direct_scale else 0.0
    self.create_variable(
        'scale',
        WeightHParams(
            shape=[self.dim],
            init=WeightInit.Constant(init_value),
            mesh_shape=self.mesh_shape,
            tensor_split_dims_mapping=wp_scale,
        ),
    )

  def __call__(
      self, inputs: JTensor, paddings: JTensor | None = None
  ) -> JTensor:
    """Applies RMS norm to inputs.

    Args:
      inputs: The inputs JTensor. Shaped [..., dim].
      paddings: unused.

    Returns:
      Output after applying RMS normalization, with the same shape as 'inputs'.
    """
    del paddings  # Unused.
    if self.intermediate_dtype is not None:
      inputs = jnp.asarray(inputs, dtype=self.intermediate_dtype)
    var = jnp.mean(jnp.square(inputs), axis=[-1], keepdims=True)
    normed_inputs = jnp.asarray(
        inputs * jax.lax.rsqrt(var + self.epsilon), self.fprop_dtype
    )
    scale = self.theta.scale if self.direct_scale else 1 + self.theta.scale
    normed_inputs *= scale
    return normed_inputs


class RmsNormNoScale(BaseNormalization):
  """RMS normalization: https://arxiv.org/abs/1910.07467 without scale.

  Attributes:
    epsilon: Tiny value to guard rsqrt.
  """
  epsilon: float = 1e-6

  def __call__(
      self, inputs: JTensor, paddings: JTensor | None = None
  ) -> JTensor:
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
    normed_inputs = (inputs * jax.lax.rsqrt(var + self.epsilon)).astype(
        inputs.dtype
    )
    return normed_inputs


class GroupNorm(BaseNormalization):
  """Group normalization layer (https://arxiv.org/abs/1803.08494).

  Attributes:
    num_groups: Number of groups for GroupNorm.
    min_group_size: Minimum group size for GroupNorm.
    cumulative: If true, only normalize by current and previous time steps.
    input_rank: Rank of input. Only 3(BTD), 4(NHWC) and 5 (NTHWC) are supported.
    epsilon: Epsilon added when computing the rsqrt.
    set_padded_output_to_zero: bool. whether to pad padding part to zero.
    use_scale: Whether to use a learned scaling.
    use_bias: Whether to use bias.
  """
  num_groups: int = 32
  min_group_size: int = 1
  cumulative: bool = False
  input_rank: int | None = None
  epsilon: float = 0.001
  set_padded_output_to_zero: bool = True
  use_scale: bool = True
  use_bias: bool = True

  def setup(self) -> None:
    """Initializes GroupNorm layer and checks parameters."""
    asserts.not_none(self.name)
    asserts.gt(self.num_groups, 0)
    asserts.gt(self.min_group_size, 0)
    asserts.le(self.min_group_size, self.dim)
    asserts.eq(self.dim % self.min_group_size, 0)

    if self.dim >= self.num_groups:
      asserts.eq(
          self.dim % self.num_groups,
          0,
          msg='p.dim({0}) is not dividable by p.num_groups({1})'.format(
              self.dim, self.num_groups
          ),
      )

    asserts.in_set(self.input_rank, (3, 4, 5))

    shape = [1] * (self.input_rank - 1) + [self.dim]
    pc = WeightHParams(
        shape,
        init=WeightInit.Constant(0.0),
        collections=[base_layer.WeightHParamsCollection.SKIP_LP_REGULARIZATION])

    if self.use_bias:
      self.create_variable('beta', pc)
    if self.use_scale:
      self.create_variable('gamma', pc)

  @property
  def _group_size(self) -> int:
    return max(self.dim // self.num_groups, self.min_group_size)

  @property
  def _num_groups(self) -> int:
    return self.dim // self._group_size

  def _normalize(self, grouped_inputs: JTensor, group_mean: JTensor,
                 group_variance: JTensor) -> JTensor:
    group_stddev_inv = jax.lax.rsqrt(group_variance + self.epsilon)

    grouped_inputs = (grouped_inputs - group_mean) * group_stddev_inv
    # Merges the last two dims.
    grouped_inputs = jnp.reshape(grouped_inputs,
                                 list(grouped_inputs.shape[:-2]) + [-1])

    outputs = grouped_inputs
    if self.use_scale:
      # Note, The real gamma to use is 1 + gamma.
      outputs *= (1 + self.theta.gamma)
    if self.use_bias:
      outputs += self.theta.beta
    return outputs

  def __call__(
      self, inputs: JTensor, paddings: JTensor | None = None
  ) -> JTensor:
    """Applies group normalization.

    Args:
      inputs: The inputs JTensor. Shaped [batch_size, height, width, channel] if
        p.rank == 4, [batch_size, time, height, width, channel] if p.rank == 5,
        else [batch, height, channel].
      paddings: The paddings JTensor. Shaped [batch_size, height]. Intended to
        be used for sequence processing where `height` is `time`.

    Returns:
      Output after applying group normalization, with the same shape as
      'inputs'.
    """
    inputs, paddings = self._cast_to_fprop_dtype((inputs, paddings))
    asserts.eq(inputs.ndim, self.input_rank)

    x = jnp.reshape(
        inputs,
        list(inputs.shape[:-1]) + [self._num_groups, self._group_size])
    expanded_rank = self.input_rank + 1
    all_dims = list(range(expanded_rank))
    if paddings is None or not self.cumulative:
      # Skips batch and num_groups.
      reduce_over_dims = all_dims[1:-2] + all_dims[-1:]
    else:
      # Skips batch, seqlen and num_groups.
      reduce_over_dims = all_dims[2:-2] + all_dims[-1:]

    if paddings is None and not self.cumulative:
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
          cumulative_axis=1 if self.cumulative else None,
          keepdims=True,
      )

    gn_output = self._normalize(x, group_mean, group_variance)
    if self.set_padded_output_to_zero and paddings is not None:
      expanded_paddings = jnp.reshape(
          paddings,
          list(inputs.shape[:2]) + [1] * (expanded_rank - 3))
      gn_output *= 1.0 - expanded_paddings
    return gn_output


class WeightNormL2(BaseNormalization):
  """Weight norm on the last axis by L2 norm https://arxiv.org/abs/1602.07868"""

  def setup(self):
    """Creates weight normalization variables."""
    self.create_variable(
        'g',
        WeightHParams(
            shape=[self.dim],
            init=WeightInit.Constant(0.0),
            dtype=self.dtype,
        ),
    )

  def __call__(
      self, inputs: JTensor, paddings: JTensor | None = None
  ) -> JTensor:
    """Applies the L2 weight normalization."""
    del paddings  # unused.

    axis = list(range(inputs.ndim - 1))
    scale = jnp.expand_dims(self.theta.g + 1.0, axis)
    return scale * py_utils.l2_normalize(inputs, axis)


class SpectralNorm(BaseNormalization):
  """Spectral normalization on the last weight dim.

  Normalizes W / σ(W), where σ(W) = max_{h≠0} |W h| / |h|.
  https://arxiv.org/abs/1802.05957.
  """

  n_power_iteration: int = 1

  def setup(self):
    """Creates weight normalization variables."""
    self.create_variable(
        'u',
        base_layer.WeightHParams(
            shape=[self.dim],
            init=base_layer.WeightInit.Gaussian(),
            dtype=self.dtype,
            collections=[base_layer.WeightHParamsCollection.REQUIRES_MEAN_SYNC],
        ),
        trainable=False,
    )

  def __call__(
      self, inputs: pytypes.JTensor, paddings: pytypes.JTensor | None = None
  ) -> pytypes.JTensor:
    del paddings  # Unused.
    w = jnp.reshape(inputs, [-1, self.dim])
    u = self.get_var('u')
    for _ in range(self.n_power_iteration):
      v = py_utils.l2_normalize(w @ u)
      u = py_utils.l2_normalize(v @ w)
    v = jax.lax.stop_gradient(v)
    u = jax.lax.stop_gradient(u)

    if not self.do_eval:
      self.update_var('u', u)

    norm = v @ w @ u
    wn = w / norm
    wn = jnp.reshape(wn, inputs.shape)
    return self._cast_to_fprop_dtype(wn)
