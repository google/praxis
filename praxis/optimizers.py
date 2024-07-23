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

"""Module for all optimizers."""

from __future__ import annotations

import abc
import copy
import dataclasses
import functools
import re
from typing import Any, Callable, Literal, NamedTuple, Protocol, Sequence, runtime_checkable

from absl import logging
import jax
from jax import lax
from jax import numpy as jnp
import optax
from praxis import asserts
from praxis import base_hyperparams
from praxis import base_layer
from praxis import gshard_utils
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis import schedules

from optax_shampoo import distributed_shampoo

# DistributedShampoo types
distributed_shampoo_optimizer = distributed_shampoo.distributed_shampoo
Preconditioner = distributed_shampoo.Preconditioner
QuantizedValue = distributed_shampoo.QuantizedValue
GraftingType = distributed_shampoo.GraftingType
ShardedShampooStats = distributed_shampoo.ShardedShampooStats
ShampooState = distributed_shampoo.ShampooState
TrainingMetrics = distributed_shampoo.TrainingMetrics
LocalShardedParameterStats = distributed_shampoo.LocalShardedParameterStats
GlobalShardedParameterStats = distributed_shampoo.GlobalShardedParameterStats

NestedMap = py_utils.NestedMap
WeightHParams = base_layer.WeightHParams
NestedWeightHParams = base_layer.NestedWeightHParams
JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor
NestedHParams = pytypes.NestedHParams

# Initializes sharding spec for the optimizer state variables.
TransformInitPartitionSpecFn = Callable[
    [NestedHParams], NestedHParams | Sequence[NestedHParams]
]

instantiate = base_hyperparams.instantiate

_WEIGHT_DECAY_DEPRECATION_TEMPLATE = (
    'DEPRECATION WARNING: p.{0} will be deprecated. In future, '
    'we will do a migration to remove p.{0} and after that, setting it will '
    'throw an exception. In future, we will use p.l2_regularizer_weight for '
    'coupled weight decay (i.e., weight decays that affect optimizer slots), '
    'and use p.decoupled_weight_decay for decoupled weight decay (i.e., '
    'weight decays that are added only to the final update).')

_WEIGHT_DECAY_DEPRECATION = _WEIGHT_DECAY_DEPRECATION_TEMPLATE.format(
    'weight_decay')

_WEIGHT_DECAY_RATE_DEPRECATION = _WEIGHT_DECAY_DEPRECATION_TEMPLATE.format(
    'weight_decay_rate')


# Extension of optax.GradientTransformation that supports spmd sharding and
# explicit annotation of sharding specs for the optimizer state variables.
@dataclasses.dataclass(frozen=True)
class ShardedGradientTransformation:
  """GradientTransformation that supports spmd."""
  # init_fn and update_fn are the same as in optax.GradientTransformation
  init: optax.TransformInitFn
  update: optax.TransformUpdateFn
  # Input is the sharding specs of the variables used in the forward
  # computation.  Output is the sharding specs of the optimizer state variables.
  #
  # Constraints: output from this function should be of identical structure as
  # that of the init() function.
  init_partition_spec: TransformInitPartitionSpecFn


GeneralGradientTransformation = (
    optax.GradientTransformation | ShardedGradientTransformation
)


def count_init_fn(_):
  """Common init_fn that initializes a count for global step."""
  return NestedMap(count=jnp.zeros([], jnp.int32))


def count_init_partition_spec_fn(
    var_hparams: NestedWeightHParams) -> NestedWeightHParams:
  """Init partition spec for only partitioning the count/step."""
  del var_hparams
  return NestedMap(
      count=WeightHParams(
          shape=[],
          init=None,
          dtype=jnp.int32,
          collections=None,
          tensor_split_dims_mapping=[],
      )
  )


def partition_params(
    grad_tx: optax.GradientTransformation,
    var_weight_hparams: NestedWeightHParams | None,
    opt_states: optax.OptState,
) -> optax.OptState:
  """Applies sharding for optimizer params with tree_map_params."""

  def sharding_function(
      params: Any, spec: base_layer.WeightHParams
  ) -> base_layer.WeightHParams | None:
    if isinstance(params, optax.MaskedNode):
      return None
    return spec

  def sharding_for_non_parameter_fields(
      params: Any,
  ) -> base_layer.WeightHParams:
    # Typically the non-parameter fields are 'count'.
    return base_layer.WeightHParams(
        # Preserve the same shape as the original params shape.
        # Specifically, the non-parameter `count` must have partition specs
        # of the same length as their rank.
        # Note that we set the shape rather than `repeat_prefix` as opposed to
        # regular Pax optimizers, since there is no easy way to retrieve a
        # `repeat_prefix_split_dims_mapping` at that point. This implies that
        # prefix vectorization won't be applied to such non parameter fields.
        # In practice, those are small arrays, so this shouldn't be a problem.
        shape=params.shape,
        init=None,
        dtype=jnp.int32,
        collections=None,
        tensor_split_dims_mapping=[],
    )

  opt_states_pspec = optax.tree_map_params(
      grad_tx,
      sharding_function,
      opt_states,
      var_weight_hparams,
      # Count is the only non-params in the current optimizers.
      transform_non_params=sharding_for_non_parameter_fields,
      is_leaf=lambda v: isinstance(v, optax.MaskedNode),
  )
  return opt_states_pspec


def sharded_sgd(
    learning_rate_fn: optax.Schedule,
    momentum: float | None = None,
    nesterov: bool = False,
) -> ShardedGradientTransformation:
  """A Stochastic Gradient Descent optimiser that supports spmd sharding.

  This implements stochastic gradient descent. It also includes support for
  momentum and Nesterov acceleration, as these are standard practice when
  using stochastic gradient descent to train deep neural networks.

  References:
    Sutskever et al, 2013: http://proceedings.mlr.press/v28/sutskever13.pdf

  Args:
    learning_rate_fn: a callable that given the current training step, returns
      the learning rate to apply.
    momentum: the `decay` rate used by the momentum term; when it is set to
      `None`, then momentum is not used at all.
    nesterov: whether Nesterov momentum is used.

  Returns:
    A `ShardedGradientTransformation`.
  """

  # optax.sgd is a trace transform followed by scale-by-learning-rate.
  sgd = optax.sgd(
      learning_rate=learning_rate_fn, momentum=momentum, nesterov=nesterov)

  def init_fn(mdl_vars: NestedJTensor):
    # If mdl_vars = {'w': jnp.ones((4,))}, the result is a tuple.
    # if momentum is None
    #   (EmptyState(), ScaleByScheduleState(count=DeviceArray(0, dtype=int32)))
    # if momentum is not None
    #   (TraceState(trace={'w': DeviceArray([0., 0., 0., 0.], dtype=float32)}),
    #    ScaleByScheduleState(count=DeviceArray(0, dtype=int32)))
    return sgd.init(mdl_vars)

  def init_partition_spec_fn(mdl_params):
    count = WeightHParams(
        shape=[], init=None, dtype=jnp.int32, collections=None)

    if momentum is None:
      return (optax.EmptyState(), optax.ScaleByScheduleState(count=count))  # pytype: disable=wrong-arg-types  # numpy-scalars

    def _opt_state_sharding_spec(var_hparams: WeightHParams) -> WeightHParams:
      """Returns optimizer sharding spec for one particular variable."""
      m_var_hparams = copy.deepcopy(var_hparams)
      m_var_hparams.init = None
      # Momentum has same sharding as mdl var.
      return m_var_hparams

    momentum_sharding = jax.tree.map(_opt_state_sharding_spec, mdl_params)
    return (optax.TraceState(trace=momentum_sharding),
            optax.ScaleByScheduleState(count=count))  # pytype: disable=wrong-arg-types  # numpy-scalars

  def update_fn(updates, state, params=None):
    del params
    return sgd.update(updates, state)

  return ShardedGradientTransformation(
      init=init_fn,
      update=update_fn,
      init_partition_spec=init_partition_spec_fn)


def sharded_adagrad(learning_rate_fn: optax.Schedule,
                    initial_accumulator_value: float,
                    epsilon: float) -> ShardedGradientTransformation:
  """Adagrad optimizer that supports spmd sharding.

  Args:
    learning_rate_fn: a callable that given the current training step, returns
      the learning rate to apply.
    initial_accumulator_value: (default `0.1`).
    epsilon (default `1e-7`): epsilon.

  Returns:
    A `ShardedGradientTransformation`.
  """

  # optax.adagrad is a trace transform followed by scale-by-learning-rate.
  adagrad = optax.adagrad(
      learning_rate=learning_rate_fn,
      initial_accumulator_value=initial_accumulator_value,
      eps=epsilon)

  def init_fn(mdl_vars: NestedJTensor):
    return adagrad.init(mdl_vars)

  def init_partition_spec_fn(mdl_params):
    count = WeightHParams(
        shape=[], init=None, dtype=jnp.int32, collections=None)

    def _opt_state_sharding_spec(var_hparams: WeightHParams) -> WeightHParams:
      """Returns optimizer sharding spec for one particular variable."""
      s_var_hparams = copy.deepcopy(var_hparams)
      s_var_hparams.init = None
      # ScaleByRssState has same sharding as `mdl_var`.
      return s_var_hparams

    scale_by_rss_sharding = jax.tree.map(_opt_state_sharding_spec, mdl_params)
    return (optax.ScaleByRssState(sum_of_squares=scale_by_rss_sharding),
            optax.ScaleByScheduleState(count=count))  # pytype: disable=wrong-arg-types  # numpy-scalars

  def update_fn(updates, state, params=None):
    del params
    return adagrad.update(updates, state)

  return ShardedGradientTransformation(
      init=init_fn,
      update=update_fn,
      init_partition_spec=init_partition_spec_fn)


class _AdamOptState:
  """Adam optimizer state.

  Attributes:
    m: first moment.
    v: second moment.
  """

  def __init__(self, *, m: WeightHParams | JTensor, v: WeightHParams | JTensor):
    self.m = m
    self.v = v


class _ShardedAdamHelper:
  """A helper class facilitates the creation of sharded_adam_optimizer.

  Attributes:
    _maybe_inf_to_nan: whether to replace infinite values with NaNs.
  """

  _maybe_inf_to_nan: bool

  def __init__(self, maybe_inf_to_nan: bool = True):
    self._maybe_inf_to_nan = maybe_inf_to_nan

  def opt_state_sharding_spec(self,
                              var_hparams: WeightHParams) -> _AdamOptState:
    """Returns optimizer sharding spec for one particular variable."""
    m_var_hparams = copy.deepcopy(var_hparams)
    m_var_hparams.init = None
    v_var_hparams = copy.deepcopy(var_hparams)
    v_var_hparams.init = None
    # m and v simply share the same sharding.
    return _AdamOptState(m=m_var_hparams, v=v_var_hparams)

  def init_opt_state(self, var_hparams: WeightHParams) -> _AdamOptState:
    """Returns optimizer state for one particular variable."""
    return _AdamOptState(
        m=jnp.zeros_like(var_hparams), v=jnp.zeros_like(var_hparams))  # pytype: disable=wrong-arg-types  # jnp-type

  def inf_to_nan(self, array: JTensor):
    """Converting Infinity values to the more sticky NaN."""
    if not self._maybe_inf_to_nan:
      return array
    return jnp.nan_to_num(array, nan=jnp.nan, posinf=jnp.nan, neginf=jnp.nan)

  def bias_corrected_decay(self, step: JTensor, decay: float) -> JTensor:
    """Incorporates bias correction into decay.

    Please see section 7.1 in https://arxiv.org/pdf/1804.04235.pdf for the
    derivation of the formulas below. With bias-corrected decay, we can simply
    do

    m_{t} = decay1 * m_{t-1} + (1 - decay1) * g
    v_{t} = decay2 * v_{t-1} + (1 - decay2) * g ^ 2

    without further bias correction.

    Args:
      step: current step, 0-based.
      decay: the raw decay. As t -> infinity, bias corrected decay converges to
        this value.

    Returns:
      Bias corrected decay.
    """
    t = step.astype(jnp.float32) + 1.
    return decay * (1. - jnp.power(decay, t - 1.)) / (1. - jnp.power(decay, t))

  def update_moments(self, step: JTensor, update: JTensor,
                     moments: _AdamOptState, beta1: float,
                     beta2: float) -> _AdamOptState:
    """Updates momentum values."""
    beta1_decay = self.bias_corrected_decay(step, beta1)
    beta2_decay = self.bias_corrected_decay(step, beta2)
    m = (1.0 - beta1_decay) * update + beta1_decay * moments.m
    v = (1.0 - beta2_decay) * (update**2) + beta2_decay * moments.v
    return _AdamOptState(m=m, v=v)

  def clip_update(self, update: JTensor, clip_threshold: float) -> JTensor:
    mean_update = self.inf_to_nan(reduce_rms(update))
    clip_threshold = jnp.array(clip_threshold, dtype=update.dtype)
    denom = jnp.maximum(1.0, mean_update / clip_threshold)
    return update / denom


class _LionOptState:
  """Lion optimizer state.

  Attributes:
    m: momentum.
  """

  def __init__(self, *, m: WeightHParams | JTensor):
    self.m = m


class _ShardedLionHelper(_ShardedAdamHelper):
  """A helper class facilitates the creation of sharded_lion_optimizer."""

  def opt_state_sharding_spec(self,  # pytype: disable=signature-mismatch  # overriding-return-type-checks
                              var_hparams: WeightHParams) -> _LionOptState:
    """Returns optimizer sharding spec for one particular variable."""
    m_var_hparams = copy.deepcopy(var_hparams)
    m_var_hparams.init = None
    # m simply share the same sharding.
    return _LionOptState(m=m_var_hparams)

  def init_opt_state(self,  # pytype: disable=signature-mismatch  # overriding-return-type-checks
                     var_hparams: WeightHParams,
                     m_dtype: jnp.dtype = jnp.float32) -> _LionOptState:
    """Returns optimizer state for one particular variable."""
    return _LionOptState(m=jnp.zeros_like(var_hparams, dtype=m_dtype))  # pytype: disable=wrong-arg-types  # jnp-type

  def update_moments(self, step: JTensor, update: JTensor,  # pytype: disable=signature-mismatch  # overriding-return-type-checks
                     moments: _LionOptState,
                     beta2: float) -> _LionOptState:
    """Updates momentum value."""
    m = (1. - beta2) * update + beta2 * moments.m
    return _LionOptState(m=m)


def sharded_chain(
    *args: GeneralGradientTransformation) -> ShardedGradientTransformation:
  """Applies a list of (possibly sharded) chainable update transformations.

  Given a sequence of chainable transforms, `sharded_chain` returns an `init_fn`
  that constructs a `state` by concatenating the states of the individual
  transforms, and returns an `update_fn` which chains the update transformations
  feeding the appropriate state to each. In addition, it differs from the optax
  `chain` function, by also supporting ShardedGradientTransformation by chaining
  also the `init_partition_spec_fn`. If there are no
  ShardedGradientTransformations in the chain, the sharding specs will be
  None, meaning all the variables are replicated.

  Args:
    *args: a sequence of chainable GradientTransformations or
      ShardedGradientTransformations or a combination of both.

  Returns:
    A single chained ShardedGradientTransformation.

  Raises:
    ValueError: If the number of updates and states do not match.
    ValueError: If attempting to `sharded_chain` an optimizer that does not have
      an `init_partition_spec` defined.
  """

  def init_fn(params):
    return tuple(fn.init(params) for fn in args)

  def update_fn(updates, state, params=None):
    if len(args) != len(state):
      raise ValueError(
          'The number of updates and states has to be the same in '
          f'sharded chain. got {len(args)=}, {len(state)=}'
      )

    new_state = []
    for s, fn in zip(state, args):
      updates, new_s = fn.update(updates, s, params)
      # Some of the new states may have None instead of optax.MaskedNode.
      new_s = jax.tree.map(
          lambda x: optax.MaskedNode() if x is None else x,
          new_s,
          is_leaf=lambda x: x is None)
      new_state.append(new_s)
    return updates, tuple(new_state)

  def init_partition_spec_fn(mdl_vars):
    partition_specs = []
    for fn in args:
      init_partition_spec = getattr(fn, 'init_partition_spec', None)
      if callable(init_partition_spec):
        nmap = init_partition_spec(mdl_vars)
        partition_specs.append(nmap)
      else:
        # Raise ValueError as we are attempting to sharded_chain an optimizer
        # that does not have an `init_partition_spec` method defined.
        raise ValueError('Attempting to use an optimizer in sharded_chain that '
                         'does not have an init_partition_spec.')
    return optax.MaskedState(inner_state=tuple(partition_specs))

  return ShardedGradientTransformation(
      init=init_fn,
      update=update_fn,
      init_partition_spec=init_partition_spec_fn)


def sharded_masked(
    inner: GeneralGradientTransformation,
    mask: NestedHParams | Callable[[NestedHParams], NestedHParams],
    use_custom_masked: bool = False,
) -> GeneralGradientTransformation:
  """Mask updates so only some are transformed, the rest are passed through.

  This differs from the Optax version in that it supports sharding annotations.

  Args:
    inner: Inner transformation to mask.
    mask: a PyTree with same structure as (or a prefix of) the params PyTree, or
      a Callable that returns such a pytree given the params/updates. The leaves
      should be booleans, ``True`` for leaves/subtrees you want to apply the
      transformation to, and ``False`` for those you want to skip.
    use_custom_masked:  Whether to use custom masked implementation. Since Optax
      masked does not work with tree_map_params. This is unused presently.

  Returns:
    New ShardedGradientTransformation wrapping ``inner``.
  """
  del use_custom_masked
  def init_partition_spec_fn(mdl_vars):
    init_partition_spec = getattr(inner, 'init_partition_spec', None)
    if callable(init_partition_spec):
      return init_partition_spec(mdl_vars)

  grad_tx = optax.masked(inner, mask)
  if not hasattr(inner, 'init_partition_spec'):
    return grad_tx
  else:
    return ShardedGradientTransformation(
        init=grad_tx.init,
        update=grad_tx.update,
        init_partition_spec=init_partition_spec_fn)


def apply_lp_regularizer(
    var_lp_mask: NestedHParams,
    regularizer_weight: float | None = 0.0,
    p: float | None = 2.0,
    skip_lp_1d_vectors: bool | None = False,
    use_optax_gradient_transformations: bool = False,
) -> GeneralGradientTransformation:
  """Applies Lp regularization by adjusting gradients.

  Note, lp regularizers add loss to final loss objective, while decoupled
  weight decay adds decay directly into weights. They are different especially
  when there are moment statistics in optimizers. A good reference can be found
  in: https://www.fast.ai/2018/07/02/adam-weight-decay/#adamw

  Args:
    var_lp_mask: mask to apply lp based on SKIP_LP_REGULARIZATION. If it is 0,
      the lp regularization is not applied.
    regularizer_weight: Weight for L2 regularization.
    p: 1 or 2 as L1/L2 regularization.
    skip_lp_1d_vectors: If True, skip L1/L2 regularization for 1d vector vars.
    use_optax_gradient_transformations: If True, returns an instance of
      optax.GradientTransformationExtraArgs

  Returns:
    A GeneralGradientTransformation applying Lp regularizers.
  """
  asserts.in_set(p, [1.0, 2.0])

  def skip_mask(var):
    if skip_lp_1d_vectors and var.ndim <= 1:
      return 0.0
    else:
      return 1.0

  def update_fn(updates, state, params):
    count = state.count
    if regularizer_weight:
      if params is None:
        raise ValueError('Params must not be empty when applying weight decay.')

      if p == 1.0:
        fn = (
            lambda g, p, m=1.0: g  # pylint: disable=g-long-lambda, g-long-ternary
            + regularizer_weight * jnp.sign(p) * skip_mask(p) * m
            if not py_utils.is_optax_masked_node(g)
            else optax.MaskedNode()
        )
      elif p == 2.0:
        fn = (
            lambda g, p, m=1.0: g + regularizer_weight * p * skip_mask(p) * m  # pylint: disable=g-long-lambda
            if not py_utils.is_optax_masked_node(g)
            else optax.MaskedNode()
        )

      if var_lp_mask is None:
        updates = jax.tree.map(fn, updates, params)
      else:
        updates = jax.tree.map(
            fn,
            updates,
            params,
            var_lp_mask,
            is_leaf=py_utils.is_optax_masked_node,
        )
    updated_state = NestedMap(count=count + 1)
    return updates, updated_state

  if use_optax_gradient_transformations:
    return optax.GradientTransformationExtraArgs(
        init=count_init_fn, update=update_fn
    )

  return ShardedGradientTransformation(
      init=count_init_fn,
      update=update_fn,
      init_partition_spec=count_init_partition_spec_fn)


def apply_decoupled_weight_decay(
    learning_rate_fn: optax.Schedule,
    var_wd_mask: NestedHParams,
    regularizer_weight: float | None = 0.0,
    use_optax_gradient_transformations: bool = False,
) -> GeneralGradientTransformation:
  """Applies decoupled weight decay on weights.

  Note, lp regularizers add loss to final loss objective, while decoupled
  weight decay adds decay directly into weights. They are different especially
  when there are moment statistics in optimizers. A good reference can be found
  in: https://www.fast.ai/2018/07/02/adam-weight-decay/#adamw

  Args:
    learning_rate_fn: An optax schedule that infers the lr given the step.
    var_wd_mask: mask to apply weight decay based on SKIP_LP_REGULARIZATION. If
      it is 0, the weight decay is not applied.
    regularizer_weight: Weight for decoupled weight decay.
    use_optax_gradient_transformations: If True, returns an instance of
      optax.GradientTransformationExtraArgs

  Returns:
    A GeneralGradientTransformation applying weight decay.
  """

  def update_fn(updates, state, params):
    count = state.count
    lr = learning_rate_fn(count)
    if regularizer_weight:
      if params is None:
        raise ValueError('Params must not be empty when applying weight decay.')

      fn = (
          lambda g, p, m=1.0: g - lr * regularizer_weight * p * m  # pylint: disable=g-long-lambda
          if not py_utils.is_optax_masked_node(g)
          else optax.MaskedNode()
      )

      if var_wd_mask is None:
        updates = jax.tree.map(fn, updates, params)
      else:
        updates = jax.tree.map(
            fn,
            updates,
            params,
            var_wd_mask,
            is_leaf=py_utils.is_optax_masked_node,
        )
    updated_state = NestedMap(count=count + 1)
    return updates, updated_state

  if use_optax_gradient_transformations:
    return optax.GradientTransformationExtraArgs(
        init=count_init_fn, update=update_fn
    )
  return ShardedGradientTransformation(
      init=count_init_fn,
      update=update_fn,
      init_partition_spec=count_init_partition_spec_fn)


def sharded_adam(
    learning_rate_fn: optax.Schedule,
    beta1: float,
    beta2: float,
    epsilon: float,
    epsilon_root: float,
    update_capping: float,
    weight_decay: float,
    maybe_inf_to_nan: bool = True) -> ShardedGradientTransformation:
  """Standard Adam optimizer that also supports sharding.

  This Adam optimizer supports optional update capping when update_capping is >
  0. Update capping can help stabilizing model learning, avoiding excessive
  updates when gradient variance estimate is stale (e.g. when data distribution
  suddenly shifts).

  Args:
    learning_rate_fn: a callable that given the current training step, returns
      the learning rate to apply.
    beta1: decay rate to track the first moment.
    beta2: decay rate to track the second moment.
    epsilon: Small constant applied to the denominator outside of the square
      root to avoid dividing by zero when rescaling.
    epsilon_root: Small constant applied to the denominator inside of the square
      root to avoid dividing by zero when rescaling.
    update_capping: If > 0, cap mean update to at most this value.
    weight_decay: If > 0, weight decay to apply.
    maybe_inf_to_nan: Will use jax.nan_to_num during update when True

  Returns:
    A `ShardedGradientTransformation`.
  """
  if weight_decay:
    logging.warning(_WEIGHT_DECAY_DEPRECATION)

  helper = _ShardedAdamHelper(maybe_inf_to_nan=maybe_inf_to_nan)

  def init_fn(mdl_vars):
    slot_vars = jax.tree.map(helper.init_opt_state, mdl_vars)
    count = jnp.array(0, dtype=jnp.int32)
    return NestedMap(
        count=count,
        m=jax.tree.map(lambda x: x.m, slot_vars),
        v=jax.tree.map(lambda x: x.v, slot_vars))

  def init_partition_spec_fn(mdl_params):
    slot_vars = jax.tree.map(helper.opt_state_sharding_spec, mdl_params)
    count = WeightHParams(
        shape=[], init=None, dtype=jnp.int32, collections=None)

    return NestedMap(
        count=count,
        m=jax.tree.map(lambda x: x.m, slot_vars),
        v=jax.tree.map(lambda x: x.v, slot_vars))

  def update_fn(updates, state, params=None):
    # Sanitize updates just in case.
    if weight_decay > 0:
      assert params is not None
    updates = jax.tree.map(helper.inf_to_nan, updates)
    count = state.count

    def _update_momentum(g, m, v):
      return helper.update_moments(count, g, _AdamOptState(m=m, v=v), beta1,
                                   beta2)

    updated_moments = jax.tree.map(_update_momentum, updates, state.m, state.v)

    m = jax.tree.map(lambda x: x.m, updated_moments)
    v = jax.tree.map(lambda x: x.v, updated_moments)

    updates = jax.tree.map(
        lambda m, v: m / (jnp.sqrt(v + epsilon_root) + epsilon), m, v)

    if update_capping > 0:
      updates = jax.tree.map(lambda x: helper.clip_update(x, update_capping),
                             updates)

    if weight_decay > 0:
      updates = jax.tree.map(lambda x, v: x + weight_decay * v, updates, params)

    step_size = -1.0 * learning_rate_fn(count)
    # Finally, fold in step size.
    updates = jax.tree.map(lambda x: step_size * x, updates)

    updated_states = NestedMap(count=count + 1, m=m, v=v)
    return updates, updated_states

  return ShardedGradientTransformation(
      init=init_fn,
      update=update_fn,
      init_partition_spec=init_partition_spec_fn)


def sharded_lion(learning_rate_fn: optax.Schedule, beta1: float,
                 beta2: float, m_dtype: jnp.dtype, update_capping: float,
                 weight_decay: float) -> ShardedGradientTransformation:
  """Standard Lion optimizer that also supports sharding.

  This Lion optimizer supports optional update capping when update_capping
  is > 0. Update capping can help stabilizing model learning, avoiding excessive
  updates when gradient variance estimate is stale (e.g. when data distribution
  suddenly shifts).

  Args:
    learning_rate_fn: a callable that given the current training step, returns
      the learning rate to apply.
    beta1: rate to combine the moment and the current gradient.
    beta2: decay rate to track the moment.
    m_dtype: momentum's dtype.
    update_capping: If > 0, cap mean update to at most this value.
    weight_decay: If > 0, weight decay to apply.

  Returns:
    A `ShardedGradientTransformation`.
  """
  if weight_decay:
    logging.warn(_WEIGHT_DECAY_DEPRECATION)

  helper = _ShardedLionHelper()
  init_opt_state = functools.partial(helper.init_opt_state, m_dtype=m_dtype)

  def init_fn(mdl_vars):
    slot_vars = jax.tree.map(init_opt_state, mdl_vars)
    count = jnp.array(0, dtype=jnp.int32)
    return NestedMap(count=count, m=jax.tree.map(lambda x: x.m, slot_vars))

  def init_partition_spec_fn(mdl_params):
    slot_vars = jax.tree.map(helper.opt_state_sharding_spec, mdl_params)
    count = WeightHParams(
        shape=[], init=None, dtype=jnp.int32, collections=None)

    return NestedMap(count=count, m=jax.tree.map(lambda x: x.m, slot_vars))

  def update_fn(updates, state, params=None):
    # Sanitize updates just in case.
    if weight_decay > 0:
      assert params is not None
    updates = jax.tree.map(helper.inf_to_nan, updates)
    count = state.count

    m_casted = jax.tree.map(lambda u, x: x.astype(u.dtype), updates, state.m)

    def _update_momentum(g, m):
      return helper.update_moments(count, g, _LionOptState(m=m), beta2)

    updated_moments = jax.tree.map(_update_momentum, updates, m_casted)

    updates = jax.tree.map(lambda g, m: jnp.sign((1.0 - beta1) * g + beta1 * m),
                           updates, m_casted)

    if update_capping > 0:
      updates = jax.tree.map(lambda x: helper.clip_update(x, update_capping),
                             updates)

    if weight_decay > 0:
      updates = jax.tree.map(lambda x, v: x + weight_decay * v, updates, params)

    step_size = -1.0 * learning_rate_fn(count)
    # Finally, fold in step size.
    updates = jax.tree.map(lambda x: step_size * x, updates)

    updated_states = NestedMap(
        count=count + 1,
        m=jax.tree.map(lambda x: x.m.astype(m_dtype), updated_moments),
    )
    return updates, updated_states

  return ShardedGradientTransformation(
      init=init_fn, update=update_fn, init_partition_spec=init_partition_spec_fn
  )


def apply_ema_weights(
    decay: float, use_optax_gradient_transformations: bool = False
) -> GeneralGradientTransformation:
  """Applies exponential moving average on weights.

  Note, this implementation averages the weight before optimization because
  trainable and non-trainable variables are handled separately. In such case
  the updates on non-trainable variables like bn stats are not available in
  updates.

  This differs from optax.ema which applies ema on gradients so it changes
  training process.

  ema = ema * decay + new_weight * (1.0 - decay)

  Args:
    decay: A float number represents the weight on the moving average.
    use_optax_gradient_transformations: Whether to use Optax based gradient
      transformation.

  Returns:
    A GeneralGradientTransformation applying ema.
  """

  def init_fn(params):
    return NestedMap(
        count=jnp.array(0, dtype=jnp.int32), ema=jax.tree.map(jnp.copy, params))

  def update_fn(updates, state, params):
    if params is None:
      raise ValueError('Params required for the EMA')

    # https://github.com/tensorflow/tensorflow/blob/v2.9.1/tensorflow/python/training/moving_averages.py#L469
    ema_decay = jnp.minimum(decay, (1. + state.count) / (10. + state.count))

    def update_func(old_v, new_v):
      if old_v.dtype == jnp.bool_ or jnp.issubdtype(old_v, jnp.integer):
        # If it is integer, we directly return the new variable
        # This is mainly supported for non_trainable
        return new_v
      else:
        return old_v - (1.0 - ema_decay) * (old_v - new_v)

    new_ema = jax.tree.map(update_func, state.ema, params)
    count_inc = state.count + jnp.array(1, jnp.int32)

    return updates, NestedMap(count=count_inc, ema=new_ema)

  def init_partition_spec_fn(params):

    def _infer_ema_pspec(x):
      return WeightHParams(
          shape=x.shape,
          init=None,
          dtype=x.dtype,
          collections=None,
          tensor_split_dims_mapping=x.tensor_split_dims_mapping,
      )

    return NestedMap(
        count=WeightHParams(
            shape=[],
            init=None,
            dtype=jnp.int32,
            collections=None,
            tensor_split_dims_mapping=[]),
        ema=jax.tree.map(_infer_ema_pspec, params))

  if use_optax_gradient_transformations:
    return optax.GradientTransformationExtraArgs(init=init_fn, update=update_fn)
  return ShardedGradientTransformation(
      init=init_fn,
      update=update_fn,
      init_partition_spec=init_partition_spec_fn)


def apply_ewc_regularization(
    learning_rate_fn: optax.Schedule,
    ewc_regularizer_weight: float = 0.0,
    ewc_weight_per_var: bool | None = False,
    use_optax_gradient_transformations: bool = False,
) -> GeneralGradientTransformation:
  """Applies EWC regularization on weights.

  paper: https://arxiv.org/abs/1612.00796
  EWC regularizer add a ewc_weight * 1 / 2 *(params - pretrain_params)**2 to
  the final objective loss.

  Args:
    learning_rate_fn: Learning rate function.
    ewc_regularizer_weight: A float number as the EWC regularization weight. if
      ewc_regularizer_weight <= 0, EWC is disabled.
    ewc_weight_per_var: EWC weight for each variable.
    use_optax_gradient_transformations: Whether to use Optax based gradient
      transformation.

  Returns:
    A GeneralGradientTransformation applying EWC regularizers.
  """
  def init_fn(params):
    if ewc_regularizer_weight > 0.0:
      var_weights = jax.tree.map(
          lambda v: jnp.array(ewc_regularizer_weight * 1.0 / 2, v.dtype), params
      )
      return NestedMap(count=jnp.array(0, dtype=jnp.int32),
                       pretrain_vars=jax.tree.map(jnp.copy, params),
                       var_weights=var_weights)
    else:
      return NestedMap(count=jnp.array(0, dtype=jnp.int32))

  def update_fn(updates, state, params):
    count = state.count
    lr = learning_rate_fn(count)
    if ewc_regularizer_weight > 0.0:
      if params is None:
        raise ValueError('Params required for the EWC')
      def fn(g, p, w, v):
        v = jnp.where(jnp.equal(state.count, 0), p, v)
        return g - lr * w * (p - v) if not py_utils.is_optax_masked_node(
            g) else optax.MaskedNode()

      def pretrain_fn(p, v):
        return jnp.where(jnp.equal(state.count, 0), p, v)

      if ewc_weight_per_var:
        if jnp.equal(state.count, 0):
          state.var_weights = jax.tree.map(
              lambda v, p, ewc_w: jnp.array(v * ewc_w, p.dtype),
              state.var_weights,
              params,
              ewc_weight_per_var,
          )

      updates = jax.tree.map(
          fn,
          updates,
          params,
          state.var_weights,
          state.pretrain_vars)

      update_states = NestedMap(
          count=state.count + 1,
          pretrain_vars=jax.tree.map(pretrain_fn, params, state.pretrain_vars),
          var_weights=state.var_weights)
    else:
      update_states = NestedMap(
          count=state.count + 1)
    return updates, update_states

  def init_partition_spec_fn(params):

    def _infer_ewc_pspec(x):
      return WeightHParams(
          shape=x.shape,
          init=None,
          dtype=x.dtype,
          collections=None,
          tensor_split_dims_mapping=x.tensor_split_dims_mapping)

    def _infer_ewc_weights_pspec(x):
      return WeightHParams(
          shape=[],
          init=None,
          dtype=x.dtype,
          collections=None,
          tensor_split_dims_mapping=[])

    if ewc_regularizer_weight > 0.0:
      return NestedMap(
          count=WeightHParams(
              shape=[],
              init=None,
              dtype=jnp.int32,
              collections=None,
              tensor_split_dims_mapping=[]),
          pretrain_vars=jax.tree.map(_infer_ewc_pspec, params),
          var_weights=jax.tree.map(_infer_ewc_weights_pspec, params))
    else:
      return NestedMap(
          count=WeightHParams(
              shape=[],
              init=None,
              dtype=jnp.int32,
              collections=None,
              tensor_split_dims_mapping=[],
          )
      )

  if use_optax_gradient_transformations:
    return optax.GradientTransformationExtraArgs(init=init_fn, update=update_fn)
  return ShardedGradientTransformation(
      init=init_fn, update=update_fn, init_partition_spec=init_partition_spec_fn
  )


@runtime_checkable
class Optimizer(Protocol):
  """A protocol for Optimizer to be used in Learner."""

  @property
  def clip_gradient_norm_to_value(self) -> float:
    """Clips gradient by global norm to this value."""
    pass

  @property
  def clip_gradient_single_norm_to_value(self) -> float:
    """Clip gradient by single tensor norm to this value."""
    pass

  @property
  def ema_decay(self) -> float:
    """Exposed hparams for optimizer."""
    pass

  @abc.abstractmethod
  def get_grad_transformation(
      self,
      var_weight_hparams: NestedWeightHParams | None = None,
      include_ema: bool = True,
  ) -> GeneralGradientTransformation:
    pass

  @abc.abstractmethod
  def get_learning_rate(self, step_count: JTensor) -> JTensor:
    pass


class OptaxOptimizer(base_hyperparams.FiddleBaseParameterizable):
  """Class that encapsulates optax transformations with common regularizers.

  Attributes:
    l2_regularizer_weight: If not None, L2 regularization to apply to the model
      weights. Otherwise, disable L2 regularization.
    l1_regularizer_weight: If not None, L1 regularization to apply to the model
      weights. Otherwise, disable L1 regularization.
    skip_lp_1d_vectors: If True, skip L1/L2 regularization for 1d vector vars.
    decoupled_weight_decay: If not None, (decoupled) weight decay to apply to
      the model weights. Otherwise, disable weight decay. Note, lp regularizers
      add loss to final loss objective, while decoupled weight decay adds decay
      directly into weights. They are different especially when there are moment
      statistics in optimizers. A good reference can be found in:
      https://www.fast.ai/2018/07/02/adam-weight-decay/#adamw
    clip_gradient_norm_to_value: Clip gradient by global norm to this value.
      This is similar to the bahaviour of tf.clip_by_global_norm. If you are
      looking for tf.clip_by_norm refer to clip_gradient_single_norm_to_value.
      Note these are mutually exclusive.
    clip_gradient_single_norm_to_value: Clip gradient by single tensor norm to
      this value. This is similar to the bahaviour of tf.clip_by_norm. Note this
      is mutually exclusive to using clip_gradient_norm_to_value.
    learning_rate: learning rate to use.
    lr_schedule: Learning rate decay schedule. The value returned by this
      schedule is *multiplied* by your base learning rate.
    ema_decay: If > 0, enable ExponentialMovingAverage during training with the
      give decay. Must be < 1. Disabled if <= 0.
    ewc_regularizer_weight: If > 0, EWC regularization is applied to the model
      weights.
    ewc_weight_per_var: If not None, set weight for each model weight, e.g.,
      refer to https://arxiv.org/abs/1612.00796 by using a Fisher information
      matrix.
    grad_tx: The gradient transformation used by this optimizer.
  """

  l2_regularizer_weight: float | None = None
  l1_regularizer_weight: float | None = None
  skip_lp_1d_vectors: bool = False
  decoupled_weight_decay: float | None = None
  clip_gradient_norm_to_value: float = 0.0
  clip_gradient_single_norm_to_value: float = 0.0
  learning_rate: float = 0.0
  lr_schedule: pax_fiddle.Config[schedules.BaseSchedule] | None = None
  ema_decay: float = 0.0
  ewc_regularizer_weight: float = 0.0
  ewc_weight_per_var: NestedMap | None = None
  grad_tx: optax.GradientTransformation | None = None

  _lr_schedule_inst: Any = dataclasses.field(init=False, repr=False)

  def __post_init__(self):
    self._lr_schedule_inst = instantiate(self.lr_schedule)
    # Should not mix L1, L2 regularizer and weight decay together.
    if self.l2_regularizer_weight and self.l1_regularizer_weight:
      raise ValueError('Should not mix L1 and L2 regularization.')
    if self.decoupled_weight_decay and (
        self.l2_regularizer_weight or self.l1_regularizer_weight
    ):
      raise ValueError(
          'Should not mix decoupled weight decay with L1 or L2 regularization.'
      )

  def get_grad_transformation(
      self,
      var_weight_hparams: NestedWeightHParams | None = None,
      include_ema: bool = True,
  ) -> optax.GradientTransformation:
    """Get the grad transformation corresponds to this optimizer config.

    This is the final gradient transformation that incorporates all
    transformations.

    Args:
      var_weight_hparams: Weight params of the vars. If provided, apply lp
        regularization and weight decay based on variable collections.
      include_ema: whether to include ema. For multi optimizer case, we disable
        it here and instead add ema in the beginning.

    Returns:
      Chained optax.GradientTransformation.
    """

    # Compute the mask for lp regularization
    if var_weight_hparams:
      var_lp_mask = jax.tree.map(
          lambda x: not base_layer.var_skip_lp_regularization(x),
          var_weight_hparams,
      )
    else:
      var_lp_mask = None

    transformations = [
        apply_lp_regularizer(
            var_lp_mask=var_lp_mask,
            regularizer_weight=self.l1_regularizer_weight,
            p=1.0,
            skip_lp_1d_vectors=self.skip_lp_1d_vectors,
            use_optax_gradient_transformations=True,
        ),
        apply_lp_regularizer(
            var_lp_mask=var_lp_mask,
            regularizer_weight=self.l2_regularizer_weight,
            p=2.0,
            skip_lp_1d_vectors=self.skip_lp_1d_vectors,
            use_optax_gradient_transformations=True,
        ),
        self.grad_tx,
        apply_decoupled_weight_decay(
            self.get_learning_rate,
            var_wd_mask=var_lp_mask,
            regularizer_weight=self.decoupled_weight_decay,
            use_optax_gradient_transformations=True,
        ),
    ]
    if self.ewc_regularizer_weight > 0.0:
      transformations.append(
          apply_ewc_regularization(
              self.get_learning_rate,
              ewc_regularizer_weight=self.ewc_regularizer_weight,
              ewc_weight_per_var=self.ewc_weight_per_var,
              use_optax_gradient_transformations=True,
          )
      )
    if self.ema_decay > 0.0 and include_ema:
      # EMA adds extra optimizer states making checkpoints not backward
      # compatible
      asserts.lt(self.ema_decay, 1.0)
      transformations.append(
          apply_ema_weights(
              decay=self.ema_decay, use_optax_gradient_transformations=True
          )
      )
    return optax.chain(*transformations)

  def get_learning_rate(self, step_count: JTensor) -> JTensor:
    """Get the learning rate of this optimizer at a particular step."""
    return self._lr_schedule_inst.value_at(step_count) * self.learning_rate


# TODO(b/277132394): OptaxOptimizer is introduced to use standard optax
# transforms with common regularizers. Learners can use OptaxOptimizer with
# optax gradient transforms directly. Custom optimizers won't need to depend on
# BaseOptimizer in the future, but will be implemented as standard optax
# transforms. Thus, BaseOptimizer class can, be removed.
class BaseOptimizer(base_hyperparams.FiddleBaseParameterizable):
  """Base class for all optimizers.

  Attributes:
    l2_regularizer_weight: If not None, L2 regularization to apply to the model
      weights. Otherwise, disable L2 regularization.
    l1_regularizer_weight: If not None, L1 regularization to apply to the model
      weights. Otherwise, disable L1 regularization.
    skip_lp_1d_vectors: If True, skip L1/L2 regularization for 1d vector vars.
    decoupled_weight_decay: If not None, (decoupled) weight decay to apply to
      the model weights. Otherwise, disable weight decay. Note, lp regularizers
      add loss to final loss objective, while decoupled weight decay adds decay
      directly into weights. They are different especially when there are moment
      statistics in optimizers. A good reference can be found in:
      https://www.fast.ai/2018/07/02/adam-weight-decay/#adamw
    clip_gradient_norm_to_value: Clip gradient by global norm to this value.
      This is similar to the behaviour of tf.clip_by_global_norm. If you are
      looking for tf.clip_by_norm refer to clip_gradient_single_norm_to_value.
      Note these are mutually exclusive.
    clip_gradient_single_norm_to_value: Clip gradient by single tensor norm to
      this value. This is similar to the behaviour of tf.clip_by_norm. Note this
      is mutually exclusive to using clip_gradient_norm_to_value.
    learning_rate: learning rate to use.
    lr_schedule: Learning rate decay schedule. The value returned by this
      schedule is *multiplied* by your base learning rate.
    ema_decay: If > 0, enable ExponentialMovingAverage during training with the
      give decay. Must be < 1. Disabled if <= 0.
    ewc_regularizer_weight: If > 0, EWC regularization is applied to the model
      weights.
    ewc_weight_per_var: If not None, set weight for each model weight, e.g.,
      refer to https://arxiv.org/abs/1612.00796 by using a Fisher information
      matrix.
  """

  l2_regularizer_weight: float | None = None
  l1_regularizer_weight: float | None = None
  skip_lp_1d_vectors: bool = False
  decoupled_weight_decay: float | None = None
  clip_gradient_norm_to_value: float = 0.0
  clip_gradient_single_norm_to_value: float = 0.0
  learning_rate: float = 0.0
  lr_schedule: pax_fiddle.Config[schedules.BaseSchedule] | None = None
  ema_decay: float = 0.0
  ewc_regularizer_weight: float = 0.0
  ewc_weight_per_var: NestedMap | None = None
  _lr_schedule_inst: Any = dataclasses.field(init=False, repr=False)

  def __post_init__(self):
    self._lr_schedule_inst = instantiate(self.lr_schedule)
    # Should not mix L1, L2 regularizer and weight decay together.
    if self.l2_regularizer_weight and self.l1_regularizer_weight:
      raise ValueError('Should not mix L1 and L2 regularization.')
    if self.decoupled_weight_decay and (
        self.l2_regularizer_weight or self.l1_regularizer_weight
    ):
      raise ValueError(
          'Should not mix decoupled weight decay with L1 or L2 regularization.')

  def get_learning_rate(self, step_count: JTensor) -> JTensor:
    """Get the learning rate of this optimizer at a particular step."""
    return self._lr_schedule_inst.value_at(step_count) * self.learning_rate

  def get_grad_transformation(
      self,
      var_weight_hparams: NestedWeightHParams | None = None,
      include_ema: bool = True,
  ) -> GeneralGradientTransformation:
    """Get the grad transformation corresponds to this optimizer config.

    This is the final gradient transformation that incorporates all
    transformations.

    Args:
      var_weight_hparams: Weight params of the vars. If provided, apply lp
        regularization and weight decay based on variable collections.
      include_ema: whether to include ema. For multi optimizer case, we disable
        it here and instead add ema in the beginning.

    Returns:
      an optax.GradientTransformation or ShardedGradientTransformation.
    """

    # Compute the mask for lp regularization
    if var_weight_hparams:
      var_lp_mask = jax.tree.map(
          lambda x: not base_layer.var_skip_lp_regularization(x),
          var_weight_hparams)
    else:
      var_lp_mask = None

    optax_list = [
        apply_lp_regularizer(
            var_lp_mask=var_lp_mask,
            regularizer_weight=self.l1_regularizer_weight,
            p=1.0,
            skip_lp_1d_vectors=self.skip_lp_1d_vectors,
        ),
        apply_lp_regularizer(
            var_lp_mask=var_lp_mask,
            regularizer_weight=self.l2_regularizer_weight,
            p=2.0,
            skip_lp_1d_vectors=self.skip_lp_1d_vectors,
        ),
        self._get_raw_grad_transformation(self.get_learning_rate),
        apply_decoupled_weight_decay(
            self.get_learning_rate,
            var_wd_mask=var_lp_mask,
            regularizer_weight=self.decoupled_weight_decay,
        ),
    ]
    if self.ewc_regularizer_weight > 0.0:
      optax_list.append(
          apply_ewc_regularization(
              self.get_learning_rate,
              ewc_regularizer_weight=self.ewc_regularizer_weight,
              ewc_weight_per_var=self.ewc_weight_per_var,
          )
      )
    if self.ema_decay > 0.0 and include_ema:
      # EMA adds new optimizer states that is not compatible
      asserts.lt(self.ema_decay, 1.0)
      optax_list.append(apply_ema_weights(decay=self.ema_decay))
    return sharded_chain(*optax_list)

  def _get_raw_grad_transformation(
      self, lr: optax.Schedule) -> GeneralGradientTransformation:
    """Get the raw optimizer transformation without taking into other ...

    transformations such l1/l2 regularization, gradient norm clipping, etc.

    Args:
      lr: an optax schedule.

    Returns:
      an optax.GradientTransformation or ShardedGradientTransformation.
    """
    raise NotImplementedError()


class Sgd(BaseOptimizer):
  """Canonical SGD optimizer.

  Attributes:
    momentum: Decay rate used by the momentum term. If set to None, momentum is
      not used.
    nesterov: Whether Nesterov momentum is used or not.
  """
  momentum: float | None = None
  nesterov: bool = False

  def _get_raw_grad_transformation(
      self, lr: optax.Schedule) -> optax.GradientTransformation:
    return optax.sgd(
        learning_rate=lr, momentum=self.momentum, nesterov=self.nesterov
    )


class Lamb(BaseOptimizer):
  """Canonical Lamb optimizer.

  Please check optax lamb doc for more details.

  Attributes:
    momentum: Decay rate used by the momentum term. If set to None, momentum is
      not used.
    b1: Exponential decay rate to track the first moment of past gradients.
      nesterov: Whether Nesterov momentum is used or not.
    b2: Exponential decay rate to track the second moment of past gradients.
    eps: A small constant applied to denominator outside of the square root (as
      in the Adam paper) to avoid dividing by zero when rescaling.
    eps_root: A small constant applied to denominator inside the square root (as
      in RMSProp), to avoid dividing by zero when rescaling.
    weight_decay: Strength of the weight decay regularization.
    mask: A boolean tree mask that determines which leaves to transform.
  """

  b1: float = 0.9
  b2: float = 0.999
  eps: float = 1e-6
  eps_root: float = 0.0
  weight_decay: float = 0.0
  mask: Any | None = None

  def _get_raw_grad_transformation(
      self, lr: optax.Schedule
  ) -> optax.GradientTransformation:
    return optax.lamb(
        learning_rate=lr,
        b1=self.b1,
        b2=self.b2,
        eps=self.eps,
        eps_root=self.eps_root,
        weight_decay=self.weight_decay,
        mask=self.mask,
    )


class Lars(BaseOptimizer):
  """Canonical Lars optimizer.

  Please check optax lars doc
  (https://optax.readthedocs.io/en/latest/api.html#optax.lars) for more details.

  Attributes:
    weight_decay: Strength of the weight decay regularization.
    weight_decay_mask: a PyTree with same structure as (or a prefix of) the
      params PyTree, or a Callable that returns such a pytree given the
      params/updates. The leaves should be booleans, ``True`` for
      leaves/subtrees you want to apply the transformation for weight decay to,
      and ``False`` for those you want to skip.
    trust_coefficient: A multiplier for the trust ratio.
    eps: Optional additive constant in the trust ratio denominator.
    trust_ratio_mask: a PyTree with same structure as (or a prefix of) the
      params PyTree, or a Callable that returns such a pytree given the
      params/updates. The leaves should be booleans, ``True`` for
      leaves/subtrees you want to apply the transformation for trust ratio to,
      and ``False`` for those you want to skip.
    momentum: Decay rate for momentum.
    nesterov: Whether to use Nesterov momentum.
  """

  weight_decay: float = 0.0
  weight_decay_mask: Any = True
  trust_coefficient: float = 0.001
  eps: float = 0.0
  trust_ratio_mask: Any = True
  momentum: float = 0.9
  nesterov: bool = False

  def _get_raw_grad_transformation(
      self, lr: optax.Schedule
  ) -> optax.GradientTransformation:
    return optax.lars(
        learning_rate=lr,
        weight_decay=self.weight_decay,
        weight_decay_mask=self.weight_decay_mask,
        trust_coefficient=self.trust_coefficient,
        eps=self.eps,
        trust_ratio_mask=self.trust_ratio_mask,
        momentum=self.momentum,
        nesterov=self.nesterov,
    )


class ShardedSgd(BaseOptimizer):
  """Sharded SGD optimizer.

  Attributes:
    momentum: Decay rate used by the momentum term. If set to None, momentum is
      not used.
    nesterov: Whether Nesterov momentum is used or not.
  """
  momentum: float | None = None
  nesterov: bool = False

  def _get_raw_grad_transformation(
      self, lr: optax.Schedule) -> ShardedGradientTransformation:
    return sharded_sgd(
        learning_rate_fn=lr, momentum=self.momentum, nesterov=self.nesterov
    )


class ShardedAdagrad(BaseOptimizer):
  """Sharded Adagrad optimizer.

  Attributes:
    initial_accumulator_value: Initial value of the accumulator.
    epsilon: Small constant applied to the denominator outside of the square
      root to avoid dividing by zero when rescaling.
  """
  initial_accumulator_value: float = 1e-12
  epsilon: float = 1e-12

  def _get_raw_grad_transformation(
      self, lr: optax.Schedule) -> ShardedGradientTransformation:
    return sharded_adagrad(
        learning_rate_fn=lr,
        initial_accumulator_value=self.initial_accumulator_value,
        epsilon=self.epsilon,
    )


class Adam(BaseOptimizer):
  """Adam optimizer.

  Attributes:
    beta1: Expenonential decay rate to track the first moment of past gradients.
    beta2: Exponential decay rate to track the second moment of past gradients.
    epsilon: Small constant applied to the denominator outside of the square
      root to avoid dividing by zero when rescaling.
    epsilon_root: Small constant applied to the denominator inside of the square
      root to avoid dividing by zero when rescaling.
    clip_threshold: An optional float to clip raw adam updates to.
    weight_decay: Decoupled weight decay to apply.
    sharded_adam: whether or not to use sharded_adam
    maybe_inf_to_nan: Will use jax.nan_to_num during update when True.
  """
  beta1: float = 0.9
  beta2: float = 0.999
  epsilon: float = 1e-6
  epsilon_root: float = 0.0
  clip_threshold: float = 1.0
  weight_decay: float = 0.0
  sharded_adam: bool = True
  maybe_inf_to_nan: bool = True

  @classmethod
  def HParamsA(cls) -> pax_fiddle.Config[Adam]:  # pylint: disable=invalid-name
    """Convenient method for a commonly used Adam config."""
    return pax_fiddle.Config(cls, beta1=0.9, beta2=0.997, epsilon=1e-9)

  @classmethod
  def HParamsB(cls) -> pax_fiddle.Config[Adam]:  # pylint: disable=invalid-name
    """Convenient method for another commonly used Adam config."""
    return pax_fiddle.Config(cls, beta1=0.9, beta2=0.98, epsilon=1e-9)

  def _get_raw_grad_transformation(
      self, lr: optax.Schedule) -> GeneralGradientTransformation:
    if self.weight_decay:
      logging.warning(_WEIGHT_DECAY_DEPRECATION)

    if self.sharded_adam:
      logging.info('Using sharded_adam.')
      return sharded_adam(
          learning_rate_fn=lr,
          beta1=self.beta1,
          beta2=self.beta2,
          epsilon=self.epsilon,
          epsilon_root=self.epsilon_root,
          update_capping=self.clip_threshold,
          weight_decay=self.weight_decay,
          maybe_inf_to_nan=self.maybe_inf_to_nan,
      )
    else:
      logging.info('Using optax.adam.')
      return optax.adam(
          learning_rate=lr,
          b1=self.beta1,
          b2=self.beta2,
          eps=self.epsilon,
          eps_root=self.epsilon_root,
      )


class Lion(BaseOptimizer):
  """Implementation of the Lion optimizer.

  Lion optimizer from the Symbolic Discovery of Optimization Algorithms paper.

  Attributes:
    beta1: Rate to combine the moment and the current gradient.
    beta2: Exponential decay rate to track the moment of past gradients.
    clip_threshold: An optional float to clip raw Lion updates to.
    weight_decay: Decoupled weight decay to apply.
  """
  beta1: float = 0.9
  beta2: float = 0.99
  clip_threshold: float = 1.0
  weight_decay: float = 0.0
  m_dtype: jnp.dtype = jnp.bfloat16

  def _get_raw_grad_transformation(
      self, lr: optax.Schedule) -> ShardedGradientTransformation:
    logging.info('Using sharded_lion.')
    return sharded_lion(
        learning_rate_fn=lr,
        beta1=self.beta1,
        beta2=self.beta2,
        m_dtype=self.m_dtype,
        update_capping=self.clip_threshold,
        weight_decay=self.weight_decay,
    )


class Adafactor(BaseOptimizer):
  """Adafactor optimizer from Optax.

  Attributes:
    min_dim_size_to_factor: Only factor the statistics if two array dimensions
      have at least this size.
    decay_rate: Controls second-moment exponential decay schedule.
    decay_offset: For finetuning, one may set this to the starting step number
      of the finetuning phase.
    multiply_by_parameter_scale: If True, then scale learning_rate by parameter
      norm. if False, provided learning_rate is absolute step size.
    clip_threshold: Optional value; if None, clipping disabled.
    momentum: Optional value between 0 and 1, enables momentum and uses extra
      memory if non-None! None by default.
    dtype_momentum: dtype of momentum buffers.
    weight_decay_rate: Optional rate at which to decay weights.
    eps: Regularization constant for root mean squared gradient.
    factored: Whether to use factored second-moment estimates.
  """
  min_dim_size_to_factor: int = 128
  decay_rate: float = 0.8
  decay_offset: int = 0
  multiply_by_parameter_scale: bool = True
  clip_threshold: float | None = 1.0
  momentum: float | None = None
  dtype_momentum: str = 'float32'
  weight_decay_rate: float | None = None
  eps: float = 1e-30
  factored: bool = True

  def _get_raw_grad_transformation(
      self, lr: optax.Schedule) -> optax.GradientTransformation:
    if self.weight_decay_rate:
      logging.warning(_WEIGHT_DECAY_RATE_DEPRECATION)
    return optax.adafactor(
        learning_rate=lr,
        min_dim_size_to_factor=self.min_dim_size_to_factor,
        decay_rate=self.decay_rate,
        decay_offset=self.decay_offset,
        multiply_by_parameter_scale=self.multiply_by_parameter_scale,
        clipping_threshold=self.clip_threshold,
        momentum=self.momentum,
        dtype_momentum=getattr(jnp, self.dtype_momentum),
        weight_decay_rate=self.weight_decay_rate,
        eps=self.eps,
        factored=self.factored,
    )


class DistributedShampoo(BaseOptimizer):
  """DistributedShampoo optimizer from Optax.

  Attributes:
    block_size: Size of the preconditioner (block size x block size).
    beta1: Momentum parameter.
    beta2: Second moment averaging parameter.
    diagonal_epsilon: Epsilon parameter for the diagonal adaptive method.
    matrix_epsilon: Epsilon parameter as part of computing the inverse-pth
      roots.
    weight_decay: Weight decay.
    start_preconditioning_step: Start preconditionining after N steps.
    preconditioning_compute_steps: How often to compute the inverse-pth roots.
    statistics_compute_steps: How often to compute the statistics.
    graft_type: Type of Grafting. 1 for SGD, 2 for AdaGrad, 3 for RMSPROP .
    batch_axis_name: Batch axis name for pmap.
    mesh_axis_names: Axis names for the mesh (used in pjit).
    num_devices_for_pjit: Number of devices to parallelize over in pjit mode.
    nesterov: Use nesterov update for momentum.
    exponent_override: Exponent override.
    inverse_failure_threshold: Numerics are hard and inverses fail sometimes; we
      determine that using this threshold.
    moving_average_for_momentum: Moving average for momentum.
    skip_preconditioning_dim_size_gt: Skips preconditioning if any dim is
      greater than this value.
    clip_by_scaled_gradient_norm: Clip by scaled gradient norm (if not None).
    best_effort_shape_interpretation: Best effort shape interpretation to
      coalesce dimensions.
    tensor_split_dims_mapping: Sharding information for statistics and
      preconditioner matrices.
    tensor_split_dims_mapping_for_inverse_pth_root: Sharding information for
      preconditioner matrices.
    best_effort_memory_usage_reduction: Experimental mode: Best effort memory
      usage reduction.
    merge_small_dims_block_size: Block size for merging dims.
    lobpcg_topk_precondition: If nonzero, specifies the number of top
      eigenvectors to subtract out before performing LOBPCG
    lobpcg_max_iter: If nonzero, specifies the maximum number of iterations to
      perform LOBPCG if activated by lobpcg_topk_precondition. If zero, uses a
      default value equal to `lobpcg_topk_precondition` itself.
    skip_preconditioning_rank_lt: Skips preconditioning if param rank is less
      than this value.
    summarize_training_metrics: Summarize training statistics (for example:
      inverse pth root)
    decoupled_weight_decay_from_momentum: Decouple weight decay from momentum.
    decoupled_learning_rate_from_momentum: Decouple learning rate from momentum.
    eigh: If True, uses eigendecomposition to compute inverse-pth roots.
    compression_rank: If nonzero, whether to use low-rank preconditioners.
    frequent_directions: Use frequent directions sketching for preconditioner.
    average_grad: Whether to average gradients before usage in FD statistics.
    reuse_preconditioner: Wire in previous preconditioner for root updates. must
      be set to true for frequent directions.
  """
  block_size: int = 1024
  beta1: float = 0.9
  beta2: float = 0.999
  diagonal_epsilon: float = 1e-16
  matrix_epsilon: float = 1e-6
  weight_decay: float = 0.0
  start_preconditioning_step: int = 101
  preconditioning_compute_steps: int = 100
  statistics_compute_steps: int = 1
  graft_type: Any = GraftingType.ADAGRAD
  batch_axis_name: str = 'batch'
  mesh_axis_names: Sequence[str] | None = None
  num_devices_for_pjit: int | None = None
  nesterov: bool = True
  exponent_override: int = 0
  inverse_failure_threshold: float = 0.1
  moving_average_for_momentum: bool = False
  skip_preconditioning_dim_size_gt: int = 4096
  clip_by_scaled_gradient_norm: float | None = None
  best_effort_shape_interpretation: bool = True
  tensor_split_dims_mapping: Sequence[int] = pax_fiddle.instance_field(
      default_factory=lambda: [-1, 1, -1]
  )
  tensor_split_dims_mapping_for_inverse_pth_root: Sequence[int] = (
      pax_fiddle.instance_field(default_factory=lambda: [-1, 1, -1])
  )
  best_effort_memory_usage_reduction: bool = False
  relative_matrix_epsilon: bool = True
  cholesky: bool = False
  qr_based_root: bool = False
  merge_small_dims_block_size: int = 4096
  lobpcg_topk_precondition: int = 0
  lobpcg_max_iter: int = 0
  skip_preconditioning_rank_lt: int = 1
  summarize_training_metrics: bool = True
  decoupled_weight_decay_from_momentum: bool = True
  decoupled_learning_rate_from_momentum: bool = False
  eigh: bool = False
  compression_rank: int = 0
  frequent_directions: bool = False
  average_grad: bool = False
  reuse_preconditioner: bool = False
  _shard_optimizer_states: Any = dataclasses.field(init=False, repr=False)
  _statistics_partition_spec: Any = dataclasses.field(init=False, repr=False)
  _preconditioner_partition_spec: Any = dataclasses.field(
      init=False, repr=False
  )

  def __post_init__(self):
    super().__post_init__()
    self._shard_optimizer_states = False
    self._statistics_partition_spec = None
    self._preconditioner_partition_spec = None

  @classmethod
  def HParamsImageClassification(cls) -> pax_fiddle.Config[DistributedShampoo]:  # pylint: disable=invalid-name
    """Common Shampoo config for Image Classification."""
    return pax_fiddle.Config(
        cls,
        beta1=0.9,
        beta2=0.95,
        block_size=128,
        weight_decay=1e-4,
        nesterov=True,
        preconditioning_compute_steps=1,
        statistics_compute_steps=1,
        graft_type=GraftingType.SGD)

  @classmethod
  def HParamsLanguageModeling(cls) -> pax_fiddle.Config[DistributedShampoo]:  # pylint: disable=invalid-name
    """Common Shampoo config for Language Modeling."""
    return pax_fiddle.Config(
        cls,
        block_size=1536,
        beta1=0.9,
        beta2=0.999,
        clip_gradient_norm_to_value=5.0,
        weight_decay=0.0,
        matrix_epsilon=1e-8,
        graft_type=GraftingType.RMSPROP_NORMALIZED,
        nesterov=False,
        exponent_override=0,
        start_preconditioning_step=51,
        preconditioning_compute_steps=50,
        skip_preconditioning_dim_size_gt=4096,
        moving_average_for_momentum=True,
        clip_by_scaled_gradient_norm=None)

  def _get_raw_grad_transformation(
      self, lr: optax.Schedule) -> optax.GradientTransformation:
    grad_transformation = self._shampoo_transformation(lr)

    def wrapped_update_fn(grads, state, params=None):
      new_params, new_state = grad_transformation.update(grads, state, params)
      if self.summarize_training_metrics:
        param_stats = new_state.stats  # pytype: disable=attribute-error  # numpy-scalars

        # Construct an almost parallel-structured pytree with key prefixes to
        # annotate per-parameter metrics for the summary name. Frustratingly,
        # this ignores non-pytree-node lists, creating separate prefixes for
        # what should be a leaf list of integers representing shapes.
        keys = py_utils.extract_prefixed_keys_from_nested_map(param_stats)

        # Luckily, extracting just the required TrainingMetrics nodes
        # collates generated prefixes perfectly for any top-down flatten
        # ordering.
        is_metrics = lambda l: isinstance(l, TrainingMetrics)
        flatten = lambda tree: jax.tree_util.tree_flatten(  # pylint: disable=g-long-lambda
            tree, is_leaf=is_metrics
        )[0]
        training_metrics = [x for x in flatten(param_stats) if is_metrics(x)]
        training_metrics_keys = [x for x in flatten(keys) if is_metrics(x)]

        assert len(training_metrics) == len(training_metrics_keys)
        for metrics, keys in zip(training_metrics, training_metrics_keys):
          # Walk all training metrics fields and summarize, assuming
          # they're scalars, using their key prefixes.
          jax.tree.map(base_layer.add_global_summary, keys, metrics)

      return new_params, new_state

    return optax.GradientTransformation(grad_transformation.init,
                                        wrapped_update_fn)

  def _shampoo_transformation(self, lr):
    return distributed_shampoo_optimizer(
        learning_rate=lr,
        block_size=self.block_size,
        beta1=self.beta1,
        beta2=self.beta2,
        diagonal_epsilon=self.diagonal_epsilon,
        matrix_epsilon=self.matrix_epsilon,
        weight_decay=self.weight_decay,
        start_preconditioning_step=self.start_preconditioning_step,
        preconditioning_compute_steps=self.preconditioning_compute_steps,
        statistics_compute_steps=self.statistics_compute_steps,
        best_effort_shape_interpretation=self.best_effort_shape_interpretation,
        graft_type=self.graft_type,
        nesterov=self.nesterov,
        exponent_override=self.exponent_override,
        batch_axis_name=self.batch_axis_name,
        num_devices_for_pjit=self.num_devices_for_pjit,
        statistics_partition_spec=self._statistics_partition_spec,
        preconditioner_partition_spec=self._preconditioner_partition_spec,
        shard_optimizer_states=self._shard_optimizer_states,
        inverse_failure_threshold=self.inverse_failure_threshold,
        moving_average_for_momentum=self.moving_average_for_momentum,
        skip_preconditioning_dim_size_gt=self.skip_preconditioning_dim_size_gt,
        clip_by_scaled_gradient_norm=self.clip_by_scaled_gradient_norm,
        precision=lax.Precision.HIGHEST,
        best_effort_memory_usage_reduction=self.best_effort_memory_usage_reduction,
        relative_matrix_epsilon=self.relative_matrix_epsilon,
        lobpcg_topk_precondition=self.lobpcg_topk_precondition,
        lobpcg_max_iter=self.lobpcg_max_iter,
        cholesky=self.cholesky,
        qr_based_root=self.qr_based_root,
        merge_small_dims_block_size=self.merge_small_dims_block_size,
        skip_preconditioning_rank_lt=self.skip_preconditioning_rank_lt,
        decoupled_weight_decay=self.decoupled_weight_decay_from_momentum,
        decoupled_learning_rate=self.decoupled_learning_rate_from_momentum,
        generate_training_metrics=self.summarize_training_metrics,
        eigh=self.eigh,
        compression_rank=self.compression_rank,
        frequent_directions=self.frequent_directions,
        average_grad=self.average_grad,
        reuse_preconditioner=self.reuse_preconditioner,
    )


class ShardedDistributedShampoo(DistributedShampoo):
  """Sharded version of distributed shampoo for model parallel training."""
  _shard_optimizer_states: Any = dataclasses.field(init=False, repr=False)
  _statistics_partition_spec: Any = dataclasses.field(init=False, repr=False)
  _preconditioner_partition_spec: Any = dataclasses.field(
      init=False, repr=False
  )

  def __post_init__(self):
    super().__post_init__()
    self._shard_optimizer_states = True
    self._statistics_partition_spec = jax.sharding.PartitionSpec(
        *self._sharded_axes(
            self.mesh_axis_names,
            self.tensor_split_dims_mapping,
        )
    )
    self._preconditioner_partition_spec = jax.sharding.PartitionSpec(
        *self._sharded_axes(
            self.mesh_axis_names,
            self.tensor_split_dims_mapping_for_inverse_pth_root,
        )
    )

  @classmethod
  def HParamsLargeLanguageModeling(
      cls,
  ) -> pax_fiddle.Config[DistributedShampoo]:  # pylint: disable=invalid-name
    """Common Shampoo config for Large Language Modeling (8B+)."""
    return pax_fiddle.Config(
        cls,
        block_size=4096,
        # Should be ~block_size x 3
        merge_small_dims_block_size=4096 * 3,
        # Larger than largest dim of dense layers.
        skip_preconditioning_dim_size_gt=4096 * 6,
        # AdaGrad is used for grafting.
        graft_type=GraftingType.RMSPROP,
        clip_gradient_norm_to_value=1.0,
        clip_by_scaled_gradient_norm=1.0,
        # TODO(rohananil): Ablate this with 0.9 which is the default for.
        # ShardedAdaFactor
        beta1=0.9,
        beta2=0.99,
        weight_decay=0.0005,
        matrix_epsilon=1e-8,
        nesterov=False,
        exponent_override=0,
        mesh_axis_names=('replica', 'data', 'mdl'),
        tensor_split_dims_mapping=[-1, 1, 2],
        tensor_split_dims_mapping_for_inverse_pth_root=[1, -1, 2],
        start_preconditioning_step=51,
        preconditioning_compute_steps=50,
        # With static accumulator summaries don't work yet.
        summarize_training_metrics=False,
        moving_average_for_momentum=True)

  def _sharded_axes(self, axes_names, tensor_split_dims_mapping):
    """Returns the axes to shard with."""
    axes = []
    if not tensor_split_dims_mapping:
      return [None]
    for tsdm in tensor_split_dims_mapping:
      if isinstance(tsdm, str):
        axes.append(tsdm)
      elif tsdm and tsdm != -1:
        axes.append(axes_names[tsdm])
      elif tsdm == -1 or not tsdm:
        axes.append(None)
    return tuple(axes)

  def init_partition_spec_fn(self, init_pspec, init_shapes_dtypes, axes_names,
                             params):
    """Annotates the PartitionSpec for optimizer states."""
    param_pspec_flattened, _ = jax.tree_util.tree_flatten(params)
    assert param_pspec_flattened
    first_param = param_pspec_flattened[0]
    assert isinstance(first_param, WeightHParams)
    assert len(axes_names) == len(self.tensor_split_dims_mapping)
    mesh_shape = first_param.mesh_shape

    partition_spec_statistics = jax.sharding.PartitionSpec(
        *self._sharded_axes(axes_names, self.tensor_split_dims_mapping)
    )

    def _pspec_from_weight_param(param):
      p = jax.sharding.PartitionSpec(
          *self._sharded_axes(axes_names, param.tensor_split_dims_mapping))
      return p

    partition_spec_params = jax.tree.map(_pspec_from_weight_param, params)
    shapes_and_dtypes = init_shapes_dtypes(params)
    partition_spec_opt_state = init_pspec(params, partition_spec_params,
                                          partition_spec_statistics)

    def _weight_param_from_pspec_shape_dtype(pspec, shapes_and_dtypes):
      if not pspec:
        if len(shapes_and_dtypes[0]) == 1:
          tensor_split_dims_mapping = [-1]
        else:
          tensor_split_dims_mapping = []
      else:
        if len(pspec) == 1 and not pspec[0]:
          if len(shapes_and_dtypes[0]) == 1:
            tensor_split_dims_mapping = [-1]
          else:
            tensor_split_dims_mapping = []
        else:
          tensor_split_dims_mapping = [
              axes_names.index(axis) if axis else -1 for axis in pspec
          ]
      assert len(shapes_and_dtypes[0]) == len(tensor_split_dims_mapping)
      return WeightHParams(
          shape=shapes_and_dtypes[0],
          init=None,
          dtype=shapes_and_dtypes[1],
          collections=None,
          mesh_shape=mesh_shape,
          tensor_split_dims_mapping=tensor_split_dims_mapping)

    return jax.tree.map(_weight_param_from_pspec_shape_dtype,
                        partition_spec_opt_state, shapes_and_dtypes)

  def _get_raw_grad_transformation(  # pytype: disable=signature-mismatch  # overriding-return-type-checks
      self, lr: optax.Schedule) -> ShardedGradientTransformation:
    result = self._shampoo_transformation(lr)
    # TODO(rohananil): Refactor after PartitionSpec layering is finalized in
    # the JAX ecosystem.
    fns = result.init(None)  # pytype: disable=wrong-arg-types  # numpy-scalars

    def _wrapped_update_fn(grads, state, params):
      new_params, new_state = result.update(grads, state, params)
      if self.summarize_training_metrics:
        local_stats = new_state.stats.local_stats  # pytype: disable=attribute-error  # numpy-scalars
        var_keys, _ = jax.tree_util.tree_flatten(
            py_utils.extract_prefixed_keys_from_nested_map(local_stats))
        var_keys = [x for x in var_keys if 'inverse_pth_root_errors' in x]
        is_stats = lambda l: isinstance(l, (LocalShardedParameterStats))
        local_stats_flattened, _ = jax.tree_util.tree_flatten(
            local_stats, is_stats)

        def add_summary(key, local_stat):
          num_statistics = len(local_stat.sizes)
          for i in range(num_statistics):
            value = local_stat.training_metrics.inverse_pth_root_errors[i]
            base_layer.add_global_summary(f'inverse_pth_root_errors/{key}_{i}',
                                          value)

        assert len(var_keys) == len(local_stats_flattened)
        for key, local_stat in zip(var_keys, local_stats_flattened):
          add_summary(key, local_stat)
      return new_params, new_state

    return ShardedGradientTransformation(
        init=fns.init_fn,
        update=_wrapped_update_fn,
        init_partition_spec=functools.partial(
            self.init_partition_spec_fn,
            fns.pspec_fn,
            fns.shape_and_dtype_fn,
            self.mesh_axis_names,
        ),
    )


class Adagrad(BaseOptimizer):
  """Adagrad optimizer.

  Attributes:
    initial_accumulator_value: Initial value of the accumulator.
    epsilon: Small constant applied to the denominator outside of the square
      root to avoid dividing by zero when rescaling.
  """
  initial_accumulator_value: float = 0.1
  epsilon: float = 1e-10

  def _get_raw_grad_transformation(
      self, lr: optax.Schedule) -> optax.GradientTransformation:
    return optax.adagrad(
        learning_rate=lr,
        initial_accumulator_value=self.initial_accumulator_value,
        eps=self.epsilon,
    )


def to_quantized(
    fvalue: JTensor, quantized_dtype: jnp.dtype
) -> tuple[JTensor, JTensor]:
  """Converts floating point values `fvalues` to quantized values.

  We use a very simple quantization scheme where the range is symmetric around
  0.0, and we simply map 0 to 0.0.

  Let x = bucket_size
  We map [-0.5x, 0.5x] to 0
         [-1.5x, -0.5x] to -1
         [0.5x, 1.5x] to 1
         and so on so forth.

  Some properties:
    a1, a2 = to_quantized(x, quantized_dtype)
    b1 = to_float(a1, a2)
    c1, c2 = to_quantized(b1, quantized_dtype)

    then a1 == c1, a2 == c2

  Args:
    fvalue: Values in floating point.
    quantized_dtype: Quantized dtype, can be either jnp.int8, or jnp.int16.

  Returns:
    A (quantized_values, bucket_size) 2-tuple.
    `quantized_values * bucket_size[jnp.newaxis, ...]` are the quantized
    values
    on the floating value axis.
  """
  float_dtype = fvalue.dtype
  if quantized_dtype == jnp.int8:
    # value -128 is not used.
    num_buckets = jnp.array(127.0, dtype=float_dtype)
  elif quantized_dtype == jnp.int16:
    # value -32768 is not used.
    num_buckets = jnp.array(32767.0, dtype=float_dtype)
  else:
    raise ValueError(f'Quantized dtype {quantized_dtype} not supported.')
  # max value is mapped to num_buckets

  # We first decide the scale.
  if fvalue.ndim < 1:
    raise ValueError(
        f'Input array {fvalue} must have a strictly positive number of '
        'dimensions.')

  max_abs = jnp.max(jnp.abs(fvalue), axis=0)
  bucket_size = max_abs / num_buckets
  bs_expanded = bucket_size[jnp.newaxis, ...]
  # To avoid divide by 0.0
  bs_nonzero = jnp.where(bs_expanded > 0.0, bs_expanded,
                         jnp.ones_like(bs_expanded))
  ratio = fvalue / bs_nonzero
  # We use rounding to remove bias.
  quantized = jnp.round(ratio)
  return quantized.astype(quantized_dtype), bucket_size


def to_float(quantized: JTensor, bucket_size: JTensor) -> JTensor:
  """Converts quantized values to float values.

  Args:
    quantized: Quantized values, of type either jnp.int8 or jnp.int16.
    bucket_size: The size of each bucket on the floating-point axis. bucket_size
      is of rank quantized.ndim - 1. For example, if quantized is of shape
      [x, ...], bucket_size is of shape [...].

  Returns:
    Unquantized values of type bucket_size.dtype.
  """
  float_dtype = bucket_size.dtype
  bucket_size = bucket_size[jnp.newaxis, ...]
  return quantized.astype(float_dtype) * bucket_size


def adafactor_decay_rate_adam(beta2: float, step_counter: JTensor) -> JTensor:
  """Second-moment decay rate like Adam, subsuming the correction factor.

  Args:
    beta2: A floating point value between 0 and 1.
    step_counter: A scalar tensor keeping track of the number of steps
      performed.

  Returns:
    The decay rate as a scalar JTensor.
  """
  step = step_counter
  beta2 = jnp.array(beta2, dtype=jnp.float32)
  t = step + 1.
  return beta2 * (1. - jnp.power(beta2, t - 1.)) / (1. - jnp.power(beta2, t))


def adafactor_decay_rate_pow(exponent: float, step_counter: JTensor) -> JTensor:
  """Second moment decay rate where memory-length grows as step_num^exponent.

  Args:
    exponent: A floating point value between 0 and 1.
    step_counter: A scalar tensor keeping track of the number of steps
      performed.

  Returns:
    The decay rate as a scalar JTensor.
  """
  step = step_counter
  exponent = jnp.array(exponent, dtype=jnp.float32)
  return 1. - jnp.power((step + 1.), -exponent)


def reduce_mean(array: JTensor) -> JTensor:
  """Computes the mean of `array` in a more numerically stable way.

  Args:
    array: Input array.

  Returns:
    The mean of the input array as a scalar array.
  """
  num_elements = array.size
  if num_elements > 1e8:
    # When x is too large, simple jnp.mean() can result in nan or inf values.
    # TODO(bf-jax): The following code snippet is consistent with the TensorFlow
    # implementation. This can be simplified into `jnp.mean(jnp.mean(x, -1))`.
    # Update to using mean() after verifying consistency.
    array_sum = jnp.sum(array, axis=-1)
    array_sum = jnp.sum(array_sum)
    return array_sum / jnp.array(num_elements, dtype=array_sum.dtype)
  else:
    return jnp.mean(array)


def reduce_rms(array: JTensor) -> JTensor:
  """Computes the RMS of `array` (in a numerically stable way).

  Args:
    array: Input array.

  Returns:
    The root mean square of the input array as a scalar array.
  """
  sq = jnp.square(array)
  sq_mean = reduce_mean(sq)
  return jnp.sqrt(sq_mean)


@dataclasses.dataclass(frozen=True)
class _ShardedAdafactorUpdateResult:
  """Structure containing per-variable info for Adafactor."""
  update: Any | None
  m: Any | None
  m_scale: Any | None
  vr: Any | None
  vc: Any | None
  v: Any | None


class ShardedAdafactorState(NamedTuple):
  """Overall state of the ShardedAdafactor optimizer."""
  count: JTensor
  m: NestedJTensor | None
  m_scale: NestedJTensor | None
  vr: NestedJTensor | None
  vc: NestedJTensor | None
  v: NestedJTensor | None


class _ShardedAdafactorHelper:
  """Helper class to implement optax-based sharded Adafactor."""

  def __init__(
      self,
      learning_rate_fn: optax.Schedule,
      weight_decay: float | None,
      layerwise_adaptation: bool,
      decay_method: str,
      decay_adam: float,
      decay_pow: float,
      beta1: float,
      clip_threshold: float | None,
      factored: bool,
      epsilon1_grad_sq_reg: float,
      quantized_dtype: jnp.dtype,
      # TODO(bf-jax) Update default value to True, once this is supported.
      respect_skip_lp_regularization: bool,
      exclude_from_layerwise_adaptation: list[str] | None,
      per_var_learning_summary: bool,
      sort_factored_second_moment_dims: bool,
      min_dim_size_to_factor: int,
      multiply_by_parameter_scale: bool,
      epsilon2_param_scale_reg: float,
      maybe_inf_to_nan: bool,
      name: str,
  ) -> None:
    """Constructor. See ShardedAdafactor() below."""
    if weight_decay:
      logging.warning(_WEIGHT_DECAY_DEPRECATION)

    self._learning_rate_fn = learning_rate_fn
    self._weight_decay = weight_decay
    self._layerwise_adaptation = layerwise_adaptation
    self._decay_method = decay_method
    self._decay_adam = decay_adam
    self._decay_pow = decay_pow
    self._beta1 = beta1
    self._clip_threshold = clip_threshold
    self._factored = factored
    self._epsilon1 = epsilon1_grad_sq_reg
    self._quantized_dtype = quantized_dtype
    self._respect_skip_lp_regularization = respect_skip_lp_regularization
    self._exclude_from_layerwise_adaptation = exclude_from_layerwise_adaptation
    self._per_var_learning_summary = per_var_learning_summary
    self._sort_factored_second_moment_dims = sort_factored_second_moment_dims
    self._min_dim_size_to_factor = min_dim_size_to_factor
    self._multiply_by_parameter_scale = multiply_by_parameter_scale
    self._epsilon2 = epsilon2_param_scale_reg
    self._maybe_inf_to_nan = maybe_inf_to_nan
    self._name = name

  def should_use_factored_second_moment_estimate(self, shape):
    """Should we use a factored second moment estimator.

    Based on the shape of the variable.

    Args:
      shape: a list of integers.

    Returns:
      A boolean.
    """
    return self.factored_second_moment_dims(shape) is not None

  def factored_second_moment_dims(self, shape):
    """Should we use a factored second moment estimator.

    We select largest and second largest var dims as row and colum dims.

    Default list of factored dims is -1, -2.

    Args:
      shape: a list of integers.

    Returns:
      either a list of 2 Dimension indices for row and col or None
    """
    if not self._factored:
      return None
    if len(shape) < 2:
      return None
    if not self._sort_factored_second_moment_dims:
      return len(shape) - 1, len(shape) - 2

    def largest_two_dim_indices():
      s = [(s, i) for i, s in enumerate(shape)]
      sorted_dims = sorted(s, key=lambda d: -d[0])
      return sorted_dims[0][1], sorted_dims[1][1]

    r_idx, c_idx = largest_two_dim_indices()
    if shape[c_idx] < self._min_dim_size_to_factor:
      return None
    return r_idx, c_idx

  def should_store_momentum_in_qint(self, shape):
    """Should we store momentum as quantized integers.

    Based on the shape of the variable.

    Args:
      shape: a list of integers

    Returns:
      A boolean.
    """
    if jnp.issubdtype(self._quantized_dtype, jnp.floating):
      return False
    if self._quantized_dtype is None:
      return False
    return len(shape) >= 1

  def to_state(self, count, result_tree):
    """Maps from a tree of (factored) values to separate trees of values."""
    return ShardedAdafactorState(  # pytype: disable=wrong-arg-types  # jax-ndarray
        count=count,
        m=jax.tree.map(lambda o: o.m, result_tree),
        m_scale=jax.tree.map(lambda o: o.m_scale, result_tree),
        vr=jax.tree.map(lambda o: o.vr, result_tree),
        vc=jax.tree.map(lambda o: o.vc, result_tree),
        v=jax.tree.map(lambda o: o.v, result_tree))

  def init(self, param):
    """Initializes the optimizer state for a given param."""
    # The actually value that will be added to a variable for updating it.
    output_update = jnp.zeros((1,))
    output_m = jnp.zeros((1,))
    output_m_scale = jnp.zeros((1,))
    output_vr = jnp.zeros((1,))
    output_vc = jnp.zeros((1,))
    output_v = jnp.zeros((1,))
    shape = param.shape
    if self._beta1:
      if jnp.issubdtype(self._quantized_dtype, jnp.floating):
        output_m = jnp.zeros(shape, dtype=self._quantized_dtype)
      elif self.should_store_momentum_in_qint(shape):
        output_m = jnp.zeros(shape, dtype=self._quantized_dtype)
        scale_shape = shape[1:]
        output_m_scale = jnp.zeros(scale_shape, dtype=jnp.float32)
      else:
        output_m = jnp.zeros(shape, dtype=jnp.float32)
    if self.should_use_factored_second_moment_estimate(shape):
      factored_dims = self.factored_second_moment_dims(shape)
      vr_axis, vc_axis = factored_dims  # pytype: disable=attribute-error
      output_vr_shape = list(shape).copy()
      del output_vr_shape[vr_axis]
      output_vc_shape = list(shape).copy()
      del output_vc_shape[vc_axis]
      output_vr = jnp.zeros(output_vr_shape, dtype=jnp.float32)
      output_vc = jnp.zeros(output_vc_shape, dtype=jnp.float32)
    else:
      output_v = jnp.zeros(shape, dtype=jnp.float32)
    return _ShardedAdafactorUpdateResult(
        update=output_update,
        m=output_m,
        m_scale=output_m_scale,
        vr=output_vr,
        vc=output_vc,
        v=output_v)

  def init_partition_spec(self, var_param):
    """Initializes the partition spec for a given param."""
    output_update = WeightHParams((1,))
    output_m = WeightHParams((1,))
    output_m_scale = WeightHParams((1,))
    output_vr = WeightHParams((1,))
    output_vc = WeightHParams((1,))
    output_v = WeightHParams((1,))
    shape = var_param.shape
    tensor_split_dims_mapping = var_param.tensor_split_dims_mapping

    if var_param.repeat_prefix:
      raise ValueError(
          'ShardedAdafactor: repeat_prefix is not empty. Consider using '
          'get_transformations_with_vectorized_repeat_prefix to vectorize '
          'prefix dimensions.')

    if tensor_split_dims_mapping is not None:
      assert len(tensor_split_dims_mapping) == len(shape)
      sharding_specified = True
    else:
      sharding_specified = False

    if self._beta1:
      if jnp.issubdtype(self._quantized_dtype, jnp.floating):
        # jnp.bfloat16 or jnp.float32
        output_m = WeightHParams(
            shape=shape,
            init=None,
            dtype=self._quantized_dtype,
            collections=None,
            mesh_shape=var_param.mesh_shape,
            tensor_split_dims_mapping=tensor_split_dims_mapping)
      elif self.should_store_momentum_in_qint(shape):
        if not jnp.issubdtype(self._quantized_dtype, jnp.integer):
          raise ValueError('jnp.integer quantized_dtype expected, got %s' %
                           self._quantized_dtype)
        output_m = WeightHParams(
            shape=shape,
            init=None,
            dtype=self._quantized_dtype,
            collections=None,
            mesh_shape=var_param.mesh_shape,
            tensor_split_dims_mapping=tensor_split_dims_mapping)
        scale_shape = shape[1:]
        m_scale_split_dims_mapping = tensor_split_dims_mapping
        # TODO(shafey): Fix logic for updating sharding annotations.
        if sharding_specified:
          m_scale_split_dims_mapping = gshard_utils.remove_dim(
              0, tensor_split_dims_mapping)
        output_m_scale = WeightHParams(
            shape=scale_shape,
            init=None,
            dtype=jnp.float32,
            collections=None,
            mesh_shape=var_param.mesh_shape,
            tensor_split_dims_mapping=m_scale_split_dims_mapping)
      else:
        output_m = WeightHParams(
            shape=shape,
            init=None,
            dtype=jnp.float32,
            collections=None,
            mesh_shape=var_param.mesh_shape,
            tensor_split_dims_mapping=tensor_split_dims_mapping)
    if self.should_use_factored_second_moment_estimate(shape):
      factored_dims = self.factored_second_moment_dims(shape)
      vr_axis, vc_axis = factored_dims  # pytype: disable=attribute-error
      # TODO(shafey): Fix logic for updating sharding annotations.
      if sharding_specified:
        vr_split_dims_mapping = gshard_utils.remove_dim(
            vr_axis, tensor_split_dims_mapping)
        vc_split_dims_mapping = gshard_utils.remove_dim(
            vc_axis, tensor_split_dims_mapping)
      else:
        vr_split_dims_mapping = tensor_split_dims_mapping
        vc_split_dims_mapping = tensor_split_dims_mapping
      output_vr_shape = list(shape).copy()
      del output_vr_shape[vr_axis]
      output_vr = WeightHParams(
          output_vr_shape,
          init=None,
          dtype=jnp.float32,
          collections=None,
          mesh_shape=var_param.mesh_shape,
          tensor_split_dims_mapping=vr_split_dims_mapping)
      output_vc_shape = list(shape).copy()
      del output_vc_shape[vc_axis]
      output_vc = WeightHParams(
          output_vc_shape,
          init=None,
          dtype=jnp.float32,
          collections=None,
          mesh_shape=var_param.mesh_shape,
          tensor_split_dims_mapping=vc_split_dims_mapping)
    else:
      output_v = WeightHParams(
          shape=shape,
          init=None,
          dtype=var_param.dtype,
          collections=None,
          mesh_shape=var_param.mesh_shape,
          tensor_split_dims_mapping=tensor_split_dims_mapping)
    return _ShardedAdafactorUpdateResult(
        update=output_update,
        m=output_m,
        m_scale=output_m_scale,
        vr=output_vr,
        vc=output_vc,
        v=output_v)

  def inf_to_nan(self, array):
    """Converting Infinity values to the more sticky NaN."""
    # For example, when we have y = 1.0 / x in code and x == inf, y will become
    # 0. Therefore the infinite value of x is hidden in the calculation,
    # leading to silent omission of numerical issues.
    if not self._maybe_inf_to_nan:
      return array
    return jnp.nan_to_num(array, nan=jnp.nan, posinf=jnp.nan, neginf=jnp.nan)

  def parameter_scale(self, var):
    """Estimate the scale of the parameters from the current values.

    We include a minimum value of 0.001 to give it a chance to escape 0
    if it was zero-initialized.

    Instead of using the value, we could impute the scale from the shape,
    as initializers do.

    Args:
      var: a variable or Tensor.

    Returns:
      a Scalar
    """
    return jnp.maximum(reduce_rms(var), jnp.asarray(self._epsilon2, var.dtype))

  def compute_subtrahend(self, x, shape, update_scale, m, m_scale):
    """Computes the subtrahend for a single variable."""
    output_m = jnp.zeros((1,))
    output_m_scale = jnp.zeros((1,))

    subtrahend = update_scale * x
    if self._beta1:
      m_init_dtype = m.dtype
      if jnp.issubdtype(self._quantized_dtype, jnp.floating):
        m = m.astype(jnp.float32)
      elif self.should_store_momentum_in_qint(shape):
        m = to_float(m, m_scale)
      subtrahend = self._beta1 * m + (1.0 - self._beta1) * subtrahend
      subtrahend = self.inf_to_nan(subtrahend)
      if self._quantized_dtype == jnp.bfloat16:
        new_m = subtrahend.astype(jnp.bfloat16)
        output_m = new_m
      elif self.should_store_momentum_in_qint(shape):
        # Update the momentum values.
        new_m_val, new_m_scale = to_quantized(subtrahend, m_init_dtype)
        output_m = new_m_val
        output_m_scale = new_m_scale
      else:
        output_m = subtrahend

    return subtrahend, output_m, output_m_scale

  def compute_var_and_slot_update(
      self, count, grad, m, m_scale, vr, vc, v, param, var_name
  ):
    """Computes the var and optimizer slots updates for a single variable."""
    # We can probably skip this step
    grad = grad.astype(jnp.float32)
    grad = self.inf_to_nan(grad)
    grad_squared = jnp.square(grad)

    prefix_name = '' if not self._name else f'{self._name}/'

    if self._per_var_learning_summary:
      grad_num_elements = jnp.array(grad.size, dtype=grad.dtype)
      num_zeros_in_grad = jnp.sum(grad_squared < 0.01 * self._epsilon1) * 1.0
      fraction_zero_grad = num_zeros_in_grad / grad_num_elements
      base_layer.add_global_summary(
          f'{prefix_name}fraction_zero_grad/{var_name}', fraction_zero_grad
      )

    # Add epsilon1_grad_sq_reg as per Algorithm 4
    # of https://arxiv.org/pdf/1804.04235.pdf
    grad_squared += self._epsilon1
    grad_squared_mean = self.inf_to_nan(reduce_mean(grad_squared))
    if self._decay_method == 'adam':
      assert self._decay_adam > 0
      decay_rate = adafactor_decay_rate_adam(self._decay_adam, count)
    elif self._decay_method == 'pow':
      assert self._decay_pow > 0
      decay_rate = adafactor_decay_rate_pow(self._decay_pow, count)
    else:
      raise ValueError(f'decay_method {self._decay_method} not supported.')

    learning_rate = self._learning_rate_fn(count)

    update_scale = learning_rate
    old_val = param

    if self._multiply_by_parameter_scale:
      update_scale *= self.parameter_scale(old_val).astype(update_scale.dtype)  # pytype: disable=attribute-error  # numpy-scalars
      if self._per_var_learning_summary:
        # Add summary for this var.
        base_layer.add_global_summary(
            f'{prefix_name}sharded_adafactor_parameter_scale/{var_name}',
            self.parameter_scale(old_val).astype(update_scale.dtype),
        )

    # Q(yonghui): Can we remove the hack now?
    # HACK: Make things dependent on grad.
    # This confounds the XLA rewriter and keeps it from fusing computations
    # across different variables.  This fusion is a bad for HBM usage, since
    # it causes the gradients to persist in memory.
    decay_rate += grad_squared_mean * 1e-30
    update_scale += grad_squared_mean * 1e-30
    # END HACK

    if self._per_var_learning_summary:
      base_layer.add_global_summary(
          f'{prefix_name}sharded_adafactor_decay_rate/{var_name}', decay_rate
      )
      base_layer.add_global_summary(
          f'{prefix_name}sharded_adafactor_update_scale/{var_name}',
          update_scale,
      )

    mixing_rate = 1.0 - decay_rate
    shape = param.shape

    output_vr = jnp.zeros((1,))
    output_vc = jnp.zeros((1,))
    output_v = jnp.zeros((1,))

    factored_second_moment_dims = self.factored_second_moment_dims(shape)
    if factored_second_moment_dims is not None:
      # Q(shafey): Should we use the more numerically stable version
      # reduce_mean().
      vr_axis, vc_axis = factored_second_moment_dims
      grad_squared_row_mean = self.inf_to_nan(
          jnp.mean(grad_squared, axis=vr_axis)
      )
      grad_squared_col_mean = self.inf_to_nan(
          jnp.mean(grad_squared, axis=vc_axis)
      )
      new_vr = decay_rate * vr + mixing_rate * grad_squared_row_mean
      new_vc = decay_rate * vc + mixing_rate * grad_squared_col_mean
      output_vr = new_vr
      output_vc = new_vc
      long_term_mean = jnp.mean(new_vr, axis=-1, keepdims=True)
      r_factor = 1.0 / jnp.sqrt(new_vr / long_term_mean)
      c_factor = 1.0 / jnp.sqrt(new_vc)
      x = (
          grad
          * jnp.expand_dims(r_factor, vr_axis)
          * jnp.expand_dims(c_factor, vc_axis)
      )
    else:
      # v with sharding annotation.
      new_v = decay_rate * v + mixing_rate * grad_squared
      output_v = new_v
      x = grad / jnp.sqrt(new_v)

    if self._per_var_learning_summary:
      # Add summary for this var.
      x_l2_scale = jnp.sqrt(reduce_mean(x * x))
      base_layer.add_global_summary(
          f'{prefix_name}sharded_adafactor_learning/{var_name}', x_l2_scale
      )

    if self._clip_threshold is not None:
      clipping_denom = jnp.maximum(1.0, reduce_rms(x) / self._clip_threshold)
      clipping_denom = self.inf_to_nan(clipping_denom)
      x /= clipping_denom
      if self._per_var_learning_summary:
        # Add summary for this var.
        base_layer.add_global_summary(
            f'{prefix_name}sharded_adafactor_clipping_denom/{var_name}',
            clipping_denom,
        )

    if self._per_var_learning_summary:
      # Add summary for this var.
      x_l2_scale_after_clipping = jnp.sqrt(reduce_mean(x * x))
      base_layer.add_global_summary(
          f'{prefix_name}sharded_adafactor_learning_after_clipping/{var_name}',
          x_l2_scale_after_clipping,
      )

    subtrahend, output_m, output_m_scale = self.compute_subtrahend(
        x, shape, update_scale, m, m_scale
    )

    if self._weight_decay is not None:
      # Apply decoupled weight decay to be consistent with AdamW.
      var_weight_decay = None
      if isinstance(self._weight_decay, dict):
        for scope_pattern in self._weight_decay.keys():
          regex_pattern = re.compile(scope_pattern)
          if regex_pattern.match(var_name):
            var_weight_decay = self._weight_decay[scope_pattern]
      else:
        var_weight_decay = self._weight_decay

      if var_weight_decay is not None:
        weight_decay = var_weight_decay * learning_rate
        subtrahend += weight_decay * old_val

    if self._layerwise_adaptation:
      include = True
      if self._exclude_from_layerwise_adaptation is not None:
        for scope_pattern in self._exclude_from_layerwise_adaptation:
          regex_pattern = re.compile(scope_pattern)
          if regex_pattern.match(var_name):
            include = False
            break
      if include:
        w_norm = reduce_rms(old_val)
        g_norm = reduce_rms(subtrahend / update_scale) + self._epsilon1
        ratio = jnp.where(
            jnp.greater(w_norm, 0),
            jnp.where(jnp.greater(g_norm, 0), (w_norm / g_norm), 1.0),
            1.0,
        )
        subtrahend *= ratio

    return _ShardedAdafactorUpdateResult(
        update=-subtrahend,
        m=output_m,
        m_scale=output_m_scale,
        vr=output_vr,
        vc=output_vc,
        v=output_v,
    )


def sharded_adafactor(
    learning_rate_fn: optax.Schedule,
    sharded_adafactor_helper_fn: Any = _ShardedAdafactorHelper,
    **kwargs,
) -> ShardedGradientTransformation:
  """Sharded AdaFactor optimizer."""
  sharded_adafactor_helper = sharded_adafactor_helper_fn(
      learning_rate_fn=learning_rate_fn, **kwargs
  )

  def init_fn(params):
    """Initializes the optimizer's state."""
    return sharded_adafactor_helper.to_state(
        jnp.zeros([], jnp.int32),
        jax.tree.map(sharded_adafactor_helper.init, params),
    )

  def init_partition_spec_fn(
      var_hparams: NestedWeightHParams,
  ) -> NestedWeightHParams:
    count = WeightHParams(
        shape=[],
        init=None,
        dtype=jnp.int32,
        collections=None,
        tensor_split_dims_mapping=[],
    )
    return sharded_adafactor_helper.to_state(
        count,
        jax.tree.map(sharded_adafactor_helper.init_partition_spec, var_hparams),
    )

  def update_fn(updates, state, params=None):
    if params is None:
      raise ValueError(
          'You are using a transformation that requires the current value of '
          'parameters, but you are not passing `params` when calling `update`.'
      )

    compute_var_and_slot_update_fn = functools.partial(
        sharded_adafactor_helper.compute_var_and_slot_update, state.count
    )
    var_names = py_utils.extract_prefixed_keys_from_nested_map(updates)
    output = jax.tree.map(
        compute_var_and_slot_update_fn,
        updates,
        state.m,
        state.m_scale,
        state.vr,
        state.vc,
        state.v,
        params,
        var_names,
    )
    updates = jax.tree.map(lambda o: o.update, output)
    count_plus_one = state.count + jnp.array(1, jnp.int32)
    updated_states = sharded_adafactor_helper.to_state(count_plus_one, output)
    return updates, updated_states

  return ShardedGradientTransformation(
      init=init_fn, update=update_fn, init_partition_spec=init_partition_spec_fn
  )


class ShardedAdafactor(BaseOptimizer):
  """AdaFactor optimizer that supports SPMD sharding.

  Reference:
    Shazeer et al, 2018: https://arxiv.org/abs/1804.04235

  Adafactor is very similar to Adam (Kingma and Ba, 2019), the major
  differences being:

  1. For a two-dimensional AxB weight matrix, Adafactor uses only A+B auxiliary
     parameters to maintain the second-moment estimator, instead of AB.
     This is advantageous on memory-limited systems.  In addition, beta1
     (momentum) is set to zero by default, saving an additional auxiliary
     parameter per weight.  Variables with >=3 dimensions are treated as
     collections of two-dimensional matrices - factorization is over the final
     two dimensions.

  2. Adafactor incorporates "update-clipping" - a scale-invariant analog of
     gradient clipping.  This improves stability.

  3. Adafactor does not require an external "learning rate".  By default, it
     incorporates a relative-update-scale schedule, corresponding to
     inverse-square-root learning-rate-decay in Adam.  We hope this works well
     for most applications.

  Attributes:
    learning_rate_fn: a callable that given the current training step, returns
      the learning rate to apply.
    weight_decay: an optional float tensor as decoupled weight decay value, or a
      dictionary with key as regex scope pattern and value as corresponding
      weight decay float tensor. The value will apply to all variables under
      that scope name.
    layerwise_adaptation: a boolean, whether or not to use layer-wise adaptive
      moments (LAMB) - https://arxiv.org/abs/1904.00962.
    decay_method: a string, deciding how decay_rate should be computed.
      Permitted values are 'adam' and 'pow'.
    decay_adam: a float, decay if decay_method == 'adam'.
    decay_pow: a float, decay if decay_method == 'pow'.
    beta1: a float value between 0 and 1 for momentum.
    clip_threshold: an optional float >= 1
    factored: a boolean, whether or not to use factored second order momentum.
    epsilon1_grad_sq_reg: Regularization constant for squared gradient.
    quantized_dtype: type of the quantized input. Allowed options are jnp.int8,
      jnp.int16, jnp.bfloat16 and jnp.float32. If floating-point type is
      specified, accumulators are stored as such type, instead of quantized
      integers.
    respect_skip_lp_regularization: whether or not to respect lingvo
      SKIP_LP_REGULARIZATION var collection that skips decoupled weight decay.
    exclude_from_layerwise_adaptation: A dictionary with key as regex scope
      pattern for variables to be skipped.
    per_var_learning_summary: a bool, whether or not to export per-var learning
      summaries.
    sort_factored_second_moment_dims: a bool, whether to select dims to factor
      by size, for the factored second moment.
    min_dim_size_to_factor: an integer, only factor the statistics if two array
      dimensions have at least this size. NOTE: min_dim_size_to_factor is only
      used when sort_factored_second_moment_dims=True.
    multiply_by_parameter_scale: a boolean, if True, then scale learning_rate by
      parameter scale. if False provided learning_rate is absolute step size.
      Set to False by default.
    epsilon2_param_scale_reg: Regularization constant for parameter scale. Only
      used when multiply_by_parameter_scale is True.
    maybe_inf_to_nan: Will use jax.nan_to_num during update when True.
  """

  weight_decay: float | dict[str, float] | None = None
  layerwise_adaptation: bool = False
  exclude_from_layerwise_adaptation: list[str] | None = None
  decay_method: Literal['adam', 'pow', ''] = ''
  decay_adam: float = 0.0
  decay_pow: float = 0.0
  beta1: float = 0.0
  clip_threshold: float | None = 1.0
  factored: bool = True
  epsilon1_grad_sq_reg: float = 1e-30
  quantized_dtype: str = 'int8'
  respect_skip_lp_regularization: bool = False
  per_var_learning_summary: bool = False
  sort_factored_second_moment_dims: bool = False
  min_dim_size_to_factor: int = 128
  multiply_by_parameter_scale: bool = False
  epsilon2_param_scale_reg: float = 1e-3
  maybe_inf_to_nan: bool = True
  name: str = ''

  @classmethod
  def HParamsAdamB(cls) -> pax_fiddle.Config[ShardedAdafactor]:  # pylint: disable=invalid-name
    """Convenient method for another commonly used Adam config."""
    return pax_fiddle.Config(
        cls,
        beta1=0.9,
        decay_method='adam',
        decay_adam=0.98,
        quantized_dtype='int8')

  def _get_sharded_adafactor_kwargs(self, lr: optax.Schedule) -> dict[str, Any]:
    if self.weight_decay:
      logging.warning(_WEIGHT_DECAY_DEPRECATION)
      # TODO(bf-jax):  skip regularization.
      assert not self.respect_skip_lp_regularization
      assert self.decay_adam >= 0
      assert self.decay_pow >= 0
      assert lr is not None
      assert self.decay_method == 'adam' or self.decay_method == 'pow', (
          f'decay_method: {self.decay_method} not supported. Supported methods '
          'are "pow", or "adam".'
      )

    return dict(
        learning_rate_fn=lr,
        weight_decay=self.weight_decay,
        layerwise_adaptation=self.layerwise_adaptation,
        decay_method=self.decay_method,
        decay_adam=self.decay_adam,
        decay_pow=self.decay_pow,
        beta1=self.beta1,
        clip_threshold=self.clip_threshold,
        factored=self.factored,
        epsilon1_grad_sq_reg=self.epsilon1_grad_sq_reg,
        quantized_dtype=getattr(jnp, self.quantized_dtype),
        respect_skip_lp_regularization=self.respect_skip_lp_regularization,
        exclude_from_layerwise_adaptation=self.exclude_from_layerwise_adaptation,
        per_var_learning_summary=self.per_var_learning_summary,
        sort_factored_second_moment_dims=self.sort_factored_second_moment_dims,
        min_dim_size_to_factor=self.min_dim_size_to_factor,
        multiply_by_parameter_scale=self.multiply_by_parameter_scale,
        epsilon2_param_scale_reg=self.epsilon2_param_scale_reg,
        maybe_inf_to_nan=self.maybe_inf_to_nan,
        name=self.name,
    )

  def _get_raw_grad_transformation(
      self, lr: optax.Schedule
  ) -> ShardedGradientTransformation:
    return sharded_adafactor(**self._get_sharded_adafactor_kwargs(lr))


def sharded_static_accumulation(
    num_sub_batches: int,
    base_tx: ShardedGradientTransformation,
    accumulation_use_cond_op: bool = False,
) -> ShardedGradientTransformation:
  """Gradient transformation for ShardedStaticAccumulator optimizer."""

  def init_fn(mdl_vars: NestedJTensor):
    base_state = base_tx.init(mdl_vars)
    # Make sure we accumulate in f32.
    accumulated_update = jax.tree.map(
        lambda v: jnp.zeros_like(v, dtype=jnp.float32), mdl_vars)
    return NestedMap(
        base_state=base_state,
        accumulated_update=accumulated_update,
        count=jnp.zeros((), dtype=jnp.int32))

  def init_partition_spec(params):

    def _weight_hparams(param):
      return WeightHParams(
          shape=param.shape,
          init=None,
          dtype=param.dtype,
          collections=None,
          mesh_shape=param.mesh_shape,
          tensor_split_dims_mapping=param.tensor_split_dims_mapping)

    accumulated_update = jax.tree.map(_weight_hparams, params)
    params_flattened, _ = jax.tree_util.tree_flatten(params)
    first_param = params_flattened[0]
    assert isinstance(first_param, WeightHParams)
    mesh_shape = first_param.mesh_shape
    count = WeightHParams(
        shape=[],
        init=None,
        dtype=jnp.int32,
        collections=None,
        mesh_shape=mesh_shape,
        tensor_split_dims_mapping=[])
    return NestedMap(
        base_state=base_tx.init_partition_spec(params),
        accumulated_update=accumulated_update,
        count=count)

  def update_fn(
      updates: NestedJTensor,
      state: NestedJTensor,
      params: NestedJTensor | None = None,
  ):
    new_accumulated_update = jax.tree.map(lambda acc, x: acc + x,
                                          state.accumulated_update, updates)  # pytype: disable=attribute-error  # jax-ndarray

    new_count = state.count + 1  # pytype: disable=attribute-error  # jax-ndarray
    should_emit = new_count >= num_sub_batches
    new_count = lax.cond(should_emit, lambda: jnp.array(0, dtype=jnp.int32),
                         lambda: new_count)

    def _run_base_tx():
      averaged_updated = jax.tree.map(lambda acc: acc / num_sub_batches,
                                      new_accumulated_update)
      emission_updates, emission_base_state = base_tx.update(
          averaged_updated, state.base_state, params)  # pytype: disable=attribute-error  # jax-ndarray
      return (emission_updates,
              jax.tree.map(lambda u: jnp.zeros_like(u, dtype=jnp.float32),
                           updates), emission_base_state)

    if accumulation_use_cond_op:

      def _continue_accumulating():
        return (
            jax.tree.map(jnp.zeros_like, updates),
            new_accumulated_update,
            state.base_state,  # pytype: disable=attribute-error # jax-ndarray
        )

      new_updates, new_accumulated_update, new_base_state = lax.cond(
          should_emit, _run_base_tx, _continue_accumulating
      )
    else:
      # PAX makes use of vmap for repeated layers. XLA currently doesn't
      # handle conds within vmap well and thus calls into both branches with a
      # select. Here we rewrite lax.cond as while_loop to get around this issue
      # and get faster step time.
      def while_cond(predicate, compute_fn, init_state, *args, **kwargs):
        """Rewrites a cond as a while loop."""

        def _iter_body(unused_state):
          results = compute_fn(*args, **kwargs)
          return tuple([False] + list(results))

        def _iter_condition(state):
          return state[0]

        results = jax.lax.while_loop(
            _iter_condition, _iter_body, tuple([predicate] + init_state)
        )
        return tuple(results[1:])

      new_updates, new_accumulated_update, new_base_state = while_cond(
          should_emit,
          _run_base_tx,
          [
              jax.tree.map(jnp.zeros_like, updates),
              new_accumulated_update,
              state.base_state,  # pytype: disable=attribute-error  # jax-ndarray
          ],
      )

    return new_updates, NestedMap(
        base_state=new_base_state,
        accumulated_update=new_accumulated_update,
        count=new_count)

  return ShardedGradientTransformation(
      init=init_fn,
      update=update_fn,
      init_partition_spec=init_partition_spec,
  )


class ShardedStaticAccumulator(BaseOptimizer):
  """Optimizer wrapper that accumulates a fixed number of sharded updates.

  Note that to enable gradient clipping, the clipping value must be set
  explicitly on ShardedStaticAccumulator and not the wrapped optimizer.

  Note that gradient clipping happens on the sub batches used for each
  accumulating step, which is different than clipping on the accumulated update,
  which is what some users may expect.

  Note we accumulate the gradients in whatever dtype they are, and then call the
  base optimizer transformation using the mean of the updates.

  When used with ShardedAdafactor turn off per_var_learning_summary since
  accessing global summary within lax.while_loop won't work. Other summaries
  should work ok.

  Attributes:
    optimizer_tpl: Parameter for base optimizer.
    num_sub_batches: The number of batches whose updates should be accumulated
      before sending to the base optimizer transformation.
    accumulation_use_cond_op: whether to use lax.while_loop (default) or
      lax.cond operator for checking if current step needs to gradient
      accumulation, using lax.cond may have memory benefits, but saw some cases
      where it can have perf regression.
  """
  optimizer_tpl: pax_fiddle.Config[BaseOptimizer] | None = None
  num_sub_batches: int = 1
  base_optimizer: Any = dataclasses.field(init=False, repr=False)
  accumulation_use_cond_op: bool = False

  def __post_init__(self):
    super().__post_init__()
    assert self.optimizer_tpl is not None
    if self.num_sub_batches < 1:
      raise ValueError('Set `p.num_sub_batches >= 1`.')

    base_opt_tpl = self.optimizer_tpl.clone()
    if base_opt_tpl.lr_schedule is None:
      base_opt_tpl.lr_schedule = self.lr_schedule
    self.base_optimizer = instantiate(base_opt_tpl)

  def _get_raw_grad_transformation(
      self, lr: optax.Schedule) -> GeneralGradientTransformation:
    base_tx = self.base_optimizer._get_raw_grad_transformation(lr)  # pylint: disable=protected-access
    return sharded_static_accumulation(
        self.num_sub_batches, base_tx, self.accumulation_use_cond_op
    )
