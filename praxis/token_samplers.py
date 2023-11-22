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

"""Algorithms for the token samplers.

This file contains token samplers with temperature including random, top_k and
top_p.
"""

import abc
from typing import Any, Sequence

import jax
from jax import numpy as jnp
from praxis import base_hyperparams
from praxis import base_layer
from praxis import decoder_utils
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes

NestedMap = py_utils.NestedMap
JTensor = base_layer.JTensor

DUMMY_PRNG_KEY = decoder_utils.DUMMY_PRNG_KEY


def _batch_rngs_random_gumbel(rngs: JTensor, shape: Sequence[int]) -> JTensor:
  """Samples gumbel random values with one or batch of PRNGKeys.

  Args:
    rngs: JTensor of int32, a single PRNGKey with shape of (2,) or a batch of
      PRNGKeys with shape of (batch_size, 2), where batch_size is the first
      dimension of `shape`.
    shape: Sequence of ints, the shape of output tensor.

  Returns:
    A tensor of sampled gumbel random values.
  """
  if rngs.ndim == 1:
    return jax.random.gumbel(rngs, shape)

  batch_size, *remaining_shape = shape

  def _impl(rng):
    return jax.random.gumbel(rng, remaining_shape)

  if rngs.ndim != 2 or rngs.shape[0] != batch_size:
    raise ValueError(
        'When sample for a batch of PRNGKeys, rngs must be 2d '
        f'with batch_size as its first dim, get {rngs.shape} '
        f'instead when batch_size is {batch_size}.'
    )

  return jax.vmap(_impl)(rngs)


def _get_argmax_ids(top_k_argmax_ids: JTensor, top_k_items: JTensor) -> JTensor:
  """Gets the original top_k items with top_k_argmax_ids.

  Args:
    top_k_argmax_ids: JTensor of shape [batch_size], the argmax among
      top_k_logits.
    top_k_items: JTensor of shape [batch_size, top_k], the top_k items.

  Returns:
    A tensor of shape [batch_size].
  """

  selectors = [
      jax.lax.broadcasted_iota(
          jnp.int32, shape=top_k_argmax_ids.shape, dimension=d
      )
      for d in range(len(top_k_argmax_ids.shape))
  ]
  selectors.append(top_k_argmax_ids)
  return top_k_items[tuple(selectors)]


def get_top_k(
    logits: JTensor,
    top_k: int,
    per_example_top_k: JTensor | None,
    top_k_recall_target: float = 1.0,
) -> Sequence[JTensor]:
  """Gets top k logits and indices from given top K.

  Args:
    logits: Logits of current step. This is a JTensor of [batch_size *
      num_samples, vocab_size].
    top_k: If non zero, only selecting among the most likely k tokens at each
      step. top_k is set to the maximum k value for sampling decode.
    per_example_top_k: Optional per example top_k of shape [batch_size *
      num_samples]. The value of per_example_top_k should be smaller or equal to
      `top_k` and larger than 0.
    top_k_recall_target: if less than 1.0, use TPU optimized approx_top_k with
      specified recall target for the top_k sampling. See
      https://arxiv.org/abs/2206.14286 for more details.

  Returns:
    A tuple of top_k_logits of shape [batch_size * num_samples, top_k] and
      top_k_indices of shape [batch_size * num_samples, top_k].
  """
  if not top_k:
    # Select all indices.
    indices = jnp.arange(logits.shape[-1])
    indices = jnp.reshape(indices, (1,) * (logits.ndim - 1) + (len(indices),))
    indices = jnp.tile(indices, logits.shape[:-1] + (1,))
    return logits, indices
  # TopK of shape [batch_size * num_samples, top_k]
  if top_k_recall_target < 1.0:
    assert top_k_recall_target > 0.0
    top_k_logits, top_k_indices = jax.lax.approx_max_k(
        logits, top_k, recall_target=top_k_recall_target
    )
  else:
    top_k_logits, top_k_indices = jax.lax.top_k(logits, top_k)

  if per_example_top_k is not None:
    per_example_top_k = jnp.minimum(per_example_top_k.astype(jnp.int32), top_k)
    iota = jnp.tile(
        jax.lax.iota(dtype=jnp.int32, size=(top_k)), [top_k_logits.shape[0], 1]
    )
    top_k_mask = iota < per_example_top_k[:, jnp.newaxis]
    top_k_logits = jnp.where(
        top_k_mask,
        top_k_logits,
        py_utils.get_large_negative_number(top_k_logits.dtype),
    )
  return top_k_logits, top_k_indices


def top_p_mask_logits(
    logits: JTensor,
    p: float | JTensor,
    logits_sorted_in_descending_order: bool = False,
    logits_sum: JTensor | None = None,
) -> JTensor:
  """Keep only logits in the top `p` percentile of the softmax distribution.

  All other logits are masked with a large negative number. Typical values of p
  are 0.95 or 0.99. Also known as nucleus sampling. Note that logits may be
  passed in descending order with the `logits_sorted_in_descending_order`

  Args:
    logits: logits of shape [B, T].
    p: A scalar or a JTensor of shape [B]. In practice this means selecting the
      highest probability tokens whose cumulative probability mass exceeds this
      pre-chosen threshold p.
    logits_sorted_in_descending_order: whether or not the logits is sorted in
      descending order.
    logits_sum: If none, apply softmax over logits to get probabilities.

  Returns:
    The masked logits.
  """
  if not isinstance(p, JTensor) and p >= 1.0:
    return logits
  batch_size = logits.shape[0]
  if logits_sorted_in_descending_order:
    logits_sorted = logits
  else:
    # Ascending order. Cumsum will go through small numbers first, which is
    # more numerically stable.
    logits_sorted = jnp.sort(logits, axis=-1)
  if logits_sum is not None:
    probs = logits_sorted.astype(jnp.float32) / logits_sum
  else:
    probs = jax.nn.softmax(logits_sorted.astype(jnp.float32), axis=-1)
  sorted_cum_probs = jnp.cumsum(probs, axis=-1)
  cutoff_threshold = p if logits_sorted_in_descending_order else 1.0 - p
  cutoff_idx = jnp.sum(
      (sorted_cum_probs <= cutoff_threshold).astype(jnp.int32), axis=-1
  )
  cutoff_logit = logits_sorted[jnp.arange(batch_size), cutoff_idx]
  logits = jnp.where(
      logits < jnp.expand_dims(cutoff_logit, -1),
      py_utils.get_large_negative_number(logits.dtype),
      logits,
  )
  return logits


class BaseLogitsSampler(
    base_hyperparams.FiddleBaseParameterizable, metaclass=abc.ABCMeta
):
  """Interface for temperature based sampling given logits.

  Final step used by a NextTokenSampler to apply temperature scaling and do
  argmax sampling given the logits.
  """

  @abc.abstractmethod
  def __call__(
      self,
      prng_key: JTensor,
      logits: JTensor,
      ids: JTensor | None,
      temperature: float | JTensor,
      decode_loop_state: NestedMap,
  ) -> tuple[JTensor, NestedMap]:
    """Samples the token ids given the logits.

    Args:
      prng_key: Random key for sampling.
      logits: Input logits.
      ids: Indices corresponding to top-k logits.
      temperature: Temperature of sampling decoding. It could be a float or a
        JTensor of shape [batch_size * num_samples].
      decode_loop_state: Decode loop state provides access to all the relevant
        sampling loop information.

    Returns:
      Tuple consisting of sampled indices and NestedMap of state variables to
      be updated. These state variables are specific to the state variables
      related to sampler; only used when sampling of tokens itself needs state
      to be maintained.
    """


class DefaultCategoricalLogitsSampler(BaseLogitsSampler):
  """Default implementation for temperature based sampling given logits."""

  def __call__(
      self,
      prng_key: JTensor,
      logits: JTensor,
      ids: JTensor | None,
      temperature: float | JTensor,
      decode_loop_state: NestedMap,
  ) -> tuple[JTensor, NestedMap]:
    """Apply temperature scaling and sample via the gumbel-max trick.

    Args:
      prng_key: Random key for sampling.
      logits: Input logits.
      ids: Indices corresponding to top-k logits.
      temperature: Temperature of sampling decoding. It could be a float or a
        JTensor of shape [batch_size * num_samples].
      decode_loop_state: Decode loop state provides access to all the relevant
        sampling loop information.

    Returns:
      Tuple consisting of sampled indices and empty NestedMap for any updated
      state variables. This sampler does not use and update any state variables.
    """
    del ids, decode_loop_state  # unused
    if temperature is None:
      temperature = 0.0
    gumbel_noise = _batch_rngs_random_gumbel(prng_key, logits.shape).astype(
        logits.dtype
    )
    logits += gumbel_noise * temperature
    return jnp.argmax(logits, axis=-1), NestedMap()


def _apply_top_p_given_top_k(
    top_k_logits: JTensor,
    top_k_indices: JTensor,
    top_p: float | JTensor | None = None,
    topk_is_sorted: bool = True,
    logits_sum: JTensor | None = None,
) -> tuple[JTensor, JTensor, JTensor]:
  """Get top_p logits given top_k logits and indices.

  Args:
    top_k_logits: Top k logits.
    top_k_indices: Indices corresponding to top-k logits.
    top_p: Optional cutoff probability. A scalar or a JTensor. Use the smallest
      number of logits whose cumulative sum of probs adds up to (at least)
      top_p. If it is a JTensor, it has shape [batch_size * num_samples, 1]
    topk_is_sorted: Whether topk logits are sorted.
    logits_sum: logits sum.

  Returns:
    A tuple of top_p_logits, top_k_logprobs, top_k_indices for sampling.
  """
  if top_p is None:
    top_p_logits = top_k_logits
  elif isinstance(top_p, JTensor):
    # Apply top_p to the mask.
    needs_top_p_mask = jnp.any(top_p < 1.0)

    def _true_fn():
      return top_p_mask_logits(
          top_k_logits,
          top_p,
          logits_sum=logits_sum,
          logits_sorted_in_descending_order=topk_is_sorted,
      )

    def _false_fn():
      return top_k_logits

    top_p_logits = jax.lax.cond(needs_top_p_mask, _true_fn, _false_fn)
  else:
    top_p_logits = top_p_mask_logits(
        top_k_logits,
        top_p,
        logits_sum=logits_sum,
        logits_sorted_in_descending_order=topk_is_sorted,
    )

  # Compute log probabilities from top_k logits
  top_k_logprobs = jax.nn.log_softmax(top_k_logits.astype(jnp.float32))
  return top_p_logits, top_k_logprobs, top_k_indices


def _apply_top_k_and_top_p(
    logits: JTensor,
    top_k: int,
    top_p: float | JTensor | None = None,
    per_example_top_k: JTensor | None = None,
    global_normalize: bool = False,
    top_k_recall_target: float = 1.0,
) -> tuple[JTensor, JTensor, JTensor]:
  """Get top_k and top_p logits.

  When both top_k and top_p are defined, top_k will be applied first.

  Args:
    logits: Logits of current step. This is a JTensor of [batch_size *
      num_samples, vocab_size].
    top_k: If nonzero, use top-k sampling, only selecting among the most likely
      k tokens at each step. top_k is set to the maximum k value for sampling
      decode.
    top_p: Optional cutoff probability. A scalar or a JTensor. Use the smallest
      number of logits whose cumulative sum of probs adds up to (at least)
      top_p. If it is a JTensor, it has shape [batch_size * num_samples, 1]
    per_example_top_k: Optional per example top_k of shape [batch_size *
      num_samples]. The value of per_example_top_k should be smaller or equal to
      `top_k` and larger than 0.
    global_normalize: Normalize the logits over top-k logits or globally in the
      whole vocabulary.
    top_k_recall_target: if less than 1.0, use TPU optimized approx_top_k with
      specified recall target for the top_k sampling. See
      https://arxiv.org/abs/2206.14286 for more details.

  Returns:
    A tuple of top_p_logits, top_k_logprobs, top_k_indices for sampling.
  """
  # TopK of shape [batch_size * num_samples, top_k]
  top_k_logits, top_k_indices = get_top_k(
      logits, top_k, per_example_top_k, top_k_recall_target
  )
  if global_normalize:
    logits_sum = jnp.sum(logits.astype(jnp.float32), axis=-1, keepdims=True)
  else:
    logits_sum = None
  return _apply_top_p_given_top_k(
      top_k_logits=top_k_logits,
      top_k_indices=top_k_indices,
      top_p=top_p,
      topk_is_sorted=True,
      logits_sum=logits_sum,
  )


# TODO(b/299978151): Consider removing or updating this public API.
def sample_from_top_p_given_top_k(
    top_k_logits: JTensor,
    top_k_indices: JTensor,
    prng_key: pytypes.PRNGKey,
    temperature: JTensor | float,
    top_p: float | JTensor | None = None,
    topk_is_sorted: bool = True,
    logits_sum: JTensor | None = None,
    logits_sampler: BaseLogitsSampler | None = None,
    decode_loop_state: NestedMap | None = None,
) -> Sequence[JTensor]:
  """Sample decode algorithm from TopP given output from TopK.

  Args:
    top_k_logits: Top k logits.
    top_k_indices: Indices corresponding to top-k logits.
    prng_key: The prng key.
    temperature: Temperature of sampling decoding. It could be a float or a
      JTensor of shape [batch_size * num_samples].
    top_p: See params of `sample_from_top_k_and_top_p`.
    topk_is_sorted: Whether topk logits are sorted.
    logits_sum: logits sum.
    logits_sampler: Callable to sample token from given logits along with
      rng_key, temperature and loop state information.
    decode_loop_state: Decode loop state provides access to all the relevant
      sampling loop information.

  Returns:
    A tuple of
      next_token_id of shape [batch_size * num_samples]
      logprobs_at_new_ids of shape [batch_size * num_samples] computed from
        the top_k_logits.
  """
  if not logits_sampler:
    logits_sampler = DefaultCategoricalLogitsSampler()
  top_p_logits, top_k_logprobs, top_k_indices = _apply_top_p_given_top_k(
      top_k_logits,
      top_k_indices,
      top_p,
      topk_is_sorted,
      logits_sum,
  )

  # Add gumbel noise.
  argmax_ids_in_topk, _ = logits_sampler(
      prng_key,
      top_p_logits,
      top_k_indices,
      temperature,
      decode_loop_state,
  )

  return (
      _get_argmax_ids(argmax_ids_in_topk, top_k_indices),
      _get_argmax_ids(argmax_ids_in_topk, top_k_logprobs),
  )


# TODO(b/299978151): Consider removing or updating this public API.
def sample_from_top_k_and_top_p(
    logits: JTensor,
    prng_key: pytypes.PRNGKey,
    temperature: JTensor | float,
    top_k: int,
    top_p: float | JTensor | None = None,
    per_example_top_k: JTensor | None = None,
    global_normalize: bool = False,
    top_k_recall_target: float = 1.0,
    logits_sampler: BaseLogitsSampler | None = None,
    decode_loop_state: NestedMap | None = None,
) -> Sequence[JTensor]:
  """Sample decode algorithm from TopK and TopP.

  When both top_k and top_p are defined, top_k will be applied first.

  Args:
    logits: Logits of current step. This is a JTensor of [batch_size *
      num_samples, vocab_size].
    prng_key: The prng key.
    temperature: Temperature of sampling decoding. It could be a float or a
      JTensor of shape [batch_size * num_samples].
    top_k: If nonzero, use top-k sampling, only selecting among the most likely
      k tokens at each step. top_k is set to the maximum k value for sampling
      decode.
    top_p: Optional cutoff probability. A scalar or a JTensor. Use the smallest
      number of logits whose cumulative sum of probs adds up to (at least)
      top_p. If it is a JTensor, it has shape [batch_size * num_samples, 1]
    per_example_top_k: Optional per example top_k of shape [batch_size *
      num_samples]. The value of per_example_top_k should be smaller or equal to
      `top_k` and larger than 0.
    global_normalize: Normalize the logits over top-k logits or globally in the
      whole vocabulary.
    top_k_recall_target: if less than 1.0, use TPU optimized approx_top_k with
      specified recall target for the top_k sampling. See
      https://arxiv.org/abs/2206.14286 for more details.
    logits_sampler: Callable to sample tokens given logits along with rng_key,
      temperature and loop state information.
    decode_loop_state: Decode loop state provides access to all the relevant
      sampling loop information.

  Returns:
    A tuple of
      next_token_id of shape [batch_size * num_samples]
      logprobs_at_new_ids of shape [batch_size * num_samples] computed from
        the top_k_logits.
  """
  if not logits_sampler:
    logits_sampler = DefaultCategoricalLogitsSampler()
  top_p_logits, top_k_logprobs, top_k_indices = _apply_top_k_and_top_p(
      logits=logits,
      top_k=top_k,
      top_p=top_p,
      per_example_top_k=per_example_top_k,
      global_normalize=global_normalize,
      top_k_recall_target=top_k_recall_target,
  )

  argmax_ids_in_topk, _ = logits_sampler(
      prng_key,
      top_p_logits,
      top_k_indices,
      temperature,
      decode_loop_state,
  )

  return (
      _get_argmax_ids(argmax_ids_in_topk, top_k_indices),
      _get_argmax_ids(argmax_ids_in_topk, top_k_logprobs),
  )


def epsilon_mask_logits(logits: JTensor, epsilon: float) -> JTensor:
  """Mask logits with absolute probability below epsilon.

  Args:
    logits: logits of shape [B, T].
    epsilon: a scalar.

  Returns:
    The masked logits.
  """
  if epsilon <= 0:
    return logits
  probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
  logits = jnp.where(
      probs < epsilon, py_utils.get_large_negative_number(logits.dtype), logits
  )
  # note: logits are no longer normalized after this, sum(exp(logits)) < 1
  return logits


class BaseNextTokenSampler(
    base_hyperparams.FiddleBaseParameterizable, metaclass=abc.ABCMeta
):

  @abc.abstractmethod
  def __call__(
      self,
      model: base_layer.BaseLayerApi,
      logits: JTensor,
      temperature: float | JTensor,
      decode_loop_state: NestedMap,
  ) -> NestedMap:
    """Samples the next token ids given the logits output.

    Args:
      model: The model object.
      logits: Logits at current step during decoding. This is a JTensor of [B,
        V] where B is the batch size and V is the vocab_size.
      temperature: Temperature of sampling decoding. It could be a float or a
        JTensor of shape [B].
      decode_loop_state: Decode loop state at current time step.

    Returns:
      A NestedMap containing 2 fields:
        new_ids: JTensor of shape [B] containing the selected next token for
          each sequence in the batch.
        logits: JTensor of shape [B, V] containing the (possibly modified)
          logits at the current step. This return value is used to ensure that
          the recorded logprobs per sequence takes into account the
          modifications made to the logits as part of the next token sampling
          logic.
        logprobs: [Optional] JTensor of shape [B] containing the log softmax
          of the selected token for each sequence in. This field allows a faster
          log probability computation within the sampler itself.
    """

  def init_decode_loop_state(
      self,
      decode_loop_state: NestedMap,
      model: base_layer.BaseLayerApi | None = None,
      batch_size: int | None = None,
      eos_id: int | Sequence[int] | JTensor | None = None,
  ) -> NestedMap:
    """Initialize any addition decode loop state."""
    return decode_loop_state

  def post_process_decode_loop_state(
      self, decode_loop_state: NestedMap
  ) -> NestedMap:
    """Delete unused decode loop state."""
    return decode_loop_state

  @classmethod
  def get_extra_kwargs(
      cls, inputs: NestedMap, num_samples: int
  ) -> dict[str, Any]:
    """Gets the supported extra arguments from model inputs."""
    return {}

  def _get_prng_key(
      self,
      model: base_layer.BaseLayerApi,
      prng_key: JTensor | None = None,
  ) -> JTensor:
    if prng_key is None:
      # Default method, use `next_prng_key` to generate noise.
      return model.next_prng_key()
    else:
      next_model_key = model.next_prng_key()
      batch_size = prng_key.shape[0]
      split_next_model_key = jax.random.split(next_model_key, batch_size)
      next_prng_key = jax.lax.cond(
          jnp.all(prng_key == DUMMY_PRNG_KEY),
          lambda: split_next_model_key,
          lambda: prng_key,
      )
      return next_prng_key


class DefaultNextTokenSampler(BaseNextTokenSampler):
  """The default sampling logic implementing top-K and top-P sampling.

  If all the values in gumbel_prng_key is set to DUMMY_PRNG_KEY, gumbel_prng_key
  will be ignored and model.next_prng_key() is used to generate random noise
  for sampling decode.

  Attributes:
    top_k: if nonzero, use top-k sampling, only selecting among the most likely
      k tokens at each step.
    top_p: if not None, use the smallest number of logits whose cumulative sum
      of probs adds up to (at least) p.
    epsilon_p: if positive, use epsilon sampling, only selecting among the
      tokens with probability at least epsilon at each step.
    global_normalize: if top_k and top_p are enabled together, this flag
      indicates whether we need to normalize the logits in top_k logits or
      globally in the whole vocabulary.
    top_k_recall_target: if less than 1.0, use TPU optimized approx_top_k with
      specified recall target for the top_k sampling. See
      https://arxiv.org/abs/2206.14286 for more details.
    use_top_k_for_logprobs: computes the log probability from the top k logits
      instead of all logits.
    logits_sampler: BaseLogitsSampler function used to sample ids given logits.
  """

  top_k: int = 40
  top_p: float | JTensor | None = None
  epsilon_p: float = 0.0
  global_normalize: bool = False
  top_k_recall_target: float = 1.0
  use_top_k_for_logprobs: bool = False
  logits_sampler: BaseLogitsSampler = pax_fiddle.instance_field(
      DefaultCategoricalLogitsSampler
  )

  def __call__(
      self,
      model: base_layer.BaseLayerApi,
      logits: JTensor,
      temperature: float | JTensor,
      decode_loop_state: NestedMap,
      per_example_top_p: JTensor | None = None,
      per_example_top_k: JTensor | None = None,
      gumbel_prng_key: JTensor | None = None,
  ) -> NestedMap:
    """The default sampling logic implementing top-K and top-P sampling."""
    assert self.top_k >= 0
    input_logits = logits

    if per_example_top_p is not None:
      top_p = per_example_top_p
    else:
      top_p = getattr(self, 'top_p', None)

    if self.epsilon_p > 0.0:
      logits = epsilon_mask_logits(logits, self.epsilon_p)

    # TODO(vbachani): Revisit, maybe combine top_k various cases.
    if self.top_k > 1:
      top_p_logits, top_k_logprobs, top_k_indices = _apply_top_k_and_top_p(
          logits=logits,
          top_k=self.top_k,
          top_p=top_p,
          per_example_top_k=per_example_top_k,
          global_normalize=self.global_normalize,
          top_k_recall_target=self.top_k_recall_target,
      )

      argmax_ids_in_topk, state_vars_to_update = self.logits_sampler(
          self._get_prng_key(model, gumbel_prng_key),
          top_p_logits,
          top_k_indices,
          temperature,
          decode_loop_state,
      )

      new_ids = _get_argmax_ids(argmax_ids_in_topk, top_k_indices)

      if self.use_top_k_for_logprobs:
        logprobs_at_new_ids = _get_argmax_ids(
            argmax_ids_in_topk, top_k_logprobs
        )
        return NestedMap(
            new_ids=new_ids,
            logits=input_logits,
            logprobs_at_new_ids=logprobs_at_new_ids,
            **state_vars_to_update,
        )
      else:
        return NestedMap(
            new_ids=new_ids, logits=input_logits, **state_vars_to_update
        )
    elif self.top_k == 0:
      if top_p is not None:
        logits = top_p_mask_logits(logits, top_p)
      if isinstance(temperature, JTensor) or temperature > 0.0:
        new_ids, state_vars_to_update = self.logits_sampler(
            self._get_prng_key(model, gumbel_prng_key),
            logits,
            None,
            temperature,
            decode_loop_state,
        )
      else:
        new_ids = jnp.argmax(logits, axis=1)
        state_vars_to_update = NestedMap()
      return NestedMap(
          new_ids=new_ids,
          logits=input_logits,
          **state_vars_to_update,
      )
    else:  #  k == 1
      new_ids = jnp.argmax(logits, axis=1)
    return NestedMap(new_ids=new_ids, logits=input_logits)
