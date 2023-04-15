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

"""Algorithms for the sample decode.

This file contains sample decode with temperature and greedy decode algorithms.
Greedy decode is a special case for sample decode.
"""

import abc
import functools
from typing import Dict, List, Optional, Sequence, Tuple, Union

from flax import linen as nn
import jax
from jax import numpy as jnp
from praxis import asserts
from praxis import base_hyperparams
from praxis import base_layer
from praxis import decoder_utils
from praxis import py_utils
from praxis import pytypes

NestedMap = py_utils.NestedMap
JTensor = base_layer.JTensor
StreamingResultCallback = decoder_utils.StreamingResultCallback

RANDOM = base_layer.RANDOM
PARAMS = base_layer.PARAMS
NON_TRAINABLE = base_layer.NON_TRAINABLE
AUX_LOSS = base_layer.AUX_LOSS
SUMMARIES = base_layer.SUMMARIES
DECODE_CACHE = base_layer.DECODE_CACHE
PREFIX_DECODE_CACHE = base_layer.PREFIX_DECODE_CACHE
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


def reorder_with_indices(x: JTensor, indices: JTensor):
  """Reorders with the given indices.

  Args:
   x: A JTensor with shape [batch, num_samples, ...].
   indices: A JTensor with shape [batch, num_samples].

  Returns:
    Result ordered at dimension 1 following the order defined by indices
  """
  assert len(indices.shape) == 2
  assert len(x.shape) >= 2
  batch_indices = jnp.repeat(
      jnp.expand_dims(jnp.arange(x.shape[0]), 1), indices.shape[1], axis=1
  )
  return x[batch_indices, indices]


def sort_samples_by_scores(result: NestedMap) -> NestedMap:
  """Sorts samples by scores.

  Args:
    result: A NestedMap with `.prefix_lengths` (indicating the lengths of
      prefixes for each target sequence), `.output_ids` (matrix of int ids with
      the decoded output), `.decode_lengths` (vector of ints indicating the
      lengths of non-padding tokens in `.output_ids`, which includes the
      prefix), and `.logprobs` (the log probability of selected tokens,
      including the prefix, where a positive value of 1.0 is used to indicate
      padded positions). The result has shape [batch, num_samples, ...].

  Returns:
    Result sorted at num_samples dimension by scores.
  """

  # Logprobs are set to 1.0 after the end of sequence(EOS) character.
  logprobs = jnp.where(
      jnp.greater_equal(result.logprobs, 1.0),
      jnp.zeros_like(result.logprobs),
      result.logprobs,
  )
  scores = jnp.sum(logprobs, -1)
  num_samples = scores.shape[-1]
  # Uses top_k to get indices.
  _, indices = jax.lax.approx_max_k(scores, num_samples, recall_target=0.85)
  return jax.tree_map(
      functools.partial(reorder_with_indices, indices=indices), result
  )


def split_batch_dim(
    x: jnp.ndarray, batch_dim: int, num_samples: int
) -> jnp.ndarray:
  """Split the tensor at batch dimension.

  Args:
    x: The input tensor of shape [batch, ...].
    batch_dim: batch dimension.
    num_samples: number of samples in sample decode.

  Returns:
    A tensor of shape [batch, num_samples, ...].
  """
  assert isinstance(x, jnp.ndarray)
  x_shape = list(x.shape)
  assert x_shape[batch_dim] % num_samples == 0
  x_shape[batch_dim] = x_shape[batch_dim] // num_samples
  x_shape.insert(batch_dim + 1, num_samples)
  return jnp.reshape(x, x_shape)


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
    per_example_top_k: JTensor,
    top_k_recall_target: float = 1.0,
) -> Sequence[JTensor]:
  """Gets top k logits and indices from given top K.

  Args:
    logits: Logits of current step. This is a JTensor of [batch_size *
      num_samples, vocab_size].
    top_k: Only selecting among the most likely k tokens at each step. top_k is
      set to the maximum k value for sampling decode.
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


def right_align_segment_position(lengths: JTensor, max_length: int) -> JTensor:
  """Get the right aligned segment position for the sequence.

  For example, if lengths = [4, 5, 6] and max_length = 6, it will return a
  tensor like:
  [[0, 0, 0, 1, 2, 3],
   [0, 0, 1, 2, 3, 4],
   [0, 1, 2, 3, 4, 5]]

  Args:
    lengths: Actual lengths of shape [batch_size].
    max_length: Max length.

  Returns:
    Segment position tensor for right aligned ids with shape
    [batch_size, max_length].
  """
  iota = jax.lax.iota(dtype=jnp.int32, size=(max_length))
  iota = iota - (max_length - jnp.expand_dims(lengths, 1))
  iota = jnp.where(iota < 0, jnp.zeros_like(iota), iota)
  return iota


def right_align_prefix_ids(
    prefix_ids: JTensor, prefix_lengths: JTensor, paddings_dtype: jnp.dtype
) -> JTensor:
  """Right align prefix ids.

  Args:
    prefix_ids: prefix ids [batch_size, prefix_len] with dtype jnp.int32.
    prefix_lengths: prefix lengths [batch_size] with dtype jnp.int32.
    paddings_dtype: the dtype for the generated prefix_paddings.

  Returns:
    Right align prefix_ids and prefix_paddings with shape [batch_size,
    prefix_len].
  """
  max_prefix_len = prefix_ids.shape[-1]

  right_align_ids = decoder_utils.right_align_tensors(
      prefix_ids, prefix_lengths
  )
  prefix_lengths = prefix_lengths[:, jnp.newaxis]
  prefix_iota = jax.lax.iota(dtype=jnp.int32, size=max_prefix_len)
  prefix_iota = prefix_iota - (max_prefix_len - prefix_lengths)
  prefix_paddings = jnp.where(
      prefix_iota < 0,
      jnp.ones_like(prefix_iota, dtype=paddings_dtype),
      jnp.zeros_like(prefix_iota, dtype=paddings_dtype),
  )
  return right_align_ids, prefix_paddings  # pytype: disable=bad-return-type  # jax-ndarray


def top_p_mask_logits(
    logits: JTensor,
    p: Union[float, JTensor],
    logits_is_sorted: bool = False,
    logits_sum: Optional[JTensor] = None,
) -> JTensor:
  """Adjust logits so that the smallest number of logits whose cumulative...

  sum of probs adds up to (at least) p. The other logits are masked with a
  large negative number.

  Args:
    logits: logits of shape [B, T].
    p: A scalar or a JTensor of shape [B]. In practice this means selecting the
      highest probability tokens whose cumulative probability mass exceeds this
      pre-chosen threshold p.
    logits_is_sorted: where or not the logits is sorted.
    logits_sum: If none, apply softmax over logits to get probabilities.

  Returns:
    The masked logits.
  """
  if not isinstance(p, JTensor) and p >= 1.0:
    return logits
  batch_size = logits.shape[0]
  # Ascending order. Cumsum will go through small numbers first, which is
  # more numerically stable.
  if logits_is_sorted:
    logits_sorted = logits
  else:
    logits_sorted = jnp.sort(logits, axis=-1)
  if logits_sum is not None:
    probs = logits_sorted.astype(jnp.float32) / logits_sum
  else:
    probs = jax.nn.softmax(logits_sorted.astype(jnp.float32), axis=-1)
  sorted_cum_probs = jnp.cumsum(probs, axis=-1)
  cutoff_idx = jnp.sum((sorted_cum_probs <= p).astype(jnp.int32), axis=-1)
  cutoff_logit = logits_sorted[jnp.arange(batch_size), cutoff_idx]
  logits = jnp.where(
      logits < jnp.expand_dims(cutoff_logit, -1),
      py_utils.get_large_negative_number(logits.dtype),
      logits,
  )
  return logits


def sample_from_top_p_given_top_k(  # pytype: disable=annotation-type-mismatch  # jax-ndarray
    top_k_logits: JTensor,
    top_k_indices: JTensor,
    prng_key: pytypes.PRNGKey,
    temperature: Union[JTensor, float],
    top_p: Optional[Union[float, JTensor]] = None,
    topk_is_sorted: bool = True,
    logits_sum: JTensor = None,
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

  Returns:
    A tuple of next_token_id of shape [batch_size * num_samples] and
      logprobs_at_new_ids of shape [batch_size * num_samples] computed from the
      top_k_logits.  next_token_id of shape [batch_size * num_samples].
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
          logits_is_sorted=topk_is_sorted,
          logits_sum=logits_sum,
      )

    def _false_fn():
      return top_k_logits

    top_p_logits = jax.lax.cond(needs_top_p_mask, _true_fn, _false_fn)
  else:
    top_p_logits = top_p_mask_logits(
        top_k_logits,
        top_p,
        logits_is_sorted=topk_is_sorted,
        logits_sum=logits_sum,
    )

  # Add gumbel noise.
  if temperature is None:
    temperature = 0.0
  gumbel_noise_shape = list(top_p_logits.shape)
  gumbel_noise = _batch_rngs_random_gumbel(prng_key, gumbel_noise_shape).astype(
      top_k_logits.dtype
  )

  # Apply gumbel noise.
  logits_with_noise = top_p_logits + gumbel_noise * temperature
  argmax_ids_in_topk = jnp.argmax(logits_with_noise, axis=-1)
  # Computes log probabilities from top_k logits
  top_k_logprobs = jax.nn.log_softmax(top_k_logits.astype(jnp.float32))
  batch_size = top_k_logits.shape[0]

  return (
      _get_argmax_ids(argmax_ids_in_topk, top_k_indices),
      _get_argmax_ids(argmax_ids_in_topk, top_k_logprobs),
  )


def sample_from_top_k_and_top_p(
    logits: JTensor,
    prng_key: pytypes.PRNGKey,
    temperature: Union[JTensor, float],
    top_k: int,
    top_p: Optional[Union[float, JTensor]] = None,
    per_example_top_k: Optional[JTensor] = None,
    global_normalize: bool = False,
    top_k_recall_target: float = 1.0,
) -> JTensor:
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
    use_top_k_for_logprobs: computes the log probability from the top k logits
      instead of all logits.

  Returns:
    A tensor of shape [batch_size * num_samples].
  """

  # TopK of shape [batch_size * num_samples, top_k]
  top_k_logits, top_k_indices = get_top_k(
      logits, top_k, per_example_top_k, top_k_recall_target
  )
  if global_normalize:
    logits_sum = jnp.sum(logits.astype(jnp.float32), axis=-1, keepdims=True)
  else:
    logits_sum = None
  return sample_from_top_p_given_top_k(
      top_k_logits,
      top_k_indices,
      prng_key,
      temperature,
      top_p,
      topk_is_sorted=True,
      logits_sum=logits_sum,
  )[0]


def sample_from_top_k_and_top_p_with_topk_logprob(
    logits: JTensor,
    prng_key: pytypes.PRNGKey,
    temperature: Union[JTensor, float],
    top_k: int,
    top_p: Optional[Union[float, JTensor]] = None,
    per_example_top_k: Optional[JTensor] = None,
    global_normalize: bool = False,
    top_k_recall_target: float = 1.0,
) -> Sequence[JTensor]:
  """Sample decode algorithm from TopK and TopP with topk log probability.

  Similar to sample_from_top_k_and_top_p, but includes the log probability
  in the returned tuple.

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

  Returns:
    A tuple of next_token_id of shape [batch_size * num_samples] and
      logprobs_at_new_ids of shape [batch_size * num_samples] computed from the
      top_k_logits.  next_token_id of shape [batch_size * num_samples].
  """

  # TopK of shape [batch_size * num_samples, top_k]
  top_k_logits, top_k_indices = get_top_k(
      logits, top_k, per_example_top_k, top_k_recall_target
  )
  if global_normalize:
    logits_sum = jnp.sum(logits.astype(jnp.float32), axis=-1, keepdims=True)
  else:
    logits_sum = None
  return sample_from_top_p_given_top_k(
      top_k_logits,
      top_k_indices,
      prng_key,
      temperature,
      top_p,
      topk_is_sorted=True,
      logits_sum=logits_sum,
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


def _condense_state(block_num_samples) -> base_layer.DecodeStateTransformFn:
  """Pads attention states after prefix fprop."""

  def _condense_state_fn(x, batch_dim, time_dim):
    del batch_dim
    assert time_dim >= 2
    temp_shape = (-1, block_num_samples) + x.shape[time_dim:]
    reshaped_x = x.reshape(temp_shape)
    reshaped_condensed = jnp.squeeze(
        jnp.take(reshaped_x, jnp.array([0]), axis=1), axis=1
    )
    new_shape = x.shape[: time_dim - 1] + (-1,) + x.shape[time_dim:]
    new_x = reshaped_condensed.reshape(new_shape)
    assert (
        x.shape[time_dim - 1] // new_x.shape[time_dim - 1] == block_num_samples
    )
    assert x.size // new_x.size == block_num_samples
    assert new_x.ndim == x.ndim
    return new_x

  return _condense_state_fn


class BaseNextTokenSampler(
    base_hyperparams.FiddleBaseParameterizable, metaclass=abc.ABCMeta
):

  @abc.abstractmethod
  def __call__(
      self,
      model: base_layer.BaseLayerApi,
      logits: JTensor,
      temperature: Union[float, JTensor],
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
      batch_size: Optional[int] = None,
      eos_id: Optional[Union[int, Sequence[int], JTensor]] = None,
  ) -> NestedMap:
    """Initialize any addition decode loop state."""
    return decode_loop_state

  def post_process_decode_loop_state(
      self, decode_loop_state: NestedMap
  ) -> NestedMap:
    """Delete unused decode loop state."""
    return decode_loop_state


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
  """

  top_k: int = 40
  top_p: Optional[Union[float, JTensor]] = None
  epsilon_p: float = 0.0
  global_normalize: bool = False
  top_k_recall_target: float = 1.0
  use_top_k_for_logprobs: bool = False

  def __call__(
      self,
      model: base_layer.BaseLayerApi,
      logits: JTensor,
      temperature: Union[float, JTensor],
      decode_loop_state: NestedMap,
      per_example_top_p: Optional[JTensor] = None,
      per_example_top_k: Optional[JTensor] = None,
      gumbel_prng_key: Optional[JTensor] = None,
  ) -> NestedMap:
    """The default sampling logic implementing top-K and top-P sampling."""
    del decode_loop_state
    assert self.top_k >= 0
    input_logits = logits

    def _get_prng_key(prng_key):
      if prng_key is None:
        # Default method, use `next_prng_key` to generate noise.
        return model.next_prng_key()
      else:
        assert prng_key.shape[0] == logits.shape[0]
        next_model_key = model.next_prng_key()
        batch_size = prng_key.shape[0]
        split_next_model_key = jax.random.split(next_model_key, batch_size)
        prng_key = jax.lax.cond(
            jnp.all(prng_key == DUMMY_PRNG_KEY),
            lambda: split_next_model_key,
            lambda: prng_key,
        )
        return prng_key

    if per_example_top_p is not None:
      top_p = per_example_top_p
    else:
      top_p = getattr(self, 'top_p', None)

    if self.epsilon_p > 0.0:
      logits = epsilon_mask_logits(logits, self.epsilon_p)

    if self.top_k > 1:
      if self.use_top_k_for_logprobs:
        new_ids, logprobs_at_new_ids = (
            sample_from_top_k_and_top_p_with_topk_logprob(
                logits,
                _get_prng_key(gumbel_prng_key),
                temperature=temperature,
                top_k=self.top_k,
                top_p=top_p,
                per_example_top_k=per_example_top_k,
                global_normalize=self.global_normalize,
                top_k_recall_target=self.top_k_recall_target,
            )
        )
        return NestedMap(
            new_ids=new_ids,
            logits=input_logits,
            logprobs_at_new_ids=logprobs_at_new_ids,
        )
      else:
        new_ids = sample_from_top_k_and_top_p(
            logits,
            _get_prng_key(gumbel_prng_key),
            temperature=temperature,
            top_k=self.top_k,
            top_p=top_p,
            per_example_top_k=per_example_top_k,
            global_normalize=self.global_normalize,
            top_k_recall_target=self.top_k_recall_target,
        )
        return NestedMap(
            new_ids=new_ids,
            logits=input_logits,
        )
    elif self.top_k == 0:
      if top_p is not None:
        logits = top_p_mask_logits(logits, top_p)
      if isinstance(temperature, JTensor) or temperature > 0.0:
        gumbel_noise = _batch_rngs_random_gumbel(
            _get_prng_key(gumbel_prng_key), logits.shape
        ).astype(logits.dtype)
        logits += gumbel_noise * temperature
      new_ids = jnp.argmax(logits, axis=1)
    else:  #  k == 1
      new_ids = jnp.argmax(logits, axis=1)
    return NestedMap(new_ids=new_ids, logits=input_logits)


# TODO(b/249483164): Rename BaseLayerApi->BaseLayer after Fiddle migration.
def sample_decode(
    model: base_layer.BaseLayerApi,
    extend_step_fn: decoder_utils.ExtendStepFn,
    transform_state_fn: Optional[decoder_utils.TransformStateFn],
    lazy_broadcast_prefix_fn: Optional[decoder_utils.LazyBroadcastPrefixFn],
    next_token_sampler: base_layer.BaseLayerApi,
    prefix_ids: JTensor,
    prefix_paddings: JTensor,
    seq_len: int,
    num_samples: int,
    fprop_fn: Optional[decoder_utils.FPropFn] = None,
    cf_guidance_scale: Optional[Union[List[float], float, JTensor]] = None,
    fprop_for_prefix: bool = False,
    temperature: Union[float, JTensor] = 1.0,
    gumbel_prng_key: Optional[JTensor] = None,
    per_example_top_p: Optional[Union[JTensor, float]] = None,
    per_example_top_k: Optional[Union[JTensor, int]] = None,
    max_prefix_len: Optional[int] = None,
    max_decode_steps: Optional[Union[int, Sequence[int]]] = None,
    per_example_max_decode_steps: Optional[JTensor] = None,
    prefix_lengths: Optional[JTensor] = None,
    eos_id: Optional[Union[int, Sequence[int], JTensor]] = None,
    result_callback: Optional[StreamingResultCallback] = None,
    decode_loop_mesh_axes_transpose: Optional[Dict[str, str]] = None,
    model_var_pspecs: Optional[base_layer.NestedPartitionSpec] = None,
    return_result_for_suffix_score: bool = False,
    sort_samples: bool = True,
    early_exit=True,
    use_top_k_for_logprobs: bool = False,
    controlled_decoding: Optional[
        decoder_utils.ControlledDecodingHParams
    ] = None,
) -> NestedMap:
  """Sampling decode the input batch.

  Top-K sampling with num_samples for each batch, in which the K most likely
  tokens are filtered and the probability mass is redistributed among only
  those K tokens.

  Args:
    model: The model object.
    extend_step_fn: A function that takes in `states` and the decoded sequence
      at the current time step (with shape [B] or [B, P] where B corresponds to
      the batch size and P corresponds to a possible prefix) and returns a tuple
      of (`NestedMap`, `JTensor`), where the first `NestedMap` corresponds to
      the `new_states` and the second `JTensor` corresponds to the logits of the
      next step.
    transform_state_fn: A function that transforms the decode state.
    lazy_broadcast_prefix_fn: A function that lazily broadcasts decode prefix.
    next_token_sampler: Layer used to sample next token ids given the logits
      output. See DefaultNextTokenSampler for an example. This can be used to
      implement decoding techniques such repetition penalty.
    prefix_ids: The token ids that correspond to the prefix sequence. A JTensor
      of shape [batch, target_sequence_length]. This should have an <SOS> token
      if one is used.
    prefix_paddings: The paddings corresponding to the prefix sequence, with a 1
      denoting padding token and 0 denoting non-padding tokens. A JTensor of
      shape [batch, target_sequence_length].
    seq_len: The output sequence length to decode to. seq_len contains prefix.
    num_samples: Number of samples.
    fprop_fn: A function that takes in the prefix information and initialize the
      decode cache states.
    cf_guidance_scale: If not 1.0, apply classifier-free guidance for
      conditioned generation assuming the inputs are with [cond_a, uncond_a,
      cond_b, uncond_b, ...]. Before sampling, we modify logits as logits =
      uncond_logits + cf_guidance_scale * (cond_logits - uncond_logits) while
      after sampling, we force align sampled token ids of conditioned and
      unconditioned branch.
    fprop_for_prefix: Use one fprop for prefix.
    temperature: Temperature of sampling decoding. It could be a float or a
      JTensor of shape [B] or [B, num_samples].
    gumbel_prng_key: PRNG key for generating gumbel random noise. If None,
      model.next_prng_key() is used; if not None, must be of shape [B] or [B,
      key_shape_dim], where key_shape_dim =
      jax.random.default_prng_impl().key_shape[0]. Usually, key_shape_dim = 2 or
      4. where B is the batch size before being duplicated wrt num_samples or
      cfg. If all the values in gumbel_prng_key is set to DUMMY_PRNG_KEY,
      gumbel_prng_key will be ignored and model.next_prng_key() is used.
    per_example_top_p: Per example top_p of sampling decoding. Optional JTensor
      of shape [B].
    per_example_top_k: Optional per example top_k of shape [batch_size]. The
      value of per_example_top_k should be smaller or equal to `top_k` and
      larger than 0.
    max_prefix_len: Python int or None, the max prefix length for decoding.
    max_decode_steps: Python int or None, the max decode step to run after the
      prefix (if any). Since the prefixes might be of unequal lengths, this
      value is not equivalent with `seq_len` above. When None, decode steps is
      only limited by `seq_len` above. If it is a list, decoding state will be
      padded at each given steps.
    per_example_max_decode_steps: Optional JTensor of shape [B], the maximum
      decode steps defined for each batch. If per_example_max_decode_steps is
      defined, the decoding for each example will be stopped either
      `per_example_max_decode_steps` is reached or `max_decode_steps` is
      reached. If EOS is reached, will also stop early. Normally,
      `per_example_max_decode_steps` should not be set to values larger than
      `max_decode_steps`.
    prefix_lengths: Optional argument supplying prefix sizes to initialize the
      model to decode from a certain target prefix for each position in the
      batch. This can either be None or a JTensor of shape [batch] signifying
      the prefix length for each sequence in the batch.
    eos_id: Optional EOS id which to terminate the decoding early. Could be a
      sequence, an integer or a JTensor. When it is a JTensor, it is 2D tensor
      of shape [batch, eos_len] with padded 0s on the left.
    result_callback: Optional callback function to be called for decoding
      results with a configurable interval.
    decode_loop_mesh_axes_transpose: Optional mesh transpose for decoding loop.
    model_var_pspecs: Optional partition specs for model variables.
    return_result_for_suffix_score: Whether or not to return result for suffix
      score.
    sort_samples: Whether to sort the samples by logprobs.
    early_exit: A bool, whether or not to allow early exit.
    use_top_k_for_logprobs: computes the log probability from the top k logits
      instead of all logits.
    controlled_decoding: Params to configure blockwise controlled decoding.

  Returns:
    A NestedMap with `.prefix_lengths` (indicating the lengths of prefixes for
    each target sequence), `.output_ids` (matrix of int ids with the
    decoded output), `.decode_lengths` (vector of ints indicating the lengths
    of non-padding tokens in `.output_ids`, which includes the prefix), and
    `.logprobs` (the log probability of selected tokens, including the prefix,
    where a positive value of 1.0 is used to indicate padded positions).
    The outputs has shape [batch, num_samples, ...].
  """
  # Init decode state using fprop_fn, state seq size is max_prefix_len.
  if fprop_fn:
    fprop_fn(model, prefix_ids, prefix_paddings)
  model = decoder_utils.maybe_reshard_mdl_for_decode(
      model,
      decode_loop_mesh_axes_transpose,
      model_var_pspecs,
      transform_state_fn,
  )
  with decoder_utils.maybe_decode_mesh_transpose(
      model, decode_loop_mesh_axes_transpose
  ):
    if fprop_fn and lazy_broadcast_prefix_fn is None and transform_state_fn:
      # Pad to full-sequence length.
      first_max_decode_steps = (
          min(max_decode_steps)
          if isinstance(max_decode_steps, Sequence)
          else max_decode_steps
      )
      pad_state_sizes = (
          first_max_decode_steps if fprop_for_prefix else seq_len - 1
      )

      transform_state_fn(model, decoder_utils.pad_state_fn(pad_state_sizes))
    return sample_decode_after_fprop(
        model,
        extend_step_fn,
        transform_state_fn,
        lazy_broadcast_prefix_fn,
        next_token_sampler,
        prefix_ids,
        prefix_paddings,
        seq_len,
        num_samples,
        cf_guidance_scale,
        fprop_for_prefix,
        temperature,
        gumbel_prng_key,
        per_example_top_p,
        per_example_top_k,
        max_prefix_len,
        max_decode_steps,
        per_example_max_decode_steps,
        prefix_lengths,
        eos_id,
        result_callback,
        return_result_for_suffix_score,
        sort_samples,
        early_exit,
        use_top_k_for_logprobs,
        controlled_decoding,
    )


# TODO(b/249483164): Rename BaseLayerApi->BaseLayer after Fiddle migration.
def sample_decode_after_fprop(
    model: base_layer.BaseLayerApi,
    extend_step_fn: decoder_utils.ExtendStepFn,
    transform_state_fn: Optional[decoder_utils.TransformStateFn],
    lazy_broadcast_prefix_fn: Optional[decoder_utils.LazyBroadcastPrefixFn],
    next_token_sampler: base_layer.BaseLayerApi,
    prefix_ids: JTensor,
    prefix_paddings: JTensor,
    seq_len: int,
    num_samples: int,
    cf_guidance_scale: Optional[Union[List[float], float, JTensor]] = None,
    fprop_for_prefix: bool = False,
    temperature: Union[float, JTensor] = 1.0,
    gumbel_prng_key: Optional[JTensor] = None,
    per_example_top_p: Optional[JTensor] = None,
    per_example_top_k: Optional[JTensor] = None,
    max_prefix_len: Optional[int] = None,
    max_decode_steps: Optional[Union[int, Sequence[int]]] = None,
    per_example_max_decode_steps: Optional[JTensor] = None,
    prefix_lengths: Optional[JTensor] = None,
    eos_id: Optional[Union[int, Sequence[int], JTensor]] = None,
    result_callback: Optional[StreamingResultCallback] = None,
    return_result_for_suffix_score: bool = False,
    sort_samples: bool = True,
    early_exit=True,
    use_top_k_for_logprobs: bool = False,
    controlled_decoding: Optional[
        decoder_utils.ControlledDecodingHParams
    ] = None,
) -> NestedMap:
  """Sampling decode after init decode state the input batch.

  fprop is called to initialize decode states. This function runs decoding
  loops after decode state are initialized.

  Top-K sampling with num_samples for each batch, in which the K most likely
  tokens are filtered and the probability mass is redistributed among only
  those K tokens.

  Args:
    model: The model object.
    extend_step_fn: A function that takes in `states` and the decoded sequence
      at the current time step (with shape [B] or [B, P] where B corresponds to
      the batch size and P corresponds to a possible prefix) and returns a tuple
      of (`NestedMap`, `JTensor`), where the first `NestedMap` corresponds to
      the `new_states` and the second `JTensor` corresponds to the logits of the
      next step.
    transform_state_fn: A function that transforms the decode state.
    lazy_broadcast_prefix_fn: A function that lazily broadcasts decode prefix.
    next_token_sampler: Layer used to sample next token ids given the logits
      output. See DefaultNextTokenSampler for an example. This can be used to
      implement decoding techniques such repetition penalty.
    prefix_ids: The token ids that correspond to the prefix sequence. A JTensor
      of shape [batch, target_sequence_length]. This should have an <SOS> token
      if one is used.
    prefix_paddings: The paddings corresponding to the prefix sequence, with a 1
      denoting padding token and 0 denoting non-padding tokens. A JTensor of
      shape [batch, target_sequence_length].
    seq_len: The output sequence length to decode to. seq_len contains prefix.
    num_samples: Number of samples.
    cf_guidance_scale: If not 1.0, apply classifier-free guidance for
      conditioned generation assuming the inputs are with [cond_a, uncond_a,
      cond_b, uncond_b, ...]. Before sampling, we modify logits as logits =
      uncond_logits + cf_guidance_scale * (cond_logits - uncond_logits) while
      after sampling, we force align sampled token ids of conditioned and
      unconditioned branch.
    fprop_for_prefix: Use one fprop for prefix.
    temperature: Temperature of sampling decoding. It could be a float or a
      JTensor of shape [B] or [B, num_samples].
    gumbel_prng_key: PRNG key for generating gumbel random noise. If None,
      model.next_prng_key() is used; if not None, must be of shape [B] or [B,
      key_shape_dim], where key_shape_dim =
      jax.random.default_prng_impl().key_shape[0]. Usually, key_shape_dim = 2 or
      4. where B is the batch size before being duplicated wrt num_samples or
      cfg. If all the values in gumbel_prng_key is set to DUMMY_PRNG_KEY,
      gumbel_prng_key will be ignored and model.next_prng_key() is used.
    per_example_top_p: Per example top_p of sampling decoding. Optional JTensor
      of shape [B].
    per_example_top_k: Optional per example top_k of shape [batch_size]. The
      value of per_example_top_k should be smaller or equal to `top_k` and
      larger than 0.
    max_prefix_len: Python int or None, the max prefix length for decoding.
    max_decode_steps: Python int or None, the max decode step to run after the
      prefix (if any). Since the prefixes might be of unequal lengths, this
      value is not equivalent with `seq_len` above. When None, decode steps is
      only limited by `seq_len` above. If it is a list, decoding state will be
      padded at each given steps.
    per_example_max_decode_steps: Optional JTensor of shape [B], the maximum
      decode steps defined for each batch. If per_example_max_decode_steps is
      defined, the decoding for each example will be stopped either
      `per_example_max_decode_steps` is reached or `max_decode_steps` is
      reached. If EOS is reached, will also stop early. Normally,
      `per_example_max_decode_steps` should not be set to values larger than
      `max_decode_steps`.
    prefix_lengths: Optional argument supplying prefix sizes to initialize the
      model to decode from a certain target prefix for each position in the
      batch. This can either be None or a JTensor of shape [batch] signifying
      the prefix length for each sequence in the batch.
    eos_id: Optional EOS id which to terminate the decoding early. Could be a
      sequence, an integer or a JTensor. When it is a JTensor, it is 2D tensor
      of shape [batch, eos_len] with padded 0s on the left.
    result_callback: Optional callback function to be called for decoding
      results with a configurable interval.
    return_result_for_suffix_score: Whether or not to return result for suffix
      score.
    sort_samples: Whether to sort the samples by logprobs.
    early_exit: A bool, whether or not to allow early exit.
    use_top_k_for_logprobs: computes the log probability from the top k logits
      instead of all logits.
    controlled_decoding: Params to configure blockwise controlled decoding.

  Returns:
    A NestedMap with `.prefix_lengths` (indicating the lengths of prefixes for
    each target sequence), `.output_ids` (matrix of int ids with the
    decoded output), `.decode_lengths` (vector of ints indicating the lengths
    of non-padding tokens in `.output_ids`, which includes the prefix), and
    `.logprobs` (the log probability of selected tokens, including the prefix,
    where a positive value of 1.0 is used to indicate padded positions).
    The outputs has shape [batch, num_samples, ...].
  """
  original_batch_size = prefix_ids.shape[0]
  original_prefix_lengths = prefix_lengths
  if controlled_decoding:
    asserts.gt(controlled_decoding.block_num_samples, 0)
    if controlled_decoding.interval > 0 and not isinstance(
        max_decode_steps, int
    ):
      raise ValueError(
          'max_decode_steps must be an int when controlled_decoding.interval is'
          ' specified.'
      )
  if isinstance(max_decode_steps, int):
    max_decode_steps = [max_decode_steps]
  max_decode_steps = sorted(max_decode_steps) if max_decode_steps else [seq_len]
  if isinstance(eos_id, int):
    eos_id = [eos_id]

  if num_samples > 1:
    # Broadcast inputs from [batch, ...] to [batch * num_samples, ...].
    # [a, b, c] and num_samples = 3 will have
    # [a, a, a, b, b, b, c, c, c]. If cf_guidance_scale is enabled, it will have
    # [cond_a, cond_a, cond_a, uncond_a, uncond_a, uncond_a, ...].
    prefix_ids = jnp.repeat(prefix_ids, axis=0, repeats=num_samples)
    prefix_paddings = jnp.repeat(prefix_paddings, axis=0, repeats=num_samples)
    prefix_lengths = (
        None
        if prefix_lengths is None
        else jnp.repeat(prefix_lengths, axis=0, repeats=num_samples)
    )

    def _broadcast_input(x: JTensor, name: str) -> JTensor:
      if cf_guidance_scale is not None:
        x_expected_shape = (original_batch_size // 2,)
      else:
        x_expected_shape = (original_batch_size,)
      if x.shape[0] != x_expected_shape[0]:  # pytype: disable=attribute-error
        raise ValueError(
            f'Dynamic {name} should have shape: '
            f'{x_expected_shape}, but it has shape: '
            f'{x.shape}.'  # pytype: disable=attribute-error
        )
      x = jnp.repeat(x, axis=0, repeats=num_samples)
      return x

    # If temperature is a JTensor of 1D shape [batch_size]
    # broadcast it to shape [batch_size * num_samples].
    # If temperature is of 2D shape [batch_size, num_samples] simply flatten it.
    if isinstance(temperature, JTensor):
      if temperature.ndim == 2:  # pytype: disable=attribute-error
        if temperature.shape != (original_batch_size, num_samples):  # pytype: disable=attribute-error
          raise ValueError(
              '2D Dynamic temperature should have shape: '
              f'({original_batch_size}, {num_samples}), but it has shape: '
              f'{temperature.shape}.'  # pytype: disable=attribute-error
          )
        temperature = jnp.reshape(temperature, (-1,))
      else:
        temperature = _broadcast_input(temperature, 'temperature')

    # Broadcast per_example_max_decode_steps if it is a JTensor.
    if per_example_max_decode_steps is not None:
      per_example_max_decode_steps = _broadcast_input(
          per_example_max_decode_steps.astype(jnp.int32),
          'per_example_max_decode_steps',
      )

    if per_example_top_p is not None:
      per_example_top_p = _broadcast_input(
          per_example_top_p, 'per_example_top_p'
      )

    if per_example_top_k is not None:
      per_example_top_k = _broadcast_input(
          per_example_top_k, 'per_example_top_k'
      )

    if eos_id is not None and isinstance(eos_id, JTensor):
      assert eos_id.ndim == 2, (
          'eos_id in sample_decode as JTensor needs to be rank 2, eos_id'
          f' has dimension: {eos_id.ndim}'
      )
      eos_id = _broadcast_input(eos_id, 'eos_id').astype(jnp.int32)

    if lazy_broadcast_prefix_fn is not None:
      assert fprop_for_prefix

      # We need to exclude the last token from prefix, and instead move it to
      # the multi-sample suffix. This is because the last token only as an Input
      # ID, but not an output ID (label), and we need to start decoding from it.
      transform_state_fn(model, decoder_utils.slice_state_fn(0, -1))
      first_decode_steps = min(max_decode_steps)
      if controlled_decoding:
        if controlled_decoding.interval:
          first_decode_steps = controlled_decoding.interval
        lazy_broadcast_prefix_fn(model, num_samples, first_decode_steps)
      else:
        # max_decode_steps + 1 to include last token from prefix.
        lazy_broadcast_prefix_fn(model, num_samples, first_decode_steps + 1)
    elif transform_state_fn is not None:
      # Broadcast prefix state for num_samples.
      transform_state_fn(
          model, decoder_utils.batch_broadcast_state_fn(num_samples)
      )

  # If cf guidance scale is a list floats with length == num_samples, we
  # convert it to the target shape to be used in decode loop_body.
  if isinstance(cf_guidance_scale, Sequence):
    assert len(cf_guidance_scale) == num_samples
    cf_guidance_scale = jnp.array(cf_guidance_scale)
    cf_guidance_scale = cf_guidance_scale[jnp.newaxis, :, jnp.newaxis]
  elif isinstance(cf_guidance_scale, JTensor):
    assert cf_guidance_scale.ndim == 2
    assert cf_guidance_scale.shape[-1] == num_samples
    cf_guidance_scale = cf_guidance_scale[:, :, jnp.newaxis]

  if isinstance(temperature, JTensor):
    temperature = temperature[:, jnp.newaxis]

  if isinstance(per_example_top_p, JTensor):
    per_example_top_p = per_example_top_p[:, jnp.newaxis]

  if gumbel_prng_key is not None and isinstance(gumbel_prng_key, JTensor):
    gumbel_prng_key = gumbel_prng_key.astype(jnp.uint32)
    dup_len = jax.random.default_prng_impl().key_shape[0]
    if len(gumbel_prng_key.shape) == 1:
      gumbel_prng_key = jnp.stack([gumbel_prng_key] * dup_len, axis=-1)

  if seq_len <= 0:
    raise ValueError(
        'The sequence length for decoding must be > 0, '
        f'current value = {seq_len}.'
    )
  last_decode_steps = max(max_decode_steps) if max_decode_steps else seq_len
  per_example_max_decode_steps = (
      last_decode_steps
      if per_example_max_decode_steps is None
      else per_example_max_decode_steps
  )
  per_example_max_decode_steps = jnp.minimum(
      per_example_max_decode_steps, last_decode_steps
  )
  if cf_guidance_scale is not None and isinstance(
      per_example_max_decode_steps, JTensor
  ) and per_example_max_decode_steps.ndim == 1:
    per_example_max_decode_steps = jnp.concatenate(
        [per_example_max_decode_steps[:, None]] * 2, axis=1
    )
    per_example_max_decode_steps = jnp.reshape(
        per_example_max_decode_steps, (-1)
    )
  batch_size = prefix_ids.shape[0]

  # If prefix length is not specified, set it to 0.
  if prefix_lengths is None:
    prefix_lengths = jnp.zeros([batch_size], dtype=jnp.int32)

  output_ids = jnp.zeros(shape=(batch_size, seq_len), dtype=jnp.int32)

  val = NestedMap()
  if fprop_for_prefix:
    # Update output_ids with prefix_ids.
    output_ids = jax.lax.dynamic_update_slice(
        output_ids, prefix_ids, [0] * output_ids.ndim
    )
    assert max_prefix_len is not None
    # Update loop init states with prefix.
    start_step = max_prefix_len - 1
    val.segment_pos = prefix_lengths - 1
  else:
    output_ids = output_ids.at[:, 0].set(prefix_ids[:, 0])
    start_step = 0
    val.segment_pos = jnp.zeros([batch_size], dtype=jnp.int32)

  val.start_step = start_step
  val.step = start_step
  val.output_ids = output_ids
  # Shape [batch_size], whether each row has terminated and should stop.
  val.done = jnp.zeros(shape=batch_size, dtype=jnp.bool_)
  val.has_eos = jnp.zeros(shape=batch_size, dtype=jnp.bool_)
  val.decode_lengths = jnp.ones_like(prefix_lengths) * seq_len
  # We use a positive value of 1.0 to indicate blank or padded positions.
  val.logprobs = jnp.ones_like(output_ids, dtype=jnp.float32)
  val = next_token_sampler.init_decode_loop_state(val, batch_size, eos_id)

  if result_callback is not None and result_callback.init_fn is not None:
    result_callback.init_fn((original_batch_size, num_samples))

  def get_cond_func(stop_at_decode_steps):
    """Gets conditional function for different decode steps."""

    def cond_func(model, val):
      """Whether the while loop should continue."""
      del model
      # We continue the decoding search iff both:
      #   (1) We have yet to exceed the max steps set by self.decoder.seqlen
      #   (2) At least one row in the batch has not terminated.
      max_steps = start_step + stop_at_decode_steps
      length_ok = val.step < min(seq_len - 1, max_steps)
      all_rows_done = jnp.all(val.done)
      return jnp.logical_and(length_ok, jnp.logical_not(all_rows_done))

    return cond_func

  def loop_body(model, val):
    """From ids at `step`, update output ids at `step + 1`."""
    step = val.step
    logits = extend_step_fn(model, val.output_ids[:, step], val.segment_pos)
    if cf_guidance_scale is not None:
      # Split cond / uncond logits.
      logits_split = split_batch_dim(logits, 0, 2 * num_samples)
      cond_logits = logits_split[:, :num_samples]
      uncond_logits = logits_split[:, num_samples:]
      logits = uncond_logits + cf_guidance_scale * (cond_logits - uncond_logits)
      logits = jnp.reshape(logits, (-1,) + logits.shape[2:])
    if gumbel_prng_key is not None:
      # Splits prng_key for num_samples.
      split_gumbel_prng_key = jax.vmap(
          lambda x: jax.random.split(x, num_samples)
      )(gumbel_prng_key)
      split_gumbel_prng_key = jnp.reshape(
          split_gumbel_prng_key, (-1, *gumbel_prng_key.shape[1:])
      )
      # Folds split prng_key for step.
      split_gumbel_prng_key = jax.vmap(lambda x: jax.random.fold_in(x, step))(
          split_gumbel_prng_key
      )
      assert split_gumbel_prng_key.shape[0] == logits.shape[0]

      split_gumbel_prng_key = jax.lax.cond(
          jnp.all(gumbel_prng_key == DUMMY_PRNG_KEY),
          lambda: jnp.ones_like(split_gumbel_prng_key) * DUMMY_PRNG_KEY,
          lambda: split_gumbel_prng_key,
      )
    else:
      split_gumbel_prng_key = None
    sampler_output = next_token_sampler(
        model,
        logits,
        temperature,
        val,
        per_example_top_p=per_example_top_p,
        per_example_top_k=per_example_top_k,
        gumbel_prng_key=split_gumbel_prng_key,
    )
    new_ids, sample_logits = sampler_output.new_ids, sampler_output.logits
    assert new_ids.shape == (sample_logits.shape[0],)
    assert new_ids.dtype == jnp.int32

    model.add_summary('new_ids', new_ids)
    model.add_summary('sample_logits', sample_logits)

    if cf_guidance_scale is not None:
      # Force-align unconditioned branch as conditioned sampled tokens ids.
      new_ids = split_batch_dim(new_ids, 0, num_samples)
      new_ids = jnp.concatenate([new_ids, new_ids], axis=1)
      new_ids = jnp.reshape(new_ids, (-1,) + new_ids.shape[2:])

    # When step becomes prefix_length - 1, the new output has index beyond
    # the known prefix.
    # If prefix_length is 0, the condition is always False, so we take the
    # decoded output rather than the prefix.
    # Selects prefix ids when step smaller than prefix_lengths when using
    # extend_step for prefix.
    if not fprop_for_prefix:
      new_ids = jnp.where(
          step < prefix_lengths - 1, prefix_ids[:, step + 1], new_ids
      )
    prev_done = val.done
    new_ids = jnp.where(prev_done, jnp.zeros_like(new_ids), new_ids)
    val.output_ids = val.output_ids.at[:, step + 1].set(new_ids)
    if eos_id is not None:
      if isinstance(eos_id, JTensor):
        has_eos = decoder_utils.end_with_sequences(
            eos_id, val.output_ids, val.step + 1
        )
      else:
        has_eos = decoder_utils.has_any_eos(new_ids, eos_id)
      val.done = jnp.logical_or(
          prev_done,
          has_eos,
      )
      val.has_eos = jnp.logical_or(val.has_eos, has_eos)
    if fprop_for_prefix:
      prefix_offset = max_prefix_len
      decode_lengths = prefix_lengths + (step - max_prefix_len + 2)
    else:
      # if eos is part of prefix, ignore it.
      val.done = jnp.where(step < prefix_lengths - 1, prev_done, val.done)
      prefix_offset = prefix_lengths
      decode_lengths = jnp.ones_like(val.decode_lengths) * (step + 2)
    val.segment_pos += 1

    max_decoding_steps_reached = (
        jnp.ones_like(prefix_lengths) * (step + 2) - prefix_offset
    ) >= per_example_max_decode_steps
    val.done = jnp.logical_or(val.done, max_decoding_steps_reached)
    done_at_this_step = jnp.logical_and(jnp.logical_not(prev_done), val.done)
    val.decode_lengths = jnp.where(
        done_at_this_step, decode_lengths, val.decode_lengths
    )

    if use_top_k_for_logprobs and sampler_output.Has('logprobs_at_new_ids'):
      logprobs_at_new_ids = sampler_output.logprobs_at_new_ids
    else:
      logprobs = jax.nn.log_softmax(logits.astype(jnp.float32))
      logprobs_at_new_ids = logprobs.at[jnp.arange(batch_size), new_ids].get()
    logprobs_at_new_ids = jnp.where(
        prev_done, jnp.ones_like(logprobs_at_new_ids), logprobs_at_new_ids
    )
    val.logprobs = val.logprobs.at[:, step + 1].set(logprobs_at_new_ids)

    if result_callback is not None:

      def _false_fn():
        """Dummy function."""
        pass

      def _get_slice(sequence):
        mod_size = (
            val.output_ids.shape[-1] - 1 - start_step
        ) % result_callback.interval_steps
        if mod_size > 0:
          sequence = jnp.pad(
              sequence,
              [[0, 0], [0, result_callback.interval_steps - mod_size]],
          )
        interval_start_id = (
            ((step - start_step) // result_callback.interval_steps)
            * result_callback.interval_steps
            + start_step
            + 1
        )
        return jax.lax.dynamic_slice(
            sequence,
            [0, interval_start_id],
            [batch_size, result_callback.interval_steps],
        )

      def _true_fn():
        """Outfeed logic."""
        # prefix_lengths: [b]
        # decode_lengths: [b * num_samples]
        # output_ids: [b * num_samples, interval_steps]
        # scores: [b * num_samples]
        outfeed_tensors = NestedMap()
        outfeed_tensors.output_ids = _get_slice(val.output_ids)
        outfeed_tensors.decode_lengths = (
            jnp.ones_like(val.decode_lengths) * result_callback.interval_steps
        )
        outfeed_tensors.scores = jnp.sum(
            # Padded logprobs can have values of 1.0, so we cap it to 0.0.
            jnp.minimum(_get_slice(val.logprobs), 0.0),
            axis=-1,
        )
        outfeed_tensors = jax.tree_map(
            lambda x: split_batch_dim(x, 0, num_samples), outfeed_tensors
        )
        outfeed_tensors.prefix_lengths = jnp.zeros_like(original_prefix_lengths)

        result_callback.callback_fn(outfeed_tensors)

      should_outfeed = jnp.logical_or(
          (step - start_step + 1) % result_callback.interval_steps == 0,
          jnp.all(val.done),
      )
      jax.lax.cond(should_outfeed, _true_fn, _false_fn)
    val.step += 1
    return val

  if controlled_decoding:
    result = val
    if controlled_decoding.interval:
      assert max_decode_steps[0] % controlled_decoding.interval == 0
      chunks = max_decode_steps[0] // controlled_decoding.interval
      decode_buckets = [
          (i + 1) * controlled_decoding.interval for i in range(chunks)
      ]
    else:
      decode_buckets = max_decode_steps
    # After the first iteration, condense the decode state since we know that
    # there are only `controlled_decoding.block_num_samples` unique samples.
    # Also perform lazy prefix broadcast and create new decode states with
    # length=controlled_decoding.interval.
    #
    # Over the course of decoding, an example of how the shape of decode states
    # will evolve is below. Given hyperparameters:
    # num_samples = 16
    # controlled_decoding.block_num_samples = 2
    # controlled_decoding.interval = 64
    #
    # Attention decode state shapes at each iteration:
    # Iter 0 (time_dim = 2): [1, 16, 64, 16, 128]
    # Iter 1 (time_dim = 3): [1, 8, 2, 64, 16, 128]
    # Iter 2 (time_dim = 4): [1, 8, 1, 2, 64, 16, 128]
    # Iter 3 (time_dim = 5): [1, 8, 1, 1, 2, 64, 16, 128]
    for i in range(len(decode_buckets)):
      if i > 0:
        pad_size = decode_buckets[i] - decode_buckets[i - 1]
        transform_state_fn(
            model, _condense_state(controlled_decoding.block_num_samples)
        )
        lazy_broadcast_prefix_fn(
            model,
            controlled_decoding.block_num_samples,  # num_suffix_samples
            pad_size,  # suffix_length
        )
      result = nn.while_loop(
          get_cond_func(decode_buckets[i]),
          loop_body,
          model,
          result,
          split_rngs={RANDOM: True},
          carry_variables=[DECODE_CACHE],
      )

  elif early_exit:
    result = val
    for i in range(len(max_decode_steps)):
      if i > 0:
        pad_size = max_decode_steps[i] - max_decode_steps[i - 1]
        transform_state_fn(model, decoder_utils.pad_state_fn(pad_size))
      result = nn.while_loop(
          get_cond_func(max_decode_steps[i]),
          loop_body,
          model,
          result,
          split_rngs={RANDOM: True},
          carry_variables=[DECODE_CACHE],
      )

  else:
    # We call nn.scan to allow propagation of summaries through decoding loop.
    # Summary and AuxLoss are concatenated on the first dim.

    def scan_body(model, val, scan_input):
      # Adapt loop_body for use in scan.
      del scan_input
      val = loop_body(model, val)
      return val, {}

    scan_fn = nn.scan(
        scan_body,
        variable_axes={AUX_LOSS: 0, SUMMARIES: 0},
        variable_broadcast=[PARAMS, NON_TRAINABLE],
        variable_carry=[DECODE_CACHE, PREFIX_DECODE_CACHE],
        split_rngs={RANDOM: True},
        in_axes=0,
        out_axes=0,
    )

    def pop_collection(module, col_name):
      """Pops all variables in a given collection."""
      col = module.scope._mutable_collection(col_name)
      col_keys = list(col.keys())
      popped = {}
      for key in col_keys:
        popped[key] = col.pop(key)
      return popped

    def reinsert_collection(module, col_name, data):
      """Inserts data into collection *without* overwriting."""
      col = module.scope._mutable_collection(col_name)

      def put(target, key, val):
        # traverse dicts to insert recursively
        if key not in target:
          target[key] = val
        elif isinstance(target[key], dict) and isinstance(val, dict):
          # traverse dict recursively
          for k, v in val.items():
            put(target[key], k, v)
        else:
          # don't overwrite existing leaves!
          pass

      for key in list(data.keys()):
        put(col, key, data[key])

    if model.is_mutable_collection(base_layer.SUMMARIES):
      # stash out the summaries temporarily
      model_summaries_copy = pop_collection(model, base_layer.SUMMARIES)

    scan_len = seq_len
    if max_prefix_len:
      scan_len -= max_prefix_len
    dummy_inputs = {'dummy': jnp.zeros([scan_len, 2])}
    result, _ = scan_fn(model, val, dummy_inputs)

    # Now merge back the summaries.
    if model.is_mutable_collection(base_layer.SUMMARIES):
      # recursively merge two dictionaries.
      reinsert_collection(model, base_layer.SUMMARIES, model_summaries_copy)

  if result_callback is not None and result_callback.done_fn is not None:
    result_callback.done_fn()

  if return_result_for_suffix_score:
    return result

  del result.segment_pos

  result.prefix_lengths = prefix_lengths
  result.original_lengths = jnp.sum(1.0 - prefix_paddings, axis=1).astype(
      jnp.int32
  )

  if fprop_for_prefix:
    prefix_ids = decoder_utils.left_align_tensor(
        prefix_ids, prefix_lengths, max_prefix_len
    )

  # We manually pad out the ids not belonging to the prefix because some
  # tokenizers tested do not always obey the lengths arg.
  indices = jnp.tile(jnp.arange(prefix_ids.shape[1]), (prefix_ids.shape[0], 1))
  prefix_lengths_2d = jnp.tile(
      prefix_lengths[:, None], (1, prefix_ids.shape[1])
  )
  prefix_ids = jnp.where(
      indices < prefix_lengths_2d, prefix_ids, jnp.zeros_like(prefix_ids)
  )
  result.prefix_ids = prefix_ids

  if fprop_for_prefix:
    # TODO(b/229679837): return decoding ids only and
    # remove left align logic here.

    # Change output_ids to left align.
    result.output_ids = decoder_utils.left_align_tensor(
        result.output_ids, prefix_lengths, max_prefix_len
    )
    result.logprobs = decoder_utils.left_align_tensor(
        result.logprobs, prefix_lengths, max_prefix_len
    )

  del result.start_step, result.step, result.done, result.has_eos
  result = next_token_sampler.post_process_decode_loop_state(result)

  if cf_guidance_scale is not None:
    # Split cond / uncond branches and only return conditioned branch.
    result = jax.tree_map(
        lambda x: split_batch_dim(x, 0, 2 * num_samples)[:, :num_samples],
        result,
    )
  else:
    result = jax.tree_map(lambda x: split_batch_dim(x, 0, num_samples), result)
  if num_samples > 1 and sort_samples:
    return sort_samples_by_scores(result)

  return result


# TODO(b/249483164): Rename BaseLayerApi->BaseLayer after Fiddle migration.
def greedy_decode(
    model: base_layer.BaseLayerApi,
    extend_step_fn: decoder_utils.ExtendStepFn,
    prefix_ids: JTensor,
    prefix_paddings: JTensor,
    seq_len: int,
    fprop_for_prefix: bool = False,
    fprop_fn: Optional[decoder_utils.FPropFn] = None,
    transform_state_fn: Optional[decoder_utils.TransformStateFn] = None,
    max_prefix_len: Optional[int] = None,
    max_decode_steps: Optional[Union[int, Sequence[int]]] = None,
    prefix_lengths: Optional[JTensor] = None,
    decode_loop_mesh_axes_transpose: Optional[Dict[str, str]] = None,
    model_var_pspecs: Optional[base_layer.NestedPartitionSpec] = None,
    eos_id: Optional[int] = None,
) -> NestedMap:
  """Greedy decode the input batch.

  Args:
    model: The model object.
    extend_step_fn: A function that takes in `states` and the decoded sequence
      at the current time step (with shape [B] or [B, P] where B corresponds to
      the batch size and P corresponds to a possible prefix) and returns a tuple
      of (`NestedMap`, `JTensor`), where the first `NestedMap` corresponds to
      the `new_states` and the second `JTensor` corresponds to the logits of the
      next step.
    prefix_ids: The token ids that correspond to the prefix sequence. This
      should contain an <SOS> token if one is used.
    prefix_paddings: The paddings corresponding to the prefix sequence, with a 1
      denoting padding token and 0 denoting non-padding tokens.
    seq_len: The output sequence length to decode to. seq_len contains prefix.
    fprop_for_prefix: Use one fprop for prefix.
    fprop_fn: A function that takes in the prefix information and initialize the
      decode cache states.
    transform_state_fn: A function that transforms the decode state.
    max_prefix_len: Python int or None, the max prefix length for decoding.
    max_decode_steps: Python int or None, the max decode step to run after the
      prefix (if any). Since the prefixes might be of unequal lengths, this
      value is not equivalent with `seq_len` above. When None, decode steps is
      only limited by `seq_len` above. If it is a list, decoding state will be
      padded at each given steps.
    prefix_lengths: Optional argument supplying prefix sizes to initialize the
      model to decode from a certain target prefix for each position in the
      batch. This can either be None or a JTensor of shape [batch] signifying
      the prefix length for each sequence in the batch.
    decode_loop_mesh_axes_transpose: Optional mesh transpose for decoding loop.
    model_var_pspecs: Optional partition specs for model variables.
    eos_id: Optional EOS id which to terminate the decoding early.

  Returns:
    A NestedMap with `.prefix_lengths` (indicating the lengths of prefixes for
    each target sequence), `.output_ids` (matrix of int ids with the
    decoded output), `.decode_lengths` (vector of ints indicating the lengths
    of non-padding tokens in `.output_ids`, which includes the prefix), and
    `.logprobs` (the log probability of selected tokens, including the prefix,
    where a positive value of 1.0 is used to indicate padded positions).
  """
  next_token_sampler = base_layer.instantiate(
      DefaultNextTokenSampler.HParams(top_k=1)
  )
  return sample_decode(
      model,
      extend_step_fn,
      transform_state_fn=transform_state_fn,
      lazy_broadcast_prefix_fn=None,
      fprop_fn=fprop_fn,
      next_token_sampler=next_token_sampler,
      prefix_ids=prefix_ids,
      prefix_paddings=prefix_paddings,
      seq_len=seq_len,
      fprop_for_prefix=fprop_for_prefix,
      max_prefix_len=max_prefix_len,
      max_decode_steps=max_decode_steps,
      prefix_lengths=prefix_lengths,
      decode_loop_mesh_axes_transpose=decode_loop_mesh_axes_transpose,
      model_var_pspecs=model_var_pspecs,
      eos_id=eos_id,
      num_samples=1,
      temperature=0.0,
  )
