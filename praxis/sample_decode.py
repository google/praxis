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

"""Algorithms for the sample decode.

This file contains sample decode with temperature and greedy decode algorithms.
Greedy decode is a special case for sample decode.
"""

import functools
from typing import List, Sequence, Optional, Union

from flax import linen as nn
import jax
from jax import numpy as jnp
from praxis import base_layer
from praxis import decoder_utils
from praxis import py_utils
from praxis import pytypes

NestedMap = py_utils.NestedMap
JTensor = base_layer.JTensor

RANDOM = base_layer.RANDOM
PARAMS = base_layer.PARAMS
NON_TRAINABLE = base_layer.NON_TRAINABLE
AUX_LOSS = base_layer.AUX_LOSS
SUMMARIES = base_layer.SUMMARIES
DECODE_CACHE = base_layer.DECODE_CACHE
PREFIX_DECODE_CACHE = base_layer.PREFIX_DECODE_CACHE


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
      jnp.expand_dims(jnp.arange(x.shape[0]), 1), indices.shape[1], axis=1)
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
      jnp.greater_equal(result.logprobs, 1.0), jnp.zeros_like(result.logprobs),
      result.logprobs)
  scores = jnp.sum(logprobs, -1)
  num_samples = scores.shape[-1]
  # Uses topk to get indices.
  _, indices = jax.lax.top_k(scores, num_samples)
  return jax.tree_map(
      functools.partial(reorder_with_indices, indices=indices), result)


def split_batch_dim(x: jnp.ndarray, batch_dim: int,
                    num_samples: int) -> jnp.ndarray:
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


def sample_from_topk_with_gumbel_noise(logits: JTensor, gumbel_noise: JTensor,
                                       temperature: Union[JTensor, float],
                                       topk: int) -> JTensor:
  """Sample decode algorithm from TopK with gumbel noise.

  Args:
    logits: Logits at current step. This is a JTensor of [batch_size *
      num_samples, vocab_size].
    gumbel_noise: Gumbel noise. This is a JTensor of [batch_size * num_samples,
      vocab_size].
    temperature: Temperature of sampling decoding. It could be a float or
      a JTensor of shape [batch_size].
    topk: If nonzero, use top-k sampling, only selecting among the most likely k
      tokens at each step.

  Returns:
    A tensor of shape [batch_size * num_samples].
  """
  # TopK of shape [batch_size * num_samples, topk]
  topk_value, topk_indices = jax.lax.top_k(logits, topk)

  # TODO(wangtao): Add more sampling logics.
  # Add gumbel noise.
  new_logits = topk_value + gumbel_noise * temperature
  # Get argmax with gumbel noise.
  top1_id = jnp.argmax(new_logits, axis=-1)
  # Get real argmax id.
  new_ids = jnp.sum(
      jax.nn.one_hot(top1_id, topk, dtype=top1_id.dtype) * topk_indices, -1)

  return new_ids


def sample_from_topk(logits: JTensor, prng_key: pytypes.PRNGKey,
                     temperature: Union[JTensor, float], topk: int) -> JTensor:
  """Sample decode algorithm from TopK.

  Args:
    logits: Logits of current step. This is a JTensor of [batch_size *
      num_samples, vocab_size].
    prng_key: The prng key.
    temperature: Temperature of sampling decoding. It could be a float or
      a JTensor of shape [batch_size].
    topk: If nonzero, use top-k sampling, only selecting among the most likely k
      tokens at each step.

  Returns:
    A tensor of shape [batch_size * num_samples].
  """
  gumbel_noise_shape = list(logits.shape)
  gumbel_noise_shape[-1] = topk
  gumbel_noise = jax.random.gumbel(
      prng_key, shape=gumbel_noise_shape).astype(logits.dtype)
  return sample_from_topk_with_gumbel_noise(logits, gumbel_noise, temperature,
                                            topk)


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


def right_align_prefix_ids(prefix_ids: JTensor, prefix_lengths: JTensor,
                           paddings_dtype: jnp.dtype) -> JTensor:
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

  right_align_ids = decoder_utils.right_align_tensors(prefix_ids,
                                                      prefix_lengths)
  prefix_lengths = prefix_lengths[:, jnp.newaxis]
  prefix_iota = jax.lax.iota(dtype=jnp.int32, size=max_prefix_len)
  prefix_iota = prefix_iota - (max_prefix_len - prefix_lengths)
  prefix_paddings = jnp.where(prefix_iota < 0,
                              jnp.ones_like(prefix_iota, dtype=paddings_dtype),
                              jnp.zeros_like(prefix_iota, dtype=paddings_dtype))
  return right_align_ids, prefix_paddings


def top_p_mask_logits(logits: JTensor, p: Union[float, JTensor]) -> JTensor:
  """Adjust logits so that the smallest number of logits whose cumulative...

  sum of probs adds up to (at least) p. The other logits are masked with a
  large negative number.

  Args:
    logits: logits of shape [B, T].
    p: a scalar.

  Returns:
    The masked logits.
  """
  if not isinstance(p, JTensor) and p >= 1.0:
    return logits
  batch_size = logits.shape[0]
  # Ascending order. Cumsum will go through small numbers first, which is
  # more numerically stable.
  logits_sorted = jnp.sort(logits, axis=-1)
  sorted_cum_probs = jnp.cumsum(
      jax.nn.softmax(logits_sorted.astype(jnp.float32), axis=-1), axis=-1)
  cutoff_idx = jnp.sum((sorted_cum_probs <= 1.0 - p).astype(jnp.int32), axis=-1)
  cutoff_logit = logits_sorted[jnp.arange(batch_size), cutoff_idx]
  logits = jnp.where(logits < jnp.expand_dims(cutoff_logit, -1),
                     py_utils.get_large_negative_number(logits.dtype), logits)
  return logits


# TODO(b/249483164): Rename BaseLayerApi->BaseLayer after Fiddle migration.
def sample_decode(model: base_layer.BaseLayerApi,
                  extend_step_fn: decoder_utils.ExtendStepFn,
                  transform_state_fn: Optional[decoder_utils.TransformStateFn],
                  lazy_broadcast_prefix_fn: Optional[
                      decoder_utils.LazyBroadcastPrefixFn],
                  target_prefix_ids: JTensor,
                  target_prefix_paddings: JTensor,
                  seq_len: int,
                  num_samples: int,
                  k: int,
                  p: Optional[Union[float, JTensor]] = None,
                  cf_guidance_scale: Optional[Union[List[float], float]] = None,
                  fprop_for_prefix: bool = False,
                  temperature: Union[float, JTensor] = 1.0,
                  max_prefix_len: Optional[int] = None,
                  max_decode_steps: Optional[int] = None,
                  prefix_lengths: Optional[JTensor] = None,
                  eos_id: Optional[int] = None,
                  return_result_for_suffix_score: bool = False,
                  sort_samples: bool = True,
                  early_exit=True) -> NestedMap:
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
    target_prefix_ids: The token ids that correspond to the target sequence. A
      JTensor of shape [batch, target_sequence_length].
    target_prefix_paddings: The paddings corresponding to the target sequence,
      with a 1 denoting padding token and 0 denoting non-padding tokens. A
      JTensor of shape [batch, target_sequence_length].
    seq_len: The output sequence length to decode to. seq_len contains prefix.
    num_samples: Number of samples.
    k: If nonzero, use top-k sampling, only selecting among the most likely k
      tokens at each step.
    p: If not None, use the smallest number of logits whose cumulative sum of
      probs adds up to (at least) p.
    cf_guidance_scale: If not 1.0, apply classifier-free guidance for
      conditioned generation assuming the inputs are with [cond_a, uncond_a,
      cond_b, uncond_b, ...]. Before sampling, we modify logits as
      logits = uncond_logits + cf_guidance_scale * (cond_logits - uncond_logits)
      while after sampling, we force align sampled token ids of conditioned and
      unconditioned branch.
    fprop_for_prefix: Use one fprop for prefix.
    temperature: Temperature of sampling decoding. It could be a float or
      a JTensor of shape [B].
    max_prefix_len: Python int or None, the max prefix length for decoding.
    max_decode_steps: Python int or None, the max decode step to run after the
      prefix (if any). Since the prefixes might be of unequal lengths, this
      value is not equivalent with `seq_len` above. When None, decode steps is
      only limited by `seq_len` above.
    prefix_lengths: Optional argument supplying prefix sizes to initialize the
      model to decode from a certain target prefix for each position in the
      batch. This can either be None or a JTensor of shape [batch] signifying
      the prefix length for each sequence in the batch.
    eos_id: Optional EOS id which to terminate the decoding early.
    return_result_for_suffix_score: Whether or not to return result for suffix
      score.
    sort_samples: Whether to sort the samples by logprobs.
    early_exit: A bool, whether or not to allow early exit.

  Returns:
    A NestedMap with `.prefix_lengths` (indicating the lengths of prefixes for
    each target sequence), `.output_ids` (matrix of int ids with the
    decoded output), `.decode_lengths` (vector of ints indicating the lengths
    of non-padding tokens in `.output_ids`, which includes the prefix), and
    `.logprobs` (the log probability of selected tokens, including the prefix,
    where a positive value of 1.0 is used to indicate padded positions).
    The outputs has shape [batch, num_samples, ...].
  """
  original_batch_size = target_prefix_ids.shape[0]
  if num_samples > 1:
    # Broadcast inputs from [batch, ...] to [batch * num_samples, ...].
    # [a, b, c] and num_samples = 3 will have
    # [a, a, a, b, b, b, c, c, c]. If cf_guidance_scale is enabled, it will have
    # [cond_a, cond_a, cond_a, uncond_a, uncond_a, uncond_a, ...].
    target_prefix_ids = jnp.repeat(
        target_prefix_ids, axis=0, repeats=num_samples)
    target_prefix_paddings = jnp.repeat(
        target_prefix_paddings, axis=0, repeats=num_samples)
    prefix_lengths = jnp.repeat(prefix_lengths, axis=0, repeats=num_samples)

    # Broadcast temperature if it is a JTensor.
    if isinstance(temperature, JTensor):
      temperature_expected_shape = (original_batch_size,)
      if temperature.shape != temperature_expected_shape:  # pytype: disable=attribute-error
        raise ValueError('Dynamic temperature should have shape: '
                         f'{temperature_expected_shape}, but it has shape: '
                         f'{temperature.shape}.'  # pytype: disable=attribute-error
                        )
      temperature = jnp.repeat(temperature, axis=0, repeats=num_samples)
      temperature = jnp.reshape(temperature, (-1, 1))

    if lazy_broadcast_prefix_fn is not None:
      assert fprop_for_prefix

      # We need to exclude the last token from prefix, and instead move it to
      # the multi-sample suffix. This is because the last token only as an Input
      # ID, but not an output ID (label), and we need to start decoding from it.
      transform_state_fn(model, decoder_utils.slice_state_fn(0, -1))
      # max_decode_steps + 1 to include last token from prefix.
      lazy_broadcast_prefix_fn(model, num_samples, max_decode_steps + 1)
    else:
      # Broadcast prefix state for num_samples.
      transform_state_fn(model,
                         decoder_utils.batch_broadcast_state_fn(num_samples))

    # If cf guidance scale is a list floats with length == num_samples, we
    # convert it to the target shape to be used in decode loop_body.
    if isinstance(cf_guidance_scale, Sequence):
      assert len(cf_guidance_scale) == num_samples
      cf_guidance_scale = jnp.array(cf_guidance_scale)
      cf_guidance_scale = cf_guidance_scale[jnp.newaxis, :, jnp.newaxis]

  if seq_len <= 0:
    raise ValueError('The sequence length for decoding must be > 0, '
                     f'current value = {seq_len}.')
  max_decode_steps = max_decode_steps or seq_len
  batch_size = target_prefix_ids.shape[0]

  # If prefix length is not specified, set it to 0.
  if prefix_lengths is None:
    prefix_lengths = jnp.zeros([batch_size], dtype=jnp.int32)

  output_ids = jnp.zeros(shape=(batch_size, seq_len), dtype=jnp.int32)

  val = NestedMap()
  if fprop_for_prefix:
    # Update output_ids with prefix_ids.
    output_ids = jax.lax.dynamic_update_slice(output_ids, target_prefix_ids,
                                              [0] * output_ids.ndim)
    assert max_prefix_len is not None
    # Update loop init states with prefix.
    val.step = max_prefix_len - 1
    val.segment_pos = prefix_lengths - 1
  else:
    output_ids = output_ids.at[:, 0].set(target_prefix_ids[:, 0])
    val.step = 0
    val.segment_pos = jnp.zeros([batch_size], dtype=jnp.int32)

  val.output_ids = output_ids
  # Shape [batch_size], whether each row has terminated and should stop.
  val.done = jnp.zeros(shape=batch_size, dtype=jnp.bool_)
  val.has_eos = jnp.zeros(shape=batch_size, dtype=jnp.bool_)
  val.decode_lengths = jnp.ones_like(prefix_lengths) * seq_len
  # We use a positive value of 1.0 to indicate blank or padded positions.
  val.logprobs = jnp.ones_like(output_ids, dtype=jnp.float32)

  def cond_func(model, val):
    """Whether the while loop should continue."""
    del model
    # We continue the decoding search iff both:
    #   (1) We have yet to exceed the max steps set by p.decoder.seqlen, AND;
    #   (2) At least one row in the batch has not terminated.
    length_ok = val.step < seq_len - 1
    all_rows_done = jnp.all(val.done)
    return jnp.logical_and(length_ok, jnp.logical_not(all_rows_done))

  def loop_body(model, val):
    """From ids at `step`, update output ids at `step + 1`."""
    step = val.step
    logits = extend_step_fn(model, val.output_ids[:, step], val.segment_pos)
    logprobs = jax.nn.log_softmax(logits.astype(jnp.float32))
    if cf_guidance_scale is not None:
      # Split cond / uncond logits.
      logits_split = split_batch_dim(logits, 0, 2 * num_samples)
      cond_logits = logits_split[:, :num_samples]
      uncond_logits = logits_split[:, num_samples:]
      logits = uncond_logits + cf_guidance_scale * (cond_logits - uncond_logits)
      logits = jnp.reshape(logits, (-1,) + logits.shape[2:])
    if p is not None:
      logits = top_p_mask_logits(logits, p)
    if k > 1:
      new_ids = sample_from_topk(
          logits,
          model.next_prng_key(),
          temperature=temperature,
          topk=k)
    elif k == 0:
      if isinstance(temperature, JTensor) or temperature > 0.0:
        gumbel_noise = jax.random.gumbel(
            model.next_prng_key(), shape=logits.shape).astype(logits.dtype)
        logits += gumbel_noise * temperature
      new_ids = jnp.argmax(logits, axis=1)
    else:
      new_ids = jnp.argmax(logits, axis=1)

    model.add_summary('new_ids', new_ids)
    model.add_summary('sample_logits', logits)

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
      new_ids = jnp.where(step < prefix_lengths - 1,
                          target_prefix_ids[:, step + 1], new_ids)
    prev_done = val.done
    new_ids = jnp.where(prev_done, jnp.zeros_like(new_ids), new_ids)
    if eos_id is not None:
      val.done = jnp.logical_or(prev_done, jnp.equal(new_ids, eos_id))
      val.has_eos = jnp.logical_or(val.has_eos, jnp.equal(new_ids, eos_id))
    if fprop_for_prefix:
      prefix_offset = max_prefix_len
      decode_lengths = prefix_lengths + (step - max_prefix_len + 2)
    else:
      # if eos is part of prefix, ignore it.
      val.done = jnp.where(step < prefix_lengths - 1, prev_done, val.done)
      prefix_offset = prefix_lengths
      decode_lengths = jnp.ones_like(val.decode_lengths) * (step + 2)
    val.segment_pos += 1

    max_decoding_steps_reached = (jnp.ones_like(prefix_lengths) * (step + 2) -
                                  prefix_offset) >= max_decode_steps
    val.done = jnp.logical_or(val.done, max_decoding_steps_reached)
    done_at_this_step = jnp.logical_and(jnp.logical_not(prev_done), val.done)
    val.decode_lengths = jnp.where(done_at_this_step, decode_lengths,
                                   val.decode_lengths)
    val.output_ids = val.output_ids.at[:, step + 1].set(new_ids)
    logprobs_at_new_ids = logprobs.at[jnp.arange(batch_size), new_ids].get()
    logprobs_at_new_ids = jnp.where(prev_done,
                                    jnp.ones_like(logprobs_at_new_ids),
                                    logprobs_at_new_ids)
    val.logprobs = val.logprobs.at[:, step + 1].set(logprobs_at_new_ids)
    val.step += 1
    return val

  if early_exit:
    result = nn.while_loop(
        cond_func,
        loop_body,
        model,
        val,
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
        variable_axes={
            AUX_LOSS: 0,
            SUMMARIES: 0
        },
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

    dummy_inputs = {'dummy': jnp.zeros([seq_len, 2])}
    result, _ = scan_fn(model, val, dummy_inputs)

    # Now merge back the summaries.
    if model.is_mutable_collection(base_layer.SUMMARIES):
      # recursively merge two dictionaries.
      reinsert_collection(model, base_layer.SUMMARIES, model_summaries_copy)


  del result.segment_pos

  if return_result_for_suffix_score:
    return result

  result.prefix_lengths = prefix_lengths
  result.original_lengths = jnp.sum(
      1.0 - target_prefix_paddings, axis=1).astype(jnp.int32)

  if fprop_for_prefix:
    prefix_ids = decoder_utils.left_align_tensor(target_prefix_ids,
                                                 prefix_lengths, max_prefix_len)
  else:
    prefix_ids = target_prefix_ids
  # We manually pad out the ids not belonging to the prefix because some
  # tokenizers tested do not always obey the lengths arg.
  indices = jnp.tile(jnp.arange(prefix_ids.shape[1]), (prefix_ids.shape[0], 1))
  prefix_lengths_2d = jnp.tile(prefix_lengths[:, None],
                               (1, prefix_ids.shape[1]))
  prefix_ids = jnp.where(indices < prefix_lengths_2d, prefix_ids,
                         jnp.zeros_like(prefix_ids))
  result.prefix_ids = prefix_ids

  if fprop_for_prefix:
    # TODO(b/229679837): return decoding ids only and
    # remove left align logic here.

    # Change output_ids to left align.
    result.output_ids = decoder_utils.left_align_tensor(result.output_ids,
                                                        prefix_lengths,
                                                        max_prefix_len)
    result.logprobs = decoder_utils.left_align_tensor(result.logprobs,
                                                      prefix_lengths,
                                                      max_prefix_len)

  del result.step, result.done, result.has_eos

  if cf_guidance_scale is not None:
    # Split cond / uncond branches and only return conditioned branch.
    result = jax.tree_map(
        lambda x: split_batch_dim(x, 0, 2 * num_samples)[:, :num_samples],
        result)
  else:
    result = jax.tree_map(lambda x: split_batch_dim(x, 0, num_samples), result)
  if num_samples > 1 and sort_samples:
    return sort_samples_by_scores(result)
  return result


# TODO(b/249483164): Rename BaseLayerApi->BaseLayer after Fiddle migration.
def greedy_decode(
    model: base_layer.BaseLayerApi,
    extend_step_fn: decoder_utils.ExtendStepFn,
    target_ids: JTensor,
    target_paddings: JTensor,
    seq_len: int,
    fprop_for_prefix: bool = False,
    max_prefix_len: Optional[int] = None,
    max_decode_steps: Optional[int] = None,
    prefix_lengths: Optional[JTensor] = None,
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
    target_ids: The token ids that correspond to the target sequence.
    target_paddings: The paddings corresponding to the target sequence, with a 1
      denoting padding token and 0 denoting non-padding tokens.
    seq_len: The output sequence length to decode to. seq_len contains prefix.
    fprop_for_prefix: Use one fprop for prefix.
    max_prefix_len: Python int or None, the max prefix length for decoding.
    max_decode_steps: Python int or None, the max decode step to run after the
      prefix (if any). Since the prefixes might be of unequal lengths, this
      value is not equivalent with `seq_len` above. When None, decode steps is
      only limited by `seq_len` above.
    prefix_lengths: Optional argument supplying prefix sizes to initialize the
      model to decode from a certain target prefix for each position in the
      batch. This can either be None or a JTensor of shape [batch] signifying
      the prefix length for each sequence in the batch.
    eos_id: Optional EOS id which to terminate the decoding early.

  Returns:
    A NestedMap with `.prefix_lengths` (indicating the lengths of prefixes for
    each target sequence), `.output_ids` (matrix of int ids with the
    decoded output), `.decode_lengths` (vector of ints indicating the lengths
    of non-padding tokens in `.output_ids`, which includes the prefix), and
    `.logprobs` (the log probability of selected tokens, including the prefix,
    where a positive value of 1.0 is used to indicate padded positions).
  """

  return sample_decode(
      model,
      extend_step_fn,
      transform_state_fn=None,
      lazy_broadcast_prefix_fn=None,
      target_prefix_ids=target_ids,
      target_prefix_paddings=target_paddings,
      seq_len=seq_len,
      fprop_for_prefix=fprop_for_prefix,
      max_prefix_len=max_prefix_len,
      max_decode_steps=max_decode_steps,
      prefix_lengths=prefix_lengths,
      eos_id=eos_id,
      num_samples=1,
      k=1,
      temperature=0.0)
