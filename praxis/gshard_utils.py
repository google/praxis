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

"""Utilities to handle XLA sharding annotations."""

from absl import logging
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from praxis import pytypes

SplitDimsMapping = pytypes.SplitDimsMapping


def remove_dim(dim: int,
               split_dims_mapping: SplitDimsMapping) -> SplitDimsMapping:
  """Returns a copy of split_dims_mapping with dimension 'dim' removed."""
  if dim < 0:
    num_dims = len(split_dims_mapping)
    dim = num_dims + dim
  assert dim >= 0 and dim < len(split_dims_mapping)
  return list(split_dims_mapping[:dim]) + list(split_dims_mapping[dim + 1:])


def cum_sum(elements, axis=0, exclusive=False, reverse=False):
  """Same as jax.np.cumsum but with the extra options from tf.cumsum.

  Args:
    elements: A Jax array. The cumulative sum is computed along the 'axis'
      dimension.
    axis: The axis to compute the cumsum over.
    exclusive: If True, perform exclusive cumsum. With exclusive=False:
      cumsum([a, b, c]) --> [a, a + b, a + b + c] With exclusive=True:
      cumprod([a, b, c]) --> [0, a, a + b]
    reverse: A bool (default: False), perform the cumulative sum in reverse.

  Returns:
    The cumulative sum.
  """
  if reverse:
    elements = jnp.flip(elements, axis=axis)

  result = jnp.cumsum(elements, axis=axis)
  if exclusive:
    result = result - elements
  if reverse:
    return jnp.flip(result, axis=axis)
  else:
    return result


def _create_over_capacity_ratio_summary(mask, position_in_expert, capacity,
                                        name):
  _ = name  # TODO(lepikhin): consider inlined summary
  masked_position_in_expert = mask * position_in_expert
  ge_capacity = jnp.greater_equal(masked_position_in_expert, capacity)
  over_capacity = jnp.sum(ge_capacity).astype(jnp.float32)
  denom = jnp.sum(mask).astype(jnp.float32)
  over_capacity_ratio = over_capacity / jnp.maximum(
      jnp.array(1.0, dtype=jnp.float32), denom)
  return over_capacity_ratio


# Close adaptation of the original gshard_layers.Top2GatingOnLogits TPU-specific
# implementation of the Algorithm 2 from the http://arxiv.org/abs/2006.16668
#
# TODO(lepikhin): convert einsums to use ... (ellipses) for HMoE support similar
# to original TF implementation.
#
# Historic unused arguments are removed:
#   - inputs: unused, since gating logits are supplied explicitly.
#   - use_xla_sharding: historic option, specifically for 1D sharding by E
#   - num_devices: historic option, specifically for 1D sharding by E
def top2_gating_on_logits(paddings,
                          logits,
                          experts_dim,
                          expert_capacity_dim,
                          fprop_dtype,
                          prng_key=None,
                          second_expert_policy='all',
                          second_expert_threshold=0.0,
                          legacy_mtf_behavior=True,
                          capacity_factor=None,
                          importance=None,
                          mask_dtype=jnp.int32,
                          gating_logit_cap=0.0):
  """Computes Top-2 gating for Mixture-of-Experts.

  This function takes gating logits, potentially sharded across tpu cores as
  inputs. We rely on sharding propagation to work universally with 1D and 2D
  sharding cases. Dispatch and combine tensors should be explicitly annotated
  with jax.with_sharding_constraint by the caller.

  We perform dispatch/combine via einsum.

  Dimensions:

    G: group dim
    S: group size dim
    E: number of experts
    C: capacity per expert
    M: model_dim (same as input_dim and output_dim as in FF layer)
    B: original batch dim
    L: original seq len dim

  Note that for local_dispatch, the original batch BLM is reshaped to GSM, each
  group `g = 0..G-1` is being dispatched independently.

  Args:
    paddings: G`S tensor.
    logits: G`SE tensor.
    experts_dim: number of experts
    expert_capacity_dim: number of examples per minibatch/group per expert. Each
      example is typically a vector of size input_dim, representing embedded
      token or an element of Transformer layer output.
    fprop_dtype: activation dtype
    prng_key: jax.random.PRNGKey used for randomness.
    second_expert_policy: 'all', 'sampling' or 'random' - 'all': we greedily
      pick the 2nd expert - 'sampling': we sample the 2nd expert from the
      softmax - 'random': we optionally randomize dispatch to second-best expert
      in proportional to (weight / second_expert_threshold).
    second_expert_threshold: threshold for probability normalization when
      second_expert_policy == 'random'
    legacy_mtf_behavior: bool, True if to match legacy mtf behavior exactly.
    capacity_factor: if set, increases expert_capacity_dim to at least
      (group_size * capacity_factor) / experts_dim
    importance: input importance weights for routing (G`S tensor or None)
    mask_dtype: using bfloat16 for fprop_dtype could be problematic for mask
      tensors, mask_dtype overrides dtype for such tensors
    gating_logit_cap: soft cap, applied for gating logits, this is a stability
      fix to avoid extreme values during initial steps. Defaults to 50.0.

  Returns:
    A tuple (aux_loss, combine_tensor, dispatch_tensor, over_capacity ratios).

    - aux_loss: auxiliary loss, for equalizing the expert assignment ratios.
    - combine_tensor: a G`SEC tensor for combining expert outputs.
    - dispatch_tensor: a G`SEC tensor, scattering/dispatching inputs to experts.
    - over_capacity ratios: tuple that represents the ratio of tokens that
      were not dispatched due to lack of capcity for top_1 and top_2 expert
      respectively, e.g. (over_capacity_1, over_capacity_2)
  """
  assert (capacity_factor or expert_capacity_dim)
  if mask_dtype is None:
    assert fprop_dtype != jnp.bfloat16, 'Using bfloat16 for mask is an error.'
    mask_dtype = fprop_dtype

  if logits.dtype != jnp.float32:
    logging.info('Upcasting gating logits')
    logits = logits.astype(jnp.float32)

  def _cap_logits(logits):
    if gating_logit_cap > 0.0:
      logging.info('gating_logit_cap: %f', gating_logit_cap)
      cap = jnp.array(gating_logit_cap, dtype=logits.dtype)
      logits = cap * jnp.tanh(logits / cap)
    return logits

  logits = _cap_logits(logits)

  raw_gates = jax.nn.softmax(logits, axis=-1)  # along E dim

  if capacity_factor:
    # Determine expert capacity automatically depending on the input size
    group_size_dim = logits.shape[1]
    auto_expert_capacity = int(group_size_dim * capacity_factor / experts_dim)
    if expert_capacity_dim is None:
      expert_capacity_dim = 1
    if expert_capacity_dim < auto_expert_capacity:
      expert_capacity_dim = auto_expert_capacity
      # Round up to a multiple of 4 to avoid possible padding.
      while expert_capacity_dim % 4:
        expert_capacity_dim += 1
      logging.info(
          'Setting expert_capacity_dim=%r (capacity_factor=%r '
          'group_size_dim=%r experts_dim=%r)', expert_capacity_dim,
          capacity_factor, group_size_dim, experts_dim)

  capacity = jnp.array(expert_capacity_dim, dtype=jnp.int32)

  # top-1 index: GS tensor
  index_1 = jnp.argmax(raw_gates, axis=-1)

  # GSE
  mask_1 = jax.nn.one_hot(index_1, experts_dim, dtype=mask_dtype)
  density_1_proxy = raw_gates

  if importance is not None:
    importance_is_one = jnp.equal(importance, 1.0)
    mask_1 *= jnp.expand_dims(importance_is_one.astype(mask_1.dtype), -1)
    density_1_proxy *= jnp.expand_dims(
        importance_is_one.astype(density_1_proxy.dtype), -1)
  else:
    assert len(mask_1.shape) == 3
    importance = jnp.ones_like(mask_1[:, :, 0]).astype(fprop_dtype)
    if paddings is not None:
      nonpaddings = 1.0 - paddings
      mask_1 *= jnp.expand_dims(nonpaddings.astype(mask_1.dtype), -1)
      density_1_proxy *= jnp.expand_dims(
          nonpaddings.astype(density_1_proxy.dtype), -1)
      importance = nonpaddings

  gate_1 = jnp.einsum('GSE,GSE->GS', raw_gates, mask_1.astype(raw_gates.dtype))
  gates_without_top_1 = raw_gates * (1.0 - mask_1.astype(raw_gates.dtype))

  if second_expert_policy == 'sampling':
    # We directly sample the 2nd expert index from the softmax over of the 2nd
    # expert by getting rid of the 1st expert already selected above. To do so,
    # we set a very negative value to the logit corresponding to the 1st expert.
    # Then we sample from the softmax distribution using the Gumbel max trick.
    prng_key, subkey = jax.random.split(prng_key)
    noise = jax.random.uniform(subkey, logits.shape, dtype=logits.dtype)
    # Generates standard Gumbel(0, 1) noise, GSE tensor.
    noise = -jnp.log(-jnp.log(noise))
    very_negative_logits = jnp.ones_like(logits) * (-0.7) * np.finfo(
        logits.dtype).max
    # Get rid of the first expert by setting its logit to be very negative.
    updated_logits = jnp.where(mask_1 > 0.0, very_negative_logits, logits)
    # Add Gumbel noise to the updated logits.
    noised_logits = updated_logits + noise
    # Pick the index of the largest noised logits as the 2nd expert. This is
    # equivalent to sampling from the softmax over the 2nd expert.
    index_2 = jnp.argmax(noised_logits, axis=-1)
  else:
    # Greedily pick the 2nd expert.
    index_2 = jnp.argmax(gates_without_top_1, axis=-1)

  mask_2 = jax.nn.one_hot(index_2, experts_dim, dtype=mask_dtype)
  if paddings is not None:
    importance_is_nonzero = importance > 0.0
    mask_2 *= jnp.expand_dims(importance_is_nonzero.astype(mask_2.dtype), -1)
  gate_2 = jnp.einsum('GSE,GSE->GS', gates_without_top_1,
                      mask_2.astype(gates_without_top_1.dtype))

  # See notes in lingvo/core/gshard_layers.py.
  if legacy_mtf_behavior:
    # Renormalize.
    denom = gate_1 + gate_2 + 1e-9
    gate_1 /= denom
    gate_2 /= denom

  # We reshape the mask as [X*S, E], and compute cumulative sums of assignment
  # indicators for each expert index e \in 0..E-1 independently.
  # First occurrence of assignment indicator is excluded, see exclusive=True
  # flag below.
  # cumsum over S dim: mask_1 is GSE tensor.
  position_in_expert_1 = cum_sum(mask_1, exclusive=True, axis=-2)

  # GE tensor (reduce S out of GSE tensor mask_1).
  # density_1[:, e] represents assignment ration (num assigned / total) to
  # expert e as top_1 expert without taking capacity into account.
  assert importance.dtype == fprop_dtype
  if legacy_mtf_behavior:
    density_denom = jnp.asarray(1.0, dtype=jnp.float32)
  else:
    density_denom = jnp.mean(
        importance, axis=1, dtype=jnp.float32)[:, jnp.newaxis] + 1e-6
  density_1 = jnp.mean(mask_1.astype(jnp.float32), axis=-2) / density_denom
  # density_1_proxy[:, e] represents mean of raw_gates for expert e, including
  # those of examples not assigned to e with top_k
  density_1_proxy = jnp.mean(
      density_1_proxy, axis=-2, dtype=jnp.float32) / density_denom

  # Compute aux_loss
  aux_loss = jnp.mean(
      density_1_proxy * density_1, dtype=jnp.float32)  # element-wise
  aux_loss *= (experts_dim * experts_dim)  # const coefficients

  # Add the over capacity ratio for expert 1
  over_capacity_1 = _create_over_capacity_ratio_summary(mask_1,
                                                        position_in_expert_1,
                                                        capacity,
                                                        'over_capacity_1')

  mask_1 *= jnp.less(position_in_expert_1,
                     expert_capacity_dim).astype(mask_1.dtype)
  position_in_expert_1 = jnp.einsum('GSE,GSE->GS', position_in_expert_1, mask_1)

  # How many examples in this sequence go to this expert?
  mask_1_count = jnp.einsum('GSE->GE', mask_1)
  # [batch, group] - mostly ones, but zeros where something didn't fit.
  mask_1_flat = jnp.sum(mask_1, axis=-1, dtype=mask_dtype)
  assert mask_1_count.dtype == mask_dtype
  assert mask_1_flat.dtype == mask_dtype

  if second_expert_policy == 'all' or second_expert_policy == 'sampling':
    pass
  else:
    assert second_expert_policy == 'random'
    # gate_2 is between 0 and 1, reminder:
    #
    #   raw_gates = jax.nn.softmax(logits)
    #   index_1 = jnp.argmax(raw_gates, axis=-1)
    #   mask_1 = jax.nn.one_hot(index_1, experts_dim, dtpe=fprop_dtype)
    #   gate_1 = jnp.einsum(`GSE,GSE->GS', raw_gates, mask_1)
    #
    # e.g., if gate_2 exceeds second_expert_threshold, then we definitely
    # dispatch to second-best expert. Otherwise, we dispatch with probability
    # proportional to (gate_2 / threshold).
    #
    prng_key, subkey = jax.random.split(prng_key)
    sampled_2 = jnp.less(
        jax.random.uniform(subkey, gate_2.shape, dtype=gate_2.dtype),
        gate_2 / max(second_expert_threshold, 1e-9))
    gate_2 *= sampled_2.astype(gate_2.dtype)
    mask_2 *= jnp.expand_dims(sampled_2, -1).astype(mask_2.dtype)

  position_in_expert_2 = cum_sum(
      mask_2, exclusive=True, axis=-2) + jnp.expand_dims(mask_1_count, -2)
  over_capacity_2 = _create_over_capacity_ratio_summary(mask_2,
                                                        position_in_expert_2,
                                                        capacity,
                                                        'over_capacity_2')

  mask_2 *= jnp.less(position_in_expert_2,
                     expert_capacity_dim).astype(mask_2.dtype)
  position_in_expert_2 = jnp.einsum('GSE,GSE->GS', position_in_expert_2, mask_2)
  mask_2_flat = jnp.sum(mask_2, axis=-1, dtype=mask_dtype)

  gate_1 *= mask_1_flat.astype(gate_1.dtype)
  gate_2 *= mask_2_flat.astype(gate_2.dtype)

  if not legacy_mtf_behavior:
    denom = gate_1 + gate_2
    # To avoid divide by 0.
    denom = jnp.where(denom > 0, denom, jnp.ones_like(denom))
    gate_1 /= denom
    gate_2 /= denom

  # GSC tensor
  b = jax.nn.one_hot(
      position_in_expert_1.astype(np.int32),
      expert_capacity_dim,
      dtype=jnp.float32)
  # GSE tensor
  a = jnp.expand_dims(
      gate_1 * mask_1_flat.astype(jnp.float32), axis=-1) * jax.nn.one_hot(
          index_1, experts_dim, dtype=jnp.float32)
  # GSEC tensor
  first_part_of_combine_tensor = jnp.einsum('GSE,GSC->GSEC', a, b)

  # GSC tensor
  b = jax.nn.one_hot(
      position_in_expert_2.astype(np.int32),
      expert_capacity_dim,
      dtype=jnp.float32)
  # GSE tensor
  a = jnp.expand_dims(
      gate_2 * mask_2_flat.astype(fprop_dtype), axis=-1) * jax.nn.one_hot(
          index_2, experts_dim, dtype=jnp.float32)
  second_part_of_combine_tensor = jnp.einsum('GSE,GSC->GSEC', a, b)

  # GSEC tensor
  combine_tensor = first_part_of_combine_tensor + second_part_of_combine_tensor

  # GSEC tensor
  dispatch_tensor = combine_tensor.astype(bool).astype(fprop_dtype)

  aux_loss = aux_loss.astype(fprop_dtype)
  combine_tensor = combine_tensor.astype(fprop_dtype)
  dispatch_tensor = dispatch_tensor.astype(fprop_dtype)

  return aux_loss, combine_tensor, dispatch_tensor, (over_capacity_1,
                                                     over_capacity_2)


def topk_gating_on_logits(paddings,
                          logits,
                          experts_dim,
                          expert_capacity_dim,
                          fprop_dtype,
                          capacity_factor=None,
                          mask_dtype=jnp.int32,
                          k=2,
                          gating_logit_cap=0.0):
  """Computes Top-k gating for Mixture-of-Experts.

  This function takes gating logits, potentially sharded across tpu cores as
  inputs. We rely on sharding propagation to work universally with 1D and 2D
  sharding cases. Dispatch and combine tensors should be explicitly annotated
  with jax.with_sharding_constraint by the caller.

  We perform dispatch/combine via einsum.

  Dimensions:

    G: group dim
    S: group size dim
    E: number of experts
    C: capacity per expert
    M: model_dim (same as input_dim and output_dim as in FF layer)
    B: original batch dim
    L: original seq len dim

  Note that for local_dispatch, the original batch BLM is reshaped to GSM, each
  group `g = 0..G-1` is being dispatched independently.

  Args:
    paddings: G`S tensor.
    logits: G`SE tensor.
    experts_dim: number of experts
    expert_capacity_dim: number of examples per minibatch/group per expert. Each
      example is typically a vector of size input_dim, representing embedded
      token or an element of Transformer layer output.
    fprop_dtype: activation dtype
    capacity_factor: if set, increases expert_capacity_dim to at least
      (group_size * capacity_factor) / experts_dim
    mask_dtype: using bfloat16 for fprop_dtype could be problematic for mask
      tensors, mask_dtype overrides dtype for such tensors
    k: number of activated experts per prediction in each MoE layer
    gating_logit_cap: soft cap, applied for gating logits, this is a stability
      fix to avoid extreme values during initial steps. Defaults to 50.0.

  Returns:
    A tuple (aux_loss, combine_tensor, dispatch_tensor, over_capacity ratios).

    - aux_loss: auxiliary loss, for equalizing the expert assignment ratios.
    - combine_tensor: a G`SEC tensor for combining expert outputs.
    - dispatch_tensor: a G`SEC tensor, scattering/dispatching inputs to experts.
    - over_capacity ratios: tuple that represents the ratio of tokens that
      were not dispatched due to lack of capcity for top_1 and top_2 expert
      respectively, e.g. (over_capacity_1, over_capacity_2)
  """
  assert (capacity_factor or expert_capacity_dim)
  if mask_dtype is None:
    assert fprop_dtype != jnp.bfloat16, 'Using bfloat16 for mask is an error.'
    mask_dtype = fprop_dtype

  if logits.dtype != jnp.float32:
    logging.info('Upcasting gating logits')
    logits = logits.astype(jnp.float32)

  def _cap_logits(logits):
    if gating_logit_cap > 0.0:
      logging.info('gating_logit_cap: %f', gating_logit_cap)
      cap = jnp.array(gating_logit_cap, dtype=logits.dtype)
      logits = cap * jnp.tanh(logits / cap)
    return logits

  logits = _cap_logits(logits)

  # GSE
  raw_gates = jax.nn.softmax(logits, axis=-1)  # along E dim

  if capacity_factor is not None:
    # Determine expert capacity automatically depending on the input size
    group_size_dim = logits.shape[1]
    auto_expert_capacity = int(group_size_dim * capacity_factor / experts_dim)
    if expert_capacity_dim < auto_expert_capacity:
      expert_capacity_dim = auto_expert_capacity
      # Round up to a multiple of 4 to avoid possible padding.
      while expert_capacity_dim % 4:
        expert_capacity_dim += 1
      logging.info(
          'Setting expert_capacity_dim=%r (capacity_factor=%r '
          'group_size_dim=%r experts_dim=%r)', expert_capacity_dim,
          capacity_factor, group_size_dim, experts_dim)

  capacity = jnp.array(expert_capacity_dim, dtype=jnp.int32)

  # top-1 index: GS tensor
  index_1 = jnp.argmax(raw_gates, axis=-1)
  # GSE
  mask_1 = jax.nn.one_hot(index_1, experts_dim, dtype=mask_dtype)
  density_1_proxy = raw_gates
  assert len(mask_1.shape) == 3

  if paddings is not None:
    nonpaddings = 1.0 - paddings
    mask_1 *= jnp.expand_dims(nonpaddings.astype(mask_1.dtype), -1)
    density_1_proxy *= jnp.expand_dims(
        nonpaddings.astype(density_1_proxy.dtype), -1)

  density_1 = jnp.mean(mask_1.astype(jnp.float32), axis=-2)
  # density_1_proxy[:, e] represents mean of raw_gates for expert e, including
  # those of examples not assigned to e with top_k
  density_1_proxy = jnp.mean(density_1_proxy, axis=-2, dtype=jnp.float32)

  # Compute aux_loss
  aux_loss = jnp.mean(
      density_1_proxy * density_1, dtype=jnp.float32)  # element-wise
  aux_loss *= (experts_dim * experts_dim)  # const coefficients

  gate_1 = jnp.einsum('GSE,GSE->GS', raw_gates, mask_1.astype(raw_gates.dtype))
  gates_list = [gate_1]  # GS
  index_list = [index_1]  # GS
  masks_list = [mask_1]  # GSE
  raw_gates_i = raw_gates  # GSE

  if k > 1:
    denom = gate_1 + 1e-9
  else:
    denom = 1.0

  for i in range(1, k):
    # Gates without the selected value from the last step
    raw_gates_i *= (1.0 - masks_list[i - 1].astype(raw_gates_i.dtype))
    # Greedily pick the current expert
    index_i = jnp.argmax(raw_gates_i, axis=-1)
    mask_i = jax.nn.one_hot(index_i, experts_dim, dtype=mask_dtype)
    if paddings is not None:
      nonpaddings = 1.0 - paddings
      mask_i *= jnp.expand_dims(nonpaddings.astype(mask_i.dtype), -1)
    gate_i = jnp.einsum('GSE,GSE->GS', raw_gates_i,
                        mask_i.astype(raw_gates_i.dtype))
    denom += gate_i
    gates_list.append(gate_i)
    masks_list.append(mask_i)
    index_list.append(index_i)

  # Renormalize.
  gates_list = [x / denom for x in gates_list]

  # Compute cumulative sums of assignment GSE
  # indicators for each expert index e \in 0..E-1 independently.
  # First occurrence of assignment indicator is excluded, see exclusive=True
  # flag below.
  # cumsum over S dim: mask_1 is GSE tensor.
  position_in_expert_1 = cum_sum(masks_list[0], exclusive=True, axis=-2)
  # Add the over capacity ratio for expert 1
  over_capacity_list = [
      _create_over_capacity_ratio_summary(masks_list[0], position_in_expert_1,
                                          capacity, 'over_capacity_1')
  ]
  # Filter valid positions for top 1 selection
  masks_list[0] *= jnp.less(position_in_expert_1,
                            expert_capacity_dim).astype(masks_list[0].dtype)
  position_in_expert_1 = jnp.einsum('GSE,GSE->GS', position_in_expert_1,
                                    masks_list[0])

  # How many examples in this sequence go to this expert?
  mask_1_count = jnp.einsum('GSE->GE', masks_list[0])
  # GS - mostly ones, but zeros where something didn't fit.
  mask_1_flat = jnp.sum(masks_list[0], axis=-1, dtype=mask_dtype)
  position_in_expert_list = [position_in_expert_1]
  mask_i_flat_list = [mask_1_flat]
  mask_count_all = mask_1_count

  for i in range(1, k):
    position_in_expert_i = cum_sum(
        masks_list[i], exclusive=True, axis=-2) + jnp.expand_dims(
            mask_count_all, -2)
    # Add the over capacity ratio for expert 1
    over_capacity_list.append(
        _create_over_capacity_ratio_summary(masks_list[i], position_in_expert_i,
                                            capacity, f'over_capacity_{i+1}'))
    # Filter invalid positions for top i selection
    masks_list[i] *= jnp.less(position_in_expert_i,
                              expert_capacity_dim).astype(masks_list[i].dtype)
    # How many examples in this sequence go to this expert?
    mask_count_all += jnp.einsum('GSE->GE', masks_list[i])
    position_in_expert_i = jnp.einsum('GSE,GSE->GS', position_in_expert_i,
                                      masks_list[i])
    position_in_expert_list.append(position_in_expert_i)
    mask_i_flat_list.append(jnp.sum(masks_list[i], axis=-1, dtype=mask_dtype))

  combine_tensor = jnp.zeros(
      [logits.shape[0], logits.shape[1], experts_dim,
       expert_capacity_dim], dtype=jnp.float32)
  for gate_i, index_i, position_in_expert_i, mask_i_flat in zip(
      gates_list, index_list, position_in_expert_list, mask_i_flat_list):
    #  GS Filter valid gate values
    gate_i *= mask_i_flat.astype(gate_i.dtype)
    # GSC
    b = jax.nn.one_hot(
        position_in_expert_i.astype(np.int32),
        expert_capacity_dim,
        dtype=jnp.float32)
    # GSE
    a = jnp.expand_dims(
        gate_i * mask_i_flat.astype(jnp.float32), axis=-1) * jax.nn.one_hot(
            index_i, experts_dim, dtype=jnp.float32)
    combine_tensor += jnp.einsum('GSE,GSC->GSEC', a, b)

  # GSEC tensor
  aux_loss = aux_loss.astype(fprop_dtype)
  combine_tensor = combine_tensor.astype(fprop_dtype)
  dispatch_tensor = combine_tensor.astype(bool).astype(fprop_dtype)

  return aux_loss, combine_tensor, dispatch_tensor, over_capacity_list


def compute_gating(paddings,
                   logits,
                   experts_dim,
                   expert_capacity_dim,
                   fprop_dtype,
                   gating_func='top2',
                   prng_key=None,
                   second_expert_policy='all',
                   second_expert_threshold=0.0,
                   legacy_mtf_behavior=True,
                   capacity_factor=None,
                   mask_dtype=jnp.int32,
                   gating_logit_cap=0.0):
  """Compute gating."""
  if gating_func == 'top2':
    gating = top2_gating_on_logits(
        paddings=paddings,
        logits=logits.astype(jnp.float32),
        experts_dim=experts_dim,
        expert_capacity_dim=expert_capacity_dim,
        fprop_dtype=fprop_dtype,
        # TODO(zhangqiaorjc): Consider a different prng key stream.
        prng_key=prng_key,
        second_expert_policy=second_expert_policy,
        second_expert_threshold=second_expert_threshold,
        # legacy_mtf_behavior=True doesn't normalize gates when one expert is
        # being dropped. This is more appropriate for routing decisions like
        # 'random'.
        legacy_mtf_behavior=legacy_mtf_behavior,
        # *2.0 because we choose top-2 experts per example
        capacity_factor=capacity_factor,
        mask_dtype=mask_dtype,
        gating_logit_cap=gating_logit_cap)
  elif gating_func == 'expert_choice':
    gating = expert_choice_gating_on_logits(
        logits=logits.astype(jnp.float32),
        experts_dim=experts_dim,
        expert_capacity_dim=expert_capacity_dim,
        fprop_dtype=fprop_dtype,
        capacity_factor=capacity_factor,
        mask_dtype=jnp.int32)
  elif gating_func == 'expert_choice_v2':
    gating = expert_choice_gating_on_logits_v2(
        paddings=paddings,
        logits=logits.astype(jnp.float32),
        experts_dim=experts_dim,
        expert_capacity_dim=expert_capacity_dim,
        fprop_dtype=fprop_dtype,
        capacity_factor=capacity_factor,
        mask_dtype=jnp.int32,
        gating_logit_cap=0.0)
  else:
    raise ValueError('Unsupported gating function type: %s' % gating_func)
  return gating


def top_k(x, k):
  """Select the top k slices from the last dimension."""
  bcast_idxs = jnp.broadcast_to(np.arange(x.shape[-1]), x.shape)
  sorted_vals, sorted_idxs = lax.sort_key_val(x, bcast_idxs)
  # TODO(levskaya): use lax.slice here instead to benefit from XLA optimization
  return sorted_vals[..., -k:], sorted_idxs[..., -k:]


def expert_choice_gating_on_logits(logits,
                                   experts_dim,
                                   expert_capacity_dim,
                                   fprop_dtype,
                                   capacity_factor=None,
                                   mask_dtype=jnp.int32):
  """Computes Expert Choice Gating for Mixture-of-Experts.

  Paper Link: https://arxiv.org/abs/2202.09368

  This function takes gating logits, potentially sharded across tpu cores as
  inputs. We rely on sharding propagation to work universally with 1D and 2D
  sharding cases. Dispatch and combine tensors should be explicitly annotated
  with jax.with_sharding_constraint by the caller.

  We perform dispatch/combine via einsum.

  Dimensions:

    G: group dim
    S: group size dim
    E: number of experts
    C: capacity per expert
    M: model_dim (same as input_dim and output_dim as in FF layer)
    B: original batch dim
    L: original seq len dim

  Note that for local_dispatch, the original batch BLM is reshaped to GSM, each
  group `g = 0..G-1` is being dispatched independently.

  Args:
    logits: G`SE tensor.
    experts_dim: number of experts
    expert_capacity_dim: number of examples per minibatch/group per expert. Each
      example is typically a vector of size input_dim, representing embedded
      token or an element of Transformer layer output.
    fprop_dtype: activation dtype
    capacity_factor: if set, increases expert_capacity_dim to at least
      (group_size * capacity_factor) / experts_dim
    mask_dtype: using bfloat16 for fprop_dtype could be problematic for mask
      tensors, mask_dtype overrides dtype for such tensors

  Returns:
    A tuple (aux_loss, combine_tensor, dispatch_tensor, over_capacity ratios).

    - aux_loss: auxiliary loss, for equalizing the expert assignment ratios.
    - combine_tensor: a G`SEC tensor for combining expert outputs.
    - dispatch_tensor: a G`SEC tensor, scattering/dispatching inputs to experts.
    - over_capacity ratios: tuple that represents the ratio of tokens that
      were not dispatched due to lack of capcity for top_1 and top_2 expert
      respectively, e.g. (over_capacity_1, over_capacity_2)
  """
  assert (capacity_factor or expert_capacity_dim)
  if mask_dtype is None:
    assert fprop_dtype != jnp.bfloat16, 'Using bfloat16 for mask is an error.'
    mask_dtype = fprop_dtype

  if logits.dtype != jnp.float32:
    logging.info('Upcasting gating logits')
    logits = logits.astype(jnp.float32)

  raw_gates = jax.nn.softmax(logits, axis=-1)  # along E dim
  if raw_gates.dtype != fprop_dtype:
    raw_gates = raw_gates.astype(fprop_dtype)

  seq_len = logits.shape[1]
  bucket_size = int(logits.shape[1] // experts_dim * capacity_factor)
  # assert bucket_size <= expert_capacity_dim

  # GEC
  gate, indices = top_k(jnp.transpose(raw_gates, [0, 2, 1]), k=bucket_size)
  # GECS
  perm = jax.nn.one_hot(indices, seq_len, dtype=mask_dtype)

  return jnp.asarray(0.0, dtype=fprop_dtype), gate, perm


def expert_choice_gating_on_logits_v2(paddings,
                                      logits,
                                      experts_dim,
                                      expert_capacity_dim,
                                      fprop_dtype,
                                      capacity_factor=None,
                                      mask_dtype=jnp.int32,
                                      gating_logit_cap=0.0):
  """Compute gating with expert choice.

  There are two expected usages of this function:

  Compared to 'expert_choice_gating_on_logits', this function

  (1) selects the tokens directly based on the raw logits for each expert so
  that the padded tokens will not affect the selection of nonpadded tokens.
  (2) and dispatches only the nonpadded tokens to the respective positions of
  the selected experts.

  Dimensions cheat sheet::

    G: group_dim
    S: group_size_dim
    E: number of experts
    C: capacity per expert
    M: model_dim (same as input_dim, same as output_dim)
    B: original batch_dim
    L: original sequence_length_dim

  Note that for local_dispatch original batch BLM is reshaped into GSM, each
  group `g = 0...G-1` is being dispatched independently.

  Args:
    paddings: G`S tensor.
    logits: G`SE Tensor.
    experts_dim: number of experts.
    expert_capacity_dim: number of examples per minibatch(group) per expert.
      Each example is typically a vector of size input_dim, representing
      embedded token or an element of Transformer layer output.
    fprop_dtype: activations datatype to use.
    capacity_factor: if set, increases expert_capacity_dim to at least
      (group_size * capacity_factor) / experts_dim
    mask_dtype: dtype for mask tensors as bfloat16 could be problematic.
    gating_logit_cap: soft cap, applied for gating logits, this is a stability
      fix to avoid extreme values during initial steps. Defaults to 50.0.

  Returns:
    A tuple (aux_loss, combine_tensor, dispatch_tensor).
    - aux_loss: Always 0, because we don't need an aux_loss in this method.
    - combine_tensor: G`EC Tensor for combining expert outputs.
    - dispatch_tensor: G`ECS Tensor, scattering/dispatching inputs to experts.
  """

  assert (capacity_factor or expert_capacity_dim)
  if mask_dtype is None:
    assert fprop_dtype != jnp.bfloat16, 'Using bfloat16 for mask is an error.'
    mask_dtype = fprop_dtype

  if logits.dtype != jnp.float32:
    logging.info('Upcasting gating logits')
    logits = logits.astype(jnp.float32)

  def _cap_logits(logits):
    if gating_logit_cap > 0.0:
      logging.info('gating_logit_cap: %f', gating_logit_cap)
      cap = jnp.array(gating_logit_cap, dtype=logits.dtype)
      logits = cap * jnp.tanh(logits / cap)
    return logits

  logits = _cap_logits(logits)

  # GSE
  gates = jax.nn.softmax(logits, axis=-1)  # along E dim

  token_scores = gates
  nonpaddings = jnp.ones([logits.shape[0], logits.shape[1]], dtype=logits.dtype)
  if paddings is not None:
    # GS
    nonpaddings = 1.0 - paddings
    token_scores *= jnp.expand_dims(nonpaddings.astype(logits.dtype), -1)

  # GES
  token_scores = jnp.transpose(token_scores, [0, 2, 1])

  group_size_dim = int(logits.shape[1])

  if capacity_factor is not None:
    # Determine expert capacity automatically depending on the input size
     # Determine expert capacity automatically depending on the input size
    group_size_dim = logits.shape[1]
    auto_expert_capacity = int(group_size_dim * capacity_factor / experts_dim)
    if expert_capacity_dim < auto_expert_capacity:
      expert_capacity_dim = auto_expert_capacity
      # Round up to a multiple of 4 to avoid possible padding.
      while expert_capacity_dim % 4:
        expert_capacity_dim += 1
      logging.info(
          'Setting expert_capacity_dim=%r (capacity_factor=%r '
          'group_size_dim=%r experts_dim=%r)', expert_capacity_dim,
          capacity_factor, group_size_dim, experts_dim)
  # GEC
  _, token_indices = top_k(token_scores, k=expert_capacity_dim)
  # GECS
  mask = jax.nn.one_hot(token_indices, group_size_dim, dtype=mask_dtype)
  # GESC
  mask = jnp.transpose(mask, [0, 1, 3, 2])
  # GES records which expert selects which tokens.
  mask = jnp.sum(mask, axis=-1)
  # GSE records which experts are selected by each token.
  mask = jnp.transpose(mask, [0, 2, 1])
  # Filtered out the padded positions
  mask *= jnp.expand_dims(nonpaddings.astype(mask.dtype), -1)
  # GSE - Indices of the selected expert per token
  expert_indices = jnp.reshape(jnp.arange(experts_dim),
                               [1, 1, experts_dim]).astype(mask.dtype) * mask

  # GSE
  position_in_expert = cum_sum(mask, exclusive=True, axis=-2)
  position_in_expert *= jnp.less(
      position_in_expert, expert_capacity_dim).astype(position_in_expert.dtype)
  position_in_expert *= mask.astype(position_in_expert.dtype)

  # GSEC
  combine_tensor = jnp.zeros(
      [logits.shape[0], logits.shape[1], experts_dim, expert_capacity_dim],
      dtype=fprop_dtype)

  for i in range(experts_dim):
    # GSE
    expert_indicator = jax.nn.one_hot(expert_indices[:, :, i], experts_dim)
    # GS
    gate_i = jnp.einsum('...GSE,...GSE->...GS', gates,
                        expert_indicator.astype(gates.dtype))
    # GS - Filters out the tokens that are not selected by the current expert
    gate_i *= mask[:, :, i].astype(gate_i.dtype)

    # GSE
    gate_i = jnp.expand_dims(
        gate_i, axis=-1) * expert_indicator.astype(gate_i.dtype)
    # GS - Position in the current expert
    pos_i = position_in_expert[:, :, i]
    # GSC
    pos_i_indicator = jax.nn.one_hot(pos_i, expert_capacity_dim)
    pos_i_indicator *= jnp.expand_dims(
        mask[:, :, i].astype(pos_i_indicator.dtype), axis=-1)
    # GSEC
    combine_tensor += jnp.einsum('GSE,GSC->GSEC', gate_i.astype(fprop_dtype),
                                 pos_i_indicator.astype(fprop_dtype))

  # GSEC tensor
  dispatch_tensor = combine_tensor.astype(bool).astype(fprop_dtype)
  combine_tensor = combine_tensor.astype(fprop_dtype)

  return jnp.asarray(0.0, dtype=fprop_dtype), combine_tensor, dispatch_tensor
