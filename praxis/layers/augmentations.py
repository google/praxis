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

"""Data augmentation layers."""

from typing import Tuple, Optional

import jax
from jax import lax
from jax import numpy as jnp
from praxis import base_layer
from praxis import py_utils
from praxis import pytypes

NestedMap = py_utils.NestedMap
JTensor = pytypes.JTensor


class MaskedLmDataAugmenter(base_layer.BaseLayer):
  """Performs data augmentation according to the BERT paper.

  https://arxiv.org/pdf/1810.04805.pdf

  Attributes:
    vocab_size: The total vocabulary size.
    mask_prob: Probability at which a token is replaced by the special <MASK>
      token.
    random_prob: Probability at which a token is replaced by a random token.
    same_prob: Probability at which a token is replaced by itself.
    mask_token_id: Id of the special <MASK> token.
  """
  vocab_size: int = 0
  mask_prob: float = 0.12
  random_prob: float = 0.015
  same_prob: float = 0.015
  mask_token_id: int = -1

  def __call__(self, inputs: JTensor,
               paddings: JTensor) -> Tuple[JTensor, JTensor]:
    """Applies data augmentation by randomly masking/replacing tokens in inputs.

    Args:
      inputs: An int32 tensor of shape [batch, length].
      paddings: A 0/1 tensor of shape [batch, length].

    Returns:
      A pair <new_inputs, mask>:
      new_inputs: An int32 tensor of shape [batch, length]. The new token ids
        after data augmentation.
      mask: A 0/1 tensor. A "1" indicates the corresponding token at that
        position had undergone the data augmentation process.
    """
    assert self.vocab_size > 0
    assert self.mask_token_id >= 0
    assert self.mask_prob + self.random_prob + self.same_prob < 1.0
    assert self.mask_prob + self.random_prob + self.same_prob > 0.0

    fprop_dtype = self.fprop_dtype

    def _uniform_sample(sample_p: float) -> JTensor:
      prng_key = self.next_prng_key()
      rnd_sample = jax.random.uniform(prng_key, inputs.shape)
      return (rnd_sample < sample_p).astype(fprop_dtype)

    total_replacement_prob = self.mask_prob + self.random_prob + self.same_prob
    # valid_tokens == 1.0 if the corresponding position is a valid token.
    valid_tokens = 1.0 - paddings.astype(fprop_dtype)
    # replacement == 1.0 if the corresponding token is to be replaced by
    # something else (mask, random, self).
    replacement_pos = valid_tokens * _uniform_sample(total_replacement_prob)
    no_replacement = 1.0 - replacement_pos

    # First sample the token positions to be masked out.
    remaining_prob = total_replacement_prob
    remaining_pos = replacement_pos
    mask_prob = self.mask_prob / remaining_prob
    # mask_pos == 1.0 if the corresponding token should be masked.
    mask_pos = remaining_pos * _uniform_sample(mask_prob)

    # Next sample the token positions to be replaced by random tokens.
    remaining_prob -= self.mask_prob
    remaining_pos -= mask_pos
    assert remaining_prob > 0.0
    random_prob = self.random_prob / remaining_prob
    random_pos = remaining_pos * _uniform_sample(random_prob)

    # Lastly, token positions to be replaced by self.
    self_pos = remaining_pos - random_pos

    random_tokens = jax.random.randint(
        self.next_prng_key(), inputs.shape, 0, self.vocab_size, inputs.dtype
    )
    mask_tokens = jnp.zeros_like(inputs) + self.mask_token_id

    input_dtype = inputs.dtype
    augmented = (
        inputs * no_replacement.astype(input_dtype) +
        mask_tokens * mask_pos.astype(input_dtype) +
        random_tokens * random_pos.astype(input_dtype) +
        inputs * self_pos.astype(input_dtype))

    return augmented, replacement_pos


# TODO(meadowlark): Add temporal warping layer.
class TemporalShifting(base_layer.BaseLayer):
  """Shifts audio signals by a random amount during training.

  Attributes:
    shift_range_ms: The maximum milliseconds to shift the signal forward or
      backward by. Must be smaller than the input signal length.
    sample_rate: The sample rate of the input signal.
    axis: The axis to treat and the time axis. Must be greater than 0.
  """
  shift_range_ms: float = 0.0
  sample_rate: float = 16000.0
  axis: int = 1

  def setup(self) -> None:
    super().setup()
    assert self.shift_range_ms >= 0.0, self.shift_range_ms
    assert self.axis > 0, self.axis

  def _shift_tensor(self, x, axis, shift_range, shift_size, *, pad_value=0.0):
    """Dynamically shifts x along axis by |shift_size| <= shift_range."""
    pad_width = [[0, 0]] * x.ndim
    pad_width[axis] = [shift_range, shift_range]
    x_padded = jnp.pad(x, pad_width, constant_values=pad_value)
    return lax.dynamic_slice_in_dim(
        x_padded,
        start_index=shift_range + shift_size,
        slice_size=x.shape[axis],
        axis=axis)

  def _shift(self, features: JTensor, paddings: Optional[JTensor],
             shift_range: int) -> Tuple[JTensor, Optional[JTensor]]:
    """Randomly shifts features and paddings by as much as shift_range.

    For example, if features is [1 2 3 4] and shift_range is 2, then the
    output features could be shifted as far as [0 0 1 2] or [3 4 0 0] in each
    direction. Each example in the batch is shifted independently.
    """

    def shift_example(features, paddings, shift_size):
      axis = self.axis - 1  # Subtract 1 to account for the vmaped batch dim.
      features = self._shift_tensor(features, axis, shift_range, shift_size)
      if paddings is not None:
        paddings = self._shift_tensor(
            paddings, axis, shift_range, shift_size, pad_value=1.0)
      return features, paddings

    # Generate a random amount to shift each example by. A negative shift_size
    # corresponds to shifting the signal to the left and then right-padding it
    # with zeros.
    shift_sizes = jax.random.randint(self.next_prng_key(), [features.shape[0]],
                                     -shift_range, shift_range + 1)
    # Use jax.vmap to independently shift each example without using gather.
    return jax.vmap(shift_example)(features, paddings, shift_sizes)

  def __call__(
      self,
      features: JTensor,
      paddings: Optional[JTensor] = None) -> Tuple[JTensor, Optional[JTensor]]:
    if self.do_eval or self.shift_range_ms == 0.0:
      return features, paddings

    shift_range = int(round(self.sample_rate * self.shift_range_ms / 1000.0))
    if shift_range >= features.shape[self.axis]:
      signal_length_ms = features.shape[self.axis] / self.sample_rate * 1000.0
      raise ValueError(
          f'p.shift_range_ms={self.shift_range_ms} is large enough '
          f'to blank out the entire {signal_length_ms}ms signal'
      )

    return self._shift(features, paddings, shift_range)
