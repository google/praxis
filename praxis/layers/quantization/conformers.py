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

"""Quantized Conformer Layers."""

from typing import Optional, Tuple

import jax.numpy as jnp
from praxis import pytypes
from praxis.layers import attentions
from praxis.layers.quantization import attentions as qattentions

JTensor = pytypes.JTensor


class DotProductAttentionWithContext(qattentions.DotProductAttention):
  """Dot-product attention with given left and right context.

  It covers several use cases:
    1 global self attention when left_context=right_context=None
    2 local self attention when left_context!=None and right_context!=None
    3 hybrid self attention when left_context or right_context is None

  For use cases (2,3) it will use emulated local self attention.
  For use case (2) it is more efficient to use LocalSelfAttention.
  """

  left_context: Optional[int] = None
  right_context: Optional[int] = None

  def _dot_atten(
      self,
      query: JTensor,
      key: JTensor,
      value: JTensor,
      atten_mask: JTensor,
      relative_bias: Optional[JTensor] = None,
  ) -> Tuple[JTensor, JTensor]:
    """Main attention function.

    Args:
      query: JTensor of shape [B, T, N, H].
      key: JTensor of shape [B, S, N, H].
      value: JTensor of shape [B, S, N, H].
      atten_mask: JTensor of shape [1|B, 1, 1|T, S] which is a mask that is
        applied to prevent attention between unwanted pairs. This has already
        been converted into large negative logits. Note that the first and third
        dimension allow size 1 if the mask is shared by every item in the batch
        or every token in the target sequence.
      relative_bias: Relative bias of shape [B, N, T, S].

    Returns:
      encoded: JTensor of shape [B, T, N, H].
      atten_probs: JTensor of shape [B, N, T, S].
    """
    time_size = query.shape[1]

    if self.left_context is not None or self.right_context is not None:
      input_atten_mask = atten_mask
      atten_mask = attentions.limited_context_mask(
          self.left_context, self.right_context, time_size
      )
      atten_mask = jnp.minimum(atten_mask, input_atten_mask)
    return super()._dot_atten(query, key, value, atten_mask, relative_bias)
