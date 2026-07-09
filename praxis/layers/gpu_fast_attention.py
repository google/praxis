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

"""Fast attention using Triton.

Experimental only. Only tested on NVIDIA A100s.
"""

from typing import Tuple

import jax
import numpy as np
from praxis import asserts
from praxis import base_layer
from praxis import pytypes
from praxis.layers import attentions
from praxis.layers import grouped_query_attention


JTensor = pytypes.JTensor
SplitDimsMapping = pytypes.SplitDimsMapping


class GpuCudnnFusedDotProductAttention(attentions.DotProductAttention):
  """Using Jax/Cudnn to call into a fused MHA kernel on NVIDIA GPU."""

  is_causal: bool = False

  def _shard_only_bn(self, x: JTensor) -> JTensor:
    """Adds sharding annotations to tensors of shape [b, n, None, None]."""
    ap = self.activation_split_dims_mapping
    if self.mesh_axis_names is None or ap.blnh is None:
      return x
    assert len(ap.blnh) == 4
    b = [ap.blnh[0], ap.blnh[2], None, None]
    return base_layer.maybe_shard(x, b, self.mesh_axis_names)

  def _dot_atten(
      self,
      query: JTensor,
      key: JTensor,
      value: JTensor,
      atten_mask: JTensor,
      relative_bias: JTensor | None = None,
  ) -> tuple[JTensor, JTensor]:
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
    query = self._shard_blnh(query)
    key = self._shard_blnh(key)
    value = self._shard_blnh(value)

    b, s, n, h = key.shape
    base_layer.assert_has_shape(value, [b, s, n, h])
    base_layer.assert_has_shape(query, [b, -1, n, h])
    t = query.shape[1]
    # If only padding bias is supplied, then atten_mask can be [B, 1, 1, S]
    # since each target token is prohibited from attending to the same set of
    # source tokens. In this case tiling is inefficient and unnecessary.
    # If there is no padding mask, and only causal mask then the shape can be
    # [1, 1, T, S]
    base_layer.assert_has_shape(atten_mask, [-1, 1, -1, s])
    asserts.in_set(atten_mask.shape[2], [t, 1])
    asserts.in_set(atten_mask.shape[0], [b, 1])

    assert self.attention_extra_logit is None
    assert not self.zero_fully_masked
    assert not self.atten_logit_cap or self.atten_logit_cap <= 0.0

    if self.atten_dropout_prob > 0.0 and not self.do_eval:
      raise NotImplementedError

    query = self._scale_query(query)
    logits_scale = 1.0 / np.sqrt(h) if self.scale_logits_by_head_dims else 1.0

    # Explicitly shard the relative_bias to ensure it has the same sharding on
    # batch and num_head dim with the query. This is required by the
    # dot_product_attention.
    if relative_bias is not None:
      relative_bias = self._shard_only_bn(relative_bias)

    encoded = jax.nn.dot_product_attention(
        query,
        key,
        value,
        bias=relative_bias,
        scale=logits_scale,
        is_causal=self.is_causal,
        implementation='cudnn',
    )
    encoded = self._shard_blnh(encoded)
    return encoded, None  # pytype: disable=bad-return-type


class GpuCudnnFusedGroupedQueryAttention(
    grouped_query_attention.GroupedQueryAttention
):
  """Using cudnn flash decoding for GroupedQueryAttention."""

  is_causal: bool = False

  def _shard_blnh(
      self, x: JTensor, split_dims_mapping: SplitDimsMapping
  ) -> JTensor:
    """Adds sharding annotations to tensors of shape [b, l, n, h]."""
    return base_layer.maybe_shard(x, split_dims_mapping, self.mesh_axis_names)

  def _atten_context(
      self,
      query: JTensor,
      key: JTensor,
      value: JTensor,
      atten_mask: JTensor,
  ) -> Tuple[JTensor, JTensor]:
    """Computes atten context."""
    if self.atten_dropout_prob > 0.0 and not self.do_eval:
      raise NotImplementedError
    if self.atten_logit_cap > 0.0:
      raise NotImplementedError

    logits_scale = (self.dim_per_head**-0.5) / self.atten_temp

    sh = self.activation_split_dims_mapping
    query = self._shard_blnh(query, sh.btnh)
    key = self._shard_blnh(key, sh.bskh)
    value = self._shard_blnh(value, sh.bskh)

    # Assume causal self-attention mask.
    assert self.is_causal

    encoded = jax.nn.dot_product_attention(
        query,
        key,
        value,
        scale=logits_scale,
        is_causal=self.is_causal,
        implementation='cudnn',
    )
    encoded = self._shard_blnh(encoded, sh.btnh)

    return encoded, None  # pytype: disable=bad-return-type  # jax-ndarray
