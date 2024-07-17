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

import functools
import logging
import os
from typing import Tuple

import jax
from jax.experimental.shard_map import shard_map
import numpy as np

from praxis import asserts
from praxis import base_layer
from praxis import py_utils
from praxis import pytypes
from praxis.layers import attentions
from praxis.layers import grouped_query_attention
from praxis.layers import multi_query_attention
from praxis.layers import normalizations

# pylint: disable=g-import-not-at-top
try:
  from jax.experimental.pallas.ops.gpu import attention
  from jax.experimental.pallas.ops.gpu import layer_norm
  from jax.experimental.pallas.ops.gpu import decode_attention
except ImportError:
  logging.warning('jax_triton not found, please `pip install jax-triton`')
# pylint: enable=g-import-not-at-top


JTensor = pytypes.JTensor


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


class GpuTritonFusedDotProductAttention(attentions.DotProductAttention):
  """Using Jax/Pallas/Triton to call into a fused MHA kernel on NVIDIA GPU."""

  is_causal: bool = False

  # Note that flash decoding may speedup MQA and GQA only.
  # XLA MHA may still run faster than Pallas Flash decoding.
  # Tune k_splits and then measure before toggling on flash decoding.
  # https://crfm.stanford.edu/2023/10/12/flashdecoding.html
  use_flash_decoding: bool = False
  flash_decoding_k_splits: int = 16

  def _blnh_pspec(self):
    """Return sharding annotations to tensors of shape [b, l, n, h]."""
    ap = self.activation_split_dims_mapping
    return base_layer.to_partition_spec(ap.blnh, self.mesh_axis_names)

  def _bnh_pspec(self):
    """Return sharding annotations to tensors of shape [b, n, h]."""
    blnh = self._blnh_pspec()
    return jax.sharding.PartitionSpec(blnh[0], blnh[2], blnh[3])

  def _get_mesh(self) -> jax.sharding.Mesh:
    device_mesh = py_utils.create_device_mesh(
        self.ici_mesh_shape,
        self.dcn_mesh_shape,
        contiguous_submeshes=self.contiguous_submeshes,
    )
    mesh = jax.sharding.Mesh(device_mesh, self.mesh_axis_names)
    return mesh

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
    assert relative_bias is None, 'Unimplemented'

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

    query = self._scale_query(query)

    # TODO(zhangqiaorjc): Apply attention masking
    # TODO(zhangqiaorjc): Apply attention dropout.

    blnh_pspec = self._blnh_pspec()

    # TODO(zhangqiaorjc): Use hparam instead of env var.
    bwd_pass_impl = os.getenv(
        'pax_flash_attention_backward_pass_impl', default='xla'
    )

    @functools.partial(
        shard_map,
        mesh=self._get_mesh(),
        in_specs=(
            blnh_pspec,
            blnh_pspec,
            blnh_pspec,
        ),
        out_specs=blnh_pspec,
        check_rep=False,
    )
    def sharded_mha(q, k, v):
      return attention.mha(
          q,
          k,
          v,
          segment_ids=None,
          causal=self.is_causal,
          backward_pass_impl=bwd_pass_impl,
      )

    encoded = sharded_mha(query, key, value)
    encoded = self._shard_blnh(encoded)
    # TODO(zhangqiaorjc): return probs.
    return encoded, None  # pytype: disable=bad-return-type  # jax-ndarray

  def _dot_atten_one_step(
      self,
      query: JTensor,
      key_state_name: str,
      value_state_name: str,
      atten_mask: JTensor,
      relative_bias: JTensor | None = None,
      time_step: JTensor | None = None,
  ) -> tuple[JTensor, JTensor]:
    """Dot attention function for queries with 1 time step.

    Args:
      query: JTensor of shape [B, N, H].
      key_state_name: Name of the decoding key state variable.
      value_state_name: Name of the decoding value state variable.
      atten_mask: JTensor of shape [1|B, 1, S] which is a mask that is applied
        to prevent attention between unwanted pairs. This has already been
        converted into large negative logits. The first dimension is allowed to
        be of size 1, if the mask is shared by all items in the batch (e.g.,
        only a causal mask).
      relative_bias: Relative bias of shape [1|B, N, 1, S].
      time_step: A scalar. The time step tensor.

    Returns:
      encoded: JTensor of shape [B, N, H].
      probs: JTensor of shape [B, N, S].
    """
    if not self.use_flash_decoding:
      return super()._dot_atten_one_step(
          query,
          key_state_name,
          value_state_name,
          atten_mask,
          relative_bias,
          time_step,
      )

    assert relative_bias is None
    assert self.attention_extra_logit is None
    assert not self.zero_fully_masked
    del time_step
    key = self._shard_blnh(self.get_decode_state(key_state_name))
    value = self._shard_blnh(self.get_decode_state(value_state_name))
    k_b = key.shape[0]
    q_b = query.shape[0]
    assert k_b == q_b, (k_b, q_b)

    # query is 3d.
    query = self._shard_bnh(query)

    b, s, n, h = key.shape
    base_layer.assert_has_shape(value, [b, s, n, h])
    base_layer.assert_has_shape(query, [b, n, h])
    base_layer.assert_has_shape(atten_mask, [-1, 1, s])
    asserts.in_set(atten_mask.shape[0], [b, 1])
    query = self._scale_query(query)

    bnh_pspec = self._bnh_pspec()
    blnh_pspec = self._blnh_pspec()

    @functools.partial(
        shard_map,
        mesh=self._get_mesh(),
        in_specs=(
            bnh_pspec,
            blnh_pspec,
            blnh_pspec,
        ),
        out_specs=bnh_pspec,
        check_rep=False,
    )
    def sharded_decode_mha(q, k, v):
      return decode_attention.gqa(
          q,
          k,
          v,
          k_splits=self.flash_decoding_k_splits,
      )

    encoded = sharded_decode_mha(query, key, value)
    encoded = self._shard_bnh(encoded)
    return encoded, None  # pytype: disable=bad-return-type  # jax-ndarray


class GpuTritonFusedGroupedQueryAttention(
    grouped_query_attention.GroupedQueryAttention
):
  """Using flash decoding for GroupedQueryAttention."""

  # Note that flash decoding may speedup MQA and GQA only.
  # XLA MHA may still run faster than Pallas Flash decoding.
  # Tune k_splits and then measure before toggling on flash decoding.
  # https://crfm.stanford.edu/2023/10/12/flashdecoding.html
  use_flash_decoding: bool = False
  flash_decoding_k_splits: int = 16

  def _get_mesh(self) -> jax.sharding.Mesh:
    device_mesh = py_utils.create_device_mesh(
        self.ici_mesh_shape,
        self.dcn_mesh_shape,
        contiguous_submeshes=self.contiguous_submeshes,
    )
    mesh = jax.sharding.Mesh(device_mesh, self.mesh_axis_names)
    return mesh

  def _atten_context(
      self,
      query: JTensor,
      key: JTensor,
      value: JTensor,
      atten_mask: JTensor,
  ) -> Tuple[JTensor, JTensor]:
    """Computes atten context."""
    b, t, n, h = query.shape
    is_decoding = t == 1
    if not is_decoding or not self.use_flash_decoding:
      return super()._atten_context(query, key, value, atten_mask)

    if self.atten_dropout_prob > 0.0 and not self.do_eval:
      raise NotImplementedError
    if self.atten_logit_cap > 0.0:
      raise NotImplementedError

    query = query * (self.dim_per_head**-0.5) / self.atten_temp
    query = query.reshape([b, n, h])

    # Assume causal self-attention mask. Not supporting cross_attention.
    sh = self.activation_split_dims_mapping
    bnh_pspec = jax.sharding.PartitionSpec(sh.btnh[0], sh.btnh[2], sh.btnh[3])
    blnh_pspec = jax.sharding.PartitionSpec(sh.bskh)

    @functools.partial(
        shard_map,
        mesh=self._get_mesh(),
        in_specs=(
            bnh_pspec,
            blnh_pspec,
            blnh_pspec,
        ),
        out_specs=bnh_pspec,
        check_rep=False,
    )
    def sharded_decode_gqa(q, k, v):
      return decode_attention.gqa(
          q,  # [batch_size, num_q_heads, head_dim]
          k,  # [batch_size, k_seq_len, num_kv_heads, head_dim]
          v,  # [batch_size, k_seq_len, num_kv_heads, head_dim]
          k_splits=self.flash_decoding_k_splits,
      )

    encoded = sharded_decode_gqa(query, key, value)
    return encoded, None  # pytype: disable=bad-return-type  # jax-ndarray


class GpuTritonFusedMultiQueryDotProductAttention(
    multi_query_attention.MultiQueryDotProductAttention
):
  """Using flash decoding for MultiQueryDotProductAttention."""

  use_flash_decoding: bool = False
  flash_decoding_k_splits: int = 16

  def _get_mesh(self) -> jax.sharding.Mesh:
    device_mesh = py_utils.create_device_mesh(
        self.ici_mesh_shape,
        self.dcn_mesh_shape,
        contiguous_submeshes=self.contiguous_submeshes,
    )
    mesh = jax.sharding.Mesh(device_mesh, self.mesh_axis_names)
    return mesh

  def _atten_context(
      self,
      query: JTensor,
      key: JTensor,
      value: JTensor,
      atten_mask: JTensor,
      relative_bias: JTensor | None = None,
  ) -> Tuple[JTensor, JTensor]:
    """Computes atten context."""
    b, t, n, h = query.shape
    is_decoding = t == 1
    if not is_decoding or not self.use_flash_decoding:
      return super()._atten_context(
          query, key, value, atten_mask, relative_bias
      )

    if self.atten_dropout_prob > 0.0 and not self.do_eval:
      raise NotImplementedError
    if self.atten_logit_cap > 0.0:
      raise NotImplementedError
    if relative_bias is not None:
      raise NotImplementedError

    query = self._scale_query(query)
    query = query.reshape([b, n, h])

    # Assume causal self-attention mask. Not supporting cross_attention.
    sh = self.activation_split_dims_mapping
    bnh_pspec = jax.sharding.PartitionSpec(sh.btnh[0], sh.btnh[2], sh.btnh[3])
    blnh_pspec = jax.sharding.PartitionSpec(sh.bskh)

    @functools.partial(
        shard_map,
        mesh=self._get_mesh(),
        in_specs=(
            bnh_pspec,
            blnh_pspec,
            blnh_pspec,
        ),
        out_specs=bnh_pspec,
        check_rep=False,
    )
    def sharded_decode_gqa(q, k, v):
      return decode_attention.gqa(
          q,  # [batch_size, num_q_heads, head_dim]
          k,  # [batch_size, k_seq_len, num_kv_heads, head_dim]
          v,  # [batch_size, k_seq_len, num_kv_heads, head_dim]
          k_splits=self.flash_decoding_k_splits,
      )

    encoded = sharded_decode_gqa(query, key, value)
    return encoded, None  # pytype: disable=bad-return-type  # jax-ndarray

  def _dot_atten_one_step(
      self,
      query: JTensor,
      key_state_name: str,
      value_state_name: str,
      atten_mask: JTensor,
      relative_bias: JTensor | None = None,
      time_step: JTensor | None = None,
  ) -> tuple[JTensor, JTensor]:
    """Dot attention function for queries with 1 time step.

    Args:
      query: JTensor of shape [B, N, H].
      key_state_name: Name of the decoding key state variable.
      value_state_name: Name of the decoding value state variable.
      atten_mask: JTensor of shape [1|B, 1, S] which is a mask that is applied
        to prevent attention between unwanted pairs. This has already been
        converted into large negative logits. The first dimension is allowed to
        be of size 1, if the mask is shared by all items in the batch (e.g.,
        only a causal mask).
      relative_bias: Relative bias of shape [1|B, N, 1, S].
      time_step: A scalar. The time step tensor.

    Returns:
      encoded: JTensor of shape [B, N, H].
      probs: JTensor of shape [B, N, S].
    """
    del time_step
    if not self.use_flash_decoding:
      return super()._dot_atten_one_step(
          query,
          key_state_name,
          value_state_name,
          atten_mask,
          relative_bias,
      )

    assert relative_bias is None
    assert self.attention_extra_logit is None

    key = self._shard_blnh(self.get_decode_state(key_state_name))
    value = self._shard_blnh(self.get_decode_state(value_state_name))
    k_b = key.shape[0]
    q_b = query.shape[0]
    assert k_b == q_b, (k_b, q_b)

    # query is 3d.
    query = self._shard_bnh(query)

    b, s, n, h = key.shape
    base_layer.assert_has_shape(value, [b, s, n, h])
    base_layer.assert_has_shape(query, [b, n * self.num_kv_heads, h])
    base_layer.assert_has_shape(atten_mask, [-1, 1, s])
    asserts.in_set(atten_mask.shape[0], [b, 1])
    query = self._scale_query(query)

    blnh_pspec = base_layer.to_partition_spec(
        self.activation_split_dims_mapping.blnh, self.mesh_axis_names
    )
    bnh_pspec = jax.sharding.PartitionSpec(
        blnh_pspec[0], blnh_pspec[2], blnh_pspec[3]
    )

    @functools.partial(
        shard_map,
        mesh=self._get_mesh(),
        in_specs=(
            bnh_pspec,
            blnh_pspec,
            blnh_pspec,
        ),
        out_specs=bnh_pspec,
        check_rep=False,
    )
    def sharded_decode_gqa(q, k, v):
      return decode_attention.gqa(
          q,
          k,
          v,
          k_splits=self.flash_decoding_k_splits,
      )

    encoded = sharded_decode_gqa(query, key, value)
    encoded = self._shard_bnh(encoded)
    return encoded, None  # pytype: disable=bad-return-type  # jax-ndarray


class GpuTritonFusedLayerNorm(normalizations.LayerNorm):

  def _ble_pspec(self):
    """Return sharding annotations to tensors of shape [b, l, e]."""
    # TODO(zhangqiaorjc): Avoid hardcode batch dim sharding..
    return base_layer.to_partition_spec(
        [('replica', 'data'), None, None], self.mesh_axis_names
    )

  def _replicated_pspec(self):
    """Return sharding annotations to weight tensor."""
    # TODO(zhangqiaorjc): Avoid hardcode batch dim sharding..
    return base_layer.to_partition_spec([None], self.mesh_axis_names)

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
    wp = self.weight_split_dims_mapping
    if self.mesh_shape is not None and wp.wt is not None:
      # Only support replicated weights.
      raise NotImplementedError
    if not self.use_scale or not self.use_bias:
      raise NotImplementedError
    ble_pspec = self._ble_pspec()

    # TODO(zhangqiaorjc): Pass a mesh from caller.
    device_mesh = py_utils.create_device_mesh(
        self.ici_mesh_shape,
        self.dcn_mesh_shape,
        contiguous_submeshes=self.contiguous_submeshes,
    )
    mesh = jax.sharding.Mesh(device_mesh, self.mesh_axis_names)

    # TODO(zhangqiaorjc): Use hparam instead of env var.
    bwd_pass_impl = os.getenv(
        'pax_fused_layernorm_backward_pass_impl', default='xla'
    )

    @functools.partial(
        shard_map,
        mesh=mesh,
        in_specs=(
            ble_pspec,
            self._replicated_pspec(),
            self._replicated_pspec(),
        ),
        out_specs=ble_pspec,
        check_rep=False,
    )
    def layernorm(x, w, b):
      return layer_norm.layer_norm(x, w, b, backward_pass_impl=bwd_pass_impl)

    return layernorm(inputs, 1 + self.theta.scale, self.theta.bias)
