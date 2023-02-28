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
import math
import os
from typing import Optional, Tuple

import jax
from jax.experimental.shard_map import shard_map
from praxis import asserts
from praxis import base_layer
from praxis import py_utils
from praxis import pytypes
from praxis.layers import attentions
from praxis.layers import normalizations

# pylint: disable=g-import-not-at-top
try:
  from jax_triton.pallas.ops import attention
  from jax_triton.pallas.ops import layer_norm
except ImportError:
  logging.warning('jax_triton not found, please `pip install jax-triton`')
# pylint: enable=g-import-not-at-top


JTensor = pytypes.JTensor


class GpuTritonFusedDotProductAttention(attentions.DotProductAttention):
  """Using Jax/Pallas/Triton to call into a fused MHA kernel on NVIDIA GPU."""

  def _blnh_pspec(self):
    """Return sharding annotations to tensors of shape [b, l, n, h]."""
    ap = self.activation_split_dims_mapping
    return base_layer.to_partition_spec(ap.blnh, self.mesh_axis_names)

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
    logging.info('Using experimental GpuTritonFusedDotProductAttention.')
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

    # TODO(zhangqiaorjc): Pass a mesh from caller.
    device_mesh = py_utils.create_device_mesh(
        self.ici_mesh_shape,
        self.dcn_mesh_shape,
        contiguous_submeshes=self.contiguous_submeshes,
    )
    mesh = jax.sharding.Mesh(device_mesh, self.mesh_axis_names)

    # TODO(zhangqiaorjc): Use hparam instead of env var.
    bwd_pass_impl = os.getenv(
        'pax_flash_attention_backward_pass_impl', default='xla'
    )

    @functools.partial(
        shard_map,
        mesh=mesh,
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
          q, k, v, sm_scale=1.0 / math.sqrt(h), backward_pass_impl=bwd_pass_impl
      )

    encoded = sharded_mha(query, key, value)
    encoded = self._shard_blnh(encoded)
    # TODO(zhangqiaorjc): return probs.
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
      self, inputs: JTensor, paddings: Optional[JTensor] = None
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
