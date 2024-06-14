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

"""Multi-Query Attention layers."""

import math
from typing import Callable, Mapping, Sequence

from flax import linen as nn
import jax
from jax import numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from praxis import asserts
from praxis import base_layer
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis.layers import attentions
from praxis.layers import base_ops
from praxis.layers import embedding_softmax
from praxis.layers import stochastics


WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams
template_field = base_layer.template_field
LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]
JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor
NestedInt = pytypes.NestedInt

SplitDimsMapping = pytypes.SplitDimsMapping
PREFIX_DECODE_CACHE = base_layer.PREFIX_DECODE_CACHE


class OneHeadedAttentionProjection(base_layer.BaseLayer):
  """Layer that computes projection with one head.

  This layer is expected to be used within MultiQueryAttention below.

  Attributes:
    input_dim: Input dimension.
    output_dim: Size of output.
    use_bias: Whether to add bias in projection or not.
  """
  input_dim: int = 0
  output_dim: int = 0
  use_bias: bool = True
  einsum_tpl: LayerTpl = template_field(base_ops.EinsumOp)

  def setup(self) -> None:
    wp = self.weight_split_dims_mapping
    has_sharding = self.mesh_shape is not None and wp.wt is not None
    wt = wp.wt
    pc_shape = [self.input_dim, self.output_dim]
    pc = WeightHParams(
        shape=pc_shape, mesh_shape=self.mesh_shape, tensor_split_dims_mapping=wt
    )
    self.create_variable('w', pc)
    if self.use_bias:
      if has_sharding:
        bias_split_dims_mapping = [wp.wt[1]]
      else:
        bias_split_dims_mapping = None
      pc_bias = WeightHParams(
          shape=[self.output_dim],
          init=WeightInit.Constant(0.0),
          mesh_shape=self.mesh_shape,
          tensor_split_dims_mapping=bias_split_dims_mapping,
      )
      self.create_variable('b', pc_bias)
    self.create_child('einsum', self.einsum_tpl.clone())

  def __call__(self, inputs: JTensor) -> JTensor:
    """Computes the multi headed projection for inputs.

    Args:
      inputs: A JTensor of shape [..., p.input_dim].

    Returns:
      The projected JTensor with shape [..., p.output_dim].
    """
    theta = self.theta

    shape = inputs.shape
    inputs = self._cast_to_fprop_dtype(inputs)
    w = theta.w

    assert (
        shape[-1] == self.input_dim
    ), f'Expecting shape[-1] == p.input_dim, {shape[-1]} != {self.input_dim}'
    eqn = '...D,DH->...H'
    ret = self.einsum(eqn, inputs, w)
    if self.use_bias:
      ret += theta.b
    return ret


class MultiQueryDotProductAttention(base_layer.BaseLayer):
  """Dot-product attention sharing keys and values across heads.

  This implementation heavily uses einsum to be efficient on TPUs.  We use the
  following capital letters to denote certain JTensor parameters.

    B = batch size
    S = length of the key/value (source)
    T = length of the query (target)
    D = model dimension
    N = number of attention heads
    H = dimensions of each attention head.

  The algorithm is sketched as follows. Each intermediate JTensor or weight
  JTensor is annotated with its shape. E.g., Wq, the weight JTensor for query's
  projection, its shape is [D, N, H].

  Trainable weights:
    Wq: [D, N, H]
    Wk, Wv: [D, H]
    Wout: [D, N, H]

  Note it also allows k, v and q to have different input dimension by setting
  input_dim as a dict: {'key': key_dim, 'value': value_dim, 'query': query_dim}.

  Input q:[B, T, D]; k:[B, S, D]; v:[B, S, D]
  q_proj:[B, T, N, H] = einsum('BTD,DNH->BTNH', x, Wq)
  k_proj:[B, S, H] = einsum('BSD,DH->BSH', x, Wk)
  v_proj:[B, S, H] = einsum('BSD,DH->BSH', x, Wv)
  logits:[B, N, T, S] = einsum('BTNH,BSH->BNTS', q_proj, k_proj) / sqrt(H)
  probs:[B, N, T, S] = softmax(logits)
  context:[B, T, N, H] = einsum('BNTS,BSH->BTNH', probs, v_proj)
  Output y:[B, T, D] = einsum('BTNH,DNH>BTD', context, Wout)

  Attributes:
    input_dim: An integer or a dict of integer values as number of input nodes.
      If input_dim is a dict, keys must be key, value and query.
    hidden_dim: Number of hidden nodes.
    num_heads: Number of attention heads.
    num_kv_heads: Number of kv heads. num_heads % num_kv_heads = 0.
    dim_per_head: Dimension of each attention head. If None then dim_per_head ==
      hidden_dim // num_heads.
    dropout_tpl: Parameterization for the dropout layer.
    atten_dropout_prob: Probability at which we apply dropout to the attention
      weights.
    proj_tpl: Parameterization for the query projection_tpl layer.
    headless_proj_tpl: Parameterization for the key/value projection_tpl layer.
    use_bias: Whether to use bias for projection_tpl layers.
    output_proj_use_nhd_shape: Whether to use NHD variable shape in output
      projection layer.
    internal_enable_query_scale: Internal. Enable scaling of query vector.
    atten_logit_cap: Cap the absolute values of logits by tanh. Enabled when a
      positive value is specified. May not be supported by a subclass.
    use_rotary_position_emb: Whether to add rotary position embedding to the
      queries and keys before computing self attention scores. This was proposed
      in https://arxiv.org/abs/2104.09864.
    consolidate_rope_key_state: Whether to consolidate `key_post_rotary_pos_emb`
      with `key_state` as `key_state` when RoPE is enabled. Only set it as
      `True` when `use_rotary_position_emb` is True.
    relative_bias_tpl: Optional parameterization of relative bias.
    attention_extra_logit: Extra logit for attention softmax.
    combine_qkv: Whether to combine qkv tensor for optimizing qkv input gradient
      computation with SPMD. Only supports self-attention.
    decode_cache: if the attention layer needs decode cache.
    scale_query_by_dim_per_head: whether to scale the query by dim_per_head,
      instead of default hidden_dim // num_heads.
    chunked_attn_num_seq_split: Whether to compute the attention context as a
      loop over the context sequence, avoiding the full attention matrix
      computation. Default value of 1 implies no loop.

  Note: dconv_qkv and ngrammer are not supported.
  """
  input_dim: int | dict[str, int] = 0
  hidden_dim: int = 0
  num_heads: int = 1
  num_kv_heads: int = 1
  dim_per_head: int | None = None
  dropout_tpl: LayerTpl = template_field(stochastics.Dropout)
  atten_dropout_prob: float = 0.0
  proj_tpl: LayerTpl = template_field(attentions.AttentionProjection)
  headless_proj_tpl: LayerTpl = template_field(OneHeadedAttentionProjection)
  internal_gshard_gaussian_init: bool = False
  use_bias: bool = True
  output_proj_use_nhd_shape: bool = False
  internal_enable_query_scale: bool = True
  atten_logit_cap: float = 0.0
  # TODO(b/300717814): Remove redundant `use_rotary_position_emb` field.
  use_rotary_position_emb: bool = False
  consolidate_rope_key_state: bool = False
  rotary_position_emb_tpl: LayerTpl = template_field(
      embedding_softmax.RotaryPositionalEmbedding
  )
  relative_bias_tpl: LayerTpl | None = template_field(None)
  attention_extra_logit: float | None = None
  dconv_qkv: bool = False
  combine_qkv: bool = False
  decode_cache: bool = True
  qk_einsum_tpl: LayerTpl = template_field(base_ops.EinsumOp)
  pv_einsum_tpl: LayerTpl = template_field(base_ops.EinsumOp)
  scale_query_by_dim_per_head: bool = False
  chunked_attn_num_seq_split: int = 1

  # SPMD partition related params.
  #
  # d - model_dim
  # n - num_heads
  # h - attention_dim_per_heads
  # b - batch_size
  # l - seq_len

  class WeightSharding(base_layer.BaseLayer.WeightSharding):
    """Represents how layer's learned parameters are partitioned across a mesh.

    Attributes:
      proj: How the projection weights should be sharded. All projection matrix
        share the same sharding.
      dconv: How the dconv weights should be sharded. All dconv weights share
        the same sharding.
    """
    proj: SplitDimsMapping = None
    dconv: SplitDimsMapping = None
    proj_headless: SplitDimsMapping = None

  class ActivationSharding(base_layer.BaseLayer.ActivationSharding):
    """Represents how intermediate values should be partitioned across a mesh.

    Attributes:
      blnh: Mesh split for query, and encoded tensors with the shape of
        [batch_size, seq_len, num_heads, dim_per_head].
      blh: Mesh split key, value, and encoded tensors with the shape of
        [batch_size, seq_len, dim_per_head].
      bld: Mesh split for output after post projection with the shape of
        [batch_size, seq_len, model_dim].
      bd: Mesh split for extend step output after post projection with the shape
        of [batch_size, model_dim].
    """
    blnh: SplitDimsMapping = None
    blh: SplitDimsMapping = None
    bld: SplitDimsMapping = None
    bd: SplitDimsMapping = None

  def setup(self) -> None:
    wp = self.weight_split_dims_mapping
    assert self.input_dim, 'input_dim is {}'.format(self.input_dim)
    assert self.hidden_dim, 'hidden_dim is {}'.format(self.hidden_dim)

    assert not self.dconv_qkv
    assert not self.combine_qkv

    dim_per_head = self.dim_per_head
    if dim_per_head is None:
      dim_per_head = self.hidden_dim // self.num_heads
      assert (
          dim_per_head * self.num_heads == self.hidden_dim
      ), f'{dim_per_head} * {self.num_heads} != {self.hidden_dim}'

    if self.mesh_shape is not None:
      assert self.weight_split_dims_mapping is not None
      assert self.activation_split_dims_mapping is not None

    if self.consolidate_rope_key_state:
      assert self.use_rotary_position_emb

    if isinstance(self.input_dim, Mapping):
      key_input_dim = self.input_dim['key']
      value_input_dim = self.input_dim['value']
      query_input_dim = self.input_dim['query']
      assert key_input_dim, f'key_input_dim is {key_input_dim}'
      assert query_input_dim, f'query_input_dim is {query_input_dim}'
    else:
      key_input_dim = self.input_dim
      value_input_dim = self.input_dim
      query_input_dim = self.input_dim

    def project_input(input_dim, dim_per_head, num_heads):
      proj_p = self.proj_tpl.clone().set(
          input_dim=input_dim,
          num_heads=num_heads,
          dim_per_head=dim_per_head,
          use_bias=self.use_bias,
      )
      proj_p.weight_split_dims_mapping.wt = wp.proj
      return proj_p

    def project_input_kv(input_dim, dim_per_head):
      if self.num_kv_heads == 1:
        proj_p = self.headless_proj_tpl.clone().set(
            input_dim=input_dim, output_dim=dim_per_head, use_bias=self.use_bias
        )
        proj_p.weight_split_dims_mapping.wt = wp.proj_headless
        return proj_p
      else:
        assert self.num_heads % self.num_kv_heads == 0
        return project_input(input_dim, dim_per_head, self.num_kv_heads)

    dim_per_head = self.dim_per_head
    if dim_per_head is None:
      dim_per_head = self.hidden_dim // self.num_heads
      assert (
          dim_per_head * self.num_heads == self.hidden_dim
      ), f'{dim_per_head} * {self.num_heads} != {self.hidden_dim}'
    self.create_child('value', project_input_kv(value_input_dim, dim_per_head))

    self.create_child('key', project_input_kv(key_input_dim, dim_per_head))
    self.create_child(
        'query', project_input(query_input_dim, dim_per_head, self.num_heads)
    )

    if self.use_rotary_position_emb:
      pos_emb_p = self.rotary_position_emb_tpl.clone().set(
          embedding_dims=dim_per_head
      )
      self.create_child('rotary_position_emb', pos_emb_p)

    if self.relative_bias_tpl is not None:
      relative_bias_p = self.relative_bias_tpl.clone()
      relative_bias_p.num_heads = self.num_heads
      self.create_child('relative_bias', relative_bias_p)

    self.create_child(
        'atten_dropout',
        self.dropout_tpl.clone().set(keep_prob=1.0 - self.atten_dropout_prob),
    )

    # Setting is_output_projection=True to set the projection direction
    # from hidden dim to input dim. Output projection follows query_input_dim.
    post_proj_p = self.proj_tpl.clone().set(
        input_dim=query_input_dim,
        num_heads=self.num_heads,
        dim_per_head=dim_per_head,
        is_output_projection=True,
        use_bias=self.use_bias,
        use_nhd_shape=self.output_proj_use_nhd_shape,
    )
    post_proj_p.weight_split_dims_mapping.wt = wp.proj

    self.create_child('post', post_proj_p)
    self.create_child('qk_einsum', self.qk_einsum_tpl.clone())
    self.create_child('pv_einsum', self.pv_einsum_tpl.clone())

  def _shard_bnh(self, x: JTensor) -> JTensor:
    """Shards tensors of shape [b, n, h].

    Single step decoder output are of shape [b, n, h].

    Args:
      x: A tensor of shape [b, n, h]

    Returns:
      x with proper sharding annotations.
    """
    ap = self.activation_split_dims_mapping
    if self.mesh_axis_names is None:
      return x
    if ap.blnh is None:
      return x
    assert len(ap.blnh) == 4
    bnh = [ap.blnh[0], ap.blnh[2], ap.blnh[3]]
    return base_layer.maybe_shard(x, bnh, self.mesh_axis_names)

  def _shard_blnh(self, x: JTensor) -> JTensor:
    """Adds sharding annotations to tensors of shape [b, l, n, h]."""
    ap = self.activation_split_dims_mapping
    return base_layer.maybe_shard(x, ap.blnh, self.mesh_axis_names)

  def _shard_blh(self, x: JTensor) -> JTensor:
    """Adds sharding annotations to tensors of shape [b, l, h]."""
    ap = self.activation_split_dims_mapping
    shard = None
    if ap.blh is not None:
      shard = ap.blh
    elif ap.blnh is not None:
      shard = [ap.blnh[0], ap.blnh[1], ap.blnh[3]]
    return base_layer.maybe_shard(x, shard, self.mesh_axis_names)

  def _shard_bld(self, x: JTensor) -> JTensor:
    """Adds sharding annotations to tensors of shape [b, l, d]."""
    ap = self.activation_split_dims_mapping
    return base_layer.maybe_shard(x, ap.bld, self.mesh_axis_names)

  def _shard_bd(self, x: JTensor) -> JTensor:
    """Adds sharding annotations to tensors of shape [b, d]."""
    ap = self.activation_split_dims_mapping
    if self.mesh_axis_names is None:
      return x
    if ap.bld is None:
      return x
    if ap.bd is not None:
      bd = ap.bd
      assert len(bd) == 2
    else:
      assert len(ap.bld) == 3
      bd = [ap.bld[0], ap.bld[2]]
    return base_layer.maybe_shard(x, bd, self.mesh_axis_names)

  def _scale_query(self, query: JTensor) -> JTensor:
    """Scales the query vector."""
    if self.scale_query_by_dim_per_head and self.dim_per_head is not None:
      dim_per_head = self.dim_per_head
    else:
      dim_per_head = self.hidden_dim // self.num_heads

    query *= dim_per_head**-0.5

    return query

  def _cap_logits(self, logits: JTensor) -> JTensor:
    """When enabled, caps the logits by p.atten_logit_cap with tanh."""
    if not self.atten_logit_cap or self.atten_logit_cap <= 0.0:
      return logits
    cap = jnp.array(self.atten_logit_cap, dtype=logits.dtype)
    # Note that since this caps the negative side as well, caller
    # must defer the pad-with-very-negative-logits logic to after
    # this function returns.
    logits = cap * jnp.tanh(logits / cap)
    return logits

  def _log_softmax_with_extra_logit(self, logits: JTensor) -> JTensor:
    """Computes log softmax with extra logit.

    self.attention_extra_logit is a user defined float value that
    helps to stabilize logit values so that they don't drift too much from it.

    Args:
      logits: input logit tensor

    Returns:
      Log softmax with extra logit value.
    """
    # Applies stop_gradient to max_logit instead of logits.
    max_logit = jnp.max(jax.lax.stop_gradient(logits), axis=-1, keepdims=True)
    extra_logit = self.attention_extra_logit
    if extra_logit is not None:
      extra_logit = jnp.asarray(extra_logit, dtype=max_logit.dtype)
      max_logit = jnp.maximum(max_logit, extra_logit)
    exp_x = jnp.exp(logits - max_logit)
    sum_exp_x = jnp.sum(exp_x, axis=-1, keepdims=True)
    if extra_logit is not None:
      sum_exp_x += jnp.exp(extra_logit - max_logit)
    return logits - jnp.log(sum_exp_x) - max_logit

  def _atten_logits(self, query: JTensor, key: JTensor) -> JTensor:
    """Compute logits from query and key."""
    query = query.transpose(0, 2, 1, 3)
    logits = self.qk_einsum('BNTH,BSH->BNTS', query, key)
    return logits

  def _atten_context(
      self,
      query: JTensor,
      key: JTensor,
      value: JTensor,
      atten_mask: JTensor,
      relative_bias: JTensor | None = None,
  ) -> tuple[JTensor, JTensor]:
    """Computes attention context."""
    _, t, n, _ = query.shape
    _, s, _ = key.shape
    logits = self._atten_logits(query, key)
    if relative_bias is not None:
      # The relative_bias has shape [1, n, t, s] or [b, n, t, s].
      base_layer.assert_has_shape(relative_bias, [-1, n, t, s])
      logits += relative_bias
    logits = checkpoint_name(logits, 'logits')
    logits = self._cap_logits(logits)
    # Attention softmax is always carried out in fp32.
    logits = logits.astype(jnp.float32)
    # Apply attention masking
    padded_logits = py_utils.apply_mask_to_logits(logits, atten_mask)
    if self.attention_extra_logit is None:
      probs = jax.nn.softmax(padded_logits, axis=-1).astype(key.dtype)
    else:
      probs = jnp.exp(self._log_softmax_with_extra_logit(padded_logits)).astype(
          key.dtype
      )
    # Apply attention dropout.
    probs = self.atten_dropout(probs)
    # Compute the attention context.
    encoded = self.pv_einsum('BNTS,BSH->BNTH', probs, value)
    encoded = encoded.transpose(0, 2, 1, 3)
    encoded = checkpoint_name(encoded, 'context')
    encoded = self._shard_blnh(encoded)
    return encoded, probs

  def _atten_context_chunked_attn_seq_split(
      self,
      query: JTensor,
      key: JTensor,
      value: JTensor,
      atten_mask: JTensor,
      relative_bias: JTensor | None = None,
  ) -> tuple[JTensor, JTensor]:
    """Computes chunked attention context."""
    b, t, n, _ = query.shape
    _, s, h = value.shape
    assert (
        s % self.chunked_attn_num_seq_split == 0
    ), 'The number of attn splits must divide the sequence length'
    w = s // self.chunked_attn_num_seq_split
    query = query.transpose(0, 2, 1, 3)
    full_encoded = jnp.zeros((b, n, t, h), dtype=value.dtype)
    for i in range(self.chunked_attn_num_seq_split):
      logits = jnp.einsum(
          'BNTH,BSH->BNTS',
          query[:, :, i * w : (i + 1) * w, :],  # current query chunk
          key[:, : (i + 1) * w, :],  # keys context up to current chunk
      )
      if relative_bias is not None:
        # The relative_bias has shape [1, n, t, s] or [b, n, t, s].
        base_layer.assert_has_shape(relative_bias, [-1, n, t, s])
        bias_slice = relative_bias[:, :, i * w : (i + 1) * w, : (i + 1) * w]
        logits += bias_slice
      logits = checkpoint_name(logits, 'logits')
      logits = self._cap_logits(logits)
      # Attention softmax is always carried out in fp32.
      logits = logits.astype(jnp.float32)
      mask_slice = atten_mask[:, :, i * w : (i + 1) * w, : (i + 1) * w]
      # Consider supporting a boolean attention mask a la
      # attention_mask_use_where
      padded_logits = logits + mask_slice.astype(jnp.float32)
      probs = jax.nn.softmax(padded_logits, axis=-1).astype(key.dtype)
      # Apply attention dropout.
      probs = self.atten_dropout(probs)
      # Compute the attention context slice.
      encoded = jnp.einsum(
          'BNTS,BSH->BNTH',
          probs,
          value[:, : (i + 1) * w, :],
      )
      encoded = checkpoint_name(encoded, 'context')
      full_encoded = full_encoded.at[:, :, i * w : (i + 1) * w, :].set(encoded)

    full_encoded = full_encoded.transpose(0, 2, 1, 3)
    full_encoded = checkpoint_name(full_encoded, 'context')
    full_encoded = self._shard_blnh(full_encoded)
    full_probs = None
    return full_encoded, full_probs  # pytype: disable=bad-return-type  # jax-ndarray

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
      key: JTensor of shape [B, S, H].
      value: JTensor of shape [B, S, H].
      atten_mask: JTensor of shape [1/B, 1, 1/T, S] which is a mask that is
        applied to prevent attention between unwanted pairs. This has already
        been converted into large negative logits. Note that the first and third
        dimension allow size 1 if the mask is shared by every item in the batch
        or every token in the target sequence.
      relative_bias: Relative bias of shape [B, N, T, S].

    Returns:
      encoded: JTensor of shape [B, T, N, H].
      atten_probs: JTensor of shape [B, N, T, S].
    """
    b, t, n, h = query.shape
    base_layer.assert_has_shape(query, [b, -1, n, h])
    _, s, _ = key.shape
    base_layer.assert_has_shape(value, [b, s, -1])
    t = query.shape[1]
    # If only padding bias is supplied, then atten_mask can be [B, 1, 1, S]
    # since each target token is prohibited from attending to the same set of
    # source tokens. In this case tiling is inefficient and unnecessary.
    # If there is no padding mask, and only causal mask then the shape can be
    # [1, 1, T, S]
    base_layer.assert_has_shape(atten_mask, [-1, 1, -1, s])
    asserts.in_set(atten_mask.shape[2], [1, t])
    asserts.in_set(atten_mask.shape[0], [1, b])
    query = self._scale_query(query)
    if self.chunked_attn_num_seq_split > 1:
      return self._atten_context_chunked_attn_seq_split(
          query, key, value, atten_mask, relative_bias
      )
    else:
      return self._atten_context(query, key, value, atten_mask, relative_bias)

  def decoding_state_sequence_length(self):
    """Returns the length of full decoding sequences."""
    return self.get_decode_state('key_state').shape[1]

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
      query: JTensor of shape [B, N, H] or [B, T, N, H].
      key_state_name: Name of the decoding key state variable.
      value_state_name: Name of the decoding value state variable.
      atten_mask: JTensor of shape [1/B, 1, S] or [B, 1, L, S] which is a mask
        that is applied to prevent attention between unwanted pairs. This has
        already been converted into large negative logits. The first dimension
        is allowed to be of size 1, if the mask is shared by all items in the
        batch (e.g., only a causal mask).
      relative_bias: Relative bias of shape [1/B, N, 1, S].
      time_step: A scalar or JTensor. Current time-step, 0-based.

    Returns:
      encoded: JTensor of shape [B, N, H].
      probs: JTensor of shape [B, N, S].
    """
    extend_one_step = len(query.shape) == 3
    key = self.get_decode_state(key_state_name)
    value = self.get_decode_state(value_state_name)
    if extend_one_step:
      query = self._shard_bnh(query)
    else:
      query = self._shard_blnh(query)
    if self.num_kv_heads == 1:
      key = self._shard_blh(key)
      value = self._shard_blh(value)
      encoded, probs = self._dot_atten_one_step_from_qkv(
          query, key, value, atten_mask, relative_bias, time_step
      )
      return self._shard_bnh(encoded), probs
    else:
      b, n, h = query.shape
      _, _, nk, _ = key.shape
      key = self._shard_blnh(key)
      value = self._shard_blnh(value)
      v_q = jnp.reshape(query, (b, nk, n // nk, h))
      if relative_bias is not None:
        v_rb = jnp.reshape(
            relative_bias,
            relative_bias.shape[:1] + (nk, n // nk) + relative_bias.shape[2:],
        )
      else:
        v_rb = None
      with self._context_for_kv_vmap():
        encoded, probs = jax.vmap(
            self._dot_atten_one_step_from_qkv,
            in_axes=(1, 2, 2, None, 1, None),
            out_axes=(1, 1),
        )(v_q, key, value, atten_mask, v_rb, time_step)
        encoded = self._shard_bnh(jnp.reshape(encoded, (b, n, h)))
        probs = jnp.reshape(probs, (b, n, -1))
        return encoded, probs

  def _dot_atten_one_step_from_qkv(
      self,
      query: JTensor,
      key: JTensor,
      value: JTensor,
      atten_mask: JTensor,
      relative_bias: JTensor | None,
      time_step: JTensor | None = None,
  ) -> tuple[JTensor, JTensor]:
    """_dot_atten_one_step with tensors instead of state names."""
    del time_step
    # query is 3d.
    extend_one_step = len(query.shape) == 3
    b, s, h = key.shape
    if extend_one_step:
      base_layer.assert_has_shape(query, [b, -1, h])
      base_layer.assert_has_shape(atten_mask, [-1, -1, s])
    else:
      base_layer.assert_has_shape(query, [b, -1, -1, h])
      base_layer.assert_has_shape(atten_mask, [-1, -1, -1, s])
    asserts.in_set(atten_mask.shape[0], [1, b])

    base_layer.assert_has_shape(value, [b, s, h])
    query = self._scale_query(query)
    if extend_one_step:
      logits = self.qk_einsum('BNH,BSH->BNS', query, key)
    else:
      logits = self.qk_einsum('BTNH,BSH->BNTS', query, key)
    if relative_bias is not None:
      if not extend_one_step:
        raise NotImplementedError(
            'MultiQueryAttention does not support extend n steps with '
            'relative bias.'
        )
      base_layer.assert_has_shape(relative_bias, [-1, -1, 1, s])
      asserts.in_set(relative_bias.shape[0], [1, b])
      relative_bias = jnp.squeeze(relative_bias, axis=2)
      logits += relative_bias
    logits = self._cap_logits(logits)
    # Attention softmax is always carried out in fp32.
    logits = logits.astype(jnp.float32)
    # Apply attention masking
    padded_logits = py_utils.apply_mask_to_logits(logits, atten_mask)
    # Of shape [b, n, s]
    if self.attention_extra_logit is None:
      probs = jax.nn.softmax(padded_logits, axis=-1).astype(key.dtype)
    else:
      probs = jnp.exp(self._log_softmax_with_extra_logit(padded_logits)).astype(
          key.dtype
      )
    # Compute the attention context.
    if extend_one_step:
      encoded = self.pv_einsum('BNS,BSH->BNH', probs, value)
    else:
      encoded = self.pv_einsum('BNTS,BSH->BTNH', probs, value)
    return encoded, probs  # pytype: disable=bad-return-type  # jax-ndarray

  def _context_for_kv_vmap(self):
    # Transpose the sharding on num_heads to None, so that the inner n // nk dim
    # is not sharded. The out nk dim is sharded on the vmap dim, which is not
    # visible by the layer code.
    if self.activation_split_dims_mapping.blnh:
      n_sharding = self.activation_split_dims_mapping.blnh[2]
      if isinstance(n_sharding, tuple):
        n_sharding = n_sharding[0]
    else:
      n_sharding = None
    assert n_sharding is None or isinstance(n_sharding, str)
    if base_layer.JaxContext.has_context():
      new_context_params = base_layer.cur_jax_context().hparams.clone()
    else:
      new_context_params = base_layer.JaxContext.HParams()
    if n_sharding is not None:
      new_context_params.mesh_axes_transpose = {n_sharding: None}
    return base_layer.JaxContext.new_context(hparams=new_context_params)

  def __call__(
      self,
      query_vec: JTensor,
      key_vec: JTensor,
      value_vec: JTensor,
      atten_mask: JTensor,
      query_segment_pos: JTensor | None = None,
      key_segment_pos: JTensor | None = None,
  ) -> tuple[JTensor, JTensor]:
    """Computes the value vector given the current query output.

    Args:
      query_vec: JTensor of shape [B, T, D].
      key_vec: JTensor of shape [B, S, D].
      value_vec: JTensor of shape [B, S, D].
      atten_mask: JTensor of shape [1/B, 1, 1/T, S] which is a mask that is
        applied to prevent attention between unwanted pairs. This has already
        been converted into large negative logits. Note that the first and third
        dimension allow size 1 if the mask is shared by every item in the batch
        or every token in the target sequence.
      query_segment_pos: JTensor of shape [B, T]
      key_segment_pos: JTensor of shape [B, S]

    Returns:
      encoded: JTensor of shape [B, T, D].
      atten_probs: JTensor of shape [B, N, T, S].
    """
    # Make sure the weight gradient matmul computes with D replicated, which
    # will be a regular reduce-scatter pattern on the result. This helps
    # scheduling MegaScale ops.
    def _rep_d(x):
      return base_layer.maybe_shard(
          x,
          [None] * x.ndim,
          self.mesh_axis_names,
          unconstrained_dims=range(x.ndim - 1),
      )

    query_vec, key_vec, value_vec = [
        _rep_d(x) for x in [query_vec, key_vec, value_vec]
    ]

    # Project inputs to query, key and value, respectively has shape
    # [B, T, N, H], [B, S, H], and [B, S, H].
    query_proj = self.query(query_vec)
    key_proj = self.key(key_vec)
    value_proj = self.value(value_vec)
    query_proj = checkpoint_name(query_proj, 'query_proj')
    key_proj = checkpoint_name(key_proj, 'key_proj')
    value_proj = checkpoint_name(value_proj, 'value_proj')

    if not self.consolidate_rope_key_state:
      self._fprop_update_decode_state('key_state', key_proj)
    self._fprop_update_decode_state('value_state', value_proj)

    # Apply rotary position embeddings.
    # Paper: https://arxiv.org/abs/2104.09864.
    if self.use_rotary_position_emb:
      query_proj = self.rotary_position_emb(query_proj, query_segment_pos)
      key_shape = key_proj.shape
      # [B, S, H] -> [B, S, N(1), H]
      if self.num_kv_heads == 1:
        key_proj = jnp.expand_dims(key_proj, axis=-2)
      key_proj = self.rotary_position_emb(key_proj, key_segment_pos)
      if self.num_kv_heads == 1:
        key_proj = jnp.reshape(key_proj, key_shape)
      if self.consolidate_rope_key_state:
        self._fprop_update_decode_state('key_state', key_proj)
      else:
        self._fprop_update_decode_state('key_post_rotary_pos_emb', key_proj)

    # Apply relative bias.
    # Paper: https://aclanthology.org/N18-2074.pdf.
    if self.relative_bias_tpl:
      relative_bias = self.relative_bias(query_segment_pos, key_segment_pos)
    else:
      relative_bias = None

    query_proj = self._shard_blnh(query_proj)
    if self.num_kv_heads == 1:
      key_proj = self._shard_blh(key_proj)
      value_proj = self._shard_blh(value_proj)
      encoded, atten_probs = self._dot_atten(
          query_proj,
          key_proj,
          value_proj,
          atten_mask,
          relative_bias,
      )
    else:
      key_proj = self._shard_blnh(key_proj)
      value_proj = self._shard_blnh(value_proj)
      b, t, n, h = query_proj.shape
      _, s, nk, _ = key_proj.shape
      assert n % nk == 0
      v_q = jnp.reshape(query_proj, (b, t, nk, n // nk, h))
      if relative_bias is not None:
        v_rb = jnp.reshape(relative_bias, (b, nk, n // nk, t, s))
      else:
        v_rb = None
      with self._context_for_kv_vmap():
        encoded, atten_probs = jax.vmap(
            self._dot_atten,
            in_axes=(2, 2, 2, None, 1),
            out_axes=(2, 1),
        )(
            v_q,
            key_proj,
            value_proj,
            atten_mask,
            v_rb,
        )
      encoded = self._shard_blnh(jnp.reshape(encoded, (b, t, n, h)))
      if atten_probs is not None:
        atten_probs = jnp.reshape(atten_probs, (b, t, n, s))

    # Post projection
    encoded = self.post(encoded)
    encoded = self._shard_bld(encoded)
    encoded = checkpoint_name(encoded, 'out_proj')

    return encoded, atten_probs

  def init_states(self, target_batch_size: int, target_max_length: int) -> None:
    """Initializes cache for autoregressive cached decoding.

    Args:
      target_batch_size: The batch size of the target to be decoded.
      target_max_length: The sequence length of the target to be decoded.
    Return: None.
    """
    raise NotImplementedError(type(self))

  @nn.nowrap
  def _fprop_update_decode_state(self, name: str, value: JTensor) -> None:
    """Updates decode state in fprop.

    This is a no-op in training.
    Args:
      name: Variable name in decoder cache.
      value: Value to extend at time step.
    """
    # Only update the state if it is decoding.
    if (
        not self.is_mutable_collection(base_layer.DECODE_CACHE)
        or not self.decode_cache
    ):
      return
    self.update_decode_state(name, value)

  @nn.nowrap
  def extend_decode_state(self, name: str, value: JTensor, time_step: JTensor,
                          time_dim: int) -> JTensor:
    """Extends decode state at time_step.

    The decode state is batch major with shape [B, T, H].
    Args:
      name: Variable name in decoder cache.
      value: Value to extend at time step.
      time_step: A scalar. Time step to update the state.
      time_dim: Time dimension in the decode state.

    Returns:
      Updated decode cache state of that variable.
    """
    if (self.num_kv_heads == 1 and len(value.shape) == time_dim + 1) or (
        self.num_kv_heads > 1 and len(value.shape) == time_dim + 2
    ):
      extend_value = jnp.expand_dims(value, axis=time_dim)
    else:
      extend_value = value
    indices = [0] * extend_value.ndim
    indices[time_dim] = time_step.astype(jnp.int32)
    state = self.get_decode_state(name)
    assert state is not None
    new_state = jax.lax.dynamic_update_slice(state,
                                             extend_value.astype(state.dtype),
                                             indices)
    self.update_decode_state(name, new_state)
    return new_state

  def extend_step(
      self,
      query_vec: JTensor,
      *,
      atten_mask: JTensor,
      time_step: JTensor,
      segment_pos: JTensor | None,
      is_cross_attention: bool = False,
  ) -> JTensor:
    """Computes the value vector given the query of the current step.

    This function is used by autoregressive decoding.

    Args:
      query_vec: JTensor of shape [B, D] or [B, T, D] corresponding to query
        vector at index time_step.
      atten_mask: JTensor of shape [B/1, 1, S] or [B/1, 1, T, S]. atten_mask
        should have already taken care of causal masking for decoding, plus
        other maskings necessary.
      time_step: A scalar or JTensor. Current time-step, 0-based.
      segment_pos: An optional JTensor of shape [B] or [B, T]. Current position
        in the same segment. If unspecified, time_step will be used.
      is_cross_attention: Whether this is a cross-attention layer. Decoding
        states will not be updated in this case.

    Returns:
      encoded: JTensor of shape [B, D] which returns the attention output at
        `time_step`.
    """
    extend_one_step = len(query_vec.shape) == 2
    time_step = jnp.array(time_step)
    # Batch major.
    time_dim = 1
    assert time_step.ndim == 0
    # Project inputs to key, value and query. Query has shape [B, N, H],
    # key/value shapes [B, H]
    if extend_one_step:
      query_vec = self._shard_bd(query_vec)
    query_proj = self.query(query_vec)

    # TODO(b/290067837): Workaround for problems when batch dim is sharded
    # differently in bld and blnh.
    ap = self.activation_split_dims_mapping
    sharding = None
    if ap.bld and ap.blnh:
      ap_b = ap.bd[0] if ap.bd else ap.bld[0]
      sharding = [ap_b, ap.blnh[2], ap.blnh[3]]
    query_proj = base_layer.maybe_shard(
        query_proj, sharding, self.mesh_axis_names
    )

    if not is_cross_attention:
      key_proj = self.key(query_vec)
      value_proj = self.value(query_vec)
      if ap.bld and ap.blnh:
        if self.num_kv_heads == 1:
          ap_b = ap.bd[0] if ap.bd else ap.bld[0]
          sharding = [ap_b, ap.blnh[3]]
        value_proj = base_layer.maybe_shard(
            value_proj,
            sharding,
            self.mesh_axis_names,
        )
        key_proj = base_layer.maybe_shard(
            key_proj,
            sharding,
            self.mesh_axis_names,
        )

    def _extend_decode_state_and_shard_blh(name: str,
                                           extend_value: JTensor) -> JTensor:
      extended_state = self.extend_decode_state(
          name, extend_value, time_step, time_dim=time_dim)
      if self.num_kv_heads == 1:
        return self._shard_blh(extended_state)
      else:
        return self._shard_blnh(extended_state)

    # Update key, value state.
    value_state_name = 'value_state'
    key_state_name = 'key_state'
    if not is_cross_attention:
      _extend_decode_state_and_shard_blh(value_state_name, value_proj)
      _extend_decode_state_and_shard_blh(key_state_name, key_proj)

    if self.use_rotary_position_emb:
      key_state_name = (
          'key_post_rotary_pos_emb'
          if not self.consolidate_rope_key_state
          else 'key_state'
      )
      if segment_pos is None:
        position = jnp.broadcast_to(time_step, [query_vec.shape[0]])
      else:
        if segment_pos.ndim == 1:
          position = segment_pos
        else:
          # uses the position of last token.
          position = segment_pos[:, -1]
      query_proj = self.rotary_position_emb.extend_step(
          query_proj, position)
      if not is_cross_attention:
        key_shape = key_proj.shape
        if self.num_kv_heads == 1:
          key_proj = jnp.expand_dims(key_proj, axis=-2)
        key_proj = self.rotary_position_emb.extend_step(key_proj, position)
        if self.num_kv_heads == 1:
          key_proj = jnp.reshape(key_proj, key_shape)
        _extend_decode_state_and_shard_blh(key_state_name, key_proj)

    if self.relative_bias_tpl:
      # Relative bias uses time_step instead of segment_pos.
      relative_bias = self.relative_bias.extend_step(
          seq_length=self.decoding_state_sequence_length(), time_step=time_step)
    else:
      relative_bias = None

    encoded, atten_prob = self._dot_atten_one_step(
        query_proj,
        key_state_name,
        value_state_name,
        atten_mask,
        relative_bias,
        time_step,
    )
    # TODO(yonghui): return atten_probs back to the caller.
    del atten_prob
    # Post projection.
    if ap.bld and ap.blnh:
      # TODO(b/290067837): Workaround for problems when batch dim is sharded
      # differently in bld and blnh.
      ap_b = ap.bd[0] if ap.bd else ap.bld[0]
      encoded = base_layer.maybe_shard(
          encoded,
          [ap_b, ap.blnh[2], ap.blnh[3]],
          self.mesh_axis_names,
      )
    encoded = self.post(encoded)
    if extend_one_step:
      encoded = self._shard_bd(encoded)
    else:
      encoded = self._shard_bld(encoded)
    return encoded

  def transform_decode_state(self,
                             transform_fn: base_layer.DecodeStateTransformFn):
    """Transforms all decode state variables based on transform_fn."""
    batch_dim = 0
    time_dim = 1
    for name, state in self.variables[base_layer.DECODE_CACHE].items():
      if not isinstance(state, JTensor):
        continue
      new_state = transform_fn(state, batch_dim, time_dim)
      if self.num_kv_heads == 1:
        new_state = self._shard_blh(new_state)
      else:
        new_state = self._shard_blnh(new_state)
      self.update_decode_state(name, new_state)

  def lazy_broadcast_prefix(self, num_suffix_samples: int,
                            suffix_length: int) -> None:
    """Performs lazy prefix broadcast on the decoding states."""
    raise NotImplementedError(
        'lazy_broadcast_prefix should be used with'
        'MultiQueryDotProductAttentionLPB instead.')


class MultiQueryDotProductAttentionLPB(MultiQueryDotProductAttention):
  # TODO(pax-dev): Implement a single base class for all LPB type models.
  """Multi-query dot-product attention with lazy prefix broadcast.

  This has the same fprop logic as MultiQueryDotProductAttention except that
  it supports Lazy Prefix Broadcasting in decoding.
  """

  def _shard_blh(self, x: JTensor) -> JTensor:
    """Adds sharding annotations to tensors of shape [b, l, h]."""
    blh = self.activation_split_dims_mapping.blh
    if blh is None:
      return x
    # It is possible that we added prefix-broadcast dimensions.
    blh = [blh[0]] + [None] * (x.ndim - len(blh)) + list(blh[1:])
    return base_layer.maybe_shard(x, blh, self.mesh_axis_names)

  def decoding_state_sequence_length(self):
    """Returns the length of full decoding sequences including prefixes."""
    key_state_length = self.get_decode_state('key_state').shape[
        1 + self._broadcast_prefixes_count]
    return key_state_length + self._broadcast_prefix_length()

  def transform_decode_state(
      self, transform_fn: base_layer.DecodeStateTransformFn
  ):
    """Transforms all decode state variables based on transform_fn."""
    batch_dim = 0
    time_dim = self._broadcast_prefixes_count + 1
    for name, state in self.variables[base_layer.DECODE_CACHE].items():
      if not isinstance(state, JTensor):
        continue
      new_state = transform_fn(state, batch_dim, time_dim)
      new_state = self._shard_blh(new_state)
      self.update_decode_state(name, new_state)

  def _dot_atten_one_step(
      self,
      query: JTensor,
      key_state_name: str,
      value_state_name: str,
      atten_mask: JTensor,
      relative_bias: JTensor | None = None,
      time_step: JTensor | None = None,
  ) -> tuple[JTensor, JTensor]:
    """Dot attention function for queries with 1 time step with LPB.

    In the shapes listed below, `...` means potential sample dims added for lazy
    broadcast prefixes.

    Args:
      query: JTensor of shape [B, ..., N, H] or [B, ..., T, N, H].
      key_state_name: Name of the decoding key state variable.
      value_state_name: Name of the decoding value state variable.
      atten_mask: JTensor of shape [1|B, 1, S] or [1|B, 1, T, S] which is a mask
        that is applied to prevent attention between unwanted pairs. This has
        already been converted into large negative logits. The first dimension
        is allowed to be of size 1, if the mask is shared by all items in the
        batch (e.g., only a causal mask).
      relative_bias: Relative bias of shape [1|B, N, 1, S].
      time_step: The time step tensor.

    Returns:
      encoded: JTensor of shape [B, ..., N, H] or [B, ..., T, N, H]
      atten_probs: JTensor of shape [B, N, T, S].
    """
    del time_step
    pfx_count = self._broadcast_prefixes_count
    # When query has shape of [B, ..., N, H], will apply extend_step to a single
    # token per batch, normal autoregressive decoding logic is applied.
    #
    # When query has shape of [B, ..., T, N, H], will apply extend_step to
    # T tokens per batch. This is used in suffix scoring of T tokens after
    # autoregressive decoding.
    extend_one_step = (len(query.shape) == pfx_count + 3)

    batch_dims = self.get_decode_state(key_state_name).shape[:1 + pfx_count]
    rb_batched = False
    if relative_bias is not None:
      rb_batched = relative_bias.shape[0] > 1
    if rb_batched:
      relative_bias = jnp.reshape(relative_bias,
                                  batch_dims + relative_bias.shape[1:])
    am_batched = atten_mask.shape[0] > 1
    if am_batched:
      atten_mask = jnp.reshape(atten_mask, batch_dims + atten_mask.shape[1:])

    def _pre_softmax(layer, batched, batched_slice, non_batched_slice, states):
      del layer
      k = states[0]
      q = batched
      if am_batched:
        am, *batched_slice = batched_slice
      else:
        am, *non_batched_slice = non_batched_slice
      if rb_batched:
        rb, *batched_slice = batched_slice
      else:
        rb, *non_batched_slice = non_batched_slice
      k = self._shard_blh(k)
      # q is 3d.
      if extend_one_step:
        q = self._shard_bnh(q)
      else:
        q = self._shard_blnh(q)

      b, s, h = k.shape
      n = self.num_heads
      if extend_one_step:
        base_layer.assert_has_shape(q, [b, n, h])
        base_layer.assert_has_shape(am, [-1, 1, s])
      else:
        base_layer.assert_has_shape(q, [b, -1, n, h])
        base_layer.assert_has_shape(am, [-1, 1, -1, s])
      asserts.in_set(am.shape[0], [b, 1])

      q = self._scale_query(q)
      if extend_one_step:
        logits = self.qk_einsum('BNH,BSH->BNS', q, k)
      else:
        logits = self.qk_einsum('BTNH,BSH->BNTS', q, k)
      if rb is not None:
        base_layer.assert_has_shape(rb, [-1, n, -1, s])
        asserts.in_set(rb.shape[0], [b, 1])
        if rb.shape[2] == 1:
          rb = jnp.squeeze(rb, axis=2)
        logits += rb
      logits = self._cap_logits(logits)
      # Attention softmax is always carried out in fp32.
      logits = logits.astype(jnp.float32)
      # Apply attention masking
      padded_logits = logits + am.astype(jnp.float32)
      return padded_logits

    batched_to_slice = []
    batched_to_slice_tdims = []
    non_batched_to_slice = []
    non_batched_to_slice_tdims = []
    if extend_one_step:
      am_tdim = 2
      concat_dim = 2
    else:
      am_tdim = 3
      concat_dim = 3

    if am_batched:
      batched_to_slice.append(atten_mask)
      batched_to_slice_tdims.append(am_tdim)
    else:
      non_batched_to_slice.append(atten_mask)
      non_batched_to_slice_tdims.append(am_tdim)
    if rb_batched:
      batched_to_slice.append(relative_bias)
      batched_to_slice_tdims.append(3)
    else:
      non_batched_to_slice.append(relative_bias)
      non_batched_to_slice_tdims.append(3)

    def _concat_logits(chunks):
      if len(chunks) == 1:
        return chunks[0]
      return jnp.concatenate(chunks, axis=pfx_count + concat_dim)

    padded_logits = self._run_with_all_decode_state_chunks(
        _pre_softmax, query, batched_to_slice, batched_to_slice_tdims,
        non_batched_to_slice, non_batched_to_slice_tdims, [key_state_name],
        _concat_logits)

    # Of shape [b, ..., n, s]
    key_dtype = self.get_decode_state(key_state_name).dtype
    if self.attention_extra_logit is None:
      probs = jax.nn.softmax(padded_logits, axis=-1).astype(key_dtype)
    else:
      probs = jnp.exp(
          self._log_softmax_with_extra_logit(padded_logits)).astype(key_dtype)

    # Compute the attention context.
    def _post_softmax(layer, batched, ps, non_batched, states):
      del layer, batched, non_batched
      v = self._shard_blh(states[0])
      if extend_one_step:
        return self._shard_bnh(self.pv_einsum('BNS,BSH->BNH', ps, v))
      return self._shard_blnh(self.pv_einsum('BNTS,BSH->BTNH', ps, v))

    # Use sum as result combiner since the time dimension is a contracting dim.
    encoded = self._run_with_all_decode_state_chunks(_post_softmax, [], probs,  # pytype: disable=wrong-arg-types  # jax-ndarray
                                                     am_tdim, [], [],
                                                     [value_state_name], sum)
    return encoded, probs  # pytype: disable=bad-return-type  # jax-ndarray

  @nn.nowrap
  def extend_decode_state(self, name: str, value: JTensor, time_step: JTensor,
                          time_dim: int) -> JTensor:
    """Extends decode state at time_step.

    The decode state is batch major with shape [B, T, H].
    Args:
      name: Variable name in decoder cache.
      value: Value to extend at time step.
      time_step: A scalar. Time step to update the state.
      time_dim: Time dimension in the decode state.

    Returns:
      Updated decode cache state of that variable.
    """
    if len(value.shape) == time_dim + 1:
      extend_value = jnp.expand_dims(value, axis=time_dim)
    else:
      extend_value = value
    indices = [0] * extend_value.ndim
    indices[time_dim] = time_step.astype(jnp.int32)
    state = self.get_decode_state(name)
    assert state is not None
    new_state = jax.lax.dynamic_update_slice(state,
                                             extend_value.astype(state.dtype),
                                             indices)
    self.update_decode_state(name, new_state)
    return new_state

  def _broadcast_prefix_length(self):
    """Returns the sum of lengths of all lazy broadcast prefixes."""
    prefix_length = 0
    for i in range(self._broadcast_prefixes_count):
      prefix_length += self.get_variable(PREFIX_DECODE_CACHE,
                                         f'key_state_{i}_pfx').shape[i + 1]
    return prefix_length

  def _vmap_on_broadcast_prefixes(self, fn: attentions.FnOnDecodeStateChunk,
                                  chunk_id: int,
                                  args_time_dims: NestedInt,
                                  broadcast_args_time_dims: NestedInt):
    """Transforms `fn` using vmap for a decoding state chunk."""

    # Wraps fn with slicing on args_to_slice and broadcast_args_to_slice.
    def _sliced_fn(layer, args, args_to_slice, broadcast_args_to_slice, states):
      sliced = jax.tree.map(
          lambda x, d: self._slice_decode_chunk(x, chunk_id, d),
          args_to_slice,
          args_time_dims,
      )
      broadcast_sliced = jax.tree.map(
          lambda x, d: (
              None if x is None else self._slice_decode_chunk(x, chunk_id, d)
          ),
          broadcast_args_to_slice,
          broadcast_args_time_dims,
          is_leaf=lambda x: x is None,
      )
      return fn(layer, args, sliced, broadcast_sliced, states)

    broadcast_dim_sizes = self.get_decode_state(
        'key_state').shape[1:1 + self._broadcast_prefixes_count]
    # There can be multiple lazy-broadcast sample dimensions, and we vmap one
    # dimension at a time. `args` and `args_to_slice` have shape
    # [b, num_samples0, num_samples1, ..., inner_dims]; after each vmap, one
    # num_samples dimension will be removed for `fn`.
    vfns = [_sliced_fn]
    # The loop works from inner vmap to outer vmap.
    for i in range(self._broadcast_prefixes_count):
      # args, args_to_slice have the sample dimensions. broadcast_args_to_slice
      # does not have them.
      in_axes = [i + 1, i + 1, None]
      if chunk_id > i:
        # This chunk has the current sample dimension to vmap. Since outer vmaps
        # (to be done at later iterations in this for loop) will handle sample
        # dimensions AFTER the current one, i + 1 is still the current vmap
        # even if there are outer vmaps. (1 in `i + 1` is the original batch
        # dim.)
        in_axes.append(i + 1)
      else:
        # This chunk does not have the current sample dimension to vmap.
        in_axes.append(None)
      # Do not vmap any state; they are handle explicitly as the `states`
      # argument in `fn`.
      vmapped_fn = nn.vmap(
          vfns[-1],
          variable_axes={
              base_layer.PARAMS: None,
              base_layer.DECODE_CACHE: None,
              base_layer.PREFIX_DECODE_CACHE: None,
              base_layer.HYPER_PARAMS: None,
          },
          in_axes=tuple(in_axes),
          out_axes=i + 1,
          split_rngs={base_layer.PARAMS: True, base_layer.RANDOM: True},
          axis_size=broadcast_dim_sizes[i],
      )
      vfns.append(vmapped_fn)
    return vfns[-1]

  def _left_concat_decode_state(self, state_name: str,
                                max_prefix_size: int) -> JTensor:
    """Left-concats the current decode state with prefixes (if any)."""
    state = self.get_decode_state(state_name)
    pfx_count = self._broadcast_prefixes_count
    if pfx_count == 0:
      return state
    batch_dims = self.get_decode_state(state_name).shape[:1 + pfx_count]
    windows = [state]
    prefix_window_size = max_prefix_size
    for i in range(pfx_count):
      if prefix_window_size == 0:
        break
      chunk_id = pfx_count - i - 1
      pfx = self.get_variable(PREFIX_DECODE_CACHE,
                              f'{state_name}_{chunk_id}_pfx')
      pfx_len = pfx.shape[chunk_id + 1]
      subwindow_len = min(pfx_len, prefix_window_size)
      prefix_window_size -= subwindow_len
      pfx = jax.lax.slice_in_dim(
          pfx, pfx_len - subwindow_len, pfx_len, axis=chunk_id + 1)
      pfx = jnp.reshape(
          pfx,
          batch_dims[:chunk_id + 1] + (1,) * (i + 1) + pfx.shape[chunk_id + 1:])
      pfx = jnp.broadcast_to(pfx, batch_dims + pfx.shape[len(batch_dims):])
      windows = [pfx] + windows
    return jnp.concatenate(windows, axis=pfx_count + 1)

  def right_align_decode_state_with_prefix(
      self, max_prefix_size: int,
      right_align_fn: base_layer.DecodeStateTransformFn) -> None:
    """Right aligns decode state with prefix decode states.

    Args:
      max_prefix_size: Max prefix length of the decode state.
      right_align_fn: Right align function for decode state.
    """
    batch_dim = 0
    time_dim = 1
    prev_pfx_count = self._broadcast_prefixes_count
    for name, state in self.variables[base_layer.DECODE_CACHE].items():
      if not isinstance(state, JTensor):
        continue
      # Left concat decode state with prefixes.
      new_state = self._left_concat_decode_state(name, max_prefix_size)

      # Merge batch dims.
      state_shape = list(new_state.shape)
      final_state_shape = state_shape.copy()
      batch_size = math.prod(state_shape[:prev_pfx_count + 1])
      state_shape = [batch_size] + state_shape[prev_pfx_count + 1:]
      new_state = jnp.reshape(new_state, state_shape)
      # Right align decode state.
      new_state = right_align_fn(new_state, batch_dim, time_dim)
      # Reshape back.
      new_state = jnp.reshape(new_state, final_state_shape)

      self.update_decode_state(name, new_state)

      # Set seq_len to 0 in prefix decode state.
      for i in range(prev_pfx_count):
        prefix_name = f'{name}_{i}_pfx'
        assert self.is_mutable_collection(PREFIX_DECODE_CACHE)
        assert prefix_name in self.variables[PREFIX_DECODE_CACHE]
        prefix_state = self.get_variable(PREFIX_DECODE_CACHE, prefix_name)
        prefix_state_shape = list(prefix_state.shape)
        prefix_state_shape[i+1] = 0
        new_prefix_state = jnp.zeros(prefix_state_shape, prefix_state.dtype)

        self.put_variable(PREFIX_DECODE_CACHE, prefix_name, new_prefix_state)

  def _decode_state_chunk_length(self, chunk_id: int) -> int:
    """Returns the length of a decode state chunk (prefix or current)."""
    t_dim = chunk_id + 1
    if chunk_id == self._broadcast_prefixes_count:
      # Current state, non-prefix.
      return self.get_decode_state('key_state').shape[t_dim]
    return self.get_variable(PREFIX_DECODE_CACHE,
                             f'key_state_{chunk_id}_pfx').shape[t_dim]

  def _slice_decode_chunk(self, x: JTensor, chunk_id: int, dim: int) -> JTensor:
    """Slices a full-sequence tensor for a decode state chunk."""
    pfx_count = self._broadcast_prefixes_count
    start = 0
    for i in range(min(pfx_count, chunk_id)):
      t_dim = i + 1
      start += self.get_variable(PREFIX_DECODE_CACHE,
                                 f'key_state_{i}_pfx').shape[t_dim]
    limit = start + self._decode_state_chunk_length(chunk_id)
    return jax.lax.slice_in_dim(x, start, limit, axis=dim)

  def _run_with_all_decode_state_chunks(
      self, fn: attentions.FnOnDecodeStateChunk, chunk_inputs: NestedJTensor,
      args_to_slice: NestedJTensor, args_time_dims: NestedInt,
      broadcast_args_to_slice: NestedJTensor,
      broadcast_args_time_dims: NestedInt, state_names: Sequence[str],
      combine_results: Callable[[Sequence[NestedJTensor]], NestedJTensor]
  ) -> NestedJTensor:
    """Runs `fn` on all decoding state chunks, then combine them."""
    pfx_count = self._broadcast_prefixes_count
    results = []
    for i in range(pfx_count + 1):
      # Get the relevant states for `fn`.
      if i == pfx_count:
        states = [self.get_decode_state(s) for s in state_names]
      else:
        states = [
            self.get_variable(PREFIX_DECODE_CACHE, f'{s}_{i}_pfx')
            for s in state_names
        ]
      # Run one chunk with vmaps.
      results.append(
          self._vmap_on_broadcast_prefixes(
              fn, i, args_time_dims,
              broadcast_args_time_dims)(self, chunk_inputs, args_to_slice,
                                        broadcast_args_to_slice, states))
    return combine_results(results)

  def extend_step(
      self,
      query_vec: JTensor,
      *,
      atten_mask: JTensor,  # pytype: disable=signature-mismatch  # overriding-parameter-name-checks
      time_step: JTensor,
      segment_pos: JTensor | None,
      is_cross_attention: bool = False,
  ) -> JTensor:
    """Computes the value vector given the query of the current step using LPB.

    This function is used by autoregressive decoding.

    Args:
      query_vec: JTensor of shape [B, D] corresponding to query vector at index
        time_step or JTensor of shape [B, T, D] to support extend n steps.
      atten_mask: JTensor of shape [1|B, 1, S] or of shape [1|B, 1, T, S] to
        support extend n steps. atten_mask should have already taken care of
        causal masking for decoding, plus other maskings necessary.
      time_step: A scalar or JTensor. Current time-step, 0-based.
      segment_pos: An optional JTensor of shape [B]. Current position in the
        same segment. If unspecified, time_step will be used.
      is_cross_attention: Whether this is a cross-attention layer. Decoding
        states will not be updated in this case.

    Returns:
      encoded: JTensor of shape [B, D] which returns the attention output at
        `time_step`.
    """
    # When query has shape of [B, D], will apply extend_step to a single
    # token per batch, normal autoregressive decoding logic is applied.
    #
    # When query has shape of [B, T, D], will apply extend_step to
    # T tokens per batch. This is used in suffix scoring of T tokens after
    # autoregressive decoding.
    extend_one_step = (len(query_vec.shape) == 2)
    # Batch major. Reshape the input batch dim to match the decoding state if
    # there are lazy broadcast prefixes.
    pfx_count = self._broadcast_prefixes_count
    batch_dims = self.get_decode_state('key_state').shape[:1 + pfx_count]
    if pfx_count > 0:
      query_vec = jnp.reshape(query_vec, batch_dims + query_vec.shape[1:])
      if segment_pos is not None:
        segment_pos = jnp.reshape(segment_pos,
                                  batch_dims + segment_pos.shape[1:])

    time_step = jnp.array(time_step)
    assert time_step.ndim == 0

    # vmap a function on the samples dimensions in lazy broadcast prefixes. This
    # is for functions that do not touch the decoding states.
    def _vmap_no_state(fn):
      vfns = [fn]
      for i in range(pfx_count):
        vmapped_fn = nn.vmap(
            vfns[-1],
            variable_axes={
                base_layer.PARAMS: None,
                base_layer.DECODE_CACHE: None,
                base_layer.PREFIX_DECODE_CACHE: None,
                base_layer.HYPER_PARAMS: None,
                base_layer.NON_TRAINABLE: None,
            },
            in_axes=i + 1,
            out_axes=i + 1,
            split_rngs={base_layer.PARAMS: True, base_layer.RANDOM: True},
            axis_size=batch_dims[1 + i],
        )
        vfns.append(vmapped_fn)
      return vfns[-1]

    def _proj_qkv(layer, q):
      if self.combine_qkv:
        # Project inputs to key, value and query using a combined weight for
        # faster performance on TPU.
        query_proj, key_proj, value_proj = layer.combined_qkv(q)
      else:
        # Project inputs to key, value and query. Each has shape [B, N, H].
        key_proj = None
        value_proj = None
        if not is_cross_attention:
          key_proj = layer.key(q)
          value_proj = layer.value(q)
        query_proj = layer.query(q)
      return query_proj, key_proj, value_proj

    query_proj, key_proj, value_proj = _vmap_no_state(_proj_qkv)(
        self, query_vec)
    prefix_length = self._broadcast_prefix_length()

    def _extend_decode_state_and_shard(name: str,
                                       extend_value: JTensor) -> JTensor:
      extended_state = self.extend_decode_state(
          name, extend_value, time_step - prefix_length, time_dim=1 + pfx_count)
      return self._shard_blh(extended_state)

    key_state_name = 'key_state'
    value_state_name = 'value_state'
    if not is_cross_attention:
      # Update key_state
      _extend_decode_state_and_shard(key_state_name, key_proj)
      # Update value state.
      _extend_decode_state_and_shard(value_state_name, value_proj)

    # Apply rotary position embeddings.
    # Paper: https://arxiv.org/abs/2104.09864.
    if self.use_rotary_position_emb:
      if segment_pos is None:
        position = jnp.broadcast_to(time_step, batch_dims)
      else:
        position = segment_pos

      def _rotary(layer, q, k, pos):
        if not is_cross_attention:
          k = jnp.expand_dims(k, axis=-2)
        key_proj = None
        if len(query_vec.shape) == pfx_count + 2:
          query_proj = layer.rotary_position_emb.extend_step(q, pos)
          if not is_cross_attention:
            key_proj = layer.rotary_position_emb.extend_step(k, pos)
        else:
          # If it is extending n steps, uses a vmap to do the computation.
          def _get_rotary(q, pos):
            return layer.rotary_position_emb.extend_step(q, pos)

          query_proj = jax.vmap(_get_rotary, in_axes=1, out_axes=1)(q, pos)
          if not is_cross_attention:
            key_proj = jax.vmap(_get_rotary, in_axes=1, out_axes=1)(k, pos)
        if not is_cross_attention:
          key_proj = jnp.squeeze(key_proj, axis=-2)
        return query_proj, key_proj

      query_proj, key_proj = _vmap_no_state(_rotary)(self,
                                                     query_proj,
                                                     key_proj,
                                                     position)

      # Update key post rotary position embedding in the cache.
      key_state_name = (
          'key_post_rotary_pos_emb'
          if not self.consolidate_rope_key_state
          else 'key_state'
      )
      if not is_cross_attention:
        _extend_decode_state_and_shard(key_state_name, key_proj)

    if self.relative_bias_tpl:
      # Relative bias uses time_step instead of segment_pos.
      if not extend_one_step:
        raise NotImplementedError(
            'MultiQueryAttention does not support extend n steps with '
            'relative bias.')
      relative_bias = self.relative_bias.extend_step(
          seq_length=self.decoding_state_sequence_length(), time_step=time_step)
    else:
      relative_bias = None

    encoded, atten_prob = self._dot_atten_one_step(query_proj,
                                                   key_state_name,
                                                   value_state_name, atten_mask,
                                                   relative_bias)
    # TODO(yonghui): return atten_probs back to the caller.

    del atten_prob
    # Post projection.
    if pfx_count > 0:
      encoded = jnp.reshape(encoded, (-1,) + encoded.shape[1 + pfx_count:])
    encoded = self.post(encoded)
    if extend_one_step:
      encoded = self._shard_bd(encoded)
    else:
      encoded = self._shard_bld(encoded)
    return encoded

  @property
  def _broadcast_prefixes_count(self):
    """Returns the number of prefixes created for lazy broadcast."""
    if PREFIX_DECODE_CACHE not in self.variables:
      return 0
    count = 0
    while f'key_state_{count}_pfx' in self.variables[PREFIX_DECODE_CACHE]:
      count += 1
    return count

  def lazy_broadcast_prefix(self, num_suffix_samples: int,
                            suffix_length: int) -> None:
    """Performs lazy prefix broadcast on the decoding states.

    Current decoding states will be moved to PREFIX_DECODE_CACHE. New decoding
    state will be created for the suffixes with multiple samples sharing
    previous prefixes. After this call, new extend_step will use a batch size
    num_suffix_samples times larger than before, which is logically 2 merged
    dimensions [previous batch dim, new num_samples dim].

    Args:
      num_suffix_samples: Number of samples that will share the same previous
        decoding state.
      suffix_length: The length of the new suffix samples.
    """
    prev_pfx_count = self._broadcast_prefixes_count

    for name, state in self.variables[base_layer.DECODE_CACHE].items():
      assert self.is_mutable_collection(PREFIX_DECODE_CACHE)
      self.put_variable(PREFIX_DECODE_CACHE, f'{name}_{prev_pfx_count}_pfx',
                        state)
      suffix_shape = state.shape[:prev_pfx_count + 1] + (
          num_suffix_samples, suffix_length) + state.shape[prev_pfx_count + 2:]
      self.update_decode_state(name, jnp.zeros(suffix_shape, dtype=state.dtype))
