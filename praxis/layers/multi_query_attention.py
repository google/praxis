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

"""Multi-Query Attention layers."""

from typing import Dict, Optional, Tuple, Union

from flax import linen as nn
import jax
from jax import numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from praxis import base_layer
from praxis import pytypes
from praxis.layers import attentions
from praxis.layers import embedding_softmax
from praxis.layers import stochastics

WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams
sub_config_field = base_layer.sub_config_field
JTensor = pytypes.JTensor

SplitDimsMapping = pytypes.SplitDimsMapping
BaseHParams = base_layer.BaseLayer.HParams
BaseWtShardingHParams = base_layer.BaseLayer.WeightShardingHParams
BaseActShardingHParams = base_layer.BaseLayer.ActivationShardingHParams


class OneHeadedAttentionProjection(base_layer.BaseLayer):
  """Layer that computes projection with one head.

  This layer is expected to be used within MultiQueryAttention below.
  """

  class HParams(BaseHParams):
    """Associated hyper-params for this layer class.

    Attributes:
      input_dim: Input dimension.
      output_dim: Size of output.
      use_bias: Whether to add bias in projection or not.
    """
    input_dim: int = 0
    output_dim: int = 0
    use_bias: bool = True

  def setup(self) -> None:
    p = self.hparams
    wp = p.weight_split_dims_mapping
    if p.mesh_shape is not None:
      assert wp.wt is not None, ('Must provide sharding annotations for the '
                                 'weights if mesh shape is provided')
    wt = wp.wt
    pc_shape = [p.input_dim, p.output_dim]
    pc = WeightHParams(
        shape=pc_shape, mesh_shape=p.mesh_shape, tensor_split_dims_mapping=wt)
    self.create_variable('w', pc)
    if p.use_bias:
      if p.mesh_shape is not None:
        bias_split_dims_mapping = [wp.wt[1]]
      else:
        bias_split_dims_mapping = None
      pc_bias = WeightHParams(
          shape=[p.output_dim],
          init=WeightInit.Constant(0.0),
          mesh_shape=p.mesh_shape,
          tensor_split_dims_mapping=bias_split_dims_mapping)
      self.create_variable('b', pc_bias)

  def __call__(self, inputs: JTensor) -> JTensor:
    """Computes the multi headed projection for inputs.

    Args:
      inputs: A JTensor of shape [..., p.input_dim].

    Returns:
      The projected JTensor with shape [..., p.output_dim].
    """
    p = self.hparams
    theta = self.theta

    shape = inputs.shape
    inputs = self._cast_to_fprop_dtype(inputs)
    w = theta.w

    assert shape[-1] == p.input_dim, (
        f'Expecting shape[-1] == p.input_dim, {shape[-1]} != {p.input_dim}')
    eqn = '...D,DH->...H'
    ret = jnp.einsum(eqn, inputs, w)
    if p.use_bias:
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
  """

  class HParams(BaseHParams):
    """Associated hyper-params for this layer class.

    Attributes:
      input_dim: An integer or a dict of integer values as number of input
        nodes. If input_dim is a dict, keys must be key, value and query.
      hidden_dim: Number of hidden nodes.
      num_heads: Number of attention heads.
      dim_per_head: Dimension of each attention head. If None then dim_per_head
        == hidden_dim // num_heads.
      dropout_tpl: Parameterization for the dropout layer.
      atten_dropout_prob: Probability at which we apply dropout to the attention
        weights.
      proj_tpl: Parameterization for the query projection_tpl layer.
      headless_proj_tpl: Parameterization for the key/value projection_tpl
        layer.
      use_bias: Whether to use bias for projection_tpl layers.
      output_proj_use_nhd_shape: Whether to use NHD variable shape in output
        projection layer.
      internal_enable_query_scale: Internal. Enable scaling of query vector.
      atten_logit_cap: Cap the absolute values of logits by tanh. Enabled when a
        positive value is specified. May not be supported by a subclass.
      use_rotary_position_emb: Whether to add rotary position embedding to the
        queries and keys before computing self attention scores. This was
        proposed in https://arxiv.org/abs/2104.09864.
      relative_bias_tpl: Optional parameterization of relative bias.
      attention_extra_logit: Extra logit for attention softmax.
      combine_qkv: Whether to combine qkv tensor for optimizing qkv input
        gradient computation with SPMD. Only supports self-attention.
      Note: dconv_qkv and ngrammer are not supported.
    """
    input_dim: Union[int, Dict[str, int]] = 0
    hidden_dim: int = 0
    num_heads: int = 1
    dim_per_head: Optional[int] = None
    dropout_tpl: BaseHParams = sub_config_field(stochastics.Dropout.HParams)
    atten_dropout_prob: float = 0.0
    proj_tpl: BaseHParams = sub_config_field(
        attentions.AttentionProjection.HParams)
    headless_proj_tpl: BaseHParams = sub_config_field(
        OneHeadedAttentionProjection.HParams)
    internal_gshard_gaussian_init: bool = False
    use_bias: bool = True
    output_proj_use_nhd_shape: bool = False
    internal_enable_query_scale: bool = True
    atten_logit_cap: float = 0.0
    use_rotary_position_emb: bool = False
    relative_bias_tpl: Optional[BaseHParams] = base_layer.sub_config_field(None)
    attention_extra_logit: Optional[float] = None
    dconv_qkv: bool = False
    combine_qkv: bool = False

  # SPMD partition related params.
  #
  # d - model_dim
  # n - num_heads
  # h - attention_dim_per_heads
  # b - batch_size
  # l - seq_len

  class WeightShardingHParams(BaseWtShardingHParams):
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

  class ActivationShardingHParams(BaseActShardingHParams):
    """Represents how intermediate values should be partitioned across a mesh.

    Attributes:
      blnh: Mesh split for query, and encoded tensors with the shape of
        [batch_size, seq_len, num_heads, dim_per_head].
      blh: Mesh split key, value, and encoded tensors with the shape of
        [batch_size, seq_len, dim_per_head].
      bld: Mesh split for output after post projection with the shape of
        [batch_size, seq_len, model_dim].
    """
    blnh: SplitDimsMapping = None
    blh: SplitDimsMapping = None
    bld: SplitDimsMapping = None

  def setup(self) -> None:
    p = self.hparams
    wp = p.weight_split_dims_mapping
    assert p.input_dim, 'input_dim is {}'.format(p.input_dim)
    assert p.hidden_dim, 'hidden_dim is {}'.format(p.hidden_dim)

    assert not p.dconv_qkv
    assert not p.combine_qkv

    dim_per_head = p.dim_per_head
    if dim_per_head is None:
      dim_per_head = p.hidden_dim // p.num_heads
      assert dim_per_head * p.num_heads == p.hidden_dim, (
          f'{dim_per_head} * {p.num_heads} != {p.hidden_dim}')

    if p.mesh_shape is not None:
      assert p.weight_split_dims_mapping is not None
      assert p.activation_split_dims_mapping is not None

    if isinstance(p.input_dim, dict):
      key_input_dim = p.input_dim['key']
      value_input_dim = p.input_dim['value']
      query_input_dim = p.input_dim['query']
      assert key_input_dim, f'key_input_dim is {key_input_dim}'
      assert query_input_dim, f'query_input_dim is {query_input_dim}'
    else:
      key_input_dim = p.input_dim
      value_input_dim = p.input_dim
      query_input_dim = p.input_dim

    def project_input(input_dim):
      proj_p = p.proj_tpl.clone().set(
          input_dim=input_dim,
          num_heads=p.num_heads,
          dim_per_head=dim_per_head,
          use_bias=p.use_bias)
      proj_p.weight_split_dims_mapping.wt = wp.proj
      return proj_p

    def project_input_no_heads(input_dim):
      proj_p = p.headless_proj_tpl.clone().set(
          input_dim=input_dim, output_dim=dim_per_head, use_bias=p.use_bias)
      proj_p.weight_split_dims_mapping.wt = wp.proj_headless
      return proj_p

    self.create_child('key', project_input_no_heads(key_input_dim))
    self.create_child('query', project_input(query_input_dim))
    self.create_child('value', project_input_no_heads(value_input_dim))

    if p.use_rotary_position_emb:
      pos_emb_p = embedding_softmax.RotaryPositionalEmbedding.HParams()
      pos_emb_p.embedding_dims = dim_per_head
      self.create_child('rotary_position_emb', pos_emb_p)

    if p.relative_bias_tpl is not None:
      relative_bias_p = p.relative_bias_tpl.clone()
      relative_bias_p.num_heads = p.num_heads
      self.create_child('relative_bias', relative_bias_p)

    self.create_child(
        'atten_dropout',
        p.dropout_tpl.clone().set(keep_prob=1.0 - p.atten_dropout_prob))

    # Setting is_output_projection=True to set the projection direction
    # from hidden dim to input dim. Output projection follows query_input_dim.
    post_proj_p = p.proj_tpl.clone().set(
        input_dim=query_input_dim,
        num_heads=p.num_heads,
        dim_per_head=dim_per_head,
        is_output_projection=True,
        use_bias=p.use_bias,
        use_nhd_shape=p.output_proj_use_nhd_shape)
    post_proj_p.weight_split_dims_mapping.wt = wp.proj

    self.create_child('post', post_proj_p)

  def _shard_bnh(self, x: JTensor) -> JTensor:
    """Shards tensors of shape [b, n, h].

    Single step decoder output are of shape [b, n, h].

    Args:
      x: A tensor of shape [b, n, h]

    Returns:
      x with proper sharding annotations.
    """
    p = self.hparams
    ap = p.activation_split_dims_mapping
    if p.mesh_axis_names is None:
      return x
    if ap.blnh is None:
      return x
    assert len(ap.blnh) == 4
    bnh = [ap.blnh[0], ap.blnh[2], ap.blnh[3]]
    return base_layer.maybe_shard(x, bnh, p.mesh_axis_names)

  def _shard_blnh(self, x: JTensor) -> JTensor:
    """Adds sharding annotations to tensors of shape [b, l, n, h]."""
    p = self.hparams
    ap = p.activation_split_dims_mapping
    return base_layer.maybe_shard(x, ap.blnh, p.mesh_axis_names)

  def _shard_blh(self, x: JTensor) -> JTensor:
    """Adds sharding annotations to tensors of shape [b, l, h]."""
    p = self.hparams
    ap = p.activation_split_dims_mapping
    return base_layer.maybe_shard(x, ap.blh, p.mesh_axis_names)

  def _shard_bld(self, x: JTensor) -> JTensor:
    """Adds sharding annotations to tensors of shape [b, l, d]."""
    p = self.hparams
    ap = p.activation_split_dims_mapping
    return base_layer.maybe_shard(x, ap.bld, p.mesh_axis_names)

  def _shard_bd(self, x: JTensor) -> JTensor:
    """Adds sharding annotations to tensors of shape [b, d]."""
    p = self.hparams
    ap = p.activation_split_dims_mapping
    if p.mesh_axis_names is None:
      return x
    if ap.bld is None:
      return x
    assert len(ap.bld) == 3
    bd = [ap.bld[0], ap.bld[2]]
    return base_layer.maybe_shard(x, bd, p.mesh_axis_names)

  def _scale_query(self, query: JTensor) -> JTensor:
    """Scales the query vector if enabled."""
    p = self.hparams
    if p.internal_enable_query_scale:
      query *= (p.hidden_dim // p.num_heads)**-0.5
    return query

  def _cap_logits(self, logits: JTensor) -> JTensor:
    """When enabled, caps the logits by p.atten_logit_cap with tanh."""
    p = self.hparams
    if not p.atten_logit_cap or p.atten_logit_cap <= 0.:
      return logits
    cap = jnp.array(p.atten_logit_cap, dtype=logits.dtype)
    # Note that since this caps the negative side as well, caller
    # must defer the pad-with-very-negative-logits logic to after
    # this function returns.
    logits = cap * jnp.tanh(logits / cap)
    return logits

  def _log_softmax_with_extra_logit(self, logits: JTensor) -> JTensor:
    """Computes log softmax with extra logit.

    self.hparams.attention_extra_logit is a user defined float value that
    helps to stabilize logit values so that they don't drift too much from it.

    Args:
      logits: input logit tensor

    Returns:
      Log softmax with extra logit value.
    """
    # Applies stop_gradient to max_logit instead of logits.
    max_logit = jnp.max(jax.lax.stop_gradient(logits), axis=-1, keepdims=True)
    extra_logit = self.hparams.attention_extra_logit
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
    logits = jnp.einsum('BNTH,BSH->BNTS', query, key)
    return logits

  def _dot_atten(
      self,
      query: JTensor,
      key: JTensor,
      value: JTensor,
      atten_mask: JTensor,
      relative_bias: Optional[JTensor] = None) -> Tuple[JTensor, JTensor]:
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
    # Add key sharding annotations.
    p = self.hparams
    query = self._shard_blnh(query)
    key = self._shard_blh(key)
    value = self._shard_blh(value)

    b, s, n, h = query.shape
    base_layer.assert_has_shape(value, [b, s, h])
    base_layer.assert_has_shape(query, [b, -1, n, h])
    t = query.shape[1]
    # If only padding bias is supplied, then atten_mask can be [B, 1, 1, S]
    # since each target token is prohibited from attending to the same set of
    # source tokens. In this case tiling is inefficient and unnecessary.
    # If there is no padding mask, and only causal mask then the shape can be
    # [1, 1, T, S]
    base_layer.assert_has_shape(atten_mask, [-1, 1, -1, s])
    assert atten_mask.shape[2] in [1, t]
    assert atten_mask.shape[0] in [1, b]
    query = self._scale_query(query)
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
    padded_logits = logits + atten_mask.astype(jnp.float32)
    if p.attention_extra_logit is None:
      probs = jax.nn.softmax(padded_logits, axis=-1).astype(key.dtype)
    else:
      probs = jnp.exp(self._log_softmax_with_extra_logit(padded_logits)).astype(
          key.dtype)
    # Apply attention dropout.
    probs = self.atten_dropout(probs)
    # Compute the attention context.
    encoded = jnp.einsum('BNTS,BSH->BNTH', probs, value)
    encoded = encoded.transpose(0, 2, 1, 3)
    encoded = checkpoint_name(encoded, 'context')
    encoded = self._shard_blnh(encoded)
    return encoded, probs

  def decoding_state_sequence_length(self):
    """Returns the length of full decoding sequences."""
    return self.get_decode_state('key_state').shape[1]

  def _dot_atten_one_step(self,
                          query: JTensor,
                          key_state_name: str,
                          value_state_name: str,
                          atten_mask: JTensor,
                          relative_bias: Optional[JTensor] = None) -> JTensor:
    """Dot attention function for queries with 1 time step.

    Args:
      query: JTensor of shape [B, N, H].
      key_state_name: Name of the decoding key state variable.
      value_state_name: Name of the decoding value state variable.
      atten_mask: JTensor of shape [1/B, 1, S] which is a mask that is applied
        to prevent attention between unwanted pairs. This has already been
        converted into large negative logits. The first dimension is allowed to
        be of size 1, if the mask is shared by all items in the batch (e.g.,
        only a causal mask).
      relative_bias: Relative bias of shape [1/B, N, 1, S].

    Returns:
      encoded: JTensor of shape [B, N, H].
      probs: JTensor of shape [B, N, S].
    """

    p = self.hparams
    key = self._shard_blh(self.get_decode_state(key_state_name))
    value = self._shard_blh(self.get_decode_state(value_state_name))
    # query is 3d.
    query = self._shard_bnh(query)

    b, s, h = key.shape
    base_layer.assert_has_shape(value, [b, s, h])
    base_layer.assert_has_shape(query, [b, -1, h])
    base_layer.assert_has_shape(atten_mask, [-1, -1, s])
    assert atten_mask.shape[0] in [1, b]
    query = self._scale_query(query)
    logits = jnp.einsum('BNH,BSH->BNS', query, key)
    if relative_bias is not None:
      base_layer.assert_has_shape(relative_bias, [-1, -1, 1, s])
      assert relative_bias.shape[0] in [1, b]
      relative_bias = jnp.squeeze(relative_bias, axis=2)
      logits += relative_bias
    logits = self._cap_logits(logits)
    # Attention softmax is always carried out in fp32.
    logits = logits.astype(jnp.float32)
    # Apply attention masking
    padded_logits = logits + atten_mask.astype(jnp.float32)
    # Of shape [b, n, s]
    if p.attention_extra_logit is None:
      probs = jax.nn.softmax(padded_logits, axis=-1).astype(key.dtype)
    else:
      probs = jnp.exp(self._log_softmax_with_extra_logit(padded_logits)).astype(
          key.dtype)
    # Compute the attention context.
    encoded = jnp.einsum('BNS,BSH->BNH', probs, value)
    encoded = self._shard_bnh(encoded)
    return encoded, probs

  def __call__(
      self,
      query_vec: JTensor,
      key_vec: JTensor,
      value_vec: JTensor,
      atten_mask: JTensor,
      query_segment_pos: Optional[JTensor] = None,
      key_segment_pos: Optional[JTensor] = None) -> Tuple[JTensor, JTensor]:
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
    p = self.hparams
    # Project inputs to key, value and query, respectively has shape
    # [B, S, N, H], [B, S, H], and [B, T, H].
    query_proj = self.query(query_vec)
    key_proj = self.key(key_vec)
    value_proj = self.value(value_vec)

    self._fprop_update_decode_state('key_state', key_proj)
    self._fprop_update_decode_state('value_state', value_proj)

    # Apply rotary position embeddings.
    # Paper: https://arxiv.org/abs/2104.09864.
    if p.use_rotary_position_emb:
      query_proj = self.rotary_position_emb(query_proj, query_segment_pos)
      key_shape = key_proj.shape
      # [B, S, H] -> [B, S, N(1), H]
      key_proj = jnp.expand_dims(key_proj, axis=-2)
      key_proj = self.rotary_position_emb(key_proj, key_segment_pos)
      key_proj = jnp.reshape(key_proj, key_shape)
      self._fprop_update_decode_state('key_post_rotary_pos_emb', key_proj)

    # Apply relative bias.
    # Paper: https://aclanthology.org/N18-2074.pdf.
    if p.relative_bias_tpl:
      relative_bias = self.relative_bias(query_segment_pos, key_segment_pos)
    else:
      relative_bias = None

    encoded, atten_probs = self._dot_atten(query_proj, key_proj, value_proj,
                                           atten_mask, relative_bias)

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
    if not self.is_mutable_collection(base_layer.DECODE_CACHE):
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
    extend_value = jnp.expand_dims(value, axis=time_dim)
    indices = [0] * extend_value.ndim
    indices[time_dim] = time_step.astype(jnp.int32)
    state = self.get_decode_state(name)
    assert state is not None
    new_state = jax.lax.dynamic_update_slice(state,
                                             extend_value.astype(state.dtype),
                                             indices)
    self.update_decode_state(name, new_state)
    return new_state

  def extend_step(self, query_vec: JTensor, *, atten_mask: JTensor,
                  time_step: JTensor,
                  segment_pos: Optional[JTensor]) -> JTensor:
    """Computes the value vector given the query of the current step.

    This function is used by autoregressive decoding.

    Args:
      query_vec: JTensor of shape [B, D] corresponding to query vector at index
        time_step.
      atten_mask: JTensor of shape [B/1, 1, S]. atten_mask should have already
        taken care of causal masking for decoding, plus other maskings
        necessary.
      time_step: A scalar or JTensor. Current time-step, 0-based.
      segment_pos: An optional JTensor of shape [B]. Current position in the
        same segment. If unspecified, time_step will be used.

    Returns:
      encoded: JTensor of shape [B, D] which returns the attention output at
        `time_step`.
    """
    p = self.hparams
    time_step = jnp.array(time_step)
    # Batch major.
    time_dim = 1
    assert time_step.ndim == 0
    # Project inputs to key, value and query. Query has shape [B, N, H],
    # key/value shapes [B, H]
    new_key_proj = self.key(query_vec)
    new_value_proj = self.value(query_vec)
    new_query_proj = self.query(query_vec)

    def _extend_decode_state_and_shard_blh(name: str,
                                           extend_value: JTensor) -> JTensor:
      extended_state = self.extend_decode_state(
          name, extend_value, time_step, time_dim=time_dim)
      return self._shard_blh(extended_state)

    # Update value state.
    value_state_name = 'value_state'
    _extend_decode_state_and_shard_blh(value_state_name, new_value_proj)
    # Update key state.
    key_state_name = 'key_state'
    _extend_decode_state_and_shard_blh(key_state_name, new_key_proj)

    if p.use_rotary_position_emb:
      if segment_pos is None:
        position = jnp.broadcast_to(time_step, [query_vec.shape[0]])
      else:
        position = segment_pos
      new_query_proj = self.rotary_position_emb.extend_step(
          new_query_proj, position)
      key_shape = new_key_proj.shape
      new_key_proj = jnp.expand_dims(new_key_proj, axis=-2)
      new_key_proj = self.rotary_position_emb.extend_step(
          new_key_proj, position)
      new_key_proj = jnp.reshape(new_key_proj, key_shape)
      key_state_name = 'key_post_rotary_pos_emb'
      _extend_decode_state_and_shard_blh(key_state_name, new_key_proj)

    if p.relative_bias_tpl:
      # Relative bias uses time_step instead of segment_pos.
      relative_bias = self.relative_bias.extend_step(
          seq_length=self.decoding_state_sequence_length(), time_step=time_step)
    else:
      relative_bias = None

    encoded, atten_prob = self._dot_atten_one_step(new_query_proj,
                                                   key_state_name,
                                                   value_state_name, atten_mask,
                                                   relative_bias)
    # TODO(yonghui): return atten_probs back to the caller.
    del atten_prob
    # Post projection.
    encoded = self.post(encoded)
    encoded = self._shard_bd(encoded)
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
      self.update_decode_state(name, new_state)

  def lazy_broadcast_prefix(self, num_suffix_samples: int,
                            suffix_length: int) -> None:
    """Performs lazy prefix broadcast on the decoding states."""
    raise NotImplementedError(
        'lazy_broadcast_prefix not implemented, use DotProductAttentionWithLPB '
        'instead.')
