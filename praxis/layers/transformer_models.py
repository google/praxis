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

"""Transformer-related layers."""

from __future__ import annotations

import enum
from typing import Any, Optional, Sequence

import jax
from jax import numpy as jnp
from praxis import asserts
from praxis import base_layer
from praxis import py_utils
from praxis import pytypes
from praxis.layers import attentions
from praxis.layers import embedding_softmax
from praxis.layers import multi_query_attention
from praxis.layers import normalizations
from praxis.layers import transformers

NestedMap = py_utils.NestedMap
JTensor = pytypes.JTensor

BaseHParams = base_layer.BaseLayer.HParams
AuxLossStruct = base_layer.AuxLossStruct

AUX_LOSS = base_layer.AUX_LOSS
sub_config_field = base_layer.sub_config_field


def _set_embedding_softmax_sharding_params_for_transformers(
    embedding_softmax_p, *, ici_mesh_shape, dcn_mesh_shape, mesh_axis_names,
    w_vd, a_blv, a_bld):
  """Sets sharding params for embedding_softmax modules in transformers.

  Args:
    embedding_softmax_p: A params of a embedding_softmax class. Currently only
      embedding_softmax.GShardSharedEmbeddingSoftmax,
      embedding_softmax.SharedEmbeddingSoftmax,
      embedding_softmax.FullSoftmax and
      embedding_softmax.Embedding are supported.
    ici_mesh_shape: Shape of logical mesh for a slice.
    dcn_mesh_shape: Shape of logical mesh between slices.
    mesh_axis_names: A list of length len(shape). Each element of the list is
      the name of the corresponding device axis.
    w_vd: sharding of the embedding weight of (vocab_size, d).
    a_blv: sharding of the logits of shape (b, l, vocab_size).
    a_bld: sharding of embedding output activation, shape (b, l, d).

  Returns:
    Params with sharding annotations added.
  """
  embedding_softmax_p.ici_mesh_shape = ici_mesh_shape
  embedding_softmax_p.dcn_mesh_shape = dcn_mesh_shape
  embedding_softmax_p.mesh_axis_names = mesh_axis_names

  if embedding_softmax_p.cls == embedding_softmax.GShardSharedEmbeddingSoftmax:
    # Softmax weight is of shape [vocab_size, input_dim].
    embedding_softmax_p.weight_split_dims_mapping.wt = w_vd
  elif (embedding_softmax_p.cls == embedding_softmax.SharedEmbeddingSoftmax or
        embedding_softmax_p.cls == embedding_softmax.FullSoftmax or
        embedding_softmax_p.cls == embedding_softmax.Embedding):
    # Softmax weight is of shape [input_dim, vocab_size].
    embedding_softmax_p.weight_split_dims_mapping.wt = [w_vd[1], w_vd[0]]
    if embedding_softmax_p.cls != embedding_softmax.FullSoftmax:
      embedding_softmax_p.lookup_style = 'matmul'
  else:
    raise NotImplementedError(
        f'embedding_softmax class {embedding_softmax_p.cls} not supported')

  if (embedding_softmax_p.cls == embedding_softmax.GShardSharedEmbeddingSoftmax
      or embedding_softmax_p.cls == embedding_softmax.SharedEmbeddingSoftmax
      or embedding_softmax_p.cls == embedding_softmax.Embedding):
    embedding_softmax_p.activation_split_dims_mapping.out = a_blv
    (embedding_softmax_p.activation_split_dims_mapping
     .emb_out_split_dims_mapping) = a_bld
  return embedding_softmax_p


def _set_stacked_transformer_sharding(stacked_transformer_p, *, w_df, w_dnh,
                                      w_emh, a_bld, a_blf, a_blnh, a_egch,
                                      a_egcm):
  """Set sharding params for the stacked transformer layer."""
  stacked_p = stacked_transformer_p
  if stacked_p.cls == transformers.PipelinedTransformer:
    stacked_p = stacked_p.pipeline_stage
  if issubclass(stacked_p.cls, transformers.StackedTransformerRepeated):
    stacked_p = stacked_p.block
  transformer_p = stacked_p.transformer_layer_params_tpl
  if isinstance(transformer_p, (list, tuple)):
    transformer_p_lst = transformer_p
  else:
    transformer_p_lst = [transformer_p]
  for t_p in transformer_p_lst:
    for atten_p in (t_p.tr_atten_tpl, t_p.cross_atten_tpl):
      if atten_p is None:
        continue
      atten_wp = atten_p.weight_split_dims_mapping
      atten_wp.proj = w_dnh
      w_n_sharding = None if w_dnh is None else w_dnh[1]
      atten_wp.dconv = [w_n_sharding, None]
      atten_ap = atten_p.activation_split_dims_mapping
      atten_ap.blnh = a_blnh
      atten_ap.bld = a_bld
      if atten_p.cls == multi_query_attention.MultiQueryDotProductAttention:
        atten_wp.proj_headless = [w_dnh[0], w_dnh[2]]
        atten_ap.blh = [a_blnh[0], a_blnh[1], a_blnh[3]]

    ff_p = t_p.tr_fflayer_tpl
    ff_wp = ff_p.weight_split_dims_mapping
    ff_wp.ffn0 = w_df
    if w_df is None:
      ff_wp.ffn1 = None
    else:
      ff_wp.ffn1 = [w_df[1], w_df[0]]
    ff_ap = ff_p.activation_split_dims_mapping
    ff_ap.ffn0 = a_blf
    ff_ap.ffn1 = a_bld

  if stacked_p.moe_layer_tpl is not None:
    # Set Moe layer sharding hparams.
    moe_p = stacked_p.moe_layer_tpl
    moe_wp = moe_p.weight_split_dims_mapping
    moe_wp.me = [None, None]  # Replicated.
    moe_wp.emh = w_emh
    w_e_sharding = None if w_emh is None else w_emh[0]
    w_m_sharding = None if w_emh is None else w_emh[1]
    w_h_sharding = None if w_emh is None else w_emh[2]
    moe_wp.ehm = [w_e_sharding, w_h_sharding, w_m_sharding]
    # Activations
    a_e_sharding = None if a_egch is None else a_egch[0]
    moe_ap = moe_p.activation_split_dims_mapping
    moe_ap.gs = [a_e_sharding, None]
    # dispatch and combine tensors
    moe_ap.gsec = [a_e_sharding, None, None, None]
    moe_ap.gecs = [a_e_sharding, None, None, None]
    moe_ap.gec = [a_e_sharding, None, None]
    moe_ap.egch = a_egch
    a_e_sharding = None if a_egcm is None else a_egcm[0]
    a_m_sharding = None if a_egcm is None else a_egcm[3]
    moe_ap.gsm = [a_e_sharding, None, a_m_sharding]
    moe_ap.egcm = a_egcm
    moe_ap.gecm = a_egcm
  return stacked_transformer_p


@enum.unique
class LanguageModelType(str, enum.Enum):
  """The different language model types based on the tokens visibility."""
  CAUSAL = 'causal'
  PREFIX = 'prefix'
  BIDIRECTIONAL = 'bidirectional'


class TransformerLm(base_layer.BaseLayer):
  """Packed Transformer LM with position embedding and shared softmax layer.

  This folds the padding with the segment mask when the inputs are not packed.
  """

  class HParams(BaseHParams):
    """Associated hyper-params for this layer class.

    Attributes:
      position_emb_tpl: The Positional Embedding layer params.
      model_dims: Model dimension in Transformer layers.
      stacked_transformer_tpl: StackedTransformer params tpl for the
        TransformerLm.
      softmax_tpl: The softmax layer params. By default the softmax layer is of
        type SharedEmbeddingSoftmax so the softmax and embedding lookup share
        parameters in this case.
      vocab_size: Size of the vocabulary for LM.
      packed_input: Whether the inputs are packed.
      model_type: The type of language model based on the tokens visibility.
      ngrammer_tpl: Params or list of params for the Ngrammer layer applied to
        the input sequence (at the beginning of the network). This param may be
        of type Ngrammer as well as VQNgrammer layer. If this is None then the
        Ngrammer layer is not used.
      post_attention_ngrammer_tpls: Sequence of params for the Ngrammer layer
        applied after every attention layer. This param must be of the form
        VQNgrammer layer, since we do not have any input ids for intermediate
        layers. The length of this sequence must match the number of layers of
        the model. To disable the application at a particular layer, set its
        value to None. To completely disable this layer set it to either None or
        a sequence of all Nones.
      separate_embedding_tpl: Optional separate embedding lookup layer params.
        By default this is None since the softmax and embedding lookup share
        parameters, however if we wish to separate the parameters of embedding
        lookup and softmax then we can set this param.
      final_ln_tpl: Parameterization of the layer normalization layer.
      skip_compute_loss: Set to skip compute_loss and output activations.
    """
    position_emb_tpl: BaseHParams = sub_config_field(
        embedding_softmax.PositionalEmbedding.HParams)
    model_dims: int = 0
    stacked_transformer_tpl: BaseHParams = sub_config_field(
        transformers.StackedTransformer.HParams)
    softmax_tpl: BaseHParams = sub_config_field(
        embedding_softmax.SharedEmbeddingSoftmax.HParams)
    vocab_size: int = 0
    packed_input: bool = False
    model_type: LanguageModelType = LanguageModelType.CAUSAL
    ngrammer_tpl: Optional[BaseHParams] = base_layer.sub_config_field(None)
    post_attention_ngrammer_tpls: Optional[Sequence[BaseHParams]] = (
        base_layer.sub_config_field(None))
    separate_embedding_tpl: Optional[BaseHParams] = base_layer.sub_config_field(
        None)
    final_ln_tpl: BaseHParams = sub_config_field(
        normalizations.LayerNorm.HParams)
    skip_compute_loss: bool = False

  @classmethod
  def set_sharding_params_v1(cls,
                             lm_p,
                             *,
                             replica_axis,
                             data_axis,
                             mdl_axis,
                             ici_mesh_shape,
                             dcn_mesh_shape=None,
                             batch_axes=None,
                             mesh_axis_names,
                             training_optimized):
    """Set Canonical sharding params.

    Args:
      lm_p: A params of this class.
      replica_axis: A string or int of the model replica axis name.
      data_axis: A string or int of the data axis name.
      mdl_axis: A string or int of the mdl axis name.
      ici_mesh_shape: Shape of logical mesh for a slice.
      dcn_mesh_shape: Shape of logical mesh between slices.
      batch_axes: A tuple with all axes used for splitting activations along
        batch dimension. Defaults to (replica_axis, data_axis).
      mesh_axis_names: A list of length len(shape). Each element of the list is
        the name of the corresponding device axis.
      training_optimized: A bool indicating whether sharding is optimized for
        training by saving activation memory between forward and backward
        passes.

    Returns:
      Params with sharding annotations added.
    """
    # In the following, model weights are layed out on the [data_axis, mdl_axis]
    # 2d mesh. Model weights are always replicated over the replica_axis mesh
    # axis.
    #
    # The batch axis of the activations are always sharded over the combination
    # of (replica_axis, data_axis).

    if batch_axes is None:
      batch_axes = (replica_axis, data_axis)
    bld = [batch_axes, None, mdl_axis
          ] if training_optimized else [batch_axes, None, None]
    egcm = [data_axis, None, None, mdl_axis
           ] if training_optimized else [batch_axes, None, None, None]

    # w_df: sharding for weight of ffn0, shape (d, f). ff1 weights will be
    # inferred from it.
    w_df = [data_axis, mdl_axis]
    # w_dnh: Sharding of qkv projection weights, shape (d, num_heads,
    # per_head_size)
    w_dnh = [data_axis, mdl_axis, None]
    # w_emh: sharding for first MoE FFN weight, shape (e, m, h). The second MoE
    # ffn weight will be inferred from it.
    w_emh = [data_axis, None, mdl_axis]
    # w_vd: sharding of the embedding weight of (vocab_size, d).
    w_vd = [mdl_axis, data_axis]
    # a_bld: sharding of output of ffn/attention, shape (b, l, d).
    a_bld = bld
    # a_blf: sharding of output of ffn0, shape (b, l, f).
    a_blf = [batch_axes, None, mdl_axis]
    # a_blnh: sharding of the attention activation of shape (b, l, num_heads,
    # per_head_size).
    a_blnh = [batch_axes, None, mdl_axis, None]
    # a_blv: sharding of the logits activation of shape (b, l, vocab_size).
    a_blv = [batch_axes, None, mdl_axis]
    # a_egch: sharding of the output of first MoE FFN, shape (e, g, c, h).
    a_egch = [data_axis, None, None, mdl_axis]
    # a_egcm: sharding of the output of second MoE FFN, shape (e, g, c, m).
    a_egcm = egcm
    return cls.set_custom_sharding_params(
        lm_p,
        ici_mesh_shape=ici_mesh_shape,
        dcn_mesh_shape=dcn_mesh_shape,
        mesh_axis_names=mesh_axis_names,
        w_df=w_df,
        w_dnh=w_dnh,
        w_emh=w_emh,
        w_vd=w_vd,
        a_bld=a_bld,
        a_blf=a_blf,
        a_blnh=a_blnh,
        a_blv=a_blv,
        a_egch=a_egch,
        a_egcm=a_egcm)

  @classmethod
  def set_custom_sharding_params(
      cls,
      lm_p,
      *,
      ici_mesh_shape,
      dcn_mesh_shape=None,
      mesh_axis_names,
      w_df=None,
      w_dnh=None,
      w_emh=None,
      w_vd=None,
      a_bld=None,
      a_blf=None,
      a_blnh=None,
      a_blv=None,
      a_egch=None,
      a_egcm=None,
  ):
    """Configure the shardings on each tensor.

    Args:
      lm_p: A params of this class.
      ici_mesh_shape: Shape of logical mesh for a slice.
      dcn_mesh_shape: Shape of logical mesh between slices.
      mesh_axis_names: A list of length len(mesh_shape). Each element of the list
        is the name of the corresponding device axis.
      w_df: sharding for weight of ffn0, shape (d, f). ff1 weights will be
        inferred from it.
      w_dnh: Sharding of qkv projection weights, shape (d, num_heads,
        per_head_size)
      w_emh: sharding for first MoE FFN weight, shape (e, m, h). The second MoE
        ffn weight will be inferred from it.
      w_vd: sharding of the embedding weight of (vocab_size, d).
      a_bld: sharding of output of ffn/attention, shape (b, l, d).
      a_blf: sharding of output of ffn0, shape (b, l, f).
      a_blnh: sharding of the attention activation of shape (b, l, num_heads,
        per_head_size).
      a_blv: sharding of the logits activation of shape (b, l, vocab_size).
      a_egch: sharding of the output of first MoE FFN, shape (e, g, c, h).
      a_egcm: sharding of the output of second MoE FFN, shape (e, g, c, m).

    Returns:
      Params with sharding annotations added.
    """
    lm_p.ici_mesh_shape = ici_mesh_shape
    lm_p.dcn_mesh_shape = dcn_mesh_shape
    lm_p.mesh_axis_names = mesh_axis_names
    pos_emb_w_ld = w_df
    if (lm_p.position_emb_tpl is not None and lm_p.position_emb_tpl.cls
        == embedding_softmax.TrainablePositionalEmbedding):
      lm_p.position_emb_tpl.weight_split_dims_mapping.wt = pos_emb_w_ld
      lm_p.position_emb_tpl.activation_split_dims_mapping.out = pos_emb_w_ld

    if lm_p.ngrammer_tpl is not None:
      lm_p.ngrammer_tpl.weight_split_dims_mapping.wt = w_vd

    mesh_kwargs = {
        'ici_mesh_shape': lm_p.ici_mesh_shape,
        'dcn_mesh_shape': lm_p.dcn_mesh_shape,
        'mesh_axis_names': mesh_axis_names,
    }
    lm_p.softmax_tpl = _set_embedding_softmax_sharding_params_for_transformers(
        lm_p.softmax_tpl, w_vd=w_vd, a_blv=a_blv, a_bld=a_bld, **mesh_kwargs)

    def _set_transformer_sharding(transformer_tpl):
      return _set_stacked_transformer_sharding(
          transformer_tpl,
          w_df=w_df,
          w_dnh=w_dnh,
          w_emh=w_emh,
          a_bld=a_bld,
          a_blf=a_blf,
          a_blnh=a_blnh,
          a_egch=a_egch,
          a_egcm=a_egcm)

    if lm_p.stacked_transformer_tpl.cls == transformers.PipelinedTransformer:
      lm_p.stacked_transformer_tpl.pipeline_stage = _set_transformer_sharding(
          lm_p.stacked_transformer_tpl.pipeline_stage)
    else:
      lm_p.stacked_transformer_tpl = _set_transformer_sharding(
          lm_p.stacked_transformer_tpl)

    if lm_p.separate_embedding_tpl is not None:
      lm_p.separate_embedding_tpl = (
          _set_embedding_softmax_sharding_params_for_transformers(
              lm_p.separate_embedding_tpl,
              w_vd=w_vd,
              a_blv=a_blv,
              a_bld=a_bld,
              **mesh_kwargs))
    return lm_p

  def setup(self) -> None:
    """Constructor."""
    p = self.hparams

    # Optional positional embedding layer.
    if p.position_emb_tpl is not None:
      pos_params = p.position_emb_tpl.clone()
      pos_params.embedding_dims = p.model_dims
      self.create_child('position_emb', pos_params)

    # Optional separate embedding layer.
    if p.separate_embedding_tpl is not None:
      emb_params = p.separate_embedding_tpl.clone()
      emb_params.input_dims = p.model_dims
      emb_params.num_classes = p.vocab_size
      self.create_child('embedding_lookup', emb_params)

    # Ngrammer layer.
    if p.ngrammer_tpl is not None:
      self.create_child('ngrammer', p.ngrammer_tpl)

    # Transformer layers.
    stacked_xformer_params = p.stacked_transformer_tpl.clone()
    xformer_params = stacked_xformer_params
    if xformer_params.cls == transformers.PipelinedTransformer:
      xformer_params = xformer_params.pipeline_stage
    if issubclass(xformer_params.cls, transformers.StackedTransformerRepeated):
      xformer_params = xformer_params.block
    if not issubclass(xformer_params.cls, transformers.StackedTransformer):
      assert False, f'{xformer_params.cls} not supported.'
    assert (xformer_params.model_dims == 0 or
            xformer_params.model_dims == p.model_dims)
    xformer_params.model_dims = p.model_dims
    # TODO(pax): we shouldn't override mask_self_attention here.
    if p.model_type == LanguageModelType.CAUSAL:
      xformer_params.mask_self_attention = True
    else:
      xformer_params.mask_self_attention = False
    xformer_params.packed_input = p.packed_input
    xformer_params.fold_padding_with_segment_mask = True
    if p.post_attention_ngrammer_tpls is not None:
      if len(p.post_attention_ngrammer_tpls) != xformer_params.num_layers:
        raise ValueError('The length of post_attention_ngrammer_tpls must match'
                         'the number of attention layers.')
      xformer_params.ngrammer_tpls = p.post_attention_ngrammer_tpls
    self.create_child('transformer', stacked_xformer_params)

    # Final layer norm.
    if p.final_ln_tpl is not None:
      ln_params = p.final_ln_tpl.clone().set(dim=p.model_dims)
      self.create_child('final_ln', ln_params)

    # Final softmax
    softmax_params = p.softmax_tpl.clone()
    softmax_params.input_dims = p.model_dims
    softmax_params.num_classes = p.vocab_size
    self.create_child('softmax', softmax_params)

  def init_states(self, *args: Any, **kwargs: Any) -> None:
    """Initialize the cache for the autoregressive decoding.

    Args:
      *args: Other arguments.
      **kwargs: Other keyword arguments.
    """
    raise NotImplementedError(type(self))

  def compute_loss(self,
                   activations: JTensor,
                   labels: Optional[NestedMap] = None) -> NestedMap:
    """Computes cross entropy loss.

    Args:
      activations: Output of last layer of shape [B, T, D].
      labels: A `.NestedMap` containing the following fields: class_weights, a
        JTensor with shape [B, T] containing weights for each target word.
        class_ids, a JTensor with shape [B, T] of int32 dtype containing the
        target class labels. class_probabilities, a JTensor with shape [B, T, V]
        of float values indicating class-membership probabilities.

    Returns:
      Returns xent_output, where `xent_output` is a `.NestedMap` as defined by
      `SoftmaxLayer`'s return. In addition, per_sequence_xent is added which
      equal to the sum of xent loss for tokens in a sequence.
    """
    if labels is None:
      logits = self.softmax.get_logits(inputs=activations)
      xent_output = NestedMap(logits=logits)
      xent_output.log_probs = jax.nn.log_softmax(logits)
      xent_output.probs = jax.nn.softmax(xent_output.logits)
    else:
      class_ids = None
      class_probabilities = None
      if 'class_ids' in labels:
        class_ids = labels.class_ids[:, :, jnp.newaxis]
      if 'class_probabilities' in labels:
        class_probabilities = labels.class_probabilities
      class_weights = labels.class_weights[:, :, jnp.newaxis]
      xent_output = self.softmax(
          activations,
          class_weights,
          class_ids=class_ids,
          class_probabilities=class_probabilities)
      per_token_xent = xent_output.per_example_xent * labels.class_weights
      xent_output.per_token_xent = per_token_xent
      xent_output.per_sequence_xent = jnp.sum(per_token_xent, -1)

      # Sum aux_loss and add to avg_xent.
      aux_loss = 0.0
      aux_loss_weight = 0.0
      if AUX_LOSS in self.variables:
        aux_loss_values = jax.tree_util.tree_leaves(
            self.variables[AUX_LOSS],
            is_leaf=lambda x: isinstance(x, AuxLossStruct))
        for v in aux_loss_values:
          assert isinstance(v, AuxLossStruct)
          aux_loss += jnp.sum(v.value)
          aux_loss_weight += jnp.sum(v.weight)
      if not isinstance(aux_loss, jnp.ndarray):
        aux_loss = jnp.array(aux_loss, dtype=self.fprop_dtype)
        aux_loss_weight = jnp.array(aux_loss_weight, dtype=self.fprop_dtype)
      self.add_summary('total_aux_loss', aux_loss)
      self.add_summary('total_aux_loss_weight', aux_loss_weight)
      xent_output.aux_loss = aux_loss
      xent_output.aux_loss_weight = aux_loss_weight
      # This is the loss to minimize.
      xent_output.total_loss = xent_output.avg_xent + xent_output.aux_loss
    return xent_output

  def _prepare_input(self,
                     inputs: JTensor,
                     paddings: JTensor,
                     segment_pos: Optional[JTensor] = None,
                     **input_kwargs) -> JTensor:
    del input_kwargs
    p = self.hparams
    _, seq_length = inputs.shape

    # Get the input embeddings.
    if self.hparams.separate_embedding_tpl is not None:
      input_emb = self.embedding_lookup.emb_lookup(inputs)
    else:
      input_emb = self.softmax.emb_lookup(inputs)

    # Add NGrammer to the source embeddings.
    if p.ngrammer_tpl is not None:
      if self.hparams.separate_embedding_tpl is not None:
        emb_var = self.embedding_lookup.theta.emb_var
      else:
        emb_var = jnp.transpose(self.softmax.logits_ffn.linear.theta.w)
      input_emb = self.ngrammer(
          input_ids=inputs,
          input_embs=input_emb,
          paddings=paddings,
          segment_pos=segment_pos,
          emb_var=emb_var)

    if p.position_emb_tpl is not None:
      position_emb = self.position_emb(
          seq_length=seq_length, position=segment_pos)
      inputs = input_emb + position_emb
    else:
      inputs = input_emb
    return inputs

  def __call__(self,
               inputs: JTensor,
               paddings: JTensor,
               labels: Optional[NestedMap] = None,
               segment_ids: Optional[JTensor] = None,
               segment_pos: Optional[JTensor] = None,
               causal_attention_mask: Optional[JTensor] = None,
               segment_mask: Optional[JTensor] = None,
               start_time_step: int = 0,
               **input_kwargs) -> NestedMap:
    """Computes xent loss given the language model inputs.

    Args:
      inputs: Input ids. An int32 JTensor of shape [B, T].
      paddings: A 0/1 JTensor of shape [B, T] with 1 denoting padding.
      labels: A `.NestedMap` containing the following fields: class_weights, a
        JTensor with shape [batch, seqlen] containing weights for each target
        word. class_ids, a JTensor with shape [B, T] of int32 dtype containing
        the target class labels. class_probabilities, a JTensor with shape [B,
        T, V] of float values indicating class-membership probabilities.
      segment_ids: A JTensor of shape [B, T]. The segment that each token
        belongs to.
      segment_pos: A JTensor of shape [B, T]. The position of each token in a
        segment.
      causal_attention_mask: A JTensor of shape [B, T] where 1 indicates a token
        position with causal attention and 0 indicates bidirectional attention.
        This overrides part of the causal mask.
      segment_mask: Optional pre-defined segment_mask passed to the transformer.
        A JTensor of shape [B, 1, T, T]. If it is None, the segment_mask will be
        inferred from the LanguageModelType `model_type` hparam.
      start_time_step: Decode extend_step start time step. When decoding after
        prefix, start_time_step will be prefix_len.
      **input_kwargs: additional input kwargs to be sent to the transformer.

    Returns:
      Returns xent_output, where
      `xent_output` is a `.NestedMap` as defined by `SoftmaxLayer`'s return. In
      addition, per_sequence_xent is added which equal to the sum of xent loss
      for tokens in a sequence.
    """
    p = self.hparams
    batch, seq_length = inputs.shape

    paddings_float32 = paddings.astype(jnp.float32)
    num_unpadded_tokens = jnp.sum(1.0 - paddings_float32)
    self.add_summary('num_unpadded_tokens', num_unpadded_tokens)
    if inputs.size != 0:
      num_tokens = jnp.array(inputs.size, jnp.float32)
      ratio_unpadded_tokens = num_unpadded_tokens / num_tokens
      self.add_summary('ratio_unpadded_tokens', ratio_unpadded_tokens)

    if segment_ids is None:
      assert segment_pos is None
      # Fold the paddings with the segment mask
      segment_ids = jnp.asarray(1 - paddings, jnp.int32)
      segment_pos = jnp.tile(
          jnp.arange(seq_length, dtype=jnp.int32)[None, :], [batch, 1])

    inputs = self._prepare_input(inputs, paddings, segment_pos=segment_pos,
                                 **input_kwargs)

    if segment_mask is None:
      if p.model_type == LanguageModelType.BIDIRECTIONAL:
        segment_mask = attentions.segment_mask(segment_ids, segment_ids,
                                               inputs.dtype)
      else:
        segment_mask = attentions.causal_segment_mask(segment_ids, inputs.dtype,
                                                      causal_attention_mask)

    self.update_decode_state('time_step', start_time_step)
    output = self.transformer(
        inputs, paddings, segment_mask=segment_mask, segment_pos=segment_pos)

    # Final layer norm
    if p.final_ln_tpl is not None:
      output = self.final_ln(output)

    if p.skip_compute_loss:
      return output
    else:
      return self.compute_loss(output, labels)

  def _emb_lookup(self, input_ids):
    """Token emb lookup.

    Args:
      input_ids: [B, T].

    Returns:
      input_emb: [B, T, D]
    """
    assert input_ids.ndim == 2, input_ids.shape

    p = self.hparams
    if p.separate_embedding_tpl is not None:
      # [B, ?, D]
      input_emb = self.embedding_lookup.emb_lookup(input_ids)
    else:
      # [B, ?, D]
      input_emb = self.softmax.emb_lookup(input_ids)
    return input_emb

  def _add_pos_emb(self, input_emb, segment_pos):
    """Adds positional emb to input_emb.

    Args:
      input_emb:   [B, T, D].
      segment_pos:  None or [B, T].

    Returns:
      [B, T, D]
    """
    assert input_emb.ndim == 3, input_emb.shape

    p = self.hparams
    if not p.position_emb_tpl:
      return input_emb

    b, t = input_emb.shape[:2]
    time_step = self.get_decode_state('time_step')

    position = segment_pos
    if segment_pos is None:
      # [1, T]
      position = (jnp.arange(t) + time_step)[jnp.newaxis, :]
      # [B, T]
      position = jnp.tile(position, (b, 1))

    # [B, T, D]
    pos_emb = self.position_emb(position=position)
    return input_emb + pos_emb

  def _emb_ngrammer(self, input_ids, input_emb, segment_pos):
    """Applies Ngrammer embedding.

    Args:
      input_ids:    [B, T].
      input_emb:    [B, T, D].
      segment_pos:  None or [B, T].

    Returns:
      if has ngrammer: ([B, 1], [B, 1, D], [B, 1])
      else, the same as the inputs.
    """
    assert input_ids.ndim == 2, input_ids.shape
    assert input_emb.ndim == 3, input_emb.shape
    if segment_pos is not None:
      assert segment_pos.ndim == 2, segment_pos.shape

    p = self.hparams
    if p.ngrammer_tpl is None:
      return input_ids, input_emb, segment_pos

    # [B, T, D]
    input_emb = self.ngrammer(input_ids, input_emb, segment_pos=segment_pos)
    # [B, 1, D]
    input_emb = input_emb[:, -1:, :]
    # [B, 1]
    input_ids = input_ids[:, -1:]
    if segment_pos is not None:
      segment_pos = segment_pos[:, -1:]

    return input_ids, input_emb, segment_pos

  def extend_step(
      self,
      inputs: JTensor,
      segment_pos: Optional[JTensor] = None,
      atten_mask: Optional[JTensor] = None,
  ) -> NestedMap:
    """Autoregressive cached decoding of Transformer LM.

    When `inputs`'s shape is [B], it does single-token extend_step per batch,
    as in regular autoregressive decoding.

    When `inputs`'s shape is [B, L], it does time-batched extend_step on L
    tokens per batch. This is for suffix scoring after autoregressive decoding.

    Args:
      inputs:       [B] or [B, L], target sequence at time_step.
        The latter is useful for Primer, Ngrammer, and suffix-scoring post
        autoregressive decoding.
      segment_pos:  [B] or [B, L], optional segment pos of each input token.
      atten_mask:   [B, 1, L, S], optional attention mask.

    Returns:
      xent_output: a `.NestedMap` object containing the log probabilities and
        probabilities.
    """
    p = self.hparams
    b = inputs.shape[0]
    # Extend step should only be called with causal or prefix LM.
    assert p.model_type != LanguageModelType.BIDIRECTIONAL, p.model_type

    # Input shape sanity checks.
    assert inputs.ndim in (1, 2), inputs.ndim
    if segment_pos is not None:
      assert inputs.shape == segment_pos.shape, (inputs.shape,
                                                 segment_pos.shape)
    if atten_mask is not None:
      prefix_len = inputs.shape[1] if inputs.ndim == 2 else 1
      assert atten_mask.shape[:3] == (b, 1, prefix_len), atten_mask.shape

    is_single_token = inputs.ndim == 1

    # Makes ids rank=2 for uniformity.
    # [B, T]
    input_ids = inputs[:, jnp.newaxis] if is_single_token else inputs

    # Makes segment_pos rank=2 for uniformity.
    if segment_pos is not None:
      # [B, T]
      segment_pos = (
          segment_pos[:, jnp.newaxis] if is_single_token else segment_pos)

    # Get the input embeddings.
    # [B, T, D]
    input_emb = self._emb_lookup(input_ids)

    # Add Ngrammer layer if applicable.
    # [B, ?], [B, ?, D], [B, ?]
    input_ids, input_emb, segment_pos = self._emb_ngrammer(
        input_ids, input_emb, segment_pos)

    # [B, ?, D]
    transformer_inputs = self._add_pos_emb(input_emb, segment_pos)

    if is_single_token or p.ngrammer_tpl is not None:
      # Ngrammer always collapses output.
      # [B, D]
      transformer_inputs = jnp.squeeze(transformer_inputs, 1)
      # [B]
      if segment_pos is not None:
        segment_pos = jnp.squeeze(segment_pos, 1)

    time_step = self.get_decode_state('time_step')
    outputs = self.transformer.extend_step(
        transformer_inputs,
        time_step=time_step,
        segment_pos=segment_pos,
        atten_mask=atten_mask)

    self.update_decode_state('time_step', time_step + 1)
    if p.final_ln_tpl is not None:
      outputs = self.final_ln(outputs)
    xent_output = self.compute_loss(outputs)
    return xent_output

  def transform_decode_state(
      self, transform_fn: base_layer.DecodeStateTransformFn) -> None:
    """Transforms all decode state variables based on transform_fn."""
    self.transformer.transform_decode_state(transform_fn)

  def lazy_broadcast_prefix(self, num_suffix_samples: int,
                            suffix_length: int) -> None:
    """Performs lazy prefix broadcast on the decoding states.

    Current decoding states will be moved to PREFIX_DECODE_CACHE. New decoding
    state will be created for the suffixes with multiple samples sharing
    previous prefixes.

    Args:
      num_suffix_samples: Number of samples that will share the same previous
        decoding state.
      suffix_length: The length of the new suffix samples.
    """
    self.transformer.lazy_broadcast_prefix(num_suffix_samples, suffix_length)

  def right_align_decode_state_with_prefix(
      self, max_prefix_size: int,
      right_align_fn: base_layer.DecodeStateTransformFn) -> None:
    """Right aligns decode state with prefix decode states."""
    self.transformer.right_align_decode_state_with_prefix(
        max_prefix_size, right_align_fn)


class TransformerEncoderDecoder(base_layer.BaseLayer):
  """Transformer encoder/decoder class.

  This uses the param `encoder_stacked_transformer_tpl` to set the configuration
  for the encoder stack, and the param `decoder_stacked_transformer_tpl` to set
  the configuration for the decoder stack.
  """

  class HParams(BaseHParams):
    """Associated hyper-params for this layer class.

    Attributes:
      position_emb_tpl: The Positional Embedding layer params for encoder and
        decoder. If this is set then encoder_position_emb_tpl and
        decoder_position_emb_tpl must be set to None.
      encoder_position_emb_tpl: Optional separate position embedding layer for
        the input ids. If this is set then position_emb_tpl must be set to None.
      encoder_stacked_transformer_tpl: StackedTransformer params tpl for the
        encoder. This must be set with a value that is not None at
        initialization time.
      encoder_ngrammer_tpl: Optional params for the Ngrammer layer for the
        encoder. This param is shared between the Ngrammer layer as well as the
        VQNgrammer layer. If this is None then the Ngrammer layer is not used.
      encoder_post_attention_ngrammer_tpls: Sequence of params for the Ngrammer
        layer applied after every attention layer in the encoder. This param
        must be of the form VQNgrammer layer, since we do not have any input ids
        for intermediate layers.
      encoder_embedding_tpl: Optional separate embedding layer for the source
        ids. By default this is set to None, so the inputs and targets share the
        same set of embeddings.
      decoder_position_emb_tpl: Optional separate position embedding layer for
        the target ids. If this is set then position_emb_tpl must be set to
        None.
      decoder_stacked_transformer_tpl: StackedTransformer params tpl for the
        decoder. This must be set with a value that is not None at
        initialization time.
      decoder_ngrammer_tpl: Optional params for the Ngrammer layer for the
        decoder. This param is shared between the Ngrammer layer as well as the
        VQNgrammer layer. If this is None then the Ngrammer layer is not used.
      decoder_post_attention_ngrammer_tpls: Sequence of params for the Ngrammer
        layer applied after every attention layer in the decoder. This param
        must be of the form VQNgrammer layer, since we do not have any input ids
        for intermediate layers.
      decoder_embedding_tpl: Optional separate embedding layer for the target
        ids. By default this is set to None, so the embedding parameters are
        shared with the softmax layer.
      model_dims: Model dimension of the Transformer layers. This must match the
        model dimension of the encoder stack and the decoder stack, as well as
        the embedding and softmax dimensions.
      softmax_tpl: The softmax layer params. By default the softmax layer is of
        type SharedEmbeddingSoftmax so the softmax and embedding lookup share
        parameters in this case.
      packed_input: Whether the inputs are packed.
      encoder_ln_tpl: Parameterization of the encoder layer normalization layer.
      decoder_ln_tpl: Parameterization of the decoder layer normalization layer.
    """
    position_emb_tpl: BaseHParams = sub_config_field(
        embedding_softmax.PositionalEmbedding.HParams)
    encoder_position_emb_tpl: Optional[
        BaseHParams] = base_layer.sub_config_field(None)
    encoder_stacked_transformer_tpl: Optional[
        BaseHParams] = base_layer.sub_config_field(None)
    encoder_ngrammer_tpl: Optional[BaseHParams] = base_layer.sub_config_field(
        None)
    encoder_post_attention_ngrammer_tpls: Optional[Sequence[BaseHParams]] = (
        base_layer.sub_config_field(None))
    encoder_embedding_tpl: Optional[BaseHParams] = base_layer.sub_config_field(
        None)
    decoder_position_emb_tpl: Optional[
        BaseHParams] = base_layer.sub_config_field(None)
    decoder_stacked_transformer_tpl: Optional[
        BaseHParams] = base_layer.sub_config_field(None)
    decoder_ngrammer_tpl: Optional[BaseHParams] = base_layer.sub_config_field(
        None)
    decoder_post_attention_ngrammer_tpls: Optional[Sequence[BaseHParams]] = (
        base_layer.sub_config_field(None))
    decoder_embedding_tpl: Optional[BaseHParams] = base_layer.sub_config_field(
        None)
    model_dims: int = 0
    softmax_tpl: BaseHParams = sub_config_field(
        embedding_softmax.SharedEmbeddingSoftmax.HParams)
    packed_input: bool = False
    encoder_ln_tpl: BaseHParams = sub_config_field(
        normalizations.LayerNorm.HParams)
    decoder_ln_tpl: BaseHParams = sub_config_field(
        normalizations.LayerNorm.HParams)

  @classmethod
  def set_sharding_params_v1(cls,
                             model_p,
                             *,
                             replica_axis,
                             data_axis,
                             mdl_axis,
                             ici_mesh_shape,
                             dcn_mesh_shape=None,
                             mesh_axis_names,
                             training_optimized):
    """Set Canonical sharding params.

    Args:
      model_p: A params of this class.
      replica_axis: A string or int of the model replica axis name.
      data_axis: A string or int of the data axis name.
      mdl_axis: A string or int of the mdl axis name.
      ici_mesh_shape: Shape of logical mesh for a slice.
      dcn_mesh_shape: Shape of logical mesh between slices.
      mesh_axis_names: A list of length len(shape). Each element of the list is
        the name of the corresponding device axis.
      training_optimized: A bool indicating whether sharding is optimized for
        training by saving activation memory between forward and backward
        passes.

    Returns:
      Params with sharding annotations added.
    """

    # In the following, model weights are layed out on the [data_axis, mdl_axis]
    # 2d mesh. Model weights are always replicated over the replica_axis mesh
    # axis.
    #
    # The batch axis of the activations are always sharded over the combination
    # of (replica_axis, data_axis).
    batch_axes = (replica_axis, data_axis)
    bld = [batch_axes, None, mdl_axis
          ] if training_optimized else [batch_axes, None, None]
    egcm = [data_axis, None, None, mdl_axis
           ] if training_optimized else [batch_axes, None, None, None]
    model_p.ici_mesh_shape = ici_mesh_shape
    model_p.dcn_mesh_shape = dcn_mesh_shape
    model_p.mesh_axis_names = mesh_axis_names
    mesh_kwargs = {
        'ici_mesh_shape': ici_mesh_shape,
        'dcn_mesh_shape': dcn_mesh_shape,
        'mesh_axis_names': mesh_axis_names,
    }

    # w_df: sharding for weight of ffn0, shape (d, f). ff1 weights will be
    # inferred from it.
    w_df = [data_axis, mdl_axis]
    # w_dnh: Sharding of qkv projection weights, shape (d, num_heads,
    # per_head_size)
    w_dnh = [data_axis, mdl_axis, None]
    # w_emh: sharding for first MoE FFN weight, shape (e, m, h). The second MoE
    # ffn weight will be inferred from it.
    w_emh = [data_axis, None, mdl_axis]
    # w_vd: sharding of the embedding weight of (vocab_size, d).
    w_vd = [mdl_axis, data_axis]
    # a_bld: sharding of output of ffn/attention, shape (b, l, d).
    a_bld = bld
    # a_blf: sharding of output of ffn0, shape (b, l, f).
    a_blf = [batch_axes, None, mdl_axis]
    # a_blnh: sharding of the attention activation of shape (b, l, num_heads,
    # per_head_size).
    a_blnh = [batch_axes, None, mdl_axis, None]
    # a_blv: sharding of the logits activation of shape (b, l, vocab_size).
    a_blv = [batch_axes, None, mdl_axis]
    # a_egch: sharding of the output of first MoE FFN, shape (e, g, c, h).
    a_egch = [data_axis, None, None, mdl_axis]
    # a_egcm: sharding of the output of second MoE FFN, shape (e, g, c, m).
    a_egcm = egcm

    pos_emb_w_ld = w_df
    if (model_p.position_emb_tpl is not None and model_p.position_emb_tpl.cls
        == embedding_softmax.TrainablePositionalEmbedding):
      model_p.position_emb_tpl.weight_split_dims_mapping.wt = pos_emb_w_ld

    if (model_p.encoder_position_emb_tpl is not None and
        model_p.encoder_position_emb_tpl.cls
        == embedding_softmax.TrainablePositionalEmbedding):
      model_p.encoder_position_emb_tpl.weight_split_dims_mapping.wt = (
          pos_emb_w_ld)

    model_p.softmax_tpl = (
        _set_embedding_softmax_sharding_params_for_transformers(
            model_p.softmax_tpl,
            w_vd=w_vd,
            a_blv=a_blv,
            a_bld=a_bld,
            **mesh_kwargs))

    def _set_transformer_sharding(transforemr_p):
      return _set_stacked_transformer_sharding(
          transforemr_p,
          w_df=w_df,
          w_dnh=w_dnh,
          w_emh=w_emh,
          a_bld=a_bld,
          a_blf=a_blf,
          a_blnh=a_blnh,
          a_egch=a_egch,
          a_egcm=a_egcm)

    model_p.encoder_stacked_transformer_tpl = _set_transformer_sharding(
        model_p.encoder_stacked_transformer_tpl)
    if model_p.encoder_ngrammer_tpl is not None:
      model_p.encoder_ngrammer_tpl.weight_split_dims_mapping.wt = w_vd
    if model_p.encoder_embedding_tpl is not None:
      model_p.encoder_embedding_tpl = (
          _set_embedding_softmax_sharding_params_for_transformers(
              model_p.encoder_embedding_tpl,
              w_vd=w_vd,
              a_blv=a_blv,
              a_bld=a_bld,
              **mesh_kwargs))

    if (model_p.decoder_position_emb_tpl is not None and
        model_p.decoder_position_emb_tpl.cls
        == embedding_softmax.TrainablePositionalEmbedding):
      model_p.decoder_position_emb_tpl.weight_split_dims_mapping.wt = (
          pos_emb_w_ld)

    model_p.decoder_stacked_transformer_tpl = _set_transformer_sharding(
        model_p.decoder_stacked_transformer_tpl)
    if model_p.decoder_ngrammer_tpl is not None:
      model_p.decoder_ngrammer_tpl.weight_split_dims_mapping.wt = w_vd
    if model_p.decoder_embedding_tpl is not None:
      model_p.decoder_embedding_tpl = (
          _set_embedding_softmax_sharding_params_for_transformers(
              model_p.decoder_embedding_tpl,
              w_vd=w_vd,
              a_blv=a_blv,
              a_bld=a_bld,
              **mesh_kwargs))

    return model_p

  def setup(self) -> None:
    """Constructor."""
    p = self.hparams

    def set_position_emb_model_dims(position_emb_tpl, model_dims):
      assert (position_emb_tpl.embedding_dims == 0 or
              position_emb_tpl.embedding_dims == p.model_dims)
      position_emb_tpl.embedding_dims = model_dims

    def set_model_dims_and_packing(stacked_transformer_tpl, model_dims,
                                   packed_input):
      if stacked_transformer_tpl.cls == transformers.StackedTransformer:
        assert (stacked_transformer_tpl.model_dims == 0 or
                stacked_transformer_tpl.model_dims == model_dims)
        stacked_transformer_tpl.model_dims = model_dims
        stacked_transformer_tpl.packed_input = packed_input
      elif issubclass(stacked_transformer_tpl.cls,
                      transformers.StackedTransformerRepeated):
        assert (stacked_transformer_tpl.block.model_dims == 0 or
                stacked_transformer_tpl.block.model_dims == model_dims)
        stacked_transformer_tpl.block.model_dims = model_dims
        stacked_transformer_tpl.block.packed_input = packed_input
      elif stacked_transformer_tpl.cls == transformers.PipelinedTransformer:
        assert (stacked_transformer_tpl.pipeline_stage.model_dims == 0 or
                stacked_transformer_tpl.pipeline_stage.model_dims == model_dims)
        stacked_transformer_tpl.pipeline_stage.model_dims = model_dims
        stacked_transformer_tpl.pipeline_stage.packed_input = packed_input
      else:
        raise ValueError(f'{stacked_transformer_tpl.cls} not supported.')

    # Create position embeddings.
    if p.position_emb_tpl is not None:
      asserts.none(
          p.encoder_position_emb_tpl,
          msg=('Separate encoder position embeddings must not be set when '
               'shared position embeddings are specified.'))
      asserts.none(
          p.decoder_position_emb_tpl,
          msg=('Separate decoder position embeddings must not be set when '
               'shared position embeddings are specified.'))
      position_emb_tpl = p.position_emb_tpl.clone()
      set_position_emb_model_dims(position_emb_tpl, p.model_dims)
      self.create_child('position_emb', position_emb_tpl)

    # Optional separate encoder position embeddings.
    if p.encoder_position_emb_tpl is not None:
      asserts.none(
          p.position_emb_tpl,
          msg=('Shared position embeddings must not be set when separate '
               'encoder position embeddings are specified.'))
      encoder_position_emb_tpl = p.encoder_position_emb_tpl.clone()
      set_position_emb_model_dims(encoder_position_emb_tpl, p.model_dims)
      self.create_child('encoder_position_emb', encoder_position_emb_tpl)

    # Create the encoder.
    if p.encoder_stacked_transformer_tpl is None:
      raise ValueError(
          'Encoder stack must be specified for TransformerEncoderDecoder.')

    # Use the user specified StackedTransformer for the encoder, assuming
    # everything is set up appropriately.
    encoder_params = p.encoder_stacked_transformer_tpl.clone()
    set_model_dims_and_packing(encoder_params, p.model_dims, p.packed_input)
    # Assert that encoder is not masked.
    if encoder_params.cls == transformers.StackedTransformer:
      mask_self_attention = encoder_params.mask_self_attention
      encoder_num_layers = encoder_params.num_layers
      stacked_encoder_block_params = encoder_params
    elif issubclass(encoder_params.cls,
                    transformers.StackedTransformerRepeated):
      mask_self_attention = encoder_params.block.mask_self_attention
      encoder_num_layers = encoder_params.block.num_layers
      stacked_encoder_block_params = encoder_params.block
    elif encoder_params.cls == transformers.PipelinedTransformer:
      mask_self_attention = encoder_params.pipeline_stage.mask_self_attention
      encoder_num_layers = encoder_params.pipeline_stage.num_layers
      stacked_encoder_block_params = encoder_params.pipeline_stage
    else:
      raise ValueError('Unknown encoder stack.')

    # No decode cache is needed in the encoder.
    if stacked_encoder_block_params.transformer_layer_params_tpl is not None:
      layer_tpl = stacked_encoder_block_params.transformer_layer_params_tpl
      if isinstance(layer_tpl, (list, tuple)):
        for tpl in layer_tpl:
          tpl.tr_atten_tpl.decode_cache = False
      else:
        layer_tpl.tr_atten_tpl.decode_cache = False

    if mask_self_attention:
      raise ValueError(
          'Encoder attention should be un-masked in TransformerEncoderDecoder.')
    self.create_child('encoder', encoder_params)

    # Optional separate embedding layer for source ids.
    if p.encoder_embedding_tpl is not None:
      encoder_embedding_params = p.encoder_embedding_tpl.clone()
      assert (encoder_embedding_params.input_dims == 0 or
              encoder_embedding_params.input_dims == p.model_dims)
      encoder_embedding_params.input_dims = p.model_dims
      self.create_child('encoder_embedding_lookup', encoder_embedding_params)

    # Optional NGrammer layer for the encoder.
    # Paper: https://openreview.net/forum?id=GxjCYmQAody
    if p.encoder_ngrammer_tpl is not None:
      self.create_child('encoder_ngrammer', p.encoder_ngrammer_tpl)

    # Optional post attention NGrammer layer for the encoder.
    if p.encoder_post_attention_ngrammer_tpls is not None:
      ngrammer_tpls = p.encoder_post_attention_ngrammer_tpls
      if len(ngrammer_tpls) != encoder_num_layers:
        raise ValueError('The length of encoder_post_attention_ngrammer_tpls'
                         'must match the number of encoder layers.')
      stacked_encoder_block_params.ngrammer_tpls = ngrammer_tpls

    # Encoder output layer norm.
    if p.encoder_ln_tpl is not None:
      encoder_ln_params = p.encoder_ln_tpl.clone().set(dim=p.model_dims)
      self.create_child('encoder_ln', encoder_ln_params)

    # Optional separate decoder position embeddings.
    if p.decoder_position_emb_tpl is not None:
      asserts.none(
          p.position_emb_tpl,
          msg=('Shared position embeddings must not be set when separate '
               'decoder position embeddings are specified.'))
      decoder_position_emb_tpl = p.decoder_position_emb_tpl.clone()
      set_position_emb_model_dims(decoder_position_emb_tpl, p.model_dims)
      self.create_child('decoder_position_emb', decoder_position_emb_tpl)

    # Create the decoder.
    if p.decoder_stacked_transformer_tpl is None:
      raise ValueError(
          'Decoder stack must be specified for TransformerEncoderDecoder.')

    # Use the user specified StackedTransformer for the decoder, assuming
    # everything is set up appropriately.
    decoder_hparams = p.decoder_stacked_transformer_tpl.clone()
    set_model_dims_and_packing(decoder_hparams, p.model_dims, p.packed_input)
    # Assert that decoder is masked.
    # Assert that encoder is not masked.
    if decoder_hparams.cls == transformers.StackedTransformer:
      mask_self_attention = decoder_hparams.mask_self_attention
      num_decoder_layers = decoder_hparams.num_layers
      stacked_decoder_block_params = decoder_hparams
    elif issubclass(decoder_hparams.cls,
                    transformers.StackedTransformerRepeated):
      mask_self_attention = decoder_hparams.block.mask_self_attention
      num_decoder_layers = decoder_hparams.block.num_layers
      stacked_decoder_block_params = decoder_hparams.block
    elif decoder_hparams.cls == transformers.PipelinedTransformer:
      mask_self_attention = decoder_hparams.pipeline_stage.mask_self_attention
      num_decoder_layers = decoder_hparams.pipeline_stage.num_layers
      stacked_decoder_block_params = decoder_hparams.pipeline_stage
    else:
      raise ValueError('Unknown decoder stack.')

    if not mask_self_attention:
      raise ValueError(
          'Decoder attention should be masked in TransformerEncoderDecoder.')
    self.create_child('decoder', decoder_hparams)

    # Optional separate embedding layer for target ids.
    if p.decoder_embedding_tpl is not None:
      decoder_embedding_params = p.decoder_embedding_tpl.clone()
      assert (decoder_embedding_params.input_dims == 0 or
              decoder_embedding_params.input_dims == p.model_dims)
      decoder_embedding_params.input_dims = p.model_dims
      self.create_child('decoder_embedding_lookup', decoder_embedding_params)

    # Optional NGrammer layer for the decoder.
    # Paper: https://openreview.net/forum?id=GxjCYmQAody
    if p.decoder_ngrammer_tpl is not None:
      self.create_child('decoder_ngrammer', p.decoder_ngrammer_tpl)

    # Optional post attention NGrammer layer for the decoder.
    if p.decoder_post_attention_ngrammer_tpls is not None:
      ngrammer_tpls = p.decoder_post_attention_ngrammer_tpls
      if len(ngrammer_tpls) != num_decoder_layers:
        raise ValueError('The length of decoder_post_attention_ngrammer_tpls'
                         'must match the number of decoder layers.')
      stacked_decoder_block_params.ngrammer_tpls = ngrammer_tpls

    # Decoder output layer norm.
    if p.decoder_ln_tpl:
      decoder_ln_params = p.decoder_ln_tpl.clone().set(dim=p.model_dims)
      self.create_child('decoder_ln', decoder_ln_params)

    # Final softmax.
    softmax_params = p.softmax_tpl.clone()
    assert (softmax_params.input_dims == 0 or
            softmax_params.input_dims == p.model_dims)
    softmax_params.input_dims = p.model_dims
    self.create_child('softmax', softmax_params)

  def encode(self,
             inputs: JTensor,
             input_paddings: JTensor,
             input_segment_ids: Optional[JTensor] = None,
             input_segment_pos: Optional[JTensor] = None,
             input_segment_mask: Optional[JTensor] = None) -> JTensor:
    """Apply the Transformer encoder to the source sequence.

    Args:
      inputs: Input ids. An int32 JTensor of shape [B, S].
      input_paddings: A 0/1 JTensor of shape [B, S] with 1 denoting padding
        correspdonding to the input sequence.
      input_segment_ids: A JTensor of shape [B,S]. The segment that each input
        token belongs to.
      input_segment_pos: A JTensor of shape [B, S]. The position of each input
        token within a segment.
      input_segment_mask: A JTensor or shape [B, 1, S, S]. The segment mask for
        packed input tokens.

    Returns:
      The encoded sequence after applying the Transformer encoder.
    """
    p = self.hparams
    batch, seq_length = inputs.shape
    if p.encoder_embedding_tpl is not None:
      # Encoder has its own embedding lookup table for source ids.
      input_emb = self.encoder_embedding_lookup.emb_lookup(inputs)
    elif p.decoder_embedding_tpl is not None:
      # Encoder shares the same embedding as the target ids.
      # The embedding lookup for target ids is separate from the softmax.
      input_emb = self.decoder_embedding_lookup.emb_lookup(inputs)
    else:
      # Encoder and decoder share the softmax and embedding params.
      input_emb = self.softmax.emb_lookup(inputs)

    if input_segment_ids is None:
      assert input_segment_pos is None
      # Fold the paddings with the segment mask.
      input_segment_ids = jnp.asarray(1 - input_paddings, jnp.int32)
      input_segment_pos = jnp.tile(
          jnp.arange(seq_length, dtype=jnp.int32)[None, :], [batch, 1])
    assert input_segment_ids is not None
    assert input_segment_pos is not None

    # Add NGrammer to the source embeddings.
    if p.encoder_ngrammer_tpl is not None:
      input_emb = self.encoder_ngrammer(
          input_ids=inputs,
          input_embs=input_emb,
          paddings=input_paddings,
          segment_pos=input_segment_pos)

    if p.position_emb_tpl is not None:
      position_emb = self.position_emb(
          seq_length=seq_length, position=input_segment_pos)
      input_emb += position_emb
    elif p.encoder_position_emb_tpl is not None:
      position_emb = self.encoder_position_emb(
          seq_length=seq_length, position=input_segment_pos)
      input_emb += position_emb

    if input_segment_mask is None:
      input_segment_mask = attentions.segment_mask(
          input_segment_ids, dtype=input_emb.dtype)
    encoder_output = self.encoder(
        input_emb,
        input_paddings,
        segment_mask=input_segment_mask,
        segment_pos=input_segment_pos)

    # Final layer norm for encoder output.
    encoder_output = self.encoder_ln(encoder_output)
    return encoder_output

  def compute_loss(self,
                   activations: JTensor,
                   labels: Optional[NestedMap] = None) -> NestedMap:
    """Computes cross entropy loss.

    Args:
      activations: Output of last layer of shape [B, T, D].
      labels: A `.NestedMap` containing the following fields: class_weights, a
        JTensor with shape [B, T] containing weights for each target word.
        class_ids, a JTensor with shape [B, T] of int32 dtype containing the
        target class labels. class_probabilities, a JTensor with shape [B, T, V]
        of float values indicating class-membership probabilities.

    Returns:
      Returns xent_output, where `xent_output` is a `.NestedMap` as defined by
      `SoftmaxLayer`'s return. In addition, per_sequence_xent is added which
      equal to the sum of xent loss for tokens in a sequence.
    """
    if labels is None:
      logits = self.softmax.get_logits(inputs=activations)
      xent_output = NestedMap(logits=logits)
      xent_output.log_probs = jax.nn.log_softmax(logits)
      xent_output.probs = jax.nn.softmax(xent_output.logits)
    else:
      class_ids = None
      class_probabilities = None
      if 'class_ids' in labels:
        class_ids = labels.class_ids[:, :, jnp.newaxis]
      if 'class_probabilities' in labels:
        class_probabilities = labels.class_probabilities
      class_weights = labels.class_weights[:, :, jnp.newaxis]
      xent_output = self.softmax(
          activations,
          class_weights,
          class_ids=class_ids,
          class_probabilities=class_probabilities)
      per_token_xent = (
          xent_output.per_example_xent *
          labels.class_weights.astype(jnp.float32))
      xent_output.per_token_xent = per_token_xent
      xent_output.per_sequence_xent = jnp.sum(
          per_token_xent, -1, dtype=jnp.float32)

      # Sum aux_loss and add to avg_xent.
      aux_loss = 0.0
      aux_loss_weight = 0.0
      if AUX_LOSS in self.variables:
        aux_loss_values = jax.tree_util.tree_leaves(
            self.variables[AUX_LOSS],
            is_leaf=lambda x: isinstance(x, AuxLossStruct))
        for v in aux_loss_values:
          assert isinstance(v, AuxLossStruct)
          aux_loss += jnp.sum(v.value)
          aux_loss_weight += jnp.sum(v.weight)
      if not isinstance(aux_loss, jnp.ndarray):
        aux_loss = jnp.array(aux_loss, dtype=self.fprop_dtype)
        aux_loss_weight = jnp.array(aux_loss_weight, dtype=self.fprop_dtype)
      xent_output.aux_loss = aux_loss
      xent_output.aux_loss_weight = aux_loss_weight
      self.add_summary('total_aux_loss', aux_loss)
      self.add_summary('total_aux_loss_weight', aux_loss_weight)

      # This is the loss to minimize.
      xent_output.total_loss = xent_output.avg_xent + xent_output.aux_loss
    return xent_output

  def __call__(
      self,
      inputs: JTensor,
      input_paddings: JTensor,
      targets: JTensor,
      target_paddings: JTensor,
      labels: Optional[NestedMap] = None,
      input_segment_ids: Optional[JTensor] = None,
      input_segment_pos: Optional[JTensor] = None,
      input_segment_mask: Optional[JTensor] = None,
      target_segment_ids: Optional[JTensor] = None,
      target_segment_pos: Optional[JTensor] = None,
      target_segment_mask: Optional[JTensor] = None,
      cross_segment_mask: Optional[JTensor] = None,
      start_time_step: int = 0,
  ) -> NestedMap:
    """Computes xent loss given the sequence model inputs.

    Args:
      inputs: Input ids. An int32 JTensor of shape [B, S].
      input_paddings: A 0/1 JTensor of shape [B, S] with 1 denoting padding
        correspdonding to the input sequence.
      targets: Target ids. An int32 JTensor of shape [B, T].
      target_paddings: A 0/1 JTensor of shape [B, T] with 1 denoting padding
        corresponding to the target sequence.
      labels: A `.NestedMap` containing the following fields: class_weights, a
        JTensor with shape [batch, seqlen] containing weights for each target
        word. class_ids, a JTensor with shape [B, T] of int32 dtype containing
        the target class labels. class_probabilities, a JTensor with shape [B,
        T, V] of float values indicating class-membership probabilities.
      input_segment_ids: A JTensor of shape [B,S]. The segment that each input
        token belongs to.
      input_segment_pos: A JTensor of shape [B, S]. The position of each input
        token within a segment.
      input_segment_mask: A JTensor or shape [B, 1, S, S]. The segment mask for
        packed input tokens.
      target_segment_ids: A JTensor of shape [B,T]. The segment that each target
        token belongs to.
      target_segment_pos: A JTensor of shape [B, T]. The position of each target
        token within a segment.
      target_segment_mask: A JTensor or shape [B, 1, T, T]. The segment mask for
        packed target tokens.
      cross_segment_mask: A JTensor or shape [B, 1, T, S]. The encoder-decoder
        segment mask.
      start_time_step: Decode extend_step start time step. When decoding after
        prefix, start_time_step will be prefix_len - 1.

    Returns:
      Returns xent_output, where
      `xent_output` is a `.NestedMap` as defined by `SoftmaxLayer`'s return. In
      addition, per_sequence_xent is added which equal to the sum of xent loss
      for tokens in a sequence.
    """
    # Get the input embeddings.
    p = self.hparams
    batch, seq_length = inputs.shape
    _, target_seq_length = targets.shape

    encoder_output = self.encode(inputs, input_paddings, input_segment_ids,
                                 input_segment_pos, input_segment_mask)

    if p.decoder_embedding_tpl is not None:
      # Targets have separate embedding params.
      target_emb = self.decoder_embedding_lookup.emb_lookup(targets)
    else:
      # Embedding parameters are shared with targets and softmax.
      target_emb = self.softmax.emb_lookup(targets)

    if p.decoder_ngrammer_tpl is not None:
      target_emb = self.decoder_ngrammer(
          input_ids=targets,
          input_embs=target_emb,
          paddings=target_paddings,
          segment_pos=target_segment_pos)

    if p.position_emb_tpl is not None:
      targets_position_emb = self.position_emb(
          seq_length=target_seq_length, position=target_segment_pos)
      target_emb += targets_position_emb
    elif p.decoder_position_emb_tpl is not None:
      targets_position_emb = self.decoder_position_emb(
          seq_length=target_seq_length, position=target_segment_pos)
      target_emb += targets_position_emb

    if input_segment_ids is None:
      assert input_segment_pos is None
      # Fold the paddings with the segment mask.
      input_segment_ids = jnp.asarray(1 - input_paddings, jnp.int32)
      input_segment_pos = jnp.tile(
          jnp.arange(seq_length, dtype=jnp.int32)[None, :], [batch, 1])

    if target_segment_ids is None:
      assert target_segment_pos is None
      # Fold the paddings with the segment mask.
      target_segment_ids = jnp.asarray(1 - target_paddings, jnp.int32)
      target_segment_pos = jnp.tile(
          jnp.arange(target_seq_length, dtype=jnp.int32)[None, :], [batch, 1])

    # Cross attention.
    if cross_segment_mask is None:
      cross_segment_mask = attentions.segment_mask(target_segment_ids,
                                                   input_segment_ids,
                                                   target_emb.dtype)
    if target_segment_mask is None:
      target_segment_mask = attentions.causal_segment_mask(
          target_segment_ids, target_emb.dtype)
    # Update caches for decode state.
    if self.is_mutable_collection(base_layer.DECODE_CACHE):
      self.update_decode_state('time_step', start_time_step)
      self.update_decode_state('encoder_output', encoder_output)
      self.update_decode_state('input_paddings', input_paddings)
    output = self.decoder(
        target_emb,
        target_paddings,
        target_segment_mask,
        cross_inputs=encoder_output,
        cross_paddings=input_paddings,
        cross_segment_mask=cross_segment_mask,
        segment_pos=target_segment_pos)

    # Final layer norm for decoder.
    output = self.decoder_ln(output)

    return self.compute_loss(output, labels)

  def transform_decode_state(
      self, transform_fn: base_layer.DecodeStateTransformFn) -> None:
    """Transforms all decode state variables based on transform_fn."""
    self.decoder.transform_decode_state(transform_fn)

  def init_states(self, inputs: JTensor, input_paddings: JTensor, *args: Any,
                  **kwargs: Any) -> None:
    """Initialize the cache for autoregressive decoding.

    Args:
      inputs: Input ids. An int32 JTensor of shape [B, S].
      input_paddings: A 0/1 JTensor of shape [B, S] with 1 denoting padding
        correspdonding to the input sequence.
      *args: Other arguments.
      **kwargs: Other keyword arguments.

    Returns:
      A `.NestedMap` corresponding to the cache.
    """
    raise NotImplementedError(type(self))

  def extend_step(self, targets: JTensor) -> NestedMap:
    """Autoregressive cached decoding of the Transformer encoder decoder.

    Args:
      targets: Target sequence of shape [B] or [B, P] corresponding to target
        sequence at index time_step. Note that the shape [B, P] corresponds to a
        prefix which is useful for decoding in some special architectures such
        as Primer or Ngrammer. B can be a multiple of the batch size for the
        encoder, where each encoder output will generate multiple samples; the
        encoder output will be repeated on the batch dimension in this case.

    Returns:
      xent_output: A `.NestedMap` object containing the log probabilities and
        probabilities.
    """
    p = self.hparams
    # Fetch encoder output from the cache.
    input_paddings = self.get_decode_state('input_paddings')

    # During autoregressive decoding inputs and targets are not packed.
    if len(targets.shape) == 1:
      targets = targets[:, jnp.newaxis]

    if p.decoder_embedding_tpl is not None:
      # Targets have separate embedding params.
      target_emb = self.decoder_embedding_lookup.emb_lookup(targets)
    else:
      # Embedding parameters are shared with targets and softmax.
      target_emb = self.softmax.emb_lookup(targets)

    time_step = self.get_decode_state('time_step')
    if p.decoder_ngrammer_tpl is not None:
      target_emb = self.decoder_ngrammer(
          targets, target_emb, paddings=None, segment_pos=None)

    targets = targets[:, -1][:, jnp.newaxis]
    target_emb = target_emb[:, -1, :][:, jnp.newaxis, :]

    # During autoregressive decoding inputs are not packed.
    segment_pos = jnp.zeros((targets.shape[0], 1)) + time_step

    # Add position embeddings to target ids.
    if p.position_emb_tpl is not None:
      target_position_emb = self.position_emb(
          seq_length=1, position=segment_pos)
      target_emb += target_position_emb
    elif p.decoder_position_emb_tpl is not None:
      target_position_emb = self.decoder_position_emb(
          seq_length=1, position=segment_pos)
      target_emb += target_position_emb

    outputs = self.decoder.extend_step(
        target_emb[:, 0, :],
        time_step=time_step,
        cross_paddings=input_paddings)

    self.update_decode_state('time_step', time_step + 1)
    outputs = self.decoder_ln(outputs)
    xent_output = self.compute_loss(outputs)
    return xent_output
