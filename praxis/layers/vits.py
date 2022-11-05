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

"""Layers for Vision Transformer (Vit).

The top level module, VisionTransformer, defines an interface where a user
can provide an entry block, a transformer block and an exit blocks. This gives
user the maximum freedom to mix-and-match different implementations and
configurations of these three blocks.

The following notations are used through this file:
B = batch size
H = height
W = width
P = patch size
C = number of channels
N = number of tokens
D = hidden dims
"""

from __future__ import annotations

from typing import Tuple, Sequence

import einops
import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_hyperparams
from praxis import base_layer
from praxis import py_utils
from praxis import pytypes
from praxis.layers import activations
from praxis.layers import embedding_softmax
from praxis.layers import linears
from praxis.layers import normalizations
from praxis.layers import poolings
from praxis.layers import stochastics
from praxis.layers import transformers

NestedMap = py_utils.NestedMap
JTensor = pytypes.JTensor

BaseHParams = base_layer.BaseLayer.HParams
WeightHParams = base_layer.WeightHParams
WeightInit = base_layer.WeightInit
sub_config_field = base_hyperparams.sub_config_field


def image_to_patch(img: JTensor, patch_size: int) -> JTensor:
  """Converts an image to patches.

  Args:
    img: JTensor, [B, H, W, C] ,
    patch_size: integer, dimension of a square patch.

  Returns:
    batched_img: [B, (H * W / P^2), P^2 * C].
  """

  if len(img.shape) < 4:
    raise ValueError('Image should be formatted as 4D [B, H, W, C], '
                     f'Shape: {img.shape}')
  height, width, channels = img.shape[1:]

  if height % patch_size != 0 or width % patch_size != 0:
    raise ValueError(
        f'Image height ({height}) and width ({width}) should be multiples '
        f'of patch_size ({patch_size}).')

  row_blocks = height // patch_size
  column_blocks = width // patch_size

  img = einops.rearrange(
      img,
      '... (m p)(n q) c->...(m n)(p q c)',
      m=row_blocks,
      n=column_blocks,
      p=patch_size,
      q=patch_size,
      c=channels)
  return img


def patch_to_image(patches: JTensor, img_shape: Sequence[int], patch_size: int):
  """Converts patches to an image with the given image shape and the patch size.

  Args:
    patches: JTensor, [batch_size, num_patches, patch_content].
    img_shape: A sequence of 4 integers specifying
      [batch_size, image_height, image_width, image_channel].
    patch_size: An integer specifying the patch size.

  Returns:
    The image converted from patches of the shape specified by img_shape.
  """
  if len(img_shape) != 4:
    raise ValueError(
        f'Image shape is expected to be [B, H, W, C], got {img_shape}.')
  height, width, channels = img_shape[1:]

  if height % patch_size != 0 or width % patch_size != 0:
    raise ValueError(
        f'Image height ({height}) and width ({width}) should be multiples '
        f'of patch_size ({patch_size}).')

  row_blocks = height // patch_size
  column_blocks = width // patch_size
  return einops.rearrange(
      patches,
      '... (m n)(p q c) -> ... (m p)(n q) c',
      m=row_blocks,
      n=column_blocks,
      p=patch_size,
      q=patch_size,
      c=channels)


def interpolate_embedding_2d(emb, source_emb_shape, target_emb_shape):
  """Interpolates a 2D positional embedding to a new shape.

  Args:
    emb: JTensor, (1, H1xW1, D), flattened 2D positional embedding.
    source_emb_shape: Tuple, (H1, W1), height and width of the source embedding.
    target_emb_shape: Tuple, (H2, W2), height and width of the target embedding.

  Returns:
    Flattened, interpolated embedding of shape (1, H2xW2, D)
  """

  if len(emb.shape) > 3 or emb.shape[0] != 1:
    raise ValueError('The shape of the embedding should be (1, H * W, D)')

  if emb.shape[1] != source_emb_shape[0] * source_emb_shape[1]:
    raise ValueError('The shape of the embedding does NOT match input specs.')

  emb_dims = emb.shape[2]
  emb = jnp.reshape(emb, (source_emb_shape[0], source_emb_shape[1], emb_dims))

  target_emb = jax.image.resize(
      emb, (target_emb_shape[0], target_emb_shape[1], emb_dims),
      method='bilinear')
  target_emb = jnp.reshape(
      target_emb, (1, target_emb_shape[0] * target_emb_shape[1], emb_dims))

  return target_emb


class VitEntryLayers(base_layer.BaseLayer):
  """Entry block of ViT.

  It performs the following operations:
    - patchifying the input
    - linear projection
    - adding positional embedding
    - adding potential dropouts
  """

  # TODO(zhangzd): If needed, add support for non-square patches.
  class HParams(BaseHParams):
    """Associated hyper-params for this layer class.

    Attributes:
      pos_embed_shapes: Height/width of the positional embedding. This param is
        used to support images of different shapes. When the embedding_size is
        not equal to image_size / patch_size, interpolation will be employed to
        generate embeddings of image_size / patch_size.
      input_dims: Dims per patch before input patch projection.
      output_dims: Dims per patch after input patch projection.
      image_channels: Number of channels of the input image.
      prepend_cls_tokens: If > 0, the layer will prepend N CLS token before
        the patch features.
      append_cls_tokens: If > 0, the layer will append N CLS token after
        the patch features.
      pos_emb_tpl: template for positional embeddings.
    """
    pos_embed_shapes: Tuple[int, int] = (0, 0)
    patch_size: int = 0
    input_dims: int = 0
    output_dims: int = 0
    pos_emb_dropout_prob: float = 0.0
    prepend_cls_tokens: int = 0
    append_cls_tokens: int = 0
    # configurable components
    pos_emb_tpl: BaseHParams = sub_config_field(
        embedding_softmax.TrainablePositionalEmbedding.HParams)

  def setup(self) -> None:
    p = self.hparams

    p_patch_projection = linears.FeedForward.HParams(
        name='proj',
        input_dims=p.input_dims,
        output_dims=p.output_dims,
        activation_tpl=activations.Identity.HParams())
    self.create_child('patch_projection', p_patch_projection)

    pos_emb = p.pos_emb_tpl.clone().set(name='emb')
    self.create_child('pos_emb', pos_emb)

    if p.pos_emb_dropout_prob > 0.0:
      p_dropout = stochastics.Dropout.HParams(
          name='dropout', keep_prob=1.0 - p.pos_emb_dropout_prob)
      self.create_child('dropout', p_dropout)

    if p.prepend_cls_tokens > 0:
      self.create_variable(
          'prepend_cls_embs',
          WeightHParams(
              shape=[1, p.prepend_cls_tokens, p.output_dims],
              init=WeightInit.Constant(0.0)))
    if p.append_cls_tokens > 0:
      self.create_variable(
          'append_cls_embs',
          WeightHParams(
              shape=[1, p.append_cls_tokens, p.output_dims],
              init=WeightInit.Constant(0.0)))

  def __call__(self, inputs: JTensor) -> JTensor:
    """Applies the vit entry operations to the input image.

    Args:
      inputs: Input image tensor of allowed shapes:
        - [B, H, W, 3], where inputs are images
        - [B, N, D], where inputs are a sequence of embeddings or patches

    Returns:
      Output tensor of shape [B, N, D].
    """
    p = self.hparams

    if len(inputs.shape) == 4:
      height, width = inputs.shape[1:3]
      if height % p.patch_size != 0 or width % p.patch_size != 0:
        raise ValueError(
            f'Image height ({height}) and width ({width}) should be multiples '
            f'of p.patch_size ({p.patch_size}).')
      patches = image_to_patch(inputs, p.patch_size)
    elif len(inputs.shape) == 3:
      patches = inputs
    else:
      raise ValueError('Input image tensor allows [B, H, W, 3] or [B, N, D].')

    features = self.patch_projection(patches)

    num_pos_embed = np.prod(p.pos_embed_shapes)
    num_pos_embed += p.prepend_cls_tokens
    num_pos_embed += p.append_cls_tokens
    pos_emb = self.pos_emb(seq_length=num_pos_embed)

    prepend_cls_pos_emb = pos_emb[:, :p.prepend_cls_tokens, :]
    input_pos_emb = pos_emb[:, p.prepend_cls_tokens:num_pos_embed -
                            p.append_cls_tokens]
    append_cls_pos_emb = pos_emb[:, -p.append_cls_tokens:, :]

    # Only support image shape for pos interpolation.
    if len(inputs.shape) == 4:
      row_patch_count = height // p.patch_size
      col_patch_count = width // p.patch_size
      if p.pos_embed_shapes != (row_patch_count, col_patch_count):
        input_pos_emb = interpolate_embedding_2d(
            input_pos_emb, p.pos_embed_shapes,
            (row_patch_count, col_patch_count))

    features = features + input_pos_emb
    if self.hparams.pos_emb_dropout_prob > 0.0:
      features = self.dropout(features)

    batch_size = inputs.shape[0]
    if p.prepend_cls_tokens > 0:
      prepend_cls_embs = jnp.tile(self.theta.prepend_cls_embs,
                                  (batch_size, 1, 1))
      prepend_cls_embs = prepend_cls_embs + prepend_cls_pos_emb
      features = jnp.concatenate((prepend_cls_embs, features), axis=1)

    if p.append_cls_tokens > 0:
      append_cls_embs = jnp.tile(self.theta.append_cls_embs, (batch_size, 1, 1))
      append_cls_embs = append_cls_embs + append_cls_pos_emb
      features = jnp.concatenate((features, append_cls_embs), axis=1)

    return features


class VitExitLayers(base_layer.BaseLayer):
  """Exit block of ViT.

  It consists of layer norm, pooling, projection and dropout.
  """

  class HParams(BaseHParams):
    """Associated hyper-params for this layer class.

    Attributes:
      hidden_dim: Number of channels of the input tensor.
      output_dim: Number of channels of the output tensor.
      output_dropout_prob: Probability to apply dropout on the output tensor.
      pooled: Global max pooling over all output tokens.
      pre_ln: If true, add a layer norm at the beginning of this layer.
      output_fc_tanh: Whether to include a linear projection layer with tanh
        activation on the output.
    """
    hidden_dim: int = 0
    output_dim: int = 0
    output_dropout_prob: float = 0.0
    pooled: bool = True
    pre_ln: bool = True
    output_fc_tanh: bool = True

  def setup(self) -> None:
    p = self.hparams

    if p.pre_ln:
      p_ln = normalizations.LayerNorm.HParams(name='ln', dim=p.hidden_dim)
      self.create_child('ln', p_ln)

    if p.pooled:
      p_pooling = poolings.GlobalPooling.HParams(
          pooling_type='MAX', pooling_dims=[1], keepdims=False)
      self.create_child('pooling', p_pooling)

    if p.output_fc_tanh:
      p_fc_tanh = linears.FeedForward.HParams(
          input_dims=p.hidden_dim,
          output_dims=p.output_dim,
          activation_tpl=activations.Tanh.HParams())
      self.create_child('fc_tanh', p_fc_tanh)
    elif p.output_dim != 0 and p.hidden_dim != p.output_dim:
      p_fc = linears.FeedForward.HParams(
          input_dims=p.hidden_dim,
          output_dims=p.output_dim,
          activation_tpl=activations.Identity.HParams())
      self.create_child('output_projection', p_fc)

    if p.output_dropout_prob > 0.0:
      p_dropout = stochastics.Dropout.HParams(keep_prob=1.0 -
                                              p.output_dropout_prob)
      self.create_child('dropout', p_dropout)

  def __call__(self, inputs: JTensor) -> JTensor:
    """FProp function.

    Args:
      inputs: Input tensor of shape [B, N, D].

    Returns:
      Output tensor of shape [B, D] or [B, N, D] if pooled == False.
    """
    p = self.hparams
    if p.pre_ln:
      inputs = self.ln(inputs)
    if p.pooled:
      inputs = self.pooling(inputs)
    if p.output_fc_tanh:
      inputs = self.fc_tanh(inputs)
    elif p.output_dim != 0 and p.hidden_dim != p.output_dim:
      inputs = self.output_projection(inputs)

    if p.output_dropout_prob > 0.0:
      inputs = self.dropout(inputs)
    return inputs


class VisionTransformer(base_layer.BaseLayer):
  """Vision transformer model."""

  class HParams(BaseHParams):
    """Associated hyper-params for this layer class.

    This class follows a minimalistic design pattern. Users need to configure
    the templates for the submodules themselves; this increases the
    generalizability of this class.

    Attributes:
      entry_layers_tpl: An integer specifying hidden dimension of transformers.
      transformer_layers_tpl: An integer specifying number of transformers.
      exit_layers_tpl: An integer specifying number of attention heads in
        transformers.
    """
    entry_layers_tpl: BaseHParams = sub_config_field(VitEntryLayers.HParams)
    transformer_layers_tpl: BaseHParams = sub_config_field(
        transformers.StackedTransformer.HParams)
    exit_layers_tpl: BaseHParams = sub_config_field(VitExitLayers.HParams)

  def setup(self) -> None:
    p = self.hparams

    if p.entry_layers_tpl is not None:
      self.create_child('entry_stack', p.entry_layers_tpl)

    if p.transformer_layers_tpl is None:
      raise ValueError('transformer_layers_tpl should not be None')
    self.create_child('transformers_stack', p.transformer_layers_tpl)

    if p.exit_layers_tpl is not None:
      self.create_child('exit_stack', p.exit_layers_tpl)

  def __call__(self, inputs: JTensor, paddings: JTensor = None) -> JTensor:
    """Applies the Vit model to the inputs.

    Args:
      inputs: Input image tensor of allowed shapes:
        - [B, H, W, 3], where inputs are images
        - [B, N, D], where inputs are a sequence of embeddings or patches
      paddings: Optional [B, N] padding field of inputs when inputs are with
        [B, N, D].

    Returns:
      Output tensor of shape [B, D] or [B, N, D] if pooled == False.
    """
    p = self.hparams
    features = inputs
    if p.entry_layers_tpl:
      features = self.entry_stack(features)  # [B, N, D]
    if paddings is None:
      paddings = jnp.zeros(features.shape[:-1], dtype=features.dtype)
    features = self.transformers_stack(features, paddings)  # [B, N, D]
    if p.exit_layers_tpl:
      features = self.exit_stack(features)  # [B, D] or [B, N, D]
    return features


def build_vision_transformer_hparams_for_test(
    pos_embed_shapes: Tuple[int, int], patch_size: int, image_channels: int,
    model_dims: int, mlp_dims: int, num_xformer_layers: int,
    num_heads: int) -> VisionTransformer.HParams:
  """Builds a minimal vision transformer layer for unit tests.

  For simplicity, only minimum number of parameters can be specified. User needs
  to set the templates for the submodules themselves to enable dropout,
  droppath, atten_logit_caps, etc.

  Args:
    pos_embed_shapes: A tuple of (int, int), height and width of the positional
      embedding.
    patch_size: An integer specifying the size of patch for ViT.
    image_channels: An integer specifying the number of channels of input image.
    model_dims: An integer specifying the model dimension of transformers.
    mlp_dims: An integer specifying mlp dimension of transformers.
    num_xformer_layers: An integer specifying number of transformers.
    num_heads: An integer specifying number of attention heads in transformers.

  Returns:
    A HParams of the VisionTransformer layer.
  """
  pos_emb_tpl = embedding_softmax.TrainablePositionalEmbedding.HParams(
      max_seq_length=np.prod(pos_embed_shapes),
      embedding_dims=model_dims,
      params_init=WeightInit.Gaussian(scale=0.02),
  )
  p_entry = VitEntryLayers.HParams(
      name='entry',
      pos_embed_shapes=pos_embed_shapes,
      patch_size=patch_size,
      input_dims=patch_size ** 2 * image_channels,
      output_dims=model_dims,
      pos_emb_tpl=pos_emb_tpl,
  )

  p_stacked_tfm = transformers.StackedTransformer.HParams(
      model_dims=model_dims,
      hidden_dims=mlp_dims,
      num_heads=num_heads,
      mask_self_attention=False,
      use_cross_attention=False,
      packed_input=False,
      num_layers=num_xformer_layers,
  )
  p_tfm = p_stacked_tfm.transformer_layer_params_tpl
  p_tfm.tr_atten_tpl.internal_enable_per_dim_scale = False  # pytype: disable=attribute-error  # enable-nested-classes

  p_exit = VitExitLayers.HParams(
      name='exit', hidden_dim=model_dims, output_dim=model_dims)

  p_vit = VisionTransformer.HParams(
      name='vit',
      entry_layers_tpl=p_entry,
      transformer_layers_tpl=p_stacked_tfm,
      exit_layers_tpl=p_exit,
  )

  return p_vit
