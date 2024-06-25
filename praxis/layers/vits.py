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

from typing import Sequence

import einops
import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_hyperparams
from praxis import base_layer
from praxis import pax_fiddle
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
LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]
WeightHParams = base_layer.WeightHParams
WeightInit = base_layer.WeightInit
template_field = base_layer.template_field


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
    raise ValueError(
        f'The shape of the embedding ({emb.shape}) does NOT match input'
        f' specs ({source_emb_shape}); expected embedding shape[1] to be the '
        'product of the input spec.'
    )

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

  Attributes:
    pos_emb_shapes: Shape of the positional embedding. This param is used to
      support images of different shapes. When the embedding is 2d and its shape
      is not equal to image_size / patch_size, interpolation will be employed to
      generate embeddings of image_size / patch_size.
    input_dims: Dims per patch before input patch projection.
    output_dims: Dims per patch after input patch projection.
    image_channels: Number of channels of the input image.
    prepend_cls_tokens: If > 0, the layer will prepend N CLS token before
      the patch features.
    append_cls_tokens: If > 0, the layer will append N CLS token after
      the patch features.
    pos_emb_tpl: template for positional embeddings.
    input_fc_has_bias: Whether the input projection layer has bias.
  """

  # TODO(zhangzd): If needed, add support for non-square patches.
  pos_emb_shapes: tuple[int, int] = (0, 0)
  patch_size: int = 0
  input_dims: int = 0
  output_dims: int = 0
  pos_emb_dropout_prob: float = 0.0
  prepend_cls_tokens: int = 0
  append_cls_tokens: int = 0
  pos_emb_tpl: LayerTpl | None = template_field(
      embedding_softmax.TrainablePositionalEmbedding
  )
  input_fc_has_bias: bool = True

  def setup(self) -> None:

    p_patch_projection = pax_fiddle.Config(
        linears.FeedForward,
        name='proj',
        input_dims=self.input_dims,
        output_dims=self.output_dims,
        has_bias=self.input_fc_has_bias,
        activation_tpl=pax_fiddle.Config(activations.Identity),
    )
    self.create_child('patch_projection', p_patch_projection)

    if self.pos_emb_tpl:
      pos_emb = self.pos_emb_tpl.clone().set(name='emb')
      self.create_child('pos_emb', pos_emb)

    if self.pos_emb_dropout_prob > 0.0:
      p_dropout = pax_fiddle.Config(
          stochastics.Dropout,
          name='dropout',
          keep_prob=1.0 - self.pos_emb_dropout_prob,
      )
      self.create_child('dropout', p_dropout)

    if self.prepend_cls_tokens > 0:
      self.create_variable(
          'prepend_cls_embs',
          WeightHParams(
              shape=[1, self.prepend_cls_tokens, self.output_dims],
              init=WeightInit.Constant(0.0),
          ),
      )
    if self.append_cls_tokens > 0:
      self.create_variable(
          'append_cls_embs',
          WeightHParams(
              shape=[1, self.append_cls_tokens, self.output_dims],
              init=WeightInit.Constant(0.0),
          ),
      )

  def __call__(self, inputs: JTensor) -> JTensor:
    """Applies the vit entry operations to the input image.

    Args:
      inputs: Input image tensor of allowed shapes:
        - [B, H, W, 3], where inputs are images
        - [B, N, D], where inputs are a sequence of embeddings or patches

    Returns:
      Output tensor of shape [B, N, D].
    """

    if len(inputs.shape) == 4:
      height, width = inputs.shape[1:3]
      if height % self.patch_size != 0 or width % self.patch_size != 0:
        raise ValueError(
            f'Image height ({height}) and width ({width}) should be multiples '
            f'of p.patch_size ({self.patch_size}).'
        )
      patches = image_to_patch(inputs, self.patch_size)
    elif len(inputs.shape) == 3:
      patches = inputs
    else:
      raise ValueError('Input image tensor allows [B, H, W, 3] or [B, N, D].')

    features = self.patch_projection(patches)

    if self.pos_emb_tpl:
      num_pos_embed = np.prod(self.pos_emb_shapes)
      pos_emb = self.pos_emb(seq_length=num_pos_embed)
      # Only support image shape and 2d pos_emb_shape for pos interpolation.
      if len(inputs.shape) == 4 and len(self.pos_emb_shapes) == 2:
        row_patch_count = height // self.patch_size
        col_patch_count = width // self.patch_size
        if self.pos_emb_shapes != (row_patch_count, col_patch_count):
          pos_emb = interpolate_embedding_2d(
              pos_emb, self.pos_emb_shapes, (row_patch_count, col_patch_count)
          )
      features = features + pos_emb

    if self.pos_emb_dropout_prob > 0.0:
      features = self.dropout(features)

    batch_size = inputs.shape[0]
    if self.prepend_cls_tokens > 0:
      prepend_cls_embs = jnp.tile(self.theta.prepend_cls_embs,
                                  (batch_size, 1, 1))
      features = jnp.concatenate((prepend_cls_embs, features), axis=1)

    if self.append_cls_tokens > 0:
      append_cls_embs = jnp.tile(self.theta.append_cls_embs, (batch_size, 1, 1))
      features = jnp.concatenate((features, append_cls_embs), axis=1)

    return features


class VitExitLayers(base_layer.BaseLayer):
  """Exit block of ViT.

  It consists of layer norm, pooling, projection and dropout.

  Attributes:
    hidden_dim: Number of channels of the input tensor.
    output_dim: Number of channels of the output tensor.
    output_dropout_prob: Probability to apply dropout on the output tensor.
    pooled: Apply pooling layer over all output tokens.
    pre_ln: If true, add a layer norm at the beginning of this layer.
    output_fc_tanh: Whether to include a linear projection layer with tanh
      activation on the output.
    output_fc_has_bias: Whether the output projection layer has bias.
    pooling_tpl: Pooling layer config to use, defaults to global
      max pooling if not set.
  """
  hidden_dim: int = 0
  output_dim: int = 0
  output_dropout_prob: float = 0.0
  pooled: bool = True
  pre_ln: bool = True
  output_fc_tanh: bool = True
  output_fc_has_bias: bool = True
  pooling_tpl: LayerTpl = template_field(poolings.GlobalPooling)
  ln_tpl: LayerTpl = template_field(normalizations.LayerNorm)

  def setup(self) -> None:

    if self.pre_ln:
      p_ln = self.ln_tpl.clone().set(name='ln', dim=self.hidden_dim)
      self.create_child('ln', p_ln)

    if self.pooled:
      self.create_child('pooling', self.pooling_tpl)

    if self.output_fc_tanh:
      p_fc_tanh = pax_fiddle.Config(
          linears.FeedForward,
          input_dims=self.hidden_dim,
          output_dims=self.output_dim,
          has_bias=self.output_fc_has_bias,
          activation_tpl=pax_fiddle.Config(activations.Tanh),
      )
      self.create_child('fc_tanh', p_fc_tanh)
    elif self.output_dim != 0 and self.hidden_dim != self.output_dim:
      p_fc = pax_fiddle.Config(
          linears.FeedForward,
          input_dims=self.hidden_dim,
          output_dims=self.output_dim,
          has_bias=self.output_fc_has_bias,
          activation_tpl=pax_fiddle.Config(activations.Identity),
      )
      self.create_child('output_projection', p_fc)

    if self.output_dropout_prob > 0.0:
      p_dropout = pax_fiddle.Config(
          stochastics.Dropout, keep_prob=1.0 - self.output_dropout_prob
      )
      self.create_child('dropout', p_dropout)

  def __call__(self, inputs: JTensor) -> JTensor:
    """FProp function.

    Args:
      inputs: Input tensor of shape [B, N, D].

    Returns:
      Output tensor of shape [B, D] or [B, N, D] if pooled == False.
    """
    if self.pre_ln:
      inputs = self.ln(inputs)
    if self.pooled:
      inputs = self.pooling(inputs)
    if self.output_fc_tanh:
      inputs = self.fc_tanh(inputs)
    elif self.output_dim != 0 and self.hidden_dim != self.output_dim:
      inputs = self.output_projection(inputs)

    if self.output_dropout_prob > 0.0:
      inputs = self.dropout(inputs)
    return inputs


class VisionTransformer(base_layer.BaseLayer):
  """Vision transformer model.

  This class follows a minimalistic design pattern. Users need to configure
  the templates for the submodules themselves; this increases the
  generalizability of this class.

  Attributes:
    entry_layers_tpl: An integer specifying hidden dimension of transformers.
    transformer_layers_tpl: An integer specifying number of transformers.
    exit_layers_tpl: An integer specifying number of attention heads in
      transformers.
    full_data_parallel_on_entry_exit: Whether to apply data parallelism over all
      devices on the entry and exit layers. This is a convenient way to shard
      small entry/exit layers.
  """
  entry_layers_tpl: LayerTpl = template_field(VitEntryLayers)
  transformer_layers_tpl: LayerTpl = template_field(
      transformers.StackedTransformer
  )
  exit_layers_tpl: LayerTpl = template_field(VitExitLayers)
  full_data_parallel_on_entry_exit: bool = False

  class ActivationSharding(base_layer.BaseLayer.ActivationSharding):
    """Represents how intermediate values should be partitioned across a mesh.

    Attributes:
      network_inputs: How the activations will be sharded in the transformer
        layers (after entry, before exits).
    """

    network_inputs: base_layer.SplitDimsMapping = None

  def setup(self) -> None:

    if self.entry_layers_tpl is not None:
      self.create_child('entry_stack', self.entry_layers_tpl)

    if self.transformer_layers_tpl is None:
      raise ValueError('transformer_layers_tpl should not be None')
    self.create_child('transformers_stack', self.transformer_layers_tpl)

    if self.exit_layers_tpl is not None:
      self.create_child('exit_stack', self.exit_layers_tpl)

  def shard_entry_exit(self, x: JTensor) -> JTensor:
    # Fully data parallel on all mesh axes.
    if (
        self.mesh_axis_names is None
        or not self.full_data_parallel_on_entry_exit
    ):
      return x
    return base_layer.maybe_shard(
        x,
        [self.mesh_axis_names] + [None] * (x.ndim - 1),
        self.mesh_axis_names,
    )

  def __call__(self, inputs: JTensor, paddings: JTensor = None) -> JTensor:  # pytype: disable=annotation-type-mismatch  # jax-ndarray
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
    features = inputs
    ap = self.activation_split_dims_mapping
    if self.entry_layers_tpl:
      features = self.shard_entry_exit(features)
      features = self.entry_stack(features)  # [B, N, D]
      features = self.shard_entry_exit(features)
      features = base_layer.maybe_shard(
          features, ap.network_inputs, self.mesh_axis_names
      )
    if paddings is None:
      paddings = jnp.zeros(features.shape[:-1], dtype=features.dtype)
    features = self.transformers_stack(features, paddings)  # [B, N, D]
    if self.exit_layers_tpl:
      features = base_layer.maybe_shard(
          features, ap.network_inputs, self.mesh_axis_names
      )
      features = self.shard_entry_exit(features)
      features = self.exit_stack(features)  # [B, D] or [B, N, D]
      features = self.shard_entry_exit(features)
    return features


def build_vision_transformer_hparams_for_test(
    pos_emb_shapes: tuple[int, int],
    patch_size: int,
    image_channels: int,
    model_dims: int,
    mlp_dims: int,
    num_xformer_layers: int,
    num_heads: int,
) -> pax_fiddle.Config[VisionTransformer]:
  """Builds a minimal vision transformer layer for unit tests.

  For simplicity, only minimum number of parameters can be specified. User needs
  to set the templates for the submodules themselves to enable dropout,
  droppath, atten_logit_caps, etc.

  Args:
    pos_emb_shapes: A tuple of (int, int), height and width of the positional
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
  pos_emb_tpl = pax_fiddle.Config(
      embedding_softmax.TrainablePositionalEmbedding,
      max_seq_length=np.prod(pos_emb_shapes),
      embedding_dims=model_dims,
      params_init=WeightInit.Gaussian(scale=0.02),
  )
  p_entry = pax_fiddle.Config(
      VitEntryLayers,
      name='entry',
      pos_emb_shapes=pos_emb_shapes,
      patch_size=patch_size,
      input_dims=patch_size**2 * image_channels,
      output_dims=model_dims,
      pos_emb_tpl=pos_emb_tpl,
  )

  p_stacked_tfm = pax_fiddle.Config(
      transformers.StackedTransformer,
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

  p_exit = pax_fiddle.Config(
      VitExitLayers, name='exit', hidden_dim=model_dims, output_dim=model_dims
  )

  p_vit = pax_fiddle.Config(
      VisionTransformer,
      name='vit',
      entry_layers_tpl=p_entry,
      transformer_layers_tpl=p_stacked_tfm,
      exit_layers_tpl=p_exit,
  )

  return p_vit
