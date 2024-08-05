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

"""VQVAE model from videogvt/models/vqvae.py."""

from typing import Sequence

import jax
from praxis import base_layer
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis.layers import activations
from praxis.layers.video import enc_dec_3dcnn
from praxis.layers.video import quantizer

template_field = base_layer.template_field
LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]
JTensor = pytypes.JTensor
NestedMap = py_utils.NestedMap


class Encoder(base_layer.BaseLayer):
  """Encoder structure with 3D CNNs."""

  filters: int = 128
  input_dim: int = 3  # RGB
  embedding_dim: int = 8
  num_res_blocks: int = 4
  temporal_downsample: Sequence[bool] = (False, True, True)
  conv_downsample: bool = True
  channel_multipliers: Sequence[int] = (1, 2, 2, 4)
  norm_tpl: LayerTpl = template_field(enc_dec_3dcnn.GroupNormSpatial)
  conv_tpl: LayerTpl = template_field(enc_dec_3dcnn.CausalConv)
  res_block_tpl: LayerTpl = template_field(enc_dec_3dcnn.ResBlock)
  activation_tpl: pax_fiddle.Config[activations.BaseActivation] = (
      template_field(activations.Swish)
  )

  def _check_input(self, x: JTensor) -> None:
    if x.ndim != 5:
      raise ValueError('Input shape must be (b, t, h, w, c).')
    temporal_downsample = 2 ** sum(self.temporal_downsample)
    if (x.shape[1] - 1) % temporal_downsample != 0:
      raise ValueError(
          'For a causal tokenizer, input must be (1 + k * %d) frames to avoid'
          ' truncation during downsampling.  Input frames = %d'
          % (temporal_downsample, x.shape[1])
      )

  def setup(self):
    if not self.conv_downsample:
      raise ValueError('conv_downsample must be True.')
    if self.num_res_blocks < 1:
      raise ValueError('num_res_blocks must be >= 1.')

    conv_p = self.conv_tpl.clone()
    conv_p.name = 'conv_first'
    conv_p.dtype = self.dtype
    conv_p.fprop_dtype = self.fprop_dtype
    conv_p.bias = False
    conv_p.filter_shape = (3, 3, 3, self.input_dim, self.filters)
    self.create_child('conv_first', conv_p)

    res_blocks = []
    conv_downsample_layers = []
    input_dim = self.filters
    output_dim = self.filters
    num_blocks = len(self.channel_multipliers)
    for i in range(num_blocks):
      output_dim = self.filters * self.channel_multipliers[i]
      res_p = self.res_block_tpl.clone()
      res_p.name = f'res_block_{i}_0'
      res_p.input_dim = input_dim
      res_p.output_dim = output_dim
      res_p.dtype = self.dtype
      res_p.fprop_dtype = self.fprop_dtype
      res_p.use_conv_shortcut = False
      res_blocks.append(res_p)
      for j in range(1, self.num_res_blocks):
        res_p = res_p.clone()
        res_p.name = f'res_block_{i}_{j}'
        res_p.input_dim = output_dim
        res_blocks.append(res_p)
      if i < num_blocks - 1:
        if self.conv_downsample:
          t_stride = 2 if self.temporal_downsample[i] else 1
          conv_p = self.conv_tpl.clone()
          conv_p.name = f'conv_downsample_{i}'
          conv_p.dtype = self.dtype
          conv_p.fprop_dtype = self.fprop_dtype
          conv_p.bias = True
          conv_p.filter_shape = (4, 4, 4, output_dim, output_dim)
          conv_p.filter_stride = (t_stride, 2, 2)
          conv_downsample_layers.append(conv_p)
      input_dim = output_dim

    for j in range(self.num_res_blocks):
      res_p = self.res_block_tpl.clone()
      res_p.name = f'res_block_{num_blocks}_{j}'
      res_p.input_dim = output_dim
      res_p.output_dim = output_dim
      res_p.dtype = self.dtype
      res_p.fprop_dtype = self.fprop_dtype
      res_p.use_conv_shortcut = False
      res_blocks.append(res_p)

    self.create_children('res_blocks', res_blocks)
    self.create_children('conv_downsample_layers', conv_downsample_layers)

    norm_p = self.norm_tpl.clone()
    norm_p.name = 'norm'
    norm_p.dtype = self.dtype
    norm_p.fprop_dtype = self.fprop_dtype
    norm_p.dim = output_dim
    self.create_child('norm', norm_p)

    self.create_child('activation', self.activation_tpl.clone())
    conv_p = self.conv_tpl.clone()
    conv_p.name = 'conv_last'
    conv_p.dtype = self.dtype
    conv_p.fprop_dtype = self.fprop_dtype
    conv_p.bias = True
    conv_p.filter_shape = (1, 1, 1, output_dim, self.embedding_dim)
    self.create_child('conv_last', conv_p)

  def __call__(self, inputs: JTensor) -> JTensor:
    self._check_input(inputs)
    x = self.conv_first(inputs)
    num_blocks = len(self.channel_multipliers)
    for i in range(num_blocks):
      for j in range(self.num_res_blocks):
        x = self.res_blocks[i * self.num_res_blocks + j](x)
      if i < num_blocks - 1:
        x = self.conv_downsample_layers[i](x)

    for j in range(self.num_res_blocks):
      x = self.res_blocks[num_blocks * self.num_res_blocks + j](x)
    x = self.norm(x)
    x = self.activation(x)
    x = self.conv_last(x)
    return x


class Decoder(base_layer.BaseLayer):
  """Decoder structure with 3D CNNs."""

  filters: int = 128
  embedding_dim: int = 8
  output_dim: int = 3  # RGB
  num_res_blocks: int = 4
  temporal_downsample: Sequence[bool] = (False, True, True)
  channel_multipliers: Sequence[int] = (1, 2, 2, 4)
  cond_norm_tpl: LayerTpl = template_field(enc_dec_3dcnn.CondNormLayer)
  norm_tpl: LayerTpl = template_field(enc_dec_3dcnn.GroupNormSpatial)
  conv_tpl: LayerTpl = template_field(enc_dec_3dcnn.CausalConv)
  res_block_tpl: LayerTpl = template_field(enc_dec_3dcnn.ResBlock)
  activation_tpl: pax_fiddle.Config[activations.BaseActivation] = (
      template_field(activations.Swish)
  )

  def setup(self):
    if self.num_res_blocks < 1:
      raise ValueError('num_res_blocks must be >= 1.')
    input_filters = self.embedding_dim
    output_filters = self.filters * self.channel_multipliers[-1]

    conv_p = self.conv_tpl.clone()
    conv_p.name = 'conv_first'
    conv_p.dtype = self.dtype
    conv_p.fprop_dtype = self.fprop_dtype
    conv_p.bias = True
    conv_p.filter_shape = (3, 3, 3, input_filters, output_filters)
    self.create_child('conv_first', conv_p)
    input_filters = self.filters * self.channel_multipliers[-1]

    res_blocks = []
    conv_upsample_layers = []
    cond_norm_layers = []
    num_blocks = len(self.channel_multipliers)

    for j in range(self.num_res_blocks):
      res_p = self.res_block_tpl.clone()
      res_p.name = f'res_block_0_{j}'
      res_p.input_dim = output_filters
      res_p.output_dim = output_filters
      res_p.dtype = self.dtype
      res_p.fprop_dtype = self.fprop_dtype
      res_p.use_conv_shortcut = False
      res_blocks.append(res_p)

    for i in reversed(range(num_blocks)):
      cond_p = self.cond_norm_tpl.clone()
      cond_p.name = f'cond_norm_{i}'
      cond_p.dtype = self.dtype
      cond_p.fprop_dtype = self.fprop_dtype
      cond_p.emb_dim = self.embedding_dim
      cond_p.dim = output_filters
      cond_norm_layers.append(cond_p)

      output_filters = self.filters * self.channel_multipliers[i]
      res_p = self.res_block_tpl.clone()
      res_p.name = f'res_block_{i}_0'
      res_p.input_dim = input_filters
      res_p.output_dim = output_filters
      res_p.dtype = self.dtype
      res_p.fprop_dtype = self.fprop_dtype
      res_p.use_conv_shortcut = False
      res_blocks.append(res_p)
      for j in range(1, self.num_res_blocks):
        res_p = res_p.clone()
        res_p.name = f'res_block_{i}_{j}'
        res_p.input_dim = output_filters
        res_blocks.append(res_p)
      if i > 0:
        t_stride = 2 if self.temporal_downsample[i - 1] else 1
        conv_p = self.conv_tpl.clone()
        conv_p.name = f'conv_upsample_{i}'
        conv_p.dtype = self.dtype
        conv_p.fprop_dtype = self.fprop_dtype
        conv_p.bias = True
        conv_p.filter_shape = (
            3,
            3,
            3,
            output_filters,
            output_filters * 4 * t_stride,
        )
        conv_upsample_layers.append(conv_p)
      input_filters = output_filters
    self.create_children('cond_norm_layers', cond_norm_layers)
    self.create_children('res_blocks', res_blocks)
    self.create_children('conv_upsample_layers', conv_upsample_layers)

    norm_p = self.norm_tpl.clone()
    norm_p.name = 'norm'
    norm_p.dtype = self.dtype
    norm_p.fprop_dtype = self.fprop_dtype
    norm_p.dim = output_filters
    self.create_child('norm', norm_p)

    self.create_child('activation', self.activation_tpl.clone())
    conv_p = self.conv_tpl.clone()
    conv_p.name = 'conv_last'
    conv_p.dtype = self.dtype
    conv_p.fprop_dtype = self.fprop_dtype
    conv_p.bias = True
    conv_p.filter_shape = (3, 3, 3, output_filters, self.output_dim)
    self.create_child('conv_last', conv_p)

  def __call__(self, inputs: JTensor) -> JTensor:
    cond = inputs
    x = self.conv_first(inputs)
    num_blocks = len(self.channel_multipliers)

    for j in range(self.num_res_blocks):
      x = self.res_blocks[j](x)

    for i in reversed(range(num_blocks)):
      x = self.cond_norm_layers[num_blocks - 1 - i](x, cond)
      output_filters = self.filters * self.channel_multipliers[i]
      for j in range(self.num_res_blocks):
        x = self.res_blocks[(num_blocks - i) * self.num_res_blocks + j](x)
      if i > 0:
        t_stride = 2 if self.temporal_downsample[i - 1] else 1
        x = self.conv_upsample_layers[num_blocks - 1 - i](x)
        x = enc_dec_3dcnn.depth_to_space(x, t_stride, output_filters)
        cond = jax.image.resize(
            cond,
            (*x.shape[:-1], cond.shape[-1]),
            method='nearest',
            antialias=False,
        )
        x = x[:, t_stride - 1 :]
        cond = cond[:, t_stride - 1 :]

    x = self.norm(x)
    x = self.activation(x)
    x = self.conv_last(x)
    return x


class VQVaeModel(base_layer.BaseLayer):
  """VQ VAE base model."""

  encoder_tpl: LayerTpl = template_field(Encoder)
  decoder_tpl: LayerTpl = template_field(Decoder)
  quantizer_tpl: LayerTpl = template_field(quantizer.LookupFreeQuantizer)

  temporal_downsample: Sequence[bool] = (False, True, True)
  channel_multipliers: Sequence[int] = (1, 2, 2, 4)
  embedding_dim: int = 8
  filters: int = 32
  num_res_blocks: int = 4

  def setup(self) -> None:
    encoder_p = self.encoder_tpl.clone()
    encoder_p.name = 'encoder'
    encoder_p.dtype = self.dtype
    encoder_p.fprop_dtype = self.fprop_dtype
    encoder_p.temporal_downsample = self.temporal_downsample
    encoder_p.channel_multipliers = self.channel_multipliers
    encoder_p.embedding_dim = self.embedding_dim
    encoder_p.filters = self.filters
    encoder_p.num_res_blocks = self.num_res_blocks
    self.create_child('encoder', encoder_p)

    decoder_p = self.decoder_tpl.clone()
    decoder_p.name = 'decoder'
    decoder_p.dtype = self.dtype
    decoder_p.fprop_dtype = self.fprop_dtype
    decoder_p.temporal_downsample = self.temporal_downsample
    decoder_p.channel_multipliers = self.channel_multipliers
    decoder_p.embedding_dim = self.embedding_dim
    decoder_p.filters = self.filters
    decoder_p.num_res_blocks = self.num_res_blocks
    self.create_child('decoder', decoder_p)

    quantizer_p = self.quantizer_tpl.clone()
    quantizer_p.name = 'quantizer'
    quantizer_p.dtype = self.dtype
    quantizer_p.fprop_dtype = self.fprop_dtype
    quantizer_p.embedding_dim = self.embedding_dim
    self.create_child('quantizer', quantizer_p)

  def encode(self, inputs: JTensor) -> tuple[JTensor, NestedMap]:
    encoded_feature = self.encoder(inputs)
    quantized, result_dict = self.quantizer(encoded_feature)
    return quantized, result_dict

  def encode_to_indices(self, inputs: JTensor) -> JTensor:
    _, result_dict = self.encode(inputs)
    ids = result_dict['encoding_indices']
    return ids

  def encode_to_latents(self, inputs: JTensor) -> JTensor:
    return self.encoder(inputs)

  def decode(self, inputs: JTensor) -> JTensor:
    return self.decoder(inputs)

  def decode_from_indices(self, ids: JTensor) -> JTensor:
    features = self.quantizer.decode_ids(ids)
    reconstructed_video = self.decode(features)
    return reconstructed_video

  def decode_from_latents(self, latents: JTensor) -> JTensor:
    quantized, _ = self.quantizer(latents)
    outputs = self.decode(quantized)
    return outputs

  def __call__(self, inputs: JTensor) -> tuple[JTensor, NestedMap]:
    quantized, result_dict = self.encode(inputs)
    outputs = self.decoder(quantized)
    return outputs, result_dict
