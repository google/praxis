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

"""Embedding and softmax layers."""

import math
from typing import Optional, Union

import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import py_utils
from praxis import pytypes
from praxis.layers import activations
from praxis.layers import linears

NestedMap = py_utils.NestedMap
WeightHParams = base_layer.WeightHParams

JTensor = pytypes.JTensor

SplitDimsMapping = pytypes.SplitDimsMapping
BaseHParams = base_layer.BaseLayer.HParams
BaseWtShardingHParams = base_layer.BaseLayer.WeightShardingHParams
BaseActShardingHParams = base_layer.BaseLayer.ActivationShardingHParams

sub_config_field = base_layer.sub_config_field


def _compute_z_loss(logits):
  """Returns a z_loss regularization which stablize logits."""
  # Applies stop_gradient to max_logit instead of logits.
  max_logit = jax.lax.stop_gradient(jnp.max(logits, axis=-1, keepdims=True))
  exp_x = jnp.exp(logits - max_logit)
  sum_exp_x = jnp.sum(exp_x, axis=-1, keepdims=True)
  log_z = jnp.log(sum_exp_x) + max_logit
  return jnp.square(log_z)


class TokenCounter(base_layer.BaseLayer):
  """Keep track of total tokens seen during training."""

  def setup(self) -> None:
    """Creates non-trainable counter."""
    # Currently, uint64 support is explicitly disabled as it causes
    # causes different type promotion behaviors. NB: Using a uint64 type
    # will fall-back to a uint32 with a UserWarning.
    #
    # Using float32 approximation, tracking millions of tokens.
    # Do not allow bfloat16 conversion.
    approx_total_tokens_mm = WeightHParams(
        shape=(),
        init=base_layer.WeightInit.Constant(0.),
        collections=[
            base_layer.WeightHParamsCollection.REQUIRES_SUM_SYNC,
            base_layer.WeightHParamsCollection.DISALLOW_BFLOAT16_CONVERSION
        ])
    self.create_variable(
        'approx_total_tokens_mm', approx_total_tokens_mm, trainable=False)

  def __call__(self, inputs: JTensor, paddings: JTensor) -> JTensor:
    """Track total non-padding tokens.

    Args:
      inputs: Input ids. An int32 JTensor of shape [B, T].
      paddings: A 0/1 JTensor of shape [B, T] with 1 denoting padding.
    """
    if not self.do_eval:
      approx_total_tokens_mm = self.get_var('approx_total_tokens_mm')
      self.add_summary('approx_total_tokens_mm', approx_total_tokens_mm)
      scale = jnp.array(1e6, dtype=jnp.float32)
      batch_total_mm = jnp.sum(1.0 - paddings).astype(jnp.float32) / scale
      # Force f32 addition.
      new_approx_total_tokens_mm = approx_total_tokens_mm.astype(
          jnp.float32) + batch_total_mm.astype(jnp.float32)
      self.update_var('approx_total_tokens_mm', new_approx_total_tokens_mm)


class Embedding(base_layer.BaseLayer):
  """A simple embedding layer that performs embedding lookups from ids."""

  class HParams(BaseHParams):
    """Associated hyper-params for this layer class.

    Attributes:
      num_classes: Number of tokens in the vocabulary.
      input_dims: Depth of the embedding output. This is called `input_dims` as
        opposed to the more appropriate `embedding_dims` to be compatible with
        other softmax/embedding layers defined in this file.
      lookup_style: Style of lookup, one of index or matmul.
      scale_sqrt_depth: If set to True, activations are scaled with
        sqrt(embedding_dim) in emb_lookup.
      set_nan_for_oob_id: If set to True, embeddings corresponding to
        out-of-boundaries ids will be set to NaN. Useful for debugging
        purposes.
    """
    num_classes: int = 0
    input_dims: int = 0
    lookup_style: str = 'index'
    scale_sqrt_depth: bool = False
    set_nan_for_oob_id: bool = False

  class ActivationShardingHParams(BaseActShardingHParams):
    """Represents how intermediate values should be partitioned across a mesh.

    Attributes:
      emb_out_split_dims_mapping: Sharding of the emb output.
    """
    emb_out_split_dims_mapping: SplitDimsMapping = None

  def setup(self) -> None:
    p = self.hparams
    assert p.num_classes > 0
    assert p.input_dims > 0

    wp = p.weight_split_dims_mapping
    self.create_variable(
        'emb_var',
        WeightHParams(
            shape=[p.num_classes, p.input_dims],
            mesh_shape=p.mesh_shape,
            tensor_split_dims_mapping=wp.wt))

  def emb_lookup(self, ids: JTensor) -> JTensor:
    p = self.hparams
    ap = p.activation_split_dims_mapping

    if p.lookup_style == 'index':
      embs = jnp.asarray(self.theta.emb_var)[(ids,)]
    elif p.lookup_style == 'matmul':
      # Explicit casting to fprop_dtype needed for bf16.
      one_hot_ids = jax.nn.one_hot(ids, p.num_classes, dtype=self.fprop_dtype)
      embs = linears.project_last_dim(one_hot_ids, self.theta.emb_var)
    else:
      raise ValueError('Unknown lookup style.')

    # map out-of-boundary ids to nan for easier debug
    if p.set_nan_for_oob_id:
      embs = jnp.where(ids[..., jnp.newaxis] < p.num_classes, embs, jnp.nan)

    if p.scale_sqrt_depth:
      embs *= p.input_dims**0.5

    embs = base_layer.maybe_shard(embs, ap.emb_out_split_dims_mapping,
                                  p.mesh_axis_names)
    return embs


class FullSoftmax(base_layer.BaseLayer):
  """A simple softmax layer with cross-entropy outputs."""

  class HParams(BaseHParams):
    """Associated hyper-params for this layer class.

    Attributes:
      input_dims: Dimension of the input when used as a softmax layer. This is
        also the depth of the output when used as an embedding layer.
      num_classes: Total number of target classes when used as a softmax layer.
        This is also the size of the vocabulary when used as an embedding layer.
      soft_cap_logits: If not None logits are soft capped to this value.
      bi_tempered_loss: If not None applies bi-tempered loss.
      label_smoothing_prob: Label smoothing probability.
      label_smoothing_apply_for_eval: If False, disables label smoothing at eval
        time, even if p.label_smoothing_prob > 0. Label smoothing is a form of
        regularization and we may want to disable it at eval time.
      z_loss_weight: If z_loss_weight is nonzero, we add a loss equal to
        z_loss_weight * square(logsumexp(logits, -1))
      bias_init: Init scale (constant) of bias terms.
      feed_forward_tpl: Sub configurable field for the feed-forward layer.
    """
    input_dims: int = 0
    num_classes: int = 0
    soft_cap_logits: Optional[float] = 0.0
    bi_tempered_loss_tpl: Optional[BaseHParams] = base_layer.sub_config_field(
        None)
    label_smoothing_prob: float = 0.0
    label_smoothing_apply_for_eval: bool = True
    z_loss_weight: float = 0.
    bias_init: Optional[float] = 0.0

    feed_forward_tpl: BaseHParams = sub_config_field(
        linears.FeedForward.HParams)

  def setup(self) -> None:
    p = self.hparams
    wp = p.weight_split_dims_mapping
    ap = p.activation_split_dims_mapping
    ff_p = p.feed_forward_tpl.clone().set(
        input_dims=p.input_dims,
        output_dims=p.num_classes,
        activation_tpl=activations.Identity.HParams(),
        bias_init=p.bias_init,
        weight_split_dims_mapping=wp.clone(),
        activation_split_dims_mapping=ap.clone())
    self.create_child('logits_ffn', ff_p)
    if p.bi_tempered_loss_tpl:
      self.create_child('bi_tempered_loss', p.bi_tempered_loss_tpl)

  def get_logits(self, inputs: JTensor) -> JTensor:
    """Returns logits given the inputs with an option to soft cap it.

    Args:
      inputs: a single JTensor with shape [..., input_dim].

    Returns:
      logits: with shape [..., num_classes]. Unnormalized softmax's logits.
    """
    p = self.hparams
    # Compute logits.
    logits = self.logits_ffn(inputs)

    # Soft cap logits if applicable.
    if p.soft_cap_logits:
      logits = p.soft_cap_logits * jnp.tanh(logits / p.soft_cap_logits)
    return logits

  def logits_to_logp(self, logits: JTensor) -> JTensor:
    """Converts logits to log probability scores."""
    return jax.nn.log_softmax(logits)

  def __call__(self,
               inputs: JTensor,
               class_weights: JTensor,
               class_ids: Optional[JTensor] = None,
               class_probabilities: Optional[JTensor] = None) -> NestedMap:
    """Computes logits, softmax cross entropy etc.

    Args:
      inputs: a single JTensor with shape [..., input_dim].
      class_weights: a JTensor with shape [..., 1] containing the weights for
        each target word.
      class_ids: a JTensor with shape [..., 1] of int32 dtype containing the
        target class labels.
      class_probabilities: a JTensor with shape [..., num_classes] of float
        values indicating class-membership probabilities.

    Returns:
      A `.NestedMap` containing the following fields

      - logits: with shape [..., num_classes]. Unnormalized softmax's logits.
      - per_example_argmax: with shape [...]. argmax of i-th example.
      - per_example_xent: with shape [...]. Cross entropy between i-th example's
        prediction and its label.
      - per_example_weight: with shape [...]. class_weights casted to
        this layer's dtype.
      - total_xent: A scalar. The sum of per_example_weight * per_example_xent.
      - total_weight: A scalar. The sum of per_example_weight.
      - avg_xent: A scalar. total_loss / total_weight.
      - z_loss [optional]: A scalar. The square of logsum logits when
        z_loss_weight > 0.
    """
    p = self.hparams
    # Assert one of class_ids or class_probabilities is not None
    if class_ids is None and class_probabilities is None:
      raise ValueError('One of class_ids or class_probabilities must be given.')

    # Compute logits
    inputs_dtype = inputs.dtype
    logits = self.get_logits(inputs)
    # We perform softmax in float32 to improve stability.
    logits = logits.astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits)

    if class_probabilities is None:
      class_probabilities = jax.nn.one_hot(
          jnp.squeeze(class_ids, axis=-1), p.num_classes, dtype=jnp.float32)
      if p.label_smoothing_prob > 0.0:
        # Label smoothing reduce the probability of the label from 1 to
        # 1 - label_smoothing_prob, and redistribute label_smoothing_prob to the
        # rest of num_classes - 1 classes where each class has a probability of
        # label_smoothing_prob / (num_classes - 1).
        if not self.do_eval or p.label_smoothing_apply_for_eval:
          # We may want to disable label smoothing at eval time.
          other_prob = p.label_smoothing_prob / (p.num_classes - 1)
          class_probabilities = (
              (1.0 - p.label_smoothing_prob) * class_probabilities +
              other_prob * (1.0 - class_probabilities)).astype(jnp.float32)
      class_probabilities = jax.lax.stop_gradient(class_probabilities)

    if p.bi_tempered_loss_tpl is None:
      per_example_xent = -jnp.sum(
          log_probs * class_probabilities, axis=-1, dtype=jnp.float32)
    else:
      per_example_xent = self.bi_tempered_loss(logits, class_probabilities)
    per_example_argmax = jax.lax.stop_gradient(jnp.argmax(logits, axis=-1))

    # Compute total softmax cross-entropy loss for the output tensor.
    total_xent = jnp.sum(
        jnp.expand_dims(per_example_xent, axis=-1) * class_weights,
        dtype=jnp.float32)
    total_weight = jnp.sum(class_weights, dtype=jnp.float32)

    if p.z_loss_weight > 0.0:
      z_loss = jnp.sum(
          _compute_z_loss(logits) * class_weights,
          dtype=jnp.float32) / total_weight
      z_loss *= p.z_loss_weight
      self.add_summary('aux_z_loss', z_loss)
      self.add_aux_loss('aux_z_loss', z_loss)

    output_nmap = NestedMap(
        logits=logits.astype(inputs_dtype),
        log_probs=log_probs.astype(inputs_dtype),
        per_example_argmax=per_example_argmax.astype(inputs_dtype),
        per_example_xent=per_example_xent.astype(jnp.float32),
        total_xent=total_xent,
        total_weight=total_weight,
        avg_xent=(total_xent / (total_weight + 1e-6)).astype(jnp.float32))
    if p.z_loss_weight > 0.0:
      output_nmap['z_loss'] = z_loss
    return output_nmap


class SharedEmbeddingSoftmax(FullSoftmax):
  """A softmax layer that also supports embedding lookups."""

  class HParams(FullSoftmax.HParams):
    """Associated hyper-params for this layer class.

    Attributes:
      lookup_style: Style of lookup, one of index or matmul.
      scale_sqrt_depth: If set True, activations are scaled with
        sqrt(embedding_dim) in emb_lookup.
    """
    lookup_style: str = 'index'
    scale_sqrt_depth: bool = False

  class ActivationShardingHParams(BaseActShardingHParams):
    """Represents how intermediate values should be partitioned across a mesh.

    Attributes:
      emb_out_split_dims_mapping: Sharding of the emb output.
    """
    emb_out_split_dims_mapping: SplitDimsMapping = None

  def emb_lookup(self, ids: JTensor) -> JTensor:
    p = self.hparams
    ap = p.activation_split_dims_mapping
    emb_var = jnp.transpose(self.logits_ffn.linear.theta.w)
    if p.lookup_style == 'index':
      embs = jnp.asarray(emb_var)[(ids,)]
    elif p.lookup_style == 'matmul':
      # Explicit casting to fprop_dtype needed for bf16.
      one_hot_ids = jax.nn.one_hot(ids, p.num_classes, dtype=self.fprop_dtype)
      embs = linears.project_last_dim(one_hot_ids, emb_var)
    else:
      raise ValueError('Unknown lookup style.')
    # Scale with sqrt(embedding dims)
    if p.scale_sqrt_depth:
      embs *= p.input_dims**0.5

    embs = base_layer.maybe_shard(embs, ap.emb_out_split_dims_mapping,
                                  p.mesh_axis_names)
    return embs


class SigmoidCrossEntropy(base_layer.BaseLayer):
  """A sigmoid cross-entropy loss layer with logits projection."""

  class HParams(BaseHParams):
    """Associated hyper-params for this layer class.

    Attributes:
      input_dims: Dimension of the input when used as a softmax layer. This is
        also the depth of the output when used as an embedding layer.
      num_classes: Total number of target classes when used as a softmax layer.
        This is also the size of the vocabulary when used as an embedding layer.
      soft_cap_logits: If not None logits are soft capped to this value.
      bias_init: Init scale (constant) of bias terms.
      feed_forward_tpl: Sub configurable field for the feed-forward layer. If
        None, skip the FFN projection.
    """
    input_dims: int = 0
    num_classes: int = 0
    soft_cap_logits: Optional[float] = 0.0
    bias_init: Optional[float] = 0.0
    feed_forward_tpl: BaseHParams = sub_config_field(
        linears.FeedForward.HParams)

  def setup(self) -> None:
    p = self.hparams
    if p.feed_forward_tpl:
      wp = p.weight_split_dims_mapping
      ap = p.activation_split_dims_mapping
      ff_p = p.feed_forward_tpl.clone().set(
          input_dims=p.input_dims,
          output_dims=p.num_classes,
          activation_tpl=activations.Identity.HParams(),
          bias_init=p.bias_init,
          weight_split_dims_mapping=wp.clone(),
          activation_split_dims_mapping=ap.clone())
      self.create_child('logits_ffn', ff_p)
    else:
      if p.input_dims != p.num_classes:
        raise ValueError(
            f'SigmoidCrossEntropy\'s input_dims {p.input_dims} has to match '
            f'num_classes if ffn is disabled.')

  def get_logits(self, inputs: JTensor) -> JTensor:
    """Returns logits given the inputs with an option to soft cap it.

    Args:
      inputs: a single JTensor with shape [..., input_dim].

    Returns:
      logits: with shape [..., num_classes]. Unnormalized softmax's logits.
    """
    p = self.hparams
    if not p.feed_forward_tpl:
      return inputs
    # Compute logits.
    logits = self.logits_ffn(inputs)

    # Soft cap logits if applicable.
    if p.soft_cap_logits:
      logits = p.soft_cap_logits * jnp.tanh(logits / p.soft_cap_logits)
    return logits

  def __call__(self,
               inputs: JTensor,
               class_weights: JTensor,
               class_ids: Optional[JTensor] = None,
               class_probabilities: Optional[JTensor] = None) -> NestedMap:
    """Computes logits, sigmoid cross entropy etc.

    Args:
      inputs: a single JTensor with shape [..., input_dim].
      class_weights: a JTensor with shape [..., 1] or [..., num_classes]
        containing the weights for each target or for each label class.
      class_ids: a JTensor with shape [..., 1] of int32 dtype containing the
        target class labels.
      class_probabilities: a JTensor with shape [..., num_classes] of float
        values indicating class-membership probabilities.

    Returns:
      A `.NestedMap` containing the following fields

      - logits: with shape [..., num_classes]. Unnormalized softmax's logits.
      - log_probs: with shape [..., num_classes]. Log prob of logits.
      - per_example_argmax: with shape [...]. argmax of i-th example.
      - per_example_xent: with shape [...]. Cross entropy between i-th example's
        prediction and its label multiplied by per_class_weight.
      - total_xent: A scalar. The sum of per_example_weight * per_example_xent.
      - total_weight: A scalar. The sum of per_example_weight.
      - avg_xent: A scalar. total_loss / total_weight.
    """
    p = self.hparams
    # Assert one of class_ids or class_probabilities is not None
    if class_ids is None and class_probabilities is None:
      raise ValueError('One of class_ids or class_probabilities must be given.')

    # Compute logits
    inputs_dtype = inputs.dtype
    logits = self.get_logits(inputs)
    # We perform softmax in float32 to improve stability.
    logits = logits.astype(jnp.float32)
    log_probs = jax.nn.log_sigmoid(logits)

    if class_probabilities is None:
      if p.num_classes == 1:
        raise ValueError(
            'one_hot with num_classes=1 has a strange behavior. Please double '
            'check this is what you intended to do.')
      class_probabilities = jax.nn.one_hot(
          jnp.squeeze(class_ids, axis=-1), p.num_classes, dtype=jnp.float32)
      class_probabilities = jax.lax.stop_gradient(class_probabilities)

    if class_weights.shape[-1] == p.num_classes:
      per_class_weight = class_weights
      per_example_weight = jnp.ones(class_weights.shape[:-1] + (1,))
    elif class_weights.shape[-1] == 1:
      per_class_weight = jnp.ones(class_weights.shape[:-1] + (p.num_classes,))
      per_example_weight = class_weights
    else:
      raise ValueError(
          f'Wrong shape of class_weights {class_weights.shape} vs '
          f'logits shape {logits.shape} vs '
          f'class_probabilities shape {class_probabilities.shape} vs '
          f'num_classes {p.num_classes}')

    # A stable implementation of sigmoid cross entropy loss.
    zeros = jnp.zeros_like(logits)
    cond = logits > zeros
    relu_logits = jnp.where(cond, logits, zeros)
    neg_abs_logits = jnp.where(cond, -logits, logits)
    per_class_xent = (
        relu_logits - logits * class_probabilities +
        jnp.log1p(jnp.exp(neg_abs_logits)))
    per_example_xent = jnp.sum(
        per_class_xent * per_class_weight, axis=-1, dtype=jnp.float32)

    per_example_argmax = jax.lax.stop_gradient(jnp.argmax(logits, axis=-1))

    # Compute total sigmoid cross-entropy loss for the output tensor.
    total_xent = jnp.sum(
        jnp.expand_dims(per_example_xent, axis=-1) * per_example_weight,
        dtype=jnp.float32)

    total_weight = jnp.sum(class_weights, dtype=jnp.float32)
    output_nmap = NestedMap(
        logits=logits.astype(inputs_dtype),
        log_probs=log_probs.astype(inputs_dtype),
        per_example_argmax=per_example_argmax.astype(inputs_dtype),
        per_example_xent=per_example_xent.astype(inputs_dtype),
        total_xent=total_xent.astype(inputs_dtype),
        total_weight=total_weight,
        avg_xent=(total_xent / (total_weight + 1e-6)).astype(inputs_dtype))
    return output_nmap


class GShardSharedEmbeddingSoftmax(base_layer.BaseLayer):
  """Softmax layer with embedding lookup and Gaussian init used in gshard.

  Features:
  1) Weight shape is [V, M] where V is num_classes and M is input_dims.
  2) No bias
  3) Apply 1/sqrt(M) to the input activations before computing the logits.
  4) Optionally using soft clipping and absolute value clipping of logits.
  5) Optional label smoothing.
  """

  class HParams(BaseHParams):
    """Associated hyper-params for this layer class.

    Attributes:
      input_dims: Dimension of the input.
      num_classes: Total number of target classes.
      use_tgt_labels_size_as_loss_denominator: False to use total number of
        non-padding tokens instead of fixed tgt_labels tensor size.
      soft_cap_logits: If not None logits are soft capped to this value before
        the absolute value clipping with p.logits_abs_max.
      logits_abs_max: Absolute logits clipping.
      z_loss_weight: If z_loss_weight is nonzero, we add a loss equal to
        z_loss_weight * square(logsumexp(logits, -1))
      label_smoothing_prob: Optional label smoothing.
    """
    input_dims: int = 0
    num_classes: int = 0
    use_tgt_labels_size_as_loss_denominator: bool = True
    soft_cap_logits: Optional[float] = 0.0
    logits_abs_max: Optional[float] = 0.0
    z_loss_weight: float = 0.
    label_smoothing_prob: float = 0.0

  class ActivationShardingHParams(BaseActShardingHParams):
    """Represents how intermediate values should be partitioned across a mesh.

    Attributes:
      emb_out_split_dims_mapping: Mesh split for embedding outputs..
    """
    emb_out_split_dims_mapping: SplitDimsMapping = None

  def setup(self) -> None:
    p = self.hparams
    wp = p.weight_split_dims_mapping
    ap = p.activation_split_dims_mapping
    emb_p = linears.Linear.HParams(
        input_dims=p.num_classes,
        output_dims=p.input_dims,
        # Same as in gshard_builder.DenseBuilder.Embedding
        params_init=base_layer.WeightInit.Gaussian(),
        weight_split_dims_mapping=wp.clone(),
        activation_split_dims_mapping=ap.clone())
    self.create_child('embedding', emb_p)

  def emb_lookup(self, ids: JTensor) -> JTensor:
    p = self.hparams
    ap = p.activation_split_dims_mapping
    # BL -> BLV
    one_hot_ids = jax.nn.one_hot(ids, p.num_classes, dtype=self.fprop_dtype)
    # BLV,VH -> BLH
    embs = linears.project_last_dim(one_hot_ids, self.embedding.theta.w)
    embs = base_layer.maybe_shard(embs, ap.emb_out_split_dims_mapping,
                                  p.mesh_axis_names)
    return embs

  def get_logits(self, inputs: JTensor) -> JTensor:
    """Returns logits given the inputs with an option to cap it.

    Args:
      inputs: a single JTensor with shape [..., input_dim].

    Returns:
      logits: with shape [..., num_classes]. Unnormalized softmax's logits.
    """
    p = self.hparams
    ap = p.activation_split_dims_mapping
    # activations are scaled with 1/sqrt(input_dims)
    inputs *= (p.input_dims**-0.5)
    # VH -> HV
    softmax_var = jnp.transpose(self.embedding.theta.w)
    # Compute logits:  BLH,HV -> BLV
    logits = linears.project_last_dim(inputs, softmax_var)
    # Adjust sharding annotation during decoding.
    ap_out = ap.out
    if ap_out is not None and len(ap_out) == 3 and logits.ndim == 2:
      ap_out = [ap_out[0], ap_out[2]]
    logits = base_layer.maybe_shard(logits, ap_out, p.mesh_axis_names)

    # Soft cap logits if applicable
    if p.soft_cap_logits:
      logits = p.soft_cap_logits * jnp.tanh(logits / p.soft_cap_logits)

    # abs cap logits if applicable
    if p.logits_abs_max:
      logits = jnp.clip(logits, -p.logits_abs_max, p.logits_abs_max)
    return logits

  def __call__(self,
               inputs: JTensor,
               class_weights: JTensor,
               class_ids: Optional[JTensor] = None,
               class_probabilities: Optional[JTensor] = None) -> NestedMap:
    """Computes logits, cross entropy etc.

    Args:
      inputs: a single JTensor with shape [..., input_dim].
      class_weights: a JTensor with shape [..., 1] containing the weights for
        each target word.
      class_ids: a JTensor with shape [..., 1] of int32 dtype containing the
        target class labels.
      class_probabilities: a JTensor with shape [..., num_classes] of float
        values indicating class-membership probabilities.

    Returns:
      A `.NestedMap` containing the following fields

      - logits: with shape [..., num_classes]. Unnormalized softmax's logits.
      - per_example_argmax: with shape [...]. argmax of i-th example.
      - per_example_xent: with shape [...]. Cross entropy between i-th example's
        prediction and its label.
      - per_example_weight: with shape [...]. class_weights casted to
        this layer's dtype.
      - total_xent: A scalar. The sum of per_example_weight * per_example_xent.
      - total_weight: A scalar. The sum of per_example_weight.
      - avg_xent: A scalar. total_loss / total_weight.
    """
    p = self.hparams
    # Assert one of class_ids or class_probabilities is not None
    if class_ids is None and class_probabilities is None:
      raise ValueError('One of class_ids or class_probabilities must be given.')

    # Compute logits
    inputs_dtype = inputs.dtype
    logits = self.get_logits(inputs)
    # We perform softmax in float32 to improve stability.
    logits = logits.astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits)

    if class_probabilities is None:
      class_probabilities = jax.nn.one_hot(
          jnp.squeeze(class_ids, axis=-1), p.num_classes, dtype=jnp.float32)

    class_probabilities_prior_to_label_smoothing = None
    if p.label_smoothing_prob > 0.0 and not self.do_eval:
      class_probabilities_prior_to_label_smoothing = class_probabilities
      off_value = p.label_smoothing_prob / p.num_classes
      on_value = 1.0 - p.label_smoothing_prob + off_value
      class_probabilities = (on_value * class_probabilities + off_value *
                             (1.0 - class_probabilities)).astype(jnp.float32)
    class_probabilities = jax.lax.stop_gradient(class_probabilities)

    per_example_xent = -jnp.sum(
        log_probs * class_probabilities, axis=-1, dtype=jnp.float32)

    per_example_argmax = jax.lax.stop_gradient(jnp.argmax(logits, axis=-1))

    # Compute total softmax for the entire sequence
    total_xent = jnp.sum(
        jnp.expand_dims(per_example_xent, axis=-1) * class_weights,
        dtype=jnp.float32)

    total_weight = jnp.sum(class_weights, dtype=jnp.float32)

    if p.use_tgt_labels_size_as_loss_denominator:
      loss_denominator = jnp.sum(
          jnp.ones_like(class_weights), dtype=jnp.float32)
    else:
      loss_denominator = total_weight
    avg_xent = (total_xent / loss_denominator).astype(jnp.float32)
    z_loss = (
        jnp.sum(_compute_z_loss(logits) * class_weights) / loss_denominator)
    z_loss *= p.z_loss_weight
    self.add_summary('aux_z_loss', z_loss)
    self.add_aux_loss('aux_z_loss', z_loss)

    if class_probabilities_prior_to_label_smoothing is not None:
      per_example_xent_prior_to_label_smoothing = -jnp.sum(
          log_probs * class_probabilities_prior_to_label_smoothing,
          axis=-1,
          dtype=jnp.float32)
      total_xent_prior_to_label_smoothing = jnp.sum(
          jnp.expand_dims(
              per_example_xent_prior_to_label_smoothing,
              axis=-1) * class_weights, dtype=jnp.float32)
      avg_xent_prior_to_label_smoothing = (
          total_xent_prior_to_label_smoothing /
          loss_denominator).astype(inputs_dtype)
      self.add_summary('avg_xent_prior_to_label_smoothing',
                       avg_xent_prior_to_label_smoothing)

    output_nmap = NestedMap(
        logits=logits.astype(inputs_dtype),
        log_probs=log_probs.astype(inputs_dtype),
        per_example_argmax=per_example_argmax.astype(inputs_dtype),
        per_example_xent=per_example_xent.astype(jnp.float32),
        total_xent=total_xent.astype(inputs_dtype),
        # base_model.py _compute_xent_loss_helper uses avg_xent_weight if set,
        # this helper is currently used by LanguageModel only, if we have
        # EncoderDecoder model we will have to adjust weighting as well.
        avg_xent_weight=loss_denominator,
        avg_xent=avg_xent,
        total_weight=total_weight)

    return output_nmap


class PositionalEmbedding(base_layer.BaseLayer):
  """Generates position embedding for a given 1-d sequence."""

  class HParams(BaseHParams):
    """Associated hyper-params for this layer class.

    Attributes:
      min_timescale: Start of the geometric index. Determines the periodicity of
        the added signal.
      max_timescale: End of the geometric index. Determines the frequency of the
        added signal.
      embedding_dims: Dimension of the embedding to be generated.
    """
    min_timescale: int = 1
    max_timescale: int = 10_000
    embedding_dims: int = 0

  def __call__(self,
               seq_length: Optional[int] = None,
               position: Optional[JTensor] = None) -> JTensor:
    """Generates a JTensor of sinusoids with different frequencies.

    Args:
      seq_length: an optional Python int definiing the output sequence length.
        if the `position` argument is specified.
      position:   [B, seq_length], optional position for each token in the
        sequence, only required when the sequence is packed.

    Returns:
      [B, seqlen, D] if `position` is specified, else [1, seqlen, D]
    """
    p = self.hparams
    if position is None:
      assert seq_length is not None
      # [1, seqlen]
      position = jnp.arange(seq_length, dtype=jnp.float32)[jnp.newaxis, :]
    else:
      assert position.ndim == 2, position.shape

    num_timescales = p.embedding_dims // 2
    log_timescale_increment = (
        math.log(float(p.max_timescale) / float(p.min_timescale)) /
        jnp.maximum(jnp.asarray(num_timescales, dtype=jnp.float32) - 1, 1))
    inv_timescales = p.min_timescale * jnp.exp(
        jnp.arange(num_timescales, dtype=jnp.float32) *
        -log_timescale_increment)
    scaled_time = (
        position[:, :, jnp.newaxis] *
        inv_timescales[jnp.newaxis, jnp.newaxis, :])
    signal = jnp.concatenate(
        [jnp.sin(scaled_time), jnp.cos(scaled_time)],
        axis=2).astype(self.fprop_dtype)
    # Force usage of `np` rather than `jnp` to compute static values at trace
    # time.
    signal = jnp.pad(signal, [[0, 0], [0, 0], [0, np.mod(p.embedding_dims, 2)]])
    return signal


class RotaryPositionalEmbedding(PositionalEmbedding):
  """Applies rotary position embedding for a given 1-d sequence.

  The Rotary position embedding is described in https://arxiv.org/abs/2104.09864
  """

  class HParams(PositionalEmbedding.HParams):
    """Associated hyper-params for this layer class.

    Attributes:
      cast_as_fprop_dtype: If True, the returned vars are cast as fprop_dtype
      to save some memory.
    """
    cast_as_fprop_dtype: bool = True

  def __call__(self,
               inputs: JTensor,
               position: Optional[JTensor] = None) -> JTensor:
    """Generates a JTensor of sinusoids with different frequencies.

    Args:
      inputs: The input sequence on which to apply the Rotary position
        embedding. Since rotary position embeddings are applied to query and
        keys after projection, it is assumed of shape [B, S, N, H].
      position: Optional position JTensor which denotes the position of each
        token in the sequence. This only needs to be supplied when the sequence
        is packed. It is of shape [B, S].

    Returns:
      a JTensor of shape [B, S, N, H] which includes the inputs together with
      the rotary position embedding incorporated in it.
    """
    p = self.hparams
    if len(inputs.shape) != 4:
      raise ValueError('Input is assumed to be a rank 4 tensor of shape'
                       '[batch, sequence, heads, dims].')
    if p.embedding_dims % 2:
      raise ValueError('Embedding dim for rotary position embedding must be a'
                       'multiple of 2.')
    if p.embedding_dims != inputs.shape[3]:
      raise ValueError('The embedding dims of the rotary position embedding'
                       'must match the hidden dimension of the inputs.')
    half_embedding_dim = p.embedding_dims // 2
    fraction = 2 * jnp.arange(0, half_embedding_dim) / p.embedding_dims
    timescale = p.min_timescale * (p.max_timescale / p.min_timescale)**fraction
    if position is None:
      seq_length = inputs.shape[1]
      position = jnp.arange(seq_length, dtype=jnp.float32)[jnp.newaxis, :]
    position = position[:, :, jnp.newaxis, jnp.newaxis]
    timescale = timescale[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
    sinusoid_inp = position / timescale
    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)
    first_half, second_half = jnp.split(inputs, 2, axis=-1)
    first_part = (first_half * cos - second_half * sin)
    second_part = (second_half * cos + first_half * sin)
    # TODO(b/252874053): Clean this up after phase 3 is done.
    if p.cast_as_fprop_dtype:
      first_part = first_part.astype(self.fprop_dtype)
      second_part = second_part.astype(self.fprop_dtype)
    return jnp.concatenate([first_part, second_part], axis=-1)

  def extend_step(self,
                  inputs: JTensor,
                  position: Optional[Union[int, JTensor]] = None) -> JTensor:
    """Generates a JTensor of sinusoids with different frequencies for a step.

    Args:
      inputs: The input sequence on which to apply the Rotary position
        embedding. Since rotary position embeddings are applied to query and
        keys after projection, it is assumed of shape [B, N, H] or of shape [B,
        P, N, H] where P may be a prefix length.
      position: The position which is being decoded, this should correspond to
        the logical position of the last token in the prefix window (P) in the
        entire sequence length S. It is a scalar or having shape [B].

    Returns:
      a JTensor of the same shape as input with the rotary position embedding
      incorporated in it.
    """
    assert len(inputs.shape) in [3, 4]
    inputs_shape = inputs.shape
    if len(inputs_shape) == 3:
      inputs = inputs[:, jnp.newaxis, :, :]
    seq_length = inputs.shape[1]
    # Adjust the prefix's position with position.
    # Note that position may be a tracer rather than an int, and so we must use
    # jax.lax.iota, rather than jnp.arange.
    prefix_position = jax.lax.iota(dtype=jnp.int32, size=seq_length)
    # [B, 1]
    position = jnp.broadcast_to(position, [inputs_shape[0]])[:, jnp.newaxis]
    # [B, P]
    prefix_position = position - jnp.flip(prefix_position)[jnp.newaxis, :]
    prefix_position = jnp.where(prefix_position < 0,
                                jnp.zeros_like(prefix_position),
                                prefix_position)
    output = self(inputs, position=prefix_position)
    if len(inputs_shape) == 3:
      output = jnp.squeeze(output, axis=1)
    return output


class TrainablePositionalEmbedding(PositionalEmbedding):
  """Generates trainable position embedding for a given 1-d sequence."""

  class HParams(PositionalEmbedding.HParams):
    """Associated hyper-params for this layer class.

    Attributes:
      max_seq_length: Max sequence length.
      lookup_style: Style of lookup, one of index or matmul.
    """
    max_seq_length: int = 10_240
    lookup_style: str = 'matmul'

  def setup(self) -> None:
    super().setup()
    p = self.hparams
    wp = p.weight_split_dims_mapping
    self.create_variable(
        'emb_var',
        WeightHParams(
            shape=[p.max_seq_length, p.embedding_dims],
            mesh_shape=p.mesh_shape,
            tensor_split_dims_mapping=wp.wt))

  def __call__(self,
               seq_length: Optional[int] = None,
               position: Optional[JTensor] = None) -> JTensor:
    """Generates a JTensor of embedding lookup result.

    Args:
      seq_length: Sequence length of the embeddings to be generated. This may be
        omitted if an explicit position JTensor is specified.
      position: Optional position JTensor which denotes the position of each
        token in the sequence. This only needs to be supplied when the sequence
        is packed. It is of shape [batch, seq_length].

    Returns:
      a JTensor of shape [batch, seq_length, embedding_dim] if position JTensor
      is specified, else of shape [1, seq_length, embedding_dim].
    """
    p = self.hparams
    if position is None:
      assert seq_length is not None
      position = jnp.arange(seq_length, dtype=jnp.float32)[jnp.newaxis, :]

    pos_emb_var = self.theta.emb_var
    pos_emb_var = jax.lax.slice_in_dim(pos_emb_var, 0, seq_length, axis=0)
    if p.lookup_style == 'index':
      embs = jnp.asarray(pos_emb_var)[(position,)]
    elif p.lookup_style == 'matmul':
      one_hot_ids = jax.nn.one_hot(position, seq_length, dtype=self.fprop_dtype)
      embs = jnp.matmul(one_hot_ids, pos_emb_var)
    else:
      raise ValueError('Unknown lookup style.')

    return embs
