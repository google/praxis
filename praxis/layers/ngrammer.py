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

"""N-grammer layers from https://arxiv.org/abs/2207.06366."""

from typing import Optional, Tuple, Union
import jax
from jax import numpy as jnp
from praxis import asserts
from praxis import base_layer
from praxis import py_utils
from praxis import pytypes
from praxis.layers import bregman
from praxis.layers import embedding_softmax
from praxis.layers import normalizations
import sympy

NestedMap = py_utils.NestedMap
WeightHParams = base_layer.WeightHParams
JTensor = pytypes.JTensor

BaseHParams = base_layer.BaseLayer.HParams


def get_bigram_ids(ids: JTensor,
                   vocab_size: int,
                   segment_pos: Optional[JTensor] = None,
                   pair_ids: Optional[JTensor] = None) -> JTensor:
  """Generate bi-gram ids from uni-gram ids.

  Args:
    ids: An int32 JTensor of shape [B, L].
    vocab_size: Vocabulary size of `ids`, must be > 0.
    segment_pos: If not None (meaning `ids` is packed, i.e. each example
      containing multiple segments), an int32 tensor of shape [B, L], containing
      the position of each id in `ids` in a segment.
    pair_ids: Optional JTensor that determines which tokens to pair to form
      bi-grams. The shape of this tensor is [B, L] where each position indexes
      into a L + 1 tensor which corresponds to `ids` with a pad suffixed at the
      end, in order to allow forming bi-grams with PAD/0. Note that not
      supplying the `pair_ids` results in consecutive tokens getting paired
      together, which is equivalent to `pair_ids = [L, 0, 1, 2, ..., L - 2]`.

  Returns:
    ngram_ids: An int64 JTensor of shape [B, L].
  """
  assert vocab_size > 0
  batch_size = ids.shape[0]
  # Cast to int64 to avoid overflow, which would affect bucket collision
  # rate and model quality.
  ids = jnp.array(ids, dtype=jnp.int64)  # [batch, time]
  pad = jnp.zeros([batch_size, 1], dtype=ids.dtype)  # [batch, 1]

  # Mechanism: for bigrams, we shift ids by one position along the time
  # dimension, and compute:
  #   bigram_id = original_id + pair_id * vocab_size.
  # If an explicit `pair_ids` is provided, we compute pair_id by explicitly
  # gathering the ids to be paired rather than shifting the sequence.
  ids_0 = jnp.concatenate([ids, pad], 1)  # [batch, time+1]
  if pair_ids is not None:
    assert pair_ids.shape == ids.shape
    # Note that `pair_ids` indexes into `ids_0` in order to pair with 0/pad.
    ids_1 = jnp.take_along_axis(ids_0, pair_ids, axis=1)
    # Pad it to be compatible with `ids_0`. No shifting is necessary.
    ids_1 = jnp.concatenate([ids_1, pad], 1)  # [batch, time+1]
  else:
    ids_1 = jnp.concatenate([pad, ids], 1)  # [batch, 1+time]

  if segment_pos is not None:
    # If input is packed, mask out the parts that cross the segment
    # boundaries.
    mask = jnp.array(jnp.equal(segment_pos, 0), dtype=ids_0.dtype)
    mask = 1 - mask
    mask = jnp.concatenate([mask, pad], 1)
    ids_1 *= mask

  ngram_ids = ids_0 + ids_1 * vocab_size  # Bigram ids.
  ngram_ids = ngram_ids[:, 0:-1]
  return ngram_ids


class VectorQuantization(base_layer.BaseLayer):
  """Implements vector quantization (VQ)/online k-means clustering.

  This layer computes a discrete latent representation of a sequence, in a
  manner similar to https://arxiv.org/abs/1805.11063, where each sequence
  position is assigned a cluster membership. This can be useful in 1) reducing
  the latency of decoding a sequence 2) reducing the vocabulary of a sequence
  which can be used to augment the sequence with n-grams and 3) for computing
  sparse attention over a long sequence as in
  https://transacl.org/ojs/index.php/tacl/article/view/2405. Note that this
  applies multi-head VQ, where each head has a separate set of centroids.

  We use the following capital letters to denote shape parameters:
    B = batch size
    L = length of the input sequence (referred to as S or T elsewhere)
    N = number of attention heads
    H = dimensions of each attention head
    K = number of clusters
  """

  class HParams(BaseHParams):
    """Associated hyper-params for this layer class.

    Attributes:
      num_clusters: Number of clusters, typically around the square root of the
        sequence length.
      num_heads: Number of attention heads.
      decay: The decay with which to update centroids.
      epsilon: Tiny value to guard against divide by 0.
      dim_per_head: The last dimension of the inputs on which to apply Vector
        Quantization.
    """
    num_clusters: int = 0
    num_heads: int = 0
    decay: float = 0.999
    epsilon: float = 1e-6
    dim_per_head: int = 0

  def setup(self) -> None:
    """Constructs an instance which tracks its own set of centroids."""
    p = self.hparams
    assert p.num_clusters
    assert p.dim_per_head

    means = WeightHParams(
        shape=[p.num_heads, p.num_clusters, p.dim_per_head],
        collections=[base_layer.WeightHParamsCollection.REQUIRES_MEAN_SYNC])
    self.create_variable('means', means, trainable=False)

  def __call__(self,
               inputs: JTensor,
               paddings: Optional[JTensor] = None) -> Tuple[JTensor, JTensor]:
    """Computes distances of the given input 'x' to all centroids.

    Args:
      inputs: Input tensor of shape [B, L, N, H] or [B, L, D].
      paddings: If not None, a tensor of shape [B, L]. The padding tensor is
        supplied when we want certain tokens to not affect the centroids.

    Returns:
      dists: "distances" of the given input 'x' to all centroids.
             Shape [B, L, N, K].
      nearest_centroid: The inputs with the input embeddings replaced by the
             centroid embeddings, it has the same shape as the inputs i.e.,
             [B, L, N, H].
    """
    p = self.hparams
    means = self.get_var('means')
    inputs = self._cast_to_fprop_dtype(inputs)
    inputs_shape = inputs.shape
    if len(inputs_shape) == 3:
      inputs = jnp.reshape(
          inputs,
          [inputs_shape[0], inputs_shape[1], p.num_heads, p.dim_per_head])

    if paddings is not None:
      # Shape [B, L, 1, 1]
      paddings_4d = paddings[:, :, jnp.newaxis, jnp.newaxis]

    dists = -2 * jnp.einsum('BLNH, NKH -> BLNK', inputs, means)
    # [B, L, N, 1]
    inputs_norm_sq = jnp.sum(jnp.square(inputs), axis=-1, keepdims=True)
    # [N, K]
    means_norm_sq = jnp.sum(jnp.square(means), axis=-1, keepdims=False)
    # [1, 1, N, K]
    means_norm_sq = means_norm_sq[jnp.newaxis, jnp.newaxis, :, :]
    dists += inputs_norm_sq + means_norm_sq

    # Shape [B, L, N, K], the same as 'dists' above.
    nearest_ids = jnp.argmin(dists, axis=-1)
    nearest_one_hot = jax.nn.one_hot(
        nearest_ids, p.num_clusters, dtype=means.dtype)
    # [B, L, N].
    # Renormalize between [0, 1] and scale to 256.
    nearest_ids /= p.num_clusters
    nearest_ids *= 256
    self.add_summary(
        'k_means/centroid/cluster_ids',
        nearest_ids[:, :, :, jnp.newaxis],
        summary_type=base_layer.SummaryType.IMAGE)

    # Apply paddings.
    if paddings is not None:
      nearest_one_hot *= (1 - paddings_4d)

    # Same shape as the input [B, L, N, H].
    nearest_centroid = jnp.einsum('BLNK, NKH -> BLNH', nearest_one_hot, means)

    means_norm = jnp.linalg.norm(means, ord=2, axis=-1)
    self.add_summary('k_means/centroid/l2_norm_avg', jnp.mean(means_norm))
    self.add_summary('k_means/centroid/l2_norm_min', jnp.min(means_norm))
    self.add_summary('k_means/centroid/l2_norm_max', jnp.max(means_norm))

    if not self.do_eval:
      # To update the centroids (self.vars.means), we apply gradient descent on
      # the mini-batch of input, which yields the following:
      #   new_centroid = centroid + (1 - decay) * (x_mean - centroid)
      # where x_mean is the average over all the input vectors closest to this
      # centroid.

      # Sum away batch and sequence length dimensions to get per cluster count.
      # Shape: [N, K]
      per_cluster_count = jnp.sum(nearest_one_hot, axis=[0, 1])
      self.add_summary('k_means/centroid/avg_cluster_count',
                       jnp.mean(per_cluster_count))

      # Sum of the input per each closest centroid.
      sum_x = jnp.einsum('BLNK, BLNH -> NKH', nearest_one_hot, inputs)

      # Sum up cluster counts across replicas.

      # If per_cluster_count for a cluster is 0, then 'nearest_one_hot' in that
      # cluster's position will always be 0, hence 'sum_x' in that dimension
      # will be 0.
      new_means = sum_x / (
          p.epsilon + jnp.expand_dims(per_cluster_count, axis=-1))
      updated_means = (1.0 - p.decay) * new_means + p.decay * means
      updated_means = jnp.array(updated_means, means.dtype)
      self.update_var('means', updated_means)
    return dists, nearest_centroid


class BregmanCompression(base_layer.BaseLayer):
  """Implements Bregman compression via Bregman PCA.

  This layer computes a continuous latent representation of a sequence, in a
  manner similar to https://arxiv.org/abs/TBA, where each token is mapped to
  its low-dimensional compression coefficients. Note that this
  applies multi-head compression, where each head has a separate set of
  coefficients.

  We use the following capital letters to denote shape parameters:
    B = batch size
    L = length of the input sequence (referred to as S or T elsewhere)
    N = number of attention heads
    H = dimensions of each attention head
    C = dimensions of compression coefficients.
  """

  class HParams(BaseHParams):
    """Associated hyper-params for this layer class.

    Attributes:
      num_heads: Number of attention heads.
      dim_per_head: The dimension per each head of the input.
      num_components: Number of PCA components, which is the same as the
        dimensionality of the compression coefficients.
      activation_type: Type of the activation function.
      negative_slope: Negative slope for leaky ReLU.
      mean_beta: EMA constant for updating the mean.
      coefficients_lr: Learning rate for the coefficients.
      coefficients_beta: EMA constant for the coefficients updates.
      coefficients_steps: Number of steps for solving the coefficients.
      components_lr: Learning rate for the PCA components.
      components_beta: EMA constant for the PCA components updates.
      start_step: Step number to start updating the components.
      end_step: Step number to end updating the components.
      constant_lr_schedule: Whether to use a constant learning rate schedule for
        the components. Applies a linearly decaying schedule if False.
    """
    num_heads: int = 0
    dim_per_head: int = 0
    num_components: int = 0
    activation_type: bregman.ActivationType = bregman.ActivationType.IDENTITY
    negative_slope: float = 0.0
    mean_beta: float = 0.99
    coefficients_lr: float = 0.01
    coefficients_beta: float = 0.9
    coefficients_steps: int = 20
    components_lr: float = 0.01
    components_beta: float = 0.9
    start_step: int = 0
    end_step: int = 0
    constant_lr_schedule: bool = True

  def setup(self) -> None:
    """Constructs an instance which updates PCA components."""
    p = self.hparams
    assert p.dim_per_head

    bregman_layers = []
    for _ in range(p.num_heads):
      bregman_layers.append(
          bregman.BregmanPCA.HParams(
              num_components=p.num_components,
              input_dims=p.dim_per_head,
              activation_type=p.activation_type,
              negative_slope=p.negative_slope,
              mean_beta=p.mean_beta,
              coefficients_lr=p.coefficients_lr,
              coefficients_beta=p.coefficients_beta,
              coefficients_steps=p.coefficients_steps,
              components_lr=p.components_lr,
              components_beta=p.components_beta,
              start_step=p.start_step,
              end_step=p.end_step,
              constant_lr_schedule=p.constant_lr_schedule))
    self.create_children('bregman_layers', bregman_layers)

  def __call__(self,
               inputs: JTensor,
               paddings: Optional[JTensor] = None) -> JTensor:
    """Computes distances of the given input 'x' to all centroids.

    Args:
      inputs: Input tensor of shape [B, L, N, H] or [B, L, D].
      paddings: If not None, a tensor of shape [B, L]. The padding tensor is
        supplied when we want certain tokens to not affect the components.

    Returns:
      coefficients: "compression" coefficients of the inputs.
        Shape [B, L, N, C].
    """
    p = self.hparams
    inputs = self._cast_to_fprop_dtype(inputs)
    inputs_shape = inputs.shape
    # Shape [B * L, N, H].
    inputs = jnp.reshape(
        inputs,
        [inputs_shape[0] * inputs_shape[1], p.num_heads, p.dim_per_head])
    paddings_2d = None
    if paddings is not None:
      # Shape [B * L, 1].
      paddings_2d = jnp.reshape(paddings, [-1, 1])

    coefficients = []
    for i in range(p.num_heads):
      # Shape [B * L, C].
      _, coefficients_i = self.bregman_layers[i](inputs[:, i, :], paddings_2d)
      # Shape [B, L, 1, C].
      coefficients_i = jnp.reshape(
          coefficients_i,
          [inputs_shape[0], inputs_shape[1], 1, p.num_components])
      coefficients.append(coefficients_i)
    # Shape [B, L, N, C].
    coefficients = jnp.concatenate(coefficients, axis=2)

    return coefficients


class Ngrammer(base_layer.BaseLayer):
  """Implements a generic N-grammer layer which looks up latent bi-gram id.

  We use the following capital letters to denote shape parameters:
    B = batch size
    L = length of the input sequence (referred to as S or T elsewhere)
    N = number of attention heads
    H = dimensions of each attention head
    K = number of clusters
    D = total dimension which is H * N
  """

  class HParams(BaseHParams):
    """Associated hyper-params for this layer class.

    Attributes:
      ngram_vocab_size: Size of the ngram vocabulary.
      unigram_vocab_size: Size of the unigram vocabulary.
      ngram_emb_dim: Size of the ngram dimension per head.
      concat_ngrams: If True, then concat ngrams.
      num_heads: Number of attention heads.
      dim_per_head: The dimension per each head of the input.
    """
    ngram_vocab_size: int = 768 * 256
    unigram_vocab_size: int = 0
    ngram_emb_dim: int = 8
    concat_ngrams: bool = True
    num_heads: int = 0
    dim_per_head: int = 0

  def setup(self) -> None:
    """Constructs an instance which looks up ngrams."""
    p = self.hparams

    if p.concat_ngrams:
      # The ngram_emb_dim must be smaller than dim_per_head.
      assert p.ngram_emb_dim <= p.dim_per_head
    else:
      # If not concatenating ngram embeddings, check the dims are compatible.
      assert p.ngram_emb_dim == p.dim_per_head

    # Create a separate layer norm per head for embedding normalization.
    # Create a separate layer norm per head for ngram embedding normalization.
    emb_layer_norm_p = []
    ngram_emb_layer_norm_p = []
    ngram_emb_table_p = []
    for i in range(p.num_heads):
      layer_norm_p = normalizations.LayerNorm.HParams().clone()
      layer_norm_p.dim = p.dim_per_head
      layer_norm_p.name = f'layer_norm_{i}'

      emb_layer_norm_p.append(layer_norm_p)
      ngram_layer_norm_p = normalizations.LayerNorm.HParams().clone()
      ngram_layer_norm_p.dim = p.ngram_emb_dim
      ngram_emb_layer_norm_p.append(ngram_layer_norm_p)

      # Create embedding table for ngram lookup.
      embedding_p = embedding_softmax.Embedding.HParams().clone()
      embedding_p.name = f'embedding_{i}'
      embedding_p.num_classes = p.ngram_vocab_size
      embedding_p.input_dims = p.ngram_emb_dim
      embedding_p.params_init = p.params_init
      # Copy sharding annotations.
      embedding_p.weight_split_dims_mapping = p.weight_split_dims_mapping
      ngram_emb_table_p.append(embedding_p)

    self.create_children('emb_layer_norm', emb_layer_norm_p)
    self.create_children('ngram_layer_norm', ngram_emb_layer_norm_p)
    self.create_children('ngram_table', ngram_emb_table_p)

  def __call__(self,
               input_ids: JTensor,
               input_embs: JTensor,
               paddings: Optional[JTensor] = None,
               segment_pos: Optional[JTensor] = None,
               merge_heads: bool = True,
               pair_ids: Optional[JTensor] = None,
               emb_var: Optional[JTensor] = None) -> JTensor:
    """Augments the input embeddings with VQ n-gram layer embeddings.

    Args:
      input_ids: Input unigram id tensor of shape [B, L] or [B, L, N].
      input_embs: Input unigram embedding tensor of shape [B, L, D] to which to
        add the ngram embedding.
      paddings: If not None, a tensor of shape [B, L] corresponding to padding.
      segment_pos: If not None, a tensor of shape [B, L] corresponding to the
        position of an id in a packed sequence.
      merge_heads: Optional argument determining whether to merge the heads in
        the output sequence.
      pair_ids: Optional JTensor that determines which tokens to pair to form
        bi-grams. The shape of this tensor is [B, N, L] where each position for
        each head indexes into a L + 1 tensor which corresponds to `ids` with a
        pad suffixed at the end, in order to allow forming bi-grams with PAD/0.
        Note that not supplying the `pair_ids` results in consecutive tokens
        getting paired together, which is equivalent to `pair_ids = [L, 0, 1, 2,
        ..., L - 2]`.
      emb_var: Embedding table for calculating cluster centers for eval. This
        is unused and is added here to be consistent with the N-grammer API.

    Returns:
      outputs: Output with the ngram embeddings added of shape [B, L, D] if
        `merge_heads` is True, or [B, L, N, H] if False.
    """
    del emb_var  # Unused.
    p = self.hparams
    if paddings is not None:
      # Shape [B, L, 1, 1]
      paddings_4d = paddings[:, :, jnp.newaxis, jnp.newaxis]

    # Cast input embeddings to fprop dtype.
    input_embs = self._cast_to_fprop_dtype(input_embs)
    inputs_shape = input_ids.shape
    batch_size = inputs_shape[0]
    seq_length = inputs_shape[1]

    # [B, L].
    if len(inputs_shape) == 2:
      input_ids_per_head = [input_ids] * p.num_heads
    else:
      input_ids_per_head = jnp.split(input_ids, p.num_heads, axis=-1)
      input_ids_per_head = [
          jnp.squeeze(ids, axis=-1) for ids in input_ids_per_head
      ]

    # Reshape to [B, L, N, H].
    input_embs = jnp.reshape(input_embs,
                             [batch_size, seq_length, p.num_heads, -1])

    def _multi_way_hash_ids(x, a, b, prime, buckets):
      return ((x * a + b) % prime) % buckets

    ngram_embs_to_concat = []
    vocab_size = p.ngram_vocab_size
    primes = list(
        sympy.primerange(p.ngram_vocab_size + 1,
                         2 * p.ngram_vocab_size))[0:p.num_heads]
    for i in range(p.num_heads):
      pair_ids_per_head = None
      if pair_ids is not None:
        pair_ids_per_head = pair_ids[:, i, :]
      ngram_ids = get_bigram_ids(
          input_ids_per_head[i],
          p.unigram_vocab_size,
          segment_pos,
          pair_ids=pair_ids_per_head)

      ngram_ids_for_head = _multi_way_hash_ids(ngram_ids, i + 1, i + 1,
                                               primes[i], vocab_size)
      ngram_embs_to_concat.append(self.ngram_table[i].emb_lookup(
          jnp.reshape(ngram_ids_for_head, [-1])))
      # [B * L, H]
      ngram_embs_to_concat[i] = self.ngram_layer_norm[i](
          ngram_embs_to_concat[i])

    # [B * L, N * H].
    ngram_embs = jnp.concatenate(ngram_embs_to_concat, 1)
    ngram_embs = jnp.reshape(
        ngram_embs, [batch_size, seq_length, p.num_heads, p.ngram_emb_dim])

    # Layer norm input embeddings independently for each head.
    input_embs_per_head = jnp.split(input_embs, p.num_heads, 2)
    for i in range(p.num_heads):
      # Reshape into [B * L, H]
      per_head_emb = jnp.reshape(input_embs_per_head[i], [-1, p.dim_per_head])
      input_embs_per_head[i] = self.emb_layer_norm[i](per_head_emb)
      # Reshape to [B, L, H]
      input_embs_per_head[i] = jnp.reshape(
          input_embs_per_head[i], [batch_size, seq_length, p.dim_per_head])

    # [B, L, N, H].
    input_embs = jnp.stack(input_embs_per_head, 2)

    if p.concat_ngrams:
      d = p.dim_per_head - p.ngram_emb_dim
      input_embs_slice = jax.lax.dynamic_slice_in_dim(
          input_embs, start_index=0, slice_size=d, axis=-1)
      input_embs = jnp.concatenate([input_embs_slice, ngram_embs], axis=-1)
    else:
      input_embs += ngram_embs

    # Apply paddings back.
    if paddings is not None:
      input_embs *= (1 - paddings_4d)

    # Merge heads.
    if merge_heads:
      # [B, L, D].
      input_embs = jnp.reshape(input_embs, [batch_size, seq_length, -1])

    return input_embs


class VQNgrammer(base_layer.BaseLayer):
  """Implements a VQ based ngrammer layer which looks up latent ngram id.

  We use the following capital letters to denote shape parameters:
    B = batch size
    L = length of the input sequence (referred to as S or T elsewhere)
    N = number of attention heads
    H = dimensions of each attention head
    K = number of clusters
    D = total dimension which is H * N
  """

  class HParams(BaseHParams):
    """Associated hyper-params for this layer class.

    Attributes:
      ngram_vocab_size: Size of the ngram vocabulary.

      ngram_emb_dim: Size of the ngram dimension per head.
      unigram_vocab_size: Size of the unigram vocabulary.
      ngram_using_attention_scores: Whether to compute n-grams using attention
        scores. If True, then consecutive tokens are not used to compute n-grams
        rather it is computed by taking the maximum over the attention scores.
      causal_attention: This argument is only relevant when using attention
        scores to compute n-grams. If this is True, then the causal order is
        respected while taking n-grams, so that a token at position i can only
        form bi-grams with tokens at position < i.
      concat_ngrams: If True, then concat ngrams.
      num_clusters: Number of clusters.
      num_heads: Number of attention heads.
      decay: The decay with which to update centroids.
      epsilon: Tiny value to guard against divide by 0.
      dim_per_head: The last dimension of the inputs on which to apply Vector
        Quantization.
      use_cached_input_ids_to_cluster_ids: Whether to use cached input ids to
        cluster ids.
    """
    ngram_vocab_size: int = 768 * 256
    unigram_vocab_size: int = 0
    ngram_emb_dim: int = 8
    ngram_using_attention_scores: bool = False
    causal_attention: bool = True
    concat_ngrams: bool = False
    num_clusters: int = 0
    num_heads: int = 0
    decay: float = 0.999
    epsilon: float = 1e-6
    dim_per_head: int = 0
    use_cached_input_ids_to_cluster_ids: bool = False

  @classmethod
  def set_canonical_sharding_params(cls, vqngrammer_p, *, replica_axis,
                                    data_axis, mdl_axis, ici_mesh_shape,
                                    dcn_mesh_shape, mesh_axis_names,
                                    training_optimized):
    del replica_axis
    vqngrammer_p.ici_mesh_shape = ici_mesh_shape
    vqngrammer_p.dcn_mesh_shape = dcn_mesh_shape
    vqngrammer_p.mesh_axis_names = mesh_axis_names
    vqngrammer_p.weight_split_dims_mapping.wt = [mdl_axis, data_axis]

    return vqngrammer_p

  def setup(self) -> None:
    """Constructs a VQ layer and an N-grammer layer."""
    p = self.hparams

    if p.concat_ngrams:
      # The ngram_emb_dim must be smaller than dim_per_head.
      assert p.ngram_emb_dim <= p.dim_per_head
    else:
      # If not concatenating ngram embeddings, check the dims are compatible.
      assert p.ngram_emb_dim == p.dim_per_head

    # Create VQ layer.
    vq_layer_p = VectorQuantization.HParams(
        num_clusters=p.num_clusters,
        num_heads=p.num_heads,
        dim_per_head=p.dim_per_head,
        decay=p.decay,
        epsilon=p.epsilon,
        params_init=p.params_init)
    self.create_child('vq_layer', vq_layer_p)

    # Create the input id to cluster id cache.
    if p.unigram_vocab_size:
      input_id_to_cluster_id_cache = WeightHParams(
          shape=[p.unigram_vocab_size, p.num_heads], dtype=jnp.int32,
          init=base_layer.WeightInit.Constant(0))
      self.create_variable('input_id_to_cluster_id_cache',
                           input_id_to_cluster_id_cache,
                           trainable=False)

    # Create N-gram lookup layer.
    ngram_layer_p = Ngrammer.HParams(
        ngram_vocab_size=p.ngram_vocab_size,
        unigram_vocab_size=p.num_clusters,
        ngram_emb_dim=p.ngram_emb_dim,
        concat_ngrams=p.concat_ngrams,
        num_heads=p.num_heads,
        dim_per_head=p.dim_per_head,
        params_init=p.params_init,
        weight_split_dims_mapping=p.weight_split_dims_mapping)
    self.create_child('ngram_layer', ngram_layer_p)

  def __call__(self,
               input_ids: JTensor,
               input_embs: JTensor,
               paddings: Optional[JTensor] = None,
               segment_pos: Optional[JTensor] = None,
               merge_heads: bool = True,
               attention_scores: Optional[JTensor] = None,
               emb_var: Optional[JTensor] = None) -> JTensor:
    """Augments the input embeddings with VQ ngram layer embeddings.

    Args:
      input_ids: Input unigram id tensor of shape [B, L] or [B, L, N]. This is
        unused and is added here to be consistent with the Ngrammger API.
      input_embs: Input unigram embedding tensor of shape [B, L, D] or [B, L, N,
        H] to which to add the ngram embedding.
      paddings: If not None, a tensor of shape [B, L] corresponding to padding.
      segment_pos: If not None, a tensor of shape [B, L] corresponding to the
        position of an id in a packed sequence.
      merge_heads: Optional argument determining whether to merge the heads in
        the output sequence.
      attention_scores: Optional argument representing the attention matrix of
        shape [B, N, L, L] used to construct n-grams if the argument
        `ngrammer_using_attention_scores` is set.
      emb_var: Embedding table for calculating cluster centers for eval. This
        is unused and is added here to be consistent with the N-grammer API.

    Returns:
      outputs: Input embedding with the VQ ngram added of shape [B, L, D] if
        `merge_heads` is True, and shape [B, L, N, H] otherwise.
    """
    del emb_var  # Unused.
    p = self.hparams
    pair_ids = None
    if p.use_cached_input_ids_to_cluster_ids:
      assert self.hparams.unigram_vocab_size > 0
      if not self.hparams.unigram_vocab_size:
        raise ValueError('unigram_vocab_size must be set if using VQ NGrammer'
                         'with use_cached_input_ids_to_cluster_ids = True.')
      if input_ids is None:
        raise ValueError('input_ids must be provided if using VQ NGrammer with'
                         'use_cached_input_ids_to_cluster_ids = True.')
      cache = self.get_var('input_id_to_cluster_id_cache')
      cluster_ids_list = []
      for i in range(p.num_heads):
        cluster_ids_list.append(cache[:, i][(input_ids,)])
      cluster_ids = jnp.stack(cluster_ids_list, axis=-1)
    else:
      # Check if `ngram_using_attention_scores` is set to True, then attention
      # scores is not None.
      if self.hparams.ngram_using_attention_scores:
        if attention_scores is None:
          raise ValueError('If ngram_using_attention_scores is set, then'
                           'attention_scores must be provided.')
        # Compute the pair ids for each token in the sequence by taking the
        # argmax at each position of the attention score.
        if self.hparams.causal_attention:
          seq_len = attention_scores.shape[2]
          pair_ids = jnp.zeros(attention_scores.shape[:-1], dtype=jnp.int32)
          for i in range(seq_len):
            if i > 0:
              # Take the token with the highest attention value less than i.
              pair_ids_at_i = jnp.argmax(attention_scores[:, :, i, :i], axis=-1)
              pair_ids = pair_ids.at[:, :, i].set(pair_ids_at_i)
            else:
              # The first token pairs with PAD due to causal attention.
              pair_ids = pair_ids.at[:, :, i].set(seq_len)
        else:
          # Take the argmax as there is no constraint on causal order.
          pair_ids = jnp.argmax(attention_scores, axis=-1)

      # Cast input embeddings to fprop dtype.
      input_embs = self._cast_to_fprop_dtype(input_embs)

      # Distances of shape [B, L, N, K].
      distances, _ = self.vq_layer(input_embs, paddings=paddings)

      # [B, L, N].
      cluster_ids = jnp.argmin(distances, -1)

      # Cache the cluster ids for future use.
      if not self.do_eval and p.unigram_vocab_size and input_ids is not None:
        cache = self.get_var('input_id_to_cluster_id_cache')
        input_ids_flat = jnp.reshape(input_ids, [-1])
        cluster_ids_flat = jnp.reshape(cluster_ids, [-1, p.num_heads])
        for i in range(p.num_heads):
          cache = cache.at[input_ids_flat, i].set(cluster_ids_flat[:, i])
        self.update_var('input_id_to_cluster_id_cache', cache)

    # [B, L, D] or [B, L, N, H].
    output_embs = self.ngram_layer(
        cluster_ids,
        input_embs,
        paddings,
        segment_pos,
        merge_heads=merge_heads,
        pair_ids=pair_ids)
    return output_embs

  def extend_step(self,
                  input_embs: JTensor,
                  step: Union[int, JTensor],
                  merge_heads: Optional[bool] = True,
                  attention_score: Optional[JTensor] = None) -> JTensor:
    """Augments the input embeddings with VQ ngram layer embeddings at a step.

    Args:
      input_embs: Input unigram embedding tensor of shape [B, L, D] or [B, L, N,
        H] to which to add the ngram embedding.
      step: Time step for which to compute the VQ ngram embedding.
      merge_heads: Optional argument determining whether to merge the heads in
        the output sequence.
      attention_score: Optional argument representing the attention matrix of
        shape [B, N, L] used to construct n-grams if the argument
        `ngrammer_using_attention_scores` is set. Note that this corresponds to
        the attention score at the particular step.

    Returns:
      outputs: Input embedding with the VQ ngram added of shape [B, D] or
      [B, N, H] corresponding to output of NGrammer at the time step.
    """
    # Cast input embeddings to fprop dtype.
    input_embs = self._cast_to_fprop_dtype(input_embs)

    # Check if `ngram_using_attention_scores` is set to True, then attention
    # score is not None.
    pair_ids = None
    if self.hparams.ngram_using_attention_scores:
      if attention_score is None:
        raise ValueError('If ngram_using_attention_scores is set, then'
                         'attention_score must be provided.')
      if not self.hparams.causal_attention:
        raise ValueError('Extend step for NGrammer must have causal attention')
      seq_len = attention_score.shape[2]
      if step > 0:
        pair_ids = jnp.argmax(
            attention_score[:, :, :step], axis=-1, keepdims=True)
        # Tile pair ids to same shape as `input_emb` to be compatible.
        pair_ids = jnp.tile(pair_ids, [1, 1, seq_len])
      else:
        pair_ids = seq_len * jnp.ones_like(attention_score, dtype=jnp.int32)

    # Distances of shape [B, L, N, K].
    distances, _ = self.vq_layer(input_embs)

    # [B, L, N].
    cluster_ids = jnp.argmin(distances, -1)

    # [B, L, D] or [B, L, N, H].
    output_embs = self.ngram_layer(
        cluster_ids, input_embs, merge_heads=merge_heads, pair_ids=pair_ids)

    # Get output at step of shape [B, D] or [B, N, H].
    output_embs = jax.lax.dynamic_slice_in_dim(
        output_embs, slice_size=1, start_index=step, axis=1)
    return jnp.squeeze(output_embs, axis=1)


class BregmanNgrammer(base_layer.BaseLayer):
  """Implements a Bregman PCA based ngrammer layer to form bi-grams.

  We use the following capital letters to denote shape parameters:
    B = batch size
    L = length of the input sequence (referred to as S or T elsewhere)
    N = number of attention heads
    H = dimensions of each attention head
    C = dimensions of compression coefficients.
    V = n-gram vocab size.
  """

  class HParams(BaseHParams):
    """Associated hyper-params for this layer class.

    Attributes:
      ngram_vocab_size: Size of the ngram vocabulary.
      ngram_emb_dim: Size of the ngram dimension per head.
      concat_ngrams: If True, then concat ngrams.
      num_heads: Number of attention heads.
      dim_per_head: The dimension per each head of the input.
      num_components: Number of PCA components, which is the same as the
        dimensionality of the compression coefficients.
      activation_type: Type of the activation function.
      negative_slope: Negative slope for leaky ReLU.
      mean_beta: EMA constant for updating the mean.
      coefficients_lr: Learning rate for the coefficients.
      coefficients_beta: EMA constant for the coefficients updates.
      coefficients_steps: Number of steps for solving the coefficients.
      components_lr: Learning rate for the PCA components.
      components_beta: EMA constant for the PCA components updates.
      start_step: Step number to start updating the components.
      end_step: Step number to end updating the components.
      constant_lr_schedule: Whether to use a constant learning rate schedule for
        the components. Applies a linearly decaying schedule if False.
    """
    ngram_vocab_size: int = 768 * 256
    ngram_emb_dim: int = 8
    concat_ngrams: bool = True
    num_heads: int = 0
    dim_per_head: int = 0
    num_components: int = 0
    activation_type: bregman.ActivationType = bregman.ActivationType.IDENTITY
    negative_slope: float = 0.0
    mean_beta: float = 0.99
    coefficients_lr: float = 0.01
    coefficients_beta: float = 0.9
    coefficients_steps: int = 20
    components_lr: float = 0.01
    components_beta: float = 0.9
    start_step: int = 0
    end_step: int = 0
    constant_lr_schedule: bool = True

  def setup(self) -> None:
    """Constructs an instance which looks up ngrams."""
    p = self.hparams

    asserts.gt(p.dim_per_head, 0)
    asserts.gt(p.num_heads, 0)
    asserts.gt(p.num_components, 0)
    asserts.le(p.num_components, p.dim_per_head)
    if p.concat_ngrams:
      # The ngram_emb_dim must be smaller than dim_per_head.
      asserts.le(p.ngram_emb_dim, p.dim_per_head)
    else:
      # If not concatenating ngram embeddings, check the dims are compatible.
      asserts.eq(p.ngram_emb_dim, p.dim_per_head)

    # Create a separate layer norm per head for embedding normalization.
    # Create a separate layer norm per head for ngram embedding normalization.
    emb_layer_norm_p = []
    ngram_emb_layer_norm_p = []
    for i in range(p.num_heads):
      layer_norm_p = normalizations.LayerNorm.HParams(
          dim=p.dim_per_head, name=f'emb_layer_norm_{i}')
      emb_layer_norm_p.append(layer_norm_p)

      ngram_layer_norm_p = normalizations.LayerNorm.HParams(
          dim=p.ngram_emb_dim, name=f'ngram_layer_norm_{i}')
      ngram_emb_layer_norm_p.append(ngram_layer_norm_p)

    self.create_children('emb_layer_norm', emb_layer_norm_p)
    self.create_children('ngram_layer_norm', ngram_emb_layer_norm_p)

    # Create correlation tensors and embedding tables for n-gram lookup.
    # [C, C, V, N]
    correlation_p = WeightHParams(
        shape=[
            p.num_components, p.num_components, p.ngram_vocab_size, p.num_heads
        ],
        init=p.params_init,
        tensor_split_dims_mapping=p.weight_split_dims_mapping,
        collections=[base_layer.WeightHParamsCollection.REQUIRES_MEAN_SYNC])
    # [V, H, N]
    embedding_table_p = WeightHParams(
        shape=[p.ngram_vocab_size, p.ngram_emb_dim, p.num_heads],
        init=p.params_init,
        tensor_split_dims_mapping=p.weight_split_dims_mapping,
        collections=[base_layer.WeightHParamsCollection.REQUIRES_MEAN_SYNC])

    self.create_variable('correlation', correlation_p)
    self.create_variable('embedding_table', embedding_table_p)

    # Create a Bregman compression layer.
    bregman_compression_layer_p = BregmanCompression.HParams(
        params_init=p.params_init,
        num_heads=p.num_heads,
        dim_per_head=p.dim_per_head,
        num_components=p.num_components,
        activation_type=p.activation_type,
        negative_slope=p.negative_slope,
        mean_beta=p.mean_beta,
        coefficients_lr=p.coefficients_lr,
        coefficients_beta=p.coefficients_beta,
        coefficients_steps=p.coefficients_steps,
        components_lr=p.components_lr,
        components_beta=p.components_beta,
        start_step=p.start_step,
        end_step=p.end_step,
        constant_lr_schedule=p.constant_lr_schedule)

    self.create_child('bregman_compression_layer', bregman_compression_layer_p)

  def __call__(self,
               input_ids: JTensor,
               input_embs: JTensor,
               paddings: Optional[JTensor] = None,
               merge_heads: bool = True,
               **kwargs) -> JTensor:
    """Augments the input embeddings with Bregman n-gram layer embeddings.

    Args:
      input_ids: Input unigram id tensor of shape [B, L] or [B, L, N]. This is
        unused and is added here to be consistent with the Ngrammger API.
      input_embs: Input unigram embedding tensor of shape [B, L, D] to which to
        add the ngram embedding.
      paddings: If not None, a tensor of shape [B, L] corresponding to padding.
      merge_heads: Optional argument determining whether to merge the heads in
        the output sequence.
      **kwargs: Unused key-word args.

    Returns:
      outputs: Output with the ngram embeddings added of shape [B, L, D] if
        `merge_heads` is True, or [B, L, N, H] if False.
    """
    del input_ids

    p = self.hparams
    if paddings is not None:
      # Shape [B, L, 1, 1].
      paddings_4d = paddings[:, :, jnp.newaxis, jnp.newaxis]

    # Cast input embeddings to fprop dtype.
    input_embs = self._cast_to_fprop_dtype(input_embs)
    inputs_shape = input_embs.shape
    batch_size = inputs_shape[0]
    seq_length = inputs_shape[1]

    # Reshape to [B, L, N, H].
    input_embs = jnp.reshape(input_embs,
                             [batch_size, seq_length, p.num_heads, -1])

    # This step can be more efficient using a lookup during inference. During
    # training, we can calculate the compression coefficeints and update the
    # coefficients table.
    input_coeffs = self.bregman_compression_layer(input_embs, paddings)

    correlations = jnp.split(self.theta.correlation, p.num_heads, axis=-1)
    correlations = [
        jnp.squeeze(correlation, axis=-1) for correlation in correlations
    ]
    embedding_tables = jnp.split(
        self.theta.embedding_table, p.num_heads, axis=-1)
    embedding_tables = [
        jnp.squeeze(embedding_table, axis=-1)
        for embedding_table in embedding_tables
    ]

    token_id = jnp.tile(jnp.arange(seq_length)[jnp.newaxis, :], [batch_size, 1])
    token_pad = jnp.zeros([batch_size, 1], dtype=token_id.dtype)  # [batch, 1]
    token_id = jnp.concatenate([token_pad, token_id], axis=1)
    # [B, L].
    curr_token_id = token_id[:, 1:]
    prev_token_id = token_id[:, :-1]
    ngram_embs_to_concat = []
    for i in range(p.num_heads):
      # [B, L, C].
      curr_coeffs_i = jnp.take_along_axis(
          input_coeffs[:, :, i, :], curr_token_id[:, :, jnp.newaxis], axis=1)
      prev_coeffs_i = jnp.take_along_axis(
          input_coeffs[:, :, i, :], prev_token_id[:, :, jnp.newaxis], axis=1)

      # [B, L, C, V].
      inner_prod = jnp.einsum('BLC, CKV -> BLKV', prev_coeffs_i,
                              correlations[i])
      # [B, L, V].
      ngram_corrs_i = jnp.einsum('BLC, BLCV -> BLV', curr_coeffs_i, inner_prod)

      # [B, L, H].
      ngram_embs_i = jnp.einsum('BLV, VH -> BLH', ngram_corrs_i,
                                embedding_tables[i])
      # [B * L, H]
      ngram_embs_i = jnp.reshape(ngram_embs_i, [-1, p.ngram_emb_dim])
      ngram_embs_to_concat.append(ngram_embs_i)

      ngram_embs_to_concat[i] = self.ngram_layer_norm[i](
          ngram_embs_to_concat[i])

    # [B * L, N * H].
    ngram_embs = jnp.concatenate(ngram_embs_to_concat, axis=-1)
    # [B, L, N, H]
    ngram_embs = jnp.reshape(
        ngram_embs, [batch_size, seq_length, p.num_heads, p.ngram_emb_dim])

    # Layer norm input embeddings independently for each head.
    input_embs_per_head = jnp.split(input_embs, p.num_heads, 2)
    for i in range(p.num_heads):
      # Reshape into [B * L, H]
      per_head_emb = jnp.reshape(input_embs_per_head[i], [-1, p.dim_per_head])
      input_embs_per_head[i] = self.emb_layer_norm[i](per_head_emb)
      # Reshape to [B, L, H]
      input_embs_per_head[i] = jnp.reshape(
          input_embs_per_head[i], [batch_size, seq_length, p.dim_per_head])

    # [B, L, N, H].
    input_embs = jnp.stack(input_embs_per_head, 2)

    if p.concat_ngrams:
      d = p.dim_per_head - p.ngram_emb_dim
      input_embs_slice = jax.lax.dynamic_slice_in_dim(
          input_embs, start_index=0, slice_size=d, axis=-1)
      input_embs = jnp.concatenate([input_embs_slice, ngram_embs], axis=-1)
    else:
      input_embs += ngram_embs

    # Apply paddings back.
    if paddings is not None:
      input_embs *= (1 - paddings_4d)

    # Merge heads.
    if merge_heads:
      # [B, L, D].
      input_embs = jnp.reshape(input_embs, [batch_size, seq_length, -1])

    return input_embs
