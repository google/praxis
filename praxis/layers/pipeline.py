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

"""GSPMD pipeline parallelism implementations."""

import functools
from typing import Callable, List, Optional

from flax import core as flax_core
from flax import linen as nn
import jax
from jax import numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from praxis import base_layer
from praxis import flax_utils
from praxis import py_utils
from praxis import pytypes
from praxis.layers import checkpoint_policy

NestedMap = py_utils.NestedMap
JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor

SplitDimsMapping = pytypes.SplitDimsMapping
BaseHParams = base_layer.BaseLayer.HParams
BaseWtShardingHParams = base_layer.BaseLayer.WeightShardingHParams

PARAMS = base_layer.PARAMS
AUX_LOSS = base_layer.AUX_LOSS
SUMMARIES = base_layer.SUMMARIES
NON_TRAINABLE = base_layer.NON_TRAINABLE
RANDOM = base_layer.RANDOM
AutodiffCheckpointType = checkpoint_policy.AutodiffCheckpointType


# Ported from LayerwiseShardablePipelinedLayer in gshard_layers.py.
class LayerwiseShardablePipelined(base_layer.BaseLayer):
  """A layer that implements pipelining across stages.

  It creates a loop over microbatches around a loop-body layer. The wrapped body
  layer represents a single stage, which will be added a leading num_stages
  dimension with vmap() in the input/output data and weights.

  It can run on a single core, or sharded using GSPMD annotations. If the stage
  dimension is sharded, GSPMD will produce a cross-core pipelining pattern.

  Inputs to LayerwiseShardablePipelined should have a leading
  num_microbatch dimension. Each microbatch will be send to each pipeline loop
  iteration.

  The high-level idea is to use a shifting buffer to communicate between stages,
  as shown below (although the real implementation uses recurrent.Recurrent() to
  manage accumulation buffers)::

      # shape: [num_microbatches, ...]
      input = ...
      # Insert a num_stages dimension after num_microbatches, then pad to shape:
      #   [num_microbatches + num_stages - 1, num_stages, ...]
      padded_input = pad(expand_dims(input, 1), ...)

      # Shifting buffer
      state = jnp.zeros([num_stages, ...])

      # Recurrent loop
      for i in range(num_microbatches + num_stages - 1):
        # shift state to the right by one stage
        shifted_state = jnp.pad(state, [[1, 0], ...])[:-1]
        in_mask = jnp.equal(jnp.arange(num_stages), 0)
        stages_in = jnp.where(in_mask, padded_input[i],  shifted_state)
        state = vmap(single_stage_body.fprop)(stages_in)
  """

  class HParams(BaseHParams):
    """Associated hyperparams for this layer class.

    Attributes:
      num_stages: Number of pipeline stages. Each variable in the wrapped layer
        will be prepended with a dimension of size `num_stages`.
      single_stage_body: Single Stage body. A leading num_stages dimension will
        be added automatically by the pipeline layer.
      num_microbatches: If not None, the input is not yet microbatched, and will
        be reshaped to [num_microbatches, microbatch_size] here. Either
        num_microbatches or microbatch_size must be set for input microbatching.
      microbatch_size: If not None, the input is not yet microbatched, and will
        be reshaped to [num_microbatches, microbatch_size] here. Either
        num_microbatches or microbatch_size must be set for input microbatching.
      unpack_summaries: If true, unpack summaries to the individual values from
        each stage.
      stream_io: If true, inputs will be initially sharded along microbatches
        across stages, but each iteration new data will be shifted towards the
        first stage. Outputs will also be stored in the same shifting buffer.
        This can help hide the input-transfer latency if 1) the collective
        permute can be implemented as asynchronous send/receive, and 2) the
        inputs are already sharded before/after the pipeline across the cores
        used as stages. It requires the number of microbatches to be divisible
        by the number of stages. This option is particularly important for
        pipelining with DCN connections where an initial blocking transfer of
        all inputs would be slow.
      polluting_bubbles_with_nan: If True, inputs to bubble iterations will be
        filled with NaNs instead of zeros. This is for testing purpose, and the
        value should not affect correctness.
      pipeline_broadcast_inputs: If true, broadcast inputs (shared between
        all stages instead of being computed by the previous stage) will be
        passed stage-by-stage instead of being replicated.
      checkpoint_policy: How to checkpoint residuals for BProp.
    """
    num_stages: int = 1
    single_stage_body: Optional[BaseHParams] = base_layer.sub_config_field(None)
    num_microbatches: Optional[int] = None
    microbatch_size: Optional[int] = None
    unpack_summaries: bool = True
    stream_io: bool = False
    polluting_bubbles_with_nan: bool = False
    pipeline_broadcast_inputs: bool = False
    checkpoint_policy: AutodiffCheckpointType = AutodiffCheckpointType.SAVE_ITERATION_INPUT

  class WeightShardingHParams(BaseWtShardingHParams):
    """Represents how layer's learned parameters are partitioned across a mesh.

    Attributes:
      stages: How the num_stages dimension should be sharded. This must be a
        list/tuple of one element.
    """
    stages: SplitDimsMapping = (None,)

  def setup(self) -> None:
    """Constructs a LayerwiseShardablePipelined object."""
    p = self.hparams
    assert p.single_stage_body
    self.create_child('body', p.single_stage_body)

  def _shard_dim_by_stages(self, x: JTensor, dim: int) -> JTensor:
    p = self.hparams
    unconstrained_dims = list(range(0, dim)) + list(range(dim + 1, x.ndim))
    dims_mapping = [None] * x.ndim
    dims_mapping[dim] = p.weight_split_dims_mapping.stages[0]
    return base_layer.maybe_shard(
        x,
        dims_mapping,
        p.mesh_axis_names,
        unconstrained_dims=unconstrained_dims)

  def _vmap_gather(self, xs, ids, ids_dim):
    """Use vmap to implement a stage-wise sharded gather.

    The stages share the same input, but they have different offsets.

    Args:
      xs: Data shared by all stages, to be gathered from.
      ids: Integer tensor of shape [num_stages], the offsets of the stages.
      ids_dim: The dimension in xs where ids are applied. In the output, this
        dimension will be [num_stages], since each stage gets one slice.

    Returns:
      The per-stage gathered values. The shape is xs.shape but with ids_dim size
        replaced with [num_stages].
    """
    def _gather_one(x, i):
      return jnp.squeeze(
          jax.lax.dynamic_slice_in_dim(x, i, 1, ids_dim), ids_dim)

    ids = self._shard_dim_by_stages(ids, 0)
    outs = jax.vmap(_gather_one, in_axes=(None, 0), out_axes=ids_dim)(xs, ids)
    return self._shard_dim_by_stages(outs, ids_dim)

  def _vmap_parallel_gather(self, xs, ids, ids_dim, xs_dim):
    """Use vmap to implement a sharded parallel gather.

    Parallel gather means each stage has its own xs, and gets one slice from it.

    Args:
      xs: Per-stage data to be gathered from.
      ids: Integer tensor of shape [num_stages], the offsets of the stages.
      ids_dim: The dimension in xs where ids are applied. The output will not
        have this dimension.
      xs_dim: The dimension in xs that represents parallel stages.

    Returns:
      The per-stage gathered values. The shape is xs.shape but with ids_dim
        removed.
    """
    def _gather_one(x, i):
      dim = ids_dim
      if xs_dim < dim:
        dim -= 1
      return jnp.squeeze(jax.lax.dynamic_slice_in_dim(x, i, 1, dim), dim)

    out_dim = xs_dim
    if ids_dim < xs_dim:
      out_dim -= 1

    ids = self._shard_dim_by_stages(ids, 0)
    xs = self._shard_dim_by_stages(xs, xs_dim)
    outs = jax.vmap(_gather_one, in_axes=(xs_dim, 0), out_axes=out_dim)(xs, ids)
    return self._shard_dim_by_stages(outs, out_dim)

  def _vmap_scatter(self, xs, updates, ids, ids_dim, xs_dim, updates_dim):
    """Use vmap to implement a sharded parallel scatter.

    Parallel scatter means each stage has its own xs and updates.

    Args:
      xs: Per-stage data that updates are to be scattered to.
      updates: Per-stage updates.
      ids: Integer tensor of shape [num_stages], the offsets of the stages.
      ids_dim: The dimension in xs where ids are applied. The updates do not
        have this dimension.
      xs_dim: The dimension in xs that represents parallel stages.
      updates_dim: The dimension in updates that represents parallel stages.

    Returns:
      The per-stage gathered values. The shape is xs.shape.
    """
    def _scatter_one(x, update, i):
      dim = ids_dim
      if xs_dim < dim:
        dim -= 1
      update = jnp.expand_dims(update, dim)
      return jax.lax.dynamic_update_slice_in_dim(x, update, i, dim)

    ids = self._shard_dim_by_stages(ids, 0)
    xs = self._shard_dim_by_stages(xs, xs_dim)
    updates = self._shard_dim_by_stages(updates, updates_dim)
    outs = jax.vmap(
        _scatter_one, in_axes=(xs_dim, updates_dim, 0),
        out_axes=xs_dim)(xs, updates, ids)
    return self._shard_dim_by_stages(outs, xs_dim)

  def _get_body_fprop_fn(self, loop_iteration: JTensor,
                         num_microbatches: int) -> Callable[..., JTensor]:
    """Returns a function that runs the fprop function of the stages."""
    del loop_iteration, num_microbatches

    def _from_nmap(nmap):
      return nmap.ToNestedDict()

    # vmap self.body.fprop to get a leading stage dimension to handle per_stage
    # inputs and args.
    def body_fn(body, is_valid_mb, per_stage_inputs, per_stage_kwargs_nmp,
                *per_stage_args):

      def xform_collections(
          var_tree: pytypes.PyTreeDef, collections: List[str],
          fn: Callable[[JTensor], JTensor]) -> pytypes.PyTreeDef:
        mapped_vars = {}
        for key in var_tree:
          if key in collections:
            mapped_vars[key] = jax.tree_map(fn, var_tree[key])
          else:
            mapped_vars[key] = var_tree[key]
        return mapped_vars

      non_trainable_backup_dict_key = '_pipeline_backup'

      def trans_in(var_tree: pytypes.PyTreeDef) -> pytypes.PyTreeDef:
        # Assume there is no AUX_LOSS and SUMMARIES before function call.
        assert AUX_LOSS not in var_tree
        assert SUMMARIES not in var_tree
        # Use stop_gradient in invalid iterations so that potential NaN
        # gradients will not be propagated to the weights. This jnp.where will
        # be optimized away by XLA, but in the backward pass it will be masking
        # with zeros.
        mapped_vars = xform_collections(
            var_tree, [base_layer.PARAMS],
            lambda x: jnp.where(is_valid_mb, x, jax.lax.stop_gradient(x)))
        if NON_TRAINABLE in var_tree:
          backups = jax.tree_map(lambda x: x, var_tree[NON_TRAINABLE])
          mapped_vars[NON_TRAINABLE][non_trainable_backup_dict_key] = backups
        return mapped_vars

      def trans_out(var_tree: pytypes.PyTreeDef) -> pytypes.PyTreeDef:
        mapped_vars = xform_collections(
            var_tree, [AUX_LOSS, SUMMARIES],
            lambda x: jnp.where(is_valid_mb, x, jnp.zeros_like(x)))
        if NON_TRAINABLE in var_tree:
          non_trainable = mapped_vars[NON_TRAINABLE]
          backups = non_trainable[non_trainable_backup_dict_key]
          del non_trainable[non_trainable_backup_dict_key]
          mapped_vars[NON_TRAINABLE] = jax.tree_map(
              lambda x, y: jnp.where(is_valid_mb, x, y), non_trainable, backups)
        return mapped_vars

      def layer_fprop(layer, *args, **kwargs):
        out = layer(*args, **kwargs)
        return out

      mapped_fn = nn.map_variables(
          layer_fprop,
          mapped_collections=True,  # Transform the entire var col tree.
          mutable=True,
          trans_in_fn=trans_in,
          trans_out_fn=trans_out)

      per_stage_kwargs = _from_nmap(per_stage_kwargs_nmp)
      return mapped_fn(body, per_stage_inputs, *per_stage_args,
                       **per_stage_kwargs)

    # This nn.vmap morally makes N copies of a single layer.
    #
    # `variable_axes` adds a leading stage axis to PARAMS/NON_TRAINABLE,
    # i.e. N copies of layer vars concatenated in the leading dimension.
    #
    # `variable_axes` for AUX_LOSS and SUMMARIES allows us to record them for
    # each layer and potentially aggregated across layers elsewhere.
    #
    # `split_rngs` for RANDOM because dropout mask should be independent for
    # each layer.
    #
    # Note that fprop should not use PARAMS rng because there is no var init.
    vmapped_fn = nn.vmap(
        body_fn,
        in_axes=0,
        out_axes=0,
        variable_axes={
            PARAMS: 0,
            AUX_LOSS: 0,
            SUMMARIES: 0,
            NON_TRAINABLE: 0
        },
        split_rngs={
            PARAMS: self.is_initializing(),
            RANDOM: True
        })

    if self.is_initializing():
      p = self.hparams
      wp = p.weight_split_dims_mapping
      assert wp.stages is not None
      assert isinstance(wp.stages, (list, tuple))
      assert len(wp.stages) == 1
      wp_sub = tuple(wp.stages)

      # nn.map_variables needs to update variable metadata: repeat_prefix and
      # repeat_prefix_split_dims_mapping for trainable and non-trainable vars as
      # we make N copies of a layer. Each layer variable metadata encapsulated
      # in the WeightHParams object in the original single layer needs to be
      # updated to reflect the newly vmapped axis.
      # `repeat_prefix` and `repeat_prefix_split_dims_mapping` are consumed by
      # the optimizer to correctly handle scan-over-layer slot variables.
      for collection in (PARAMS, NON_TRAINABLE):
        vmapped_fn = nn.map_variables(
            vmapped_fn,
            collection,
            mutable=self.is_mutable_collection(collection),
            trans_in_fn=functools.partial(
                flax_utils.remove_axis_to_metadata,
                sub_weight_split_dims_mapping=wp_sub,
                x_times=p.num_stages),
            trans_out_fn=functools.partial(
                flax_utils.add_axis_to_metadata,
                sub_weight_split_dims_mapping=wp_sub,
                x_times=p.num_stages))
    return vmapped_fn

  def num_total_iterations(self, num_microbatches: int) -> int:
    p = self.hparams
    return num_microbatches + p.num_stages - 1

  def num_valid_iterations(self, num_microbatches: int) -> int:
    return num_microbatches

  def get_valid_microbatch_mask(self, loop_iteration: JTensor,
                                num_microbatches: int):
    p = self.hparams
    stage_id = jnp.arange(p.num_stages, dtype=jnp.int32)
    return jnp.logical_and(
        stage_id <= loop_iteration,
        loop_iteration - stage_id < self.num_valid_iterations(num_microbatches))

  def body_fprop(self, loop_iteration: JTensor, num_microbatches: int,
                 per_stage_inputs: JTensor, *per_stage_args,
                 **per_stage_kwargs) -> NestedJTensor:
    p = self.hparams
    per_stage_is_valid_mb = self.get_valid_microbatch_mask(
        loop_iteration, num_microbatches)
    if p.mesh_axis_names is not None:

      def annotate(x):
        return self._shard_dim_by_stages(x, 0)

      per_stage_inputs = jax.tree_map(annotate, per_stage_inputs)
      per_stage_args = jax.tree_map(annotate, per_stage_args)
      per_stage_kwargs = jax.tree_map(annotate, per_stage_kwargs)

    # nn.vmap does not support kwargs, so we use a NestedMap for kwargs.
    def _to_nmap(**kwargs):
      nmap = NestedMap()
      for k, v in kwargs.items():
        nmap.Set(k, v)
      return nmap

    body_fprop_fn = self._get_body_fprop_fn(loop_iteration, num_microbatches)
    # per_mb_vars: per-microbatch vars that need to be adjusted.
    return body_fprop_fn(self.body, per_stage_is_valid_mb, per_stage_inputs,
                         _to_nmap(**per_stage_kwargs), *per_stage_args)

  def _get_init_loop_state(self, microbatched_inputs: NestedJTensor,
                           num_microbatches: int) -> NestedJTensor:
    p = self.hparams
    L = p.num_stages  # pylint: disable=invalid-name
    # The shifting loop state has shape [num_stages, ...]
    # Inputs are not the loop state: they are not changed during the loop. The
    # state (shifting buffer) does not have a num_microbatches dimension.
    shift = jax.tree_map(lambda x: jnp.zeros((L,) + x.shape[1:], dtype=x.dtype),
                         microbatched_inputs)
    state = py_utils.NestedMap(shift=shift)
    if p.stream_io:
      # The streaming buffer has shape [L, num_microbatches // L, ...]. In each
      # iteration, one slice of it (at the num_microbatches // L dim) will be
      # used and shifted.
      def _to_stream(x):
        reshaped = jnp.reshape(x, (L, num_microbatches // L) + x.shape[1:])
        return self._shard_dim_by_stages(reshaped, 0)

      state.stream = jax.tree_map(_to_stream, microbatched_inputs)

    return state

  def _should_checkpoint_stages_in(self) -> bool:
    # We use a slice of the stream buffer at a time, so we checkpoint the slice
    # to avoid saving full buffers in each iteration.
    return self.hparams.stream_io

  def _get_iteration_inputs(self, loop_iteration: JTensor,
                            num_microbatches: int,
                            per_stage_inputs: Optional[NestedJTensor],
                            loop_state: NestedJTensor) -> NestedJTensor:
    p = self.hparams
    if p.stream_io:
      stream_buf_idx = loop_iteration % (num_microbatches // p.num_stages)
      stream_slice = jax.tree_map(lambda x: x[:, stream_buf_idx],
                                  loop_state.stream)
      inputs = stream_slice
    else:
      inputs = jax.tree_map(lambda x: x[loop_iteration % num_microbatches],
                            per_stage_inputs)
    return inputs

  def _get_new_loop_state(self, loop_iteration: JTensor, num_microbatches: int,
                          body_outputs: NestedJTensor,
                          old_state: NestedJTensor) -> NestedJTensor:
    p = self.hparams
    L = p.num_stages  # pylint: disable=invalid-name

    # Shift state to the right by 1.
    def _shift_right(x):
      padding = [[1, 0]] + [[0, 0]] * (x.ndim - 1)
      # Use lax.slice to guarantee the gradient is a pad.
      return jax.lax.slice(jnp.pad(x, padding), [0] * x.ndim, x.shape)

    shifted_out_state = jax.tree_map(_shift_right, body_outputs)
    new_state = NestedMap(shift=shifted_out_state)

    if p.stream_io:
      stream_buf_idx = loop_iteration % (num_microbatches // L)
      stream_slice = jax.tree_map(lambda x: x[:, stream_buf_idx],
                                  old_state.stream)

      def _updated_stream(x, sslice, out):
        # Shift the current slice to the left, then fill the last stage with
        # the final output.
        padding = [[0, 1]] + [[0, 0]] * (sslice.ndim - 1)
        sslice = jax.lax.slice_in_dim(
            jnp.pad(sslice, padding), 1, sslice.shape[0] + 1, axis=0)
        sslice = jnp.where(
            jax.lax.broadcasted_iota('int32', sslice.shape, 0) == L - 1, out,
            sslice)
        sslice = jnp.expand_dims(sslice, 1)
        return jax.lax.dynamic_update_slice_in_dim(
            x, sslice, stream_buf_idx, axis=1)

      new_state.stream = jax.tree_map(_updated_stream, old_state.stream,
                                      stream_slice, body_outputs)

    return new_state

  def _unpack_summary(self, key: str, vectorized_summary: JTensor):
    """Unpack vectorized summaries to per-stage ones."""
    p = self.hparams
    assert p.num_stages == vectorized_summary.shape[0]
    per_stage = {}
    for i in range(p.num_stages):
      per_stage[f'{key}.stage{i}'] = vectorized_summary[i]
    return per_stage

  def __call__(self, inputs: NestedJTensor, *broadcast_inputs,
               **broadcast_kwargs) -> NestedJTensor:
    """FProp inputs through the pipeline body.

    self.body.fprop is expected to be of the following signature:
    outputs = self.body(theta, inputs,
                              *broadcast_inputs, **broadcast_kwargs)

    outputs are expected to be of the same structure as inputs.

    Args:
      inputs: Inputs to body_fprop, same structure as outputs.
      *broadcast_inputs: Broadcasted args to body_fprop.
      **broadcast_kwargs: Broadcasted kwargs to body_fprop

    Returns:
      Output from the last pipeline stage.
    """
    p = self.hparams
    L = p.num_stages  # pylint: disable=invalid-name

    # If inputs are already microbatched, the leading dimension is the number of
    # microbatches.
    needs_microbatching = False
    flat_inputs = jax.tree_util.tree_leaves(inputs)
    assert flat_inputs
    num_microbatches = flat_inputs[0].shape[0]

    # If not, users must only specify either num_microbatches or microbatch_size
    # but not both.
    if p.num_microbatches is not None:
      assert p.microbatch_size is None
      num_microbatches = p.num_microbatches
      needs_microbatching = True

    if p.microbatch_size is not None:
      assert p.num_microbatches is None
      batch_size = flat_inputs[0].shape[0]
      assert batch_size % p.microbatch_size == 0, (batch_size,
                                                   p.microbatch_size)
      num_microbatches = batch_size // p.microbatch_size
      needs_microbatching = True

    # Reshape inputs into [num_microbatches, microbatch_size, ...] if needed.
    if needs_microbatching:

      def _to_microbatches(x):
        batch = x.shape[0]
        assert batch % num_microbatches == 0
        # For streaming, the outer-most dim is number of stages/streams, so that
        # existing sharding on it can be preserved.
        streams = L if p.stream_io else 1
        assert num_microbatches % streams == 0
        # Within each stream, we put num_microbatches in the inner dimension
        # then transpose it. This allows the sharding on the batch (if any) to
        # be propagated to the microbatch dimension because otherwise XLA SPMD
        # propagates sharding to the major dimension (num_microbatches) when we
        # split batch to num_microbatches and microbatch_sizes. We cannot shard
        # the num_microbatches dimension since it's indexed by the loop
        # iteration.
        reshaped = x.reshape((streams, batch // num_microbatches,
                              num_microbatches // streams) + x.shape[1:])
        streams = reshaped.transpose([0, 2, 1] +
                                     list(range(3, len(reshaped.shape))))
        return jnp.reshape(streams, (num_microbatches,) + streams.shape[2:])

      inputs = jax.tree_map(_to_microbatches, inputs)
      broadcast_inputs = jax.tree_map(_to_microbatches, broadcast_inputs)
      broadcast_kwargs = jax.tree_map(_to_microbatches, broadcast_kwargs)

    if p.pipeline_broadcast_inputs:
      inputs = py_utils.NestedMap(
          inputs=inputs,
          broadcast_inputs=broadcast_inputs,
          broadcast_kwargs=broadcast_kwargs)
      broadcast_inputs = []
      broadcast_kwargs = {}

    total_iterations = self.num_total_iterations(num_microbatches)

    if p.stream_io:
      per_stage_inputs = None

      # Broadcast inputs need to be replicated across stages.
      def _replicate_broadcast_inputs(x):
        return base_layer.maybe_shard(
            x, [None] * x.ndim,
            p.mesh_axis_names,
            unconstrained_dims=list(range(1, x.ndim)))

      broadcast_inputs = jax.tree_map(_replicate_broadcast_inputs,
                                      broadcast_inputs)
      broadcast_kwargs = jax.tree_map(_replicate_broadcast_inputs,
                                      broadcast_kwargs)
    else:
      # Create a L dimension, but use pad instead of broadcast to avoid gradient
      # accumulation in the backward pass (only the first stage produces
      # non-zero gradients.)
      def _add_l_dim(x):
        return jnp.pad(
            jnp.expand_dims(x, 1),
            [[0, 0], [0, L - 1]] + [[0, 0]] * (x.ndim - 1))

      per_stage_inputs = jax.tree_map(_add_l_dim, inputs)

    loop_state0 = self._get_init_loop_state(inputs, num_microbatches)

    def _scan_fn(model, carry):
      in_state = carry.data.shift
      loop_iter = carry.loop_iter

      # Different stages need args from different microbatches.
      microbatch_ids = jnp.maximum(loop_iter - jnp.arange(L), 0)
      microbatch_ids = microbatch_ids % num_microbatches

      # Bring in the next microbatch.
      def _select_state_or_input(x, s):
        return jnp.where(
            jax.lax.broadcasted_iota('int32', s.shape, 0) == 0, x, s)

      stages_in = self._get_iteration_inputs(loop_iter, num_microbatches,
                                             per_stage_inputs, carry.data)
      if p.polluting_bubbles_with_nan:
        is_valid_mb = self.get_valid_microbatch_mask(loop_iter,
                                                     num_microbatches)

        def _fill_nan_for_bubbles(x):
          if not jnp.issubdtype(x.dtype, jnp.floating):
            return x
          mask = jnp.reshape(is_valid_mb, (L,) + (1,) * (x.ndim - 1))
          return jnp.where(mask, x, jnp.zeros_like(x) + jnp.nan)

        stages_in = jax.tree_map(_fill_nan_for_bubbles, stages_in)

      if self._should_checkpoint_stages_in():
        stages_in = jax.tree_map(
            lambda x: checkpoint_name(x, 'iteration_input'), stages_in)
      stages_in = jax.tree_map(_select_state_or_input, stages_in, in_state)

      if p.pipeline_broadcast_inputs:
        per_stage_args = stages_in.broadcast_inputs
        per_stage_kwargs = stages_in.broadcast_kwargs
        stages_in = stages_in.inputs
      else:
        per_stage_args = jax.tree_map(
            functools.partial(self._vmap_gather, ids=microbatch_ids, ids_dim=0),
            broadcast_inputs)
        per_stage_kwargs = jax.tree_map(
            functools.partial(self._vmap_gather, ids=microbatch_ids, ids_dim=0),
            broadcast_kwargs)
      # Run through pipeline body.
      out_state = model.body_fprop(loop_iter, num_microbatches, stages_in,
                                   *per_stage_args, **per_stage_kwargs)
      y_out = out_state
      py_utils.assert_same_shape_and_dtype(stages_in, out_state)
      if p.pipeline_broadcast_inputs:
        out_state = py_utils.NestedMap(
            inputs=out_state,
            broadcast_inputs=per_stage_args,
            broadcast_kwargs=per_stage_kwargs,
        )
      new_carry_data = self._get_new_loop_state(loop_iter, num_microbatches,
                                                out_state, carry.data)

      # Accumulator saves out_state for final output retrieval.
      ys = NestedMap(data=y_out if not p.stream_io else None)
      carry = NestedMap(data=new_carry_data, loop_iter=loop_iter + 1)
      return carry, ys

    # Loop over num_microbatches + (num_stages - 1), where input to each iter
    # has the same shape as the loop state.
    # TODO(zhangqiaorjc): Allow checkpoint policy customization.
    rematted_scan_fn = nn.remat(
        _scan_fn,
        prevent_cse=False,  # prevent_cse not required for scan.
        policy=jax.checkpoint_policies.save_from_both_policies(
            checkpoint_policy.custom_policy(p.checkpoint_policy),
            checkpoint_policy.custom_policy(
                AutodiffCheckpointType.SAVE_ITERATION_INPUT)))

    # This nn.scan morally iterates through microbatches and feed each
    # microbatch to a full pipeline body (N layers, not 1 layer).
    #
    # `variable_axes` for AUX_LOSS and SUMMARIES allows us to record them for
    # each layer and potentially aggregated across layers elsewhere.
    #
    # `variable_carry` for NON_TRAINABLE (if mutable) because we want to
    # accumulate batch stats (moving_mean etc) across layers instead of
    # recording one for each layer.
    #
    # `variable_broadcast` for PARAMS because we already have a full pipeline
    # due to the inner vmap. Note NON_TRAINABLE cannot be broadcast because
    # broadcast var cannot be mutated (needs to be loop invariant).
    # Additionally, if NON_TRAINABLE is immutable, it should also be included in
    # this category because it does not have a corresponding output as required
    # by `variable_carry`.
    #
    # `split_rngs` for RANDOM because dropout mask should be independent for
    # each microbatch of the same batch.
    #
    # Note that fprop should not use PARAMS rng because there is no var init.
    variable_carry = []
    variable_broadcast = [PARAMS]
    if self.is_mutable_collection(NON_TRAINABLE):
      variable_carry.append(NON_TRAINABLE)
    else:
      variable_broadcast.append(NON_TRAINABLE)
    scan_fn = nn.scan(
        rematted_scan_fn,
        variable_axes={
            SUMMARIES: 0,
            AUX_LOSS: 0,
        },
        variable_carry=variable_carry,
        variable_broadcast=variable_broadcast,
        # Dropout keys will be split for each iteration.
        split_rngs={RANDOM: True},
        length=total_iterations,
    )

    def post_process(var_tree: pytypes.PyTreeDef) -> pytypes.PyTreeDef:
      if AUX_LOSS in var_tree:
        # Normalize aux_loss by num_microbatches.
        var_tree[AUX_LOSS] = jax.tree_map(
            lambda x: jnp.sum(x, axis=0) / num_microbatches, var_tree[AUX_LOSS])
        # Q(yonghui): Why do we further aggregate aux-loss across pipeline
        # stages?
        var_tree[AUX_LOSS] = jax.tree_map(lambda x: jnp.sum(x, axis=0),
                                          var_tree[AUX_LOSS])
      if SUMMARIES in var_tree:
        # Normalize summaries by num_microbatches.
        summaries = jax.tree_map(
            lambda x: jnp.sum(x, axis=0) / num_microbatches,
            var_tree[SUMMARIES])
        if p.unpack_summaries:

          def _unpack(vars_dict):
            per_stage = {}
            for k, v in vars_dict.items():
              if not isinstance(v, JTensor):
                per_stage[k] = _unpack(v)
                continue
              summary_suffix = base_layer.get_summary_type_suffix(
                  base_layer.get_summary_type_from_key(k))
              k = base_layer.trim_summary_type_from_key(k)
              unpacked = self._unpack_summary(k, v)
              for uk, uv in unpacked.items():
                per_stage[f'{uk}{summary_suffix}'] = uv
            return per_stage

          var_tree[SUMMARIES] = _unpack(summaries)
        else:
          var_tree[SUMMARIES] = summaries
      return var_tree

    after_post_process = nn.map_variables(
        scan_fn,
        mapped_collections=True,  # Transform the entire var col tree.
        mutable=True,
        trans_out_fn=post_process)

    init_carry = NestedMap(
        data=loop_state0, loop_iter=jnp.array(0, dtype=jnp.int32))

    if self.is_initializing():
      # Variable initializations doesn't require scanning through microbatches.
      # One scan body is enough to trigger variable initializations.
      out, accum = _scan_fn(self, init_carry)
      # Get a single-stage output and broadcast it to the right output shape.
      shift_buf = out.data.shift
      if p.pipeline_broadcast_inputs:
        shift_buf = shift_buf.inputs
      output = jax.tree_map(
          lambda x: jax.lax.broadcast(x[0], [num_microbatches]), shift_buf)
    else:
      # The following is layer.apply codepath.
      out, accum = after_post_process(self, init_carry)

      if not p.stream_io:
        # Extract output from the last stage after num_stages-1 bubbles.
        def _extract_out(x):
          # Use lax.slice to guarantee the gradient is a pad.
          return jnp.squeeze(
              jax.lax.slice(
                  x, [total_iterations - num_microbatches, x.shape[1] - 1] +
                  [0] * (x.ndim - 2), x.shape), 1)

        output = jax.tree_map(_extract_out, accum.data)
      else:
        # Extract output the streaming buffer.
        # The output can be misaligned on the (num_microbatches // L) dim,
        # because the implementation uses an offset based the input (consumed
        # by the first stage), but outputs are produced by the last stage. We
        # rotate the buffer to fix the misalignment.
        first_last_offset = (L - 1) % (num_microbatches // L)

        def _extract_out(x):
          if first_last_offset > 0:
            x = jnp.concatenate(
                [x[:, first_last_offset:], x[:, :first_last_offset]], axis=1)
          return jnp.reshape(x, (num_microbatches,) + x.shape[2:])

        if p.pipeline_broadcast_inputs:
          output = jax.tree_map(_extract_out, out.data.stream.inputs)
        else:
          output = jax.tree_map(_extract_out, out.data.stream)

    if needs_microbatching:

      def _to_batches(x):
        streams = L if p.stream_io else 1
        x = jnp.reshape(x, (streams, num_microbatches // streams) + x.shape[1:])
        x = x.transpose([0, 2, 1] + list(range(3, len(x.shape))))
        return x.reshape((num_microbatches * x.shape[1],) + x.shape[3:])

      output = jax.tree_map(_to_batches, output)

    return output


class CircularLayerwiseShardablePipelined(LayerwiseShardablePipelined):
  """A layer that implements circular pipelining across stages.

  It is based on the same shifting buffer mechanism in the base class
  LayerwiseShardablePipelined, but implements a circular schedule. Each stage
  will have an additional leading dimension of size `circular_repeat`; the total
  number of layers is (circular_repeat * num_stages), and each microbatch will
  go through layers with repeat_index 0 in all stage (0 to num_stages - 1), then
  go back to stage 0 and begin the next repeat_index of all stages. This is used
  to reduce the ratio of pipeline bubbles when the number of microbatches is
  small, but adds more cross-stage transfers.
  """

  class HParams(LayerwiseShardablePipelined.HParams):
    """Associated hyperparams for this layer class in addition to.

    Attributes:
      circular_repeat: Number of round-robin layers in each stage for the
        circular pipeline schedule.
      share_weights: Whether layers in the same stage share the weights. This
        can be useful for token-level autoregressive decoding.
    """
    circular_repeat: int = 1
    share_weights: bool = False

  def _async_circular_transfer(self, num_microbatches: int) -> bool:
    """Whether to delay circular transfers by 1 iteration."""
    p = self.hparams
    if num_microbatches < p.num_stages:
      # TODO(yuanzx): Implement padding on small number of microbatches.
      raise NotImplementedError('num_microbatches must be at least num_stages')
    # If we have at least num_stages + 1 microbatches, we can delay the
    # transfer. This is to overlap it with the computation of the next
    # iteration.
    return p.num_stages < num_microbatches

  def num_total_iterations(self, num_microbatches: int) -> int:
    p = self.hparams
    return num_microbatches * p.circular_repeat + p.num_stages - 1

  def num_valid_iterations(self, num_microbatches: int) -> int:
    return num_microbatches * self.hparams.circular_repeat

  def _get_body_fprop_fn(self, loop_iteration: JTensor,
                         num_microbatches: int) -> Callable[..., JTensor]:
    p = self.hparams
    vmapped_fn = super()._get_body_fprop_fn(loop_iteration, num_microbatches)
    if p.share_weights:
      return vmapped_fn

    if self.is_initializing():
      for collection in (PARAMS, NON_TRAINABLE):
        vmapped_fn = nn.map_variables(
            vmapped_fn,
            collection,
            mutable=self.is_mutable_collection(collection),
            trans_in_fn=functools.partial(
                flax_utils.remove_axis_to_metadata,
                sub_weight_split_dims_mapping=(None,),
                x_times=p.circular_repeat),
            trans_out_fn=functools.partial(
                flax_utils.add_axis_to_metadata,
                sub_weight_split_dims_mapping=(None,),
                x_times=p.circular_repeat))

    backup_vars = self.body.variables
    microbatch_ids = jnp.maximum(loop_iteration - jnp.arange(p.num_stages), 0)
    repeat_ids = microbatch_ids // num_microbatches

    # Gather per-stage layers; they have different repeat_ids.
    def trans_in(var_tree: pytypes.PyTreeDef) -> pytypes.PyTreeDef:
      assert SUMMARIES not in var_tree
      return jax.tree_map(
          functools.partial(
              self._vmap_parallel_gather, ids=repeat_ids, ids_dim=0, xs_dim=1),
          var_tree)

    def trans_out(var_tree: pytypes.PyTreeDef) -> pytypes.PyTreeDef:
      mapped_vars = {}
      for collection, tree in var_tree.items():
        if collection in backup_vars:
          original_vars = flax_core.unfreeze(backup_vars[collection])
        else:
          original_vars = jax.tree_map(
              lambda x: jnp.zeros((p.circular_repeat,) + x.shape, x.dtype),
              tree)
        mapped_vars[collection] = jax.tree_map(
            functools.partial(
                self._vmap_scatter,
                ids=repeat_ids,
                ids_dim=0,
                xs_dim=1,
                updates_dim=0), original_vars, tree)
      return mapped_vars

    vmapped_fn = nn.map_variables(
        vmapped_fn,
        mapped_collections=[PARAMS, NON_TRAINABLE, SUMMARIES],
        mutable=True,
        trans_in_fn=trans_in,
        trans_out_fn=trans_out)

    return vmapped_fn

  def _should_checkpoint_stages_in(self) -> bool:
    return True

  def _get_init_loop_state(self, microbatched_inputs: NestedJTensor,
                           num_microbatches: int) -> NestedJTensor:
    p = self.hparams
    state = super()._get_init_loop_state(microbatched_inputs, num_microbatches)
    if self._async_circular_transfer(num_microbatches):
      state.last_iter_result = state.shift
      state.circular_inputs = jax.tree_map(
          lambda x: jnp.zeros((p.num_stages,) + x.shape, x.dtype),
          microbatched_inputs)
    return state

  def _get_iteration_inputs(self, loop_iteration: JTensor,
                            num_microbatches: int,
                            per_stage_inputs: Optional[NestedJTensor],
                            loop_state: NestedJTensor) -> NestedJTensor:
    inputs = super()._get_iteration_inputs(loop_iteration, num_microbatches,
                                           per_stage_inputs, loop_state)
    if self._async_circular_transfer(num_microbatches):
      # After all the microbatches finish repeat_index 0, the first stage will
      # use data received from earlier iterations from the last stage
      # (circular_inputs).
      circular_slice = jax.tree_map(
          lambda x: x[:, loop_iteration % num_microbatches],
          loop_state.circular_inputs)
    else:
      # shift is a circular buffer in this case.
      circular_slice = loop_state.shift
    return jax.tree_map(
        lambda x, c: jnp.where(loop_iteration < num_microbatches, x, c), inputs,
        circular_slice)

  def _get_new_loop_state(self, loop_iteration: JTensor, num_microbatches: int,
                          body_outputs: NestedJTensor,
                          old_state: NestedJTensor) -> NestedJTensor:
    p = self.hparams
    L = p.num_stages  # pylint: disable=invalid-name
    new_state = super()._get_new_loop_state(loop_iteration, num_microbatches,
                                            body_outputs, old_state)

    # Rotate state to the right by 1.
    def _rotate_right(x):
      # Use lax.slice to avoid generating a gather.
      last = jax.lax.slice_in_dim(x, L - 1, L, axis=0)
      except_last = jax.lax.slice_in_dim(x, 0, L - 1, axis=0)
      return jnp.concatenate([last, except_last], axis=0)

    if self._async_circular_transfer(num_microbatches):
      # Transfer data from the last stage to the first. We delay the transfer by
      # one iteration, which means the current iteration's output will be saved
      # in last_iter_result, and in the next iteration, the data in
      # last_iter_result will be transferred using a rotate pattern. We delay
      # the transfer to make it easy to be overlapped with the computation in
      # the iteration.
      def _rotate_right_and_update(x, inp_buf):
        rotated = _rotate_right(x)
        rotated = jnp.expand_dims(rotated, 1)
        # The offset is the last stage's last microbatch ID.
        offset = (loop_iteration - (L - 1) - 1) % num_microbatches
        return jax.lax.dynamic_update_slice_in_dim(
            inp_buf, rotated, offset, axis=1)

      new_state.circular_inputs = jax.tree_map(_rotate_right_and_update,
                                               old_state.last_iter_result,
                                               old_state.circular_inputs)
      new_state.last_iter_result = body_outputs
    else:
      new_state.shift = jax.tree_map(_rotate_right, body_outputs)
    return new_state

  def _unpack_summary(self, key: str, vectorized_summary: JTensor):
    """Unpack vectorized summaries to per-stage ones."""
    p = self.hparams
    if p.share_weights:
      # Average over layers sharing the weights.
      return super()._unpack_summary(key,
                                     vectorized_summary / p.circular_repeat)
    per_layer = {}
    assert vectorized_summary.shape[0] == p.circular_repeat
    assert vectorized_summary.shape[1] == p.num_stages
    for i in range(p.circular_repeat):
      for j in range(p.num_stages):
        per_layer[f'{key}.circular_layer{i}.stage{j}'] = vectorized_summary[i][
            j]
    return per_layer
