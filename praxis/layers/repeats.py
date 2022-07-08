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

"""Generic repeat layer that stacks a sub-layer multiple times.

This simply passes input through the layer stack.
"""

import enum
import functools
from typing import Any, Callable, Optional

from flax import linen as nn
import jax
from jax import numpy as jnp
from praxis import base_layer
from praxis import flax_utils
from praxis import py_utils
from praxis import pytypes
import tensorflow.compat.v2 as tf

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
DECODE_CACHE = base_layer.DECODE_CACHE
PREFIX_DECODE_CACHE = base_layer.PREFIX_DECODE_CACHE

SCAN_VARIABLE_AXES = {
    PARAMS: 0,
    AUX_LOSS: 0,
    SUMMARIES: 0,
    NON_TRAINABLE: 0,
    DECODE_CACHE: 0,
    PREFIX_DECODE_CACHE: 0
}
# PARAMS is vmapped. Scan does not need to init vars, so PARAMS rng key is not
# needed.
SCAN_SPLIT_RNGS = {PARAMS: False, RANDOM: True}


@enum.unique
class AutodiffCheckpointType(str, enum.Enum):
  """jax.checkpoint policy types."""
  SAVE_NOTHING = 'save_nothing'
  SAVE_EVERYTHING = 'save_everything'
  SAVE_QKV_OUT_PROJ = 'save_qkv_out_proj'
  SAVE_OUT_PROJ = 'save_out_proj'
  SAVE_CONTEXT = 'save_context'
  SAVE_CONTEXT_AND_OUT_PROJ = 'save_encoded_and_out_proj'
  SAVE_DOT_ONLY = 'save_dot_only'
  SAVE_DOT_WITH_NO_BATCH_DIM = 'save_dot_with_no_batch_dims'
  SAVE_DOT_FOR_MLPERF_200B = 'save_dot_for_mlperf_200b'


def _custom_policy(checkpoint_policy: AutodiffCheckpointType):
  """Returns a JAX Autodiff checkpointing policy from the enum value."""
  # TODO(zhangqiaorjc): Configure custom checkpoint policy in expt config
  # without introducing enum.
  if checkpoint_policy == AutodiffCheckpointType.SAVE_EVERYTHING:
    return jax.checkpoint_policies.everything_saveable
  if checkpoint_policy == AutodiffCheckpointType.SAVE_DOT_ONLY:
    return jax.checkpoint_policies.checkpoint_dots
  if checkpoint_policy == AutodiffCheckpointType.SAVE_DOT_WITH_NO_BATCH_DIM:
    return jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
  if checkpoint_policy == AutodiffCheckpointType.SAVE_QKV_OUT_PROJ:
    return jax.checkpoint_policies.save_only_these_names(
        'combined_qkv_proj', 'out_proj')
  if checkpoint_policy == AutodiffCheckpointType.SAVE_CONTEXT:
    return jax.checkpoint_policies.save_only_these_names('context')
  if checkpoint_policy == AutodiffCheckpointType.SAVE_OUT_PROJ:
    return jax.checkpoint_policies.save_only_these_names('out_proj')
  if checkpoint_policy == AutodiffCheckpointType.SAVE_CONTEXT_AND_OUT_PROJ:
    return jax.checkpoint_policies.save_only_these_names('context', 'out_proj')
  if checkpoint_policy == AutodiffCheckpointType.SAVE_DOT_FOR_MLPERF_200B:
    return jax.checkpoint_policies.save_only_these_names(
        'combined_qkv_proj', 'query_proj', 'value_proj', 'key_proj', 'context',
        'out_proj')
  assert checkpoint_policy == AutodiffCheckpointType.SAVE_NOTHING
  return jax.checkpoint_policies.nothing_saveable


def _sum_aux_loss(tree):
  return jax.tree_map(jnp.sum, tree)


class Repeat(base_layer.BaseLayer):
  """A generic repeat layer."""

  class HParams(BaseHParams):
    """Associated hyperparams for this layer class.

    Attributes:
      sub: The parameterization of the sub-layer.
      x_times: The number of times to repeat sub.
      unpack_summaries: If true, unpack summaries to the individual values from
        each loop iterations.
      checkpoint_policy: How to checkpoint residuals for BProp: save nothing,
        dot only or dot with no batch dimensions.
      unroll_in_decode: Whether to unroll the layers during extend_step. The
        scan loop within a decoding loop can cause large overheads for data
        copy/formatting.
    """
    sub: Optional[BaseHParams] = None
    x_times: int = 0
    unpack_summaries: bool = False
    checkpoint_policy: AutodiffCheckpointType = AutodiffCheckpointType.SAVE_NOTHING
    unroll_in_decode: bool = False

  class WeightShardingHParams(BaseWtShardingHParams):
    """Represents how layer's learned parameters are partitioned across a mesh.

    Attributes:
      sub: How the list of subs should be sharded.
    """
    sub: SplitDimsMapping = None

  def setup(self) -> None:
    """Constructor."""
    p = self.hparams
    assert p.x_times > 0
    assert p.sub is not None

    self.create_child('sub', p.sub)

  def force_init(self):
    p = self.hparams

    # nn.vmap variable_axes adds a leading stage axis to variable shapes.
    def fn(model, _):
      model.force_init(None)
      return None, None

    # nn.vmap variable_axes adds a leading stage axis to variable shapes.
    vmapped_fn = nn.vmap(
        fn,
        variable_axes=SCAN_VARIABLE_AXES,
        split_rngs={
            PARAMS: True,
            RANDOM: True
        },
        axis_size=p.x_times)

    wp = p.weight_split_dims_mapping
    if wp.sub is not None:
      assert isinstance(wp.sub, (list, tuple))
      assert len(wp.sub) == 1
      wp_sub = tuple(wp.sub)
    else:
      wp_sub = (-1,)

    mapped_fn = vmapped_fn
    for collection in (PARAMS, NON_TRAINABLE):
      mapped_fn = nn.map_variables(
          mapped_fn,
          collection,
          mutable=self.is_mutable_collection(collection),
          trans_out_fn=functools.partial(
              flax_utils.add_axis_to_metadata,
              sub_weight_split_dims_mapping=wp_sub,
              x_times=p.x_times))

    mapped_fn(self.sub, None)  # scan requires a dummy carry input

  # TODO(zhangqiaorjc): Allow callers to customize. body_fn.
  def __call__(self, inputs: NestedJTensor, *args: Any, **kwargs: Any) -> Any:
    """FProp inputs through the sub layer stack.

    outputs are expected to be of the same structure as inputs. extra can be any
    structure.

    Args:
      inputs: A NestedMap of inputs that goes through the sub layer stack.
      *args: Positional args to be passed to sub.fprop method.
      **kwargs: Keyward args to be passed to sub.fprop method.

    Returns:
      Output from the last sub layer.
    """
    p = self.hparams

    def body_fn(sub, layer_in):
      layer_out = sub(layer_in, *args, **kwargs)
      tf.nest.assert_same_structure(layer_in, layer_out)
      return layer_out, None

    # TODO(zhangqiaorjc): Use remat-scan?
    rematted_body_fn = nn.remat(
        body_fn,
        prevent_cse=False,  # prevent_cse not required for scan.
        policy=_custom_policy(p.checkpoint_policy))

    scan_fn = nn.scan(
        rematted_body_fn,
        variable_axes=SCAN_VARIABLE_AXES,
        split_rngs=SCAN_SPLIT_RNGS,
        length=p.x_times)

    mapped_scan_fn = nn.map_variables(
        scan_fn,
        SUMMARIES,
        mutable=self.is_mutable_collection(SUMMARIES),
        trans_in_fn=functools.partial(
            flax_utils.maybe_repack_summary,
            unpack_summaries=p.unpack_summaries,
            x_times=p.x_times),
        trans_out_fn=functools.partial(
            flax_utils.maybe_unpack_summary,
            unpack_summaries=p.unpack_summaries,
            x_times=p.x_times))

    mapped_scan_fn = nn.map_variables(
        mapped_scan_fn,
        AUX_LOSS,
        mutable=self.is_mutable_collection(AUX_LOSS),
        trans_out_fn=_sum_aux_loss)

    if p.unroll_in_decode:

      def _unstack_cache(tree):
        new_tree = {}
        for collection, subtree in tree.items():
          new_tree[collection] = {}
          for i in range(p.x_times):

            def _slice(x, i=i):
              return x[i]

            new_tree[collection][f'layer{i}'] = jax.tree_map(_slice, subtree)
        return new_tree

      mapped_scan_fn = nn.map_variables(
          mapped_scan_fn, [DECODE_CACHE, PREFIX_DECODE_CACHE],
          mutable=True,
          trans_out_fn=_unstack_cache)

    layer_out, _ = mapped_scan_fn(self.sub, inputs)
    return layer_out

  def init_states(self, *args: Any, **kwargs: Any) -> Any:
    """Inits decoder states for all sub layers.

    sub.init_states() should be of signature

    init_states = sub.init_states(*args, **kwargs)

    Args:
      *args: Positional args to pass to the sub.init_states() method.
      **kwargs: Keyward args to pass to the sub.init_states() method.

    Returns:
      Initial decoder states.
    """
    # TODO(team): Configure for spmd.
    p = self.hparams

    def body_fn(sub, _):
      sub.init_states(*args, **kwargs)
      return None, None

    scan_fn = nn.scan(
        body_fn,
        variable_axes=SCAN_VARIABLE_AXES,
        split_rngs=SCAN_SPLIT_RNGS,
        length=p.x_times)

    mapped_scan_fn = nn.map_variables(
        scan_fn,
        SUMMARIES,
        mutable=self.is_mutable_collection(SUMMARIES),
        trans_in_fn=functools.partial(
            flax_utils.maybe_repack_summary,
            unpack_summaries=p.unpack_summaries,
            x_times=p.x_times),
        trans_out_fn=functools.partial(
            flax_utils.maybe_unpack_summary,
            unpack_summaries=p.unpack_summaries,
            x_times=p.x_times))

    # Calls scan_fn with a None carry_in and ignores the carry_out.
    mapped_scan_fn(self.sub, None)

  def _run_unrolled_for_decoding(self, fn: Callable[[base_layer.BaseLayer, Any],
                                                    Any], inputs: Any) -> Any:
    p = self.hparams

    def _run_one_layer(i, inp):

      def _map_in(tree):
        for collection, subtree in tree.items():
          if collection not in [DECODE_CACHE, PREFIX_DECODE_CACHE]:
            layer_i = jax.tree_map(lambda x: x[i], subtree)
            subtree.clear()
            subtree.update(layer_i)
          else:
            layer_i = subtree[f'layer{i}']
            del subtree[f'layer{i}']
            other_layers = subtree.copy()
            subtree.clear()
            subtree.update(layer_i)
            subtree['_repeats_other_layers'] = other_layers
        return tree

      def _map_out(tree):
        for collection, subtree in tree.items():
          if collection not in [DECODE_CACHE, PREFIX_DECODE_CACHE]:
            continue
          other_layers = subtree['_repeats_other_layers']
          del subtree['_repeats_other_layers']
          layer_i = subtree.copy()
          subtree.clear()
          subtree.update(other_layers)
          subtree[f'layer{i}'] = layer_i
        return tree

      mapped_fn = nn.map_variables(
          fn, [
              DECODE_CACHE, PREFIX_DECODE_CACHE, PARAMS, NON_TRAINABLE,
              SUMMARIES, AUX_LOSS
          ],
          mutable=True,
          trans_in_fn=_map_in,
          trans_out_fn=_map_out)
      mapped_fn = nn.map_variables(
          mapped_fn,
          SUMMARIES,
          mutable=False,
          trans_in_fn=functools.partial(
              flax_utils.maybe_repack_summary,
              unpack_summaries=p.unpack_summaries,
              x_times=p.x_times))
      # TODO(yuanzx): we do not support summaries/aux_losses yet.
      mapped_fn = nn.map_variables(
          mapped_fn, [PARAMS, NON_TRAINABLE, SUMMARIES, AUX_LOSS],
          mutable=False)
      return mapped_fn(self.sub, inp)

    out = inputs
    for i in range(p.x_times):
      out, _ = _run_one_layer(i, out)
    return out

  def extend_step(self, step_inputs: NestedJTensor, *args: Any,
                  **kwargs: Any) -> Any:
    """Extends decoder states by one step.

    Args:
      step_inputs: Input to the bottom decoder layer.
      *args: Additional positional input.
      **kwargs: Additional keyword input.

    Returns:
      new_states, top_decoder_out, where new_states is the updated decoder
      states, and top_decoder_out is the output from the top decoder layer.
    """
    p = self.hparams

    # TODO(zhangqiaorjc): Apply remat?
    def body_fn(sub, layer_in):
      layer_out = sub.extend_step(layer_in, *args, **kwargs)
      tf.nest.assert_same_structure(layer_in, layer_out)
      return layer_out, None

    if p.unroll_in_decode:
      return self._run_unrolled_for_decoding(body_fn, step_inputs)

    # Note that in_axes specification skips `carry` and supports prefix spec.
    scan_fn = nn.scan(
        body_fn,
        in_axes=0,  # scan over axis 0 for layer_states
        variable_axes=SCAN_VARIABLE_AXES,
        split_rngs=SCAN_SPLIT_RNGS,
        length=p.x_times)

    mapped_scan_fn = nn.map_variables(
        scan_fn,
        SUMMARIES,
        mutable=self.is_mutable_collection(SUMMARIES),
        trans_in_fn=functools.partial(
            flax_utils.maybe_repack_summary,
            unpack_summaries=p.unpack_summaries,
            x_times=p.x_times),
        trans_out_fn=functools.partial(
            flax_utils.maybe_unpack_summary,
            unpack_summaries=p.unpack_summaries,
            x_times=p.x_times))

    mapped_scan_fn = nn.map_variables(
        mapped_scan_fn,
        AUX_LOSS,
        mutable=self.is_mutable_collection(AUX_LOSS),
        trans_out_fn=_sum_aux_loss)

    # PREFIX_DECODE_CACHE is kept constant during decoding.
    mapped_scan_fn = nn.map_variables(
        mapped_scan_fn, PREFIX_DECODE_CACHE, mutable=False)

    layer_out, _ = mapped_scan_fn(self.sub, step_inputs)
    return layer_out

  def transform_decode_state(
      self, transform_fn: base_layer.DecodeStateTransformFn) -> None:
    """Transforms all decode state variables based on transform_fn."""
    p = self.hparams

    def body_fn(sub, _):
      sub.transform_decode_state(transform_fn)
      return None, None

    if p.unroll_in_decode:
      self._run_unrolled_for_decoding(body_fn, None)
      return

    scan_fn = nn.scan(
        body_fn,
        variable_axes=SCAN_VARIABLE_AXES,
        split_rngs=SCAN_SPLIT_RNGS,
        length=p.x_times)

    mapped_scan_fn = nn.map_variables(
        scan_fn,
        SUMMARIES,
        mutable=self.is_mutable_collection(SUMMARIES),
        trans_in_fn=functools.partial(
            flax_utils.maybe_repack_summary,
            unpack_summaries=p.unpack_summaries,
            x_times=p.x_times),
        trans_out_fn=functools.partial(
            flax_utils.maybe_unpack_summary,
            unpack_summaries=p.unpack_summaries,
            x_times=p.x_times))

    mapped_scan_fn(self.sub, None)

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
    p = self.hparams

    def body_fn(sub, _):
      sub.lazy_broadcast_prefix(num_suffix_samples, suffix_length)
      return None, None

    if p.unroll_in_decode:
      return self._run_unrolled_for_decoding(body_fn, None)

    scan_fn = nn.scan(
        body_fn,
        variable_axes=SCAN_VARIABLE_AXES,
        split_rngs=SCAN_SPLIT_RNGS,
        length=p.x_times)

    mapped_scan_fn = nn.map_variables(
        scan_fn,
        SUMMARIES,
        mutable=self.is_mutable_collection(SUMMARIES),
        trans_in_fn=functools.partial(
            flax_utils.maybe_repack_summary,
            unpack_summaries=p.unpack_summaries,
            x_times=p.x_times),
        trans_out_fn=functools.partial(
            flax_utils.maybe_unpack_summary,
            unpack_summaries=p.unpack_summaries,
            x_times=p.x_times))

    mapped_scan_fn(self.sub, None)
