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

"""Generic repeat layer that stacks a sub-layer multiple times.

This simply passes input through the layer stack.
"""

import functools
from typing import Any, Callable, Optional, Sequence, Tuple, Union

from flax import linen as nn
from flax.core import meta
import jax
from jax import numpy as jnp
from praxis import asserts
from praxis import base_layer
from praxis import flax_utils
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis.layers import checkpoint_policy

NestedMap = py_utils.NestedMap
JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor

SplitDimsMapping = pytypes.SplitDimsMapping
LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]

PARAMS = base_layer.PARAMS
AUX_LOSS = base_layer.AUX_LOSS
SUMMARIES = base_layer.SUMMARIES
NON_TRAINABLE = base_layer.NON_TRAINABLE
RANDOM = base_layer.RANDOM
DECODE_CACHE = base_layer.DECODE_CACHE
PREFIX_DECODE_CACHE = base_layer.PREFIX_DECODE_CACHE
INTERMEDIATES = base_layer.INTERMEDIATES
AutodiffCheckpointType = checkpoint_policy.AutodiffCheckpointType

SCAN_VARIABLE_AXES = {
    PARAMS: 0,
    AUX_LOSS: 0,
    SUMMARIES: 0,
    NON_TRAINABLE: 0,
    DECODE_CACHE: 0,
    PREFIX_DECODE_CACHE: 0
}


def _sum_aux_loss(tree):
  return jax.tree_map(jnp.sum, tree)


class Repeat(base_layer.BaseLayer):
  """A generic repeat layer.

  Attributes:
    sub_tpl: The parameterization of the sub-layer.
    x_times: The number of times to repeat sub.
    unpack_summaries: If true, unpack summaries to the individual values from
      each loop iterations.
    checkpoint_policy: How to checkpoint residuals for BProp: save nothing, dot
      only or dot with no batch dimensions.
    unroll_in_decode: Whether to unroll the layers during extend_step. The scan
      loop within a decoding loop can cause large overheads for data
      copy/formatting.
    sublayer_name: Name of the sublayer. This affects the checkpoint variable
      paths.
    optimizer_dims_mapping: Tensor split dims mapping used for the optimizer
      state variables corresponding to the repeat prefix dims.
    collect_intermediate_outputs: If True, makes intermediate sublayers' outputs
      available for flax capture_intermediates.
    nd_prefix_shape: If not None, there are multiple prefix dims of this shape
      and np.prod(nd_prefix_shape) == x_times. It enables circular
      pipeline-compatible repeat layer decoding.
    positional_args_as_scan_carry: Passing positional args as scan carry instead
      of broadcast args.
  """
  sub_tpl: Optional[LayerTpl] = base_layer.template_field(None)
  x_times: int = 0
  unpack_summaries: bool = False
  checkpoint_policy: AutodiffCheckpointType = AutodiffCheckpointType.SAVE_NOTHING
  unroll_in_decode: bool = False
  sublayer_name: str = 'sub'
  optimizer_dims_mapping: SplitDimsMapping = None
  collect_intermediate_outputs: bool = False
  nd_prefix_shape: Optional[Sequence[int]] = None
  positional_args_as_scan_carry: bool = False

  class WeightSharding(base_layer.BaseLayer.WeightSharding):
    """Represents how layer's learned parameters are partitioned across a mesh.

    Attributes:
      sub: How the list of subs should be sharded.
    """
    sub: SplitDimsMapping = None

  def setup(self) -> None:
    """Constructor."""
    assert self.x_times > 0
    assert self.sub_tpl is not None
    nd_shape = self.nd_prefix_shape
    if nd_shape is not None:
      assert len(nd_shape) >= 1
      assert functools.reduce(lambda x, y: x * y, nd_shape) == self.x_times

    self.create_child(self.sublayer_name, self.sub_tpl)

  def _wrap_for_nd(
      self, fn: Callable[..., Any], unrolled: bool = False
  ) -> Callable[..., Any]:
    nd_shape = self.nd_prefix_shape
    if nd_shape is None or len(nd_shape) <= 1:
      return fn
    # Do not support prefix sharding.
    assert self.weight_split_dims_mapping.sub is None
    assert self.optimizer_dims_mapping is None

    if not unrolled:
      # Remove leading axis for 1D repeat.
      metadata_params = {
          'is_initializing': self.is_initializing(),
          'sub_weight_split_dims_mapping': (-1,),
          'x_times': self.x_times,
          'optimizer_dims_mapping': None,
      }
      fn = nn.map_variables(
          fn,
          [PARAMS, NON_TRAINABLE],
          trans_in_fn=lambda x: meta.add_axis(x, 0, metadata_params),
          trans_out_fn=lambda x: meta.remove_axis(x, 0, metadata_params),
          mutable=True,
      )

    n = len(nd_shape)

    def merge_dims(tree: pytypes.PyTree) -> pytypes.PyTree:
      return jax.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[n:]), tree)

    def split_dims(tree: pytypes.PyTree) -> pytypes.PyTree:
      return jax.tree_map(
          lambda x: jnp.reshape(x, tuple(nd_shape) + x.shape[1:]), tree
      )

    mapped_fn = nn.map_variables(
        fn,
        [PARAMS, NON_TRAINABLE],
        mutable=True,
        trans_in_fn=merge_dims,
        trans_out_fn=split_dims,
    )

    # Add leading axes for ND repeat.
    for i in range(n):
      mapped_fn = nn.add_metadata_axis(
          mapped_fn,
          variable_axes={PARAMS: 0, NON_TRAINABLE: 0},
          metadata_params={
              'is_initializing': self.is_initializing(),
              'sub_weight_split_dims_mapping': (-1,),
              'x_times': nd_shape[n - i - 1],
              'optimizer_dims_mapping': None,
          },
      )
    return mapped_fn

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

    def body_fn(sub, layer_in):
      if self.positional_args_as_scan_carry:
        layer_out = _ensure_tuple(sub(*layer_in, **kwargs))
      else:
        layer_out = sub(layer_in, *args, **kwargs)

      asserts.assert_same_structure(layer_in, layer_out)
      if self.collect_intermediate_outputs:
        return layer_out, layer_out
      else:
        return layer_out, None

    # TODO(zhangqiaorjc): Use remat-scan?
    rematted_body_fn = nn.remat(
        body_fn,
        prevent_cse=False,  # prevent_cse not required for scan.
        policy=checkpoint_policy.custom_policy(self.checkpoint_policy),
    )

    scan_fn = nn.scan(
        rematted_body_fn,
        variable_axes=SCAN_VARIABLE_AXES,
        split_rngs={PARAMS: self.is_initializing(), RANDOM: True},
        length=self.x_times,
        metadata_params={
            'is_initializing': self.is_initializing(),
            'sub_weight_split_dims_mapping': self.weight_split_dims_mapping.sub,
            'x_times': self.x_times,
            'optimizer_dims_mapping': self.optimizer_dims_mapping,
        },
    )
    scan_fn = self._wrap_for_nd(scan_fn)

    mapped_scan_fn = nn.map_variables(
        scan_fn,
        SUMMARIES,
        mutable=self.is_mutable_collection(SUMMARIES),
        trans_in_fn=functools.partial(
            flax_utils.maybe_repack_summary,
            unpack_summaries=self.unpack_summaries,
            x_times=self.x_times,
        ),
        trans_out_fn=functools.partial(
            flax_utils.maybe_unpack_summary,
            unpack_summaries=self.unpack_summaries,
            x_times=self.x_times,
        ),
    )

    mapped_scan_fn = nn.map_variables(
        mapped_scan_fn,
        AUX_LOSS,
        mutable=self.is_mutable_collection(AUX_LOSS),
        trans_out_fn=_sum_aux_loss)

    if self.unroll_in_decode:
      def _clear_decode_cache(tree):
        del tree
        return {}

      def _unstack_cache(tree):
        new_tree = {}
        for collection, subtree in tree.items():
          new_tree[collection] = {}
          for i in range(self.x_times):
            def _slice(x, i=i):
              return x[i]

            new_tree[collection][f'layer{i}'] = jax.tree_map(_slice, subtree)
        return new_tree

      mapped_scan_fn = nn.map_variables(
          mapped_scan_fn, [DECODE_CACHE, PREFIX_DECODE_CACHE],
          mutable=True,
          trans_in_fn=_clear_decode_cache,
          trans_out_fn=_unstack_cache)

    if self.positional_args_as_scan_carry:
      scan_inputs = (inputs,) + args
    else:
      scan_inputs = inputs
    layer_out, intermediates = mapped_scan_fn(self.sublayer, scan_inputs)
    if self.collect_intermediate_outputs:
      self.sow(INTERMEDIATES, 'repeat_intermediates', intermediates)
    return layer_out

  def quantize_weight(self) -> NestedJTensor:
    """Quantize the current layer and its children layer(s).

    Returns:
      a nested map from names to quantized weights.
    """
    return self._quantize_fn(return_pspec=False)

  def quantized_partition_specs(self) -> Any:
    """Get quantization spec for the current layer and its children layer(s).

    Returns:
      a nested map from names to partition spec.
    """
    return self._quantize_fn(return_pspec=True)

  def _quantize_fn(self, return_pspec: bool) -> Union[NestedJTensor, Any]:
    """Get the quantized weight or partition specs of the sublayer.

    Args:
      return_pspec: a boolean to control if returning ParititionSpecs for
        quantized tensors. If True, returns the partition specs. If False,
        returns quantized tensors.

    Returns:
      a nested map from names to quantized layer or partition spec.
    """

    def body_fn(sub, _):
      if return_pspec:
        res = sub.quantized_partition_specs()
      else:
        res = sub.quantize_weight()
      return None, res

    def add_leading_none(x):
      # Adding a leading 'None' to PartitionSpec for repeats.
      if isinstance(x, base_layer.BoxedPartitionSpec):
        return base_layer.BoxedPartitionSpec(
            meta=jax.sharding.PartitionSpec(None, *x.meta))
      else:
        return x

    scan_fn = nn.scan(
        body_fn,
        variable_axes=SCAN_VARIABLE_AXES,
        split_rngs={RANDOM: True},
        length=self.x_times,
    )
    scan_fn = self._wrap_for_nd(scan_fn)

    _, res = scan_fn(self.sublayer, None)
    ret = {}
    for collection in [PARAMS, NON_TRAINABLE]:
      if collection in res:
        ret[collection] = {self.sublayer_name: res[collection]}

    if return_pspec:
      ret = jax.tree_map(
          add_leading_none,
          ret,
          is_leaf=lambda x: isinstance(x, base_layer.BoxedPartitionSpec))
    return ret

  @property
  def sublayer(self) -> base_layer.BaseLayer:
    return getattr(self, self.sublayer_name)

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

    assert not self.is_initializing()

    def body_fn(sub, _):
      sub.init_states(*args, **kwargs)
      return None, None

    scan_fn = nn.scan(
        body_fn,
        variable_axes=SCAN_VARIABLE_AXES,
        split_rngs={PARAMS: self.is_initializing(), RANDOM: True},
        length=self.x_times,
    )
    scan_fn = self._wrap_for_nd(scan_fn)

    mapped_scan_fn = nn.map_variables(
        scan_fn,
        SUMMARIES,
        mutable=self.is_mutable_collection(SUMMARIES),
        trans_in_fn=functools.partial(
            flax_utils.maybe_repack_summary,
            unpack_summaries=self.unpack_summaries,
            x_times=self.x_times,
        ),
        trans_out_fn=functools.partial(
            flax_utils.maybe_unpack_summary,
            unpack_summaries=self.unpack_summaries,
            x_times=self.x_times,
        ),
    )

    # Calls scan_fn with a None carry_in and ignores the carry_out.
    mapped_scan_fn(self.sublayer, None)

  def _run_unrolled_for_decoding(
      self, fn: Callable[[base_layer.BaseLayer, Any], Any], inputs: Any
  ) -> Any:

    def _run_one_layer(i, inp):

      def _map_in(tree):
        for collection, subtree in tree.items():
          if collection not in [DECODE_CACHE, PREFIX_DECODE_CACHE]:
            layer_i = jax.tree_map(lambda x: x[i], subtree)
            subtree.clear()
            subtree.update(layer_i)
          else:
            if f'layer{i}' in subtree:
              layer_i = subtree[f'layer{i}']
              del subtree[f'layer{i}']
            else:
              # This could happen during lazy_broadcast_prefix since the vars in
              # PREFIX_DECODE_CACHE are newly created.
              layer_i = {}
            other_layers = subtree.copy()
            subtree.clear()
            subtree.update(layer_i)
            subtree['_repeats_other_layers'] = other_layers
        return tree

      def _map_out(tree):
        for collection, subtree in tree.items():
          if collection not in [DECODE_CACHE, PREFIX_DECODE_CACHE]:
            continue
          if '_repeats_other_layers' in subtree:
            other_layers = subtree['_repeats_other_layers']
            del subtree['_repeats_other_layers']
          else:
            # This could happen during the first layer of lazy_broadcast_prefix.
            other_layers = {}
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
              unpack_summaries=self.unpack_summaries,
              x_times=self.x_times,
          ),
      )
      # TODO(yuanzx): we do not support summaries/aux_losses yet.
      mapped_fn = nn.map_variables(
          mapped_fn, [PARAMS, NON_TRAINABLE, SUMMARIES, AUX_LOSS],
          mutable=False)
      mapped_fn = self._wrap_for_nd(mapped_fn, unrolled=True)
      return mapped_fn(self.sublayer, inp)

    out = inputs
    for i in range(self.x_times):
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

    assert not self.is_initializing()

    # TODO(zhangqiaorjc): Apply remat?
    def body_fn(sub, layer_in):
      layer_out = sub.extend_step(layer_in, *args, **kwargs)
      asserts.assert_same_structure(layer_in, layer_out)
      return layer_out, None

    if self.unroll_in_decode:
      return self._run_unrolled_for_decoding(body_fn, step_inputs)

    # Note that in_axes specification skips `carry` and supports prefix spec.
    scan_fn = nn.scan(
        body_fn,
        in_axes=0,  # scan over axis 0 for layer_states
        variable_axes=SCAN_VARIABLE_AXES,
        split_rngs={RANDOM: True},
        length=self.x_times,
    )
    scan_fn = self._wrap_for_nd(scan_fn)

    mapped_scan_fn = nn.map_variables(
        scan_fn,
        SUMMARIES,
        mutable=self.is_mutable_collection(SUMMARIES),
        trans_in_fn=functools.partial(
            flax_utils.maybe_repack_summary,
            unpack_summaries=self.unpack_summaries,
            x_times=self.x_times,
        ),
        trans_out_fn=functools.partial(
            flax_utils.maybe_unpack_summary,
            unpack_summaries=self.unpack_summaries,
            x_times=self.x_times,
        ),
    )

    mapped_scan_fn = nn.map_variables(
        mapped_scan_fn,
        AUX_LOSS,
        mutable=self.is_mutable_collection(AUX_LOSS),
        trans_out_fn=_sum_aux_loss)

    # PREFIX_DECODE_CACHE is kept constant during decoding.
    mapped_scan_fn = nn.map_variables(
        mapped_scan_fn, PREFIX_DECODE_CACHE, mutable=False)

    layer_out, _ = mapped_scan_fn(self.sublayer, step_inputs)
    return layer_out

  def transform_decode_state(
      self, transform_fn: base_layer.DecodeStateTransformFn) -> None:
    """Transforms all decode state variables based on transform_fn."""

    assert not self.is_initializing()

    def body_fn(sub, _):
      sub.transform_decode_state(transform_fn)
      return None, None

    if self.unroll_in_decode:
      self._run_unrolled_for_decoding(body_fn, None)
      return

    scan_fn = nn.scan(
        body_fn,
        variable_axes=SCAN_VARIABLE_AXES,
        split_rngs={RANDOM: True},
        length=self.x_times,
    )
    scan_fn = self._wrap_for_nd(scan_fn)

    mapped_scan_fn = nn.map_variables(
        scan_fn,
        SUMMARIES,
        mutable=self.is_mutable_collection(SUMMARIES),
        trans_in_fn=functools.partial(
            flax_utils.maybe_repack_summary,
            unpack_summaries=self.unpack_summaries,
            x_times=self.x_times,
        ),
        trans_out_fn=functools.partial(
            flax_utils.maybe_unpack_summary,
            unpack_summaries=self.unpack_summaries,
            x_times=self.x_times,
        ),
    )

    mapped_scan_fn(self.sublayer, None)

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

    assert not self.is_initializing()

    def body_fn(sub, _):
      sub.lazy_broadcast_prefix(num_suffix_samples, suffix_length)
      return None, None

    if self.unroll_in_decode:
      return self._run_unrolled_for_decoding(body_fn, None)

    scan_fn = nn.scan(
        body_fn,
        variable_axes=SCAN_VARIABLE_AXES,
        split_rngs={RANDOM: True},
        length=self.x_times,
    )
    scan_fn = self._wrap_for_nd(scan_fn)

    mapped_scan_fn = nn.map_variables(
        scan_fn,
        SUMMARIES,
        mutable=self.is_mutable_collection(SUMMARIES),
        trans_in_fn=functools.partial(
            flax_utils.maybe_repack_summary,
            unpack_summaries=self.unpack_summaries,
            x_times=self.x_times,
        ),
        trans_out_fn=functools.partial(
            flax_utils.maybe_unpack_summary,
            unpack_summaries=self.unpack_summaries,
            x_times=self.x_times,
        ),
    )

    mapped_scan_fn(self.sublayer, None)

  def right_align_decode_state_with_prefix(
      self, max_prefix_size: int,
      right_align_fn: base_layer.DecodeStateTransformFn) -> None:
    """Right aligns decode state with prefix decode states.

    Args:
      max_prefix_size: Max prefix length of the decode state.
      right_align_fn: Right align function for decode state.
    """

    def body_fn(sub, _):
      sub.right_align_decode_state_with_prefix(max_prefix_size, right_align_fn)
      return None, None

    if self.unroll_in_decode:
      return self._run_unrolled_for_decoding(body_fn, None)

    scan_fn = nn.scan(
        body_fn,
        variable_axes=SCAN_VARIABLE_AXES,
        split_rngs={RANDOM: True},
        length=self.x_times,
    )
    scan_fn = self._wrap_for_nd(scan_fn)

    mapped_scan_fn = nn.map_variables(
        scan_fn,
        SUMMARIES,
        mutable=self.is_mutable_collection(SUMMARIES),
        trans_in_fn=functools.partial(
            flax_utils.maybe_repack_summary,
            unpack_summaries=self.unpack_summaries,
            x_times=self.x_times,
        ),
        trans_out_fn=functools.partial(
            flax_utils.maybe_unpack_summary,
            unpack_summaries=self.unpack_summaries,
            x_times=self.x_times,
        ),
    )

    mapped_scan_fn(self.sublayer, None)


def _ensure_tuple(x: Any) -> Tuple[Any, ...]:
  """Ensures that `x` is a tuple."""
  return x if isinstance(x, tuple) else (x,)
