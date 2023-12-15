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

"""Functional RNN-related layers."""

from flax import linen as nn
import jax
from jax import numpy as jnp
from praxis import base_layer
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis.layers import rnn_cell

NestedMap = py_utils.NestedMap

JTensor = pytypes.JTensor
LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]

PARAMS = base_layer.PARAMS
AUX_LOSS = base_layer.AUX_LOSS
HYPER_PARAMS = base_layer.HYPER_PARAMS
SUMMARIES = base_layer.SUMMARIES
NON_TRAINABLE = base_layer.NON_TRAINABLE
RANDOM = base_layer.RANDOM

# RNN share weights across time dimension, so PARAMS are never split.
SCAN_SPLIT_RNGS = {PARAMS: False, RANDOM: True}


def reset_mask(
    segment_ids: JTensor,
    paddings: JTensor | None = None,
    dtype: jnp.dtype = jnp.float32,
) -> JTensor:
  """Computes reset mask.

  Reset mask is a 0/1 tensor where 0 indicating the position to reset rnn
  cell state. When padding is provided, reset_mask at padding positions are
  all 1, otherwise, reset_mask at the start of padding positions takes 0 values.
  The shape of segment_ids, paddings and return value must all match.

  Args:
    segment_ids: A JTensor of shape [B, S] or [B, S, 1], the segment that each
      token belongs to.
    paddings: A 0/1 JTensor of shape [B, S] or [B, S, 1], value 1 indicates the
      current position is padding.
    dtype: data type of the output.

  Returns:
    A JTensor of shape [B, S] or [B, S, 1]
  """
  batch_size = segment_ids.shape[0]
  left_padding_shape = [1] * segment_ids.ndim
  left_padding_shape[0] = batch_size
  left_padding = jnp.full(left_padding_shape, -1, dtype=segment_ids.dtype)
  # Segment ids of previous step. Using -1 for the first step.
  segment_ids_of_previous_step = jnp.concatenate(
      [left_padding, segment_ids[:, :-1]], axis=1
  )
  # Boolean valued mask where False indicating the position to reset.
  mask = jnp.equal(segment_ids_of_previous_step, segment_ids)
  if paddings is None:
    return mask.astype(dtype)
  return jnp.logical_or(mask, paddings).astype(dtype)


def _sum_aux_loss(tree):
  return jax.tree_map(jnp.sum, tree)


class FRnn(base_layer.BaseLayer):
  """A generic Rnn layer that works with any RnnCell.

  Attributes:
    cell_tpl: Configs for the RnnCell.
    reverse: Whether or not to unroll the sequence in reversed order.
    unroll: Number of steps to unroll in the scan function (using >1 can speed
      up gradient computation).
  """
  cell_tpl: LayerTpl | None = base_layer.template_field(None)
  reverse: bool = False
  unroll: int = 1

  def setup(self) -> None:
    assert self.unroll > 0, 'Unroll must be positive.'
    self.create_child('cell', self.cell_tpl)

  def init_states(self, batch_size: int) -> NestedMap:
    return self.cell.init_states(batch_size)

  def extend_step(
      self, inputs: NestedMap, state0: NestedMap
  ) -> tuple[NestedMap, JTensor]:
    """Extends Rnn for one step on 'inputs' from 'state0'.

    Args:
      inputs: A nestedMap of inputs for one time-step. Must contain two elements
        'act' and 'padding'. 'act' can be a single tensor or a list of tensors,
        all of shape [b, inner_dim], 'padding' is a tensor of shape [b, 1].
      state0: The initial state to extend from.

    Returns:
      A pair (state1, act), where state1 is the new state, and act is the
      output for the current time-step.
    """
    # Reset cell state is not supported during decoding.
    inputs.reset_mask = jnp.ones_like(inputs.padding)
    state1 = self.cell(state0, inputs)
    return state1, self.get_output(state1)

  def get_output(self, state: NestedMap) -> JTensor:
    return self.cell.get_output(state)

  def __call__(
      self, inputs: NestedMap, state0: NestedMap | None = None
  ) -> tuple[JTensor, NestedMap]:
    """Computes frnn forward pass.

    Args:
      inputs: A NestedMap of inputs. 'inputs' must contain two elements, 'act'
        and 'padding'. 'act' can be a single tensor, or a list/tuple of tensors,
        all of shape [b, t, dim], and 'padding' is of shape [b, t, 1]. 'inputs'
        can optionally contain other tensors. For example, when 'inputs' is
        packed, 'segment_ids' which has shape [b, t, 1] is expected.
      state0: If not None, the initial rnn state in a `NestedMap`. Defaults to
        the cell's zero-state.

    Returns:
      act: A tensor of shape [batch, time, dims]. The output.
      state: Final state.
    """
    # Make a copy of the input structure to avoid side-effect.
    inputs = jax.tree_map(lambda x: x, inputs)
    assert hasattr(inputs, 'act')
    assert hasattr(inputs, 'padding')
    assert isinstance(self.cell, rnn_cell.BaseRnnCell)

    if not isinstance(inputs.act, (list, tuple)):
      inputs.act = [inputs.act]

    if hasattr(inputs, 'segment_ids'):
      inputs.reset_mask = reset_mask(
          inputs.segment_ids, inputs.padding, dtype=self.fprop_dtype
      )
    else:
      inputs.reset_mask = jnp.ones_like(inputs.padding, dtype=self.fprop_dtype)

    if self.reverse:
      inputs = jax.tree_map(lambda x: jnp.flip(x, axis=[1]), inputs)

    if not state0:
      batch_size = inputs.padding.shape[0]
      state0 = self.init_states(batch_size)

    def body_fn(sub, state0, inputs):
      state1 = sub(state0, inputs)
      return state1, sub.get_output(state1)

    # Flax nn.scan has the limitation that variables cannot be created inside.
    # So we must create variables before we enter nn.scan by calling body_fn
    # once to initialize the variables with the right shape expected by the
    # scan during layer.init.
    if self.is_initializing():
      # inputs has shape [b, t, dim] or [b, t, 1]
      # sliced_inputs has shape [b, dim] or [b, 1].
      sliced_inputs = jax.tree_map(lambda x: x[:, 1], inputs)
      _ = body_fn(self.cell, state0, sliced_inputs)

    # NON_TRAINABLE variables are carried over from one iteration to another.
    # For example, frnn iteration n+1 is able to see updated batch norm
    # variables from frnn iteration n. AUX_LOSS and SUMMARIES are scanned over.
    scan_fn = nn.scan(
        body_fn,
        variable_axes={AUX_LOSS: 0, SUMMARIES: 0, HYPER_PARAMS: 0},
        variable_broadcast=[PARAMS],
        variable_carry=[NON_TRAINABLE],
        split_rngs=SCAN_SPLIT_RNGS,
        in_axes=1,
        out_axes=1,
        unroll=self.unroll,
    )
    # Sum-up aux losses.
    mapped_scan_fn = nn.map_variables(
        scan_fn,
        AUX_LOSS,
        mutable=self.is_mutable_collection(AUX_LOSS),
        trans_out_fn=_sum_aux_loss)

    # NOTE(yonghui): typically, sequences are pretty long in RNN. Unpacking the
    # summaries will result in too many individual summaries.

    final_state, act = mapped_scan_fn(self.cell, state0, inputs)

    if self.reverse:
      act = jnp.flip(act, axis=[1])
    return act, final_state


class StackFrnn(base_layer.BaseLayer):
  """A stacked FRNN which includes multiple layers.

  Attributes:
    frnn_tpl: Configs for the frnn.
    num_layers: number of frnn layers.
    num_input_nodes: Number of input nodes.
    num_output_nodes: Number of output nodes. If num_hidden_nodes is 0, also
      used as cell size.
  """
  frnn_tpl: LayerTpl | None = base_layer.template_field(None)
  num_layers: int = 1
  num_input_nodes: int = 0
  num_output_nodes: int = 0

  def setup(self) -> None:

    assert self.num_layers > 0
    input_nodes = self.num_input_nodes
    frnns_p = []
    assert self.frnn_tpl is not None
    for _ in range(self.num_layers):
      frnn_p = self.frnn_tpl.clone()
      frnn_p.cell_tpl.set(
          num_input_nodes=input_nodes, num_output_nodes=self.num_output_nodes
      )
      frnns_p.append(frnn_p)
      input_nodes = self.num_output_nodes

    self.create_children('frnn', frnns_p)

  def init_states(self, batch_size: int) -> list[NestedMap]:
    return [
        self.frnn[i].init_states(batch_size) for i in range(self.num_layers)
    ]

  def extend_step(
      self, inputs: NestedMap, state: list[NestedMap]
  ) -> tuple[list[NestedMap], JTensor]:
    inputs = jax.tree_map(lambda x: x, inputs)
    new_states = []
    for i in range(self.num_layers):
      new_state, act_i = self.frnn[i].extend_step(inputs, state[i])
      inputs.act = act_i
      new_states.append(new_state)
    return new_states, inputs.act

  def get_output(self, state: list[NestedMap]) -> JTensor:
    return self.frnn[-1].get_output(state[-1])

  def __call__(
      self, inputs: NestedMap, state0: list[NestedMap] | None = None
  ) -> tuple[JTensor, list[NestedMap]]:
    """Computes Stacked LSTM forward pass.

    Args:
      inputs: A NestedMap of inputs. 'inputs' must contain two elements, 'act'
        and 'padding', 'act' can be a single tensor, or a list/tuple of
        tensors, all of shape [b, t, dim], and 'padding' are of shape [b, t, 1].
        'inputs' can optionally contain other tensors.
      state0: If not None, the initial rnn state in a List of `.NestedMap`.

    Returns:
      act: A tensor of [batch, time, dims]. The output.
      state: Final state.
    """
    inputs = jax.tree_map(lambda x: x, inputs)

    if not state0:
      batch_size = inputs.padding.shape[0]
      state0 = self.init_states(batch_size)

    final_states = []
    for i in range(self.num_layers):
      act_i, state = self.frnn[i](inputs=inputs, state0=state0[i])
      inputs.act = act_i
      final_states.append(state)
    return inputs.act, final_states


class StackBiFrnn(base_layer.BaseLayer):
  """A stack of a bidrectional FRNN which includes multiple layers.

    Each layer is composed of an RNN and a reverse RNN. We pass the input to
    both RNNs, and then concat both of their outputs and pass to the next layer.

    Note that the output of each RNN is half the dim of the total output.


  Attributes:
    frnn_tpl: Configs for the frnn.
    num_layers: number of frnn layers.
    num_input_nodes: Number of input nodes.
    num_output_nodes: Number of output nodes. If num_hidden_nodes is 0, also
      used as cell size.
  """

  frnn_tpl: LayerTpl | None = base_layer.template_field(None)
  num_layers: int = 1
  num_input_nodes: int = 0
  num_output_nodes: int = 0

  def setup(self) -> None:
    assert self.num_layers > 0
    input_nodes = self.num_input_nodes
    fwd_frnns_p = []
    bwd_frnns_p = []
    # The output of both directions are concatenated and passed to the next
    # layer, thus the divsion of the output_notes by 2.
    out_nodes = self.num_output_nodes // 2
    assert self.frnn_tpl is not None
    for _ in range(self.num_layers):
      # Add the forward cells.
      fwd_p = self.frnn_tpl.clone()
      fwd_p.cell_tpl.set(
          num_input_nodes=input_nodes, num_output_nodes=out_nodes
      )
      fwd_p.reverse = False  # Forward.
      fwd_frnns_p.append(fwd_p)

      # Add the backward cells.
      bwd_p = self.frnn_tpl.clone()
      bwd_p.cell_tpl.set(
          num_input_nodes=input_nodes, num_output_nodes=out_nodes
      )
      bwd_p.reverse = True  # Backward.
      bwd_frnns_p.append(bwd_p)

      input_nodes = self.num_output_nodes

    self.create_children('fwd_frnn', fwd_frnns_p)
    self.create_children('bwd_frnn', bwd_frnns_p)

  def fwd_init_states(self, batch_size: int) -> list[NestedMap]:
    return [
        self.fwd_frnn[i].init_states(batch_size) for i in range(self.num_layers)
    ]

  def bwd_init_states(self, batch_size: int) -> list[NestedMap]:
    return [
        self.bwd_frnn[i].init_states(batch_size) for i in range(self.num_layers)
    ]

  def init_states(self, batch_size: int) -> NestedMap:
    state = NestedMap()
    state.fwd = self.fwd_init_states(batch_size)
    state.bwd = self.bwd_init_states(batch_size)
    return state

  def __call__(
      self, inputs: NestedMap, state0: NestedMap | None = None
  ) -> tuple[JTensor, list[NestedMap]]:
    """Computes the bidrectional stacked RNN forward pass.

    Args:
      inputs: A NestedMap of inputs. 'inputs' must contain two elements, 'act'
        and 'padding', 'act' can be a single tensor, or a list/tuple of tensors,
        all of shape [b, t, dim], and 'padding' are of shape [b, t, 1]. 'inputs'
        can optionally contain other tensors.
      state0: If not None, the initial rnn state of `.NestedMap`.

    Returns:
      act: A tensor of [batch, time, dims]. The output.
      state: Final state - a list of NestedMap of fwd and bwd states.
    """
    # This is to create a copy.
    inputs = jax.tree_map(lambda x: x, inputs)

    if not state0:
      batch_size = inputs.padding.shape[0]
      state0 = self.init_states(batch_size)

    final_states = []
    for i in range(self.num_layers):
      fwd_act_i, fwd_state = self.fwd_frnn[i](
          inputs=inputs, state0=state0.fwd[i]
      )
      bwd_act_i, bwd_state = self.bwd_frnn[i](
          inputs=inputs, state0=state0.bwd[i]
      )
      inputs.act = jnp.concatenate([fwd_act_i, bwd_act_i], axis=-1)
      final_states.append(NestedMap(fwd=fwd_state, bwd=bwd_state))
    return inputs.act, final_states


class LstmFrnn(FRnn):
  """A FRNN for LSTMCellSimple cell.

  It exploits the parallelism in input projection across time steps for better
  efficiency.
  """

  @property
  def num_input_nodes(self) -> int:
    return self.cell.num_input_nodes

  @property
  def num_output_nodes(self) -> int:
    return self.cell.num_output_nodes

  def __call__(
      self, inputs: NestedMap, state0: NestedMap | None = None
  ) -> tuple[JTensor, NestedMap]:
    """Computes LSTM forward pass.

    Args:
      inputs: A NestedMap of inputs. 'inputs' must contain two elements, 'act'
        and 'padding', 'act' can be a single tensor, or a list/tuple of tensors,
        all of shape [b, t, dim], and 'padding' are of shape [b, t, 1]. 'inputs'
        can optionally contain other tensors. For example, when 'inputs' is
        packed, 'segment_ids' which has shape [b, t, 1] is expected.
      state0: If not None, the initial rnn state in a `.NestedMap`. Defaults to
        the cell's zero-state.

    Returns:
      act: A tensor of [batch, time, dims]. The output.
      state: Final state.
    """
    # Make a copy of the input structure to avoid side-effect.
    inputs = jax.tree_map(lambda x: x, inputs)
    assert hasattr(inputs, 'act')
    assert hasattr(inputs, 'padding')
    assert isinstance(self.cell, rnn_cell.BaseRnnCell)

    if not isinstance(inputs.act, (list, tuple)):
      inputs.act = [inputs.act]

    if hasattr(inputs, 'segment_ids'):
      inputs.reset_mask = reset_mask(
          inputs.segment_ids, inputs.padding, dtype=self.fprop_dtype
      )
    else:
      inputs.reset_mask = jnp.ones_like(inputs.padding, dtype=self.fprop_dtype)

    if self.reverse:
      inputs = jax.tree_map(lambda x: jnp.flip(x, axis=[1]), inputs)

    if not state0:
      batch_size = inputs.padding.shape[0]
      state0 = self.init_states(batch_size)

    # [T, B, H]
    proj_inputs = self.cell.project_input(inputs)
    inputs.proj_inputs = proj_inputs

    # TODO(nanxinchen): test whether pre-slicing wm_h improves efficiency
    def body_fn(sub, state0, inputs):
      state1 = sub.fprop_with_projected_inputs(state0, inputs)
      return state1, sub.get_output(state1)

    # Flax nn.scan has the limitation that variables cannot be created inside.
    # So we must create variables before we enter nn.scan by calling body_fn
    # once to initialize the variables with the right shape expected by the
    # scan during layer.init.
    if self.is_initializing():
      # inputs has shape [b, t, dim] or [b, t, 1]
      # sliced_inputs has shape [b, dim] or [b, 1].
      sliced_inputs = jax.tree_map(lambda x: x[:, 1], inputs)
      # `body_fn` is sufficient to trigger PARAMS initialization.
      _ = body_fn(self.cell, state0, sliced_inputs)

    # NON_TRAINABLE variables are carried over from one iteration to another.
    # For example, frnn iteration n+1 is able to see updated batch norm
    # variables from frnn iteration n.
    scan_fn = nn.scan(
        body_fn,
        variable_axes={AUX_LOSS: 0, SUMMARIES: 0, HYPER_PARAMS: 0},
        variable_broadcast=[PARAMS],
        variable_carry=[NON_TRAINABLE],
        split_rngs=SCAN_SPLIT_RNGS,
        in_axes=1,
        out_axes=1,
        unroll=self.unroll,
    )
    # Sum-up aux losses.
    mapped_scan_fn = nn.map_variables(
        scan_fn,
        AUX_LOSS,
        mutable=self.is_mutable_collection(AUX_LOSS),
        trans_out_fn=_sum_aux_loss)

    final_state, act = mapped_scan_fn(self.cell, state0, inputs)

    if self.reverse:
      act = jnp.flip(act, axis=[1])
    return act, final_state
