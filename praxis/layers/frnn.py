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

"""Functional RNN-related layers."""

from typing import List, Optional, Tuple

from flax import linen as nn
import jax
from jax import numpy as jnp
from praxis import base_layer
from praxis import py_utils
from praxis import pytypes
from praxis.layers import rnn_cell

NestedMap = py_utils.NestedMap

JTensor = pytypes.JTensor

BaseHParams = base_layer.BaseLayer.HParams

PARAMS = base_layer.PARAMS
AUX_LOSS = base_layer.AUX_LOSS
SUMMARIES = base_layer.SUMMARIES
NON_TRAINABLE = base_layer.NON_TRAINABLE
RANDOM = base_layer.RANDOM

# RNN share weights across time dimension, so PARAMS are never split.
SCAN_SPLIT_RNGS = {PARAMS: False, RANDOM: True}


def _sum_aux_loss(tree):
  return jax.tree_map(jnp.sum, tree)


class FRnn(base_layer.BaseLayer):
  """A generic Rnn layer that works with any RnnCell."""

  class HParams(BaseHParams):
    """Associated hyper-params for this layer class.

    Attributes:
      cell_tpl: Configs for the RnnCell.
      reverse: Whether or not to unroll the sequence in reversed order.
    """
    cell_tpl: Optional[BaseHParams] = base_layer.sub_config_field(None)
    reverse: bool = False

  def setup(self) -> None:
    p = self.hparams
    self.create_child('cell', p.cell_tpl)

  def init_states(self, batch_size: int) -> NestedMap:
    return self.cell.init_states(batch_size)

  def extend_step(self, inputs: NestedMap,
                  state0: NestedMap) -> Tuple[NestedMap, JTensor]:
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
    state1 = self.cell(state0, inputs)
    return state1, self.get_output(state1)

  def get_output(self, state: NestedMap) -> JTensor:
    return self.cell.get_output(state)

  def __call__(self,
               inputs: NestedMap,
               state0: Optional[NestedMap] = None) -> Tuple[JTensor, NestedMap]:
    """Computes frnn forward pass.

    Args:
      inputs: A NestedMap of inputs. 'inputs' must contain two elements, 'act'
        and 'padding'. 'act' can be a single tensor, or a list/tuple of tensors,
        all of shape [b, t, dim], and 'padding' is of shape [b, t, 1]. 'inputs'
        can optionally contain other tensors too.
      state0: If not None, the initial rnn state in a `NestedMap`. Defaults to
        the cell's zero-state.

    Returns:
      act: A tensor of shape [batch, time, dims]. The output.
      state: Final state.
    """
    p = self.hparams
    # Make a copy of the input structure to avoid side-effect.
    inputs = jax.tree_map(lambda x: x, inputs)
    assert isinstance(inputs, NestedMap)
    assert hasattr(inputs, 'act')
    assert hasattr(inputs, 'padding')
    assert isinstance(self.cell, rnn_cell.BaseRnnCell)

    if p.reverse:
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
        variable_axes={
            AUX_LOSS: 0,
            SUMMARIES: 0
        },
        variable_broadcast=[PARAMS],
        variable_carry=[NON_TRAINABLE],
        split_rngs=SCAN_SPLIT_RNGS,
        in_axes=1,
        out_axes=1,
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

    if p.reverse:
      act = jnp.flip(act, axis=[1])
    return act, final_state


class StackFrnn(base_layer.BaseLayer):
  """A stacked FRNN which includes multiple layers."""

  class HParams(BaseHParams):
    """Associated hyper-params for this layer class.

    Attributes:
      frnn_tpl: Configs for the frnn.
      num_layers: number of frnn layers.
      num_input_nodes: Number of input nodes.
      num_output_nodes: Number of output nodes. If num_hidden_nodes is 0, also
        used as cell size.
    """
    frnn_tpl: Optional[BaseHParams] = base_layer.sub_config_field(None)
    num_layers: int = 1
    num_input_nodes: int = 0
    num_output_nodes: int = 0

  def setup(self) -> None:
    p = self.hparams

    assert p.num_layers > 0
    input_nodes = p.num_input_nodes
    frnns_p = []
    for _ in range(p.num_layers):
      frnn_p = p.frnn_tpl.clone()
      frnn_p.cell_tpl.set(
          num_input_nodes=input_nodes, num_output_nodes=p.num_output_nodes)
      frnns_p.append(frnn_p)
      input_nodes = p.num_output_nodes

    self.create_children('frnn', frnns_p)

  @property
  def num_input_nodes(self) -> int:
    return self.hparams.num_input_nodes

  @property
  def num_output_nodes(self) -> int:
    return self.hparams.num_output_nodes

  def init_states(self, batch_size: int) -> List[NestedMap]:
    return [
        self.frnn[i].init_states(batch_size)
        for i in range(self.hparams.num_layers)
    ]

  def extend_step(self, inputs: NestedMap,
                  state: List[NestedMap]) -> Tuple[List[NestedMap], JTensor]:
    p = self.hparams
    inputs = jax.tree_map(lambda x: x, inputs)
    new_states = []
    for i in range(p.num_layers):
      new_state, act_i = self.frnn[i].extend_step(inputs, state[i])
      inputs.act = act_i
      new_states.append(new_state)
    return new_states, inputs.act

  def get_output(self, state: List[NestedMap]) -> JTensor:
    return self.frnn[-1].get_output(state[-1])

  def __call__(self,
               inputs: NestedMap,
               state0: Optional[List[NestedMap]] = None
              ) -> Tuple[JTensor, List[NestedMap]]:
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
    p = self.hparams
    inputs = jax.tree_map(lambda x: x, inputs)

    if not state0:
      batch_size = inputs.padding.shape[0]
      state0 = self.init_states(batch_size)

    final_states = []
    for i in range(p.num_layers):
      act_i, state = self.frnn[i](inputs=inputs, state0=state0[i])
      inputs.act = act_i
      final_states.append(state)
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

  def __call__(self,
               inputs: NestedMap,
               state0: Optional[NestedMap] = None) -> Tuple[JTensor, NestedMap]:
    """Computes LSTM forward pass.

    Args:
      inputs: A NestedMap of inputs. 'inputs' must contain two elements, 'act'
        and 'padding', 'act' can be a single tensor, or a list/tuple of tensors,
        all of shape [b, t, dim], and 'padding' are of shape [b, t, 1]. 'inputs'
        can optionally contain other tensors.
      state0: If not None, the initial rnn state in a `.NestedMap`. Defaults to
        the cell's zero-state.

    Returns:
      act: A tensor of [batch, time, dims]. The output.
      state: Final state.
    """
    p = self.hparams
    # Make a copy of the inputs nested structure.
    inputs = jax.tree_map(lambda x: x, inputs)
    assert isinstance(self.cell, rnn_cell.BaseRnnCell)

    if not isinstance(inputs.act, (list, tuple)):
      inputs.act = [inputs.act]

    # TODO(pax): support packed input.
    inputs.reset_mask = jnp.zeros_like(inputs.padding)

    if p.reverse:
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
        variable_axes={
            AUX_LOSS: 0,
            SUMMARIES: 0
        },
        in_axes=1,
        out_axes=1,
        variable_broadcast=[PARAMS],
        variable_carry=[NON_TRAINABLE],
        split_rngs=SCAN_SPLIT_RNGS)
    # Sum-up aux losses.
    mapped_scan_fn = nn.map_variables(
        scan_fn,
        AUX_LOSS,
        mutable=self.is_mutable_collection(AUX_LOSS),
        trans_out_fn=_sum_aux_loss)

    final_state, act = mapped_scan_fn(self.cell, state0, inputs)

    if p.reverse:
      act = jnp.flip(act, axis=[1])
    return act, final_state
