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

"""RNN-related layers."""

import jax
from jax import numpy as jnp
from praxis import asserts
from praxis import base_layer
from praxis import py_utils
from praxis import pytypes

NestedMap = py_utils.NestedMap
WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams

Params = py_utils.HParams
JTensor = pytypes.JTensor

BaseHParams = base_layer.BaseLayer.HParams


def _zoneout_helper(prev_v: JTensor, cur_v: JTensor, padding_v: JTensor,
                    zo_prob: float, is_eval: bool,
                    random_uniform: JTensor) -> JTensor:
  """A helper function to apply ZoneOut regularlization to cur_v.

  Implements ZoneOut regularization as described in
  https://arxiv.org/abs/1606.01305

  Args:
    prev_v: Values from the previous timestep.
    cur_v: Values from the current timestep.
    padding_v: The paddings vector for the cur timestep.
    zo_prob: Probability at which to apply ZoneOut regularization.
    is_eval: Whether or not in eval mode.
    random_uniform: Random uniform numbers. This can be None if zo_prob=0.0

  Returns:
    cur_v after ZoneOut regularization has been applied.
  """
  prev_v = jnp.array(prev_v)
  cur_v = jnp.array(cur_v)
  padding_v = jnp.array(padding_v)
  if zo_prob == 0.0:
    # Special case for when ZoneOut is not enabled.
    return jnp.where(padding_v, prev_v, cur_v)

  if is_eval:
    mix_prev = jnp.full(prev_v.shape, zo_prob) * prev_v
    mix_curr = jnp.full(cur_v.shape, 1.0 - zo_prob) * cur_v
    mix = mix_prev + mix_curr

    # If padding_v is 1, it always carries over the previous state.
    return jnp.where(padding_v, prev_v, mix)
  else:
    asserts.not_none(random_uniform)
    zo_p = (random_uniform < zo_prob).astype(padding_v.dtype)
    zo_p += padding_v
    # If padding_v is 1, we always carry over the previous state.
    zo_p = jnp.minimum(zo_p, 1.0)
    zo_p = jax.lax.stop_gradient(zo_p)
    return jnp.where(zo_p, prev_v, cur_v)


class BaseRnnCell(base_layer.BaseLayer):
  """Base class for all RnnCell.

  An RNN cell encapsulates the logic for performing one single step in an RNN
  layer.

  RNNCell represents recurrent state in a `NestedMap`.

  `init_states(batch_size)` returns the initial state, which is defined
  by each subclass. From the state, each subclass defines `get_output()`
  to extract the output tensor.

  `RNNCell.fprop` defines the forward function::

      (state0, inputs) -> state1

  All arguments and return values are `NestedMap`. Each subclass defines
  what fields these `NestedMap` are expected to have.

  `init_states(batch_size)`, `state0` and `state1` are all compatible
  `NestedMap` (see `NestedMap.IsCompatible`).
  I.e., they have the same keys recursively. Furthermore, the corresponding
  tensors in these `NestedMap` have the same shape and dtype.
  """

  def init_states(self, batch_size: int) -> NestedMap:
    """Returns the initial state given the batch size."""
    raise NotImplementedError('Abstract method')

  def get_output(self, state: NestedMap) -> NestedMap:
    """Returns the output value given the current state."""
    raise NotImplementedError('Abstract method')

  def __call__(self, state0: NestedMap, inputs: NestedMap) -> NestedMap:
    """Forward function.

    Args:
      state0: The previous recurrent state.
      inputs: The inputs to the cell. 'inputs' is expected to have two elements
        at the minimal, 'act' and 'padding'. 'act' is expected to have shape [b,
        dim] and 'padding' expected to be of shape [b, 1]. 'act' can be a single
        tensor or a list/tuple of tensors.

    Returns:
      state1: The next recurrent state.
    """
    raise NotImplementedError('Abstract method')


class LstmCellSimple(BaseRnnCell):
  """Simple LSTM cell.

  theta:

  - wm: the parameter weight matrix. All gates combined.
  - b: the combined bias vector.

  state:

  - m: the lstm output. [batch, cell_nodes]
  - c: the lstm cell state. [batch, cell_nodes]

  inputs:

  - act: a list of input activations, of shape [batch, input_nodes]
  - padding: the padding. [batch, 1].
  - reset_mask: optional 0/1 float input to support packed input training.
    Shape [batch, 1]
  """

  class HParams(BaseRnnCell.HParams):
    """Associated hyper-params for this layer class.

    Attributes:
      inputs_arity: Number of tensors expected for the inputs.act field.
      num_input_nodes: Number of input nodes.
      num_output_nodes: Number of output nodes. If num_hidden_nodes is 0, also
        used as cell size.
      num_hidden_nodes: Number of projection hidden nodes (see
        https://arxiv.org/abs/1603.08042). Set to 0 to disable projection.
      reset_cell_state: Set True to support resetting cell state in scenarios
        where multiple inputs are packed into a single training example. The RNN
        layer should provide reset_mask inputs in addition to act and padding if
        this flag is set.
      cell_value_cap: Cell values are capped to be within [-cell_value_cap,
        +cell_value_cap] if the value is not None. It can be a scalar, a scalar
        tensor or None. When set to None, no capping is applied.
      forget_gate_bias: Bias to apply to the forget gate.
      output_nonlinearity: Whether or not to apply tanh non-linearity on lstm
        output.
      zo_prob: If > 0, applies ZoneOut regularization with the given prob.
      bias_init: Initialization parameters for bias.
    """
    inputs_arity: int = 1
    num_input_nodes: int = 0
    num_output_nodes: int = 0
    num_hidden_nodes: int = 0
    reset_cell_state: bool = False
    cell_value_cap: float = 10.0
    forget_gate_bias: float = 0.0
    output_nonlinearity: bool = True
    zo_prob: float = 0.0
    bias_init: WeightInit = WeightInit.Constant(0.0)

  @property
  def num_gates(self) -> int:
    return 4

  @property
  def output_size(self) -> int:
    return self.hparams.num_output_nodes

  @property
  def hidden_size(self) -> int:
    return self.hparams.num_hidden_nodes or self.hparams.num_output_nodes

  def setup(self) -> None:
    """Initializes LSTMCellSimple."""
    p = self.hparams
    if p.cell_value_cap is not None:
      asserts.instance(p.cell_value_cap, (int, float))

    # Define weights.
    wm_pc = WeightHParams(shape=[
        p.num_input_nodes + self.output_size, self.num_gates * self.hidden_size
    ])
    self.create_variable('wm', wm_pc)

    if p.num_hidden_nodes:
      w_proj = WeightHParams(shape=[self.hidden_size, self.output_size])
      self.create_variable('w_proj', w_proj)

    bias_pc = WeightHParams(
        shape=[self.num_gates * self.hidden_size], init=p.bias_init)
    self.create_variable('b', bias_pc)

  def init_states(self, batch_size: int) -> NestedMap:
    zero_m = jnp.zeros((batch_size, self.output_size))
    zero_c = jnp.zeros((batch_size, self.hidden_size))
    return NestedMap(m=zero_m, c=zero_c)

  def _reset_state(self, state: NestedMap, inputs: NestedMap) -> NestedMap:
    state.m = inputs.reset_mask * state.m
    state.c = inputs.reset_mask * state.c
    return state

  def get_output(self, state: NestedMap) -> JTensor:
    return state.m

  def __call__(self, state0: NestedMap, inputs: NestedMap) -> NestedMap:
    """Forward function.

    `_reset_state` is optionally applied if `reset_cell_state` is True. The RNN
    layer should provide `reset_mask` inputs in addition to other inputs.
    `reset_mask` inputs are expected to be 0 at timesteps where state0 should be
    reset to default (zeros) before running `fprop`, and 1
    otherwise. This is meant to support use cases like packed inputs, where
    multiple samples are fed in a single input example sequence, and need to be
    masked from each other. For example, if the two examples packed together
    are ['good', 'day'] -> ['guten-tag'] and ['thanks'] -> ['danke']
    to produce ['good', 'day', 'thanks'] -> ['guten-tag', 'danke'], the
    source reset_masks would be [1, 1, 0] and target reset masks would be
    [1, 0]. These ids are meant to enable masking computations for
    different examples from each other.

    Args:
      state0: The previous recurrent state.
      inputs: The inputs to the cell.

    Returns:
      state1: The next recurrent state.
    """
    p = self.hparams
    inputs = jax.tree_map(lambda x: x, inputs)
    if not isinstance(inputs.act, (list, tuple)):
      inputs.act = [inputs.act]
    asserts.eq(self.hparams.inputs_arity, len(inputs.act))

    if p.reset_cell_state:
      state0 = self._reset_state(state0, inputs)

    concat = jnp.concatenate(inputs.act + [state0.m], 1)
    xmw = jnp.einsum('bd,dc->bc', concat, self.theta.wm)
    xmw += self._bias_adjustment()
    i_i, i_g, f_g, o_g = jnp.split(xmw, self.num_gates, axis=1)
    state1 = self._gates_internal(state0, i_i, i_g, f_g, o_g)
    state1 = self._apply_zoneout(state0, inputs, state1)
    return state1

  def _bias_adjustment(self) -> JTensor:
    p = self.hparams
    bias = self.theta.b
    if p.forget_gate_bias != 0.0:
      adjustment = (
          jnp.ones([self.num_gates, self.hidden_size]) *
          jnp.expand_dims(jnp.array([0., 0., p.forget_gate_bias, 0.]), axis=1))
      adjustment = jnp.reshape(adjustment, [self.num_gates * self.hidden_size])
      bias += adjustment
    return bias

  def _gates_internal(self, state0: NestedMap, i_i: JTensor, i_g: JTensor,
                      f_g: JTensor, o_g: JTensor) -> NestedMap:
    p = self.hparams
    forget_gate = jax.nn.sigmoid(f_g) * state0.c
    input_gate = jax.nn.sigmoid(i_g) * jnp.tanh(i_i)
    new_c = forget_gate + input_gate
    # Clip the cell states to reasonable value.
    if p.cell_value_cap is not None:
      new_c = jnp.clip(new_c, -p.cell_value_cap, p.cell_value_cap)

    if p.output_nonlinearity:
      new_m = jax.nn.sigmoid(o_g) * jnp.tanh(new_c)
    else:
      new_m = jax.nn.sigmoid(o_g) * new_c
    if p.num_hidden_nodes:
      new_m = jnp.einsum('bd,dc->bc', new_m, self.theta.w_proj)

    return NestedMap(c=new_c, m=new_m)

  def _apply_zoneout(self, state0: NestedMap, inputs: NestedMap,
                     state1: NestedMap) -> NestedMap:
    """Apply Zoneout and returns the updated states."""
    p = self.hparams

    if p.zo_prob > 0.0:
      c_random_uniform = jax.random.uniform(self.next_prng_key(),
                                            state0.c.shape)
      m_random_uniform = jax.random.uniform(self.next_prng_key(),
                                            state0.m.shape)
    else:
      c_random_uniform = None
      m_random_uniform = None

    new_c = _zoneout_helper(state0.c, state1.c, inputs.padding, p.zo_prob,
                            self.do_eval, c_random_uniform)
    new_m = _zoneout_helper(state0.m, state1.m, inputs.padding, p.zo_prob,
                            self.do_eval, m_random_uniform)

    return NestedMap(m=new_m, c=new_c)

  def project_input(self, inputs: NestedMap) -> JTensor:
    """Applies input projection for the entire sequence.

    Args:
      inputs: A NestedMap with the following fields:
        - act: A list of Tensors of shape [seqlen, batch, input_dim].

    Returns:
      A Tensor of shape [seqlen, batch, 4 * hidden_dim].
    """
    act = inputs.act
    if not isinstance(act, (list, tuple)):
      act = [act]
    if len(act) > 1:
      x = jnp.concatenate(inputs.act, axis=-1)
    else:
      x = act[0]
    # [T, B, 4 * H]
    num_input_nodes = self.hparams.num_input_nodes
    assert num_input_nodes == x.shape[-1]
    wm_i = self.theta.wm[:num_input_nodes, :]

    proj_inputs = jnp.einsum('TBD,DH->TBH', x, wm_i)
    return proj_inputs

  def fprop_with_projected_inputs(self, state0: NestedMap,
                                  inputs: NestedMap) -> NestedMap:
    """FProp with inputs already projected.

    This method is for parallelizing the input projection across time steps to
    accelerate training.

    The following are equivalent:

    >>> inputs = <a tensor of [T, B, D]>
    >>> padding = tf.zeros([T, B])
    >>> state = cell.zero_state(B)

    # a. Use fprop().
    >>> for i in range(T):
    ...  state, _ = cell(inputs[i, :, :], padding, state)

    # b. Use fprop_with_projected_inputs().
    >>> proj_inputs = cell.project_input(inputs)
    >>> for i in range(T):
    ...  state, _ = cell.fprop_with_projected_inputs(
    ...    proj_inputs[i, :, :], padding, state)

    Args:
      state0: A NestedMap with the same structure as return value of
        `self.zero_state()`.
      inputs: A NestedMap with the following fields:
        - proj_inputs: A single Tensors of shape [batch, 4 * hidden_dim].
        - padding: A Tensor of shape [batch, 1].
        - reset_mask: A Tensor of shape [batch, 1].

    Returns:
      state1: A NestedMap of the same structure as `state0`.
    """
    p = self.hparams
    if p.reset_cell_state:
      state0 = self._reset_state(state0, inputs)

    num_input_nodes = self.hparams.num_input_nodes
    wm_h = self.theta.wm[num_input_nodes:, :]
    proj_m = jnp.einsum('bd,dc->bc', state0.m, wm_h)
    xmw = inputs.proj_inputs + proj_m

    xmw += self._bias_adjustment()
    i_i, i_g, f_g, o_g = jnp.split(xmw, self.num_gates, axis=1)

    state1 = self._gates_internal(state0, i_i, i_g, f_g, o_g)
    state1 = self._apply_zoneout(state0, inputs, state1)
    return state1


class CifgLstmCellSimple(LstmCellSimple):
  """CIFG variant LSTM which couple the input and output gate."""

  @property
  def num_gates(self) -> int:
    return 3

  def __call__(self, state0: NestedMap, inputs: NestedMap) -> NestedMap:
    """Forward function.

    Please see LstmCellSimple.fprop for more details.

    Args:
      state0: The previous recurrent state.
      inputs: The inputs to the cell.

    Returns:
      state1: The next recurrent state.
    """
    p = self.hparams
    p: CifgLstmCellSimple.HParams

    inputs = jax.tree_map(lambda x: x, inputs)
    if not isinstance(inputs.act, (list, tuple)):
      inputs.act = [inputs.act]

    asserts.eq(self.hparams.inputs_arity, len(inputs.act))

    if p.reset_cell_state:
      state0 = self._reset_state(state0, inputs)

    concat = jnp.concatenate(inputs.act + [state0.m], 1)
    xmw = jnp.einsum('bd,dc->bc', concat, self.theta.wm)
    # CifgLstmCellSimple doesn't support forget gate bias.
    assert p.forget_gate_bias == 0.0
    xmw += self.theta.b
    i_i, f_g, o_g = jnp.split(xmw, self.num_gates, axis=1)
    state1 = self._gates_internal(state0, i_i, f_g, o_g)
    state1 = self._apply_zoneout(state0, inputs, state1)
    return state1

  def _gates_internal(self, state0: NestedMap, i_i: JTensor, f_g: JTensor,
                      o_g: JTensor) -> NestedMap:
    p = self.hparams
    forget_gate = jax.nn.sigmoid(f_g) * state0.c
    # Coupled input and forget gate.
    input_gate = (1.0 - jax.nn.sigmoid(f_g)) * jnp.tanh(i_i)
    new_c = forget_gate + input_gate
    # Clip the cell states to reasonable value.
    if p.cell_value_cap is not None:
      new_c = jnp.clip(new_c, -p.cell_value_cap, p.cell_value_cap)

    if p.output_nonlinearity:
      new_m = jax.nn.sigmoid(o_g) * jnp.tanh(new_c)
    else:
      new_m = jax.nn.sigmoid(o_g) * new_c
    if p.num_hidden_nodes:
      new_m = jnp.einsum('bd,dc->bc', new_m, self.theta.w_proj)

    return NestedMap(c=new_c, m=new_m)

  def fprop_with_projected_inputs(self, state0: NestedMap,
                                  inputs: NestedMap) -> NestedMap:
    """FProp with inputs already projected.

    Please see LstmCellSimple.fprop_with_projected_inputs for more details.

    Args:
      state0: A NestedMap with the same structure as return value of
        `self.zero_state()`.
      inputs: A NestedMap with the following fields:
        - proj_inputs: A single Tensors of shape [batch, 4 * hidden_dim].
        - padding: A Tensor of shape [batch, 1].
        - reset_mask: A Tensor of shape [batch, 1].

    Returns:
      state1: A NestedMap of the same structure as `state0`.
    """
    p = self.hparams
    if p.reset_cell_state:
      state0 = self._reset_state(state0, inputs)

    num_input_nodes = self.hparams.num_input_nodes
    wm_h = self.theta.wm[num_input_nodes:, :]
    proj_m = jnp.einsum('bd,dc->bc', state0.m, wm_h)
    xmw = inputs.proj_inputs + proj_m

    # CifgLstmCellSimple doesn't support forget gate bias.
    assert p.forget_gate_bias == 0.0

    xmw += self.theta.b
    i_i, f_g, o_g = jnp.split(xmw, self.num_gates, axis=1)

    state1 = self._gates_internal(state0, i_i, f_g, o_g)
    state1 = self._apply_zoneout(state0, inputs, state1)
    return state1
