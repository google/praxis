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

"""Adapter layers."""

from typing import Optional

import jax
import jax.numpy as jnp
from praxis import asserts
from praxis import base_layer
from praxis import py_utils
from praxis import pytypes
from praxis.layers import activations
from praxis.layers import normalizations

NestedMap = py_utils.NestedMap
JTensor = pytypes.JTensor
weight_init = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams
sub_config_field = base_layer.sub_config_field

BaseHParams = base_layer.BaseLayer.HParams


class MultitaskResidualAdapter(base_layer.BaseLayer):
  """A multitask residual adapter layer.

  https://arxiv.org/pdf/1902.00751.pdf

  Residual adapter introduces a small network which includes two fc layer
  with bottleneck in the middle. Normalization is used on inputs before
  feeding into these fc layers. Residual connection is also used.

  This is a multi-task residual adapter and each task has its own adapter
  conditioning on the task id which is provided during fprop.

  """

  class HParams(BaseHParams):
    """Associated hyperparams for this layer class.

    Attributes:
      input_dims: input dimension to the Adapter
      bottleneck_dims: bottleneck dimension of the adapter
      num_tasks: total number of tasks
      norm_tpl: normalization used in the beginning
      activation_tpl: activation template to use.
    """
    input_dims: int = 0
    bottleneck_dims: int = 0
    num_tasks: int = 1
    norm_tpl: BaseHParams = sub_config_field(normalizations.LayerNorm.HParams)
    activation_tpl: activations.BaseActivation.HParams = sub_config_field(
        activations.ReLU.HParams)

  def setup(self) -> None:
    p = self.hparams
    norm_tpl = p.norm_tpl.clone()
    if norm_tpl.cls in {
        normalizations.BatchNorm, normalizations.GroupNorm,
        normalizations.LayerNorm
    }:
      norm_tpl.dim = p.input_dims
    else:
      raise NotImplementedError('%s is not supported' % norm_tpl.cls)
    self.create_child('norm', norm_tpl)

    down_w_pc = WeightHParams(
        shape=[p.num_tasks, p.input_dims, p.bottleneck_dims])
    self.create_variable('down_w', down_w_pc)
    down_b_pc = WeightHParams(
        shape=[p.num_tasks, p.bottleneck_dims], init=weight_init.Constant(0.))
    self.create_variable('down_b', down_b_pc)
    up_w_pc = WeightHParams(
        shape=[p.num_tasks, p.bottleneck_dims, p.input_dims])
    self.create_variable('up_w', up_w_pc)
    up_b_pc = WeightHParams(
        shape=[p.num_tasks, p.input_dims], init=weight_init.Constant(0.))
    self.create_variable('up_b', up_b_pc)

    self.create_child('activation', p.activation_tpl)

  def __call__(self,
               inputs: JTensor,
               paddings: Optional[JTensor] = None,
               tasks: Optional[JTensor] = None) -> JTensor:
    """Fprop for multitask adapter.

    Args:
      inputs: A tensor containing the activations from the previous layer.
        [batch, ..., input_dims].
      paddings: A tensor indicating whether it is padding (1.0) or not (0.0).
        Optional and only used for BatchNorm.
      tasks: An int32 tensor containing the task ID for each input. The shape
        should match the first n dimension of inputs and the differences between
        dimensions should be less than two. For example, inputs with [batch,
        time, input_dims] and tasks [batch]. Another possibility is to have
        inputs with shape [batch, time, input_dims] and tasks [batch, time].

    Returns:
      A tensor containing the adapted activations with the same shape as inputs.
    """
    p = self.hparams

    if tasks is None:
      asserts.eq(1, p.num_tasks, msg='tasks is not specified but num_tasks!=1')
      tasks = jnp.zeros(shape=inputs.shape[:-1])

    asserts.eq(tasks.shape, inputs.shape[:len(tasks.shape)])
    asserts.gt(len(inputs.shape) - len(tasks.shape), 0)
    asserts.le(len(inputs.shape) - len(tasks.shape), 2)
    tasks_onehot = jax.nn.one_hot(
        tasks, p.num_tasks, axis=-1, dtype=inputs.dtype)

    # Einsum axis names:
    # k - task
    # i - input_dims
    # n - bottleneck_dims

    down_w = jnp.einsum('...k,kin->...in', tasks_onehot, self.theta.down_w)
    if len(down_w.shape) <= len(inputs.shape):
      down_w = jnp.expand_dims(down_w, -3)
    down_b = jnp.einsum('...k,kn->...n', tasks_onehot, self.theta.down_b)
    if len(down_b.shape) < len(inputs.shape):
      down_b = jnp.expand_dims(down_b, -2)
    up_w = jnp.einsum('...k,kni->...ni', tasks_onehot, self.theta.up_w)
    if len(up_w.shape) <= len(inputs.shape):
      up_w = jnp.expand_dims(up_w, -3)
    up_b = jnp.einsum('...k,ki->...i', tasks_onehot, self.theta.up_b)
    if len(up_b.shape) < len(inputs.shape):
      up_b = jnp.expand_dims(up_b, -2)

    # Norm -> down-projection -> non-linearity -> up-projection
    norm_inputs = self.norm(inputs)
    if p.norm_tpl.cls in {
        normalizations.BatchNorm, normalizations.GroupNorm,
        normalizations.LayerNorm
    }:
      norm_inputs = self.norm(inputs, paddings)
    else:
      raise NotImplementedError('%s is not supported' % p.norm_tpl.cls)

    down_projected = jnp.einsum('...i,...in->...n', norm_inputs,
                                down_w) + down_b
    down_projected = self.activation(down_projected)
    up_projected = jnp.einsum('...n,...ni->...i', down_projected, up_w) + up_b
    return inputs + up_projected
