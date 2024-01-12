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

"""A layer that computes an Einsum with a weight, and optionally adds a bias."""

from typing import Sequence

from praxis import base_layer
from praxis import pax_fiddle
from praxis import pytypes
from praxis.layers.quantization import quantizer

JTensor = pytypes.JTensor
LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]
template_field = base_layer.template_field


class Einsum(quantizer.QuantizationLayer):
  """Layer that computes an einsum and maybe a bias.

  The fan-in, fan-out and bias dimensions are inferred from the einsum equation.
  If bias is used, the fan-out dims must appear at the end of the output tensor.

  Attributes:
    eqn: Einsum equation. It should be in the format of input,w->output. E.g.,
      '...d,df->...f'.
    w_shape: Weight shape.
    use_bias: Whether to add a bias.
  """
  eqn: str = ''
  w_shape: Sequence[int] = ()
  use_bias: bool = False
  _PACK_4BIT_DIM = 0

  def setup(self) -> None:
    operands, out = self.eqn.split('->')
    x, w = operands.split(',')
    assert '.' not in w
    fan_in = sorted(w.index(d) for d in (set(x) - set(out)))
    fan_out = sorted(w.index(d) for d in (set(out) - set(x)))
    w_sharding = self.weight_split_dims_mapping.wt
    pc = base_layer.WeightHParams(
        shape=self.w_shape,
        fan_in_axes=fan_in,
        fan_out_axes=fan_out,
        mesh_shape=self.mesh_shape,
        tensor_split_dims_mapping=w_sharding,
    )
    out_bias_dims = sorted(out.index(d) for d in (set(out) - set(x)))
    bias_shape = [self.w_shape[w.index(out[d])] for d in out_bias_dims]
    self.set_up_weights(
        weight_name='w',
        weight_params=pc,
        scale_shape=bias_shape,
    )
    if self.use_bias:
      # Fan-out dims must be at the end of `out`.
      assert all(d >= len(out) - len(out_bias_dims) for d in out_bias_dims)
      if w_sharding is not None:
        b_sharding = [w_sharding[w.index(out[d])] for d in out_bias_dims]
      else:
        b_sharding = None
      pc_bias = base_layer.WeightHParams(
          shape=bias_shape,
          init=base_layer.WeightInit.Constant(0.0),
          mesh_shape=self.mesh_shape,
          tensor_split_dims_mapping=b_sharding,
      )
      self.create_variable('b', pc_bias)

  def __call__(self, inputs: JTensor) -> JTensor:
    """Computes the einsum and maybe bias.

    Args:
      inputs: A JTensor of shape as described in the equation.

    Returns:
      The result of the einsum with maybe a bias added.
    """
    ret = self.quantized_einsum(
        eqn=self.eqn,
        x=inputs,
        w=self.theta.w,
        reshape=[],
    )
    if self.use_bias:
      ret += self.theta.b
    return base_layer.maybe_shard(
        ret, self.activation_split_dims_mapping.out, self.mesh_axis_names
    )
