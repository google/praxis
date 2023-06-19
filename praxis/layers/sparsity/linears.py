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

"""Sparse Linear Layers."""

from praxis import base_layer
from praxis import pytypes
from praxis.layers import linears
from praxis.layers.sparsity import sparse_base_layer
from praxis.layers.sparsity import sparsity_hparams


SparsityMode = sparsity_hparams.SparsityMode
SparsityScore = sparsity_hparams.SparsityScore
SparsityType = sparsity_hparams.SparsityType
SparsityHParams = sparsity_hparams.SparsityHParams
WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams

instance_field = base_layer.instance_field
JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor


class Linear(sparse_base_layer.SparsityBaseLayer, linears.Linear):  # pytype: disable=signature-mismatch
  """Sparsed Linear layer without bias.

  Attributes:
    sparsity: The relevant information related to the kind of sparsity that is
      applied to this layer.
  """

  def setup(self) -> None:
    wp = self.weight_split_dims_mapping
    weight_hp = WeightHParams(
        shape=[self.input_dims, self.output_dims],
        init=self.weight_init,
        mesh_shape=self.mesh_shape,
        tensor_split_dims_mapping=wp.wt,
    )
    name = 'w'
    self.create_variable(name, weight_hp)
    self.create_child('einsum', self.einsum_tpl.clone())
    self.create_aux_variables(name, weight_hp)

  def __call__(self, inputs: JTensor) -> JTensor:
    """Apply projection to inputs.

    Args:
      inputs: The inputs JTensor.  Shaped [..., input_dims].

    Returns:
      Projected inputs.
    """
    ap = self.activation_split_dims_mapping

    w = self.sparsifiy(
        self.theta.w, inputs=inputs, name='w'
    )  # sparsify weight.
    out = self.einsum('...y,yz->...z', inputs, w)

    # Adjust sharding annotation during decoding.
    # TODO(pax): This logic should likely be lifted somewhere else.
    ap_out = ap.out
    if ap_out is not None and len(ap_out) == 3 and out.ndim == 2:
      ap_out = [ap_out[0], ap_out[2]]
    out = base_layer.maybe_shard(out, ap_out, self.mesh_axis_names)
    return out
