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

import jax
import jax.numpy as jnp

from praxis import base_layer
from praxis import pytypes
from praxis.layers import linears
from praxis.layers.sparsity import sparsity
from praxis.layers.sparsity import sparsity_hparams

SparsityMode = sparsity_hparams.SparsityMode
SparsityType = sparsity_hparams.SparsityType
SparsityHParams = sparsity_hparams.SparsityHParams
WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams

instance_field = base_layer.instance_field
JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor


class Linear(linears.Linear):
  """Sparsed Linear layer without bias.

  Attributes:
    sparsity: The relevant information related to the kind of sparsity that is
      applied to this layer.
  """

  sparsity: SparsityHParams = instance_field(SparsityHParams)

  def setup(self) -> None:
    wp = self.weight_split_dims_mapping
    pc = WeightHParams(
        shape=[self.input_dims, self.output_dims],
        mesh_shape=self.mesh_shape,
        tensor_split_dims_mapping=wp.wt,
    )
    if self.sparsity.mode == SparsityMode.INFERENCE:
      self.create_variable('w', pc)
    else:
      self.create_sparse_variable('w', pc)
      count_pc = WeightHParams(
          shape=[], init=WeightInit.Constant(0), dtype=jnp.int32
      )
      if (
          self.sparsity.mode == SparsityMode.ONESHOT
          or self.sparsity.mode == SparsityMode.FEWSHOT
      ):
        self.create_variable('mask_update_count', count_pc, trainable=False)

  def _update_mask(self, weight):
    return sparsity.get_sparsity_mask(
        weight,
        n_sparsity=self.sparsity.weight_params.prune_rate[0],
        m_sparsity=self.sparsity.weight_params.prune_rate[1],
    )

  def _maybe_update_mask(self, update_count, weight, mask):
    def _true_fn():
      return self._update_mask(weight), update_count + 1

    def _false_fn():
      return mask, update_count

    return jax.lax.cond(
        update_count < self.sparsity.num_shots, _true_fn, _false_fn
    )

  def __call__(self, inputs: JTensor) -> JTensor:
    """Apply projection to inputs.

    Args:
      inputs: The inputs JTensor.  Shaped [..., input_dims].

    Returns:
      Projected inputs.
    """
    ap = self.activation_split_dims_mapping
    if self.sparsity.mode == SparsityMode.INFERENCE:
      out = linears.project_last_dim(inputs, self.theta.w)
    else:
      w, m = self.get_sparse_weight('w')
      if self.sparsity.sparsity_type == SparsityType.UNSTRUCTURED:
        raise NotImplementedError(
            'Unstructured sparsity is not currently supported.'
        )
      elif self.sparsity.sparsity_type == SparsityType.STRUCTURED_NM:
        if (
            self.sparsity.mode == SparsityMode.ONESHOT
            or self.sparsity.mode == SparsityMode.FEWSHOT
        ):
          up_cnt = self.get_var('mask_update_count')
          m, up_cnt = self._maybe_update_mask(up_cnt, w, m)
          self.update_var('mask_update_count', up_cnt)
        else:
          m = self._update_mask(w)

        # update_var function needs to be out of jax.lax.cond, due to
        # flax.errors.JaxTransformError: Jax transforms and Flax models cannot
        # be mixed.
        self.update_var('w' + base_layer.SPARSITY_NAME_POSTFIX, m)
        w = sparsity.apply_sparsity(w, m)
        out = linears.project_last_dim(inputs, w)
      else:
        raise ValueError('Unknown sparsity_type. It should be UNSTRUCTURED or'
                         'STRUCTURED_NM.')
    # Adjust sharding annotation during decoding.
    # TODO(pax): This logic should likely be lifted somewhere else.
    ap_out = ap.out
    if ap_out is not None and len(ap_out) == 3 and out.ndim == 2:
      ap_out = [ap_out[0], ap_out[2]]
    out = base_layer.maybe_shard(out, ap_out, self.mesh_axis_names)
    return out
