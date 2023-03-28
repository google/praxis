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

"""Collection of hyper-parameters for sparsity."""

import dataclasses
import enum
from typing import Optional, Tuple, Union
from praxis import base_hyperparams


@enum.unique
class SparsityType(str, enum.Enum):
  """The different types for sparsity.

  `STRUCTURED_NM` implementes N:M structured sparsity.
  `UNSTRUCTURED` implementes unstructured sparsity.
  """

  STRUCTURED_NM = 'structured_nm'
  UNSTRUCTURED = 'unstructured'


@enum.unique
class SparsityMode(str, enum.Enum):
  """The different modes for sparsity.

  TRAINING indicates that the model is in the sparse training mode.
  MATERIALIZE indicates that the model weights are being materialized as
    sparsed weights. After materialization mode is set to inference.
  INFERENCE indicates that the model is in the inference mode with sparsed
    weights.
  """

  TRAINING = 'training'
  MATERIALIZE = 'materialize'
  INFERENCE = 'inference'


# TODO(ayazdan): Define parameters for activation sparsity.
@dataclasses.dataclass
class WeightSparsityParams:
  """Parameters for sparsity.

  prune_rate: Defines the rate of pruning, either for unstructured sparsity
    or N:M structured sparsity.
  """

  # TODO(ayazdan): Add additional sparsity parameters (order, offset, etc.)
  prune_rate: Union[None, float, Tuple[int, int]]


class SparsityHParams(base_hyperparams.BaseHyperParams):
  """Collection of hyper-parameters for sparsity.

  Attributes:
    sparsity_type: Defines sparsity types.
  """

  sparsity_type: SparsityType = SparsityType.STRUCTURED_NM
  weight_params: Optional[WeightSparsityParams] = None
  mode: SparsityMode = SparsityMode.INFERENCE

  def __post_init__(self):
    if self.weight_params is not None:
      if self.weight_params.prune_rate is not None:
        if self.sparsity_type == SparsityType.STRUCTURED_NM:
          assert isinstance(self.weight_params.prune_rate, Tuple), (
              'Prune rate must be either None '
              'for no pruning or a Tuple[int, int] for '
              'N:M structured sparsity.'
          )
        elif self.sparsity_type == SparsityType.UNSTRUCTURED:
          assert isinstance(self.weight_params.prune_rate, float), (
              'Prune rate must be either None for no pruning or float '
              'for unstructured sparsity.'
          )
        else:
          assert False, f'Unrecognized sparsity type {self.sparsity_type}.'
