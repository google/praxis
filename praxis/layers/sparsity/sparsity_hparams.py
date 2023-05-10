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
  # Oneshot pruning do sparsity mask updating for the very first step.
  ONESHOT = 'oneshot'
  # Fewshot pruning do sparsity mask updating for the beginning steps, user
  # needs to define num_shots > 1 correspondingly.
  FEWSHOT = 'fewshot'
  MATERIALIZE = 'materialize'
  INFERENCE = 'inference'


# TODO(ayazdan): Define parameters for activation sparsity.
@dataclasses.dataclass
class WeightSparsityParams:
  """Parameters for sparsity.

  Attributes:
    prune_rate:  Defines the rate of pruning, either for unstructured sparsity
      or N:M structured sparsity.
    structure_decay: If True, a decaying schedule is applied for the structured
      sparsity, the algorithm is described in:
      https://arxiv.org/pdf/2209.07617.pdf.
    mask_decay_weight: If 0.0, no mask decay is applied. The mask value start
      with 1.0 and each time `num_update_sparsity` * `mask_decay_weight` is
      subtracted from 1.0. Due to overhead of jit, we limited the number of
      updates to `num_update_sparsity` to 16. After 16 iterations, we forcefully
      set `mask_decay_value` to zero. Mask decaying works for both structured
      and unstructured sparsity. The algorithm is described in:
      https://arxiv.org/pdf/2209.07617.pdf.
    sparse_ste: If True, a sparse-refined straight-through estimator (SR-STE) is
      applied, following the algorithm described in:
        https://arxiv.org/abs/2102.04010
    sparse_ste_weight: Denotes the relative weight for the sparse-refined term.
      As mentioned in the paper (https://arxiv.org/abs/2102.04010), the best
      default value is 0.0002 (lambda_w in the paper).
  """

  # TODO(ayazdan): Add additional sparsity parameters (order, offset, etc.)
  prune_rate: Union[None, float, Tuple[int, int]]
  structure_decay: bool = False
  mask_decay_weight: float = 0.0
  sparse_ste: bool = False
  sparse_ste_weight: float = 0.0002

  def __post_init__(self):
    assert self.mask_decay_weight >= 0.0, (
        'Invalid value for '
        f'{self.mask_decay_weight}. '
        '`mask_decay_weight` must be positive.'
    )

    assert self.sparse_ste_weight > 0.0, (
        'Invalid value for '
        f'{self.sparse_ste_weight}. '
        '`sparse_ste_weight` must be positive.'
    )

    if self.sparse_ste:
      if self.mask_decay_weight != 0.0:
        raise ValueError('SR-STE only works with non-decaying mask.')
      if self.structure_decay:
        raise ValueError(
            'SR-STE only works with non-decaying sparse structure.'
        )


@dataclasses.dataclass
class SparsityHParams:
  """Collection of hyper-parameters for sparsity.

  Attributes:
    sparsity_type: Defines sparsity types.
  """

  sparsity_type: SparsityType = SparsityType.STRUCTURED_NM
  weight_params: Optional[WeightSparsityParams] = None
  mode: SparsityMode = SparsityMode.INFERENCE
  num_shots: int = 0

  def __post_init__(self):
    if (
        self.weight_params is not None
        and self.weight_params.prune_rate is not None
    ):
      # Check sparsity types.
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
        if self.weight_params.sparse_ste:
          raise ValueError('SR-STE only works with structured sparsity.')

      else:
        assert False, f'Unrecognized sparsity type {self.sparsity_type}.'

      # Check sparsity mode.
      if self.mode == SparsityMode.ONESHOT:
        assert (
            self.num_shots == 1
        ), '`num_shots should be set for ONESHOT sparse.`'
      elif self.mode == SparsityMode.FEWSHOT:
        assert (
            self.num_shots > 1
        ), '`num_shots should be set for FEWSHOT sparse.`'
