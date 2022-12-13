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

"""Collection of hyper-parameters for quantization."""

import dataclasses
import enum
from typing import Optional

from praxis import base_hyperparams


@enum.unique
class QuantizationType(str, enum.Enum):
  """The different types for quantization.

  PTQ indicates Post Training Quantization.
  AQT indicates Accurate Quantized Training, which is one flavor of QAT.
  FQ  indicates Fake Quantization, which is one flavor of QAT.
  """

  PTQ = 'ptq'
  AQT = 'aqt'
  FQ = 'fq'


@enum.unique
class QuantizationMode(str, enum.Enum):
  """The different modes for quantization.

  TRAINING indicates that the model is in the training mode.
  MATERIALIZE indicates that the model weights are being materialized as
    quantized weights and scales. After materialization mode is set to
    inference.
  INFERENCE indicates that the model is in inference mode.
  """
  TRAINING = 'training'
  MATERIALIZE = 'materialize'
  INFERENCE = 'inference'


@dataclasses.dataclass
class ActQuantizationParams:
  """Parameters for activation quantization.

  precision: the precision (number of bits) for activation quantization.
  unsigned_int_bounds: whether or not to use unsigned_int_bounds.
  clipping_coeff: the coefficient to shrink the hard range for activation
    quantization.
  stats_config: static values used for quantize activation.
    stats_config == None: dynamic activation quantization
    otherwise: static activation quantization
  """
  precision: int = 8
  unsigned_int_bounds: bool = False
  clipping_coeff: float = 0.0
  # TODO(jihwanlee): Define stats config for static quantization
  stats_config = None


@dataclasses.dataclass
class WeightQuantizationParams:
  """Parameters for weight quantization.

  precision: the precision (number of bits) for activation quantization.
  unsigned_int_bounds: whether or not to use unsigned_int_bounds.
  clipping_coeff: the coefficient to shrink the hard range for activation
    quantization.
  """
  precision: int = 8
  unsigned_int_bounds: bool = False
  clipping_coeff: float = 0.0


class QuantizationHParams(base_hyperparams.BaseHyperParams):
  """Parameters for quantization.

  Attributes:
    quantization_type: quantization type.
    mode: the quantization mode associated with this quantization parameter.
    act_params: Config for activation quantization.
    weight_params: Config for weight quantization.
  """

  quantization_type: QuantizationType = QuantizationType.PTQ
  mode: QuantizationMode = QuantizationMode.INFERENCE
  act_params: Optional[ActQuantizationParams] = None
  weight_params: WeightQuantizationParams = WeightQuantizationParams()
