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

import enum

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
class ActivationQuantizationType(str, enum.Enum):
  """The different types for activation quantization.

  NONE indicates no quantization on the activation.
  DYNAMIC indicates dynamic quantization on the activation.
  STATIC indicates static quantization on the activation.
  """
  NONE = 'none'
  DYNAMIC = 'dynamic'
  STATIC = 'static'


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


class QuantizationHParams(base_hyperparams.BaseHyperParams):
  """Parameters for quantization.

  Attributes:
    quantization_type: quantization type.
    mode: the quantization mode associated with this quantization parameter.
    activation_quantization_type: quantization type for activation.
  """

  quantization_type: QuantizationType = QuantizationType.PTQ
  mode: QuantizationMode = QuantizationMode.INFERENCE
  activation_quantization_type: ActivationQuantizationType = ActivationQuantizationType.NONE