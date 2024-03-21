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

"""Collection of hyper-parameters for quantization."""

import dataclasses
import enum

import jax.numpy as jnp

# Internal import for internal quantization hyper parameters.


@enum.unique
class QuantizationType(str, enum.Enum):
  """The different types for quantization.

  PTQ: Post Training Quantization.
  AQT: Accurate Quantized Training, which is one flavor of QAT.
  FQ:  Fake Quantization, which is one flavor of QAT.
  FQ_VN:  Use variational noise to emulate quantization noise.
  FR: Fr quantization.
  """
  PTQ = 'ptq'
  AQT = 'aqt'
  FQ = 'fq'
  FQ_VN = 'fq_vn'
  FR = 'fr'
  # Internal quantization type.


@enum.unique
class QuantizationMode(str, enum.Enum):
  """The different modes for quantization.

  TRAINING indicates that the model is in the training mode.
  MATERIALIZE indicates that the model weights are being materialized as
    quantized weights and scales. After materialization the mode is set to
    INFERENCE. This mode is referenced only by `ServableModelParams` for
    serving.
  INFERENCE indicates that the model is in inference mode.
  QT indicates the model will train with quantization.
  CALIB inidates that the model is going to be calibrated.
  """
  TRAINING = 'training'
  MATERIALIZE = 'materialize'
  INFERENCE = 'inference'
  QT = 'qt'
  CALIB = 'calib'


@enum.unique
class TransformerLayer(str, enum.Enum):
  """Transformer layer types used in quantization.

  Users can use this enum to specify mixed precision quantization
  for transformer models.

  LINEAR: The FFN layer weight in a transformer.
  LINEAR_ACT: The FFN layer activation in a transformer.
  ATTENTION: The attention layer weight.
  ATTENTION_ACT: The attention layer activation.
  EMBEDDING_SOFTMAX: Embedding for the softmax layer.
  EMBEDDING_SOFTMAX_ACT: Embedding activation for the softmax layer.
  EMBEDDING_NGRAMMER: Embedding for the ngrammar layer.
  """

  LINEAR = 'linear'
  LINEAR_ACT = 'linear_activation'
  ATTENTION = 'attention'
  ATTENTION_ACT = 'attention_activation'
  EMBEDDING_SOFTMAX = 'embedding_softmax'
  EMBEDDING_SOFTMAX_ACT = 'embedding_softmax_activation'
  EMBEDDING_NGRAMMER = 'embedding_ngrammer'


@dataclasses.dataclass
class ActQuantizationParams:
  """Parameters for activation quantization.

  precision: The precision (number of bits) for activation quantization.
  unsigned_int_bounds: Whether or not to use unsigned_int_bounds.
  clipping_coeff: The coefficient to shrink the hard range for activation
    quantization. 1.0 means using hard min/max.
  stats_config: Static values used for quantize activation.
    stats_config == None: dynamic activation quantization
    otherwise: static activation quantization
  stop_scale_gradient: Stop the gradient of the quantization scale for numerical
    stability. Note: this is numerically incorrect.
  fp16: clip activation to fp16. This overrides the int8 activation quantization
    for QAT.
  per_channel: Whether or not to quantize activation channel-wisely.
  symmetric: Whether or not to quantize activation symmetrically.
  """
  precision: int = 8
  unsigned_int_bounds: bool = False
  clipping_coeff: float = 1.0
  # TODO(jihwanlee): Define stats config for static quantization
  stats_config = None
  stop_scale_gradient: bool = False
  fp16: bool = False
  per_channel: bool = True
  symmetric: bool = True


@dataclasses.dataclass
class WeightQuantizationParams:
  """Parameters for weight quantization.

  precision: The precision (number of bits) for weight quantization.
  unsigned_int_bounds: Whether or not to use unsigned_int_bounds.
  clipping_coeff: The coefficient to shrink the hard range for weight
    quantization. 1.0 means using hard min/max.
  stop_scale_gradient: Stop the gradient of the quantization scale for numerical
    stability. Note: this is numerically incorrect.
    Note stop_scale_gradient is also used by VN.
  min_clipping: Clipping value which will be used for clipping optimization
    in range [min_clipping ... 1].
  num_optimize_clipping: Number of optimization steps used for
    scale estimation with search over clipping values in
    range [min_clipping ... 1].
  use_symmetric: Do symmetric quantization for weights.
  add_scale_eps: If True add epsilon to scale to avoid division by zero,
    else it will replace zero scale by 1.
  dequant_upfront: Dequantize weights before it goes into matmul.
  dtype: The datatype for weight quantization. Defaults to int8.
  quant_loss_weight: Weight for quantization loss.
  optimize_clipping_per_channel: If True choose the best clipping value
    per channel, else per-tensor. It only works when min_clipping
    and num_optimize_clipping are set.
  sub_channels: Number of sub channels for splitting channelwise quantization.
  calculation_dtype: The type used for calculation around quantization.
  use_step_count: If True step_count non-trainable variable will added.
    It is used for counting forward propagation training steps.
    By default it is disabled for backward compatibility with prod.
  use_int4_packed_weights: If True, pack/unpack int4 weights into int32 or int8.
    It is for int4 weights only and has not effect on other type.
    If False int4 weights will be kept in int8.
    There are several edge cases:
      1. use_int4_packed_weights=True, precision=4, dtype=jnp.int8
        It keeps int4 values in int8 type and packs it into int8 or int32
        depending on int4_packed_weights_container_dtype.
      2. use_int4_packed_weights=False, precision=4, dtype=jnp.int8
        It keeps int4 values in int8 type.
      3. use_int4_packed_weights=False, precision=4, dtype=jnp.int4
        It will use native jnp.int4 type.
      4. use_int4_packed_weights=True, precision=4, dtype=jnp.int4
        it will raise an error.
  int4_packed_weights_container_dtype: Container type for int4 weights:
    int32 to pack 8 int4s, or int8 to pack 2 int4s.
  vn_scale: Scale coefficient for VN quantization. TODO(rybakov) use bits.
  vn_start_step: Step number after which VN is applied. If training step is less
    than vn_start_step then standard float model is trained.
  vn_noise_type: Noise type, can be 'uniform' of 'normal'.
    else use noise with normal distribution.
  vn_weight_norm_type: Type of weight normalization: 'L2', 'Linf',
    'PerChannelLinf'. Default value is 'PerChannelLinf' it is a
      standard scale normalization used by QAT.
  kurt_loss_weight: Weight for Kurtosis loss.
  kurt: Kurtosis target. By default it is 1.8 (uniform distribution).
    It is based on paper: "Robust Quantization: One Model to Rule Them All".
  block_size: block size for sub channel quantization. 0 to set it off. Defaults
    to off.
  quant_method: Quantization method:
    * 'default' - extracts min and max for quantization scale estimation.
      It is well applied for int8, in4, int2 quantization.
    * 'bin' - binarization, where scale is defined by mean|w|.
    * 'bin_norm' - binarization with weight normalization.
  """
  precision: int = 8
  unsigned_int_bounds: bool = False
  clipping_coeff: float = 1.0
  stop_scale_gradient: bool = False
  min_clipping: float | None = None
  num_optimize_clipping: int | None = None
  use_symmetric: bool = True
  add_scale_eps: bool | None = True
  dequant_upfront: bool = False
  dtype: jnp.dtype = jnp.int8
  quant_loss_weight: float | None = None
  optimize_clipping_per_channel: bool = False
  sub_channels: int | None = None
  calculation_dtype: jnp.dtype = jnp.float32
  use_step_count: bool = False
  use_int4_packed_weights: bool = True
  int4_packed_weights_container_dtype: jnp.dtype = jnp.int32
  vn_scale: float | None = None
  vn_start_step: int = 0
  vn_noise_type: str = 'uniform'
  vn_weight_norm_type: str = 'PerChannelLinf'
  kurt_loss_weight: float | None = None
  kurt: float = 1.8
  block_size: int = 0
  # Internal quantization parameters.
  quant_method: str = 'default'


@dataclasses.dataclass
class QuantizationParams:
  """Parameters for quantization.

  Attributes:
    quantization_type: Quantization type.
    mode: The quantization mode associated with this quantization parameter.
    act_params: Config for activation quantization.
    weight_params: Config for weight quantization.
  """
  quantization_type: QuantizationType = QuantizationType.PTQ
  mode: QuantizationMode = QuantizationMode.INFERENCE
  act_params: ActQuantizationParams | None = None
  weight_params: WeightQuantizationParams = dataclasses.field(
      default_factory=WeightQuantizationParams
  )
