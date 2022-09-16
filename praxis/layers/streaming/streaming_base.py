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

"""Base class for streaming aware layers."""
from typing import Optional
from flax import linen as nn

from praxis import base_layer
from praxis import py_utils
from praxis import pytypes

NestedMap = py_utils.NestedMap
NestedJTensor = pytypes.NestedJTensor
JTensor = pytypes.JTensor


class StreamingBase(base_layer.BaseLayer):
  """Abstract class for streaming aware layer.

  All sublayers need to implement:
    get_stride(), init_states(), streaming_step(), get_right_context().
  Also need to override other methods if different from BaseLayer.
  """

  @classmethod
  def get_right_context(cls, hparams) -> int:
    # Returns the right context(number of samples in the future) in time dim.
    # Together with stride they define layer delay.
    raise NotImplementedError(cls)

  @property
  def right_context(self) -> int:
    return self.get_right_context(self.hparams)

  @classmethod
  def get_stride(cls, hparams) -> int:
    # Returns stride in time dimension.
    # It defines a minimum number of input samples to produce one output.
    raise NotImplementedError(cls)

  @property
  def stride(self) -> int:
    return self.get_stride(self.hparams)

  def init_states(self, batch_size: int, with_paddings: bool = True):
    """Creates streaming states in base_layer.DECODE_CACHE.

    It will use _update_streaming_state() for states creation.
    Args:
      batch_size: Defines batch size of streaming states.
      with_paddings: If True it creates streaming states
        for padding processing, else sets it to None (it can save some memory).
    """
    raise NotImplementedError()

  def streaming_step(
      self,
      inputs: NestedJTensor,
  ) -> NestedJTensor:
    raise NotImplementedError()

  @nn.nowrap
  def get_streaming_state(self, name: str) -> JTensor:
    return self.get_decode_state(name)

  @nn.nowrap
  def _update_streaming_state(self, name: str, value: JTensor) -> None:
    """Updates streaming state.

    This is a no-op in training.
    It is used for creating streaming initial state
    and for state update in streaming_step() function.
    Args:
      name: Variable name in DECODE_CACHE which will be updated with new value.
      value: New value which will be stored in DECODE_CACHE.
    """
    # Only update the state if it is in streaming mode.
    if not self.is_mutable_collection(base_layer.DECODE_CACHE):
      return
    self.update_decode_state(name, value)
