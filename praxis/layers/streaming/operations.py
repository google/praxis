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

"""Utils for streaming aware layers."""

from typing import List

from jax import numpy as jnp
from praxis import base_layer
from praxis import py_utils
from praxis import pytypes
from praxis.layers.streaming.streaming_base import StreamingBase

NestedMap = py_utils.NestedMap
NestedJTensor = pytypes.NestedJTensor
JTensor = pytypes.JTensor


def run_streaming(layer: StreamingBase,
                  initial_vars: NestedJTensor,
                  inputs: NestedJTensor,
                  output_names: List[str],
                  step: int) -> JTensor:
  """Runs layer in streaming mode.

  It will concatenate all streaming output packets and return non streaming
  output sequence. It is convenient for numerical validation.
  This function assumes that all features of "inputs"
  will be used as streaming inputs and all of them have shape [batch, time,...].

  Args:
    layer: Streaming aware layer.
    initial_vars: Initial layer variables.
    inputs: NestedMap with features of shape [Batch, Time,...] and
      paddings with shape [Batch, Time] or it can be None.
    output_names: List of output names which will be returned as
      streaming outputs.
    step: Defines how many samples to process per streaming step.
      It also defines how many steps to split the input audio = time_size/step.

  Returns:
    NestedMap with outputs.
  """

  batch_size = None
  time_size = None
  for key in inputs:
    if inputs[key] is not None:
      if batch_size is not None and batch_size != inputs[key].shape[0]:
        raise ValueError('batch_size has to be the same in all inputs')
      batch_size = inputs[key].shape[0]

      # Time dimension is the second dim: [batch, time, ...]
      if time_size is not None and time_size != inputs[key].shape[1]:
        raise ValueError('time_size has to be the same in all inputs')
      time_size = inputs[key].shape[1]
      if time_size % step:
        raise ValueError(f'Input time dimension: {inputs[key].shape[1]} '
                         f'of feature: {key}'
                         f'is not aligned with streaming step: {step}')

  with_paddings = 'paddings' in inputs and inputs.paddings is not None

  outputs = NestedMap((name, None) for name in output_names)

  context_p = base_layer.JaxContext.HParams(do_eval=True)
  with base_layer.JaxContext.new_context(hparams=context_p):
    # Get streaming states.
    _, streaming_states = layer.apply(
        initial_vars,
        batch_size=batch_size,
        with_paddings=with_paddings,
        method=layer.init_states,
        mutable=[base_layer.DECODE_CACHE])

    for i in range(0, time_size, step):
      # Prepare streaming input.
      in_nmap = NestedMap()
      for key in inputs:
        if inputs[key] is None:
          in_nmap[key] = None
        else:
          # Streaming dimension is time dim: [batch, time, ...]
          in_nmap[key] = inputs[key][:, i:i + step]

      # Combine streaming state with model vars and run one streaming step:
      updated_vars = py_utils.MergeDictsWithValueCheck(
          streaming_states, initial_vars)
      output_step, streaming_states = layer.apply(
          updated_vars,
          in_nmap,
          method=layer.streaming_step,
          mutable=[base_layer.DECODE_CACHE])

      # Concatenate streaming output with the final non streaming output.
      for key in outputs:
        if key not in output_step:
          raise ValueError(f'Output name: {key} is missing in '
                           f'streaming outputs: {output_step.keys()}')
        if outputs[key] is None:
          outputs[key] = output_step[key]
        else:
          outputs[key] = jnp.concatenate([outputs[key], output_step[key]],
                                         axis=1)
    return outputs
