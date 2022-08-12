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

"""Gradient checkpoint policies that are supported by the `checkpoint` transform."""

import enum
import jax


@enum.unique
class AutodiffCheckpointType(str, enum.Enum):
  """jax.checkpoint policy types."""
  SAVE_NOTHING = 'save_nothing'
  SAVE_EVERYTHING = 'save_everything'
  SAVE_QKV_OUT_PROJ = 'save_qkv_out_proj'
  SAVE_OUT_PROJ = 'save_out_proj'
  SAVE_CONTEXT = 'save_context'
  SAVE_CONTEXT_AND_OUT_PROJ = 'save_encoded_and_out_proj'
  SAVE_DOT_ONLY = 'save_dot_only'
  SAVE_DOT_WITH_NO_BATCH_DIM = 'save_dot_with_no_batch_dims'
  SAVE_DOT_FOR_MLPERF_200B = 'save_dot_for_mlperf_200b'
  SAVE_ITERATION_INPUT = 'save_iteration_input'


def custom_policy(checkpoint_policy: AutodiffCheckpointType):
  """Returns a JAX Autodiff checkpointing policy from the enum value."""
  # TODO(zhangqiaorjc): Configure custom checkpoint policy in expt config
  # without introducing enum.
  if checkpoint_policy == AutodiffCheckpointType.SAVE_EVERYTHING:
    return jax.checkpoint_policies.everything_saveable
  if checkpoint_policy == AutodiffCheckpointType.SAVE_DOT_ONLY:
    return jax.checkpoint_policies.checkpoint_dots
  if checkpoint_policy == AutodiffCheckpointType.SAVE_DOT_WITH_NO_BATCH_DIM:
    return jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
  if checkpoint_policy == AutodiffCheckpointType.SAVE_QKV_OUT_PROJ:
    return jax.checkpoint_policies.save_only_these_names(
        'combined_qkv_proj', 'out_proj')
  if checkpoint_policy == AutodiffCheckpointType.SAVE_CONTEXT:
    return jax.checkpoint_policies.save_only_these_names('context')
  if checkpoint_policy == AutodiffCheckpointType.SAVE_OUT_PROJ:
    return jax.checkpoint_policies.save_only_these_names('out_proj')
  if checkpoint_policy == AutodiffCheckpointType.SAVE_CONTEXT_AND_OUT_PROJ:
    return jax.checkpoint_policies.save_only_these_names('context', 'out_proj')
  if checkpoint_policy == AutodiffCheckpointType.SAVE_DOT_FOR_MLPERF_200B:
    return jax.checkpoint_policies.save_only_these_names(
        'combined_qkv_proj', 'query_proj', 'value_proj', 'key_proj', 'context',
        'out_proj')
  if checkpoint_policy == AutodiffCheckpointType.SAVE_ITERATION_INPUT:
    return jax.checkpoint_policies.save_only_these_names('iteration_input')
  assert checkpoint_policy == AutodiffCheckpointType.SAVE_NOTHING
  return jax.checkpoint_policies.nothing_saveable
