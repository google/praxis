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

"""Layers with overflow checks."""

import dataclasses
from typing import Any

from absl import logging
import jax
from jax import numpy as jnp
from praxis import base_layer
from praxis import pax_fiddle
from praxis import pytypes
from praxis.layers import attentions
from praxis.layers import linears
from praxis.layers import multi_query_attention

instance_field = base_layer.instance_field
JTensor = pytypes.JTensor


@dataclasses.dataclass
class OverflowLimits:
  """Overflow limits."""

  max_val: float = jnp.finfo(jnp.float16).max
  min_val: float = jnp.finfo(jnp.float16).min


class OverflowChecker(base_layer.BaseLayer):
  """Adds overflow checks."""

  overflow_limits: OverflowLimits = instance_field(OverflowLimits)

  def check_overflow(self, inputs, class_name, name):
    max_val = jnp.max(inputs)
    min_val = jnp.min(inputs)
    overflow = jnp.logical_or(
        jnp.greater(max_val, self.overflow_limits.max_val),
        jnp.less(min_val, self.overflow_limits.min_val),
    )

    def log_overflow(max_val, min_val):
      logging.warning(  # pylint: disable=logging-fstring-interpolation
          f"Overflow in: {class_name}, {name}, max_val: {max_val}, min_val:"
          f" {min_val}"
      )

    def _true_fn(max_val, min_val):
      jax.debug.callback(log_overflow, max_val, min_val)

    def _false_fn(max_val, min_val):
      del max_val, min_val
      pass

    jax.lax.cond(overflow, _true_fn, _false_fn, max_val, min_val)


class FeedForwardOverflowCheck(linears.FeedForward, OverflowChecker):
  """Linear layer with overflow checks."""

  def __call__(
      self,
      inputs: JTensor,
  ) -> JTensor:
    out = super().__call__(inputs)
    self.check_overflow(out, "FeedForward", self.name)
    return out


class AttentionProjectionOverflowCheck(
    attentions.AttentionProjection, OverflowChecker
):
  """Attention projection layer with overflow checks."""

  def __call__(
      self,
      inputs: JTensor,
  ) -> JTensor:
    out = super().__call__(inputs)
    self.check_overflow(out, "AttentionProjection", self.name)
    return out

  def extend_step(self, inputs: JTensor, *, time_step: JTensor) -> JTensor:
    del time_step  # Not used.
    return self.__call__(inputs)


class CombinedQKVProjectionLayerOverflowCheck(
    attentions.CombinedQKVProjectionLayer, OverflowChecker
):
  """Combined QKV projection layer with overflow checks."""

  def __call__(
      self,
      inputs: JTensor,
  ) -> tuple[JTensor, JTensor, JTensor]:
    q, k, v = super().__call__(inputs)
    self.check_overflow(q, "CombinedQKVProjection q", self.name)
    self.check_overflow(k, "CombinedQKVProjection k", self.name)
    self.check_overflow(v, "CombinedQKVProjection v", self.name)
    return q, k, v

  def extend_step(self, inputs: JTensor, *, time_step: JTensor) -> JTensor:
    del time_step  # Not used.
    return self.__call__(inputs)  # pytype: disable=bad-return-type  # jax-ndarray


class OneHeadedAttentionProjectionOverflowCheck(
    multi_query_attention.OneHeadedAttentionProjection, OverflowChecker
):
  """Attention projection layer with overflow checks."""

  def __call__(
      self,
      inputs: JTensor,
  ) -> JTensor:
    out = super().__call__(inputs)
    self.check_overflow(out, "OneHeadedAttentionProjection", self.name)
    return out


def add_overflow_checks(
    task_p: Any,
    max_val: float = jnp.finfo(jnp.float16).max,
    min_val: float = jnp.finfo(jnp.float16).min,
) -> Any:
  """Adds overflow checks to the task."""
  overwrite_task_p = task_p.clone()

  transformer_layer_p = (
      overwrite_task_p.model.lm_tpl.stacked_transformer_tpl.block.transformer_layer_params_tpl
  )

  overflow_check_tpl = OverflowLimits(max_val=max_val, min_val=min_val)
  feed_forward = pax_fiddle.Config(
      FeedForwardOverflowCheck, overflow_limits=overflow_check_tpl
  )
  transformer_layer_p.tr_fflayer_tpl.fflayer_tpl = feed_forward

  proj_tpl = pax_fiddle.Config(
      AttentionProjectionOverflowCheck,
      overflow_limits=overflow_check_tpl,
  )
  transformer_layer_p.tr_atten_tpl.proj_tpl = proj_tpl

  if issubclass(
      transformer_layer_p.tr_atten_tpl.cls,
      multi_query_attention.MultiQueryDotProductAttention,
  ):
    headless_proj_tpl = pax_fiddle.Config(
        OneHeadedAttentionProjectionOverflowCheck,
        overflow_limits=overflow_check_tpl,
    )
    transformer_layer_p.tr_atten_tpl.headless_proj_tpl = headless_proj_tpl

  if issubclass(
      transformer_layer_p.tr_atten_tpl.cls, attentions.DotProductAttention
  ):
    combined_qkv_proj_tpl = pax_fiddle.Config(
        CombinedQKVProjectionLayerOverflowCheck,
        overflow_limits=overflow_check_tpl,
    )
    transformer_layer_p.tr_atten_tpl.combined_qkv_proj_tpl = (
        combined_qkv_proj_tpl
    )

  return overwrite_task_p
