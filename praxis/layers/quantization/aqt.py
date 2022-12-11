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

"""Quantization Aware Training ops."""

from typing import Optional

import jax
import jax.numpy as jnp
from praxis import base_layer
from praxis import pytypes

WeightHParams = base_layer.WeightHParams
JTensor = pytypes.JTensor


def _pass_through(x: JTensor, fn) -> JTensor:
  # Create an exactly-zero expression with Sterbenz lemma that has an
  # exactly-one gradient.
  return x - jax.lax.stop_gradient(x) + jax.lax.stop_gradient(fn(x))


class TensorQuantizer(base_layer.BaseLayer):
  """Maintains state associated with the quantization of an input tensor.

  Attributes:
    precision: Number of bits to quantize to (e.g 4 for int4). Must be positive.
  """
  precision: Optional[int] = None

  def setup(self):
    assert (
        self.precision is None or self.precision <= 23
    ), 'Too many bits, float32 has less precision.'

  def __call__(self):
    # Since TensorQuantizer does nothing when initialized, __call__ is a no-op.
    pass

  def _get_clip_bound(self) -> float:
    bucket_count = 2.0**self.precision
    bucket_count -= 1.0
    return bucket_count / 2.0

  def _safe_clip_bound(self) -> float:
    cb_unsafe = self._get_clip_bound()
    cb = cb_unsafe - 2.0 ** (-20 + self.precision)
    assert cb < cb_unsafe, 'Internal error, epsilon too small.'
    return cb

  def get_quant_scale(self,
                      sample,
                      contract_dims,
                      dtype=jnp.bfloat16) -> JTensor:
    if self.precision is None:
      return jnp.ones(shape=(1,) * sample.ndim, dtype=dtype)

    x_bound = jnp.max(
        jnp.abs(sample),
        axis=contract_dims,
        keepdims=True)
    clip_bound = self._get_clip_bound()
    scale = clip_bound / x_bound
    return scale.astype(dtype)

  def update(self, sample: JTensor):
    # This function is no-op for now. Once static quantization is supported,
    # statistics update will be performed through this function.
    pass

  def to_quant(self, x: JTensor, dtype=jnp.bfloat16):
    if self.precision is None:
      return x.astype(dtype)

    clip_bound = self._safe_clip_bound()
    x = jnp.clip(x, -clip_bound, clip_bound)
    x = _pass_through(x + 0.5, jnp.floor)

    return x.astype(dtype)
