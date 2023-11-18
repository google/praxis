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

"""Chunk functions."""

import jax
from jax import numpy as jnp
from praxis import pytypes

JTensor = pytypes.JTensor


def chunk(x: JTensor, *, chunk_size: int, axis: int = 1) -> JTensor:
  """Splits the x tensor into chunks with `chunk_size`.

  Args:
    x: JTensor of shape [..., L, ...].
    chunk_size: int, the chunk size.
    axis: axis for chunking.

  Returns:
    JTensor of shape [L/C, ..., C, ...].
  """
  # Maybe pad paddings to multiple of chunk_size.
  extra_pad = -x.shape[axis] % chunk_size
  if extra_pad:
    # Use default pad value unless specified.
    x = jnp.pad(x, [[0, extra_pad if i == axis else 0] for i in range(x.ndim)])

  x_shape = jnp.shape(x)
  num_chunks = x_shape[axis] // chunk_size
  # Reshape to [..., L/C, C, ...].
  new_shape = x_shape[:axis] + (num_chunks, chunk_size) + x_shape[axis + 1 :]
  chunk_x = jnp.reshape(x, new_shape)
  # [L/C, ..., C, ...]
  return jnp.moveaxis(chunk_x, axis, 0)


def unchunk(
    chunk_x: JTensor, axis: int = 1, seqlen: int | None = None
) -> JTensor:
  """Reshapes the x tensor from chunks to the original shape.

  Args:
    chunk_x: JTensor of shape [L/C, ..., C, ...].
    axis: axis for chunking.
    seqlen: int, the original sequence length.

  Returns:
    JTensor of shape [..., L, ...].
  """
  chunk_size = chunk_x.shape[axis + 1]
  # [..., L/C, C, ...]
  chunk_x = jnp.moveaxis(chunk_x, 0, axis)
  # [..., L, ...]
  new_shape = chunk_x.shape[:axis] + (-1,) + chunk_x.shape[axis + 2 :]
  x = jnp.reshape(chunk_x, new_shape)
  if seqlen is not None and seqlen % chunk_size:
    x = jax.lax.slice_in_dim(x, 0, seqlen, axis=axis)
  return x
