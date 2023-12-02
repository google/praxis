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

"""GroupedQueryAttention layer test."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental import pjit
import jax.numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import pax_fiddle as fdl
from praxis import py_utils
from praxis import test_utils
from praxis.layers import grouped_query_attention

P = jax.sharding.PartitionSpec


class AttentionTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  @parameterized.parameters(
      (8, 1, 4, 2, False),
      (8, 1, 4, 2, True),
      (8, 2, 4, 2, False),
      (8, 2, 6, 2, True),
      (8, 4, 4, 2, False),
      (8, 6, 6, 2, True),
      (8, 1, 6, 2, True),
  )
  def test_fprop_extend_step_consistency(self, d, k, n, h, use_rope):
    if use_rope:
      rope_min_max_timescales = (1, 10_000)
    else:
      rope_min_max_timescales = None
    p = fdl.Config(
        grouped_query_attention.GroupedQueryAttention,
        name='attention',
        input_dim=d,
        num_heads=n,
        num_kv_heads=k,
        dim_per_head=h,
        rope_min_max_timescales=rope_min_max_timescales,
    )
    layer = p.Instantiate()
    b = 2
    l = 8
    inputs = np.random.normal(1.0, 0.5, (b, l, d)).astype('float32')
    mask = grouped_query_attention.causal_mask(l)
    prng_key = jax.random.PRNGKey(seed=123)
    positions = jnp.arange(l, dtype=jnp.int32)[None, :]
    initial_vars = layer.init(
        prng_key, inputs, inputs, inputs, mask, positions, positions
    )
    outputs, attention_states = layer.apply(
        initial_vars,
        inputs,
        inputs,
        inputs,
        mask,
        positions,
        positions,
        mutable=[base_layer.DECODE_CACHE],
    )
    decoder_output = np.zeros(shape=[l, b, d])
    updated_vars = py_utils.merge_dict(attention_states, initial_vars)
    starting_index = 0
    for t in range(starting_index, l):
      encoded, attention_states = layer.apply(
          updated_vars,
          query_vec=inputs[:, t, :],
          atten_mask=mask[:, :, t, :],
          # Needs to be a Jax scalar because loop unrolling occurs on-device.
          time_step=jnp.asarray(t),
          segment_pos=None,
          method=layer.extend_step,
          mutable=[base_layer.DECODE_CACHE],
      )
      updated_vars = py_utils.merge_dict(attention_states, initial_vars)
      decoder_output[t] = encoded
    decoder_output = np.transpose(decoder_output[starting_index:], [1, 0, 2])
    fprop_out = outputs[0][:, starting_index:]
    self.assertAllClose(fprop_out, decoder_output)

  @parameterized.parameters(
      (8, 1, 4, 2, False, False),
      (8, 1, 4, 2, True, False),
      (8, 2, 4, 2, False, False),
      (8, 2, 6, 2, True, False),
      (8, 4, 4, 2, False, False),
      (8, 6, 6, 2, True, False),
      (8, 6, 6, 2, True, True),
  )
  def test_sharded_attention(
      self,
      d,
      k,
      n,
      h,
      use_rope,
      seq_shard,
  ):
    if use_rope:
      rope_min_max_timescales = (1, 10_000)
    else:
      rope_min_max_timescales = None
    dnh = (None, 'y', None)
    btd = (None, 'x', 'y') if seq_shard else ('x', None, 'y')
    btnh = (None, 'x', 'y', None) if seq_shard else ('x', None, 'y', None)
    if k == 1:
      dkh = (None, None, None)
      bskh = (None, 'x', None, None) if seq_shard else ('x', None, None, None)
    else:
      dkh = dnh
      bskh = btnh
    p = fdl.Config(
        grouped_query_attention.GroupedQueryAttention,
        name='attention',
        input_dim=d,
        num_heads=n,
        num_kv_heads=k,
        dim_per_head=h,
        rope_min_max_timescales=rope_min_max_timescales,
        weight_split_dims_mapping=fdl.Config(
            grouped_query_attention.GroupedQueryAttention.WeightSharding,
            dnh=dnh,
            dkh=dkh,
        ),
        activation_split_dims_mapping=fdl.Config(
            grouped_query_attention.GroupedQueryAttention.ActivationSharding,
            btd=btd,
            btnh=btnh,
            bskh=bskh,
        ),
    )
    layer = fdl.build(p)
    b = 2
    l = 8
    inputs = np.random.normal(1.0, 0.5, (b, l, d)).astype('float32')
    mask = grouped_query_attention.causal_mask(l)
    prng_key = jax.random.PRNGKey(seed=123)
    positions = jnp.arange(l, dtype=jnp.int32)[None, :]
    initial_vars = layer.init(
        prng_key, inputs, inputs, inputs, mask, positions, positions
    )
    outputs = layer.apply(
        initial_vars,
        inputs,
        inputs,
        inputs,
        mask,
        positions,
        positions,
    )
    var_hparams = layer.abstract_init_with_metadata(
        inputs, inputs, inputs, mask, positions, positions
    )
    var_pspecs = base_layer.var_partition_specs(var_hparams, (2, 2), ('x', 'y'))

    @functools.partial(
        pjit.pjit,
        in_shardings=(var_pspecs, P(*btd), None, None),
        out_shardings=None,
    )
    def sharded_step(variables, inputs, mask, positions):
      return layer.apply(
          variables,
          inputs,
          inputs,
          inputs,
          mask,
          positions,
          positions,
      )

    devices = np.array(jax.devices()).reshape((2, 2))
    mesh = jax.sharding.Mesh(devices, axis_names=('x', 'y'))
    with mesh:
      sharded_outputs = sharded_step(initial_vars, inputs, mask, positions)

    self.assertAllClose(outputs[0], sharded_outputs[0])


if __name__ == '__main__':
  absltest.main()
