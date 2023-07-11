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

"""Tests for overflow checking."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import layers
from praxis import pax_fiddle
from praxis import test_utils
from praxis.layers import multi_query_attention
from praxis.layers import quantization
from praxis.layers.quantization import overflow_check

instantiate = base_layer.instantiate


class CheckOverflowTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  def test_feed_forward_overflow_check(self):
    overflow_check_tpl = overflow_check.OverflowLimits(
        max_val=65500, min_val=-65000
    )
    p = pax_fiddle.Config(
        quantization.FeedForwardOverflowCheck,
        overflow_limits=overflow_check_tpl,
        input_dims=8,
        output_dims=4,
        name='feed_forward_1',
    )

    feed_forward_1 = instantiate(p)

    p = pax_fiddle.Config(
        quantization.FeedForwardOverflowCheck,
        overflow_limits=overflow_check_tpl,
        input_dims=8,
        output_dims=4,
        name='feed_forward_2',
    )

    feed_forward_2 = instantiate(p)

    inputs_small = jnp.array(
        [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]],
        dtype=p.dtype,
    )
    inputs_large = (
        jnp.array(
            [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]],
            dtype=p.dtype,
        )
        * 100_000
    )

    with self.assertLogs(level='WARNING') as log_output:
      with base_layer.JaxContext.new_context():
        prng_key = jax.random.PRNGKey(seed=123)
        initial_vars = feed_forward_1.init(prng_key, inputs_small)
        _ = feed_forward_1.apply(initial_vars, inputs_small)
        _ = feed_forward_1.apply(initial_vars, inputs_large)
        _ = feed_forward_2.apply(initial_vars, inputs_small)
        _ = feed_forward_2.apply(initial_vars, inputs_large)

    self.assertLen(log_output[0], 2)
    self.assertStartsWith(
        log_output[0][0].msg, 'Overflow in: FeedForward, feed_forward_1'
    )
    self.assertStartsWith(
        log_output[0][1].msg, 'Overflow in: FeedForward, feed_forward_2'
    )

  @parameterized.named_parameters(
      ('MHA', False),
      ('MQA', True),
  )
  def test_transformer_overflow_check(self, use_mqa):
    p = pax_fiddle.Config(
        layers.transformers.Transformer,
        name='jax_transformer_layer',
        input_dims=8,
        hidden_dims=4,
        num_heads=8,
    )
    if use_mqa:
      p.tr_atten_tpl = pax_fiddle.Config(
          multi_query_attention.MultiQueryDotProductAttention
      )
    p.tr_atten_tpl.combine_qkv = not use_mqa
    p.tr_fflayer_tpl.use_gated_activation = True

    overflow_check_tpl = overflow_check.OverflowLimits(max_val=-1, min_val=1)
    feed_forward = pax_fiddle.Config(
        quantization.FeedForwardOverflowCheck,
        overflow_limits=overflow_check_tpl,
    )
    p.tr_fflayer_tpl.fflayer_tpl = feed_forward

    proj_tpl = pax_fiddle.Config(
        quantization.AttentionProjectionOverflowCheck,
        overflow_limits=overflow_check_tpl,
    )
    p.tr_atten_tpl.proj_tpl = proj_tpl

    if not use_mqa:
      combined_qkv_proj_tpl = pax_fiddle.Config(
          quantization.CombinedQKVProjectionLayerOverflowCheck,
          overflow_limits=overflow_check_tpl,
      )
      p.tr_atten_tpl.combined_qkv_proj_tpl = combined_qkv_proj_tpl
    else:
      headless_proj_tpl = pax_fiddle.Config(
          quantization.OneHeadedAttentionProjectionOverflowCheck,
          overflow_limits=overflow_check_tpl,
      )
      p.tr_atten_tpl.headless_proj_tpl = headless_proj_tpl

    transformer = instantiate(p)

    inputs = jnp.expand_dims(
        jnp.array(
            [[1, 2, 3, 4, 5, 6, 7, 8]],
            dtype=p.dtype,
        ),
        -1,
    )
    paddings = jnp.array(
        [[0] * 8],
        dtype=p.dtype,
    )
    atten_mask = jnp.expand_dims(
        jnp.expand_dims(jnp.array([[False] * 8], dtype=p.dtype), 0), 0
    )

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = transformer.init(prng_key, inputs, paddings, atten_mask)
      with self.assertLogs(level='WARNING') as log_output:
        _ = transformer.apply(initial_vars, inputs, paddings, atten_mask)

    self.assertLen(log_output[0], 7)
    self.assertStartsWith(
        log_output[0][0].msg,
        'Overflow in: AttentionProjection, query'
        if use_mqa
        else 'Overflow in: CombinedQKVProjection q, combined_qkv',
    )
    self.assertStartsWith(
        log_output[0][1].msg,
        'Overflow in: OneHeadedAttentionProjection, key'
        if use_mqa
        else 'Overflow in: CombinedQKVProjection k, combined_qkv',
    )
    self.assertStartsWith(
        log_output[0][2].msg,
        'Overflow in: OneHeadedAttentionProjection, value'
        if use_mqa
        else 'Overflow in: CombinedQKVProjection v, combined_qkv',
    )
    self.assertStartsWith(
        log_output[0][3].msg, 'Overflow in: AttentionProjection, post'
    )
    self.assertStartsWith(
        log_output[0][4].msg, 'Overflow in: FeedForward, ffn_layer1_gate'
    )
    self.assertStartsWith(
        log_output[0][5].msg, 'Overflow in: FeedForward, ffn_layer1'
    )
    self.assertStartsWith(
        log_output[0][6].msg, 'Overflow in: FeedForward, ffn_layer2'
    )


if __name__ == '__main__':
  absltest.main()
