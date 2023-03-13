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

"""Tests for quantized OneHeadedAttentionProjection layer."""
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import pax_fiddle
from praxis.layers import multi_query_attention
from praxis.layers.quantization import multi_query_attention as qmulti_query_attention
from praxis.layers.quantization import quantization_hparams
from praxis.layers.quantization.tests import test_util as quantization_test_util

QuantizationHParams = quantization_hparams.QuantizationHParams
QuantizationMode = quantization_hparams.QuantizationMode
QuantizationType = quantization_hparams.QuantizationType
instantiate = base_layer.instantiate
to_list = quantization_test_util.to_list
generate_quantization_test_config = (
    quantization_test_util.generate_one_headed_attention_projection_test_config
)


def _add_expected_training_results(cur_key, cur_samples):
  sample_input = [
      3.25011,
      1.68019,
      1.35529,
      -1.79903,
      -0.53809,
      -0.76695,
      0.27459,
      1.06138,
      0.41519,
      0.31456,
      -1.88078,
      -1.13559,
      -0.89631,
      -0.03403,
      -0.76315,
      0.17997,
      1.33960,
      0.60980,
  ]
  sample_init_weight = [
      -1.94768,
      0.04102,
      -1.95673,
      0.02697,
      0.91925,
      0.96507,
      0.35018,
      0.87926,
      0.22423,
      0.56189,
      0.28390,
      2.62069,
  ]
  pseudo_answer = [
      0.07560,
      -1.28025,
      -0.40961,
      0.46229,
      1.03244,
      2.05577,
  ]
  expected_trained_weights = [
      [
          [-1.96978, 0.02787],
          [-1.97054, 0.01343],
          [0.90658, 0.96003],
          [0.35850, 0.88578],
          [0.24302, 0.56728],
          [0.29646, 2.62718],
      ],
      [
          [-1.96978, 0.02787],
          [-1.97059, 0.01339],
          [0.90656, 0.96003],
          [0.35850, 0.88578],
          [0.24302, 0.56728],
          [0.29646, 2.62719],
      ],
      [
          [-1.96978, 0.02787],
          [-1.97054, 0.01343],
          [0.90658, 0.96003],
          [0.35850, 0.88578],
          [0.24302, 0.56728],
          [0.29646, 2.62718],
      ],
      [
          [-1.96978, 0.02787],
          [-1.97059, 0.01339],
          [0.90656, 0.96003],
          [0.35850, 0.88578],
          [0.24302, 0.56728],
          [0.29646, 2.62719],
      ],
  ]

  updated_key = cur_key + [
      'sample_input',
      'sample_init_weight',
      'pseudo_answer',
      'expected_trained_weight',
  ]
  ret = []
  for sample, expected_trained_weight in zip(
      cur_samples, expected_trained_weights
  ):
    sample.append(sample_input)
    sample.append(sample_init_weight)
    sample.append(pseudo_answer)
    sample.append(expected_trained_weight)

    ret.append(sample)

  return updated_key, ret


def _add_expected_quantized_weights(cur_key, cur_samples):
  target_weight = [
      -0.19159,
      0.78929,
      0.98149,
      -1.34752,
      0.29173,
      -1.16955,
      0.44791,
      2.10438,
      1.75480,
      0.11929,
      -0.12547,
      -0.77088,
      1.45661,
      0.94992,
      0.96217,
      1.13777,
      -1.51503,
      -0.26329,
  ]
  expected_weights = [
      [
          [-17, 48, 71],
          [-117, 18, -85],
          [39, 127, 127],
          [10, -8, -56],
          [127, 57, 70],
          [99, -91, -19],
      ],
      [
          [-23, 34, 60],
          [-128, -1, -128],
          [35, 127, 127],
          [5, -30, -93],
          [127, 46, 58],
          [98, -128, -49],
      ],
      [
          [-17, 48, 71],
          [-117, 18, -85],
          [39, 127, 127],
          [10, -8, -56],
          [127, 57, 70],
          [99, -91, -19],
      ],
      [
          [-23, 34, 60],
          [-128, -1, -128],
          [35, 127, 127],
          [5, -30, -93],
          [127, 46, 58],
          [98, -128, -49],
      ],
  ]
  expected_scales = [
      [0.01146, 0.01657, 0.01381],
      [0.01099, 0.01419, 0.01146],
      [0.01146, 0.01657, 0.01381],
      [0.01099, 0.01419, 0.01146],
  ]
  expected_zps = [
      None,
      [-0.06005, -0.30178, -0.29837],
      None,
      [-0.06005, -0.30178, -0.29837],
  ]

  updated_key = cur_key + [
      'target_weight',
      'expected_weight',
      'expected_scale',
      'expected_zp',
  ]
  ret = []
  for sample, expected_weight, expected_scale, expected_zp in zip(
      cur_samples, expected_weights, expected_scales, expected_zps
  ):
    sample.append(target_weight)
    sample.append(expected_weight)
    sample.append(expected_scale)
    sample.append(expected_zp)

    ret.append(sample)

  return updated_key, ret


def _add_expected_quantization_results(cur_key, cur_samples):
  sample_weight = [
      -0.65354,
      0.36260,
      1.96744,
      -0.07008,
      -1.33874,
      -0.98434,
      1.74267,
      -0.21951,
      0.75616,
      -0.08952,
      0.35862,
      -0.49218,
  ]
  sample_input = [
      1.90758,
      0.25165,
      -0.89873,
      1.04120,
      0.00274,
      0.25572,
      -0.90040,
      1.08367,
  ]
  expected_results = [
      [[-2.93991, 0.91891, 2.30518], [-1.69255, 0.24635, -1.46885]],
      [[-2.92416, 0.92125, 2.32290], [-1.68628, 0.24053, -1.45073]],
      [[-2.93991, 0.91891, 2.30518], [-1.69255, 0.24635, -1.46885]],
      [[-2.92416, 0.92125, 2.32290], [-1.68628, 0.24053, -1.45073]],
  ]
  updated_key = cur_key + ['sample_weight', 'sample_input', 'expected_result']
  ret = []
  for sample, expected_result in zip(cur_samples, expected_results):
    sample.append(sample_weight)
    sample.append(sample_input)
    sample.append(expected_result)

    ret.append(sample)

  return updated_key, ret


class OneHeadedAttentionProjectionAQTTest(
    quantization_test_util.QuantizationTestCase):
  """Test cases for QuantizationType.AQT.

  Following tests are required:
  1. Training test: The results compared to the non-quantized model should be
                    the same.
  2. Weight quantization test: The weights should be properly converted.
  3. Partition spec test.
  4. Inference test.
  """

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  def train_and_compare(
      self,
      p_f,
      p_q,
      inputs,
      sample_init_weight,
      pseudo_answer,
      expected_trained_weight,
  ):
    # Train the two models, and compare if the results are equal.
    step_size = 0.03
    one_headed_attn_f = instantiate(p_f)
    one_headed_attn_q = instantiate(p_q)

    def update(one_headed_attn_layer, params, inputs, targets):
      def loss(params, inputs, targets):
        outputs = one_headed_attn_layer.apply(params, inputs)
        return -jnp.mean(jnp.abs(outputs - targets))

      grads = jax.grad(loss)(params, inputs, targets)

      out_params = dict()
      out_params[base_layer.PARAMS] = dict()
      for k, v in params[base_layer.PARAMS].items():
        v_grad = grads[base_layer.PARAMS][k]
        out_params[base_layer.PARAMS][k] = v - step_size * v_grad

      return out_params

    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars_f = one_headed_attn_f.init(prng_key, inputs)
    initial_vars_q = one_headed_attn_q.init(prng_key, inputs)
    weight_shape_f = initial_vars_f[base_layer.PARAMS]['w'].shape
    weight_shape_q = initial_vars_q[base_layer.PARAMS]['w'].shape

    initial_vars_f[base_layer.PARAMS]['w'] = np.array(
        sample_init_weight
    ).reshape(weight_shape_f)
    initial_vars_q[base_layer.PARAMS]['w'] = np.array(
        sample_init_weight
    ).reshape(weight_shape_q)

    outputs_f = one_headed_attn_f.apply(initial_vars_f, inputs)
    pseudo_answer = np.array(pseudo_answer[: np.prod(outputs_f.shape)]).reshape(
        outputs_f.shape
    )

    updated_vars_f = update(
        one_headed_attn_f, initial_vars_f, inputs, pseudo_answer)
    updated_vars_q = update(
        one_headed_attn_q, initial_vars_q, inputs, pseudo_answer)

    # 1. Weight must not be the same, but they should be close.
    updated_w_f_tensor = updated_vars_f[base_layer.PARAMS]['w']
    updated_w_q_tensor = updated_vars_q[base_layer.PARAMS]['w']

    updated_w_f = to_list(updated_w_f_tensor)
    updated_w_q = to_list(updated_w_q_tensor)

    self.assertNotEqual(updated_w_f, updated_w_q)
    self.assertAllClose(updated_w_f_tensor, updated_w_q_tensor, atol=1e-4)

    # 2. Value check.
    self.assertNestedListClose(updated_w_q, expected_trained_weight)

  # Test the training with AQT quantization.
  @parameterized.parameters(
      generate_quantization_test_config([_add_expected_training_results])
  )
  def test_train(
      self,
      use_bias,
      is_weight_symmetric,
      sample_input,
      sample_init_weight,
      pseudo_answer,
      expected_trained_weight
  ):
    quantization_option = QuantizationHParams(
        quantization_type=QuantizationType.AQT,
        mode=QuantizationMode.TRAINING,
        weight_params=quantization_hparams.WeightQuantizationParams(
            use_symmetric=is_weight_symmetric
        ),
    )
    p_f = pax_fiddle.Config(
        multi_query_attention.OneHeadedAttentionProjection,
        name='_one_headed_attn_proj_f',
        use_bias=use_bias,
    )
    p_q = pax_fiddle.Config(
        qmulti_query_attention.OneHeadedAttentionProjection,
        name='_one_headed_attn_proj_q',
        use_bias=use_bias,
        quantization=quantization_option,
    )
    for p in [p_f, p_q]:
      p.input_dim = 6
      p.output_dim = 2

    inputs = np.array(sample_input).astype(np.float32).reshape([3, 6])
    self.train_and_compare(
        p_f,
        p_q,
        inputs,
        sample_init_weight,
        pseudo_answer,
        expected_trained_weight,
    )

  # Test AQT weight quantization.
  @parameterized.parameters(
      generate_quantization_test_config([_add_expected_quantized_weights])
  )
  def test_weight_quantization(
      self,
      use_bias,
      is_weight_symmetric,
      target_weight,
      expected_weight,
      expected_scale,
      expected_zp
  ):
    quantization_option = QuantizationHParams(
        quantization_type=QuantizationType.AQT,
        mode=QuantizationMode.MATERIALIZE,
        weight_params=quantization_hparams.WeightQuantizationParams(
            use_symmetric=is_weight_symmetric
        ),
    )

    p = pax_fiddle.Config(
        qmulti_query_attention.OneHeadedAttentionProjection,
        name='_one_headed_attn_proj_q',
        mesh_axis_names=['replica', 'mdl', 'data'],
        weight_split_dims_mapping=base_layer.BaseLayer.WeightSharding(
            wt=['mdl', 'data']
        ),
        use_bias=use_bias,
        quantization=quantization_option,
    )
    p.input_dim = 6
    p.output_dim = 3

    layer = instantiate(p)
    inputs = np.random.normal(1.5, 2.0, [3, 6]).astype(np.float32)

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = layer.init(prng_key, inputs)
      weight_shape = initial_vars[base_layer.PARAMS]['w'].shape
      initial_vars[base_layer.PARAMS]['w'] = np.array(target_weight).reshape(
          weight_shape
      )

      res, _ = layer.apply(
          initial_vars, mutable=[], method=layer.quantize_weight
      )

    expected_params = ['w', 'w_quantized_scale']
    if not quantization_option.weight_params.use_symmetric:
      expected_params.append('w_quantized_zp')

    # Check quantized parameters values.
    weight = res[base_layer.PARAMS].get('w', None)
    weight_scale = res[base_layer.PARAMS].get('w_quantized_scale', None)
    weight_zp = res[base_layer.PARAMS].get('w_quantized_zp', None)

    self.assertNestedListClose(to_list(weight), expected_weight)
    self.assertNestedListClose(to_list(weight_scale), expected_scale)
    self.assertNestedListClose(to_list(weight_zp), expected_zp)

  # Check Q specification.
  @parameterized.parameters(generate_quantization_test_config())
  def test_quantization_partition_spec(
      self,
      use_bias,
      is_weight_symmetric
  ):
    quantization_option = QuantizationHParams(
        quantization_type=QuantizationType.AQT,
        mode=QuantizationMode.MATERIALIZE,
        weight_params=quantization_hparams.WeightQuantizationParams(
            use_symmetric=is_weight_symmetric
        ),
    )

    p = pax_fiddle.Config(
        qmulti_query_attention.OneHeadedAttentionProjection,
        name='_one_headed_attn_proj_q',
        mesh_axis_names=['replica', 'mdl', 'data'],
        weight_split_dims_mapping=base_layer.BaseLayer.WeightSharding(
            wt=['replica', 'mdl', 'data']
        ),
        use_bias=use_bias,
        quantization=quantization_option,
    )
    p.input_dim = 8
    p.output_dim = 3

    layer = instantiate(p)
    inputs = np.random.normal(1.5, 2.0, [4, 8]).astype(np.float32)

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = layer.init(prng_key, inputs)

    pspec, _ = layer.apply(
        initial_vars, mutable=[], method=layer.quantized_partition_specs
    )

    expected_pspec = {
        'params': {
            'w': base_layer.BoxedPartitionSpec(
                meta=jax.sharding.PartitionSpec('replica', 'mdl', 'data')
            ),
            'w_quantized_scale': base_layer.BoxedPartitionSpec(
                meta=jax.sharding.PartitionSpec('mdl',)
            ),
        }
    }
    if not is_weight_symmetric:
      expected_pspec['params']['w_quantized_zp'] = (
          base_layer.BoxedPartitionSpec(
              meta=jax.sharding.PartitionSpec('mdl',)
          )
      )

    self.assertEqual(pspec, expected_pspec)

  # Check inference result.
  @parameterized.parameters(
      generate_quantization_test_config([_add_expected_quantization_results])
  )
  def test_inference_call(
      self,
      use_bias,
      is_weight_symmetric,
      sample_weight,
      sample_input,
      expected_result
  ):
    quantization_option = QuantizationHParams(
        quantization_type=QuantizationType.AQT,
        mode=QuantizationMode.INFERENCE,
        weight_params=quantization_hparams.WeightQuantizationParams(
            use_symmetric=is_weight_symmetric
        ),
    )

    p_f = pax_fiddle.Config(
        multi_query_attention.OneHeadedAttentionProjection,
        name='_one_headed_attn_proj_f',
        use_bias=use_bias,
    )

    p_q = pax_fiddle.Config(
        qmulti_query_attention.OneHeadedAttentionProjection,
        name='_one_headed_attn_proj_q',
        use_bias=use_bias,
        quantization=quantization_option,
    )

    for p in [p_f, p_q]:
      p.input_dim = 4
      p.output_dim = 3

    layer_f = instantiate(p_f)
    layer_q = instantiate(p_q)
    inputs = np.array(sample_input).astype(np.float32)
    inputs = inputs.reshape([2, 4])

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars_f = layer_f.init(prng_key, inputs)
      initial_vars_q = layer_q.init(prng_key, inputs)
      weight_shape_f = initial_vars_f[base_layer.PARAMS]['w'].shape
      weight_shape_q = initial_vars_q[base_layer.PARAMS]['w'].shape
      sample_weight = sample_weight[:np.prod(weight_shape_f)]
      initial_vars_f[base_layer.PARAMS]['w'] = np.array(sample_weight).reshape(
          weight_shape_f
      )
      initial_vars_q[base_layer.PARAMS]['w'] = np.array(sample_weight).reshape(
          weight_shape_q
      )
      quantized_vars, _ = layer_q.apply(
          initial_vars_q, mutable=[], method=layer_q.quantize_weight
      )
      for k, v in quantized_vars[base_layer.PARAMS].items():
        initial_vars_q[base_layer.PARAMS][k] = v

      res_f = layer_f.apply(initial_vars_f, inputs)
      res_q = layer_q.apply(initial_vars_q, inputs)

    # Since they are quantized results, they may not be exactly equal,
    # but they should be similar in some way.
    result_f = to_list(res_f)
    result_q = to_list(res_q)

    self.assertNotEqual(result_f, result_q)
    self.assertAllClose(res_f, res_q, atol=1e-1)

    self.assertNestedListClose(result_q, expected_result)

if __name__ == '__main__':
  absltest.main()
