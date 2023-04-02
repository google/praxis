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

"""FQ Tests for quantized CombinedQKVProjection layer."""
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import pax_fiddle
from praxis.layers import attentions
from praxis.layers.quantization import attentions as qattentions
from praxis.layers.quantization import quantization_hparams
from praxis.layers.quantization.tests import test_util as quantization_test_util

QuantizationHParams = quantization_hparams.QuantizationHParams
QuantizationMode = quantization_hparams.QuantizationMode
QuantizationType = quantization_hparams.QuantizationType
instantiate = base_layer.instantiate
to_list = quantization_test_util.to_list
generate_quantization_test_config = (
    quantization_test_util.generate_combined_qkv_projection_test_config
)


def _add_expected_training_results(cur_key, cur_samples):
  sample_input = [
      -2.00876,
      0.73863,
      -0.72689,
      -0.62825,
      -0.92254,
      -0.85069,
      -2.18116,
      -1.4566,
      -2.15637,
      -1.64758,
      -0.15066,
      -2.97152,
  ]
  sample_init_weight = [
      -1.87005,
      -0.46876,
      -0.70401,
      -2.04259,
      0.44374,
      -0.88963,
      -1.61684,
      -0.99906,
      -1.9828,
      -1.21553,
      -1.24802,
      -1.55197,
      -0.58044,
      0.00342,
      -0.00905,
      0.50119,
      0.93848,
      0.2393,
      -0.05777,
      0.44242,
      -0.21066,
      -1.12345,
      0.28552,
      -1.3532,
      -4.35933,
      0.0171,
      -4.54504,
      -0.55903,
      -1.76804,
      -2.82254,
      -1.55866,
      -0.91616,
      0.05137,
      -1.85107,
      -0.89163,
      -1.57415,
      -1.32169,
      -0.35597,
      0.8457,
      -1.00789,
      -2.57792,
      -0.9877,
      -0.09236,
      -2.25711,
      0.28498,
      -2.85576,
      0.18087,
      -1.28404,
      -0.18007,
      -1.61576,
      0.98753,
      -2.69071,
      -1.52242,
      -1.31156,
      -1.7391,
      -2.49235,
      0.13903,
      -0.8255,
      -0.49299,
      -0.23885,
      -2.03163,
      -1.09647,
      -2.96585,
      -1.67375,
      -0.44178,
      -2.62476,
      -0.9364,
      -0.63199,
      0.41509,
      -1.95427,
      -0.11317,
      -1.0383,
  ]
  pseudo_answer_q = [
      -1.88934,
      -2.13782,
      -0.71917,
      -1.1674,
      0.79667,
      -1.44687,
      -1.66122,
      -0.80103,
      -1.56184,
      -0.35587,
      -0.17155,
      0.82231,
      -2.40681,
      0.41624,
      -2.61967,
      -2.42596,
      -1.34162,
      -1.30046,
  ]
  pseudo_answer_k = [
      -0.21392,
      -0.43585,
      0.39435,
      -0.30145,
      0.27945,
      -1.15479,
      -1.61763,
      -1.27135,
      0.17339,
      -0.0956,
      -2.80184,
      0.35549,
      -0.31812,
      0.127,
      -2.54108,
      -0.12027,
      -0.94138,
      -1.50289,
  ]
  pseudo_answer_v = [
      -0.97338,
      -1.31178,
      -1.75928,
      -1.84029,
      -0.80379,
      -1.27508,
      -0.83843,
      -0.46398,
      -0.18629,
      -1.44834,
      0.54771,
      -1.15359,
      -0.94106,
      -1.6752,
      0.38545,
      -0.79464,
      0.85592,
      -1.71085,
  ]
  expected_trained_weights = [
      [
          [
              [-1.8729, -0.47159, -0.70684, -2.04544, 0.44417, -0.89246],
              [-1.61782, -1.00007, -1.9838, -1.21651, -1.24889, -1.55295],
              [-0.58214, 0.00172, -0.01075, 0.49949, 0.94001, 0.2376],
              [-0.06058, 0.43961, -0.21347, -1.12626, 0.28503, -1.35601],
          ],
          [
              [-4.36216, 0.01427, -4.54791, -0.56186, -1.77087, -2.82539],
              [-1.55964, -0.91714, 0.05039, -1.85205, -0.89261, -1.57513],
              [-1.32339, -0.35767, 0.844, -1.00959, -2.57962, -0.9894],
              [-0.09517, -2.25992, 0.28217, -2.85857, 0.17806, -1.28685],
          ],
          [
              [-0.1829, -1.61859, 0.9871, -2.69356, -1.52523, -1.31439],
              [-1.74008, -2.49333, 0.13989, -0.82648, -0.49397, -0.23983],
              [-2.03333, -1.09817, -2.9674, -1.67545, -0.44348, -2.62646],
              [-0.93921, -0.6348, 0.41558, -1.95708, -0.11598, -1.04111],
          ],
      ],
      [
          [
              [-1.87284, -0.47159, -0.70684, -2.04544, 0.44417, -0.89246],
              [-1.61782, -1.00004, -1.98377, -1.21651, -1.24887, -1.55293],
              [-0.58214, 0.00172, -0.01071, 0.49951, 0.94001, 0.23762],
              [-0.06066, 0.43961, -0.21347, -1.12626, 0.28503, -1.35601],
          ],
          [
              [-4.36218, 0.01423, -4.54789, -0.5619, -1.77087, -2.82539],
              [-1.55964, -0.91714, 0.05039, -1.85205, -0.89261, -1.57513],
              [-1.32339, -0.35767, 0.84402, -1.00959, -2.57962, -0.9894],
              [-0.09515, -2.25992, 0.28217, -2.85853, 0.17806, -1.28685],
          ],
          [
              [-0.1829, -1.61859, 0.9871, -2.69358, -1.52523, -1.31439],
              [-1.74008, -2.49333, 0.13988, -0.82646, -0.49396, -0.23983],
              [-2.03333, -1.09817, -2.96738, -1.67545, -0.44348, -2.62646],
              [-0.93921, -0.6348, 0.41558, -1.95708, -0.11604, -1.04111],
          ],
      ],
      [
          [
              [[-1.8729, -0.47159, -0.70684], [-2.04544, 0.44417, -0.89246]],
              [[-1.61782, -1.00007, -1.9838], [-1.21651, -1.24889, -1.55295]],
              [[-0.58214, 0.00172, -0.01075], [0.49949, 0.94001, 0.2376]],
              [[-0.06058, 0.43961, -0.21347], [-1.12626, 0.28503, -1.35601]],
          ],
          [
              [[-4.36216, 0.01427, -4.54791], [-0.56186, -1.77087, -2.82539]],
              [[-1.55964, -0.91714, 0.05039], [-1.85205, -0.89261, -1.57513]],
              [[-1.32339, -0.35767, 0.844], [-1.00959, -2.57962, -0.9894]],
              [[-0.09517, -2.25992, 0.28217], [-2.85857, 0.17806, -1.28685]],
          ],
          [
              [[-0.1829, -1.61859, 0.9871], [-2.69356, -1.52523, -1.31439]],
              [[-1.74008, -2.49333, 0.13989], [-0.82648, -0.49397, -0.23983]],
              [[-2.03333, -1.09817, -2.9674], [-1.67545, -0.44348, -2.62646]],
              [[-0.93921, -0.6348, 0.41558], [-1.95708, -0.11598, -1.04111]],
          ],
      ],
      [
          [
              [[-1.87284, -0.47159, -0.70684], [-2.04544, 0.44417, -0.89246]],
              [[-1.61782, -1.00004, -1.98377], [-1.21651, -1.24887, -1.55293]],
              [[-0.58214, 0.00172, -0.01071], [0.49951, 0.94001, 0.23762]],
              [[-0.06066, 0.43961, -0.21347], [-1.12626, 0.28503, -1.35601]],
          ],
          [
              [[-4.36218, 0.01423, -4.54789], [-0.5619, -1.77087, -2.82539]],
              [[-1.55964, -0.91714, 0.05039], [-1.85205, -0.89261, -1.57513]],
              [[-1.32339, -0.35767, 0.84402], [-1.00959, -2.57962, -0.9894]],
              [[-0.09515, -2.25992, 0.28217], [-2.85853, 0.17806, -1.28685]],
          ],
          [
              [[-0.1829, -1.61859, 0.9871], [-2.69358, -1.52523, -1.31439]],
              [[-1.74008, -2.49333, 0.13988], [-0.82646, -0.49396, -0.23983]],
              [[-2.03333, -1.09817, -2.96738], [-1.67545, -0.44348, -2.62646]],
              [[-0.93921, -0.6348, 0.41558], [-1.95708, -0.11604, -1.04111]],
          ],
      ],
      [
          [
              [-1.8729, -0.47159, -0.70684, -2.04544, 0.44417, -0.89246],
              [-1.61782, -1.00007, -1.9838, -1.21651, -1.24889, -1.55295],
              [-0.58214, 0.00172, -0.01075, 0.49949, 0.94001, 0.2376],
              [-0.06058, 0.43961, -0.21347, -1.12626, 0.28503, -1.35601],
          ],
          [
              [-4.36216, 0.01427, -4.54791, -0.56186, -1.77087, -2.82539],
              [-1.55964, -0.91714, 0.05039, -1.85205, -0.89261, -1.57513],
              [-1.32339, -0.35767, 0.844, -1.00959, -2.57962, -0.9894],
              [-0.09517, -2.25992, 0.28217, -2.85857, 0.17806, -1.28685],
          ],
          [
              [-0.1829, -1.61859, 0.9871, -2.69356, -1.52523, -1.31439],
              [-1.74008, -2.49333, 0.13989, -0.82648, -0.49397, -0.23983],
              [-2.03333, -1.09817, -2.9674, -1.67545, -0.44348, -2.62646],
              [-0.93921, -0.6348, 0.41558, -1.95708, -0.11598, -1.04111],
          ],
      ],
      [
          [
              [-1.87284, -0.47159, -0.70684, -2.04544, 0.44417, -0.89246],
              [-1.61782, -1.00004, -1.98377, -1.21651, -1.24887, -1.55293],
              [-0.58214, 0.00172, -0.01071, 0.49951, 0.94001, 0.23762],
              [-0.06066, 0.43961, -0.21347, -1.12626, 0.28503, -1.35601],
          ],
          [
              [-4.36218, 0.01423, -4.54789, -0.5619, -1.77087, -2.82539],
              [-1.55964, -0.91714, 0.05039, -1.85205, -0.89261, -1.57513],
              [-1.32339, -0.35767, 0.84402, -1.00959, -2.57962, -0.9894],
              [-0.09515, -2.25992, 0.28217, -2.85853, 0.17806, -1.28685],
          ],
          [
              [-0.1829, -1.61859, 0.9871, -2.69358, -1.52523, -1.31439],
              [-1.74008, -2.49333, 0.13988, -0.82646, -0.49396, -0.23983],
              [-2.03333, -1.09817, -2.96738, -1.67545, -0.44348, -2.62646],
              [-0.93921, -0.6348, 0.41558, -1.95708, -0.11604, -1.04111],
          ],
      ],
      [
          [
              [[-1.8729, -0.47159, -0.70684], [-2.04544, 0.44417, -0.89246]],
              [[-1.61782, -1.00007, -1.9838], [-1.21651, -1.24889, -1.55295]],
              [[-0.58214, 0.00172, -0.01075], [0.49949, 0.94001, 0.2376]],
              [[-0.06058, 0.43961, -0.21347], [-1.12626, 0.28503, -1.35601]],
          ],
          [
              [[-4.36216, 0.01427, -4.54791], [-0.56186, -1.77087, -2.82539]],
              [[-1.55964, -0.91714, 0.05039], [-1.85205, -0.89261, -1.57513]],
              [[-1.32339, -0.35767, 0.844], [-1.00959, -2.57962, -0.9894]],
              [[-0.09517, -2.25992, 0.28217], [-2.85857, 0.17806, -1.28685]],
          ],
          [
              [[-0.1829, -1.61859, 0.9871], [-2.69356, -1.52523, -1.31439]],
              [[-1.74008, -2.49333, 0.13989], [-0.82648, -0.49397, -0.23983]],
              [[-2.03333, -1.09817, -2.9674], [-1.67545, -0.44348, -2.62646]],
              [[-0.93921, -0.6348, 0.41558], [-1.95708, -0.11598, -1.04111]],
          ],
      ],
      [
          [
              [[-1.87284, -0.47159, -0.70684], [-2.04544, 0.44417, -0.89246]],
              [[-1.61782, -1.00004, -1.98377], [-1.21651, -1.24887, -1.55293]],
              [[-0.58214, 0.00172, -0.01071], [0.49951, 0.94001, 0.23762]],
              [[-0.06066, 0.43961, -0.21347], [-1.12626, 0.28503, -1.35601]],
          ],
          [
              [[-4.36218, 0.01423, -4.54789], [-0.5619, -1.77087, -2.82539]],
              [[-1.55964, -0.91714, 0.05039], [-1.85205, -0.89261, -1.57513]],
              [[-1.32339, -0.35767, 0.84402], [-1.00959, -2.57962, -0.9894]],
              [[-0.09515, -2.25992, 0.28217], [-2.85853, 0.17806, -1.28685]],
          ],
          [
              [[-0.1829, -1.61859, 0.9871], [-2.69358, -1.52523, -1.31439]],
              [[-1.74008, -2.49333, 0.13988], [-0.82646, -0.49396, -0.23983]],
              [[-2.03333, -1.09817, -2.96738], [-1.67545, -0.44348, -2.62646]],
              [[-0.93921, -0.6348, 0.41558], [-1.95708, -0.11604, -1.04111]],
          ],
      ],
  ]

  updated_key = cur_key + [
      'sample_input',
      'sample_init_weight',
      'pseudo_answer_q',
      'pseudo_answer_k',
      'pseudo_answer_v',
      'expected_trained_weight',
  ]
  ret = []
  for sample, expected_trained_weight in zip(
      cur_samples, expected_trained_weights
  ):
    sample.append(sample_input)
    sample.append(sample_init_weight)
    sample.append(pseudo_answer_q)
    sample.append(pseudo_answer_k)
    sample.append(pseudo_answer_v)
    sample.append(expected_trained_weight)

    ret.append(sample)

  return updated_key, ret


class CombinedQKVProjectionFQTest(quantization_test_util.QuantizationTestCase):
  """Test cases for QuantizationType.FQ.

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
      ans_q,
      ans_k,
      ans_v,
      expected_trained_weight
  ):
    # Train the two models, and compare if the results are equal.
    step_size = 0.01
    combined_qkv_f = instantiate(p_f)
    combined_qkv_q = instantiate(p_q)

    def update(combined_qkv_layer, params, inputs, ans_q, ans_k, ans_v):
      def loss(params, inputs, ans_q, ans_k, ans_v):
        proj_q, proj_k, proj_v = combined_qkv_layer.apply(params, inputs)
        return (
            -jnp.mean(jnp.abs(proj_q - ans_q))
            -jnp.mean(jnp.abs(proj_k - ans_k))
            -jnp.mean(jnp.abs(proj_v - ans_v))
        )

      grads = jax.grad(loss)(params, inputs, ans_q, ans_k, ans_v)

      out_params = dict()
      out_params[base_layer.PARAMS] = dict()
      for k, v in params[base_layer.PARAMS].items():
        v_grad = grads[base_layer.PARAMS][k]
        out_params[base_layer.PARAMS][k] = v - step_size * v_grad

      return out_params

    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars_f = combined_qkv_f.init(prng_key, inputs)
    initial_vars_q = combined_qkv_q.init(prng_key, inputs)
    weight_shape_f = initial_vars_f[base_layer.PARAMS]['w'].shape
    weight_shape_q = initial_vars_q[base_layer.PARAMS]['w'].shape

    out_q_f, out_k_f, out_v_f = combined_qkv_f.apply(initial_vars_f, inputs)
    ans_q = np.array(ans_q).reshape(out_q_f.shape)
    ans_k = np.array(ans_k).reshape(out_k_f.shape)
    ans_v = np.array(ans_v).reshape(out_v_f.shape)

    initial_vars_f[base_layer.PARAMS]['w'] = np.array(
        sample_init_weight
    ).reshape(weight_shape_f)
    initial_vars_q[base_layer.PARAMS]['w'] = np.array(
        sample_init_weight
    ).reshape(weight_shape_q)

    updated_vars_f = update(
        combined_qkv_f, initial_vars_f, inputs, ans_q, ans_k, ans_v
    )
    updated_vars_q = update(
        combined_qkv_q, initial_vars_q, inputs, ans_q, ans_k, ans_v
    )

    # 1. Weight must not be the same, but they should be close.
    updated_w_f_tensor = updated_vars_f[base_layer.PARAMS]['w']
    updated_w_q_tensor = updated_vars_q[base_layer.PARAMS]['w']

    updated_w_f = to_list(updated_w_f_tensor)
    updated_w_q = to_list(updated_w_q_tensor)

    self.assertNotEqual(updated_w_f, updated_w_q)
    self.assertAllClose(updated_w_f_tensor, updated_w_q_tensor, atol=1e-3)

    # 2. Value check.
    self.assertNestedListClose(updated_w_q, expected_trained_weight)

  # Test the training with FQ quantization.
  @parameterized.parameters(
      generate_quantization_test_config([_add_expected_training_results])
  )
  def test_train(
      self,
      use_bias,
      attention_combine_dims,
      is_weight_symmetric,
      sample_input,
      sample_init_weight,
      pseudo_answer_q,
      pseudo_answer_k,
      pseudo_answer_v,
      expected_trained_weight
  ):
    quantization_option = QuantizationHParams(
        quantization_type=QuantizationType.FQ,
        mode=QuantizationMode.TRAINING,
        weight_params=quantization_hparams.WeightQuantizationParams(
            use_symmetric=is_weight_symmetric,
            calculation_dtype=jnp.bfloat16,
        ),
    )
    p_f = pax_fiddle.Config(
        attentions.CombinedQKVProjectionLayer,
        name='_combined_qkv_f',
        use_bias=use_bias,
        attention_combine_dims=attention_combine_dims,
    )
    p_q = pax_fiddle.Config(
        qattentions.CombinedQKVProjectionLayer,
        name='_combined_qkv_q',
        use_bias=use_bias,
        attention_combine_dims=attention_combine_dims,
        quantization=quantization_option,
    )
    for p in [p_f, p_q]:
      p.input_dim = 4
      p.num_heads = 2
      p.dim_per_head = 3

    inputs = np.array(sample_input).astype(np.float32).reshape([3, 4])
    self.train_and_compare(
        p_f,
        p_q,
        inputs,
        sample_init_weight,
        pseudo_answer_q,
        pseudo_answer_k,
        pseudo_answer_v,
        expected_trained_weight,
    )

  # Test FQ weight quantization.
  # Currently, weight quantization is not supported for FQ.
  @parameterized.parameters(
      generate_quantization_test_config()
  )
  def test_weight_quantization(
      self,
      use_bias,
      attention_combine_dims,
      is_weight_symmetric
  ):
    return

  # Check Q specification.
  @parameterized.parameters(generate_quantization_test_config())
  def test_quantization_partition_spec(
      self,
      use_bias,
      attention_combine_dims,
      is_weight_symmetric
  ):
    quantization_option = QuantizationHParams(
        quantization_type=QuantizationType.FQ,
        mode=QuantizationMode.MATERIALIZE,
        weight_params=quantization_hparams.WeightQuantizationParams(
            use_symmetric=is_weight_symmetric,
            calculation_dtype=jnp.bfloat16,
        ),
    )

    p = pax_fiddle.Config(
        qattentions.CombinedQKVProjectionLayer,
        name='_combined_qkv_proj_q',
        ici_mesh_shape=[0, 1, 2],
        mesh_axis_names=['replica', 'mdl', 'data'],
        weight_split_dims_mapping=base_layer.BaseLayer.WeightSharding(
            wt=[None, 'mdl', 'data']
            if attention_combine_dims
            else ['replica', 'mdl', 'data']
        ),
        use_bias=use_bias,
        attention_combine_dims=attention_combine_dims,
        quantization=quantization_option,
    )
    p.input_dim = 8
    p.num_heads = 2
    p.dim_per_head = 4

    layer = instantiate(p)
    inputs = np.random.normal(1.5, 2.0, [4, 8]).astype(np.float32)

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = layer.init(prng_key, inputs)

    pspec, _ = layer.apply(
        initial_vars, mutable=[], method=layer.quantized_partition_specs
    )

    if attention_combine_dims:
      expected_pspec = {
          'params': {
              'w': base_layer.BoxedPartitionSpec(
                  meta=jax.sharding.PartitionSpec(None, 'mdl', 'data')
              ),
              'w_quantized_scale': base_layer.BoxedPartitionSpec(
                  meta=jax.sharding.PartitionSpec(None, 'mdl')
              ),
          }
      }
    else:
      expected_pspec = {
          'params': {
              'w': base_layer.BoxedPartitionSpec(
                  meta=jax.sharding.PartitionSpec(
                      None, 'replica', 'mdl', 'data'
                  )
              ),
              'w_quantized_scale': base_layer.BoxedPartitionSpec(
                  meta=jax.sharding.PartitionSpec(None, 'mdl', 'data')
              ),
          }
      }
    if not is_weight_symmetric:
      expected_pspec['params']['w_quantized_zp'] = expected_pspec['params'][
          'w_quantized_scale'
      ]

    self.assertEqual(pspec, expected_pspec)

  # Check inference result.
  # Currently, the weight quantization for FQ is not being supported.
  # Since it is not being supported, we cannot test the inference results with
  # quantized weights.
  @parameterized.parameters(
      generate_quantization_test_config()
  )
  def test_inference_call(
      self,
      use_bias,
      attention_combine_dims,
      is_weight_symmetric
  ):
    return

if __name__ == '__main__':
  absltest.main()

