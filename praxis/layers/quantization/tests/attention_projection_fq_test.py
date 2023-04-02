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

"""FQ Tests for quantized AttentionProjection layer."""
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
    quantization_test_util.generate_attention_projection_test_config
)


def _add_expected_training_results(cur_key, cur_samples):
  sample_input = [
      -1.26807,
      -2.04221,
      0.23409,
      -1.13823,
      -1.58314,
      -0.76971,
      -1.31358,
      -0.32611,
      -0.53641,
      -0.40342,
      -0.81108,
      -1.86804,
      -0.67612,
      -1.80399,
      0.01075,
      -1.64826,
      -0.47334,
      -2.09314,
  ]
  sample_init_weight = [
      -0.7554,
      -1.52106,
      -0.6777,
      -1.18297,
      -1.63098,
      0.1629,
      -1.52358,
      -1.70192,
      -0.58419,
      0.81529,
      -0.79822,
      -1.10159,
      -1.75617,
      -2.05341,
      -0.76527,
      -1.7593,
      -2.22355,
      0.77632,
      0.52507,
      0.65367,
      -2.64026,
      -0.82232,
      -0.94633,
      -0.08249,
  ]
  pseudo_answer = [
      -3.29654,
      -1.70126,
      -1.08829,
      -1.52927,
      0.75383,
      -1.49826,
      -3.56215,
      -0.90436,
      0.2494,
      -0.12776,
      -1.21297,
      -0.41449,
      -0.7843,
      -1.00443,
      -3.11404,
      -0.35362,
      0.22589,
      -0.77327,
  ]
  expected_trained_weights = [
      [
          [
              [-0.75811, -1.52377, -0.68041, -1.18568],
              [-1.63446, 0.15942, -1.52706, -1.7054],
              [-0.58443, 0.81505, -0.79846, -1.10183],
          ],
          [
              [-1.75883, -2.05603, -0.76793, -1.76203],
              [-2.22594, 0.77393, 0.52267, 0.65128],
              [-2.64421, -0.82627, -0.95026, -0.08644],
          ],
      ],
      [
          [
              [-0.75811, -1.52377, -0.68041, -1.18568],
              [-1.63446, 0.15942, -1.52706, -1.7054],
              [-0.58459, 0.81509, -0.79846, -1.10183],
          ],
          [
              [-1.75883, -2.05609, -0.76793, -1.762],
              [-2.22594, 0.77393, 0.52268, 0.65127],
              [-2.64419, -0.82627, -0.95028, -0.08644],
          ],
      ],
      [
          [-0.75811, -1.52377, -0.68041, -1.18568],
          [-1.63446, 0.15942, -1.52706, -1.7054],
          [-0.58443, 0.81505, -0.79846, -1.10183],
          [-1.75883, -2.05603, -0.76793, -1.76203],
          [-2.22594, 0.77393, 0.52267, 0.65128],
          [-2.64421, -0.82627, -0.95026, -0.08644],
      ],
      [
          [-0.75811, -1.52377, -0.68041, -1.18568],
          [-1.63446, 0.15942, -1.52706, -1.7054],
          [-0.58459, 0.81509, -0.79846, -1.10183],
          [-1.75883, -2.05609, -0.76793, -1.762],
          [-2.22594, 0.77393, 0.52268, 0.65127],
          [-2.64419, -0.82627, -0.95028, -0.08644],
      ],
      [
          [[-0.75811, -1.52454, -0.67794], [-1.18563, -1.63337, 0.15897]],
          [[-1.52629, -1.7054, -0.58443], [0.81263, -0.80062, -1.10554]],
          [[-1.75888, -2.05689, -0.76551], [-1.76196, -2.2259, 0.77237]],
          [[0.52236, 0.65019, -2.64053], [-0.82498, -0.94872, -0.08644]],
      ],
      [
          [[-0.75811, -1.52454, -0.67794], [-1.18563, -1.63336, 0.15892]],
          [[-1.52629, -1.7054, -0.58443], [0.81259, -0.80061, -1.10554]],
          [[-1.75888, -2.05689, -0.76551], [-1.76196, -2.22599, 0.77237]],
          [[0.52236, 0.65019, -2.64057], [-0.82498, -0.94872, -0.08644]],
      ],
      [
          [-0.75811, -1.52454, -0.67794, -1.18563, -1.63337, 0.15897],
          [-1.52629, -1.7054, -0.58443, 0.81263, -0.80062, -1.10554],
          [-1.75888, -2.05689, -0.76551, -1.76196, -2.2259, 0.77237],
          [0.52236, 0.65019, -2.64053, -0.82498, -0.94872, -0.08644],
      ],
      [
          [-0.75811, -1.52454, -0.67794, -1.18563, -1.63336, 0.15892],
          [-1.52629, -1.7054, -0.58443, 0.81259, -0.80061, -1.10554],
          [-1.75888, -2.05689, -0.76551, -1.76196, -2.22599, 0.77237],
          [0.52236, 0.65019, -2.64057, -0.82498, -0.94872, -0.08644],
      ],
      [
          [[-0.75728, -1.52294, -0.67958], [-1.18485, -1.63286, 0.16102]],
          [[-1.52537, -1.70371, -0.58598], [0.8135, -0.80001, -1.10339]],
          [[-1.7572, -2.05446, -0.76632], [-1.76034, -2.22462, 0.77527]],
          [[0.52322, 0.65181, -2.6421], [-0.82417, -0.94819, -0.08435]],
      ],
      [
          [[-0.75728, -1.52294, -0.67958], [-1.18485, -1.63286, 0.16102]],
          [[-1.52537, -1.70371, -0.58594], [0.81349, -0.8, -1.10337]],
          [[-1.75722, -2.05446, -0.76632], [-1.76036, -2.22459, 0.77529]],
          [[0.52323, 0.65182, -2.64213], [-0.82418, -0.94819, -0.08435]],
      ],
      [
          [-0.75728, -1.52294, -0.67958, -1.18485, -1.63286, 0.16102],
          [-1.52537, -1.70371, -0.58598, 0.8135, -0.80001, -1.10339],
          [-1.7572, -2.05446, -0.76632, -1.76034, -2.22462, 0.77527],
          [0.52322, 0.65181, -2.6421, -0.82417, -0.94819, -0.08435],
      ],
      [
          [-0.75728, -1.52294, -0.67958, -1.18485, -1.63286, 0.16102],
          [-1.52537, -1.70371, -0.58594, 0.81349, -0.8, -1.10337],
          [-1.75722, -2.05446, -0.76632, -1.76036, -2.22459, 0.77529],
          [0.52323, 0.65182, -2.64213, -0.82418, -0.94819, -0.08435],
      ],
      [
          [[-0.75728, -1.52294, -0.67958], [-1.18485, -1.63286, 0.16102]],
          [[-1.52537, -1.70371, -0.58598], [0.8135, -0.80001, -1.10339]],
          [[-1.7572, -2.05446, -0.76632], [-1.76034, -2.22462, 0.77527]],
          [[0.52322, 0.65181, -2.6421], [-0.82417, -0.94819, -0.08435]],
      ],
      [
          [[-0.75728, -1.52294, -0.67958], [-1.18485, -1.63286, 0.16102]],
          [[-1.52537, -1.70371, -0.58594], [0.81349, -0.8, -1.10337]],
          [[-1.75722, -2.05446, -0.76632], [-1.76036, -2.22459, 0.77529]],
          [[0.52323, 0.65182, -2.64213], [-0.82418, -0.94819, -0.08435]],
      ],
      [
          [-0.75728, -1.52294, -0.67958, -1.18485, -1.63286, 0.16102],
          [-1.52537, -1.70371, -0.58598, 0.8135, -0.80001, -1.10339],
          [-1.7572, -2.05446, -0.76632, -1.76034, -2.22462, 0.77527],
          [0.52322, 0.65181, -2.6421, -0.82417, -0.94819, -0.08435],
      ],
      [
          [-0.75728, -1.52294, -0.67958, -1.18485, -1.63286, 0.16102],
          [-1.52537, -1.70371, -0.58594, 0.81349, -0.8, -1.10337],
          [-1.75722, -2.05446, -0.76632, -1.76036, -2.22459, 0.77529],
          [0.52323, 0.65182, -2.64213, -0.82418, -0.94819, -0.08435],
      ],
      [
          [[-0.75728, -1.52294, -0.67958], [-1.18485, -1.63286, 0.16102]],
          [[-1.52537, -1.70371, -0.58598], [0.8135, -0.80001, -1.10339]],
          [[-1.7572, -2.05446, -0.76632], [-1.76034, -2.22462, 0.77527]],
          [[0.52322, 0.65181, -2.6421], [-0.82417, -0.94819, -0.08435]],
      ],
      [
          [[-0.75728, -1.52294, -0.67958], [-1.18485, -1.63286, 0.16102]],
          [[-1.52537, -1.70371, -0.58594], [0.81349, -0.8, -1.10337]],
          [[-1.75722, -2.05446, -0.76632], [-1.76036, -2.22459, 0.77529]],
          [[0.52323, 0.65182, -2.64213], [-0.82418, -0.94819, -0.08435]],
      ],
      [
          [[-0.75811, -1.52454, -0.67794], [-1.18563, -1.63337, 0.15897]],
          [[-1.52629, -1.7054, -0.58443], [0.81263, -0.80062, -1.10554]],
          [[-1.75888, -2.05689, -0.76551], [-1.76196, -2.2259, 0.77237]],
          [[0.52236, 0.65019, -2.64053], [-0.82498, -0.94872, -0.08644]],
      ],
      [
          [[-0.75811, -1.52454, -0.67794], [-1.18563, -1.63336, 0.15892]],
          [[-1.52629, -1.7054, -0.58443], [0.81259, -0.80061, -1.10554]],
          [[-1.75888, -2.05689, -0.76551], [-1.76196, -2.22599, 0.77237]],
          [[0.52236, 0.65019, -2.64057], [-0.82498, -0.94872, -0.08644]],
      ],
      [
          [[-0.75728, -1.52294, -0.67958], [-1.18485, -1.63286, 0.16102]],
          [[-1.52537, -1.70371, -0.58598], [0.8135, -0.80001, -1.10339]],
          [[-1.7572, -2.05446, -0.76632], [-1.76034, -2.22462, 0.77527]],
          [[0.52322, 0.65181, -2.6421], [-0.82417, -0.94819, -0.08435]],
      ],
      [
          [[-0.75728, -1.52294, -0.67958], [-1.18485, -1.63286, 0.16102]],
          [[-1.52537, -1.70371, -0.58594], [0.81349, -0.8, -1.10337]],
          [[-1.75722, -2.05446, -0.76632], [-1.76036, -2.22459, 0.77529]],
          [[0.52323, 0.65182, -2.64213], [-0.82418, -0.94819, -0.08435]],
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


class AttentionProjectionFQTest(quantization_test_util.QuantizationTestCase):
  """Test cases for QuantizationType.FQ.

  Following tests are required:
  1. Training test.
  2. Weight quantization test.
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
    step_size = 0.01
    attn_f = instantiate(p_f)
    attn_q = instantiate(p_q)

    def update(attn_layer, params, inputs, targets):
      def loss(params, inputs, targets):
        outputs = attn_layer.apply(params, inputs)
        return -jnp.mean(jnp.abs(outputs - targets))

      grads = jax.grad(loss)(params, inputs, targets)

      out_params = dict()
      out_params[base_layer.PARAMS] = dict()
      for k, v in params[base_layer.PARAMS].items():
        v_grad = grads[base_layer.PARAMS][k]
        out_params[base_layer.PARAMS][k] = v - step_size * v_grad

      return out_params

    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars_f = attn_f.init(prng_key, inputs)
    initial_vars_q = attn_q.init(prng_key, inputs)
    weight_shape_f = initial_vars_f[base_layer.PARAMS]['w'].shape
    weight_shape_q = initial_vars_q[base_layer.PARAMS]['w'].shape

    initial_vars_f[base_layer.PARAMS]['w'] = np.array(
        sample_init_weight
    ).reshape(weight_shape_f)
    initial_vars_q[base_layer.PARAMS]['w'] = np.array(
        sample_init_weight
    ).reshape(weight_shape_q)

    outputs_f = attn_f.apply(initial_vars_f, inputs)
    pseudo_answer = np.array(pseudo_answer[: np.prod(outputs_f.shape)]).reshape(
        outputs_f.shape
    )

    updated_vars_f = update(attn_f, initial_vars_f, inputs, pseudo_answer)
    updated_vars_q = update(attn_q, initial_vars_q, inputs, pseudo_answer)

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
      is_output_projection,
      use_bias,
      attention_combine_dims,
      use_nhd_shape,
      is_weight_symmetric,
      sample_input,
      sample_init_weight,
      pseudo_answer,
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
        attentions.AttentionProjection,
        name='_attn_proj_f',
        is_output_projection=is_output_projection,
        use_bias=use_bias,
        attention_combine_dims=attention_combine_dims,
        use_nhd_shape=use_nhd_shape,
    )
    p_q = pax_fiddle.Config(
        qattentions.AttentionProjection,
        name='_attn_proj_q',
        is_output_projection=is_output_projection,
        use_bias=use_bias,
        attention_combine_dims=attention_combine_dims,
        use_nhd_shape=use_nhd_shape,
        quantization=quantization_option,
    )
    for p in [p_f, p_q]:
      p.input_dim = 4
      p.num_heads = 2
      p.dim_per_head = 3

    if is_output_projection:
      inputs = np.array(sample_input).astype(np.float32).reshape([3, 2, 3])
    else:
      inputs = np.array(sample_input[:12]).astype(np.float32).reshape([3, 4])
    self.train_and_compare(
        p_f,
        p_q,
        inputs,
        sample_init_weight,
        pseudo_answer,
        expected_trained_weight,
    )

  # Test FQ weight quantization: Currently the weight quantization for FQ is not
  # supported.
  @parameterized.parameters(generate_quantization_test_config())
  def test_weight_quantization(
      self,
      is_output_projection,
      use_bias,
      attention_combine_dims,
      use_nhd_shape,
      is_weight_symmetric
  ):
    return

  # Check Q specification.
  @parameterized.parameters(generate_quantization_test_config())
  def test_quantization_partition_spec(
      self,
      is_output_projection,
      use_bias,
      attention_combine_dims,
      use_nhd_shape,
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
        qattentions.AttentionProjection,
        name='_attn_proj_q',
        mesh_axis_names=['replica', 'mdl', 'data'],
        weight_split_dims_mapping=base_layer.BaseLayer.WeightSharding(
            wt=['mdl', 'data']
            if is_output_projection
            else ['replica', 'mdl', 'data']
        ),
        is_output_projection=is_output_projection,
        use_bias=use_bias,
        attention_combine_dims=attention_combine_dims,
        use_nhd_shape=use_nhd_shape,
        quantization=quantization_option,
    )
    p.input_dim = 8
    p.num_heads = 2
    p.dim_per_head = 4

    layer = instantiate(p)
    if is_output_projection:
      inputs = np.random.normal(1.5, 2.0, [4, 2, 4]).astype(np.float32)
    else:
      inputs = np.random.normal(1.5, 2.0, [4, 8]).astype(np.float32)

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = layer.init(prng_key, inputs)

    pspec, _ = layer.apply(
        initial_vars, mutable=[], method=layer.quantized_partition_specs
    )

    if is_output_projection:
      expected_pspec = {
          'params': {
              'w': base_layer.BoxedPartitionSpec(
                  meta=jax.sharding.PartitionSpec('mdl', 'data')
              ),
              'w_quantized_scale': base_layer.BoxedPartitionSpec(
                  meta=jax.sharding.PartitionSpec('mdl')
              ),
          }
      }
      if not is_weight_symmetric:
        expected_pspec['params']['w_quantized_zp'] = (
            base_layer.BoxedPartitionSpec(
                meta=jax.sharding.PartitionSpec('mdl')
            )
        )
    else:
      expected_pspec = {
          'params': {
              'w': base_layer.BoxedPartitionSpec(
                  meta=jax.sharding.PartitionSpec('replica', 'mdl', 'data')
              ),
              'w_quantized_scale': base_layer.BoxedPartitionSpec(
                  meta=jax.sharding.PartitionSpec('mdl', 'data')
              ),
          }
      }
      if not is_weight_symmetric:
        expected_pspec['params']['w_quantized_zp'] = (
            base_layer.BoxedPartitionSpec(
                meta=jax.sharding.PartitionSpec('mdl', 'data')
            )
        )

    self.assertEqual(pspec, expected_pspec)

  # Check inference result.
  # Currently, the weight quantization for FQ is not being supported.
  # Since it is not being supported, we cannot test the inference results with
  # quantized weights.
  @parameterized.parameters(generate_quantization_test_config())
  def test_inference_call(
      self,
      is_output_projection,
      use_bias,
      attention_combine_dims,
      use_nhd_shape,
      is_weight_symmetric
  ):
    return

if __name__ == '__main__':
  absltest.main()
