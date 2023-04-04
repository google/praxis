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

"""PTQ Tests for quantized Linears layer."""
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from jax import tree_util
import numpy as np
from praxis import base_layer
from praxis import pax_fiddle
from praxis.layers import linears
from praxis.layers.quantization import linears as qlinears
from praxis.layers.quantization import quantization_hparams
from praxis.layers.quantization.tests import test_util as quantization_test_util

QuantizationHParams = quantization_hparams.QuantizationHParams
QuantizationMode = quantization_hparams.QuantizationMode
QuantizationType = quantization_hparams.QuantizationType
instantiate = base_layer.instantiate
to_list = quantization_test_util.to_list
generate_quantization_test_config = (
    quantization_test_util.generate_linears_test_config
)


def _add_expected_quantized_weights(cur_key, cur_samples):
  target_weight = [
      1.76405235,
      0.40015721,
      0.97873798,
      2.2408932,
      1.86755799,
      -0.97727788,
      0.95008842,
      -0.15135721,
      -0.10321885,
      0.4105985,
      0.14404357,
      1.45427351,
  ]
  expected_weights = [
      [[100, 27, 85], [127, 127, -85], [54, -10, -9], [23, 10, 127]],
      [[61, -58, 77], [127, 127, -128], [-53, -128, -36], [-128, -91, 127]],
  ]
  expected_scales = [
      [0.01764483, 0.01470518, 0.01145097],
      [0.00717763, 0.00791731, 0.0095355],
  ]
  expected_zps = [None, [-1.3293346, -0.86205906, -0.24326566]]

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
      1.76405235,
      0.40015721,
      0.97873798,
      2.2408932,
      1.86755799,
      -0.97727788,
      0.95008842,
      -0.15135721,
      -0.10321885,
      0.4105985,
      0.14404357,
      1.45427351,
  ]
  sample_input = [-0.6712612, 3.494691, 2.065957, -1.5125895]
  expected_results = [
      [[8.0014305, 5.733789, -6.4674907]],
      [[7.984358, 5.729262, -6.4777813]],
  ]

  updated_key = cur_key + ['sample_weight', 'sample_input', 'expected_result']
  ret = []
  for sample, expected_result in zip(cur_samples, expected_results):
    sample.append(sample_weight)
    sample.append(sample_input)
    sample.append(expected_result)

    ret.append(sample)

  return updated_key, ret


class LinearsPTQTest(quantization_test_util.QuantizationTestCase):
  """Test cases for QuantizationType.PTQ on Linears layer.

  Following tests are required:
  1. Training test: The results compared to the non-quantized model should be
                    the same.
  2. Weight quantization test: The weights should be properly converted.
  3. Partition spec test.
  4. Inference test.
  """

  def setUp(self):
    super().setUp()
    np.random.seed(123)

  def train_and_compare(self, cfg_float, cfg_quantized, inputs):
    # Train the two models, and compare if the results are equal.
    float_model = instantiate(cfg_float)
    quantized_model = instantiate(cfg_quantized)

    def update(model, params, x, y, lr=0.01):
      def loss(params, x, y):
        outputs = model.apply(params, x)
        return jnp.mean(jnp.abs(outputs - y))

      grads = jax.grad(loss)(params, x, y)
      return tree_util.tree_map(
          lambda theta, grad: theta - lr * grad, params, grads
      )

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      params_float = float_model.init(prng_key, inputs)
      params_quantized = quantized_model.init(prng_key, inputs)
      outputs_float = float_model.apply(params_float, inputs)
    pseudo_answer = np.random.normal(0.0, 2.0, outputs_float.shape)
    pseudo_answer = jnp.asarray(pseudo_answer)

    with base_layer.JaxContext.new_context():
      params_float = update(float_model, params_float, inputs, pseudo_answer)
      params_quantized = update(
          quantized_model, params_quantized, inputs, pseudo_answer
      )

    # 1. Check if the trained weights are the same.
    self.assertAllClose(params_float['params']['w'],
                        params_quantized['params']['w'])

    # 2. Check if the inference result with updated results are the same.
    with base_layer.JaxContext.new_context():
      outputs_float = float_model.apply(params_float, inputs)
      outputs_quantized = quantized_model.apply(params_quantized, inputs)
    self.assertAllClose(outputs_float, outputs_quantized)

  # See if the training results of PTQ-quantized model and original model are
  # the same.
  @parameterized.parameters(generate_quantization_test_config())
  def test_train(
      self,
      is_weight_symmetric,
  ):
    quantization_option = QuantizationHParams(
        quantization_type=QuantizationType.PTQ,
        mode=QuantizationMode.TRAINING,
        weight_params=quantization_hparams.WeightQuantizationParams(
            use_symmetric=is_weight_symmetric
        ),
    )
    linear_cfg_float = pax_fiddle.Config(linears.Linear, name='_linear_float')
    linear_cfg_quantized = pax_fiddle.Config(
        qlinears.Linear,
        name='_linear_quantized',
        quantization=quantization_option,
    )
    for cfg in [linear_cfg_float, linear_cfg_quantized]:
      cfg.input_dims = 4
      cfg.output_dims = 3

    inputs = np.random.normal(1.5, 2.0, [2, 4]).astype(np.float32)
    inputs = jnp.asarray(inputs)
    self.train_and_compare(linear_cfg_float, linear_cfg_quantized, inputs)

  # Test PTQ weight quantization.
  @parameterized.parameters(
      generate_quantization_test_config([_add_expected_quantized_weights])
  )
  def test_weight_quantization(
      self,
      is_weight_symmetric,
      target_weight,
      expected_weight,
      expected_scale,
      expected_zp,
  ):
    quantization_option = QuantizationHParams(
        quantization_type=QuantizationType.PTQ,
        mode=QuantizationMode.MATERIALIZE,
        weight_params=quantization_hparams.WeightQuantizationParams(
            use_symmetric=is_weight_symmetric
        ),
    )

    cfg = pax_fiddle.Config(
        qlinears.Linear,
        name='_linear_quantized',
        mesh_axis_names=['replica', 'mdl', 'data'],
        weight_split_dims_mapping=base_layer.BaseLayer.WeightSharding(
            wt=['mdl', 'data']
        ),
        quantization=quantization_option,
        input_dims=4,
        output_dims=3,
    )

    layer = instantiate(cfg)
    inputs = np.random.normal(1.5, 2.0, [1, 4]).astype(np.float32)

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = layer.init(prng_key, inputs)
      weight_shape = initial_vars[base_layer.PARAMS]['w'].shape
      # set the weight (before quantization) to the desired target weight
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
      is_weight_symmetric,
  ):
    quantization_option = QuantizationHParams(
        quantization_type=QuantizationType.PTQ,
        mode=QuantizationMode.MATERIALIZE,
        weight_params=quantization_hparams.WeightQuantizationParams(
            use_symmetric=is_weight_symmetric
        ),
    )

    cfg = pax_fiddle.Config(
        qlinears.Linear,
        name='_linear_quantized',
        mesh_axis_names=['replica', 'mdl', 'data'],
        weight_split_dims_mapping=base_layer.BaseLayer.WeightSharding(
            wt=['mdl', 'data']
        ),
        quantization=quantization_option,
        input_dims=4,
        output_dims=3,
    )

    layer = instantiate(cfg)
    inputs = np.random.normal(1.5, 2.0, [1, 4]).astype(np.float32)

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = layer.init(prng_key, inputs)

      pspec, _ = layer.apply(
          initial_vars, mutable=[], method=layer.quantized_partition_specs
      )

    expected_pspec = {
        'params': {
            'w': base_layer.BoxedPartitionSpec(
                meta=jax.sharding.PartitionSpec('mdl', 'data')
            ),
            'w_quantized_scale': base_layer.BoxedPartitionSpec(
                meta=jax.sharding.PartitionSpec(
                    'data',
                )
            ),
        }
    }

    if not is_weight_symmetric:
      expected_pspec['params']['w_quantized_zp'] = (
          base_layer.BoxedPartitionSpec(
              meta=jax.sharding.PartitionSpec(
                  'data',
              )
          )
      )

    self.assertEqual(pspec, expected_pspec)

  # Check inference result.
  @parameterized.parameters(
      generate_quantization_test_config([_add_expected_quantization_results])
  )
  def test_inference_call(
      self,
      is_weight_symmetric,
      sample_weight,
      sample_input,
      expected_result,
  ):
    quantization_option = QuantizationHParams(
        quantization_type=QuantizationType.PTQ,
        mode=QuantizationMode.INFERENCE,
        weight_params=quantization_hparams.WeightQuantizationParams(
            use_symmetric=is_weight_symmetric
        ),
    )
    linear_cfg_float = pax_fiddle.Config(linears.Linear, name='_linear_float')
    linear_cfg_quantized = pax_fiddle.Config(
        qlinears.Linear,
        name='_linear_quantized',
        quantization=quantization_option,
    )
    for cfg in [linear_cfg_float, linear_cfg_quantized]:
      cfg.input_dims = 4
      cfg.output_dims = 3

    layer_f = instantiate(linear_cfg_float)
    layer_q = instantiate(linear_cfg_quantized)
    inputs = np.array(sample_input).astype(np.float32).reshape([1, 4])

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars_f = layer_f.init(prng_key, inputs)
      initial_vars_q = layer_q.init(prng_key, inputs)
      weight_shape_f = initial_vars_f[base_layer.PARAMS]['w'].shape
      weight_shape_q = initial_vars_q[base_layer.PARAMS]['w'].shape
      sample_weight = sample_weight[: np.prod(weight_shape_f)]
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
