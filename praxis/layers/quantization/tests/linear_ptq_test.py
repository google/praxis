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


class LinearsPTQTest(quantization_test_util.QuantizationTestCase):
  """Test cases for QuantizationType.PTQ on Linears layer

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

    prng_key = jax.random.PRNGKey(seed=123)
    params_float = float_model.init(prng_key, inputs)
    params_quantized = quantized_model.init(prng_key, inputs)
    outputs_float = float_model.apply(params_float, inputs)
    pseudo_answer = np.random.normal(0.0, 2.0, outputs_float.shape)
    pseudo_answer = jnp.asarray(pseudo_answer)

    params_float = update(float_model, params_float, inputs, pseudo_answer)
    params_quantized = update(
        quantized_model, params_quantized, inputs, pseudo_answer
    )

    # 1. Check if the trained weights are the same.
    params_float_flat = tree_util.tree_flatten(params_float)[0]
    params_quantized_flat = tree_util.tree_flatten(params_quantized)[0]
    self.assertAllClose(params_float_flat, params_quantized_flat)

    # 2. Check if the inference result with updated results are the same.
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


if __name__ == '__main__':
  absltest.main()
