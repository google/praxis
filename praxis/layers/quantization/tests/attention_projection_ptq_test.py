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

"""PTQ Tests for quantized AttentionProjection layer."""
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


def _add_expected_quantized_weights(cur_key, cur_samples):
  target_weight = [
      0.67381,
      -1.02029,
      -2.96073,
      0.4568,
      -2.06743,
      -0.71279,
      -0.87012,
      -2.57819,
      -0.519,
      -2.14745,
      -0.66781,
      0.46878,
      -0.1505,
      -2.11084,
      -0.43236,
      -0.89053,
      -1.69517,
      -1.24534,
      -0.37589,
      -1.56385,
      -0.9495,
      -0.98122,
      -0.20156,
      -1.89354,
      -1.59769,
      -0.0634,
      -1.43138,
      0.64709,
      -0.74104,
      -1.41583,
      -1.65185,
      -0.34277,
      -0.38108,
      -1.06516,
      -1.96323,
      -1.18998,
  ]
  expected_weights = [
      [
          [
              [52, -50, -127, 27, -127, -48],
              [-67, -127, -22, -127, -41, 31],
              [-12, -104, -19, -53, -104, -84],
          ],
          [
              [-29, -77, -41, -58, -12, -127],
              [-123, -3, -61, 38, -46, -95],
              [-127, -17, -16, -63, -121, -80],
          ],
      ],
      [
          [
              [127, 30, -128, 110, -128, -1],
              [-42, -128, 113, -128, 63, 127],
              [37, -81, 122, -13, -77, -58],
          ],
          [
              [12, -25, 71, -22, 127, -128],
              [-122, 127, 23, 127, 53, -76],
              [-128, 99, 127, -29, -114, -52],
          ],
      ],
      [
          [52, -50, -127, 27, -127, -48],
          [-67, -127, -22, -127, -41, 31],
          [-12, -104, -19, -53, -104, -84],
          [-29, -77, -41, -58, -12, -127],
          [-123, -3, -61, 38, -46, -95],
          [-127, -17, -16, -63, -121, -80],
      ],
      [
          [127, 30, -128, 110, -128, -1],
          [-42, -128, 113, -128, 63, 127],
          [37, -81, 122, -13, -77, -58],
          [12, -25, 71, -22, 127, -128],
          [-122, 127, 23, 127, 53, -76],
          [-128, 99, 127, -29, -114, -52],
      ],
      [
          [[29, -44, -127], [20, -89, -31]],
          [[-43, -127, -26], [-106, -33, 23]],
          [[-9, -127, -26], [-54, -102, -75]],
          [[-25, -105, -64], [-66, -14, -127]],
          [[-127, -5, -114], [51, -59, -113]],
          [[-107, -22, -25], [-69, -127, -77]],
      ],
      [
          [[127, 8, -128], [112, -65, 30]],
          [[15, -128, 44], [-92, 32, 127]],
          [[127, -128, 90], [31, -74, -15]],
          [[101, -78, 14], [9, 127, -128]],
          [[-128, 46, -109], [127, -31, -107]],
          [[-79, 127, 121], [13, -128, -6]],
      ],
      [
          [29, -44, -127, 20, -89, -31],
          [-43, -127, -26, -106, -33, 23],
          [-9, -127, -26, -54, -102, -75],
          [-25, -105, -64, -66, -14, -127],
          [-127, -5, -114, 51, -59, -113],
          [-107, -22, -25, -69, -127, -77],
      ],
      [
          [127, 8, -128, 112, -65, 30],
          [15, -128, 44, -92, 32, 127],
          [127, -128, 90, 31, -74, -15],
          [101, -78, 14, 9, 127, -128],
          [-128, 46, -109, 127, -31, -107],
          [-79, 127, 121, 13, -128, -6],
      ],
      [
          [[52, -50, -127], [27, -127, -48]],
          [[-67, -127, -22], [-127, -41, 31]],
          [[-12, -104, -19], [-53, -104, -84]],
          [[-29, -77, -41], [-58, -12, -127]],
          [[-123, -3, -61], [38, -46, -95]],
          [[-127, -17, -16], [-63, -121, -80]],
      ],
      [
          [[127, 30, -128], [110, -128, -1]],
          [[-42, -128, 113], [-128, 63, 127]],
          [[37, -81, 122], [-13, -77, -58]],
          [[12, -25, 71], [-22, 127, -128]],
          [[-122, 127, 23], [127, 53, -76]],
          [[-128, 99, 127], [-29, -114, -52]],
      ],
      [
          [52, -50, -127, 27, -127, -48],
          [-67, -127, -22, -127, -41, 31],
          [-12, -104, -19, -53, -104, -84],
          [-29, -77, -41, -58, -12, -127],
          [-123, -3, -61, 38, -46, -95],
          [-127, -17, -16, -63, -121, -80],
      ],
      [
          [127, 30, -128, 110, -128, -1],
          [-42, -128, 113, -128, 63, 127],
          [37, -81, 122, -13, -77, -58],
          [12, -25, 71, -22, 127, -128],
          [-122, 127, 23, 127, 53, -76],
          [-128, 99, 127, -29, -114, -52],
      ],
      [
          [[52, -50, -127], [27, -127, -48]],
          [[-67, -127, -22], [-127, -41, 31]],
          [[-12, -104, -19], [-53, -104, -84]],
          [[-29, -77, -41], [-58, -12, -127]],
          [[-123, -3, -61], [38, -46, -95]],
          [[-127, -17, -16], [-63, -121, -80]],
      ],
      [
          [[127, 30, -128], [110, -128, -1]],
          [[-42, -128, 113], [-128, 63, 127]],
          [[37, -81, 122], [-13, -77, -58]],
          [[12, -25, 71], [-22, 127, -128]],
          [[-122, 127, 23], [127, 53, -76]],
          [[-128, 99, 127], [-29, -114, -52]],
      ],
      [
          [52, -50, -127, 27, -127, -48],
          [-67, -127, -22, -127, -41, 31],
          [-12, -104, -19, -53, -104, -84],
          [-29, -77, -41, -58, -12, -127],
          [-123, -3, -61, 38, -46, -95],
          [-127, -17, -16, -63, -121, -80],
      ],
      [
          [127, 30, -128, 110, -128, -1],
          [-42, -128, 113, -128, 63, 127],
          [37, -81, 122, -13, -77, -58],
          [12, -25, 71, -22, 127, -128],
          [-122, 127, 23, 127, 53, -76],
          [-128, 99, 127, -29, -114, -52],
      ],
      [
          [[52, -50, -127], [27, -127, -48]],
          [[-67, -127, -22], [-127, -41, 31]],
          [[-12, -104, -19], [-53, -104, -84]],
          [[-29, -77, -41], [-58, -12, -127]],
          [[-123, -3, -61], [38, -46, -95]],
          [[-127, -17, -16], [-63, -121, -80]],
      ],
      [
          [[127, 30, -128], [110, -128, -1]],
          [[-42, -128, 113], [-128, 63, 127]],
          [[37, -81, 122], [-13, -77, -58]],
          [[12, -25, 71], [-22, 127, -128]],
          [[-122, 127, 23], [127, 53, -76]],
          [[-128, 99, 127], [-29, -114, -52]],
      ],
      [
          [[29, -44, -127], [20, -89, -31]],
          [[-43, -127, -26], [-106, -33, 23]],
          [[-9, -127, -26], [-54, -102, -75]],
          [[-25, -105, -64], [-66, -14, -127]],
          [[-127, -5, -114], [51, -59, -113]],
          [[-107, -22, -25], [-69, -127, -77]],
      ],
      [
          [[127, 8, -128], [112, -65, 30]],
          [[15, -128, 44], [-92, 32, 127]],
          [[127, -128, 90], [31, -74, -15]],
          [[101, -78, 14], [9, 127, -128]],
          [[-128, 46, -109], [127, -31, -107]],
          [[-79, 127, 121], [13, -128, -6]],
      ],
      [
          [[52, -50, -127], [27, -127, -48]],
          [[-67, -127, -22], [-127, -41, 31]],
          [[-12, -104, -19], [-53, -104, -84]],
          [[-29, -77, -41], [-58, -12, -127]],
          [[-123, -3, -61], [38, -46, -95]],
          [[-127, -17, -16], [-63, -121, -80]],
      ],
      [
          [[127, 30, -128], [110, -128, -1]],
          [[-42, -128, 113], [-128, 63, 127]],
          [[37, -81, 122], [-13, -77, -58]],
          [[12, -25, 71], [-22, 127, -128]],
          [[-122, 127, 23], [127, 53, -76]],
          [[-128, 99, 127], [-29, -114, -52]],
      ],
  ]
  expected_scales = [
      [0.01301, 0.0203, 0.02331, 0.01691, 0.01628, 0.01491],
      [0.00912, 0.00986, 0.01012, 0.01096, 0.00732, 0.00926],
      [0.01301, 0.0203, 0.02331, 0.01691, 0.01628, 0.01491],
      [0.00912, 0.00986, 0.01012, 0.01096, 0.00732, 0.00926],
      [0.02331, 0.0203, 0.01662, 0.01491, 0.01258, 0.01546],
      [0.01425, 0.01195, 0.00769, 0.00664, 0.0088, 0.00635],
      [0.02331, 0.0203, 0.01662, 0.01491, 0.01258, 0.01546],
      [0.01425, 0.01195, 0.00769, 0.00664, 0.0088, 0.00635],
      [[0.01301, 0.0203, 0.02331], [0.01691, 0.01628, 0.01491]],
      [[0.00912, 0.00986, 0.01012], [0.01096, 0.00732, 0.00926]],
      [0.01301, 0.0203, 0.02331, 0.01691, 0.01628, 0.01491],
      [0.00912, 0.00986, 0.01012, 0.01096, 0.00732, 0.00926],
      [[0.01301, 0.0203, 0.02331], [0.01691, 0.01628, 0.01491]],
      [[0.00912, 0.00986, 0.01012], [0.01096, 0.00732, 0.00926]],
      [0.01301, 0.0203, 0.02331, 0.01691, 0.01628, 0.01491],
      [0.00912, 0.00986, 0.01012, 0.01096, 0.00732, 0.00926],
      [[0.01301, 0.0203, 0.02331], [0.01691, 0.01628, 0.01491]],
      [[0.00912, 0.00986, 0.01012], [0.01096, 0.00732, 0.00926]],
      [0.02331, 0.0203, 0.01662, 0.01491, 0.01258, 0.01546],
      [0.01425, 0.01195, 0.00769, 0.00664, 0.0088, 0.00635],
      [[0.01301, 0.0203, 0.02331], [0.01691, 0.01628, 0.01491]],
      [[0.00912, 0.00986, 0.01012], [0.01096, 0.00732, 0.00926]],
  ]
  expected_zps = [
      None,
      [0.48446, 1.31586, 1.66585, 0.7447, 1.13084, 0.70775],
      None,
      [0.48446, 1.31586, 1.66585, 0.7447, 1.13084, 0.70775],
      None,
      [1.13633, 1.04873, 1.12683, 1.04423, 0.4709, 1.14982],
      None,
      [1.13633, 1.04873, 1.12683, 1.04423, 0.4709, 1.14982],
      None,
      [[0.48446, 1.31586, 1.66585], [0.7447, 1.13084, 0.70775]],
      None,
      [0.48446, 1.31586, 1.66585, 0.7447, 1.13084, 0.70775],
      None,
      [[0.48446, 1.31586, 1.66585], [0.7447, 1.13084, 0.70775]],
      None,
      [0.48446, 1.31586, 1.66585, 0.7447, 1.13084, 0.70775],
      None,
      [[0.48446, 1.31586, 1.66585], [0.7447, 1.13084, 0.70775]],
      None,
      [1.13633, 1.04873, 1.12683, 1.04423, 0.4709, 1.14982],
      None,
      [[0.48446, 1.31586, 1.66585], [0.7447, 1.13084, 0.70775]],
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
      -2.22199,
      -1.88602,
      -0.02927,
      -2.69641,
      -0.5878,
      -0.50946,
      -1.78383,
      -1.29524,
      -1.64884,
      -1.91264,
      -0.42079,
      0.37078,
      -1.55856,
      -0.79459,
      -1.83468,
      -3.27913,
      -1.97442,
      -1.47083,
      -1.4707,
      -1.20945,
      -1.77924,
      -1.21964,
      -1.51418,
      -2.44872,
      -2.03366,
      -1.61235,
      -2.10163,
      0.08583,
      -2.45877,
      -0.35326,
      0.28335,
      -0.45909,
      0.11562,
      -1.32482,
      -0.8984,
      -1.35707,
  ]
  sample_input = [
      0.10513,
      -1.12749,
      -1.47827,
      -0.51852,
      -0.40626,
      -1.91471,
      -0.34426,
      -0.39641,
      -1.89249,
      -0.62093,
      -0.28789,
      -0.6004,
      -1.41112,
      -2.02689,
      -1.42512,
      -2.57246,
      0.20096,
      -1.73225,
  ]
  expected_results = [
      [
          [7.89588, 6.55599, 7.07401, 7.5288],
          [6.7207, 6.12363, 3.98131, 4.61029],
          [13.38474, 10.2899, 11.29049, 18.33138],
      ],
      None,
      [
          [7.89588, 6.55599, 7.07401, 7.5288],
          [6.7207, 6.12363, 3.98131, 4.61029],
          [13.38474, 10.2899, 11.29049, 18.33138],
      ],
      None,
      [
          [4.54285, 4.14229, 8.76503, 9.77506],
          [3.70807, 5.3148, 7.80463, 7.00603],
          [14.70574, 11.65554, 17.01151, 14.11841],
      ],
      [
          [4.55392, 4.16677, 8.75893, 9.77384],
          [3.72, 5.34187, 7.81188, 7.01467],
          [14.71869, 11.69615, 17.00218, 14.13585],
      ],
      [
          [4.54285, 4.14229, 8.76503, 9.77506],
          [3.70807, 5.3148, 7.80463, 7.00603],
          [14.70574, 11.65554, 17.01151, 14.11841],
      ],
      [
          [4.55392, 4.16677, 8.75893, 9.77384],
          [3.72, 5.34187, 7.81188, 7.01467],
          [14.71869, 11.69615, 17.00218, 14.13585],
      ],
      [
          [[4.84244, 3.06763, 5.48731], [7.34865, 4.11183, 2.97018]],
          [[5.43833, 3.99293, 4.50102], [6.35924, 2.32123, 0.97739]],
          [[6.64387, 5.3246, 2.67231], [7.94088, 2.8525, 2.61333]],
      ],
      [
          [[4.84406, 3.06414, 5.49206], [7.3545, 4.11531, 2.97915]],
          [[5.43637, 3.99957, 4.50875], [6.37345, 2.32231, 0.97751]],
          [[6.64407, 5.3287, 2.67602], [7.97013, 2.84479, 2.63713]],
      ],
      [
          [[4.84244, 3.06763, 5.48731], [7.34865, 4.11183, 2.97018]],
          [[5.43833, 3.99293, 4.50102], [6.35924, 2.32123, 0.97739]],
          [[6.64387, 5.3246, 2.67231], [7.94088, 2.8525, 2.61333]],
      ],
      [
          [[4.84406, 3.06414, 5.49206], [7.3545, 4.11531, 2.97915]],
          [[5.43637, 3.99957, 4.50875], [6.37345, 2.32231, 0.97751]],
          [[6.64407, 5.3287, 2.67602], [7.97013, 2.84479, 2.63713]],
      ],
      [
          [[4.84244, 3.06763, 5.48731], [7.34865, 4.11183, 2.97018]],
          [[5.43833, 3.99293, 4.50102], [6.35924, 2.32123, 0.97739]],
          [[6.64387, 5.3246, 2.67231], [7.94088, 2.8525, 2.61333]],
      ],
      [
          [[4.84406, 3.06414, 5.49206], [7.3545, 4.11531, 2.97915]],
          [[5.43637, 3.99957, 4.50875], [6.37345, 2.32231, 0.97751]],
          [[6.64407, 5.3287, 2.67602], [7.97013, 2.84479, 2.63713]],
      ],
      [
          [[4.84244, 3.06763, 5.48731], [7.34865, 4.11183, 2.97018]],
          [[5.43833, 3.99293, 4.50102], [6.35924, 2.32123, 0.97739]],
          [[6.64387, 5.3246, 2.67231], [7.94088, 2.8525, 2.61333]],
      ],
      [
          [[4.84406, 3.06414, 5.49206], [7.3545, 4.11531, 2.97915]],
          [[5.43637, 3.99957, 4.50875], [6.37345, 2.32231, 0.97751]],
          [[6.64407, 5.3287, 2.67602], [7.97013, 2.84479, 2.63713]],
      ],
      [
          [[4.84244, 3.06763, 5.48731], [7.34865, 4.11183, 2.97018]],
          [[5.43833, 3.99293, 4.50102], [6.35924, 2.32123, 0.97739]],
          [[6.64387, 5.3246, 2.67231], [7.94088, 2.8525, 2.61333]],
      ],
      [
          [[4.84406, 3.06414, 5.49206], [7.3545, 4.11531, 2.97915]],
          [[5.43637, 3.99957, 4.50875], [6.37345, 2.32231, 0.97751]],
          [[6.64407, 5.3287, 2.67602], [7.97013, 2.84479, 2.63713]],
      ],
      [
          [4.54285, 4.14229, 8.76503, 9.77506],
          [3.70807, 5.3148, 7.80463, 7.00603],
          [14.70574, 11.65554, 17.01151, 14.11841],
      ],
      [
          [4.55392, 4.16677, 8.75893, 9.77384],
          [3.72, 5.34187, 7.81188, 7.01467],
          [14.71869, 11.69615, 17.00218, 14.13585],
      ],
      [
          [[4.84244, 3.06763, 5.48731], [7.34865, 4.11183, 2.97018]],
          [[5.43833, 3.99293, 4.50102], [6.35924, 2.32123, 0.97739]],
          [[6.64387, 5.3246, 2.67231], [7.94088, 2.8525, 2.61333]],
      ],
      [
          [[4.84406, 3.06414, 5.49206], [7.3545, 4.11531, 2.97915]],
          [[5.43637, 3.99957, 4.50875], [6.37345, 2.32231, 0.97751]],
          [[6.64407, 5.3287, 2.67602], [7.97013, 2.84479, 2.63713]],
      ],
  ]

  updated_key = cur_key + ['sample_weight', 'sample_input', 'expected_result']
  ret = []
  for sample, expected_result in zip(cur_samples, expected_results):
    sample.append(sample_weight)
    sample.append(sample_input)
    sample.append(expected_result)

    ret.append(sample)

  return updated_key, ret


class AttentionProjectionPTQTest(quantization_test_util.QuantizationTestCase):
  """Test cases for QuantizationType.PTQ.

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

  def train_and_compare(self, p_f, p_q, inputs):
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
    outputs_f = attn_f.apply(initial_vars_f, inputs)
    pseudo_answer = np.random.normal(0.0, 2.0, outputs_f.shape)

    updated_vars_f = update(attn_f, initial_vars_f, inputs, pseudo_answer)
    updated_vars_q = update(attn_q, initial_vars_q, inputs, pseudo_answer)

    # 1. Check if the trained weights are the same.
    for k, v in updated_vars_f[base_layer.PARAMS].items():
      self.assertAllClose(v, updated_vars_q[base_layer.PARAMS][k])

    # 2. Check if the inference result with updated results are the same.
    outputs_f = attn_f.apply(updated_vars_f, inputs)
    outputs_q = attn_q.apply(updated_vars_q, inputs)
    self.assertAllClose(outputs_f, outputs_q)

  # See if the training results of PTQ-quantized model and original model are
  # the same.
  @parameterized.parameters(generate_quantization_test_config())
  def test_train(
      self,
      is_output_projection,
      use_bias,
      attention_combine_dims,
      use_nhd_shape,
      is_weight_symmetric,
  ):
    quantization_option = QuantizationHParams(
        quantization_type=QuantizationType.PTQ,
        mode=QuantizationMode.TRAINING,
        weight_params=quantization_hparams.WeightQuantizationParams(
            use_symmetric=is_weight_symmetric
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
      p.input_dim = 16
      p.num_heads = 2
      p.dim_per_head = 5

    if is_output_projection:
      inputs = np.random.normal(1.5, 2.0, [5, 2, 5]).astype(np.float32)
    else:
      inputs = np.random.normal(1.5, 2.0, [5, 16]).astype(np.float32)
    self.train_and_compare(p_f, p_q, inputs)

  # Test PTQ weight quantization.
  @parameterized.parameters(
      generate_quantization_test_config([_add_expected_quantized_weights])
  )
  def test_weight_quantization(
      self,
      is_output_projection,
      use_bias,
      attention_combine_dims,
      use_nhd_shape,
      is_weight_symmetric,
      target_weight,
      expected_weight,
      expected_scale,
      expected_zp
  ):
    # There exist a shape mismatch bug when the flag attention_combine_dims is
    # turned on.
    # The bug will be fixed in a next CL; until then, I blocked the related test
    # cases.
    if attention_combine_dims:
      return

    quantization_option = QuantizationHParams(
        quantization_type=QuantizationType.PTQ,
        mode=QuantizationMode.MATERIALIZE,
        weight_params=quantization_hparams.WeightQuantizationParams(
            use_symmetric=is_weight_symmetric
        ),
    )

    p = pax_fiddle.Config(
        qattentions.AttentionProjection,
        name='_attn_proj_q',
        mesh_axis_names=['replica', 'mdl', 'data'],
        weight_split_dims_mapping=base_layer.BaseLayer.WeightSharding(
            wt=['mdl', 'data']
        ),
        is_output_projection=is_output_projection,
        use_bias=use_bias,
        attention_combine_dims=attention_combine_dims,
        use_nhd_shape=use_nhd_shape,
        quantization=quantization_option,
    )
    p.input_dim = 6
    p.num_heads = 2
    p.dim_per_head = 3

    layer = instantiate(p)
    if is_output_projection:
      inputs = np.random.normal(1.5, 2.0, [3, 2, 3]).astype(np.float32)
    else:
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
      is_output_projection,
      use_bias,
      attention_combine_dims,
      use_nhd_shape,
      is_weight_symmetric
  ):
    quantization_option = QuantizationHParams(
        quantization_type=QuantizationType.PTQ,
        mode=QuantizationMode.MATERIALIZE,
        weight_params=quantization_hparams.WeightQuantizationParams(
            use_symmetric=is_weight_symmetric
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
  @parameterized.parameters(
      generate_quantization_test_config([_add_expected_quantization_results])
  )
  def test_inference_call(
      self,
      is_output_projection,
      use_bias,
      attention_combine_dims,
      use_nhd_shape,
      is_weight_symmetric,
      sample_weight,
      sample_input,
      expected_result
  ):
    # There exist a shape mismatch bug when the flag attention_combine_dims is
    # turned on.
    # The bug will be fixed in a next CL; until then, I blocked the related test
    # cases.
    if attention_combine_dims:
      return

    # Not supported.
    if is_output_projection and use_nhd_shape and not is_weight_symmetric:
      return

    quantization_option = QuantizationHParams(
        quantization_type=QuantizationType.PTQ,
        mode=QuantizationMode.INFERENCE,
        weight_params=quantization_hparams.WeightQuantizationParams(
            use_symmetric=is_weight_symmetric
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

    layer_f = instantiate(p_f)
    layer_q = instantiate(p_q)
    inputs = np.array(sample_input).astype(np.float32)
    if is_output_projection:
      inputs = inputs.reshape([3, 2, 3])
    else:
      inputs = inputs[:12].reshape([3, 4])

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
