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

"""Tests for quantized AttentionProjection layer."""
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
      -0.54386,
      -2.53066,
      -1.18443,
      -1.95979,
      -2.17336,
      -2.28523,
      -1.05954,
      -1.00019,
      -0.54596,
      -1.84391,
      -3.41523,
      -2.00566,
      -1.49485,
      -2.9653,
      -0.7045,
      -0.83795,
      -1.66723,
      -1.13179,
  ]
  sample_init_weight = [
      -2.15535,
      0.30676,
      -1.24707,
      -0.62521,
      -1.33668,
      -1.16038,
      0.51353,
      -0.29848,
      -1.31042,
      0.46827,
      0.85347,
      -1.02423,
      -0.3762,
      -1.18879,
      -2.07196,
      0.65845,
      1.407,
      -0.27715,
      -1.77313,
      -1.20534,
      -0.49318,
      -2.13344,
      -1.51106,
      -1.33449,
  ]
  pseudo_answer = [
      -0.34306,
      -0.69014,
      -1.49167,
      -1.02093,
      0.63741,
      1.35343,
      1.118,
      -0.53484,
      -0.15856,
      -1.26393,
      0.64037,
      -0.64181,
      -0.96042,
      -1.54216,
      -0.7629,
      -2.50346,
      -0.18399,
      -1.49924,
  ]
  expected_trained_weights = [
      [
          [
              [-2.15793, 0.30418, -1.24965, -0.62779],
              [-1.34209, -1.16579, 0.50812, -0.30389],
              [-1.31245, 0.46624, 0.85144, -1.02626],
          ],
          [
              [-0.38007, -1.19266, -2.07587, 0.65458],
              [1.40095, -0.2832, -1.77918, -1.21139],
              [-0.4977, -2.13794, -1.51558, -1.33898],
          ],
      ],
      None,
      [
          [-2.15793, 0.30418, -1.24965, -0.62779],
          [-1.34209, -1.16579, 0.50812, -0.30389],
          [-1.31245, 0.46624, 0.85144, -1.02626],
          [-0.38007, -1.19266, -2.07587, 0.65458],
          [1.40095, -0.2832, -1.77918, -1.21139],
          [-0.4977, -2.13794, -1.51558, -1.33898],
      ],
      None,
      [
          [[-2.15793, 0.30135, -1.2491], [-0.62908, -1.34273, -1.1649]],
          [[0.51271, -0.30223, -1.31154], [0.46748, 0.85312, -1.02541]],
          [[-0.37702, -1.19254, -2.07309], [0.65766, 1.40665, -0.27833]],
          [[-1.77571, -1.21075, -0.49521], [-2.13731, -1.51711, -1.33901]],
      ],
      [
          [[-2.15794, 0.30135, -1.2491], [-0.62908, -1.34273, -1.1649]],
          [[0.51271, -0.30223, -1.31154], [0.46748, 0.85312, -1.02541]],
          [[-0.37702, -1.19254, -2.07307], [0.65766, 1.40664, -0.27833]],
          [[-1.77571, -1.21075, -0.49521], [-2.13731, -1.51711, -1.33901]],
      ],
      [
          [-2.15793, 0.30135, -1.2491, -0.62908, -1.34273, -1.1649],
          [0.51271, -0.30223, -1.31154, 0.46748, 0.85312, -1.02541],
          [-0.37702, -1.19254, -2.07309, 0.65766, 1.40665, -0.27833],
          [-1.77571, -1.21075, -0.49521, -2.13731, -1.51711, -1.33901],
      ],
      [
          [-2.15794, 0.30135, -1.2491, -0.62908, -1.34273, -1.1649],
          [0.51271, -0.30223, -1.31154, 0.46748, 0.85312, -1.02541],
          [-0.37702, -1.19254, -2.07307, 0.65766, 1.40664, -0.27833],
          [-1.77571, -1.21075, -0.49521, -2.13731, -1.51711, -1.33901],
      ],
      [
          [[-2.15716, 0.30495, -1.24888], [-0.62702, -1.33728, -1.16219]],
          [[0.50983, -0.30218, -1.31412], [0.46457, 0.85463, -1.02793]],
          [[-0.37934, -1.19193, -2.07508], [0.65531, 1.40897, -0.28029]],
          [[-1.77589, -1.20808, -0.49594], [-2.1362, -1.50941, -1.33722]],
      ],
      [
          [[-2.15716, 0.30494, -1.24888], [-0.62702, -1.33728, -1.16219]],
          [[0.50982, -0.30218, -1.31412], [0.46457, 0.85463, -1.02793]],
          [[-0.37934, -1.19193, -2.07511], [0.6553, 1.40897, -0.2803]],
          [[-1.77589, -1.2081, -0.49594], [-2.13619, -1.50941, -1.33725]],
      ],
      [
          [-2.15716, 0.30495, -1.24888, -0.62702, -1.33728, -1.16219],
          [0.50983, -0.30218, -1.31412, 0.46457, 0.85463, -1.02793],
          [-0.37934, -1.19193, -2.07508, 0.65531, 1.40897, -0.28029],
          [-1.77589, -1.20808, -0.49594, -2.1362, -1.50941, -1.33722],
      ],
      [
          [-2.15716, 0.30494, -1.24888, -0.62702, -1.33728, -1.16219],
          [0.50982, -0.30218, -1.31412, 0.46457, 0.85463, -1.02793],
          [-0.37934, -1.19193, -2.07511, 0.6553, 1.40897, -0.2803],
          [-1.77589, -1.2081, -0.49594, -2.13619, -1.50941, -1.33725],
      ],
      [
          [[-2.15716, 0.30495, -1.24888], [-0.62702, -1.33728, -1.16219]],
          [[0.50983, -0.30218, -1.31412], [0.46457, 0.85463, -1.02793]],
          [[-0.37934, -1.19193, -2.07508], [0.65531, 1.40897, -0.28029]],
          [[-1.77589, -1.20808, -0.49594], [-2.1362, -1.50941, -1.33722]],
      ],
      [
          [[-2.15716, 0.30494, -1.24888], [-0.62702, -1.33728, -1.16219]],
          [[0.50982, -0.30218, -1.31412], [0.46457, 0.85463, -1.02793]],
          [[-0.37934, -1.19193, -2.07511], [0.6553, 1.40897, -0.2803]],
          [[-1.77589, -1.2081, -0.49594], [-2.13619, -1.50941, -1.33725]],
      ],
      [
          [-2.15716, 0.30495, -1.24888, -0.62702, -1.33728, -1.16219],
          [0.50983, -0.30218, -1.31412, 0.46457, 0.85463, -1.02793],
          [-0.37934, -1.19193, -2.07508, 0.65531, 1.40897, -0.28029],
          [-1.77589, -1.20808, -0.49594, -2.1362, -1.50941, -1.33722],
      ],
      [
          [-2.15716, 0.30494, -1.24888, -0.62702, -1.33728, -1.16219],
          [0.50982, -0.30218, -1.31412, 0.46457, 0.85463, -1.02793],
          [-0.37934, -1.19193, -2.07511, 0.6553, 1.40897, -0.2803],
          [-1.77589, -1.2081, -0.49594, -2.13619, -1.50941, -1.33725],
      ],
      [
          [[-2.15716, 0.30495, -1.24888], [-0.62702, -1.33728, -1.16219]],
          [[0.50983, -0.30218, -1.31412], [0.46457, 0.85463, -1.02793]],
          [[-0.37934, -1.19193, -2.07508], [0.65531, 1.40897, -0.28029]],
          [[-1.77589, -1.20808, -0.49594], [-2.1362, -1.50941, -1.33722]],
      ],
      [
          [[-2.15716, 0.30494, -1.24888], [-0.62702, -1.33728, -1.16219]],
          [[0.50982, -0.30218, -1.31412], [0.46457, 0.85463, -1.02793]],
          [[-0.37934, -1.19193, -2.07511], [0.6553, 1.40897, -0.2803]],
          [[-1.77589, -1.2081, -0.49594], [-2.13619, -1.50941, -1.33725]],
      ],
      [
          [[-2.15793, 0.30135, -1.2491], [-0.62908, -1.34273, -1.1649]],
          [[0.51271, -0.30223, -1.31154], [0.46748, 0.85312, -1.02541]],
          [[-0.37702, -1.19254, -2.07309], [0.65766, 1.40665, -0.27833]],
          [[-1.77571, -1.21075, -0.49521], [-2.13731, -1.51711, -1.33901]],
      ],
      [
          [[-2.15794, 0.30135, -1.2491], [-0.62908, -1.34273, -1.1649]],
          [[0.51271, -0.30223, -1.31154], [0.46748, 0.85312, -1.02541]],
          [[-0.37702, -1.19254, -2.07307], [0.65766, 1.40664, -0.27833]],
          [[-1.77571, -1.21075, -0.49521], [-2.13731, -1.51711, -1.33901]],
      ],
      [
          [[-2.15716, 0.30495, -1.24888], [-0.62702, -1.33728, -1.16219]],
          [[0.50983, -0.30218, -1.31412], [0.46457, 0.85463, -1.02793]],
          [[-0.37934, -1.19193, -2.07508], [0.65531, 1.40897, -0.28029]],
          [[-1.77589, -1.20808, -0.49594], [-2.1362, -1.50941, -1.33722]],
      ],
      [
          [[-2.15716, 0.30494, -1.24888], [-0.62702, -1.33728, -1.16219]],
          [[0.50982, -0.30218, -1.31412], [0.46457, 0.85463, -1.02793]],
          [[-0.37934, -1.19193, -2.07511], [0.6553, 1.40897, -0.2803]],
          [[-1.77589, -1.2081, -0.49594], [-2.13619, -1.50941, -1.33725]],
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
      0.61393,
      -0.59483,
      0.64299,
      0.23731,
      -2.70565,
      -1.53161,
      0.0562,
      -2.47793,
      -1.79071,
      -0.25653,
      -1.80287,
      -0.96573,
      -2.51167,
      0.34429,
      -0.4883,
      -0.67177,
      -2.90302,
      -2.6505,
      -2.51485,
      -1.9554,
      -2.37668,
      -1.41808,
      -1.06503,
      -2.29008,
  ]
  expected_weights = [
      [
          [[27, -29, 32, 12], [-118, -73, 3, -127], [-78, -12, -91, -49]],
          [
              [-110, 16, -25, -34],
              [-127, -127, -127, -100],
              [-104, -68, -54, -117],
          ],
      ],
      None,
      None,
      None,
      [
          [[29, -28, 30], [11, -127, -72]],
          [[3, -127, -92], [-13, -92, -49]],
          [[-110, 15, -21], [-29, -127, -116]],
          [[-127, -99, -120], [-72, -54, -116]],
      ],
      [
          [[125, 33, 127], [96, -128, -39]],
          [[127, -128, -59], [96, -60, 24]],
          [[-97, 127, 62], [47, -128, -108]],
          [[-128, -30, -104], [65, 127, -88]],
      ],
      None,
      None,
      [
          [[31, -30, 34], [21, -118, -73]],
          [[3, -127, -96], [-23, -79, -46]],
          [[-127, 18, -26], [-60, -127, -127]],
          [[-127, -100, -127], [-127, -47, -110]],
      ],
      [
          [[127, 42, 127], [127, -101, 41]],
          [[82, -128, -79], [51, 25, 127]],
          [[-128, 127, 31], [-13, -128, -128]],
          [[-128, -81, -128], [-128, 127, -73]],
      ],
      None,
      None,
      [
          [[31, -30, 34], [21, -118, -73]],
          [[3, -127, -96], [-23, -79, -46]],
          [[-127, 18, -26], [-60, -127, -127]],
          [[-127, -100, -127], [-127, -47, -110]],
      ],
      [
          [[127, 42, 127], [127, -101, 41]],
          [[82, -128, -79], [51, 25, 127]],
          [[-128, 127, 31], [-13, -128, -128]],
          [[-128, -81, -128], [-128, 127, -73]],
      ],
      None,
      None,
      [
          [[31, -30, 34], [21, -118, -73]],
          [[3, -127, -96], [-23, -79, -46]],
          [[-127, 18, -26], [-60, -127, -127]],
          [[-127, -100, -127], [-127, -47, -110]],
      ],
      [
          [[127, 42, 127], [127, -101, 41]],
          [[82, -128, -79], [51, 25, 127]],
          [[-128, 127, 31], [-13, -128, -128]],
          [[-128, -81, -128], [-128, 127, -73]],
      ],
      [
          [[29, -28, 30], [11, -127, -72]],
          [[3, -127, -92], [-13, -92, -49]],
          [[-110, 15, -21], [-29, -127, -116]],
          [[-127, -99, -120], [-72, -54, -116]],
      ],
      [
          [[125, 33, 127], [96, -128, -39]],
          [[127, -128, -59], [96, -60, 24]],
          [[-97, 127, 62], [47, -128, -108]],
          [[-128, -30, -104], [65, 127, -88]],
      ],
      [
          [[31, -30, 34], [21, -118, -73]],
          [[3, -127, -96], [-23, -79, -46]],
          [[-127, 18, -26], [-60, -127, -127]],
          [[-127, -100, -127], [-127, -47, -110]],
      ],
      [
          [[127, 42, 127], [127, -101, 41]],
          [[82, -128, -79], [51, 25, 127]],
          [[-128, 127, 31], [-13, -128, -128]],
          [[-128, -81, -128], [-128, 127, -73]],
      ],
  ]
  expected_scales = [
      [0.02286, 0.02087, 0.0198, 0.01951],
      None,
      None,
      None,
      [0.0213, 0.01951, 0.02286, 0.0198],
      [0.01313, 0.00994, 0.01273, 0.00569],
      None,
      None,
      [[0.0198, 0.01951, 0.01871], [0.01117, 0.02286, 0.02087]],
      [[0.01227, 0.01107, 0.01184], [0.00649, 0.00721, 0.00661]],
      None,
      None,
      [[0.0198, 0.01951, 0.01871], [0.01117, 0.02286, 0.02087]],
      [[0.01227, 0.01107, 0.01184], [0.00649, 0.00721, 0.00661]],
      None,
      None,
      [[0.0198, 0.01951, 0.01871], [0.01117, 0.02286, 0.02087]],
      [[0.01227, 0.01107, 0.01184], [0.00649, 0.00721, 0.00661]],
      [0.0213, 0.01951, 0.02286, 0.0198],
      [0.01313, 0.00994, 0.01273, 0.00569],
      [[0.0198, 0.01951, 0.01871], [0.01117, 0.02286, 0.02087]],
      [[0.01227, 0.01107, 0.01184], [0.00649, 0.00721, 0.00661]],
  ]
  expected_zps = [
      None,
      None,
      None,
      None,
      None,
      [1.02475, 1.20588, 1.27298, 1.78708],
      None,
      None,
      None,
      [[0.94431, 1.06127, 0.86091], [0.58712, 1.98041, 1.8048]],
      None,
      None,
      None,
      [[0.94431, 1.06127, 0.86091], [0.58712, 1.98041, 1.8048]],
      None,
      None,
      None,
      [[0.94431, 1.06127, 0.86091], [0.58712, 1.98041, 1.8048]],
      None,
      [1.02475, 1.20588, 1.27298, 1.78708],
      None,
      [[0.94431, 1.06127, 0.86091], [0.58712, 1.98041, 1.8048]],
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
      -1.37332,
      -1.37516,
      -1.09758,
      0.46481,
      -0.86398,
      -1.31529,
      -1.07837,
      -1.08196,
      -0.61218,
      -1.77383,
      1.73423,
      -0.55323,
      -2.43194,
      -0.75468,
      -1.62792,
      -1.11507,
      -0.73443,
      -1.90432,
      -1.92588,
      -0.8019,
      -0.10628,
      0.6925,
      -0.90212,
      0.44655,
  ]
  sample_input = [
      -0.90493,
      0.82371,
      -1.96339,
      0.00497,
      -0.73834,
      -0.7162,
      -1.69343,
      -0.40696,
      -2.10805,
      -1.75733,
      -0.28249,
      -1.74786,
      -2.11565,
      -0.11866,
      -1.35722,
      -1.36858,
      -1.18304,
      -2.30437,
  ]
  expected_results = [
      [
          [2.34845, 4.54375, -1.23836, 0.03905],
          [8.65739, 7.25295, 3.60215, 2.22014],
          [8.30482, 7.16612, 6.65223, 1.33384],
      ],
      None,
      None,
      None,
      [
          [3.83921, 0.40653, 6.67087, 1.63829],
          [6.91039, 7.15894, 13.32442, 2.07045],
          [7.96367, 4.90317, 14.19237, 3.40436],
      ],
      [
          [3.85624, 0.39379, 6.67449, 1.63893],
          [6.93882, 7.14854, 13.35028, 2.07538],
          [7.98725, 4.87839, 14.22751, 3.40519],
      ],
      None,
      None,
      [
          [[5.12962, 1.83672, 3.68653], [0.31909, 3.65034, 4.47828]],
          [[6.69144, 3.40052, 4.05312], [2.53812, 1.00863, 4.41341]],
          [[8.85845, 6.41645, 4.04424], [1.24059, 0.54949, 3.50833]],
      ],
      [
          [[5.12185, 1.82991, 3.68294], [0.31709, 3.64941, 4.47819]],
          [[6.69077, 3.39496, 4.04921], [2.53875, 1.00814, 4.40675]],
          [[8.85111, 6.41538, 4.03582], [1.2515, 0.55176, 3.49292]],
      ],
      None,
      None,
      [
          [[5.12962, 1.83672, 3.68653], [0.31909, 3.65034, 4.47828]],
          [[6.69144, 3.40052, 4.05312], [2.53812, 1.00863, 4.41341]],
          [[8.85845, 6.41645, 4.04424], [1.24059, 0.54949, 3.50833]],
      ],
      [
          [[5.12185, 1.82991, 3.68294], [0.31709, 3.64941, 4.47819]],
          [[6.69077, 3.39496, 4.04921], [2.53875, 1.00814, 4.40675]],
          [[8.85111, 6.41538, 4.03582], [1.2515, 0.55176, 3.49292]],
      ],
      None,
      None,
      [
          [[5.12962, 1.83672, 3.68653], [0.31909, 3.65034, 4.47828]],
          [[6.69144, 3.40052, 4.05312], [2.53812, 1.00863, 4.41341]],
          [[8.85845, 6.41645, 4.04424], [1.24059, 0.54949, 3.50833]],
      ],
      [
          [[5.12185, 1.82991, 3.68294], [0.31709, 3.64941, 4.47819]],
          [[6.69077, 3.39496, 4.04921], [2.53875, 1.00814, 4.40675]],
          [[8.85111, 6.41538, 4.03582], [1.2515, 0.55176, 3.49292]],
      ],
      [
          [3.83921, 0.40653, 6.67087, 1.63829],
          [6.91039, 7.15894, 13.32442, 2.07045],
          [7.96367, 4.90317, 14.19237, 3.40436],
      ],
      [
          [3.85624, 0.39379, 6.67449, 1.63893],
          [6.93882, 7.14854, 13.35028, 2.07538],
          [7.98725, 4.87839, 14.22751, 3.40519],
      ],
      [
          [[5.12962, 1.83672, 3.68653], [0.31909, 3.65034, 4.47828]],
          [[6.69144, 3.40052, 4.05312], [2.53812, 1.00863, 4.41341]],
          [[8.85845, 6.41645, 4.04424], [1.24059, 0.54949, 3.50833]],
      ],
      [
          [[5.12185, 1.82991, 3.68294], [0.31709, 3.64941, 4.47819]],
          [[6.69077, 3.39496, 4.04921], [2.53875, 1.00814, 4.40675]],
          [[8.85111, 6.41538, 4.03582], [1.2515, 0.55176, 3.49292]],
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


class AttentionProjectionAQTTest(quantization_test_util.QuantizationTestCase):
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
    self.assertAllClose(updated_w_f_tensor, updated_w_q_tensor, atol=1e-4)

    # 2. Value check.
    self.assertNestedListClose(updated_w_q, expected_trained_weight)

  # Test the training with AQT quantization.
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
    # Not supported.
    if is_output_projection and use_nhd_shape and not is_weight_symmetric:
      return

    quantization_option = QuantizationHParams(
        quantization_type=QuantizationType.AQT,
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

  # Test AQT weight quantization.
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
    # TODO(dhchoi): Unblock this once the fix is applied.
    if attention_combine_dims:
      return

    # Not supported.
    if is_output_projection and use_nhd_shape and not is_weight_symmetric:
      return

    quantization_option = QuantizationHParams(
        quantization_type=QuantizationType.AQT,
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
    p.input_dim = 4
    p.num_heads = 2
    p.dim_per_head = 3

    layer = instantiate(p)
    if is_output_projection:
      inputs = np.random.normal(1.5, 2.0, [3, 2, 3]).astype(np.float32)
    else:
      inputs = np.random.normal(1.5, 2.0, [3, 4]).astype(np.float32)

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
    # Not supported.
    if is_output_projection and use_nhd_shape and not is_weight_symmetric:
      return

    quantization_option = QuantizationHParams(
        quantization_type=QuantizationType.AQT,
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
    # TODO(dhchoi): Unblock this once the fix is applied.
    if attention_combine_dims:
      return

    # Not supported.
    if is_output_projection and use_nhd_shape and not is_weight_symmetric:
      return

    quantization_option = QuantizationHParams(
        quantization_type=QuantizationType.AQT,
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
