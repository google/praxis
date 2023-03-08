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

"""PTQ Tests for quantized CombinedQKVProjection layer."""
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


def _add_expected_quantized_weights(cur_key, cur_samples):
  target_weight = [
      -1.26162,
      -0.40528,
      0.54522,
      -1.17459,
      0.21987,
      -1.35148,
      -1.79563,
      -0.05499,
      -1.09215,
      -2.16327,
      -1.778,
      -2.21668,
      -2.74413,
      -1.21846,
      -1.88004,
      -0.95879,
      -1.47911,
      -0.94022,
      -0.723,
      -1.75646,
      -1.57047,
      1.59422,
      -1.50573,
      -2.38593,
      0.31808,
      -1.94686,
      -0.64351,
      -1.99628,
      -0.4796,
      -2.58131,
      -1.23005,
      -0.96793,
      -0.08183,
      -1.12138,
      -1.29185,
      -1.10986,
      -0.39464,
      -1.15535,
      -2.60466,
      -1.62579,
      -0.35469,
      -1.53606,
      0.82421,
      -2.33424,
      -0.48014,
      0.2766,
      -0.82778,
      -0.94868,
      -1.01986,
      -0.75793,
      0.66379,
      -1.58638,
      -0.65796,
      -0.95057,
      -0.88032,
      -0.74686,
      0.25205,
      -0.51437,
      -1.8619,
      0.5897,
      0.52245,
      -1.9469,
      -1.81492,
      0.108,
      -0.89185,
      -1.53911,
      -1.85232,
      -0.84103,
      -1.7795,
      -1.02871,
      0.37477,
      1.72783,
  ]
  expected_weights = [
      [
          [
              [-58, -29, 37, -69, 16, -72],
              [-83, -4, -74, -127, -127, -118],
              [-127, -88, -127, -56, -106, -50],
              [-33, -127, -106, 94, -108, -127],
          ],
          [
              [33, -106, -31, -127, -47, -127],
              [-127, -53, -4, -71, -127, -55],
              [-41, -63, -127, -103, -35, -76],
              [85, -127, -23, 18, -81, -47],
          ],
          [
              [-70, -49, 46, -127, -45, -70],
              [-60, -49, 18, -41, -127, 43],
              [36, -127, -127, 9, -61, -113],
              [-127, -55, -125, -82, 26, 127],
          ],
      ],
      [
          [
              [59, 75, 127, -61, 127, 54],
              [-8, 127, -45, -128, -128, -98],
              [-128, -47, -128, -46, -90, 127],
              [127, -128, -95, 127, -93, -128],
          ],
          [
              [64, -56, 70, -128, 93, -128],
              [-128, 127, 127, -30, -128, 102],
              [-24, 92, -128, -86, 127, 35],
              [127, -128, 87, 127, -2, 127],
          ],
          [
              [-39, 125, 127, -128, 9, -82],
              [-24, 127, 85, 33, -128, 38],
              [127, -128, -128, 127, -17, -128],
              [-128, 107, -124, -44, 127, 127],
          ],
      ],
      [
          [
              [[-58, -29, 37], [-69, 16, -72]],
              [[-83, -4, -74], [-127, -127, -118]],
              [[-127, -88, -127], [-56, -106, -50]],
              [[-33, -127, -106], [94, -108, -127]],
          ],
          [
              [[33, -106, -31], [-127, -47, -127]],
              [[-127, -53, -4], [-71, -127, -55]],
              [[-41, -63, -127], [-103, -35, -76]],
              [[85, -127, -23], [18, -81, -47]],
          ],
          [
              [[-70, -49, 46], [-127, -45, -70]],
              [[-60, -49, 18], [-41, -127, 43]],
              [[36, -127, -127], [9, -61, -113]],
              [[-127, -55, -125], [-82, 26, 127]],
          ],
      ],
      [
          [
              [[59, 75, 127], [-61, 127, 54]],
              [[-8, 127, -45], [-128, -128, -98]],
              [[-128, -47, -128], [-46, -90, 127]],
              [[127, -128, -95], [127, -93, -128]],
          ],
          [
              [[64, -56, 70], [-128, 93, -128]],
              [[-128, 127, 127], [-30, -128, 102]],
              [[-24, 92, -128], [-86, 127, 35]],
              [[127, -128, 87], [127, -2, 127]],
          ],
          [
              [[-39, 125, 127], [-128, 9, -82]],
              [[-24, 127, 85], [33, -128, 38]],
              [[127, -128, -128], [127, -17, -128]],
              [[-128, 107, -124], [-44, 127, 127]],
          ],
      ],
      [
          [
              [-58, -29, 37, -69, 16, -72],
              [-83, -4, -74, -127, -127, -118],
              [-127, -88, -127, -56, -106, -50],
              [-33, -127, -106, 94, -108, -127],
          ],
          [
              [33, -106, -31, -127, -47, -127],
              [-127, -53, -4, -71, -127, -55],
              [-41, -63, -127, -103, -35, -76],
              [85, -127, -23, 18, -81, -47],
          ],
          [
              [-70, -49, 46, -127, -45, -70],
              [-60, -49, 18, -41, -127, 43],
              [36, -127, -127, 9, -61, -113],
              [-127, -55, -125, -82, 26, 127],
          ],
      ],
      [
          [
              [59, 75, 127, -61, 127, 54],
              [-8, 127, -45, -128, -128, -98],
              [-128, -47, -128, -46, -90, 127],
              [127, -128, -95, 127, -93, -128],
          ],
          [
              [64, -56, 70, -128, 93, -128],
              [-128, 127, 127, -30, -128, 102],
              [-24, 92, -128, -86, 127, 35],
              [127, -128, 87, 127, -2, 127],
          ],
          [
              [-39, 125, 127, -128, 9, -82],
              [-24, 127, 85, 33, -128, 38],
              [127, -128, -128, 127, -17, -128],
              [-128, 107, -124, -44, 127, 127],
          ],
      ],
      [
          [
              [[-58, -29, 37], [-69, 16, -72]],
              [[-83, -4, -74], [-127, -127, -118]],
              [[-127, -88, -127], [-56, -106, -50]],
              [[-33, -127, -106], [94, -108, -127]],
          ],
          [
              [[33, -106, -31], [-127, -47, -127]],
              [[-127, -53, -4], [-71, -127, -55]],
              [[-41, -63, -127], [-103, -35, -76]],
              [[85, -127, -23], [18, -81, -47]],
          ],
          [
              [[-70, -49, 46], [-127, -45, -70]],
              [[-60, -49, 18], [-41, -127, 43]],
              [[36, -127, -127], [9, -61, -113]],
              [[-127, -55, -125], [-82, 26, 127]],
          ],
      ],
      [
          [
              [[59, 75, 127], [-61, 127, 54]],
              [[-8, 127, -45], [-128, -128, -98]],
              [[-128, -47, -128], [-46, -90, 127]],
              [[127, -128, -95], [127, -93, -128]],
          ],
          [
              [[64, -56, 70], [-128, 93, -128]],
              [[-128, 127, 127], [-30, -128, 102]],
              [[-24, 92, -128], [-86, 127, 35]],
              [[127, -128, 87], [127, -2, 127]],
          ],
          [
              [[-39, 125, 127], [-128, 9, -82]],
              [[-24, 127, 85], [33, -128, 38]],
              [[127, -128, -128], [127, -17, -128]],
              [[-128, 107, -124], [-44, 127, 127]],
          ],
      ],
  ]
  expected_scales = [
      [
          [0.02161, 0.01383, 0.0148, 0.01703, 0.014, 0.01879],
          [0.00969, 0.01838, 0.02051, 0.01572, 0.01017, 0.02033],
          [0.01459, 0.01533, 0.01429, 0.01249, 0.01466, 0.0136],
      ],
      [
          [0.00793, 0.00667, 0.00951, 0.01474, 0.00783, 0.00567],
          [0.00806, 0.00536, 0.00989, 0.00891, 0.00368, 0.0064],
          [0.00931, 0.00471, 0.00972, 0.00664, 0.00877, 0.01281],
      ],
      [
          [[0.02161, 0.01383, 0.0148], [0.01703, 0.014, 0.01879]],
          [[0.00969, 0.01838, 0.02051], [0.01572, 0.01017, 0.02033]],
          [[0.01459, 0.01533, 0.01429], [0.01249, 0.01466, 0.0136]],
      ],
      [
          [[0.00793, 0.00667, 0.00951], [0.01474, 0.00783, 0.00567]],
          [[0.00806, 0.00536, 0.00989], [0.00891, 0.00368, 0.0064]],
          [[0.00931, 0.00471, 0.00972], [0.00664, 0.00877, 0.01281]],
      ],
      [
          [0.02161, 0.01383, 0.0148, 0.01703, 0.014, 0.01879],
          [0.00969, 0.01838, 0.02051, 0.01572, 0.01017, 0.02033],
          [0.01459, 0.01533, 0.01429, 0.01249, 0.01466, 0.0136],
      ],
      [
          [0.00793, 0.00667, 0.00951, 0.01474, 0.00783, 0.00567],
          [0.00806, 0.00536, 0.00989, 0.00891, 0.00368, 0.0064],
          [0.00931, 0.00471, 0.00972, 0.00664, 0.00877, 0.01281],
      ],
      [
          [[0.02161, 0.01383, 0.0148], [0.01703, 0.014, 0.01879]],
          [[0.00969, 0.01838, 0.02051], [0.01572, 0.01017, 0.02033]],
          [[0.01459, 0.01533, 0.01429], [0.01249, 0.01466, 0.0136]],
      ],
      [
          [[0.00793, 0.00667, 0.00951], [0.01474, 0.00783, 0.00567]],
          [[0.00806, 0.00536, 0.00989], [0.00891, 0.00368, 0.0064]],
          [[0.00931, 0.00471, 0.00972], [0.00664, 0.00877, 0.01281]],
      ],
  ]
  expected_zps = [
      None,
      [
          [1.7296, 0.90239, 0.66265, 0.27716, 0.77515, 1.66024],
          [0.19889, 1.64841, 1.3383, 0.85538, 0.82143, 1.76179],
          [0.66028, 1.34453, 0.5707, 0.73587, 0.73918, -0.10077],
      ],
      None,
      [
          [[1.7296, 0.90239, 0.66265], [0.27716, 0.77515, 1.66024]],
          [[0.19889, 1.64841, 1.3383], [0.85538, 0.82143, 1.76179]],
          [[0.66028, 1.34453, 0.5707], [0.73587, 0.73918, -0.10077]],
      ],
      None,
      [
          [1.7296, 0.90239, 0.66265, 0.27716, 0.77515, 1.66024],
          [0.19889, 1.64841, 1.3383, 0.85538, 0.82143, 1.76179],
          [0.66028, 1.34453, 0.5707, 0.73587, 0.73918, -0.10077],
      ],
      None,
      [
          [[1.7296, 0.90239, 0.66265], [0.27716, 0.77515, 1.66024]],
          [[0.19889, 1.64841, 1.3383], [0.85538, 0.82143, 1.76179]],
          [[0.66028, 1.34453, 0.5707], [0.73587, 0.73918, -0.10077]],
      ],
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
      -2.84852,
      0.62403,
      -0.98237,
      -1.47302,
      -0.91375,
      -0.85424,
      -1.42235,
      -1.75659,
      1.33605,
      -1.02721,
      -2.16927,
      -3.21512,
      -1.14124,
      -3.10371,
      -1.56947,
      -0.7308,
      -1.92597,
      -0.37188,
      -0.33085,
      -1.84185,
      -1.59925,
      -1.3135,
      -0.42755,
      -0.97155,
      0.38865,
      -0.77274,
      -2.51332,
      -2.61059,
      0.81915,
      0.02247,
      -1.8461,
      -1.78431,
      -1.86007,
      -1.70471,
      -0.66714,
      -1.39178,
      -0.80946,
      -2.03366,
      -1.87245,
      0.11786,
      -1.50098,
      1.16665,
      -1.19206,
      -2.11419,
      -0.39098,
      -2.68357,
      -0.92308,
      -0.0871,
      0.91143,
      0.40047,
      -2.09876,
      0.75099,
      -0.19849,
      -0.69452,
      -1.69859,
      -0.68488,
      0.15054,
      0.21664,
      2.11651,
      -0.15126,
      -0.76478,
      -0.22708,
      -1.11234,
      0.182,
      -2.09573,
      1.31382,
      0.12477,
      -1.78133,
      -2.72183,
      -2.02121,
      -1.49994,
      -0.27957,
  ]
  sample_input = [
      -1.24926,
      -2.03837,
      -2.83991,
      -0.20428,
      -1.05729,
      -0.16742,
      -1.43561,
      -0.37401,
      -0.6705,
      0.42014,
      -0.63765,
      -0.28283,
  ]

  expected_proj_qs = [
      [
          [[9.75613, 11.98158, 3.30311], [6.28723, 11.12136, 8.90384]],
          [[5.0163, 4.76402, 3.67289], [3.26944, 4.25096, 2.35328]],
          [[2.14081, 1.33217, 2.6754], [1.39059, 1.04712, -0.25945]],
      ],
      [
          [[9.77321, 11.99697, 3.26833], [6.2786, 11.11129, 8.8722]],
          [[5.01157, 4.78101, 3.65444], [3.26966, 4.24909, 2.33653]],
          [[2.13127, 1.34398, 2.6669], [1.39322, 1.0478, -0.26739]],
      ],
      [
          [[9.75613, 11.98158, 3.30311], [6.28723, 11.12136, 8.90384]],
          [[5.0163, 4.76402, 3.67289], [3.26944, 4.25096, 2.35328]],
          [[2.14081, 1.33217, 2.6754], [1.39059, 1.04712, -0.25945]],
      ],
      [
          [[9.77321, 11.99697, 3.26833], [6.2786, 11.11129, 8.8722]],
          [[5.01157, 4.78101, 3.65444], [3.26966, 4.24909, 2.33653]],
          [[2.13127, 1.34398, 2.6669], [1.39322, 1.0478, -0.26739]],
      ],
      [
          [[9.75613, 11.98158, 3.30311], [6.28723, 11.12136, 8.90384]],
          [[5.0163, 4.76402, 3.67289], [3.26944, 4.25096, 2.35328]],
          [[2.14081, 1.33217, 2.6754], [1.39059, 1.04712, -0.25945]],
      ],
      [
          [[9.77321, 11.99697, 3.26833], [6.2786, 11.11129, 8.8722]],
          [[5.01157, 4.78101, 3.65444], [3.26966, 4.24909, 2.33653]],
          [[2.13127, 1.34398, 2.6669], [1.39322, 1.0478, -0.26739]],
      ],
      [
          [[9.75613, 11.98158, 3.30311], [6.28723, 11.12136, 8.90384]],
          [[5.0163, 4.76402, 3.67289], [3.26944, 4.25096, 2.35328]],
          [[2.14081, 1.33217, 2.6754], [1.39059, 1.04712, -0.25945]],
      ],
      [
          [[9.77321, 11.99697, 3.26833], [6.2786, 11.11129, 8.8722]],
          [[5.01157, 4.78101, 3.65444], [3.26966, 4.24909, 2.33653]],
          [[2.13127, 1.34398, 2.6669], [1.39322, 1.0478, -0.26739]],
      ],
  ]
  expected_proj_ks = [
      [
          [[5.82799, 10.7871, 12.35168], [6.95024, 4.7813, -0.47147]],
          [[1.50855, 4.81424, 5.81579], [3.87851, 1.7482, -1.42504]],
          [[-0.18259, 1.65807, 2.21437], [1.71588, 0.39297, -1.31536]],
      ],
      [
          [[5.82631, 10.81121, 12.33706], [6.94698, 4.78029, -0.48693]],
          [[1.50848, 4.82808, 5.80375], [3.87607, 1.74349, -1.43332]],
          [[-0.18222, 1.66486, 2.20654], [1.71493, 0.38875, -1.31926]],
      ],
      [
          [[5.82799, 10.7871, 12.35168], [6.95024, 4.7813, -0.47147]],
          [[1.50855, 4.81424, 5.81579], [3.87851, 1.7482, -1.42504]],
          [[-0.18259, 1.65807, 2.21437], [1.71588, 0.39297, -1.31536]],
      ],
      [
          [[5.82631, 10.81121, 12.33706], [6.94698, 4.78029, -0.48693]],
          [[1.50848, 4.82808, 5.80375], [3.87607, 1.74349, -1.43332]],
          [[-0.18222, 1.66486, 2.20654], [1.71493, 0.38875, -1.31926]],
      ],
      [
          [[5.82799, 10.7871, 12.35168], [6.95024, 4.7813, -0.47147]],
          [[1.50855, 4.81424, 5.81579], [3.87851, 1.7482, -1.42504]],
          [[-0.18259, 1.65807, 2.21437], [1.71588, 0.39297, -1.31536]],
      ],
      [
          [[5.82631, 10.81121, 12.33706], [6.94698, 4.78029, -0.48693]],
          [[1.50848, 4.82808, 5.80375], [3.87607, 1.74349, -1.43332]],
          [[-0.18222, 1.66486, 2.20654], [1.71493, 0.38875, -1.31926]],
      ],
      [
          [[5.82799, 10.7871, 12.35168], [6.95024, 4.7813, -0.47147]],
          [[1.50855, 4.81424, 5.81579], [3.87851, 1.7482, -1.42504]],
          [[-0.18259, 1.65807, 2.21437], [1.71588, 0.39297, -1.31536]],
      ],
      [
          [[5.82631, 10.81121, 12.33706], [6.94698, 4.78029, -0.48693]],
          [[1.50848, 4.82808, 5.80375], [3.87607, 1.74349, -1.43332]],
          [[-0.18222, 1.66486, 2.20654], [1.71493, 0.38875, -1.31926]],
      ],
  ]
  expected_proj_vs = [
      [
          [[4.46662, 1.89401, 6.03898], [-1.4729, 2.20537, -2.49188]],
          [[0.37222, 0.67342, 4.81343], [-0.32354, 3.43263, -1.02286]],
          [[-0.87138, 0.08543, 2.95173], [0.0521, 2.7865, -0.35922]],
      ],
      [
          [[4.47725, 1.89872, 6.03076], [-1.49756, 2.18906, -2.49892]],
          [[0.37597, 0.67945, 4.81056], [-0.34117, 3.42304, -1.02289]],
          [[-0.87045, 0.08993, 2.95112], [0.04144, 2.7816, -0.35719]],
      ],
      [
          [[4.46662, 1.89401, 6.03898], [-1.4729, 2.20537, -2.49188]],
          [[0.37222, 0.67342, 4.81343], [-0.32354, 3.43263, -1.02286]],
          [[-0.87138, 0.08543, 2.95173], [0.0521, 2.7865, -0.35922]],
      ],
      [
          [[4.47725, 1.89872, 6.03076], [-1.49756, 2.18906, -2.49892]],
          [[0.37597, 0.67945, 4.81056], [-0.34117, 3.42304, -1.02289]],
          [[-0.87045, 0.08993, 2.95112], [0.04144, 2.7816, -0.35719]],
      ],
      [
          [[4.46662, 1.89401, 6.03898], [-1.4729, 2.20537, -2.49188]],
          [[0.37222, 0.67342, 4.81343], [-0.32354, 3.43263, -1.02286]],
          [[-0.87138, 0.08543, 2.95173], [0.0521, 2.7865, -0.35922]],
      ],
      [
          [[4.47725, 1.89872, 6.03076], [-1.49756, 2.18906, -2.49892]],
          [[0.37597, 0.67945, 4.81056], [-0.34117, 3.42304, -1.02289]],
          [[-0.87045, 0.08993, 2.95112], [0.04144, 2.7816, -0.35719]],
      ],
      [
          [[4.46662, 1.89401, 6.03898], [-1.4729, 2.20537, -2.49188]],
          [[0.37222, 0.67342, 4.81343], [-0.32354, 3.43263, -1.02286]],
          [[-0.87138, 0.08543, 2.95173], [0.0521, 2.7865, -0.35922]],
      ],
      [
          [[4.47725, 1.89872, 6.03076], [-1.49756, 2.18906, -2.49892]],
          [[0.37597, 0.67945, 4.81056], [-0.34117, 3.42304, -1.02289]],
          [[-0.87045, 0.08993, 2.95112], [0.04144, 2.7816, -0.35719]],
      ],
  ]

  updated_key = cur_key + [
      'sample_weight',
      'sample_input',
      'expected_proj_q',
      'expected_proj_k',
      'expected_proj_v',
  ]
  ret = []
  for sample, expected_proj_q, expected_proj_k, expected_proj_v in zip(
      cur_samples, expected_proj_qs, expected_proj_ks, expected_proj_vs
  ):
    sample.append(sample_weight)
    sample.append(sample_input)
    sample.append(expected_proj_q)
    sample.append(expected_proj_k)
    sample.append(expected_proj_v)

    ret.append(sample)

  return updated_key, ret


class CombinedQKVProjectionPTQTest(quantization_test_util.QuantizationTestCase):
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
    combined_qkv_f = instantiate(p_f)
    combined_qkv_q = instantiate(p_q)

    def update(combined_qkv_layer, params, inputs, targets):
      def loss(params, inputs, targets):
        proj_q, proj_k, proj_v = combined_qkv_layer.apply(params, inputs)
        ans_q, ans_k, ans_v = targets
        return (
            -jnp.mean(jnp.abs(proj_q - ans_q))
            -jnp.mean(jnp.abs(proj_k - ans_k))
            -jnp.mean(jnp.abs(proj_v - ans_v))
        )

      grads = jax.grad(loss)(params, inputs, targets)

      out_params = dict()
      out_params[base_layer.PARAMS] = dict()
      for k, v in params[base_layer.PARAMS].items():
        v_grad = grads[base_layer.PARAMS][k]
        out_params[base_layer.PARAMS][k] = v - step_size * v_grad

      return out_params

    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars_f = combined_qkv_f.init(prng_key, inputs)
    initial_vars_q = combined_qkv_q.init(prng_key, inputs)
    proj_q_f, proj_k_f, proj_v_f = combined_qkv_f.apply(initial_vars_f, inputs)

    pseudo_answer_q = np.random.normal(0.0, 2.0, proj_q_f.shape)
    pseudo_answer_k = np.random.normal(0.0, 2.0, proj_k_f.shape)
    pseudo_answer_v = np.random.normal(0.0, 2.0, proj_v_f.shape)
    pseudo_answer = [pseudo_answer_q, pseudo_answer_k, pseudo_answer_v]

    updated_vars_f = update(
        combined_qkv_f, initial_vars_f, inputs, pseudo_answer
    )
    updated_vars_q = update(
        combined_qkv_q, initial_vars_q, inputs, pseudo_answer
    )

    # 1. Check if the trained weights are the same.
    for k, v in updated_vars_f[base_layer.PARAMS].items():
      self.assertAllClose(v, updated_vars_q[base_layer.PARAMS][k])

    # 2. Check if the inference result with updated results are the same.
    outputs_f = combined_qkv_f.apply(updated_vars_f, inputs)
    outputs_q = combined_qkv_q.apply(updated_vars_q, inputs)
    self.assertAllClose(outputs_f, outputs_q)

  # See if the training results of PTQ-quantized model and original model are
  # the same.
  @parameterized.parameters(generate_quantization_test_config())
  def test_train(
      self,
      use_bias,
      attention_combine_dims,
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
      p.input_dim = 16
      p.num_heads = 2
      p.dim_per_head = 5

    inputs = np.random.normal(1.5, 2.0, [5, 16]).astype(np.float32)
    self.train_and_compare(p_f, p_q, inputs)

  # Test PTQ weight quantization.
  @parameterized.parameters(
      generate_quantization_test_config([_add_expected_quantized_weights])
  )
  def test_weight_quantization(
      self,
      use_bias,
      attention_combine_dims,
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

    quantization_option = QuantizationHParams(
        quantization_type=QuantizationType.PTQ,
        mode=QuantizationMode.MATERIALIZE,
        weight_params=quantization_hparams.WeightQuantizationParams(
            use_symmetric=is_weight_symmetric
        ),
    )

    p = pax_fiddle.Config(
        qattentions.CombinedQKVProjectionLayer,
        name='_combined_qkv_q',
        mesh_axis_names=['replica', 'mdl', 'data'],
        weight_split_dims_mapping=base_layer.BaseLayer.WeightSharding(
            wt=['mdl', 'data']
        ),
        use_bias=use_bias,
        attention_combine_dims=attention_combine_dims,
        quantization=quantization_option,
    )
    p.input_dim = 4
    p.num_heads = 2
    p.dim_per_head = 3

    layer = instantiate(p)
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
      attention_combine_dims,
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
  @parameterized.parameters(
      generate_quantization_test_config([_add_expected_quantization_results])
  )
  def test_inference_call(
      self,
      use_bias,
      attention_combine_dims,
      is_weight_symmetric,
      sample_weight,
      sample_input,
      expected_proj_q,
      expected_proj_k,
      expected_proj_v
  ):
    # There exist a shape mismatch bug when the flag attention_combine_dims is
    # turned on.
    # TODO(dhchoi): Unblock this once the fix is applied.
    if attention_combine_dims:
      return

    quantization_option = QuantizationHParams(
        quantization_type=QuantizationType.PTQ,
        mode=QuantizationMode.INFERENCE,
        weight_params=quantization_hparams.WeightQuantizationParams(
            use_symmetric=is_weight_symmetric
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

    layer_f = instantiate(p_f)
    layer_q = instantiate(p_q)
    inputs = np.array(sample_input).astype(np.float32).reshape([3, 4])

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

      result_f = layer_f.apply(initial_vars_f, inputs)
      result_q = layer_q.apply(initial_vars_q, inputs)

    # Since they are quantized results, they may not be exactly equal,
    # but they should be similar in some way.
    for proj_f, proj_q in zip(result_f, result_q):
      self.assertAllClose(proj_f, proj_q, atol=1e-1)

    proj_q_q, proj_k_q, proj_v_q = [to_list(proj) for proj in result_q]
    self.assertNestedListClose(proj_q_q, expected_proj_q)
    self.assertNestedListClose(proj_k_q, expected_proj_k)
    self.assertNestedListClose(proj_v_q, expected_proj_v)


if __name__ == '__main__':
  absltest.main()

