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

"""AQT Tests for quantized CombinedQKVProjection layer."""
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
      -0.59239,
      -1.72215,
      -0.55673,
      -1.72355,
      -2.56228,
      -0.61437,
      0.003,
      -0.47187,
      0.93212,
      -1.69607,
      1.41366,
      -0.37353,
  ]
  sample_init_weight = [
      -0.32352,
      -0.97161,
      -1.62819,
      -0.18048,
      -1.49884,
      -0.13276,
      -1.09476,
      0.32922,
      -0.40419,
      -0.25421,
      -0.56297,
      0.10578,
      -1.08966,
      -1.18284,
      -0.76431,
      1.54825,
      -0.53929,
      -2.90381,
      -1.04242,
      -2.17317,
      -2.33143,
      -1.64313,
      -2.22273,
      -0.43115,
      -1.55855,
      -1.79694,
      -0.61316,
      -1.87027,
      -0.61849,
      -0.91998,
      0.23792,
      -1.66416,
      -2.29665,
      -0.88466,
      -1.52589,
      -1.28175,
      -1.26917,
      -1.32261,
      -2.08742,
      -1.61839,
      0.01025,
      -0.09026,
      -1.55829,
      -2.07232,
      -1.6655,
      -1.18358,
      -0.39116,
      -2.48106,
      0.59847,
      -0.05372,
      -0.32531,
      -1.69596,
      -1.13281,
      -0.0813,
      -1.41043,
      -1.35228,
      -1.28001,
      -0.14764,
      -0.68219,
      -1.25881,
      -0.9347,
      -1.27462,
      0.54037,
      -0.14835,
      -1.51232,
      -2.14686,
      -1.43539,
      -1.89926,
      -1.37009,
      -1.03534,
      -0.4249,
      -0.43537,
  ]
  pseudo_answer_q = [
      -2.24506,
      -1.3769,
      0.00378,
      -1.80912,
      -0.72076,
      -1.48066,
      -1.1879,
      -0.81428,
      -1.34753,
      -2.96608,
      -0.99648,
      0.20376,
      -1.79093,
      -2.7384,
      0.67075,
      -0.35657,
      -1.73234,
      -2.21357,
  ]
  pseudo_answer_k = [
      0.59968,
      0.9523,
      -0.65297,
      -2.17226,
      -1.91835,
      -1.75239,
      -2.64038,
      -2.4811,
      0.43448,
      -1.59541,
      -0.21765,
      -0.84044,
      -0.27934,
      -0.3125,
      -0.13864,
      -1.96368,
      -1.21525,
      -1.7714,
  ]
  pseudo_answer_v = [
      -0.32063,
      -0.42031,
      -1.4065,
      -1.5592,
      -1.71117,
      0.76558,
      -0.57686,
      -1.63547,
      -0.32274,
      -2.64875,
      0.0998,
      -1.25093,
      -1.07829,
      -1.91498,
      -0.5928,
      0.13895,
      0.56315,
      0.00803,
  ]
  expected_trained_weights = [
      [
          [
              [-0.32475, -0.97284, -1.63046, -0.18171, -1.50007, -0.13503],
              [-1.09701, 0.32698, -0.40455, -0.25645, -0.56521, 0.10542],
              [-1.08918, -1.18236, -0.7654, 1.54873, -0.53881, -2.90491],
              [-1.04385, -2.1746, -2.33245, -1.64457, -2.22416, -0.43216],
          ],
          [
              [-1.56082, -1.79817, -0.61439, -1.87254, -0.61972, -0.92121],
              [0.23756, -1.6664, -2.29889, -0.88502, -1.52813, -1.28399],
              [-1.27026, -1.32213, -2.08694, -1.61948, 0.01073, -0.08978],
              [-1.5593, -2.07375, -1.66693, -1.18459, -0.39259, -2.48249],
          ],
          [
              [0.59724, -0.05495, -0.32654, -1.69823, -1.13508, -0.08357],
              [-1.41267, -1.35452, -1.28225, -0.148, -0.68255, -1.25917],
              [-0.93422, -1.27414, 0.54085, -0.14944, -1.51341, -2.14796],
              [-1.43682, -1.90068, -1.37152, -1.03635, -0.42591, -0.43638],
          ],
      ],
      [
          [
              [-0.32475, -0.97284, -1.63046, -0.18171, -1.50007, -0.13503],
              [-1.097, 0.32698, -0.40454, -0.25645, -0.56521, 0.10542],
              [-1.08918, -1.18236, -0.7654, 1.54873, -0.53881, -2.9049],
              [-1.04385, -2.1746, -2.33244, -1.64456, -2.22416, -0.43216],
          ],
          [
              [-1.56133, -1.79817, -0.6144, -1.87254, -0.61972, -0.92121],
              [0.23756, -1.6664, -2.29889, -0.88502, -1.52813, -1.28399],
              [-1.27026, -1.32214, -2.08694, -1.61948, 0.01073, -0.08979],
              [-1.5588, -2.07374, -1.66693, -1.18459, -0.39259, -2.48248],
          ],
          [
              [0.59724, -0.05496, -0.32654, -1.69823, -1.13508, -0.08357],
              [-1.41267, -1.35452, -1.28225, -0.14854, -0.68255, -1.25917],
              [-0.93422, -1.27414, 0.54085, -0.1489, -1.51341, -2.14795],
              [-1.43682, -1.90068, -1.37152, -1.03635, -0.42591, -0.43638],
          ],
      ],
      [
          [
              [[-0.32475, -0.97284, -1.63046], [-0.18171, -1.50007, -0.13503]],
              [[-1.09701, 0.32698, -0.40455], [-0.25645, -0.56521, 0.10542]],
              [[-1.08918, -1.18236, -0.7654], [1.54873, -0.53881, -2.90491]],
              [[-1.04385, -2.1746, -2.33245], [-1.64457, -2.22416, -0.43216]],
          ],
          [
              [[-1.56082, -1.79817, -0.61439], [-1.87254, -0.61972, -0.92121]],
              [[0.23756, -1.6664, -2.29889], [-0.88502, -1.52813, -1.28399]],
              [[-1.27026, -1.32213, -2.08694], [-1.61948, 0.01073, -0.08978]],
              [[-1.5593, -2.07375, -1.66693], [-1.18459, -0.39259, -2.48249]],
          ],
          [
              [[0.59724, -0.05495, -0.32654], [-1.69823, -1.13508, -0.08357]],
              [[-1.41267, -1.35452, -1.28225], [-0.148, -0.68255, -1.25917]],
              [[-0.93422, -1.27414, 0.54085], [-0.14944, -1.51341, -2.14796]],
              [[-1.43682, -1.90068, -1.37152], [-1.03635, -0.42591, -0.43638]],
          ],
      ],
      [
          [
              [[-0.32475, -0.97284, -1.63046], [-0.18171, -1.50007, -0.13503]],
              [[-1.097, 0.32698, -0.40454], [-0.25645, -0.56521, 0.10542]],
              [[-1.08918, -1.18236, -0.7654], [1.54873, -0.53881, -2.9049]],
              [[-1.04385, -2.1746, -2.33244], [-1.64456, -2.22416, -0.43216]],
          ],
          [
              [[-1.56133, -1.79817, -0.6144], [-1.87254, -0.61972, -0.92121]],
              [[0.23756, -1.6664, -2.29889], [-0.88502, -1.52813, -1.28399]],
              [[-1.27026, -1.32214, -2.08694], [-1.61948, 0.01073, -0.08979]],
              [[-1.5588, -2.07374, -1.66693], [-1.18459, -0.39259, -2.48248]],
          ],
          [
              [[0.59724, -0.05496, -0.32654], [-1.69823, -1.13508, -0.08357]],
              [[-1.41267, -1.35452, -1.28225], [-0.14854, -0.68255, -1.25917]],
              [[-0.93422, -1.27414, 0.54085], [-0.1489, -1.51341, -2.14795]],
              [[-1.43682, -1.90068, -1.37152], [-1.03635, -0.42591, -0.43638]],
          ],
      ],
      [
          [
              [-0.32475, -0.97284, -1.63046, -0.18171, -1.50007, -0.13503],
              [-1.09701, 0.32698, -0.40455, -0.25645, -0.56521, 0.10542],
              [-1.08918, -1.18236, -0.7654, 1.54873, -0.53881, -2.90491],
              [-1.04385, -2.1746, -2.33245, -1.64457, -2.22416, -0.43216],
          ],
          [
              [-1.56082, -1.79817, -0.61439, -1.87254, -0.61972, -0.92121],
              [0.23756, -1.6664, -2.29889, -0.88502, -1.52813, -1.28399],
              [-1.27026, -1.32213, -2.08694, -1.61948, 0.01073, -0.08978],
              [-1.5593, -2.07375, -1.66693, -1.18459, -0.39259, -2.48249],
          ],
          [
              [0.59724, -0.05495, -0.32654, -1.69823, -1.13508, -0.08357],
              [-1.41267, -1.35452, -1.28225, -0.148, -0.68255, -1.25917],
              [-0.93422, -1.27414, 0.54085, -0.14944, -1.51341, -2.14796],
              [-1.43682, -1.90068, -1.37152, -1.03635, -0.42591, -0.43638],
          ],
      ],
      [
          [
              [-0.32475, -0.97284, -1.63046, -0.18171, -1.50007, -0.13503],
              [-1.097, 0.32698, -0.40454, -0.25645, -0.56521, 0.10542],
              [-1.08918, -1.18236, -0.7654, 1.54873, -0.53881, -2.9049],
              [-1.04385, -2.1746, -2.33244, -1.64456, -2.22416, -0.43216],
          ],
          [
              [-1.56133, -1.79817, -0.6144, -1.87254, -0.61972, -0.92121],
              [0.23756, -1.6664, -2.29889, -0.88502, -1.52813, -1.28399],
              [-1.27026, -1.32214, -2.08694, -1.61948, 0.01073, -0.08979],
              [-1.5588, -2.07374, -1.66693, -1.18459, -0.39259, -2.48248],
          ],
          [
              [0.59724, -0.05496, -0.32654, -1.69823, -1.13508, -0.08357],
              [-1.41267, -1.35452, -1.28225, -0.14854, -0.68255, -1.25917],
              [-0.93422, -1.27414, 0.54085, -0.1489, -1.51341, -2.14795],
              [-1.43682, -1.90068, -1.37152, -1.03635, -0.42591, -0.43638],
          ],
      ],
      [
          [
              [[-0.32475, -0.97284, -1.63046], [-0.18171, -1.50007, -0.13503]],
              [[-1.09701, 0.32698, -0.40455], [-0.25645, -0.56521, 0.10542]],
              [[-1.08918, -1.18236, -0.7654], [1.54873, -0.53881, -2.90491]],
              [[-1.04385, -2.1746, -2.33245], [-1.64457, -2.22416, -0.43216]],
          ],
          [
              [[-1.56082, -1.79817, -0.61439], [-1.87254, -0.61972, -0.92121]],
              [[0.23756, -1.6664, -2.29889], [-0.88502, -1.52813, -1.28399]],
              [[-1.27026, -1.32213, -2.08694], [-1.61948, 0.01073, -0.08978]],
              [[-1.5593, -2.07375, -1.66693], [-1.18459, -0.39259, -2.48249]],
          ],
          [
              [[0.59724, -0.05495, -0.32654], [-1.69823, -1.13508, -0.08357]],
              [[-1.41267, -1.35452, -1.28225], [-0.148, -0.68255, -1.25917]],
              [[-0.93422, -1.27414, 0.54085], [-0.14944, -1.51341, -2.14796]],
              [[-1.43682, -1.90068, -1.37152], [-1.03635, -0.42591, -0.43638]],
          ],
      ],
      [
          [
              [[-0.32475, -0.97284, -1.63046], [-0.18171, -1.50007, -0.13503]],
              [[-1.097, 0.32698, -0.40454], [-0.25645, -0.56521, 0.10542]],
              [[-1.08918, -1.18236, -0.7654], [1.54873, -0.53881, -2.9049]],
              [[-1.04385, -2.1746, -2.33244], [-1.64456, -2.22416, -0.43216]],
          ],
          [
              [[-1.56133, -1.79817, -0.6144], [-1.87254, -0.61972, -0.92121]],
              [[0.23756, -1.6664, -2.29889], [-0.88502, -1.52813, -1.28399]],
              [[-1.27026, -1.32214, -2.08694], [-1.61948, 0.01073, -0.08979]],
              [[-1.5588, -2.07374, -1.66693], [-1.18459, -0.39259, -2.48248]],
          ],
          [
              [[0.59724, -0.05496, -0.32654], [-1.69823, -1.13508, -0.08357]],
              [[-1.41267, -1.35452, -1.28225], [-0.14854, -0.68255, -1.25917]],
              [[-0.93422, -1.27414, 0.54085], [-0.1489, -1.51341, -2.14795]],
              [[-1.43682, -1.90068, -1.37152], [-1.03635, -0.42591, -0.43638]],
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


def _add_expected_quantized_weights(cur_key, cur_samples):
  target_weight = [
      -2.85875,
      -0.26966,
      -0.4157,
      -1.65228,
      -0.90998,
      -0.9159,
      -1.57794,
      -1.75943,
      0.73056,
      -1.44157,
      -1.14173,
      -2.26072,
      -1.74432,
      -0.31976,
      -0.57841,
      -1.62655,
      0.0788,
      -0.54099,
      1.75434,
      -2.07819,
      -2.06707,
      0.36832,
      -1.7024,
      -0.90404,
      0.0036,
      -0.48214,
      -1.56216,
      -1.52713,
      -0.11998,
      -0.90835,
      -0.87592,
      -0.80171,
      -1.27282,
      -2.16859,
      -0.93238,
      0.2701,
      -1.39285,
      -1.83631,
      -1.57724,
      -0.70324,
      0.48421,
      -1.1211,
      -0.40343,
      -2.71717,
      -2.33469,
      -0.1898,
      -2.47444,
      -1.57077,
      -1.12741,
      -1.33565,
      -0.80041,
      -0.5173,
      -2.37988,
      0.12862,
      -0.11754,
      -0.0451,
      -0.27503,
      -1.22885,
      -0.10287,
      -2.79219,
      -0.02054,
      -0.07701,
      -0.53349,
      -1.39526,
      -0.38721,
      -2.00721,
      -2.4433,
      -0.75067,
      0.13569,
      -2.32686,
      -0.08036,
      0.4129,
  ]
  expected_weights = [
      [
          [
              [-127, -16, -26, -127, -68, -51],
              [-70, -108, 45, -111, -85, -127],
              [-77, -20, -36, -125, 6, -30],
              [78, -127, -127, 28, -127, -51],
          ],
          [
              [0, -23, -85, -89, -6, -73],
              [-80, -37, -69, -127, -48, 22],
              [-127, -86, -86, -41, 25, -91],
              [-37, -127, -127, -11, -127, -127],
          ],
          [
              [-59, -127, -127, -28, -127, 6],
              [-6, -4, -44, -67, -5, -127],
              [-1, -7, -85, -76, -21, -91],
              [-127, -71, 22, -127, -4, 19],
          ],
      ],
      [
          [
              [-128, 127, 23, -128, -15, 71],
              [-57, -83, 127, -101, -48, -128],
              [-66, 120, 8, -125, 127, 127],
              [127, -128, -128, 127, -128, 73],
          ],
          [
              [127, 127, 58, -45, 75, -36],
              [-34, 91, 127, -128, 5, 127],
              [-128, -28, 54, 61, 127, -66],
              [53, -128, -128, 127, -128, -128],
          ],
          [
              [10, -128, -128, 127, -128, 104],
              [117, 127, 15, 27, 125, -128],
              [127, 121, -55, 3, 93, -66],
              [-128, -12, 127, -128, 127, 127],
          ],
      ],
      [
          [
              [[-127, -16, -26], [-127, -68, -51]],
              [[-70, -108, 45], [-111, -85, -127]],
              [[-77, -20, -36], [-125, 6, -30]],
              [[78, -127, -127], [28, -127, -51]],
          ],
          [
              [[0, -23, -85], [-89, -6, -73]],
              [[-80, -37, -69], [-127, -48, 22]],
              [[-127, -86, -86], [-41, 25, -91]],
              [[-37, -127, -127], [-11, -127, -127]],
          ],
          [
              [[-59, -127, -127], [-28, -127, 6]],
              [[-6, -4, -44], [-67, -5, -127]],
              [[-1, -7, -85], [-76, -21, -91]],
              [[-127, -71, 22], [-127, -4, 19]],
          ],
      ],
      [
          [
              [[-128, 127, 23], [-128, -15, 71]],
              [[-57, -83, 127], [-101, -48, -128]],
              [[-66, 120, 8], [-125, 127, 127]],
              [[127, -128, -128], [127, -128, 73]],
          ],
          [
              [[127, 127, 58], [-45, 75, -36]],
              [[-34, 91, 127], [-128, 5, 127]],
              [[-128, -28, 54], [61, 127, -66]],
              [[53, -128, -128], [127, -128, -128]],
          ],
          [
              [[10, -128, -128], [127, -128, 104]],
              [[117, 127, 15], [27, 125, -128]],
              [[127, 121, -55], [3, 93, -66]],
              [[-128, -12, 127], [-128, 127, 127]],
          ],
      ],
      [
          [
              [-127, -16, -26, -127, -68, -51],
              [-70, -108, 45, -111, -85, -127],
              [-77, -20, -36, -125, 6, -30],
              [78, -127, -127, 28, -127, -51],
          ],
          [
              [0, -23, -85, -89, -6, -73],
              [-80, -37, -69, -127, -48, 22],
              [-127, -86, -86, -41, 25, -91],
              [-37, -127, -127, -11, -127, -127],
          ],
          [
              [-59, -127, -127, -28, -127, 6],
              [-6, -4, -44, -67, -5, -127],
              [-1, -7, -85, -76, -21, -91],
              [-127, -71, 22, -127, -4, 19],
          ],
      ],
      [
          [
              [-128, 127, 23, -128, -15, 71],
              [-57, -83, 127, -101, -48, -128],
              [-66, 120, 8, -125, 127, 127],
              [127, -128, -128, 127, -128, 73],
          ],
          [
              [127, 127, 58, -45, 75, -36],
              [-34, 91, 127, -128, 5, 127],
              [-128, -28, 54, 61, 127, -66],
              [53, -128, -128, 127, -128, -128],
          ],
          [
              [10, -128, -128, 127, -128, 104],
              [117, 127, 15, 27, 125, -128],
              [127, 121, -55, 3, 93, -66],
              [-128, -12, 127, -128, 127, 127],
          ],
      ],
      [
          [
              [[-127, -16, -26], [-127, -68, -51]],
              [[-70, -108, 45], [-111, -85, -127]],
              [[-77, -20, -36], [-125, 6, -30]],
              [[78, -127, -127], [28, -127, -51]],
          ],
          [
              [[0, -23, -85], [-89, -6, -73]],
              [[-80, -37, -69], [-127, -48, 22]],
              [[-127, -86, -86], [-41, 25, -91]],
              [[-37, -127, -127], [-11, -127, -127]],
          ],
          [
              [[-59, -127, -127], [-28, -127, 6]],
              [[-6, -4, -44], [-67, -5, -127]],
              [[-1, -7, -85], [-76, -21, -91]],
              [[-127, -71, 22], [-127, -4, 19]],
          ],
      ],
      [
          [
              [[-128, 127, 23], [-128, -15, 71]],
              [[-57, -83, 127], [-101, -48, -128]],
              [[-66, 120, 8], [-125, 127, 127]],
              [[127, -128, -128], [127, -128, 73]],
          ],
          [
              [[127, 127, 58], [-45, 75, -36]],
              [[-34, 91, 127], [-128, 5, 127]],
              [[-128, -28, 54], [61, 127, -66]],
              [[53, -128, -128], [127, -128, -128]],
          ],
          [
              [[10, -128, -128], [127, -128, 104]],
              [[117, 127, 15], [27, 125, -128]],
              [[127, 121, -55], [3, 93, -66]],
              [[-128, -12, 127], [-128, 127, 127]],
          ],
      ],
  ]
  expected_scales = [
      [
          [0.02251, 0.01636, 0.01628, 0.01301, 0.0134, 0.0178],
          [0.01097, 0.0214, 0.01838, 0.01708, 0.01948, 0.01237],
          [0.01924, 0.01052, 0.0063, 0.01832, 0.01874, 0.02199],
      ],
      [
          [0.01809, 0.00709, 0.01097, 0.00792, 0.00699, 0.00674],
          [0.00548, 0.00876, 0.00416, 0.00776, 0.0116, 0.00722],
          [0.0095, 0.00506, 0.00367, 0.0071, 0.00902, 0.01257],
      ],
      [
          [[0.02251, 0.01636, 0.01628], [0.01301, 0.0134, 0.0178]],
          [[0.01097, 0.0214, 0.01838], [0.01708, 0.01948, 0.01237]],
          [[0.01924, 0.01052, 0.0063], [0.01832, 0.01874, 0.02199]],
      ],
      [
          [[0.01809, 0.00709, 0.01097], [0.00792, 0.00699, 0.00674]],
          [[0.00548, 0.00876, 0.00416], [0.00776, 0.0116, 0.00722]],
          [[0.0095, 0.00506, 0.00367], [0.0071, 0.00902, 0.01257]],
      ],
      [
          [0.02251, 0.01636, 0.01628, 0.01301, 0.0134, 0.0178],
          [0.01097, 0.0214, 0.01838, 0.01708, 0.01948, 0.01237],
          [0.01924, 0.01052, 0.0063, 0.01832, 0.01874, 0.02199],
      ],
      [
          [0.01809, 0.00709, 0.01097, 0.00792, 0.00699, 0.00674],
          [0.00548, 0.00876, 0.00416, 0.00776, 0.0116, 0.00722],
          [0.0095, 0.00506, 0.00367, 0.0071, 0.00902, 0.01257],
      ],
      [
          [[0.02251, 0.01636, 0.01628], [0.01301, 0.0134, 0.0178]],
          [[0.01097, 0.0214, 0.01838], [0.01708, 0.01948, 0.01237]],
          [[0.01924, 0.01052, 0.0063], [0.01832, 0.01874, 0.02199]],
      ],
      [
          [[0.01809, 0.00709, 0.01097], [0.00792, 0.00699, 0.00674]],
          [[0.00548, 0.00876, 0.00416], [0.00776, 0.0116, 0.00722]],
          [[0.0095, 0.00506, 0.00367], [0.0071, 0.00902, 0.01257]],
      ],
  ]
  expected_zps = [
      None,
      [
          [0.54314, 1.17036, 0.66275, 0.638, 0.80829, 1.39747],
          [0.69187, 1.59526, 1.80166, 1.1753, 0.9893, 0.64671],
          [1.22715, 0.68783, 0.33051, 1.41852, 1.2256, 1.18335],
      ],
      None,
      [
          [[0.54314, 1.17036, 0.66275], [0.638, 0.80829, 1.39747]],
          [[0.69187, 1.59526, 1.80166], [1.1753, 0.9893, 0.64671]],
          [[1.22715, 0.68783, 0.33051], [1.41852, 1.2256, 1.18335]],
      ],
      None,
      [
          [0.54314, 1.17036, 0.66275, 0.638, 0.80829, 1.39747],
          [0.69187, 1.59526, 1.80166, 1.1753, 0.9893, 0.64671],
          [1.22715, 0.68783, 0.33051, 1.41852, 1.2256, 1.18335],
      ],
      None,
      [
          [[0.54314, 1.17036, 0.66275], [0.638, 0.80829, 1.39747]],
          [[0.69187, 1.59526, 1.80166], [1.1753, 0.9893, 0.64671]],
          [[1.22715, 0.68783, 0.33051], [1.41852, 1.2256, 1.18335]],
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
      -0.612,
      -0.85997,
      -1.94558,
      -1.70384,
      -1.59824,
      -0.27225,
      -0.47487,
      0.13027,
      -2.59563,
      -3.81417,
      -2.76993,
      -2.67496,
      -1.77842,
      -0.09677,
      0.26129,
      0.64258,
      -0.65508,
      -0.69901,
      -2.51849,
      -0.03826,
      -0.63144,
      -2.54013,
      -0.24506,
      -1.26556,
      -2.46572,
      -1.47424,
      -0.83684,
      -0.99288,
      -0.67241,
      -0.97733,
      -1.53587,
      0.35062,
      -2.65814,
      -1.09059,
      -1.45458,
      -0.44545,
      -1.06511,
      0.02425,
      0.02991,
      0.25805,
      -1.208,
      3.06458,
      -1.63139,
      -1.29872,
      -2.13012,
      -1.36993,
      -3.59238,
      -3.84707,
      0.45985,
      -0.36217,
      1.06711,
      -2.47725,
      -3.12246,
      -0.73996,
      -0.8313,
      -2.23025,
      -2.30276,
      -2.09248,
      -0.72215,
      -0.22492,
      0.10892,
      -0.13143,
      -1.14005,
      -2.37122,
      -0.8541,
      -1.43616,
      1.09905,
      -0.89697,
      0.60749,
      1.00183,
      0.08349,
      -1.098,
  ]
  sample_input = [
      -2.47713,
      -0.13888,
      0.49473,
      -1.02615,
      -1.72124,
      0.16843,
      -1.50292,
      -1.58378,
      0.34937,
      0.5749,
      -1.58083,
      -2.40122,
  ]

  expected_proj_qs = [
      None,
      None,
      [
          [[3.29031, 2.10722, 5.95174], [7.70184, 4.25119, 2.00272]],
          [[7.6491, 1.70874, 3.50895], [5.39933, 3.63733, 3.06693]],
          [[8.38051, 0.02094, -1.06923], [2.34197, -0.53824, 2.49987]],
      ],
      [
          [[3.28283, 2.10284, 5.96138], [7.66191, 4.28129, 1.99595]],
          [[7.63754, 1.70295, 3.52495], [5.33695, 3.65657, 3.06246]],
          [[8.3767, 0.01263, -1.06042], [2.2927, -0.53448, 2.49702]],
      ],
      None,
      None,
      [
          [[3.29031, 2.10722, 5.95174], [7.70184, 4.25119, 2.00272]],
          [[7.6491, 1.70874, 3.50895], [5.39933, 3.63733, 3.06693]],
          [[8.38051, 0.02094, -1.06923], [2.34197, -0.53824, 2.49987]],
      ],
      [
          [[3.28283, 2.10284, 5.96138], [7.66191, 4.28129, 1.99595]],
          [[7.63754, 1.70295, 3.52495], [5.33695, 3.65657, 3.06246]],
          [[8.3767, 0.01263, -1.06042], [2.2927, -0.53448, 2.49702]],
      ],
  ]
  expected_proj_ks = [
      None,
      None,
      [
          [[7.4662, 4.94917, 4.64413], [4.14346, 4.9666, 7.92561]],
          [[8.17361, 4.62043, 4.34308], [3.30526, 8.44313, 3.08671]],
          [[3.86093, 2.77035, 3.2726], [1.90722, 9.48239, 3.80128]],
      ],
      [
          [[7.46798, 4.943, 4.63784], [4.14509, 4.95451, 7.94024]],
          [[8.16875, 4.61585, 4.31874], [3.308, 8.42187, 3.08619]],
          [[3.85436, 2.76299, 3.25013], [1.90843, 9.47069, 3.78888]],
      ],
      None,
      None,
      [
          [[7.4662, 4.94917, 4.64413], [4.14346, 4.9666, 7.92561]],
          [[8.17361, 4.62043, 4.34308], [3.30526, 8.44313, 3.08671]],
          [[3.86093, 2.77035, 3.2726], [1.90722, 9.48239, 3.80128]],
      ],
      [
          [[7.46798, 4.943, 4.63784], [4.14509, 4.95451, 7.94024]],
          [[8.16875, 4.61585, 4.31874], [3.308, 8.42187, 3.08619]],
          [[3.85436, 2.76299, 3.25013], [1.90843, 9.47069, 3.78888]],
      ],
  ]
  expected_proj_vs = [
      None,
      None,
      [
          [[-2.09294, 2.08149, -3.52796], [4.22822, 7.33238, 2.26731]],
          [[-2.83916, 1.86234, -1.4888], [5.91343, 6.43092, 5.12284]],
          [[-3.13432, 0.93388, -0.62463], [-0.69218, -0.31758, 4.51747]],
      ],
      [
          [[-2.10662, 2.06142, -3.50795], [4.22783, 7.32401, 2.27623]],
          [[-2.83939, 1.86513, -1.46917], [5.88338, 6.4124, 5.13108]],
          [[-3.125, 0.95269, -0.60097], [-0.73215, -0.34743, 4.52195]],
      ],
      None,
      None,
      [
          [[-2.09294, 2.08149, -3.52796], [4.22822, 7.33238, 2.26731]],
          [[-2.83916, 1.86234, -1.4888], [5.91343, 6.43092, 5.12284]],
          [[-3.13432, 0.93388, -0.62463], [-0.69218, -0.31758, 4.51747]],
      ],
      [
          [[-2.10662, 2.06142, -3.50795], [4.22783, 7.32401, 2.27623]],
          [[-2.83939, 1.86513, -1.46917], [5.88338, 6.4124, 5.13108]],
          [[-3.125, 0.95269, -0.60097], [-0.73215, -0.34743, 4.52195]],
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


class CombinedQKVProjectionAQTTest(quantization_test_util.QuantizationTestCase):
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

  # Test the training with AQT quantization.
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
        quantization_type=QuantizationType.AQT,
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

  # Test AQT weight quantization.
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
        quantization_type=QuantizationType.AQT,
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
        quantization_type=QuantizationType.AQT,
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
        quantization_type=QuantizationType.AQT,
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

