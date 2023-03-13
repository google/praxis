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

"""AQT Tests for quantized DotProductAttention layer.

Since the DotProductAttention layer does not have any weight by itself,
current weight-only PTQ and FQ cannot be applied to DotProductAttention.
"""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import pax_fiddle
from praxis import py_utils
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
    quantization_test_util.generate_dotproduct_attention_test_config
)


def _add_expected_inference_results_after_training(cur_key, cur_samples):
  expected_inference_results = [
      [[
          [0, 0, 0, 0.00125],
          [0, 0, 0, 0.00125],
          [0, 0, 0, 0.00125],
          [-0.63688, -13.98827, -12.41746, -10.72078],
      ]],
      [[
          [-7.15844, -11.41326, -6.8178, -9.48724],
          [-18.83258, -25.33003, -30.2528, -22.75027],
          [-18.34432, -24.99047, -28.62995, -22.48166],
          [-8.24178, -26.48544, -28.27686, -21.40897],
      ]],
      [[
          [-6.48754, -14.52376, -15.52597, -12.05844],
          [-6.48754, -14.52376, -15.52597, -12.05844],
          [-6.48754, -14.52376, -15.52597, -12.05844],
          [-1.45293, -14.92399, -13.10168, -11.52248],
      ]],
      [[
          [-7.15844, -11.41326, -6.8178, -9.48724],
          [-18.83258, -25.33003, -30.2528, -22.75027],
          [-18.34432, -24.99047, -28.62995, -22.48166],
          [-8.24178, -26.48544, -28.27686, -21.40897],
      ]],
      [[
          [0, 0, 0, 0.00125],
          [0, 0, 0, 0.00125],
          [0, 0, 0, 0.00125],
          [-0.63688, -13.98827, -12.41746, -10.72078],
      ]],
      [[
          [-7.15844, -11.41326, -6.8178, -9.48724],
          [-18.83258, -25.33003, -30.2528, -22.75027],
          [-18.34432, -24.99047, -28.62995, -22.48166],
          [-8.24178, -26.48544, -28.27686, -21.40897],
      ]],
      [[
          [-6.48754, -14.52376, -15.52597, -12.05844],
          [-6.48754, -14.52376, -15.52597, -12.05844],
          [-6.48754, -14.52376, -15.52597, -12.05844],
          [-1.45293, -14.92399, -13.10168, -11.52248],
      ]],
      [[
          [-7.15844, -11.41326, -6.8178, -9.48724],
          [-18.83258, -25.33003, -30.2528, -22.75027],
          [-18.34432, -24.99047, -28.62995, -22.48166],
          [-8.24178, -26.48544, -28.27686, -21.40897],
      ]],
      None,
      None,
      None,
      None,
      [[
          [0, 0, 0, 0.00125],
          [0, 0, 0, 0.00125],
          [0, 0, 0, 0.00125],
          [-0.63688, -13.98827, -12.41746, -10.72078],
      ]],
      [[
          [-7.40531, -11.63147, -6.979, -9.65194],
          [-19.45521, -25.81805, -30.72656, -23.13909],
          [-19.42682, -25.79952, -30.66985, -23.12003],
          [-16.88581, -27.93965, -26.70293, -25.48205],
      ]],
      [[
          [-6.48754, -14.52376, -15.52597, -12.05844],
          [-6.48754, -14.52376, -15.52597, -12.05844],
          [-6.48754, -14.52376, -15.52597, -12.05844],
          [-1.45293, -14.92399, -13.10168, -11.52248],
      ]],
      [[
          [-7.40531, -11.63147, -6.979, -9.65194],
          [-19.45521, -25.81805, -30.72656, -23.13909],
          [-19.42682, -25.79952, -30.66985, -23.12003],
          [-16.88581, -27.93965, -26.70293, -25.48205],
      ]],
      [[
          [0, 0.00125, 0, 0.00125],
          [0, 0.00125, 0, 0.00125],
          [0, 0.00125, 0, 0.00125],
          [-35.10808, 1.70227, -3.29638, -12.56385],
      ]],
      [[
          [-13.16125, -1.48178, -1.62388, -7.34667],
          [-40.18428, -12.13143, -6.74566, -19.35975],
          [-45.1223, -10.24395, -6.54652, -21.87124],
          [-58.01292, -4.32096, -7.17834, -24.08418],
      ]],
      [[
          [-28.89788, -3.08438, -3.7559, -12.03867],
          [-28.89788, -3.08438, -3.7559, -12.03867],
          [-28.89788, -3.08438, -3.7559, -12.03867],
          [-36.55595, 1.22932, -3.63835, -13.80034],
      ]],
      [[
          [-13.16125, -1.48178, -1.62388, -7.34667],
          [-40.18428, -12.13143, -6.74566, -19.35975],
          [-45.1223, -10.24395, -6.54652, -21.87124],
          [-58.01292, -4.32096, -7.17834, -24.08418],
      ]],
      [[
          [0, 0.00125, 0, 0.00125],
          [0, 0.00125, 0, 0.00125],
          [0, 0.00125, 0, 0.00125],
          [-35.10808, 1.70227, -3.29638, -12.56385],
      ]],
      [[
          [-13.16125, -1.48178, -1.62388, -7.34667],
          [-40.18428, -12.13143, -6.74566, -19.35975],
          [-45.1223, -10.24395, -6.54652, -21.87124],
          [-58.01292, -4.32096, -7.17834, -24.08418],
      ]],
      [[
          [-28.89788, -3.08438, -3.7559, -12.03867],
          [-28.89788, -3.08438, -3.7559, -12.03867],
          [-28.89788, -3.08438, -3.7559, -12.03867],
          [-36.55595, 1.22932, -3.63835, -13.80034],
      ]],
      [[
          [-13.16125, -1.48178, -1.62388, -7.34667],
          [-40.18428, -12.13143, -6.74566, -19.35975],
          [-45.1223, -10.24395, -6.54652, -21.87124],
          [-58.01292, -4.32096, -7.17834, -24.08418],
      ]],
      None,
      None,
      None,
      None,
      [[
          [0, 0.00125, 0, 0.00125],
          [0, 0.00125, 0, 0.00125],
          [0, 0.00125, 0, 0.00125],
          [-35.10808, 1.70227, -3.29638, -12.56385],
      ]],
      [[
          [-13.19857, -1.58171, -1.6515, -7.46626],
          [-40.27155, -12.53879, -6.84212, -19.76836],
          [-40.41632, -12.47416, -6.83475, -19.83953],
          [-55.96181, -6.74058, -6.58032, -23.88404],
      ]],
      [[
          [-28.89788, -3.08438, -3.7559, -12.03867],
          [-28.89788, -3.08438, -3.7559, -12.03867],
          [-28.89788, -3.08438, -3.7559, -12.03867],
          [-36.55595, 1.22932, -3.63835, -13.80034],
      ]],
      [[
          [-13.19857, -1.58171, -1.6515, -7.46626],
          [-40.27155, -12.53879, -6.84212, -19.76836],
          [-40.41632, -12.47416, -6.83475, -19.83953],
          [-55.96181, -6.74058, -6.58032, -23.88404],
      ]],
      [[
          [0.00125, 0, 0, 0.00125],
          [0.00125, 0, 0, 0.00125],
          [0.00125, 0, 0, 0.00125],
          [3.73169, -17.92372, -29.78185, -11.05944],
      ]],
      [[
          [-8.37813, -12.72894, -14.68908, -11.05933],
          [-16.98026, -37.42053, -26.69418, -29.48494],
          [-6.20387, -32.43212, -33.00116, -23.20395],
          [-0.75136, -36.84155, -46.49342, -24.95195],
      ]],
      [[
          [-4.13064, -20.04366, -22.18038, -14.56186],
          [-4.13064, -20.04366, -22.18038, -14.56186],
          [-4.13064, -20.04366, -22.18038, -14.56186],
          [2.93601, -19.5608, -32.36147, -12.42099],
      ]],
      [[
          [-8.37813, -12.72894, -14.68908, -11.05933],
          [-16.98026, -37.42053, -26.69418, -29.48494],
          [-6.20387, -32.43212, -33.00116, -23.20395],
          [-0.75136, -36.84155, -46.49342, -24.95195],
      ]],
      [[
          [0.00125, 0, 0, 0.00125],
          [0.00125, 0, 0, 0.00125],
          [0.00125, 0, 0, 0.00125],
          [3.73169, -17.92372, -29.78185, -11.05944],
      ]],
      [[
          [-8.37813, -12.72894, -14.68908, -11.05933],
          [-16.98026, -37.42053, -26.69418, -29.48494],
          [-6.20387, -32.43212, -33.00116, -23.20395],
          [-0.75136, -36.84155, -46.49342, -24.95195],
      ]],
      [[
          [-4.13064, -20.04366, -22.18038, -14.56186],
          [-4.13064, -20.04366, -22.18038, -14.56186],
          [-4.13064, -20.04366, -22.18038, -14.56186],
          [2.93601, -19.5608, -32.36147, -12.42099],
      ]],
      [[
          [-8.37813, -12.72894, -14.68908, -11.05933],
          [-16.98026, -37.42053, -26.69418, -29.48494],
          [-6.20387, -32.43212, -33.00116, -23.20395],
          [-0.75136, -36.84155, -46.49342, -24.95195],
      ]],
      None,
      None,
      None,
      None,
      [[
          [0.00125, 0, 0, 0.00125],
          [0.00125, 0, 0, 0.00125],
          [0.00125, 0, 0, 0.00125],
          [3.73169, -17.92372, -29.78185, -11.05944],
      ]],
      [[
          [-8.41529, -12.8069, -14.76298, -11.12315],
          [-17.04082, -37.52143, -26.74599, -29.57402],
          [-7.72792, -33.64466, -31.53459, -24.5944],
          [-0.72331, -36.69995, -46.36712, -24.83733],
      ]],
      [[
          [-4.13064, -20.04366, -22.18038, -14.56186],
          [-4.13064, -20.04366, -22.18038, -14.56186],
          [-4.13064, -20.04366, -22.18038, -14.56186],
          [2.93601, -19.5608, -32.36147, -12.42099],
      ]],
      [[
          [-8.41529, -12.8069, -14.76298, -11.12315],
          [-17.04082, -37.52143, -26.74599, -29.57402],
          [-7.72792, -33.64466, -31.53459, -24.5944],
          [-0.72331, -36.69995, -46.36712, -24.83733],
      ]],
      [[
          [0, 0, 0, 0.00125],
          [0, 0, 0, 0.00125],
          [0, 0, 0, 0.00125],
          [-24.90825, -16.29055, -8.73898, -23.97112],
      ]],
      [[
          [-11.71323, -8.17542, -3.4262, -9.16296],
          [-45.08155, -10.65862, -8.81668, -30.63462],
          [-43.53231, -16.31478, -10.72185, -36.92304],
          [-49.92963, -24.48606, -14.61028, -44.84933],
      ]],
      [[
          [-26.07055, -11.2625, -6.84954, -21.79811],
          [-26.07055, -11.2625, -6.84954, -21.79811],
          [-26.07055, -11.2625, -6.84954, -21.79811],
          [-26.99276, -18.18575, -9.60366, -26.97277],
      ]],
      [[
          [-11.71323, -8.17542, -3.4262, -9.16296],
          [-45.08155, -10.65862, -8.81668, -30.63462],
          [-43.53231, -16.31478, -10.72185, -36.92304],
          [-49.92963, -24.48606, -14.61028, -44.84933],
      ]],
      [[
          [0, 0, 0, 0.00125],
          [0, 0, 0, 0.00125],
          [0, 0, 0, 0.00125],
          [-24.90825, -16.29055, -8.73898, -23.97112],
      ]],
      [[
          [-11.71323, -8.17542, -3.4262, -9.16296],
          [-45.08155, -10.65862, -8.81668, -30.63462],
          [-43.53231, -16.31478, -10.72185, -36.92304],
          [-49.92963, -24.48606, -14.61028, -44.84933],
      ]],
      [[
          [-26.07055, -11.2625, -6.84954, -21.79811],
          [-26.07055, -11.2625, -6.84954, -21.79811],
          [-26.07055, -11.2625, -6.84954, -21.79811],
          [-26.99276, -18.18575, -9.60366, -26.97277],
      ]],
      [[
          [-11.71323, -8.17542, -3.4262, -9.16296],
          [-45.08155, -10.65862, -8.81668, -30.63462],
          [-43.53231, -16.31478, -10.72185, -36.92304],
          [-49.92963, -24.48606, -14.61028, -44.84933],
      ]],
      None,
      None,
      None,
      None,
      [[
          [0, 0, 0, 0.00125],
          [0, 0, 0, 0.00125],
          [0, 0, 0, 0.00125],
          [-24.90825, -16.29055, -8.73898, -23.97112],
      ]],
      [[
          [-11.75114, -8.1691, -3.43306, -9.17747],
          [-45.15388, -10.64365, -8.82501, -30.67708],
          [-42.64843, -15.73957, -10.44966, -35.82434],
          [-49.63036, -24.42527, -14.54575, -44.57124],
      ]],
      [[
          [-26.07055, -11.2625, -6.84954, -21.79811],
          [-26.07055, -11.2625, -6.84954, -21.79811],
          [-26.07055, -11.2625, -6.84954, -21.79811],
          [-26.99276, -18.18575, -9.60366, -26.97277],
      ]],
      [[
          [-11.75114, -8.1691, -3.43306, -9.17747],
          [-45.15388, -10.64365, -8.82501, -30.67708],
          [-42.64843, -15.73957, -10.44966, -35.82434],
          [-49.63036, -24.42527, -14.54575, -44.57124],
      ]],
      [[
          [0, 0, 0, 0.00125],
          [0, 0, 0, 0.00125],
          [0, 0, 0, 0.00125],
          [-1.24254, -27.74252, -24.80173, -21.24504],
      ]],
      [[
          [-11.73337, -20.44285, -12.45408, -17.2051],
          [-28.15127, -38.63865, -51.78885, -35.62572],
          [-28.15141, -38.63868, -51.78917, -35.62577],
          [-1.9752, -28.44204, -25.4672, -21.8889],
      ]],
      [[
          [-11.44834, -27.3315, -29.74196, -22.73614],
          [-11.44834, -27.3315, -29.74196, -22.73614],
          [-11.44834, -27.3315, -29.74196, -22.73614],
          [-2.12593, -28.58394, -25.59468, -22.00729],
      ]],
      [[
          [-11.73337, -20.44285, -12.45408, -17.2051],
          [-28.15127, -38.63865, -51.78885, -35.62572],
          [-28.15141, -38.63868, -51.78917, -35.62577],
          [-1.9752, -28.44204, -25.4672, -21.8889],
      ]],
      [[
          [0, 0, 0, 0.00125],
          [0, 0, 0, 0.00125],
          [0, 0, 0, 0.00125],
          [-1.24254, -27.74252, -24.80173, -21.24504],
      ]],
      [[
          [-11.73337, -20.44285, -12.45408, -17.2051],
          [-28.15127, -38.63865, -51.78885, -35.62572],
          [-28.15141, -38.63868, -51.78917, -35.62577],
          [-1.9752, -28.44204, -25.4672, -21.8889],
      ]],
      [[
          [-11.44834, -27.3315, -29.74196, -22.73614],
          [-11.44834, -27.3315, -29.74196, -22.73614],
          [-11.44834, -27.3315, -29.74196, -22.73614],
          [-2.12593, -28.58394, -25.59468, -22.00729],
      ]],
      [[
          [-11.73337, -20.44285, -12.45408, -17.2051],
          [-28.15127, -38.63865, -51.78885, -35.62572],
          [-28.15141, -38.63868, -51.78917, -35.62577],
          [-1.9752, -28.44204, -25.4672, -21.8889],
      ]],
      None,
      None,
      None,
      None,
      [[
          [0, 0, 0, 0.00125],
          [0, 0, 0, 0.00125],
          [0, 0, 0, 0.00125],
          [-1.24254, -27.74252, -24.80173, -21.24504],
      ]],
      [[
          [-11.80017, -20.50523, -12.51208, -17.2597],
          [-28.44493, -38.91343, -52.04216, -35.86406],
          [-28.44493, -38.91343, -52.04216, -35.86406],
          [-28.44493, -38.91343, -52.04216, -35.86406],
      ]],
      [[
          [-11.44834, -27.3315, -29.74196, -22.73614],
          [-11.44834, -27.3315, -29.74196, -22.73614],
          [-11.44834, -27.3315, -29.74196, -22.73614],
          [-2.12593, -28.58394, -25.59468, -22.00729],
      ]],
      [[
          [-11.80017, -20.50523, -12.51208, -17.2597],
          [-28.44493, -38.91343, -52.04216, -35.86406],
          [-28.44493, -38.91343, -52.04216, -35.86406],
          [-28.44493, -38.91343, -52.04216, -35.86406],
      ]],
      [[
          [0, 0.00125, 0, 0.00125],
          [0, 0.00125, 0, 0.00125],
          [0, 0.00125, 0, 0.00125],
          [-68.20838, 2.90668, -6.61268, -25.13028],
      ]],
      [[
          [-25.22124, -2.26573, -3.09566, -13.32517],
          [-68.61777, -19.88044, -11.66021, -29.49759],
          [-68.61803, -19.88071, -11.66029, -29.49792],
          [-68.91206, 2.34384, -7.05911, -25.89133],
      ]],
      [[
          [-54.72446, -5.83041, -7.29869, -22.53382],
          [-54.72446, -5.83041, -7.29869, -22.53382],
          [-54.72446, -5.83041, -7.29869, -22.53382],
          [-69.05771, 2.25457, -7.10509, -26.05802],
      ]],
      [[
          [-25.22124, -2.26573, -3.09566, -13.32517],
          [-68.61777, -19.88044, -11.66021, -29.49759],
          [-68.61803, -19.88071, -11.66029, -29.49792],
          [-68.91206, 2.34384, -7.05911, -25.89133],
      ]],
      [[
          [0, 0.00125, 0, 0.00125],
          [0, 0.00125, 0, 0.00125],
          [0, 0.00125, 0, 0.00125],
          [-68.20838, 2.90668, -6.61268, -25.13028],
      ]],
      [[
          [-25.22124, -2.26573, -3.09566, -13.32517],
          [-68.61777, -19.88044, -11.66021, -29.49759],
          [-68.61803, -19.88071, -11.66029, -29.49792],
          [-68.91206, 2.34384, -7.05911, -25.89133],
      ]],
      [[
          [-54.72446, -5.83041, -7.29869, -22.53382],
          [-54.72446, -5.83041, -7.29869, -22.53382],
          [-54.72446, -5.83041, -7.29869, -22.53382],
          [-69.05771, 2.25457, -7.10509, -26.05802],
      ]],
      [[
          [-25.22124, -2.26573, -3.09566, -13.32517],
          [-68.61777, -19.88044, -11.66021, -29.49759],
          [-68.61803, -19.88071, -11.66029, -29.49792],
          [-68.91206, 2.34384, -7.05911, -25.89133],
      ]],
      None,
      None,
      None,
      None,
      [[
          [0, 0.00125, 0, 0.00125],
          [0, 0.00125, 0, 0.00125],
          [0, 0.00125, 0, 0.00125],
          [-68.20838, 2.90668, -6.61268, -25.13028],
      ]],
      [[
          [-25.27854, -2.46446, -3.13277, -13.42151],
          [-68.87882, -20.46275, -11.8018, -29.8799],
          [-68.87882, -20.46275, -11.8018, -29.8799],
          [-68.87882, -20.46275, -11.8018, -29.8799],
      ]],
      [[
          [-54.72446, -5.83041, -7.29869, -22.53382],
          [-54.72446, -5.83041, -7.29869, -22.53382],
          [-54.72446, -5.83041, -7.29869, -22.53382],
          [-69.05771, 2.25457, -7.10509, -26.05802],
      ]],
      [[
          [-25.27854, -2.46446, -3.13277, -13.42151],
          [-68.87882, -20.46275, -11.8018, -29.8799],
          [-68.87882, -20.46275, -11.8018, -29.8799],
          [-68.87882, -20.46275, -11.8018, -29.8799],
      ]],
      [[
          [0.00125, 0, 0, 0.00125],
          [0.00125, 0, 0, 0.00125],
          [0.00125, 0, 0, 0.00125],
          [7.43458, -34.94469, -57.71767, -21.46405],
      ]],
      [[
          [-14.51658, -22.65409, -26.14117, -19.87947],
          [-23.69559, -59.62869, -38.44277, -46.28018],
          [10.27693, -44.20252, -41.28938, -30.361],
          [7.16902, -35.66841, -58.43946, -22.12231],
      ]],
      [[
          [-6.99133, -36.81042, -40.09488, -26.57466],
          [-6.99133, -36.81042, -40.09488, -26.57466],
          [-6.99133, -36.81042, -40.09488, -26.57466],
          [6.47384, -35.86517, -58.5898, -22.30547],
      ]],
      [[
          [-14.51658, -22.65409, -26.14117, -19.87947],
          [-23.69559, -59.62869, -38.44277, -46.28018],
          [10.27693, -44.20252, -41.28938, -30.361],
          [7.16902, -35.66841, -58.43946, -22.12231],
      ]],
      [[
          [0.00125, 0, 0, 0.00125],
          [0.00125, 0, 0, 0.00125],
          [0.00125, 0, 0, 0.00125],
          [7.43458, -34.94469, -57.71767, -21.46405],
      ]],
      [[
          [-14.51658, -22.65409, -26.14117, -19.87947],
          [-23.69559, -59.62869, -38.44277, -46.28018],
          [10.27693, -44.20252, -41.28938, -30.361],
          [7.16902, -35.66841, -58.43946, -22.12231],
      ]],
      [[
          [-6.99133, -36.81042, -40.09488, -26.57466],
          [-6.99133, -36.81042, -40.09488, -26.57466],
          [-6.99133, -36.81042, -40.09488, -26.57466],
          [6.47384, -35.86517, -58.5898, -22.30547],
      ]],
      [[
          [-14.51658, -22.65409, -26.14117, -19.87947],
          [-23.69559, -59.62869, -38.44277, -46.28018],
          [10.27693, -44.20252, -41.28938, -30.361],
          [7.16902, -35.66841, -58.43946, -22.12231],
      ]],
      None,
      None,
      None,
      None,
      [[
          [0.00125, 0, 0, 0.00125],
          [0.00125, 0, 0, 0.00125],
          [0.00125, 0, 0, 0.00125],
          [7.43458, -34.94469, -57.71767, -21.46405],
      ]],
      [[
          [-14.50935, -22.67908, -26.16159, -19.90079],
          [-23.6452, -59.7604, -38.54902, -46.39432],
          [8.34754, -45.22635, -41.22205, -31.39761],
          [13.75056, -44.4755, -48.27301, -29.01481],
      ]],
      [[
          [-6.99133, -36.81042, -40.09488, -26.57466],
          [-6.99133, -36.81042, -40.09488, -26.57466],
          [-6.99133, -36.81042, -40.09488, -26.57466],
          [6.47384, -35.86517, -58.5898, -22.30547],
      ]],
      [[
          [-14.50935, -22.67908, -26.16159, -19.90079],
          [-23.6452, -59.7604, -38.54902, -46.39432],
          [8.34754, -45.22635, -41.22205, -31.39761],
          [13.75056, -44.4755, -48.27301, -29.01481],
      ]],
      [[
          [0, 0, 0, 0.00125],
          [0, 0, 0, 0.00125],
          [0, 0, 0, 0.00125],
          [-48.40785, -31.00853, -16.93446, -45.64339],
      ]],
      [[
          [-22.06349, -14.34444, -6.23847, -15.75029],
          [-76.74578, -13.75991, -14.06797, -49.03659],
          [-59.71566, -20.10871, -15.78629, -48.31999],
          [-49.20071, -31.67142, -17.46412, -46.51015],
      ]],
      [[
          [-48.40355, -19.65101, -12.54035, -38.29126],
          [-48.40355, -19.65101, -12.54035, -38.29126],
          [-48.40355, -19.65101, -12.54035, -38.29126],
          [-49.33721, -31.74284, -17.50723, -46.65035],
      ]],
      [[
          [-22.06349, -14.34444, -6.23847, -15.75029],
          [-76.74578, -13.75991, -14.06797, -49.03659],
          [-59.71566, -20.10871, -15.78629, -48.31999],
          [-49.20071, -31.67142, -17.46412, -46.51015],
      ]],
      [[
          [0, 0, 0, 0.00125],
          [0, 0, 0, 0.00125],
          [0, 0, 0, 0.00125],
          [-48.40785, -31.00853, -16.93446, -45.64339],
      ]],
      [[
          [-22.06349, -14.34444, -6.23847, -15.75029],
          [-76.74578, -13.75991, -14.06797, -49.03659],
          [-59.71566, -20.10871, -15.78629, -48.31999],
          [-49.20071, -31.67142, -17.46412, -46.51015],
      ]],
      [[
          [-48.40355, -19.65101, -12.54035, -38.29126],
          [-48.40355, -19.65101, -12.54035, -38.29126],
          [-48.40355, -19.65101, -12.54035, -38.29126],
          [-49.33721, -31.74284, -17.50723, -46.65035],
      ]],
      [[
          [-22.06349, -14.34444, -6.23847, -15.75029],
          [-76.74578, -13.75991, -14.06797, -49.03659],
          [-59.71566, -20.10871, -15.78629, -48.31999],
          [-49.20071, -31.67142, -17.46412, -46.51015],
      ]],
      None,
      None,
      None,
      None,
      [[
          [0, 0, 0, 0.00125],
          [0, 0, 0, 0.00125],
          [0, 0, 0, 0.00125],
          [-48.40785, -31.00853, -16.93446, -45.64339],
      ]],
      [[
          [-22.10061, -14.35311, -6.24811, -15.77763],
          [-76.92857, -13.80955, -14.12574, -49.16965],
          [-61.16026, -19.67361, -15.71037, -48.49237],
          [-54.71908, -26.81585, -17.16926, -55.37206],
      ]],
      [[
          [-48.40355, -19.65101, -12.54035, -38.29126],
          [-48.40355, -19.65101, -12.54035, -38.29126],
          [-48.40355, -19.65101, -12.54035, -38.29126],
          [-49.33721, -31.74284, -17.50723, -46.65035],
      ]],
      [[
          [-22.10061, -14.35311, -6.24811, -15.77763],
          [-76.92857, -13.80955, -14.12574, -49.16965],
          [-61.16026, -19.67361, -15.71037, -48.49237],
          [-54.71908, -26.81585, -17.16926, -55.37206],
      ]],
  ]

  updated_key = cur_key + ['expected_inference_result']

  ret = []
  for sample, expected_inference_result in zip(
      cur_samples, expected_inference_results
  ):
    sample.append(expected_inference_result)

    ret.append(sample)

  return updated_key, ret


def _add_sample_inputs(cur_key, cur_samples):
  init_proj_weight = [
      0.49263,
      -0.47125,
      -2.74662,
      -1.50045,
      -1.444,
      -1.54899,
      0.92554,
      -1.49259,
      1.62225,
      -1.03616,
      -2.08021,
      0.06764,
      -0.71702,
      -1.31121,
      -3.65516,
      -0.38274,
      -1.96408,
      -0.36709,
      0.64298,
      -1.98362,
      0.28688,
      -1.37332,
      -1.33783,
      -1.84077,
      -1.13351,
      -2.23542,
      -1.41229,
      0.74241,
      -1.38769,
      -0.40825,
      -0.06061,
      -1.74407,
      -0.9262,
      -0.45272,
      -1.18325,
      -1.05958,
      -1.17578,
      -0.9783,
      0.13367,
      -0.653,
      -1.82506,
      -1.83417,
      -0.69623,
      0.19598,
      0.26787,
      -0.93888,
      -1.38779,
      -1.54724,
      1.083,
      -0.14147,
      -2.10915,
      -0.99369,
      -1.7054,
      -2.14023,
      -1.38681,
      -1.50564,
      1.15452,
      -3.58734,
      -1.81326,
      -2.23135,
      0.43774,
      -0.78058,
      -0.12287,
      -2.02514,
      -0.3775,
      -0.73769,
      -0.26285,
      -3.00276,
      -1.70136,
      -2.42964,
      -1.08255,
      -0.0313,
      0.32341,
      -1.11681,
      -0.46007,
      0.04985,
      -2.37018,
      -0.14432,
      0.45345,
      -2.37759,
      -1.27379,
      -0.15513,
      -1.6202,
      -0.01434,
      -1.65144,
      0.27582,
      -0.95931,
      -0.78677,
      -0.23684,
      -2.10283,
      -0.96119,
      -0.19732,
      -0.79705,
      -1.36458,
      -1.82397,
      -0.23553,
  ]
  init_post_weight = [
      -2.11696,
      -0.39401,
      -0.20965,
      -0.8457,
      -3.2395,
      -2.37503,
      -0.62421,
      -1.69202,
      -0.08903,
      0.06258,
      -2.64418,
      0.03128,
      1.10451,
      -0.49107,
      -1.28301,
      0.5186,
      -0.06033,
      0.0944,
      -0.58937,
      -0.29472,
      0.09253,
      -0.46955,
      -0.62756,
      -0.18464,
      -0.27736,
      -1.22081,
      -2.74254,
      -1.11865,
      1.18502,
      -2.08555,
      1.05182,
      -1.65488,
  ]
  query = [
      0.24065,
      -1.3093,
      0.02357,
      -1.48774,
      -0.19496,
      -0.80272,
      -3.52127,
      -2.24813,
      -1.8162,
      -1.06886,
      -1.74048,
      0.26431,
      -1.87161,
      -2.40834,
      -1.62859,
      0.5879,
  ]
  pseudo_ans = [
      0.67653,
      0.09715,
      -1.58487,
      -1.70358,
      -1.45268,
      -1.14612,
      -1.0698,
      -0.74192,
      -0.23311,
      -0.3604,
      0.18244,
      -0.61323,
      -0.35955,
      -0.84001,
      -0.39822,
      -0.88943,
  ]

  # Current dconv init algorithm. Written here in case of their initialization
  # algorithm update.
  init_dconv_weight = [0.5] * 24 + [0.5 / 3] * 48

  updated_key = cur_key + [
      'init_proj_weight',
      'init_post_weight',
      'init_dconv_weight',
      'query',
      'pseudo_ans',
  ]

  ret = []
  for sample in cur_samples:
    sample.append(init_proj_weight)
    sample.append(init_post_weight)
    sample.append(init_dconv_weight)
    sample.append(query)
    sample.append(pseudo_ans)

    ret.append(sample)

  return updated_key, ret


class DotProductAttentionAQTTest(quantization_test_util.QuantizationTestCase):
  """Test cases for QuantizationType.AQT.

  Since there are no quantized weights to update during the training, we do not
  check the trained weights value. Instead, we do the following tests:
  1. Check the result of __call__ after training.
  2. Compare the results of extend_step and __call__ after training.
  """

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  def _update(self, atten_layer, params, q, k, v, mask, segment_pos, ans):
    step_size = 0.01

    def loss(params, q, k, v, mask, segment_pos, ans):
      attended, _ = atten_layer.apply(
          params,
          q,
          k,
          v,
          mask,
          query_segment_pos=segment_pos,
          key_segment_pos=segment_pos,
      )
      return -jnp.mean(jnp.abs(attended - ans))

    grads = jax.grad(loss)(params, q, k, v, mask, segment_pos, ans)

    def _update_params(step_size, params, grads):
      out_params = dict()
      for k, v in params.items():
        if isinstance(v, dict):
          out_params[k] = _update_params(step_size, v, grads[k])
        else:
          out_params[k] = v - grads[k] * step_size
      return out_params

    out_params = dict()
    out_params['params'] = _update_params(
        step_size, params['params'], grads['params']
    )

    return out_params

  def _initialize_weights(
      self,
      initial_vars,
      init_proj_weight,
      init_post_weight,
      init_dconv,
      dconv_kernel_size,
      combine_qkv,
      dconv_qkv,
  ):
    if combine_qkv:
      proj_shape = initial_vars['params']['combined_qkv']['w'].shape
      initial_vars['params']['combined_qkv']['w'] = np.array(
          init_proj_weight
      ).reshape(proj_shape)
    else:
      proj_weight_len = len(init_proj_weight) // 3
      proj_shape = initial_vars['params']['query']['w'].shape
      initial_vars['params']['query']['w'] = np.array(
          init_proj_weight[:proj_weight_len]
      ).reshape(proj_shape)
      initial_vars['params']['key']['w'] = np.array(
          init_proj_weight[:proj_weight_len]
      ).reshape(proj_shape)
      initial_vars['params']['value']['w'] = np.array(
          init_proj_weight[:proj_weight_len]
      ).reshape(proj_shape)

    post_shape = initial_vars['params']['post']['w'].shape
    initial_vars['params']['post']['w'] = np.array(init_post_weight).reshape(
        post_shape
    )

    if dconv_qkv:
      dconv_kernel_weight_len = len(init_dconv) // dconv_kernel_size // 3
      weight_start = 0
      for i in range(dconv_kernel_size):
        for p in ['q', 'k', 'v']:
          shape = initial_vars['params'][f'dconv_{p}'][f'dconv_{i}'].shape
          initial_vars['params'][f'dconv_{p}'][f'dconv_{i}'] = np.array(
              init_dconv[weight_start : weight_start + dconv_kernel_weight_len]
          ).reshape(shape)
          weight_start += dconv_kernel_weight_len

    return initial_vars

  # Check the result of __call__ after training with AQT quantization.
  @parameterized.parameters(
      generate_quantization_test_config(
          [_add_sample_inputs, _add_expected_inference_results_after_training]
      )
  )
  def test_inference_after_training(
      self,
      dconv_qkv,
      combine_qkv,
      output_proj_use_nhd_shape,
      use_rotary_position_emb,
      cast_rotary_position_emb,
      zero_fully_masked,
      simulate_packed,
      init_proj_weight,
      init_post_weight,
      init_dconv_weight,
      query,
      pseudo_ans,
      expected_inference_result
  ):
    if not use_rotary_position_emb and cast_rotary_position_emb:
      return

    batch_size = 1
    num_heads = 2
    max_seq_len = 4
    input_dim = 4
    dconv_kernel_size = 3
    hidden_dim = 8

    query = (
        np.array(query)
        .reshape([batch_size, max_seq_len, input_dim])
        .astype('f4')
    )
    pseudo_ans = (
        np.array(pseudo_ans)
        .reshape([batch_size, max_seq_len, input_dim])
        .astype('f4')
    )

    quantization_option = QuantizationHParams(
        quantization_type=QuantizationType.AQT,
        mode=QuantizationMode.TRAINING,
        act_params=quantization_hparams.ActQuantizationParams(precision=16),
        weight_params=None,
    )
    atten_f = pax_fiddle.Config(
        attentions.DotProductAttention,
        name='_dotprod_f'
    )
    atten_q = pax_fiddle.Config(
        qattentions.DotProductAttention,
        name='_dotprod_q',
        quantization=quantization_option,
    )

    for atten in [atten_f, atten_q]:
      atten.input_dim = input_dim
      atten.num_heads = num_heads
      atten.hidden_dim = hidden_dim
      atten.dconv_qkv = dconv_qkv
      atten.dconv_kernel_size = dconv_kernel_size
      atten.combine_qkv = combine_qkv
      atten.output_proj_use_nhd_shape = output_proj_use_nhd_shape
      atten.use_rotary_position_emb = use_rotary_position_emb
      atten.cast_rotary_position_emb = cast_rotary_position_emb
      atten.zero_fully_masked = zero_fully_masked

    atten_f = instantiate(atten_f)
    atten_q = instantiate(atten_q)

    fake_query = jnp.zeros_like(query)

    atten_mask = attentions.causal_mask(query)
    segment_pos = np.tile(np.arange(max_seq_len), (batch_size, 1))

    starting_index = 0
    if simulate_packed:
      starting_index = dconv_kernel_size
      atten_mask = atten_mask.at[:, :, :, :starting_index].set(-2.3819763e38)
      segment_pos = jnp.maximum(segment_pos - starting_index, 0)

    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)
    initial_vars = atten_f.init(
        init_key,
        fake_query,
        fake_query,
        fake_query,
        atten_mask,
        query_segment_pos=segment_pos,
        key_segment_pos=segment_pos,
    )

    initial_vars = self._initialize_weights(
        initial_vars,
        init_proj_weight,
        init_post_weight,
        init_dconv_weight,
        dconv_kernel_size,
        combine_qkv,
        dconv_qkv,
    )

    fprop_out_f_init, _ = atten_f.apply(
        initial_vars,
        query,
        query,
        query,
        atten_mask,
        query_segment_pos=segment_pos,
        key_segment_pos=segment_pos,
    )
    fprop_out_q_init, _ = atten_q.apply(
        initial_vars,
        query,
        query,
        query,
        atten_mask,
        query_segment_pos=segment_pos,
        key_segment_pos=segment_pos,
    )

    # Without the training, the two inference results needs to be almost
    # similar.
    self.assertAllClose(fprop_out_f_init, fprop_out_q_init, atol=1e-1)

    updated_vars_q = self._update(
        atten_q,
        initial_vars,
        query,
        query,
        query,
        atten_mask,
        segment_pos,
        pseudo_ans,
    )

    fprop_out_q, _ = atten_q.apply(
        updated_vars_q,
        query,
        query,
        query,
        atten_mask,
        query_segment_pos=segment_pos,
        key_segment_pos=segment_pos,
    )

    # Check inference result after training.
    self.assertNestedListClose(
        to_list(fprop_out_q), expected_inference_result, places=3
    )

  # Check the result of extend_step after training with AQT quantization.
  @parameterized.parameters(
      generate_quantization_test_config([_add_sample_inputs])
  )
  def test_extend_step(
      self,
      dconv_qkv,
      combine_qkv,
      output_proj_use_nhd_shape,
      use_rotary_position_emb,
      cast_rotary_position_emb,
      zero_fully_masked,
      simulate_packed,
      init_proj_weight,
      init_post_weight,
      init_dconv_weight,
      query,
      pseudo_ans
  ):
    if not use_rotary_position_emb and cast_rotary_position_emb:
      return

    batch_size = 1
    num_heads = 2
    max_seq_len = 4
    input_dim = 4
    dconv_kernel_size = 3
    hidden_dim = 8

    query = (
        np.array(query)
        .reshape([batch_size, max_seq_len, input_dim])
        .astype('f4')
    )
    pseudo_ans = (
        np.array(pseudo_ans)
        .reshape([batch_size, max_seq_len, input_dim])
        .astype('f4')
    )

    quantization_option = QuantizationHParams(
        quantization_type=QuantizationType.AQT,
        mode=QuantizationMode.TRAINING,
        act_params=quantization_hparams.ActQuantizationParams(precision=23),
        weight_params=None,
    )

    atten_q = pax_fiddle.Config(
        qattentions.DotProductAttention,
        name='_dotprod_q',
        input_dim=input_dim,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        dconv_qkv=dconv_qkv,
        dconv_kernel_size=dconv_kernel_size,
        combine_qkv=combine_qkv,
        output_proj_use_nhd_shape=output_proj_use_nhd_shape,
        use_rotary_position_emb=use_rotary_position_emb,
        cast_rotary_position_emb=cast_rotary_position_emb,
        zero_fully_masked=zero_fully_masked,
        quantization=quantization_option,
    )

    atten_q = instantiate(atten_q)
    fake_query = jnp.zeros_like(query)
    atten_mask = attentions.causal_mask(query)
    segment_pos = np.tile(np.arange(max_seq_len), (batch_size, 1))

    starting_index = 0
    if simulate_packed:
      starting_index = dconv_kernel_size
      atten_mask = atten_mask.at[:, :, :, :starting_index].set(-2.3819763e38)
      segment_pos = jnp.maximum(segment_pos - starting_index, 0)

    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)
    initial_vars = atten_q.init(
        init_key,
        fake_query,
        fake_query,
        fake_query,
        atten_mask,
        query_segment_pos=segment_pos,
        key_segment_pos=segment_pos,
    )

    initial_vars = self._initialize_weights(
        initial_vars,
        init_proj_weight,
        init_post_weight,
        init_dconv_weight,
        dconv_kernel_size,
        combine_qkv,
        dconv_qkv,
    )

    updated_vars_q = self._update(
        atten_q,
        initial_vars,
        query,
        query,
        query,
        atten_mask,
        segment_pos,
        pseudo_ans,
    )

    # 1. Full Inference result.
    fprop_out_q, attention_states_init = atten_q.apply(
        updated_vars_q,
        query,
        query,
        query,
        atten_mask,
        query_segment_pos=segment_pos,
        key_segment_pos=segment_pos,
        mutable=[base_layer.DECODE_CACHE]
    )
    fprop_out_q = fprop_out_q[0]

    for k, v in attention_states_init[base_layer.DECODE_CACHE].items():
      shape = v.shape
      attention_states_init[base_layer.DECODE_CACHE][k] = jnp.zeros(shape)
    updated_vars = py_utils.merge_dict(attention_states_init, updated_vars_q)

    # 3. Step-by-Step decoding.
    decoder_output_q = jnp.zeros(shape=[max_seq_len, batch_size, input_dim])
    for t in range(starting_index, max_seq_len):
      encoded, attention_states = atten_q.apply(
          updated_vars,
          query_vec=query[:, t, :],
          atten_mask=atten_mask[:, :, t, :],
          time_step=t,
          segment_pos=None,
          method=atten_q.extend_step,
          mutable=[base_layer.DECODE_CACHE],
      )
      updated_vars = py_utils.merge_dict(attention_states, updated_vars_q)
      decoder_output_q = decoder_output_q.at[t].set(encoded)

    decoder_output_q = decoder_output_q[starting_index:]
    decoder_out_transposed_q = jnp.transpose(decoder_output_q, [1, 0, 2])

    self.assertAllClose(
        fprop_out_q[:, starting_index:, :], decoder_out_transposed_q, atol=1e-3
    )


if __name__ == '__main__':
  absltest.main()
