# coding=utf-8
# Copyright 2022 Google LLC.
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

"""Tests for frontend."""

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.experimental.jax2tf
import numpy as np
from praxis import base_layer
from praxis import py_utils
from praxis import pytypes
from praxis import test_utils
from praxis.layers.asr import frontend
import tensorflow as tf


RANDOM = base_layer.RANDOM
PARAMS = base_layer.PARAMS

FLAGS = flags.FLAGS
JTensor = pytypes.JTensor
NestedMap = py_utils.NestedMap
NestedJTensor = py_utils.NestedMap
instantiate = base_layer.instantiate


class FramingTest(test_utils.TestCase):

  def test_basic_framing(self):
    """Smoke test for Framing."""
    frame_size = 3
    frame_step = 2
    shape = (2, 8, 1)
    features = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
    paddings = np.zeros(shape[:2]).astype(np.float32)
    paddings[0, 4:] = 1.0
    paddings[1, 5:] = 1.0

    expected_frames = np.array([[
        [0., 1., 2.],
        [2., 3., 4.],
        [4., 5., 6.],
        [6., 7., 0.],
    ], [
        [8., 9., 10.],
        [10., 11., 12.],
        [12., 13., 14.],
        [14., 15., 0.],
    ]])
    # Add singleton channels dim.
    expected_frames = expected_frames[:, :, :, None]
    expected_paddings = np.array([[0., 0., 1., 1.], [0., 0., 0., 1.]])

    framing_p = frontend.Framing.HParams(
        name='framing',
        frame_size=frame_size,
        frame_step=frame_step,
        pad_end=True)
    framing_layer = instantiate(framing_p)

    with base_layer.JaxContext.new_context():
      initial_vars = framing_layer.init(
          jax.random.PRNGKey(123), features=features, paddings=paddings)
      framed_features, framed_paddings = framing_layer.apply(
          initial_vars, features, paddings)

    self.assertAllClose(framed_features, expected_frames)
    self.assertAllClose(framed_paddings, expected_paddings)


class SpectrogramFrontendTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(1234)

  @parameterized.named_parameters(
      dict(testcase_name='default', pad_end=False, dtype=np.int32),
      dict(testcase_name='with_float', pad_end=False, dtype=np.float32),
      dict(testcase_name='with_pad_end', pad_end=True, dtype=np.int32))
  def test_spectrogram_with_trivial_case_zero(self, pad_end, dtype):
    signal_length = 999
    padded_signal_length = 1401
    input_signal = np.zeros((1, padded_signal_length), dtype=dtype)
    input_paddings = np.ones((1, padded_signal_length))
    input_paddings[:, :signal_length] = 0.0

    # First 5 frames are expected to be valid
    num_expected_pads = 3 if pad_end else 1
    expected_paddings = np.array([[0.0] * 5 + [1.0] * num_expected_pads])

    p = frontend.SpectrogramFrontend.HParams(
        name='spectrogram',
        frame_size_ms=25.0,
        frame_step_ms=12.5,
        pad_end=pad_end,
        noise_scale=0.0)
    spectrogram_layer = instantiate(p)
    with base_layer.JaxContext.new_context():
      params_key, random_key = jax.random.split(jax.random.PRNGKey(seed=123))
      initial_vars = spectrogram_layer.init(
          rngs={
              PARAMS: params_key,
              RANDOM: random_key
          },
          inputs=input_signal,
          input_paddings=input_paddings)
      spect, spect_paddings = spectrogram_layer.apply(initial_vars,
                                                      input_signal,
                                                      input_paddings)

    expected_shape = (1, 8, 257, 1) if pad_end else (1, 6, 257, 1)
    self.assertSequenceEqual(spect.shape, expected_shape)
    self.assertAllClose(spect, np.zeros_like(spect, dtype=np.float32))
    self.assertSequenceEqual(spect.shape[:2], spect_paddings.shape[:2])
    self.assertAllClose(spect_paddings, expected_paddings)

  @parameterized.product(
      pad_end=[False, True],
      fft_length=[512, 1024],
      input_length=[1024, 1357, 2048, 2123])
  def test_spectrogram_against_tf(self, pad_end, fft_length, input_length):
    frame_size_ms = 25.0
    frame_step_ms = 10.0
    sample_rate = 16000.0
    input_shape = (2, input_length, 1)
    input_signal = np.random.uniform(-32768.0, 32767.0, size=input_shape)
    input_paddings = np.zeros(input_shape[:2])

    ref_spect_tf = tf.signal.stft(
        input_signal.squeeze(-1),
        int(sample_rate * frame_size_ms / 1000),
        int(sample_rate * frame_step_ms / 1000),
        fft_length=fft_length,
        window_fn=tf.signal.hann_window,
        pad_end=pad_end)
    ref_spect = tf.abs(ref_spect_tf).numpy()

    spectrogram_layer = instantiate(
        frontend.SpectrogramFrontend.HParams(
            name='spectrogram',
            sample_rate=sample_rate,
            frame_size_ms=frame_size_ms,
            frame_step_ms=frame_step_ms,
            noise_scale=0.0,
            fft_size=fft_length,
            pad_end=pad_end,
            preemph=0.0))
    with base_layer.JaxContext.new_context():
      params_key, random_key = jax.random.split(jax.random.PRNGKey(seed=123))
      initial_vars = spectrogram_layer.init(
          rngs={
              PARAMS: params_key,
              RANDOM: random_key
          },
          inputs=input_signal,
          input_paddings=input_paddings)
      spect, unused_spect_paddings = spectrogram_layer.apply(
          initial_vars, input_signal, input_paddings)

    self.assertAllClose(spect.squeeze(-1), ref_spect, rtol=1e-4)


class MelFilterbankFrontendTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(1234)

  @parameterized.named_parameters(
      ('40-to-12', 16000.0, 40, 12, 125.0, 7600.0),
      ('513-to-80', 16000.0, 513, 80, 125.0, 7600.0),
      ('1025-to-64', 16000.0, 1025, 64, 1.0, 8000.0))
  def test_mel_weight_matrix(self, sample_rate: float,
                             num_spectrogram_bins: int, num_mel_bins: int,
                             lower_edge_hertz: float, upper_edge_hertz: float):

    def mel_weight_gen():
      return tf.signal.linear_to_mel_weight_matrix(
          num_mel_bins=num_mel_bins,
          num_spectrogram_bins=num_spectrogram_bins,
          sample_rate=sample_rate,
          lower_edge_hertz=lower_edge_hertz,
          upper_edge_hertz=upper_edge_hertz)

    ref_mel_weights = jax.experimental.jax2tf.call_tf(mel_weight_gen)()
    test_mel_weights = frontend.linear_to_mel_weight_matrix(
        num_mel_bins=num_mel_bins,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=sample_rate,
        lower_edge_hertz=lower_edge_hertz,
        upper_edge_hertz=upper_edge_hertz)
    self.assertAllClose(ref_mel_weights, test_mel_weights, atol=1e-4)

  def test_logmelfb_shapes(self):
    signal_length = 799
    padded_signal_length = 1201  # +1 needed for pre-emphasis
    input_signal = np.zeros((1, padded_signal_length, 1), dtype=np.float32)
    input_paddings = np.ones((1, padded_signal_length))
    input_paddings[:, :signal_length] = 0.0
    # First 5 frames are expected to be valid
    expected_paddings = np.array([[0.0] * 5 + [1.0]])

    p = frontend.MelFilterbankFrontend.HParams(
        name='mfb',
        num_bins=80,
        sample_rate=16000.0,
    )
    mfb_layer = instantiate(p)

    with base_layer.JaxContext.new_context():
      params_key, random_key = jax.random.split(jax.random.PRNGKey(seed=123))
      layer_vars = mfb_layer.init(
          rngs={
              PARAMS: params_key,
              RANDOM: random_key
          },
          inputs=input_signal,
          input_paddings=input_paddings)
      feats, feat_paddings = mfb_layer.apply(
          layer_vars, input_signal, input_paddings, rngs={RANDOM: random_key})

    self.assertSequenceEqual(feats.shape, (1, 6, 80, 1))
    self.assertAllClose(feat_paddings, expected_paddings)

  def test_normalizer(self):
    """Tests normalizer routine."""
    signal_length = 1205
    padded_signal_length = 1500
    input_signal = np.random.uniform(
        -1.0, 1.0, size=(1, padded_signal_length, 1)) * 32768.0
    input_paddings = np.ones(input_signal.shape[:2])
    input_paddings[:, :signal_length] = 0.0

    raw_p = frontend.MelFilterbankFrontend.HParams(name='mfb', noise_scale=0.0)
    raw_layer = instantiate(raw_p)
    raw_vars = raw_layer.init(
        jax.random.PRNGKey(0), input_signal, input_paddings)

    normalizer_mean = np.random.normal(size=(raw_p.num_bins,))
    normalizer_stddev = np.random.gamma(1.0, size=(raw_p.num_bins,))
    norm_p = frontend.MelFilterbankFrontend.HParams(
        name='mfb',
        noise_scale=0.0,
        per_bin_mean=normalizer_mean.tolist(),
        per_bin_stddev=normalizer_stddev.tolist())
    norm_layer = instantiate(norm_p)
    norm_vars = norm_layer.init(
        jax.random.PRNGKey(0), input_signal, input_paddings)

    raw_feats, raw_feat_paddings = raw_layer.apply(raw_vars, input_signal,
                                                   input_paddings)
    norm_feats, norm_feat_paddings = norm_layer.apply(norm_vars, input_signal,
                                                      input_paddings)

    self.assertAllClose(raw_feat_paddings, norm_feat_paddings)
    # Just redo the same logic as in the tested code.
    #  1. mean subtraction
    expected_norm_feats = (
        raw_feats - normalizer_mean[np.newaxis, np.newaxis, :, np.newaxis])
    #  2. scale inverse stddev
    expected_norm_feats /= (
        normalizer_stddev[np.newaxis, np.newaxis, :, np.newaxis])
    self.assertAllClose(norm_feats, expected_norm_feats)


class IdFrontendLayer(base_layer.BaseLayer):

  def __call__(self, *args):
    return args


class StackedFeatureFrontendTest(test_utils.TestCase):

  @parameterized.product(repeat_left=[True, False], repeat_right=[True, False])
  def test_fprop_default(self, repeat_left, repeat_right):
    fe_p = frontend.FeatureStackingFrontend.HParams(
        name='stacked',
        frontend=IdFrontendLayer.HParams(),
        pad_with_left_frame=repeat_left,
        pad_with_right_frame=repeat_right,
        left_context=3,
        right_context=0,
        stride=3)
    fe = fe_p.Instantiate()
    features = np.array([2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8])
    paddings = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.float32)

    features = features[np.newaxis, :, np.newaxis, np.newaxis]
    paddings = paddings[np.newaxis, :]

    fe_vars = fe.init(jax.random.PRNGKey(0), features, paddings)
    stacked_features, stacked_paddings = fe.apply(fe_vars, features, paddings)

    expected_features = np.array([
        [0, 0, 0, 2],
        [2, 1, 2, 3],
        [3, 4, 5, 6],
        [6, 7, 8, 9],  # this is a padding row but keeps the original values.
    ])[np.newaxis, :, :, np.newaxis]
    if repeat_left:
      expected_features[0, 0, 0:3, :] = 2
    expected_paddings = np.array([0, 0, 0, 1], dtype=np.float32)[np.newaxis, :]
    self.assertAllClose(stacked_features, expected_features)
    self.assertAllClose(stacked_paddings, expected_paddings)


if __name__ == '__main__':
  absltest.main()
