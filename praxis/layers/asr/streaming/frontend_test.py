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

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import py_utils
from praxis import pytypes
from praxis.layers.asr.streaming import frontend
from praxis.layers.streaming import test_utils

RANDOM = base_layer.RANDOM
PARAMS = base_layer.PARAMS

JTensor = pytypes.JTensor
NestedMap = py_utils.NestedMap
NestedJTensor = py_utils.NestedMap
instantiate = base_layer.instantiate


class StreamingFramingTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(1234)

  def test_pad_end_call_error(self):
    features = jnp.zeros((1, 8, 1))
    paddings = None
    framing_p = frontend.Framing.HParams(
        name='framing', frame_size=2, frame_step=2, pad_end=True)
    framing_layer = instantiate(framing_p)
    with base_layer.JaxContext.new_context():
      with self.assertRaises(NotImplementedError):
        framing_layer.init(
            jax.random.PRNGKey(123), features=features, paddings=paddings)

  def test_features_dtype(self):
    features = jnp.zeros((1, 8, 1))
    paddings = None
    framing_p = frontend.Framing.HParams(
        name='framing',
        frame_size=2,
        frame_step=2,
        pad_end=False,
        features_dtype=np.int16)
    framing_layer = instantiate(framing_p)
    with base_layer.JaxContext.new_context():
      with self.assertRaisesRegex(ValueError, 'float32'):
        framing_layer.init(
            jax.random.PRNGKey(123), features=features, paddings=paddings)

  @parameterized.product(
      frame_size=[1, 2, 3],
      frame_step=[1, 2, 3],
      stream_step_multiplier=[1, 2, 3])
  def test_stream_step_against_call(self, frame_size, frame_step,
                                    stream_step_multiplier):
    """Tests that streaming is numerically identical to non-streaming."""
    # run_streaming requires input_length to be a multiple of
    # stream_step_size, and framing requires stream_step_size to be a multiple
    # of frame_step.
    stream_step_size = frame_step * stream_step_multiplier
    input_length = stream_step_size * 10
    shape = (2, input_length, 1)
    features = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
    paddings = np.random.randint(2, size=shape[:2]).astype(np.float32)

    framing_p = frontend.Framing.HParams(
        name='framing',
        frame_size=frame_size,
        frame_step=frame_step,
        pad_end=False)

    self._compare_stream_non_stream(
        features,
        paddings,
        # Due to padding in Framing, streaming and non-streaming versions of the
        # frontend layers are numerically different, but stream_step and
        # __call__ for each streaming layer must match.
        p_non_stream=framing_p,
        p_stream=framing_p,
        step=stream_step_size,
    )


class StreamingFrontendTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(1234)

  def _generate_inputs(self):
    shape = (2, 1600, 1)
    input_signal = np.random.uniform(-32768.0, 32767.0, size=shape)
    input_paddings = np.random.randint(2, size=shape[:2]).astype(np.float32)
    return input_signal, input_paddings

  @parameterized.product(
      compute_energy=[True, False],
      window_fn=['HANNING', 'HANNING_NONZERO'],
      output_log=[True, False],
      preemph=[0, 0.97],
      preemph_htk_flavor=[True, False],
  )
  def test_spectrogram_frontend_streaming_step(self, compute_energy, window_fn,
                                               output_log, preemph,
                                               preemph_htk_flavor):
    """Tests that the duplicated streaming_step matches the parent __call__."""
    input_signal, input_paddings = self._generate_inputs()
    layer_p = frontend.SpectrogramFrontend.HParams(
        name='spectrogram',
        sample_rate=16000.0,
        frame_size_ms=25.0,
        frame_step_ms=10.0,
        compute_energy=compute_energy,
        window_fn=window_fn,
        output_log=output_log,
        preemph=preemph,
        preemph_htk_flavor=preemph_htk_flavor,
        noise_scale=0.0)
    self._compare_stream_non_stream(
        input_signal,
        input_paddings,
        # Due to padding in Framing, streaming and non-streaming versions of the
        # frontend layers are numerically different, but stream_step and
        # __call__ for each streaming layer must match.
        p_non_stream=layer_p,
        p_stream=layer_p,
        step=frontend.SpectrogramFrontend.get_stride(layer_p),
    )

  @parameterized.product(
      compute_energy=[True, False],
      window_fn=['HANNING', 'HANNING_NONZERO'],
      preemph=[0, 0.97],
      preemph_htk_flavor=[True, False],
      use_divide_stream=[True, False],
  )
  def test_mel_filterbank_frontend_streaming_step(self, compute_energy,
                                                  window_fn, preemph,
                                                  preemph_htk_flavor,
                                                  use_divide_stream):
    """Tests that the duplicated streaming_step matches the parent __call__."""
    input_signal, input_paddings = self._generate_inputs()
    layer_p = frontend.MelFilterbankFrontend.HParams(
        name='mel_spectrogram',
        sample_rate=16000.0,
        frame_size_ms=25.0,
        frame_step_ms=10.0,
        compute_energy=compute_energy,
        window_fn=window_fn,
        preemph=preemph,
        use_divide_stream=use_divide_stream,
        preemph_htk_flavor=preemph_htk_flavor,
        noise_scale=0.0)
    self._compare_stream_non_stream(
        input_signal,
        input_paddings,
        # Due to padding in Framing, streaming and non-streaming versions of the
        # frontend layers are numerically different, but stream_step and
        # __call__ for each streaming layer must match.
        p_non_stream=layer_p,
        p_stream=layer_p,
        step=frontend.MelFilterbankFrontend.get_stride(layer_p),
    )


if __name__ == '__main__':
  absltest.main()
