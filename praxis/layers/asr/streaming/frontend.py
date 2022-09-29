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

"""A collection of streaming frontend modules for speech processing."""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import py_utils
from praxis import pytypes
from praxis.layers import streaming
from praxis.layers.asr import frontend


NestedMap = py_utils.NestedMap
NestedJTensor = pytypes.NestedJTensor
JTensor = pytypes.JTensor
NpTensor = pytypes.NpTensor

sub_config_field = base_layer.sub_config_field
BaseHParams = base_layer.BaseLayer.HParams


class Framing(  # pytype: disable=signature-mismatch
    frontend.Framing, streaming.StreamingBase):
  """Slides a window and extracts values.

  This layer is not numerically equivalent to frontend.Framing if there is an
  overlap between frames as the start of the input in __call__ must be padded
  for equivalence with streaming_step.
  """

  class HParams(frontend.Framing.HParams):
    features_dtype: np.dtype = np.float32

  def setup(self) -> None:
    if self.hparams.pad_end:
      raise NotImplementedError(
          'pad_end=True is not supported in streaming mode')

  @classmethod
  def get_right_context(cls, p: frontend.Framing.HParams) -> int:
    return 0

  @classmethod
  def get_stride(cls, p: frontend.Framing.HParams) -> int:
    return p.frame_step

  @property
  def frame_overlap(self) -> int:
    p = self.hparams
    return max(0, p.frame_size - p.frame_step)

  def _validate_features(self, features):
    if features.shape[-1] > 1:
      raise NotImplementedError(
          'Streaming is only supported for single-channel features')
    if features.dtype != self.hparams.features_dtype:
      raise ValueError(
          f'features.dtype={features.dtype} does not match p.feature_dtype='
          f'{self.hparams.features_dtype}')

  def _maybe_pad_start(self, x):
    """Pad the start of the input for compatibility with streaming_step."""
    # Padding is only required for numerical equivalence with streaming_step if
    # there's an overlap between frames.
    if self.frame_overlap > 0:
      pad_width = ((0, 0), (self.frame_overlap, 0))
      # No padding for any dims after the time dim.
      pad_width += ((0, 0),) * (len(x.shape) - 2)
      x = jnp.pad(x, pad_width, 'constant', constant_values=0.0)
    return x

  def __call__(
      self,
      features: JTensor,
      paddings: Optional[JTensor] = None) -> Tuple[JTensor, Optional[JTensor]]:
    """Audio signal framing.

    Args:
      features: Input sequence with shape (batch_size, timesteps, channels)
      paddings: Optional paddings with shape (batch_size, timesteps)

    Returns:
      Features with shape (batch_size, num_frames, frame_size, channels) and
      optional framed paddings with shape (batch_size, num_frames).
    """
    self._validate_features(features)
    # Pad the input if needed for compatibility with streaming_step.
    features = self._maybe_pad_start(features)
    if paddings is not None:
      paddings = self._maybe_pad_start(paddings)
    return super().__call__(features, paddings)

  def init_states(self, batch_size: int, with_paddings: bool = True):
    """Creates streaming states in base_layer.DECODE_CACHE.

    Args:
      batch_size: defines batch size of streaming states.
      with_paddings: if True it will creates streaming states for padding
        processing, else will set it None (it can save some memory).
    """
    if self.frame_overlap > 0:
      self._update_streaming_state(
          'features_context',
          jnp.zeros([batch_size, self.frame_overlap, 1],
                    dtype=self.hparams.features_dtype))
    else:
      self._update_streaming_state('features_context', None)

    if self.frame_overlap > 0 and with_paddings:
      self._update_streaming_state(
          'paddings_context',
          jnp.zeros([batch_size, self.frame_overlap], dtype=jnp.float32))
    else:
      self._update_streaming_state('paddings_context', None)

  def streaming_step(self, inputs: NestedJTensor) -> NestedJTensor:
    """Streaming audio signal framing.

    Args:
      inputs: A nested map of features with shape (batch_size, timesteps,
        channels) and optional paddings with shape (batch_size, timesteps).

    Returns:
      A nested map of framed features with shape (batch_size, num_frames,
      frame_size, channels) and optional framed paddings with shape
      (batch_size, num_frames).
    """
    self._validate_features(inputs.features)
    timesteps = inputs.features.shape[1]
    if timesteps % self.stride != 0:
      raise ValueError('Expected the time dim of features to be a multiple of '
                       f'self.stride = {self.stride} but got {timesteps}')

    # Add the context from the previous streaming_step to features and paddings.
    # If the contexts are None then the layer uses stateless streaming.
    features_context = self.get_streaming_state('features_context')
    if features_context is not None:
      features = jnp.concatenate([features_context, inputs.features], axis=1)
      self._update_streaming_state('features_context',
                                   features[:, -self.frame_overlap:, :])
    else:
      features = inputs.features

    paddings_context = self.get_streaming_state('paddings_context')
    if paddings_context is not None:
      if inputs.paddings is None:
        raise ValueError('The streaming paddings_context was created '
                         '(with_paddings=True) but the input paddings are None')
      paddings = jnp.concatenate([paddings_context, inputs.paddings], axis=1)
      self._update_streaming_state('paddings_context',
                                   paddings[:, -self.frame_overlap:])
    else:
      paddings = inputs.paddings

    # The superclass call is used to avoid padding the start of each input.
    features, paddings = super().__call__(features, paddings)
    return NestedMap(features=features, paddings=paddings)


class SpectrogramFrontend(  # pytype: disable=signature-mismatch
    frontend.SpectrogramFrontend, streaming.StreamingBase):
  """Perform STFT to convert 1D signals to spectrograms.

  The `fprop` input to this layer is assumed to have a shape
  `(batch_size, timesteps, channel)`, and an optional padding array should have
  shape `(batchsize, timesteps)`.  The output frame-level padding indicators
  are calculated when the sample-level paddings are given as an argument.
  """

  class HParams(frontend.SpectrogramFrontend.HParams):
    # Replace frontend.Framing by its streaming aware version:
    _attribute_overrides: Tuple[str, ...] = ('framing_tpl',)
    framing_tpl: BaseHParams = sub_config_field(Framing.HParams)

  def setup(self) -> None:
    if not issubclass(self.hparams.framing_tpl.cls, streaming.StreamingBase):
      raise TypeError(
          'Expected p.framing_tpl to be a streamable layer inheriting from '
          f'streaming.StreamingBase but got {self.hparams.framing_tpl.cls}')
    super().setup()

  def init_states(self, batch_size: int, with_paddings: bool = True):
    self.framing.init_states(batch_size, with_paddings)

  @classmethod
  def get_right_context(cls, p: frontend.SpectrogramFrontend.HParams) -> int:
    framing_p = cls.configure_framing(p)
    return framing_p.cls.get_right_context(framing_p)

  @classmethod
  def get_stride(cls, p: frontend.SpectrogramFrontend.HParams) -> int:
    framing_p = cls.configure_framing(p)
    return framing_p.cls.get_stride(framing_p)

  def streaming_step(self, inputs: NestedJTensor) -> NestedJTensor:
    """Convert 1D-inputs and optional paddings to spectrogram.

    Identical to frontend.SpectrogramFrontend.__call__ except for:
      - self.framing.streaming_step() is used instead of self.framing()
      - the inputs and outputs are NestedJTensors
    Must be updated if frontend.SpectrogramFrontend.__call__ is updated.

    Args:
      inputs: An nested map of features with shape (batchsize, timesteps,
        channels) and optional paddings with shape (batchsize, timesteps).

    Returns:
      A nested map with `features` and `paddings` where `features` is a
      tensor with shape `(batch_size, frames, fft_size // 2 + 1, channels)`
      where frames is the number of frames extracted from the signal with
      `timesteps`-length. `paddings[b, n]` is the padding indicator for
      (b, n)-th frame that is 1.0 if all the samples corresponding to this frame
      are paddings, and 0.0 otherwise.
    """
    inputs, input_paddings = inputs.features.astype(np.float32), inputs.paddings
    p = self.hparams

    # Expand to have a channel axis
    if inputs.ndim == 2:
      inputs = jnp.expand_dims(inputs, -1)

    if input_paddings is not None:
      inputs = inputs * jnp.expand_dims(1.0 - input_paddings, -1)

    pcm_audio_chunk = inputs * p.input_scale_factor

    # framed_signal.shape = [batch, frames, frame_size + 1, channels]
    framed_nmap = self.framing.streaming_step(
        NestedMap(features=pcm_audio_chunk, paddings=input_paddings))
    framed_signal, output_paddings = framed_nmap.features, framed_nmap.paddings

    # preemphasized.shape = [batch, frames, frame_size, channels]
    if p.preemph != 0.0:
      preemphasized = self._apply_preemphasis(framed_signal)
    else:
      preemphasized = framed_signal[..., :-1, :]

    if p.noise_scale > 0.0:
      noise_signal = jax.random.normal(self.next_prng_key(),
                                       preemphasized.shape) * p.noise_scale
    else:
      noise_signal = jnp.zeros(preemphasized.shape)

    windowed_signal = preemphasized + noise_signal
    # Window here
    if self._window_fn is not None:
      window = self._window_fn(self.frame_size, framed_signal.dtype)
      window = window.reshape((1, 1, self.frame_size, 1))
      windowed_signal *= window

    spectrum = jnp.fft.rfft(windowed_signal, self.fft_size, axis=2)
    spectrum = jnp.abs(spectrum)
    if p.compute_energy:
      spectrum = spectrum**2.0

    outputs = spectrum
    if p.output_log:
      outputs = jnp.log(jnp.maximum(outputs, p.output_log_floor))

    return NestedMap(features=outputs, paddings=output_paddings)


class MelFilterbankFrontend(  # pytype: disable=signature-mismatch
    frontend.MelFilterbankFrontend, streaming.StreamingBase):
  """Compute Log-mel-filterbank outputs from 1D signal.

  This layer partially achieve parameter-level compatibility with its
  counterpart in TF (`lingvo.tasks.asr.frontend.MelAsrFrontend`) except the
  parameter related to post-filterbank processing.
  """

  class HParams(frontend.MelFilterbankFrontend.HParams):
    # Replace frontend.SpectrogramFrontend by its streaming aware version:
    _attribute_overrides: Tuple[str, ...] = ('stft_tpl',)
    stft_tpl: BaseHParams = sub_config_field(SpectrogramFrontend.HParams)

  def setup(self) -> None:
    if not issubclass(self.hparams.stft_tpl.cls, streaming.StreamingBase):
      raise TypeError(
          'Expected p.stft_tpl to be a streamable layer inheriting from '
          f'streaming.StreamingBase but got {self.hparams.stft_tpl.cls}')
    super().setup()

  def init_states(self, batch_size: int, with_paddings: bool = True):
    self.stft.init_states(batch_size, with_paddings)

  @classmethod
  def get_right_context(cls, p: frontend.MelFilterbankFrontend.HParams) -> int:
    stft_p = cls.configure_stft(p)
    return stft_p.cls.get_right_context(stft_p)

  @classmethod
  def get_stride(cls, p: frontend.MelFilterbankFrontend.HParams) -> int:
    stft_p = cls.configure_stft(p)
    return stft_p.cls.get_stride(stft_p)

  def streaming_step(self, inputs: NestedJTensor) -> NestedJTensor:
    """Computed log-mel-filterbank features.

    Identical to frontend.MelFilterbankFrontend.__call__ except for:
      - self.stft.streaming_step() is used instead of self.stft()
      - the inputs and outputs are NestedJTensors
    Must be updated if frontend.MelFilterbankFrontend.__call__ is updated.

    Args:
      inputs: A nested map with features and paddings to be processed by the
        `stft` sublayer.

    Returns:
      A nested map with `features` and `paddings`. `features` is an array with
      shape `(batch_size, num_frames, p.num_bins, num_channels)` where
      `num_frames` is the number of frames obtained from `stft` sublayer.
      `paddings` is None if `inputs.paddings` is None, otherwise it
      contains frame-level padding indicators obtained from `stft`.
    """
    p = self.hparams

    spect_nmap = self.stft.streaming_step(inputs)
    spect, spect_paddings = spect_nmap.features, spect_nmap.paddings
    mel_weights = frontend.linear_to_mel_weight_matrix(
        num_mel_bins=self.hparams.num_bins,
        num_spectrogram_bins=spect.shape[2],
        sample_rate=self.hparams.sample_rate,
        lower_edge_hertz=self.hparams.lower_edge_hertz,
        upper_edge_hertz=self.hparams.upper_edge_hertz)

    mel_spectrogram = jnp.einsum('fn,btfc->btnc', mel_weights, spect)
    logmel_spectrogram = jnp.log(jnp.maximum(mel_spectrogram, p.output_floor))

    normalized_logmel_spectrogram = (
        (logmel_spectrogram - self._normalizer_mean) / self._normalizer_stddev)

    return NestedMap(
        features=normalized_logmel_spectrogram, paddings=spect_paddings)
