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

"""A collection of frontend modules for speech processing."""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import py_utils
from praxis import pytypes
from praxis.layers import linears

NestedMap = py_utils.NestedMap
JTensor = pytypes.JTensor
NpTensor = pytypes.NpTensor

sub_config_field = base_layer.sub_config_field
BaseHParams = base_layer.BaseLayer.HParams
StackingOverTime = linears.StackingOverTime

# mel spectrum constants.
_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0


def _hertz_to_mel(frequencies_hertz: Union[float, int, NpTensor]) -> NpTensor:
  """Convert hertz to mel."""
  return _MEL_HIGH_FREQUENCY_Q * np.log(1.0 + (frequencies_hertz /
                                               _MEL_BREAK_FREQUENCY_HERTZ))


def _pad_end_length(num_timesteps: int, frame_step: int,
                    frame_size: int) -> int:
  """Returns how many sample needed to be padded for pad_end feature."""
  # The number of frames that can be extracted from the signal.
  num_frames = int(np.ceil(num_timesteps / frame_step))
  # Signal length required for computing `num_frames` frames.
  padded_length = frame_step * (num_frames - 1) + frame_size
  return padded_length - num_timesteps


def frame(x: JTensor,
          frame_length: int,
          frame_step: int,
          pad_end: bool = False,
          pad_value: Union[int, float] = 0.0) -> JTensor:
  """Slides a window and extracts frames.

  This function extracts `x[:, n:n+frame_length, :]` with sliding `n` with
  stride of `frame_step`, and returns an array `y` with the shape
  `(batch_size, num_frames, frame_length, num_channels)`. Unlike the
  counterpart in Tensorflow (`tf.signal.frame`), this function currently does
  not take `axis` argument, and the input tensor `x` is expected to have a
  shape of `(batch_size, timesteps, channels)`.

  Args:
    x: An input array with `(batch_size, timesteps, channels)`-shape.
    frame_length: The frame length.
    frame_step: The frame hop size.
    pad_end: If True, the end of the signal is padded so the window can continue
      sliding while the starting point of the window is in the valid range.
    pad_value: A scalar used as a padding value when `pad_end` is True.

  Returns:
    A tensor with shape `(batch_size, num_frames, frame_length, num_channels)`.
  """
  unused_batch_size, num_timesteps, num_channels = x.shape

  if pad_end:
    num_extends = _pad_end_length(num_timesteps, frame_step, frame_length)
    x = jnp.pad(
        x, ((0, 0), (0, num_extends), (0, 0)),
        'constant',
        constant_values=pad_value)

  flat_y = jax.lax.conv_general_dilated_patches(
      x, (frame_length,), (frame_step,),
      'VALID',
      dimension_numbers=('NTC', 'OIT', 'NTC'))
  ret = flat_y.reshape(flat_y.shape[:-1] + (num_channels, frame_length))
  return ret.transpose((0, 1, 3, 2))


class Framing(base_layer.BaseLayer):
  """Slides a window and extracts frames."""

  class HParams(BaseHParams):
    """Associated hyper-params for this layer class.

    Attributes:
      frame_size: The frame size.
      frame_step: The frame hop size.
      pad_end: If True, the end of the signal is padded so the window can
        continue sliding while the starting point of the window is in the valid
        range.
      pad_value: A scalar used as a padding value when `pad_end` is True.
    """
    frame_size: int = 0
    frame_step: int = 0
    pad_end: bool = False
    pad_value: Union[int, float] = 0.0

  def frame_paddings(self, paddings: JTensor) -> JTensor:
    """Specialized implementation for paddings using reduce_window."""
    p = self.hparams
    if p.pad_end:
      num_extends = _pad_end_length(paddings.shape[1], p.frame_step,
                                    p.frame_size)
      paddings = jnp.pad(
          paddings, ((0, 0), (0, num_extends)), constant_values=1.0)
    return jax.lax.reduce_window(
        paddings,
        init_value=1.0,
        computation=jax.lax.min,
        window_dimensions=[1, p.frame_size],
        window_strides=[1, p.frame_step],
        padding='valid')

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
    p = self.hparams
    features = frame(features, p.frame_size, p.frame_step, p.pad_end,
                     p.pad_value)
    if paddings is not None:
      paddings = self.frame_paddings(paddings)
    return features, paddings


def linear_to_mel_weight_matrix(num_mel_bins: int = 20,
                                num_spectrogram_bins: int = 129,
                                sample_rate: Union[int, float] = 8000,
                                lower_edge_hertz: Union[int, float] = 125.0,
                                upper_edge_hertz: Union[int, float] = 3800.0,
                                dtype: Any = np.float32) -> NpTensor:
  r"""Numpy-port of `tf.signal.linear_to_mel_weight_matrix`.

  Note that this function works purely on numpy because mel-weights are
  shape-dependent constants that usually should not be computed in an
  accelerators.

  Args:
    num_mel_bins: Python int. How many bands in the resulting mel spectrum.
    num_spectrogram_bins: An integer `Tensor`. How many bins there are in the
      source spectrogram data, which is understood to be `fft_size // 2 + 1`,
      i.e. the spectrogram only contains the nonredundant FFT bins.
    sample_rate: An integer or float `Tensor`. Samples per second of the input
      signal used to create the spectrogram. Used to figure out the frequencies
      corresponding to each spectrogram bin, which dictates how they are mapped
      into the mel scale.
    lower_edge_hertz: Python float. Lower bound on the frequencies to be
      included in the mel spectrum. This corresponds to the lower edge of the
      lowest triangular band.
    upper_edge_hertz: Python float. The desired top edge of the highest
      frequency band.
    dtype: The `DType` of the result matrix. Must be a floating point type.

  Returns:
    An array of shape `[num_spectrogram_bins, num_mel_bins]`.
  Raises:
    ValueError: If `num_mel_bins`/`num_spectrogram_bins`/`sample_rate` are not
      positive, `lower_edge_hertz` is negative, frequency edges are incorrectly
      ordered, `upper_edge_hertz` is larger than the Nyquist frequency.
  [mel]: https://en.wikipedia.org/wiki/Mel_scale
  """

  # Input validator from tensorflow/python/ops/signal/mel_ops.py#L71
  if num_mel_bins <= 0:
    raise ValueError('num_mel_bins must be positive. Got: %s' % num_mel_bins)
  if lower_edge_hertz < 0.0:
    raise ValueError('lower_edge_hertz must be non-negative. Got: %s' %
                     lower_edge_hertz)
  if lower_edge_hertz >= upper_edge_hertz:
    raise ValueError('lower_edge_hertz %.1f >= upper_edge_hertz %.1f' %
                     (lower_edge_hertz, upper_edge_hertz))
  if sample_rate <= 0.0:
    raise ValueError('sample_rate must be positive. Got: %s' % sample_rate)
  if upper_edge_hertz > sample_rate / 2:
    raise ValueError('upper_edge_hertz must not be larger than the Nyquist '
                     'frequency (sample_rate / 2). Got %s for sample_rate: %s' %
                     (upper_edge_hertz, sample_rate))

  # HTK excludes the spectrogram DC bin.
  bands_to_zero = 1
  nyquist_hertz = sample_rate / 2.0
  linear_frequencies = np.linspace(
      0.0, nyquist_hertz, num_spectrogram_bins, dtype=dtype)[bands_to_zero:]
  spectrogram_bins_mel = _hertz_to_mel(linear_frequencies)[:, np.newaxis]

  # Compute num_mel_bins triples of (lower_edge, center, upper_edge). The
  # center of each band is the lower and upper edge of the adjacent bands.
  # Accordingly, we divide [lower_edge_hertz, upper_edge_hertz] into
  # num_mel_bins + 2 pieces.
  edges = np.linspace(
      _hertz_to_mel(lower_edge_hertz),
      _hertz_to_mel(upper_edge_hertz),
      num_mel_bins + 2,
      dtype=dtype)

  # Split the triples up and reshape them into [1, num_mel_bins] tensors.
  lower_edge_mel, center_mel, upper_edge_mel = (edges[:-2][np.newaxis, :],
                                                edges[1:-1][np.newaxis, :],
                                                edges[2:][np.newaxis, :])

  # Calculate lower and upper slopes for every spectrogram bin.
  # Line segments are linear in the mel domain, not Hertz.
  lower_slopes = (spectrogram_bins_mel - lower_edge_mel) / (
      center_mel - lower_edge_mel)
  upper_slopes = (upper_edge_mel - spectrogram_bins_mel) / (
      upper_edge_mel - center_mel)

  # Intersect the line segments with each other and zero.
  mel_weights_matrix = np.maximum(0.0, np.minimum(lower_slopes, upper_slopes))

  # Re-add the zeroed lower bins we sliced out above.
  return np.pad(mel_weights_matrix, [[bands_to_zero, 0], [0, 0]])


def _hanning_nonzero(win_support, frame_size, dtype):
  """Computes modified version of Hanning window.

  This version of Hanning window doesn't have zero on the both tail, and
  therefore the `frame_size` parameter can be more accurate.

  Args:
    win_support: Number of samples for non-zero support in the window
    frame_size: Total size of the window (frame_size >= win_support)
    dtype: numpy data type of output and computation

  Returns:
    Numpy array of size frame_size with the window to apply.
  """
  assert frame_size >= win_support
  arg = np.pi * 2.0 / (win_support)
  hann = 0.5 - (0.5 * np.cos(arg * (np.arange(win_support, dtype=dtype) + 0.5)))
  zero_size = frame_size - win_support
  return np.pad(hann, [(0, zero_size)])


def _next_pow_of_two(x: Union[int, float]) -> int:
  return int(2**np.ceil(np.log2(x)))


def _milliseconds_to_samples(length_ms: float, sample_rate: float) -> int:
  return int(round(sample_rate * length_ms / 1000.0))


class SpectrogramFrontend(base_layer.BaseLayer):
  """Perform STFT to convert 1D signals to spectrograms.

  The `fprop` input to this layer is assumed to have a shape
  `(batch_size, timesteps, channel)`, and an optional padding array should have
  shape `(batchsize, timesteps)`.  The output frame-level padding indicators
  are calculated when the sample-level paddings are given as an argument.
  """

  class HParams(BaseHParams):
    """Associated hyperparams for this layer class.

    Attributes:
      sample_rate: Sample rate in Hz.
      frame_size_ms: Amount of data grabbed for each frame during analysis.
      frame_step_ms: Number of ms to jump between frames.
      compute_energy: If True, returns squares of magnitudes of each frequency
        bin.
      fft_size: If set, override FFT size.
      window_fn: Window function to apply (valid values are "HANNING", and
        None).
      output_log: If True, returns logarithmic magnitude (or energy).
      output_log_floor: Minimum output of filterbank output prior to taking
        logarithm. Used only when p.output_log == True.
      input_scale_factor: Input scale-factor applied first before any
        processing.
      pad_end: Whether to pad the end of `signals` with zeros when the provided
        frame length and step produces a frame that lies partially past its end.
      preemph: The first-order filter coefficient used for preemphasis. When it
        is 0.0, preemphasis is turned off.
      preemph_htk_flavor: preemphasis is applied as in HTK.
      noise_scale: The amount of noise to add.
      framing_tpl: The Framing class or subclass to use. Parameter values will
        be overridden.
    """
    sample_rate: float = 16000.0
    frame_size_ms: float = 25.0
    frame_step_ms: float = 10.0
    compute_energy: bool = False
    fft_size: Optional[int] = None
    window_fn: str = 'HANNING'
    output_log: bool = False
    output_log_floor: float = 1.0
    input_scale_factor: float = 1.0
    pad_end: bool = False
    preemph: float = 0.97
    preemph_htk_flavor: bool = False
    noise_scale: float = 8.0
    framing_tpl: BaseHParams = sub_config_field(Framing.HParams)

  @classmethod
  def get_frame_size(cls, p: HParams) -> int:
    return _milliseconds_to_samples(p.frame_size_ms, p.sample_rate)

  @property
  def frame_size(self) -> int:
    return self.get_frame_size(self.hparams)

  @classmethod
  def get_frame_step(cls, p: HParams) -> int:
    return _milliseconds_to_samples(p.frame_step_ms, p.sample_rate)

  @property
  def frame_step(self) -> int:
    return self.get_frame_step(self.hparams)

  @classmethod
  def configure_framing(cls, p: HParams) -> BaseHParams:
    return p.framing_tpl.clone().set(
        name='framing',
        frame_size=cls.get_frame_size(p) + 1,  # +1 for the preemph
        frame_step=cls.get_frame_step(p),
        pad_end=p.pad_end)

  def setup(self) -> None:
    p = self.hparams
    self.create_child('framing', self.configure_framing(p))

    # TF-version has maximum of 512, but it's not always necessary
    if p.fft_size is None:
      self.fft_size = _next_pow_of_two(self.frame_size)
    else:
      self.fft_size = p.fft_size

    self._create_window_function()

  def _create_window_function(self):
    p = self.hparams
    if p.window_fn is None:
      self._window_fn = None
    elif p.window_fn.upper() == 'HANNING':

      def _hanning_window(frame_size, dtype):
        # Preparing 1-point longer window to follow TF's definition
        if frame_size % 2 == 0:
          # simulate periodic=True in tf.signal.hann_window
          return jnp.hanning(frame_size + 1).astype(dtype)[:-1]
        else:
          return jnp.hanning(frame_size).astype(dtype)

      self._window_fn = _hanning_window
    elif p.window_fn.upper() == 'HANNING_NONZERO':
      def f(frame_size, dtype):
        return _hanning_nonzero(frame_size, frame_size, dtype)

      self._window_fn = f
    else:
      raise ValueError('Illegal value %r for window_fn param' % p.window_fn)

  def _apply_preemphasis(self, framed_signal: JTensor) -> JTensor:
    p = self.hparams
    if p.preemph_htk_flavor:
      return jnp.concatenate([
          framed_signal[:, :, :1, :] * (1. - p.preemph),
          (framed_signal[:, :, 1:-1, :] -
           p.preemph * framed_signal[:, :, 0:-2, :])
      ],
                             axis=2)
    else:
      return (framed_signal[:, :, 1:, :] -
              p.preemph * framed_signal[:, :, 0:-1, :])

  def __call__(
      self,
      inputs: JTensor,
      input_paddings: Optional[JTensor] = None
  ) -> Tuple[JTensor, Optional[JTensor]]:
    """Convert 1D-inputs and optional paddings to spectrogram.

    Args:
      inputs: An array with the shape `(batchsize, timesteps, channels)`.
      input_paddings: Optional padding indicator.  This layer handles paddings
        as zeroes in the input.

    Returns:
      `(spectrogram, output_paddings)` where `spectrogram` is a tensor with
      shape `(batch_size, frames, fft_size // 2 + 1, channels)` where frames
      is the number of frames extracted from the signal with `timesteps`-length.
      `output_paddings[b, n]` is the padding indicator for (b, n)-th frame that
      is 1.0 if all the samples corresponding to this frame are paddings, and
      0.0 otherwise.
    """
    inputs = inputs.astype(np.float32)
    p = self.hparams

    # Expand to have a channel axis
    if inputs.ndim == 2:
      inputs = jnp.expand_dims(inputs, -1)

    if input_paddings is not None:
      inputs = inputs * jnp.expand_dims(1.0 - input_paddings, -1)

    pcm_audio_chunk = inputs * p.input_scale_factor

    # framed_signal.shape = [batch, frames, frame_size + 1, channels]
    framed_signal, output_paddings = self.framing(pcm_audio_chunk,
                                                  input_paddings)

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

    return outputs, output_paddings


class MelFilterbankFrontend(base_layer.BaseLayer):
  """Compute Log-mel-filterbank outputs from 1D signal.

  This layer partially achieve parameter-level compatibility with its
  counterpart in TF (`lingvo.tasks.asr.frontend.MelAsrFrontend`) except the
  parameter related to post-filterbank processing.
  """

  class HParams(BaseHParams):
    """Associated hyperparams for this layer class.

    Attributes:
      sample_rate: Sample rate in Hz.
      frame_size_ms: Amount of data grabbed for each frame during analysis.
      frame_step_ms: Number of ms to jump between frames.
      num_bins: Number of bins in the mel-spectrogram output.
      lower_edge_hertz: The lowest frequency of the mel-spectrogram analysis.
      upper_edge_hertz: The highest frequency of the mel-spectrogram analysis.
      preemph: The first-order filter coefficient used for preemphasis. When it
        is 0.0, preemphasis is turned off.
      preemph_htk_flavor: preemphasis is applied as in HTK.
      noise_scale: The amount of noise (in 16-bit LSB units) to add.
      window_fn: Window function to apply (valid values are "HANNING", and
        None).
      pad_end: Whether to pad the end of `signals` with zeros when the provided
        frame length and step produces a frame that lies partially past its end.
      fft_overdrive: Whether to use twice the minimum fft resolution.
      output_floor: Minimum output of filterbank output prior to taking
        logarithm.
      compute_energy: Whether to compute filterbank output on the energy of
        spectrum rather than just the magnitude.
      use_divide_stream: Whether use a divide stream to the input signal.
      feature_dtype: Output feature type.
      per_bin_mean: Per-bin (num_bins) means for normalizing the spectrograms.
        Defaults to zeros.
      per_bin_stddev: Per-bin (num_bins) standard deviations. Defaults to ones.
      stft_tpl: The SpectrogramFrontend class or subclass to use. Parameter
        values will be overridden.
    """
    sample_rate: float = 16000.0
    frame_size_ms: float = 25.0
    frame_step_ms: float = 10.0
    num_bins: int = 80
    lower_edge_hertz: float = 125.0
    upper_edge_hertz: float = 7600.0
    preemph: float = 0.97
    preemph_htk_flavor: bool = False
    noise_scale: float = 8.0
    window_fn: str = 'HANNING'
    pad_end: bool = False
    fft_overdrive: bool = True
    output_floor: float = 1.0
    compute_energy: bool = False
    use_divide_stream: bool = False
    feature_dtype: jnp.dtype = jnp.float32
    per_bin_mean: Optional[float] = None
    per_bin_stddev: Optional[float] = None
    stft_tpl: BaseHParams = sub_config_field(SpectrogramFrontend.HParams)

  @classmethod
  def configure_stft(cls, p: HParams) -> BaseHParams:
    stft_p = p.stft_tpl.clone().set(
        name='spectrogram',
        sample_rate=p.sample_rate,
        frame_size_ms=p.frame_size_ms,
        frame_step_ms=p.frame_step_ms,
        compute_energy=p.compute_energy,
        pad_end=p.pad_end,
        preemph=p.preemph,
        preemph_htk_flavor=p.preemph_htk_flavor,
        noise_scale=p.noise_scale,
        window_fn=p.window_fn,
        input_scale_factor=2**-15 if p.use_divide_stream else 1.0,
        output_log=False)

    if p.fft_overdrive:
      fft_input_size = int(round(p.sample_rate * p.frame_size_ms / 1000.0))
      stft_p.fft_size = _next_pow_of_two(fft_input_size) * 2
    return stft_p

  def setup(self) -> None:
    p = self.hparams
    self.create_child('stft', self.configure_stft(p))

    # Mean/stddev.
    per_bin_mean = p.per_bin_mean
    if per_bin_mean is None:
      per_bin_mean = [0.0] * p.num_bins
    per_bin_stddev = p.per_bin_stddev
    if per_bin_stddev is None:
      per_bin_stddev = [1.0] * p.num_bins
    if len(per_bin_mean) != p.num_bins:
      raise ValueError('Size of per_bin_mean does not match with p.num_bins '
                       f'[{len(per_bin_mean)} vs {p.num_bins}]')
    if len(per_bin_stddev) != p.num_bins:
      raise ValueError('Size of per_bin_stddev does not match with p.num_bins '
                       f'[{len(per_bin_stddev)} vs {p.num_bins}]')
    self._normalizer_mean = jnp.array(per_bin_mean)[np.newaxis, np.newaxis, :,
                                                    np.newaxis]
    self._normalizer_stddev = jnp.array(per_bin_stddev)[np.newaxis,
                                                        np.newaxis, :,
                                                        np.newaxis]

  def __call__(
      self,
      inputs: JTensor,
      input_paddings: Optional[JTensor] = None
  ) -> Tuple[JTensor, Optional[JTensor]]:
    """Computed log-mel-filterbank features.

    Args:
      inputs: An array that will be processed with `stft` sublayer.
      input_paddings: A padding indicator that will be processed by `stft`.

    Returns:
      `(features, feature_paddings)`. `features` is an array with shape
      `(batch_size, num_frames, p.num_bins, num_channels)` where `num_frames`
      is the number of frames obtained from `stft` sublayer.
      `feature_paddings` is None if `input_paddings` is None, otherwise it
      contains frame-level padding indicators obtained from `stft`.
    """
    p = self.hparams

    spect, spect_paddings = self.stft(inputs, input_paddings)
    mel_weights = linear_to_mel_weight_matrix(
        num_mel_bins=self.hparams.num_bins,
        num_spectrogram_bins=spect.shape[2],
        sample_rate=self.hparams.sample_rate,
        lower_edge_hertz=self.hparams.lower_edge_hertz,
        upper_edge_hertz=self.hparams.upper_edge_hertz)

    mel_spectrogram = jnp.einsum('fn,btfc->btnc', mel_weights, spect)
    logmel_spectrogram = jnp.log(jnp.maximum(mel_spectrogram, p.output_floor))

    normalized_logmel_spectrogram = (
        (logmel_spectrogram - self._normalizer_mean) / self._normalizer_stddev)

    return normalized_logmel_spectrogram, spect_paddings


class FeatureStackingFrontend(base_layer.BaseLayer):
  """A wrapper for another frontend layer for applying feature stacking."""

  class HParams(BaseHParams):
    """Associated hyperparams for this layer class.

    Attributes:
      frontend: Base frontend layer to be augmented.
      left_context: Number of time steps to stack on the left to the central
        step.
      right_context: Number of time steps to stack on the right to the central
        step.
      stride: The stride for emitting the stacked output.
      pad_with_left_frame: Whether to use the left frame for padding instead of
        0s.
      pad_with_right_frame: Whether to use the right frame for padding instead
        of 0s.
      padding_reduce_option: reduce_max or reduce_min. How to reduce stacked
        padding from [b, t / stride, stride] to [b, t / stride, 1].
    """
    frontend: Optional[BaseHParams] = None
    left_context: int = 0
    right_context: int = 0
    stride: int = 0
    pad_with_left_frame: bool = False
    pad_with_right_frame: bool = False
    padding_reduce_option: str = 'reduce_min'

  def setup(self):
    p = self.hparams
    if p.stride < 1:
      raise ValueError('`p.stride` must be 1 or greater.')

    if p.frontend is None:
      raise ValueError('`p.frontend` must be set.')
    self.create_child('frontend', p.frontend)
    self.create_child(
        'stacker',
        StackingOverTime.HParams(
            left_context=p.left_context,
            right_context=p.right_context,
            stride=p.stride,
            pad_with_left_frame=p.pad_with_left_frame,
            pad_with_right_frame=p.pad_with_right_frame,
            padding_reduce_option=p.padding_reduce_option))

  def __call__(
      self,
      inputs: JTensor,
      input_paddings: Optional[JTensor] = None
  ) -> Tuple[JTensor, Optional[JTensor]]:
    p = self.hparams
    features, feature_paddings = self.frontend(inputs, input_paddings)
    if features.ndim != 4:
      raise ValueError(
          'Output of inner layer must have [batch, time, freq, ch]-shape.')
    batch_size, time_steps, freq_bins, channels = features.shape
    flat_features = features.reshape(
        (batch_size, time_steps, freq_bins * channels))
    flat_features, feature_paddings = self.stacker(
        flat_features, feature_paddings[..., np.newaxis])
    feature_paddings = feature_paddings[..., 0]

    batch_size, time_steps, unused_dims = flat_features.shape
    window_size = p.left_context + p.right_context + 1
    features = flat_features.reshape(
        (batch_size, time_steps, freq_bins * window_size, channels))
    return features, feature_paddings
