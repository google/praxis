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

"""HParams for the decoder."""

import copy
import dataclasses
from typing import Sequence, TypeVar

from praxis import base_hyperparams
from praxis import decoder_utils
from praxis import pax_fiddle
from praxis import pytypes
from praxis import sample_decode
JTensor = pytypes.JTensor


_TDecoderHParams = TypeVar('_TDecoderHParams', bound='DecoderHParams')


@dataclasses.dataclass
class DecoderHParams:
  """Decoder parameters to tune decoding.

  Attributes:
    seqlen: Maximum output sequence length.
    min_prefix_len: Minimum number of tokens picked to be used as decoding
      prefix.
    eos_id: The id of EOS token indicating the termination decoding. Could be
      either a sequence or an integer.
    min_decode_steps: If not 0, the minimum decode steps for each example.
      eos_id does not get generated until these many steps have been decoded.
    max_decode_steps: If not None, the maximum decode steps for each example. If
      None, this is set to `seqlen`, which contains prefix. If it is a list,
      decoding state will be padded at each given steps.
    fprop_for_prefix: Whether or not uses fprop instead of extend_step for the
      prefix.
    lazy_prefix_broadcast: Whether to enable the lazy-prefix-broadcast
      optimization for multi-sample decoding with shared prefixes. Requires
      fprop_for_prefix. This requires an implementation of the
      lazy_broadcast_prefix() method in the attention layer, which is
      DotProductAttentionWithLPB.
    decode_loop_mesh_axes_transpose: Optional mesh transpose for decoding loop.
    process_result_fn: Optional function that further processes the results,
      such as performing suffix scoring.
    emb_lookup_style: lookup style for the softmax embedding layer.
    use_extra_input_kwargs: if True, pass through any extra_input_kwargs from
      the input batch to the decode data.
      num_cache_slots: if num_cache_slots > 0, continuous batching will be
        enabled with max batch_size = num_cache_slots
  """

  seqlen: int = 0
  min_prefix_len: int = 5
  eos_id: int | Sequence[int] = 2
  min_decode_steps: int = 0
  max_decode_steps: int | Sequence[int] | None = None
  fprop_for_prefix: bool = False
  lazy_prefix_broadcast: bool = False
  decode_loop_mesh_axes_transpose: dict[str, str] | None = None
  process_result_fn: decoder_utils.ProcessResultFn | None = None
  emb_lookup_style: str = 'matmul'
  use_extra_input_kwargs: bool = False
  num_cache_slots: int = 0

  def clone(self: _TDecoderHParams) -> _TDecoderHParams:
    return copy.deepcopy(self)


@dataclasses.dataclass
class GreedyDecoderHParams(DecoderHParams):
  """HParams for greedy decode."""

@dataclasses.dataclass
class BeamSearchHParams(DecoderHParams):
  """HParams for beam search.

  Attributes:
    beam_size: Beam size for decoding.
    tokens_per_beam: Number of tokens to explore per beam, defaulting to
      beam_size if None.
    length_norm_alpha: Length norm alpha for beam search.
    early_exit: A bool, whether or not to allow early exit.
  """
  beam_size: int = 1
  tokens_per_beam: int | None = None
  length_norm_alpha: float = 0.8
  early_exit: bool = False
  use_matmul_beam_shuffle: bool = False


@dataclasses.dataclass
class FlatBeamSearchHParams(BeamSearchHParams):
  """HParams for flat beam search."""


@dataclasses.dataclass
class SampleDecoderHParams(DecoderHParams):
  """HParams for sample decode.

  Attributes:
    num_samples: Beam size limit in number of hyps
    temperature: Temperature of sampling decoding.
    k: if nonzero, use top-k sampling, only selecting amongthe most likely k
      tokens at each step.
    top_k_recall_target: if less than 1.0, use TPU optimized approx_top_k with
      specified recall target for the top_k sampling. See
      https://arxiv.org/abs/2206.14286 for more details.
    use_top_k_for_logprobs: computes the log probability from the top k logits
      instead of all logits.
    p: if not None, use the smallest number of logits whose cumulative sum of
      probs adds up to (at least) p. Notice that it should not be used with k at
      the same time.
    next_token_sampler_tpl: HParams for the layer used to sample next token ids
      given the logits output.
    sample_constraint: HParams for the layer used to terminate samples early if
      they don't conform to specific constraints.
    global_normalize: Normalize the logits over top-k logits or globally in the
      whole vocabulary. It is used if k is nonzero and p is also not None.
    cf_guidance_scale: If not None, apply classifier-free guidance.
    controlled_decoding: Parameters for controlled decoding if used.
    sort_samples:  Whether to sort the samples by logprobs.
    override_next_token_sampler_params: Whether to override, the next token
      sampler params from the decoder ones. Ideally, this should not be
      performed, but it is currently enabled for back-compatibility reasons.
    optimize_eos: Whether to optimize the eos prediction by recording eos
      probability at each step.
    vanilla_sample_decode: Switch to vanilla sample decode.
  """
  num_samples: int = 1
  # TODO(wangtao): supports per-example temperature.
  temperature: float = 1.0
  k: int = 40
  top_k_recall_target: float = 1.0
  use_top_k_for_logprobs: bool = False
  p: float | JTensor | None = None
  next_token_sampler_tpl: pax_fiddle.Config[
      sample_decode.BaseNextTokenSampler] = (
          pax_fiddle.template_field(sample_decode.DefaultNextTokenSampler))
  sample_constraint: pax_fiddle.Config[
      sample_decode.BaseSampleTerminationConstraint
  ] | None = None
  global_normalize: bool = False
  cf_guidance_scale: list[float] | float | None = None
  controlled_decoding: decoder_utils.ControlledDecodingHParams | None = None
  sort_samples: bool | None = True
  override_next_token_sampler_params: bool = True
  optimize_eos: bool = False
  vanilla_sample_decode: bool = False
