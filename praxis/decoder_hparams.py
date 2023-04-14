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
from typing import Dict, List, Optional, Sequence, Union

from praxis import base_hyperparams
from praxis import base_layer
from praxis import decoder_utils
from praxis import pax_fiddle
from praxis import pytypes
from praxis import sample_decode
BaseHyperParams = base_hyperparams.BaseHyperParams
JTensor = pytypes.JTensor


class DecoderHParams(BaseHyperParams):
  """HParams for decoder.

  Attributes:
    seqlen: Maximum output sequence length.
    min_prefix_len: Minimum number of tokens picked to be used as decoding
      prefix.
    eos_id: The id of EOS token indicating the termination decoding. Could be
      either a sequence or an integer.
    max_decode_steps: If not None, the max decode steps for each example. If
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
    emb_lookup_style: lookup style for the softmax embedding layer.
  """
  # TODO(b/229679837): remove seqlen and uses max_decode_steps.
  seqlen: int = 0
  min_prefix_len: int = 5
  eos_id: Union[int, Sequence[int]] = 2
  max_decode_steps: Optional[Union[int, Sequence[int]]] = None
  fprop_for_prefix: bool = False
  lazy_prefix_broadcast: bool = False
  decode_loop_mesh_axes_transpose: Optional[Dict[str, str]] = None
  emb_lookup_style: str = 'matmul'


class GreedyDecoderHParams(DecoderHParams):
  """HParams for greedy decode."""


class BeamSearchHParams(DecoderHParams):
  """HParams for beam search.

  Attributes:
    beam_size: Beam size for decoding.
    length_norm_alpha: Length norm alpha for beam search.
  """
  beam_size: int = 1
  length_norm_alpha: float = 0.8


class FlatBeamSearchHParams(BeamSearchHParams):
  """HParams for flat beam search."""


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
    global_normalize: Normalize the logits over top-k logits or globally in the
      whole vocabulary. It is used if k is nonzero and p is also not None.
    cf_guidance_scale: If not None, apply classifier-free guidance.
    sort_samples:  Whether to sort the samples by logprobs.
  """
  num_samples: int = 1
  # TODO(wangtao): supports per-example temperature.
  temperature: float = 1.0
  k: int = 40
  top_k_recall_target: float = 1.0
  use_top_k_for_logprobs: bool = False
  p: Optional[Union[float, JTensor]] = None
  next_token_sampler_tpl: pax_fiddle.Config[
      sample_decode.BaseNextTokenSampler] = (
          pax_fiddle.template_field(sample_decode.DefaultNextTokenSampler))
  global_normalize: bool = False
  cf_guidance_scale: Optional[Union[List[float], float]] = None
  controlled_decoding: Optional[decoder_utils.ControlledDecodingHParams] = None
  sort_samples: Optional[bool] = True
