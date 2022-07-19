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

"""HParams for the decoder."""
from typing import List, Optional, Union

from praxis import base_hyperparams
from praxis import pytypes
BaseHyperParams = base_hyperparams.BaseHyperParams
JTensor = pytypes.JTensor


class DecoderHParams(BaseHyperParams):
  """HParams for decoder.

  Attributes:
    seqlen: Maximum output sequence length.
    min_prefix_len: Minimum number of tokens picked to be used as decoding
      prefix.
    eos_id: The id of EOS token indicating the termination of greedy search.
    max_decode_steps: If not None, the max decode steps for each example. If
      None, this is set to `seqlen`, which contains prefix.
    fprop_for_prefix: Whether or not uses fprop instead of extend_step for
      the prefix.
    lazy_prefix_broadcast: Whether to enable the lazy-prefix-broadcast
      optimization for multi-sample decoding with shared prefixes. Requires
      fprop_for_prefix. This requires an implementation of the
      lazy_broadcast_prefix() method in the attention layer, which is
      DotProductAttentionWithLPB.
  """
  # TODO(b/229679837): remove seqlen and uses max_decode_steps.
  seqlen: int = 0
  min_prefix_len: int = 5
  eos_id: int = 2
  max_decode_steps: Optional[int] = None
  fprop_for_prefix: bool = False
  lazy_prefix_broadcast: bool = False


class GreedyDecoderHParams(DecoderHParams):
  """HParams for greedy decode."""


class BeamSearchHParams(DecoderHParams):
  """HParams for beam search.

  Attributes:
    beam_size: Beam size for decoding.
    length_norm_alpha: Length norm alpha for beam search.
    parse_tokens: Token ids used for parsing out answers from model outputs.
  """
  beam_size: int = 1
  length_norm_alpha: float = 0.8
  parse_tokens: Optional[List[int]] = None


class FlatBeamSearchHParams(BeamSearchHParams):
  """HParams for flat beam search."""


class SampleDecoderHParams(DecoderHParams):
  """HParams for sample decode.

  Attributes:
    num_samples: Beam size limit in number of hyps
    temperature: Temperature of sampling decoding.
    k: if nonzero, use top-k sampling, only selecting amongthe most likely k
      tokens at each step.
    p: if not None, use the smallest number of logits whose cumulative sum of
      probs adds up to (at least) p. Notice that it should not be used with k
      at the same time.
  """
  num_samples: int = 1
  # TODO(wangtao): supports per-example temperature.
  temperature: float = 1.0
  k: int = 40
  p: Optional[Union[float, JTensor]] = None
