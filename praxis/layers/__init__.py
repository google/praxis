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

"""Exposes the public layer functionalities."""

from praxis.layers.activations import BaseActivation
from praxis.layers.activations import CubedReLU
from praxis.layers.activations import GELU
from praxis.layers.activations import Identity
from praxis.layers.activations import LeakyReLU
from praxis.layers.activations import ReLU
from praxis.layers.activations import ReLU6
from praxis.layers.activations import Sigmoid
from praxis.layers.activations import SiLU
from praxis.layers.activations import SquaredReLU
from praxis.layers.activations import Swish
from praxis.layers.activations import Tanh
from praxis.layers.adapters import AdaptedTransformerFeedForward
from praxis.layers.adapters import MultitaskResidualAdapter
from praxis.layers.attentions import AttentionProjection
from praxis.layers.attentions import causal_mask
from praxis.layers.attentions import causal_segment_mask
from praxis.layers.attentions import CausalDepthwiseConv1D
from praxis.layers.attentions import convert_paddings_to_mask
from praxis.layers.attentions import DotProductAttention
from praxis.layers.attentions import DotProductAttentionXL
from praxis.layers.attentions import LocalSelfAttention
from praxis.layers.attentions import LocalSelfAttentionXL
from praxis.layers.attentions import PerDimScale
from praxis.layers.attentions import RelativeBias
from praxis.layers.attentions import segment_mask
from praxis.layers.augmentations import MaskedLmDataAugmenter
from praxis.layers.bregman import BregmanPCA
from praxis.layers.checkpoint_policy import AutodiffCheckpointType
from praxis.layers.conformers import Conformer
from praxis.layers.conformers import DotProductAttentionWithContext
from praxis.layers.conformers import DotProductAttentionWithContextXL
from praxis.layers.conformers import SelfAttentionWithNormAndResidual
from praxis.layers.convolutions import Conv2D
from praxis.layers.convolutions import ConvBNAct
from praxis.layers.convolutions import ConvBNActWithPadding
from praxis.layers.convolutions import DepthwiseConv1D
from praxis.layers.convolutions import LightConv1D
from praxis.layers.embedding_softmax import Embedding
from praxis.layers.embedding_softmax import FullSoftmax
from praxis.layers.embedding_softmax import GShardSharedEmbeddingSoftmax
from praxis.layers.embedding_softmax import PositionalEmbedding
from praxis.layers.embedding_softmax import SharedEmbeddingSoftmax
from praxis.layers.embedding_softmax import SigmoidCrossEntropy
from praxis.layers.embedding_softmax import TrainablePositionalEmbedding
from praxis.layers.frnn import FRnn
from praxis.layers.frnn import LstmFrnn
from praxis.layers.frnn import StackFrnn
from praxis.layers.linears import Bias
from praxis.layers.linears import FeedForward
from praxis.layers.linears import Linear
from praxis.layers.linears import project_last_dim
from praxis.layers.linears import StackingOverTime
from praxis.layers.losses import BiTemperedLoss
from praxis.layers.models import BertModel
from praxis.layers.models import ClassificationMLPModel
from praxis.layers.models import ClassificationModel
from praxis.layers.models import LanguageModel
from praxis.layers.models import SequenceModel
from praxis.layers.ngrammer import get_bigram_ids
from praxis.layers.ngrammer import Ngrammer
from praxis.layers.ngrammer import VectorQuantization
from praxis.layers.ngrammer import VQNgrammer
from praxis.layers.normalizations import BaseNormalization
from praxis.layers.normalizations import BatchNorm
from praxis.layers.normalizations import compute_moments
from praxis.layers.normalizations import GroupNorm
from praxis.layers.normalizations import LayerNorm
from praxis.layers.normalizations import RmsNorm
from praxis.layers.normalizations import RmsNormNoScale
from praxis.layers.pipeline import LayerwiseShardablePipelined
from praxis.layers.poolings import GlobalPooling
from praxis.layers.poolings import Pooling
from praxis.layers.quantizer import quantize_vector
from praxis.layers.quantizer import RandomVectorQuantizer
from praxis.layers.quantizer import VectorQuantizer
from praxis.layers.repeats import Repeat
from praxis.layers.resnets import ResNet
from praxis.layers.resnets import ResNetBlock
from praxis.layers.rnn_cell import CifgLstmCellSimple
from praxis.layers.rnn_cell import LstmCellSimple
from praxis.layers.spectrum_augmenter import SpectrumAugmenter
from praxis.layers.stochastics import Dropout
from praxis.layers.stochastics import StochasticResidual
from praxis.layers.transformer_models import LanguageModelType
from praxis.layers.transformer_models import TransformerEncoderDecoder
from praxis.layers.transformer_models import TransformerLm
from praxis.layers.transformers import compute_attention_masks_for_extend_step
from praxis.layers.transformers import compute_attention_masks_for_fprop
from praxis.layers.transformers import PipelinedTransformer
from praxis.layers.transformers import StackedTransformer
from praxis.layers.transformers import StackedTransformerRepeated
from praxis.layers.transformers import Transformer
from praxis.layers.transformers import TransformerFeedForward
from praxis.layers.transformers import TransformerFeedForwardMoe
from praxis.layers.vanillanets import VanillaBlock
from praxis.layers.vanillanets import VanillaNet
from praxis.layers.vits import VisionTransformer
from praxis.layers.vits import VitEntryLayers
from praxis.layers.vits import VitExitLayers
