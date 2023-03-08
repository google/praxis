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

"""Unittest for Sequential."""

from absl.testing import absltest
import jax
from jax import numpy as jnp

from praxis import base_layer
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis import test_utils
from praxis.layers.sequential import Sequential

JTensor = pytypes.JTensor
NestedMap = py_utils.NestedMap
instantiate = base_layer.instantiate

PARAMS = base_layer.PARAMS
RANDOM = base_layer.RANDOM
AUX_LOSS = base_layer.AUX_LOSS
SUMMARIES = base_layer.SUMMARIES
NON_TRAINABLE = base_layer.NON_TRAINABLE
NON_PAX_VAR_COLLECTION = base_layer.NON_PAX_VAR_COLLECTION
DECODE_CACHE = base_layer.DECODE_CACHE
DEFAULT_INIT_MUTABLE_LIST = base_layer.DEFAULT_INIT_MUTABLE_LIST


class Feature(base_layer.BaseLayer):
  def __call__(self, features: JTensor, paddings: JTensor) -> NestedMap:
    return NestedMap(features=features, paddings=paddings)


class SpecAugment(base_layer.BaseLayer):
  def __call__(self, features: JTensor, paddings: JTensor) -> NestedMap:
    return NestedMap(features=features, paddings=paddings)


class Encoder(base_layer.BaseLayer):
  def __call__(self, features: JTensor, paddings: JTensor) -> NestedMap:
    return NestedMap(embeddings=features, paddings=paddings)


class DecodingPrep(base_layer.BaseLayer):
  def __call__(self, embeddings: JTensor, paddings: JTensor) -> NestedMap:
    return NestedMap(embeddings=embeddings, paddings=paddings)


class SequentialTest(test_utils.TestCase):

  def test_simple_sequence(self):
    specaug_p = pax_fiddle.Config(SpecAugment, name='specaugment')
    feature_p = pax_fiddle.Config(Feature, name='feature')
    encoder_p = pax_fiddle.Config(Encoder, name='encoder')
    decodingprep_p = pax_fiddle.Config(DecodingPrep, name='decodingprep')

    sequence_p = pax_fiddle.Config(
        Sequential, layers=[specaug_p, feature_p, encoder_p, decodingprep_p])
    sequence = instantiate(sequence_p)

    features = jax.random.normal(jax.random.PRNGKey(123), shape=[2, 1, 4])
    paddings = jnp.zeros(shape=[2, 1])

    k1 = jax.random.PRNGKey(123)
    k2 = jax.random.PRNGKey(456)
    with base_layer.JaxContext.new_context():
      initial_vars = sequence.init(
          rngs={RANDOM: k1, PARAMS: k2},
          mutable=DEFAULT_INIT_MUTABLE_LIST,
          features=features,
          paddings=paddings)

      outputs = sequence.apply(initial_vars, features, paddings)

    self.assertIn('embeddings', outputs)
    self.assertIn('paddings', outputs)
    self.assertAllClose(outputs.embeddings, features)


if __name__ == '__main__':
  absltest.main()

  