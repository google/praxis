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

from absl.testing import absltest
from absl.testing import parameterized
import jax
from praxis import base_layer
from praxis import pax_fiddle
from praxis import test_utils
from praxis.layers.video import quantizer


class QuantizerTest(test_utils.TestCase):

  @parameterized.parameters(
      (8, True, 0, 0, 0),
      (8, False, -0.331989, 0.10211816, -0.43410715),
      (16, True, 0, 0, 0),
      (16, False, -0.3675179, 0.1005669, -0.46808478),
  )
  def test_encode_decode_id_and_loss(
      self, embedding_dim, do_eval, quantzer_loss, e_latent_loss, entropy_loss
  ):
    prng_key = jax.random.PRNGKey(seed=123)
    x = jax.random.normal(prng_key, (1, 3, 6, 6, embedding_dim))
    config = pax_fiddle.Config(
        quantizer.LookupFreeQuantizer,
        name="quantizer",
        embedding_dim=embedding_dim,
    )
    context_p = base_layer.JaxContext.HParams(do_eval=do_eval)
    with base_layer.JaxContext.new_context(hparams=context_p):
      lookup_free_quantizer = base_layer.instantiate(config)
      pax_vars = lookup_free_quantizer.init(prng_key, x)
      self.assertEmpty(pax_vars)
      value, result_dict = lookup_free_quantizer.apply(pax_vars, x)
      self.assertEqual(value.shape, x.shape)
      self.assertAllClose(x, result_dict.raw)

      if not do_eval:
        ids = result_dict.encoding_indices
        decoded = lookup_free_quantizer.decode_ids(ids)
        self.assertAllClose(decoded, result_dict.encodings)
        self.assertEqual(result_dict.quantizer_loss.shape, ())
        self.assertEqual(result_dict.e_latent_loss.shape, ())
        self.assertEqual(result_dict.entropy_loss.shape, ())
        self.assertAllClose(result_dict.quantizer_loss, quantzer_loss)
        self.assertAllClose(result_dict.e_latent_loss, e_latent_loss)
        self.assertAllClose(result_dict.entropy_loss, entropy_loss)


if __name__ == "__main__":
  absltest.main()
