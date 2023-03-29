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

from typing import Optional

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
import numpy as np
from praxis import base_hyperparams
from praxis import base_layer
from praxis import pax_fiddle
from praxis import py_utils
from praxis import test_utils
from praxis.layers import activations

from praxis.layers import chain


BaseLayer = base_layer.BaseLayer
Config = pax_fiddle.Config
JTensor = py_utils.JTensor


def _residual():
  """A dummy residual stack with padding."""
  return chain.chain(
      chain.add_residual(Config(activations.Identity), chain.apply_padding())
  )


class _Scale(BaseLayer):
  """Simple scaling layer, useful for testing."""

  factor: float = 1.0

  def __call__(self, inputs: JTensor, paddings: JTensor) -> JTensor:
    scaled = inputs * self.factor
    return scaled, paddings


def _scale(factor: float, **kwargs) -> Config[_Scale]:
  """`Config(Scale)`; scales the input by a factor (mostly testing)."""
  return Config(
      _Scale,
      factor=factor,
      **chain.kwargs_with_name('scale', **kwargs),
  )


class ChainTest(test_utils.TestCase):

  def _run(self, layer_tpl, *args, **kwargs):
    layer_tpl.set(name='nnnn')
    layer = base_hyperparams.instantiate(layer_tpl)
    with base_layer.JaxContext.new_context():
      layer_vars = layer.init(jax.random.PRNGKey(111), *args, **kwargs)
      return layer.apply(layer_vars, *args, **kwargs)

  def test_chain_of_different_type(self):
    p = chain.chain(chain.full_like(2.0), chain.apply_padding(), _scale(3.0))
    outputs, paddings = self._run(
        p, jnp.ones((2, 2)), jnp.array([[0.0, 1], [0, 0]])
    )
    self.assertArraysEqual([[6.0, 0], [6, 6]], outputs)
    self.assertArraysEqual([[0.0, 1], [0, 0]], paddings)

  def test_chain_with_none_layer(self):
    p = chain.chain(
        chain.chain(),
        chain.full_like(2.0),
        chain.chain(None),
        chain.apply_padding(),
        None,
        _scale(3.0),
        chain.chain(),
    )
    outputs, paddings = self._run(
        p, jnp.ones((2, 2)), jnp.array([[0.0, 1], [0, 0]])
    )
    self.assertArraysEqual([[6.0, 0], [6, 6]], outputs)
    self.assertArraysEqual([[0.0, 1], [0, 0]], paddings)

  def test_chain_with_special_kwarg(self):
    class SpecialKWarg(BaseLayer):

      def __call__(self, inputs, input_batch):
        return input_batch.override

    p = chain.chain(
        chain.full_like(2.0),
        pax_fiddle.Config(SpecialKWarg),
        chain.apply_padding(),
    )
    input_batch = py_utils.NestedMap(override=jnp.ones((2, 2)) * 6)
    outputs, paddings = self._run(
        p,
        jnp.ones((2, 2)),
        jnp.array([[0.0, 1], [0, 0]]),
        input_batch=input_batch,
    )
    self.assertArraysEqual([[6.0, 0], [6, 6]], outputs)
    self.assertArraysEqual([[0.0, 1], [0, 0]], paddings)

  def test_chain_with_more_outputs_than_inputs(self):
    class ReturnOneMore(BaseLayer):

      def __call__(self, *args):
        return args[0], *args

    p = chain.chain(
        pax_fiddle.Config(ReturnOneMore), pax_fiddle.Config(ReturnOneMore)
    )
    ones = jnp.ones((2, 2))
    outputs = self._run(p, ones)
    self.assertArraysEqual([ones] * 3, outputs)

  def test_chain_with_two_chain(self):
    p = chain.chain(
        chain.chain(chain.full_like(2.0)),
        chain.chain(chain.apply_padding(), _scale(3.0)),
    )
    outputs, paddings = self._run(
        p, jnp.ones((2, 2)), jnp.array([[0.0, 1], [0, 0]])
    )
    self.assertArraysEqual([[6.0, 0], [6, 6]], outputs)
    self.assertArraysEqual([[0.0, 1], [0, 0]], paddings)

  def test_copy_n_times(self):
    p = chain.copy_n_times(3, _scale(2.0), chain.apply_padding())
    outputs = self._run(p, jnp.ones((2, 2)), jnp.array([[0.0, 1], [0, 0]]))
    pow_val = pow(2.0, 3)
    self.assertArraysEqual([[pow_val, 0], [pow_val, pow_val]], outputs[0])
    self.assertArraysEqual([[0.0, 1], [0, 0]], outputs[1])

  def test_repeat_n_times(self):
    p = chain.repeat(3, _scale(2.0), chain.apply_padding())
    outputs = self._run(p, jnp.ones((2, 2)), jnp.array([[0.0, 1], [0, 0]]))
    pow_val = pow(2.0, 3)
    self.assertArraysEqual([[pow_val, 0], [pow_val, pow_val]], outputs[0])
    self.assertArraysEqual([[0.0, 1], [0, 0]], outputs[1])

  def test_chain_with_wrong_args_raises_error(self):
    with self.assertRaises(ValueError):
      _ = self._run(
          chain.chain(chain.apply_padding()),
          jnp.ones((2, 2)),
      )

  def test_doc_example(self):
    """Example code used in the doc string."""
    n_times = 3
    input_dims, hidden_dims, output_dims = 2, 8, 3
    # A simple mlp stacking a few layers and using `copy_n_times()`.
    act = Config(activations.Swish)
    my_mlp = chain.chain(
        chain.feed_forward(input_dims, hidden_dims, act),
        chain.copy_n_times(
            n_times, chain.feed_forward(hidden_dims, hidden_dims, act)
        ),
        chain.feed_forward(hidden_dims, output_dims),
    )

    inputs, paddings = jnp.ones((2, 2)), jnp.array([[0.0, 1], [0, 0]])
    outputs, out_paddings = self._run(my_mlp, inputs, paddings)
    self.assertGreater(np.mean(np.abs(outputs)), 0)
    self.assertArraysEqual(paddings, out_paddings)

  @parameterized.parameters(
      (_residual(), (1 + 1)),
      (chain.copy_n_times(2, _residual(), _residual()), pow((1 + 1 + 2), 2)),
      (chain.chain(chain.feed_forward(2, 2))),
      (chain.chain(chain.log_args('hello'))),
  )
  def test_chain_extensions(
      self,
      layer_tpl: Config[BaseLayer],
      expected_factor: Optional[float] = None,
  ) -> None:
    inputs = jnp.ones((2, 2, 2))
    paddings = jnp.zeros((2, 2))
    outputs, out_paddings = self._run(layer_tpl, inputs, paddings)
    if expected_factor is None:
      self.assertLess(0, jnp.sum(jnp.abs(outputs)))
    else:
      self.assertArraysEqual(expected_factor * inputs, outputs)
    self.assertArraysEqual(paddings, out_paddings)

  def test_dict_to_args(self):
    p = chain.chain(chain.dict_to_args('inputs', 'paddings'))
    input_batch = {'inputs': jnp.ones((2, 2, 2)), 'paddings': jnp.zeros((2, 2))}
    outputs, out_paddings = self._run(p, input_batch)
    self.assertArraysEqual(input_batch['inputs'], outputs)
    self.assertArraysEqual(input_batch['paddings'], out_paddings)


if __name__ == '__main__':
  absltest.main()
