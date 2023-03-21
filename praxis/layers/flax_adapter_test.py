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

"""Tests for Flax adapter."""

from typing import Any, Optional, Tuple

from absl.testing import absltest
import fiddle as fdl
import flax.linen as flax_nn
from flax.linen import partitioning as flax_partitioning
import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import pax_fiddle
from praxis import pytypes
from praxis import test_utils
from praxis.layers import flax_adapter
from praxis.layers import normalizations


instantiate = base_layer.instantiate
JTensor = pytypes.JTensor
WeightHParamsCollection = base_layer.WeightHParamsCollection


class CNN(flax_nn.Module):
  """A simple CNN model."""

  @flax_nn.compact
  def __call__(self,
               x: JTensor,
               *,
               use_running_average: bool = True) -> JTensor:
    x = flax_nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = flax_nn.BatchNorm(use_running_average=use_running_average)(x)
    x = flax_nn.relu(x)
    x = flax_nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = flax_nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = flax_nn.BatchNorm(use_running_average=use_running_average)(x)
    x = flax_nn.relu(x)
    x = flax_nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = flax_nn.Dense(features=256)(x)
    x = flax_nn.BatchNorm(use_running_average=use_running_average)(x)
    x = flax_nn.relu(x)
    x = flax_nn.Dense(features=10)(x)
    x = flax_nn.log_softmax(x)
    return x


class MixLayer(base_layer.BaseLayer):
  """A layer that mixes Pax native layer with nn.Module wrapper layer.

  Attributes:
    use_running_average: bool as if BN layers are using running average or not.
  """
  use_running_average: bool = False

  def setup(self) -> None:
    super().setup()

    cnn_p = pax_fiddle.Config(
        flax_adapter.FlaxModuleAdapter, module_factory_method=CNN
    )

    self.create_child('cnn_p1', cnn_p.clone())
    self.create_child('cnn_p2', cnn_p.clone())
    bn_p = pax_fiddle.Config(normalizations.BatchNorm, dim=10)
    self.create_child('bn', bn_p)

  def __call__(self, x: JTensor) -> Tuple[JTensor, JTensor, JTensor]:
    # Call cnn_p1 twice to verify this doesn't break initialization.
    out1 = self.cnn_p1(
        x, use_running_average=self.use_running_average
    ) + self.cnn_p1(x / 2.0, use_running_average=self.use_running_average)
    out2 = self.cnn_p2(x, use_running_average=self.use_running_average)
    out = self.bn(out1 + out2)
    return out1, out2, out


class DirectMixLayer(base_layer.BaseLayer):
  """Direct instantiation version of the mix layer above."""

  cnn_p1: Optional[flax_nn.Module] = None
  cnn_p2: Optional[flax_nn.Module] = None
  bn: Optional[flax_nn.Module] = None
  use_running_average: bool = False

  def __call__(self, x: JTensor) -> Tuple[JTensor, JTensor, JTensor]:
    # Call cnn_p1 twice to verify this doesn't break initialization.
    # pylint: disable=not-callable
    out1 = self.cnn_p1(
        x, use_running_average=self.use_running_average
    ) + self.cnn_p1(x / 2.0, use_running_average=self.use_running_average)
    out2 = self.cnn_p2(x, use_running_average=self.use_running_average)
    out = self.bn(out1 + out2)
    # pylint: enable=not-callable
    return out1, out2, out


@pax_fiddle.auto_config
def direct_mix_layer_fixture():
  return DirectMixLayer(
      cnn_p1=flax_adapter.DirectFlaxModuleAdapter(cld=CNN()),
      cnn_p2=flax_adapter.DirectFlaxModuleAdapter(cld=CNN()),
      bn=normalizations.BatchNorm(dim=10),
  )


class FlaxWrapperTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  def test_mix_layer(self):
    test_layer_p = pax_fiddle.Config(MixLayer, name='test_layer')
    test_layer = instantiate(test_layer_p)

    prng_key = jax.random.PRNGKey(seed=123)

    input_x = jnp.zeros((256, 256, 3))
    with base_layer.JaxContext.new_context():
      init_var_meta = test_layer.abstract_init_with_metadata(input_x)

      def assert_learnable(x):
        assert not x.collections

      jax.tree_map(assert_learnable, init_var_meta['params'])

      def assert_non_learnable(x):
        assert WeightHParamsCollection.NON_TRAINABLE in x.collections
        assert WeightHParamsCollection.REQUIRES_MEAN_SYNC in x.collections

      jax.tree_map(assert_non_learnable, init_var_meta['batch_stats'])
      jax.tree_map(assert_non_learnable, init_var_meta['non_trainable'])
      init_vars = test_layer.init(prng_key, input_x)
      _ = test_layer.apply(init_vars, input_x, mutable=True)

  def test_literal_init_args(self):
    """Tests construction with literal (non-callable) init args."""
    test_layer_p = flax_adapter.FlaxModuleAdapter.config(
        name='test_layer', module_factory_method=CNN)
    test_layer = instantiate(test_layer_p)

    input_x = jnp.zeros((256, 256, 3))
    with base_layer.JaxContext.new_context():
      init_vars = test_layer.init(jax.random.PRNGKey(seed=123), input_x)
      _ = test_layer.apply(init_vars, input_x)

  def test_direct_mix_layer_is_same(self):
    old_layer_config = pax_fiddle.Config(MixLayer, name='test_layer')
    old_layer = instantiate(old_layer_config)

    new_layer_config = direct_mix_layer_fixture.as_buildable()
    new_layer = pax_fiddle.build(new_layer_config)

    input_x = jnp.zeros((256, 256, 3))
    with base_layer.JaxContext.new_context():
      init_vars = old_layer.init(jax.random.PRNGKey(seed=123), input_x)
      old_outputs, _ = old_layer.apply(init_vars, input_x, mutable=True)
      new_outputs, _ = new_layer.apply(init_vars, input_x, mutable=True)

    self.assertLen(old_outputs, len(new_outputs))
    for old, new in zip(old_outputs, new_outputs):
      self.assertAllClose(old, new)

  def test_wrap_sharding_spec(self):

    class SimpleLinear(flax_nn.Module):

      def setup(self):
        self.w = flax_partitioning.param_with_axes(
            'w',
            jax.nn.initializers.ones, (3, 4),
            jnp.float32,
            axes=('input', 'output'))
        # Do not annotate with param_axes on purpose since users may mix
        # Flax modules which do not have sharding annotations with Flaxformer
        # models which do.
        self.b = self.param('b', jax.nn.initializers.zeros, (4,), jnp.float32)

      def __call__(self, x):
        return jnp.dot(x, self.w) + self.b

    logical_axes_rules = [
        ('input', 'mdl'),
        ('output', 'data'),
    ]
    layer_p = flax_adapter.FlaxModuleAdapter.config(
        name='flax_adapter',
        module_factory_method=SimpleLinear,
        logical_axes_rules=logical_axes_rules,
        ici_mesh_shape=[4, 2],
        mesh_axis_names=['data', 'model'],
    )
    layer = instantiate(layer_p)
    variables = layer.abstract_init_with_metadata(jnp.ones((2, 3)))
    w_sharding = variables['params']['cld']['w'].tensor_split_dims_mapping
    self.assertEqual(w_sharding, ['mdl', 'data'])
    b_sharding = variables['params']['cld']['b'].tensor_split_dims_mapping
    self.assertEqual(b_sharding, [None])

  def test_adapter_with_fdl_partial(self):
    test_layer_p = flax_adapter.FlaxModuleAdapter.config(
        name='test_layer',
        module_factory_method=fdl.Partial(flax_nn.Dense, features=32),
    )
    test_layer = instantiate(test_layer_p)

    inputs = jnp.zeros(5)
    with base_layer.JaxContext.new_context():
      init_vars = test_layer.init(jax.random.PRNGKey(seed=123), inputs)
      out = test_layer.apply(init_vars, inputs)
      self.assertEqual(out.shape, (32,))


class PaxWrapperTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  def test_wrap_pax_params(self):
    bn_p = pax_fiddle.Config(normalizations.BatchNorm, name='pax_bn', dim=3)

    class SomeFlaxModel(flax_nn.Module):
      bn_p: Any

      @flax_nn.compact
      def __call__(self, x: JTensor) -> JTensor:
        x = self.bn_p.Instantiate()(x)
        return x

    key = jax.random.PRNGKey(1)
    m = SomeFlaxModel(bn_p)
    input_x = jnp.ones((4, 4, 3))
    with base_layer.JaxContext.new_context():
      initial_vars = m.init(key, input_x)
      wrapped_layer_res = m.apply(initial_vars, input_x)

      pax_layer = bn_p.Instantiate()
      initial_vars = pax_layer.init(key, input_x)
      pax_layer_res = pax_layer.apply(initial_vars, input_x)
      self.assertAllClose(wrapped_layer_res, pax_layer_res)

  def test_wrap_pax_layer_then_adapter(self):
    bn_p = pax_fiddle.Config(normalizations.BatchNorm, name='pax_bn', dim=5)

    # The flax module contains praxis layers.
    # But it is also adapted into a praxis layer.
    class SomeFlaxModel(flax_nn.Module):
      bn_p: Any

      @flax_nn.compact
      def __call__(self, x: JTensor) -> JTensor:
        x = flax_nn.Dense(5)(x)
        x = self.bn_p.Instantiate()(x)
        return x

    test_layer_p = pax_fiddle.Config(
        flax_adapter.FlaxModuleAdapter,
        module_factory_method=lambda: SomeFlaxModel(bn_p),
        name='test_layer',
    )
    test_layer = instantiate(test_layer_p)
    inputs = jnp.zeros((5, 5))
    with base_layer.JaxContext.new_context():
      init_vars = test_layer.init(jax.random.PRNGKey(seed=123), inputs)
      leaves = jax.tree_util.tree_leaves(
          init_vars, is_leaf=lambda x: isinstance(x, base_layer.BoxedParam))
      # Check that init variables are unboxed.
      self.assertFalse(
          any(isinstance(x, base_layer.BoxedParam) for x in leaves))
      test_layer.apply(init_vars, inputs)

  def test_wrap_pax_layer(self):
    bn_p = pax_fiddle.Config(normalizations.BatchNorm, name='pax_bn', dim=3)
    pax_layer = bn_p.Instantiate()

    class SomeFlaxModel(flax_nn.Module):
      pax_layer: Any

      @flax_nn.compact
      def __call__(self, x: JTensor) -> JTensor:
        x = self.pax_layer(x)
        return x

    key = jax.random.PRNGKey(1)
    m = SomeFlaxModel(pax_layer)
    input_x = jnp.ones((4, 4, 3))
    with base_layer.JaxContext.new_context():
      initial_vars = m.init(key, input_x)
      wrapped_layer_res = m.apply(initial_vars, input_x)

      initial_vars = pax_layer.init(key, input_x)
      pax_layer_res = pax_layer.apply(initial_vars, input_x)
      self.assertAllClose(wrapped_layer_res, pax_layer_res)


if __name__ == '__main__':
  absltest.main()
