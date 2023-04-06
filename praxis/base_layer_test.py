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

"""Tests for base_layer."""

import copy
import dataclasses
import sys
import typing
from typing import Any, Callable, Dict, List, Optional, Tuple

from absl.testing import absltest
from absl.testing import parameterized
import fiddle as fdl
from flax import core as flax_core
from flax import linen as nn
import jax
import jax.numpy as jnp
from praxis import base_hyperparams
from praxis import base_layer
from praxis import pax_fiddle
from praxis import test_utils

LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]


class Identity(base_layer.BaseLayer):
  """Layer for testing summary writing."""

  def setup(self):
    pass

  def __call__(self, x):
    self.add_summary('debug', x, verbosity=3)
    self.add_summary('info', x, verbosity=2)
    # tensor value can be a deferred callable as well:
    self.add_summary('callable', lambda: x, verbosity=2)
    return x


class AddBias(base_layer.BaseLayer):
  """A layer that adds bias to an input tensor."""

  @nn.compact
  def __call__(self, x: base_layer.JTensor) -> base_layer.JTensor:
    var_p = base_layer.WeightHParams(
        shape=(x.shape[-1],), init=base_layer.WeightInit.Constant(0.))
    b = self.create_variable('b', var_hparams=var_p, trainable=True)
    x, b = self._cast_to_fprop_dtype((x, b))
    return x + b


class MultipleBiasLayer(base_layer.BaseLayer):
  """A dummy layer that adds multiple biases to an input tensor.

  Attributes:
    num_child: number of independent child AddBias layers to test.
    num_children: number of children to be added into a self.create_children.
  """
  num_child: int = 0
  num_children: int = 0

  @nn.compact
  def __call__(self, x: base_layer.JTensor) -> base_layer.JTensor:
    b_p = pax_fiddle.Config(AddBias)
    for i in range(self.num_child):
      x = self.create_child(f'child_{i}', b_p)(x)
    layers = self.create_children('children', [b_p] * self.num_children)
    for layer in layers:
      x = layer(x)
    return x


class TrivialFiddleLayer(base_layer.BaseLayer):
  pass


class SimpleBaseLayer(base_layer.BaseLayer):
  x: int = 0


class BaseLayerTest(test_utils.TestCase):

  def test_summary_same_input_name(self):
    a = jnp.array([1., 2.], dtype=jnp.float32)
    b = jnp.array([3., 4.], dtype=jnp.float32)
    with base_layer.JaxContext.new_context() as context:
      summary_dict = context.summary_dict
      summary_dict.add_summary('my_custom_summary', a,
                               base_layer.SummaryType.SCALAR)
      summary_dict.add_summary('my_custom_summary', b,
                               base_layer.SummaryType.SCALAR)

      for key in summary_dict.dict:
        summary_type = base_layer.get_summary_type_from_key(key)
        self.assertEqual(summary_type, base_layer.SummaryType.SCALAR)
      self.assertSameElements(
          list(summary_dict.dict.keys()),
          {'my_custom_summary_scalar', 'my_custom_summary1_scalar'})

  def test_layer_summary_verbosity_log(self):
    layer_p = pax_fiddle.Config(Identity, name='test_identity')
    layer = base_layer.instantiate(layer_p)

    x = jnp.array([1., 2.], dtype=jnp.float32)
    init_vars = layer.init(jax.random.PRNGKey(0), x)
    _, updated_vars = layer.apply(init_vars, x, mutable=[base_layer.SUMMARIES])
    summaries = updated_vars[base_layer.SUMMARIES]

    self.assertIn('debug_scalar', summaries)
    self.assertIn('info_scalar', summaries)
    self.assertIn('callable_scalar', summaries)
    self.assertArraysEqual(x, summaries['debug_scalar'])
    self.assertArraysEqual(x, summaries['info_scalar'])
    self.assertArraysEqual(x, summaries['callable_scalar'])

  def test_layer_summary_verbosity_no_log(self):
    context_p = base_layer.JaxContext.HParams(
        do_eval=True, summary_verbosity=2)
    with base_layer.JaxContext.new_context(hparams=context_p):
      layer_p = pax_fiddle.Config(Identity, name='test_identity')
      layer = base_layer.instantiate(layer_p)

      x = jnp.array([1., 2.], dtype=jnp.float32)
      init_vars = layer.init(jax.random.PRNGKey(0), x)
      _, updated_vars = layer.apply(
          init_vars, x, mutable=[base_layer.SUMMARIES])
      summaries = updated_vars[base_layer.SUMMARIES]

    self.assertNotIn('debug_scalar', summaries)
    self.assertIn('info_scalar', summaries)
    self.assertArraysEqual(x, summaries['info_scalar'])

  @parameterized.named_parameters(
      ('log', 2, 2, True),
      ('no_log', 2, 3, False),
  )
  def test_global_summary_verbosity(
      self, ctx_verbosity, summary_verbosity, should_log):
    context_p = base_layer.JaxContext.HParams(
        do_eval=True, summary_verbosity=ctx_verbosity)
    with base_layer.JaxContext.new_context(hparams=context_p):
      summary = jnp.array([1., 2.], dtype=jnp.float32)
      base_layer.add_global_summary('test', summary,
                                    verbosity=summary_verbosity)
      all_summaries = base_layer.all_global_summaries()

      self.assertEqual('test_scalar' in all_summaries, should_log)
      if should_log:
        self.assertArraysEqual(summary, all_summaries['test_scalar'])

  def test_get_summary_base_type(self):
    self.assertEqual(
        base_layer.SummaryType.SCALAR,
        base_layer.get_summary_base_type(base_layer.SummaryType.SCALAR))
    self.assertEqual(
        base_layer.SummaryType.SCALAR,
        base_layer.get_summary_base_type(
            base_layer.SummaryType.AGGREGATE_SCALAR))
    self.assertEqual(
        base_layer.SummaryType.IMAGE,
        base_layer.get_summary_base_type(base_layer.SummaryType.IMAGE))
    self.assertEqual(
        base_layer.SummaryType.IMAGE,
        base_layer.get_summary_base_type(
            base_layer.SummaryType.AGGREGATE_IMAGE))
    self.assertEqual(
        base_layer.SummaryType.TEXT,
        base_layer.get_summary_base_type(
            base_layer.SummaryType.TEXT))
    self.assertEqual(
        base_layer.SummaryType.VIDEO,
        base_layer.get_summary_base_type(
            base_layer.SummaryType.VIDEO))

  def test_quantize(self):
    layer_p = pax_fiddle.Config(Identity, name='test_identity')
    layer = base_layer.instantiate(layer_p)

    x = jnp.array([1., 2.], dtype=jnp.float32)
    init_vars = layer.init(jax.random.PRNGKey(0), x)
    res, _ = layer.apply(init_vars, mutable=[], method=layer.quantize_weight)
    self.assertEqual(res, {})
    pspec, _ = layer.apply(
        init_vars, mutable=[], method=layer.quantized_partition_specs
    )
    self.assertEqual(pspec, {})

  @parameterized.parameters((0, 2), (3, 0), (1, 4))
  def test_layer_building_nn_compact(self, num_child: int, num_children: int):
    x = jnp.array([[0., 1.], [2., 3.]], dtype=jnp.float32)

    p = pax_fiddle.Config(
        MultipleBiasLayer,
    )
    p.name = 'multi_bias'
    p.num_child = num_child
    p.num_children = num_children
    layer = base_layer.instantiate(p)

    with base_layer.JaxContext.new_context():
      params = layer.init(jax.random.PRNGKey(0), x)

    flattened_params, _ = jax.tree_util.tree_flatten(params)
    self.assertLen(flattened_params, num_children + num_child)

    y = layer.apply(params, x)
    self.assertAllClose(x, y)

  def test_copy_base_hparams(self):

    class ChildLayer(base_layer.BaseLayer):
      params_init: base_layer.WeightInit = base_layer.instance_field(
          lambda: base_layer.WeightInit.Uniform(0.5)  # override default
      )

    class ParentLayer(base_layer.BaseLayer):
      child: Any = None
      child_tpl: pax_fiddle.Config = base_layer.template_field(ChildLayer)
      params_init: base_layer.WeightInit = base_layer.instance_field(
          base_layer.WeightInit.Gaussian)

    source = pax_fiddle.Config(
        base_layer.BaseLayer,
        dtype=jnp.float64,
        ici_mesh_shape=[2, 3, 4],
        dcn_mesh_shape=[5, 6, 7],
        params_init=base_layer.default_param_init(),
    )

    with self.subTest('fiddle_to_fiddle'):
      target = pax_fiddle.Config(
          base_layer.BaseLayer, dtype=jnp.float16, ici_mesh_shape=None
      )
      base_layer.BaseLayerApi.copy_base_hparams(source, target)
      self.assertEqual(target.dtype, jnp.float16)
      self.assertEqual(target.ici_mesh_shape, [2, 3, 4])
      self.assertEqual(target.dcn_mesh_shape, [5, 6, 7])

  def test_copy_base_hparams_instantiate(self):
    class ChildLayer(base_layer.BaseLayer):
      pass

    class ParentLayer(base_layer.BaseLayer):
      # instance fields:
      a: base_layer.BaseLayer = base_layer.instance_field(ChildLayer)
      bs: List[base_layer.BaseLayer] = base_layer.instance_field(list)
      # template fields:
      x_tpl: LayerTpl = base_layer.template_field(ChildLayer)
      y_tpls: List[LayerTpl] = base_layer.template_field(list)

      def setup(self):
        self.create_child('x', self.x_tpl)
        self.create_children('ys', self.y_tpls)

      def __call__(self):
        validate(self)  # Defined below.
        return 0

    @pax_fiddle.auto_config
    def make_model():
      return ParentLayer(
          dtype=jnp.int64,
          ici_mesh_shape=(1,),
          dcn_mesh_shape=(2,),
          params_init=base_layer.WeightInit.Gaussian(2.0),
          a=ParentLayer(
              ici_mesh_shape=(3,),
              bs=[ChildLayer(dcn_mesh_shape=(4,))],
              params_init=base_layer.WeightInit.Uniform(4.0),
          ),
          bs=[ChildLayer(ici_mesh_shape=(5,))],
          x_tpl=pax_fiddle.Config(ChildLayer, dtype=jnp.int32),
          y_tpls=[
              pax_fiddle.Config(
                  ParentLayer,
                  dcn_mesh_shape=(6,),
                  a=ChildLayer(dtype=jnp.float16),
                  y_tpls=[
                      pax_fiddle.Config(
                          ChildLayer, dtype=jnp.float16, dcn_mesh_shape=(7,)
                      )
                  ],
              )
          ],
      )

    model = make_model.as_buildable().Instantiate()

    def validate(model):
      # model
      self.assertIsInstance(model, ParentLayer)
      self.assertEqual(model.dtype, jnp.int64)
      self.assertEqual(model.ici_mesh_shape, (1,))
      self.assertEqual(model.params_init.scale, 2.0)
      # model.a
      self.assertIsInstance(model.a, ParentLayer)
      self.assertIsInstance(model.a.a, ChildLayer)
      self.assertLen(model.a.bs, 1)
      self.assertIsInstance(model.a.bs[0], ChildLayer)
      self.assertEqual(model.a.dtype, jnp.int64)
      self.assertEqual(model.a.ici_mesh_shape, (3,))
      self.assertEqual(model.a.dcn_mesh_shape, (2,))
      self.assertEqual(model.a.params_init.scale, 4.0)
      self.assertEqual(model.a.a.dtype, jnp.int64)
      self.assertEqual(model.a.a.ici_mesh_shape, (3,))
      self.assertEqual(model.a.a.dcn_mesh_shape, (2,))
      self.assertEqual(model.a.bs[0].dtype, jnp.int64)
      self.assertEqual(model.a.bs[0].ici_mesh_shape, (3,))
      self.assertEqual(model.a.bs[0].dcn_mesh_shape, (4,))
      self.assertEqual(model.a.bs[0].params_init.scale, 4.0)
      self.assertEqual(model.a.x.dtype, jnp.int64)
      self.assertEqual(model.a.x.ici_mesh_shape, (3,))
      self.assertEqual(model.a.x.dcn_mesh_shape, (2,))
      # model.bs
      self.assertLen(model.bs, 1)
      self.assertIsInstance(model.bs[0], ChildLayer)
      self.assertEqual(model.bs[0].dtype, jnp.int64)
      self.assertEqual(model.bs[0].ici_mesh_shape, (5,))
      self.assertEqual(model.bs[0].dcn_mesh_shape, (2,))
      self.assertEqual(model.bs[0].params_init.scale, 2.0)
      # model.x
      self.assertIsInstance(model.x, ChildLayer)
      self.assertEqual(model.x.dtype, jnp.int32)
      self.assertEqual(model.x.ici_mesh_shape, (1,))
      self.assertEqual(model.x.dcn_mesh_shape, (2,))
      self.assertEqual(model.x.params_init.scale, 2.0)
      # model.ys
      self.assertLen(model.ys, 1)
      self.assertIsInstance(model.ys[0], ParentLayer)
      self.assertIsInstance(model.ys[0].a, ChildLayer)
      self.assertLen(model.ys[0].bs, 0)
      self.assertIsInstance(model.ys[0].x, ChildLayer)
      self.assertLen(model.ys[0].ys, 1)
      self.assertIsInstance(model.ys[0].ys[0], ChildLayer)
      self.assertEqual(model.ys[0].dtype, jnp.int64)
      self.assertEqual(model.ys[0].ici_mesh_shape, (1,))
      self.assertEqual(model.ys[0].dcn_mesh_shape, (6,))
      self.assertEqual(model.ys[0].params_init.scale, 2.0)
      self.assertEqual(model.ys[0].a.dtype, jnp.float16)
      self.assertEqual(model.ys[0].a.ici_mesh_shape, (1,))
      self.assertEqual(model.ys[0].a.dcn_mesh_shape, (6,))
      self.assertEqual(model.ys[0].x.dtype, jnp.int64)
      self.assertEqual(model.ys[0].x.ici_mesh_shape, (1,))
      self.assertEqual(model.ys[0].x.dcn_mesh_shape, (6,))
      self.assertEqual(model.ys[0].ys[0].dtype, jnp.float16)
      self.assertEqual(model.ys[0].ys[0].ici_mesh_shape, (1,))
      self.assertEqual(model.ys[0].ys[0].dcn_mesh_shape, (7,))
      self.assertEqual(model.ys[0].ys[0].params_init.scale, 2.0)

    # Note: this invokes model.setup() and then model.__call__(), which calls
    # the `validate` function.
    model.init(jax.random.PRNGKey(0))

  def test_post_init_hparams(self):

    class FiddleChild(base_layer.BaseLayer):
      x: int = 0

    class FiddleParent(base_layer.BaseLayer):

      child_tpl: pax_fiddle.Config = base_layer.template_field(FiddleChild)
      child_tpl_list: List[pax_fiddle.Config] = base_layer.template_field(None)
      child_tpl_dict: Dict[str, pax_fiddle.Config] = base_layer.template_field(
          None
      )
      child_instance_list: Optional[List[base_layer.BaseLayer]] = None
      child_instance_dict: Optional[List[base_layer.BaseLayer]] = None

      def setup(self):
        child_tpl = self.child_tpl.clone()
        child_tpl.x += 2
        self.create_child('child', child_tpl)

      def __call__(self):
        return 0

    p = pax_fiddle.Config(FiddleParent, name='test')
    p.child_tpl = pax_fiddle.Config(FiddleChild, x=5)
    p.child_tpl_list = [
        pax_fiddle.Config(FiddleChild, x=7),
        pax_fiddle.Config(FiddleChild, x=12),
    ]
    p.child_tpl_dict = {'x': pax_fiddle.Config(FiddleChild, x=12)}
    p.child_instance_list = p.child_tpl_list
    p.child_instance_dict = p.child_tpl_dict
    layer = p.Instantiate()

    model = layer.bind(
        layer.init(jax.random.PRNGKey(0)),
        mutable=[base_layer.HYPER_PARAMS])
    model.post_init_hparams()
    hyper_params = jax.tree_map(
        lambda x: x.meta,
        model.variables[base_layer.HYPER_PARAMS],
        is_leaf=lambda x: isinstance(x, base_layer.WrappedHParams))

    self.assertEqual(hyper_params['_hparams'].dtype, jnp.float32)
    self.assertEqual(hyper_params['child']['_hparams'].dtype, jnp.float32)
    self.assertEqual(hyper_params['child']['_hparams'].x, 7)
    self.assertIsNone(hyper_params['_hparams'].child_tpl)
    self.assertIsNone(hyper_params['_hparams'].child_tpl_list)
    self.assertIsNone(hyper_params['_hparams'].child_tpl_dict)
    self.assertIsNone(hyper_params['_hparams'].child_instance_list)
    self.assertIsNone(hyper_params['_hparams'].child_instance_dict)

  @parameterized.parameters([
      (pax_fiddle.Config(SimpleBaseLayer),
       pax_fiddle.Config(SimpleBaseLayer), True),
      (pax_fiddle.Config(SimpleBaseLayer),
       pax_fiddle.Config(SimpleBaseLayer, name='foo'), True),
      (pax_fiddle.Config(SimpleBaseLayer),
       pax_fiddle.Config(SimpleBaseLayer, dtype=jnp.float16), False),
  ])
  def test_compatible_hparams(self, lhs, rhs, expected):
    self.assertEqual(base_layer.compatible_hparams(lhs, rhs), expected)

  def test_flax_parent_can_assign_name(self):

    class Parent(nn.Module):
      get_layer: Callable[[], nn.Module]

      def setup(self):
        self.layer = self.get_layer()

      def __call__(self, x):
        return self.layer(x)

    class CompactParent(nn.Module):
      get_layer: Callable[[], nn.Module]

      @nn.compact
      def __call__(self, x):
        return self.get_layer()(x)

    key = jax.random.PRNGKey(0)
    for parent_cls, assigned_name in [(Parent, 'layer'),
                                      (CompactParent, 'AddBias_0')]:
      mod = parent_cls(
          get_layer=lambda: pax_fiddle.Config(AddBias).Instantiate()
      )
      prms = mod.init({'params': key}, jnp.ones((3, 3)))
      self.assertIn(assigned_name, prms['params'])

      mod = parent_cls(
          get_layer=lambda: pax_fiddle.Config(AddBias, name='x').Instantiate()
      )
      prms = mod.init({'params': key}, jnp.ones((3, 3)))
      self.assertIn('x', prms['params'])

  @parameterized.parameters([
      dict(expected=None),
      dict(ici_mesh_shape=[1, 2], mesh_axis_names=['a', 'b'], expected=[1, 2]),
      dict(
          ici_mesh_shape=[1, 2],
          dcn_mesh_shape=[3, 4],
          contiguous_submeshes=False,
          mesh_axis_names=['a', 'b'],
          expected=[1 * 3, 2 * 4],
      ),
  ])
  def test_mesh_shape_property(self, expected, **kwargs):
    layer = base_layer.BaseLayer(**kwargs)
    self.assertEqual(layer.mesh_shape, expected)

  def test_hparams_instance_stub(self):

    class Layer(base_layer.BaseLayer):
      x: int = 0

    layer = Layer(
        x=3,
        fprop_dtype=jnp.float16,
        ici_mesh_shape=[1, 2],
        dcn_mesh_shape=[3, 4],
        mesh_axis_names=['a', 'b'],
        parent=flax_core.Scope({}),
        name='my_layer')

    hparams_stub = layer.hparams
    with self.subTest('fields'):
      self.assertIsInstance(hparams_stub, pax_fiddle.Config)
      self.assertEqual(hparams_stub.x, 3)
      self.assertEqual(hparams_stub.fprop_dtype, jnp.float16)
      self.assertEqual(hparams_stub.dtype, jnp.float32)
      self.assertEqual(hparams_stub.mesh_shape, [3, 8])
      self.assertEqual(hparams_stub.name, 'my_layer')

    with self.subTest('clone'):
      cloned = hparams_stub.clone()
      self.assertIsInstance(cloned, pax_fiddle.Config)
      self.assertEqual(cloned.cls, Layer)
      self.assertEqual(fdl.get_callable(cloned), Layer)
      self.assertEqual(cloned.x, 3)
      self.assertEqual(cloned.fprop_dtype, jnp.float16)
      self.assertEqual(cloned.dtype, jnp.float32)
      self.assertEqual(cloned.name, 'my_layer')
      self.assertNotIn('parent', cloned.__arguments__)

    with self.subTest('to_text'):
      expected_to_text = (
          '\n'.join([
              'activation_split_dims_mapping.out : NoneType',
              f'cls : {Layer!r}',
              'contiguous_submeshes : False',
              'dcn_mesh_shape : [3, 4]',
              'dtype : type/jax.numpy/float32',
              'fprop_dtype : type/jax.numpy/float16',
              'ici_mesh_shape : [1, 2]',
              "mesh_axis_names : ['a', 'b']",
              "name : 'my_layer'",
              "params_init.method : 'xavier'",
              'params_init.scale : 1.000001',
              'shared_weight_layer_id : NoneType',
              'skip_lp_regularization : NoneType',
              'weight_split_dims_mapping.wt : NoneType',
              'x : 3',
          ])
          + '\n'
      )
      actual_to_text = base_hyperparams.nested_struct_to_text(hparams_stub)
      self.assertEqual(actual_to_text, expected_to_text)

    with self.subTest('can_deepcopy'):
      copy.deepcopy(hparams_stub)

  def test_override_weight_sharding_hparams(self):

    class Layer(base_layer.BaseLayer):

      class WeightSharding(base_layer.BaseLayer.WeightSharding):
        x: int = 5

      class ActivationSharding(base_layer.BaseLayer.ActivationSharding):
        y: str = 'y'

    with self.subTest('construct_layer_directly'):
      layer = Layer()
      self.assertIsInstance(layer.weight_split_dims_mapping,
                            pax_fiddle.Config)
      self.assertEqual(
          fdl.get_callable(layer.weight_split_dims_mapping),
          Layer.WeightSharding,
      )
      self.assertIsNone(layer.weight_split_dims_mapping.wt)
      self.assertEqual(layer.weight_split_dims_mapping.x, 5)
      self.assertIsInstance(layer.activation_split_dims_mapping,
                            pax_fiddle.Config)
      self.assertEqual(
          fdl.get_callable(layer.activation_split_dims_mapping),
          Layer.ActivationSharding,
      )
      self.assertIsNone(layer.activation_split_dims_mapping.out)
      self.assertEqual(layer.activation_split_dims_mapping.y, 'y')

    with self.subTest('build_layer_from_fiddle_config'):
      cfg = pax_fiddle.Config(Layer)
      cfg.weight_split_dims_mapping.x = 12
      cfg.activation_split_dims_mapping.y = 'yellow'
      layer2 = pax_fiddle.build(cfg)
      self.assertEqual(layer2.weight_split_dims_mapping.x, 12)
      self.assertEqual(layer2.activation_split_dims_mapping.y, 'yellow')

  def test_hparam_is_instance_of_fdl_buildable(self):

    class Child(base_layer.BaseLayer):
      size: int = 5

    with self.assertRaisesRegex(
        ValueError, 'default value is a mutable instance of fdl.Buildable'):

      # Allowing the default value of `child_tpl` to be a Config object here
      # would be problematic, because that mutable default value would be
      # shared by all instances of Parent.  E.g., if `a` and `b` were two
      # instances of Parent that did not override `child_tpl`, then setting
      # `a.child_tpl.size = 20` would also modify `b.child_tpl.size` to be 20
      # (since `a.child_tpl is b.child_tpl`).  We therefore raise an exception,
      # indicating that the user should use a `default_factory` rather than a
      # default value.
      class Parent(base_layer.BaseLayer):
        child_tpl: pax_fiddle.Config = pax_fiddle.Config(Child, size=2)

      del Parent  # unused.

  def test_fprop_dtype(self):
    with self.subTest('default'):
      layer = base_layer.BaseLayer()
      self.assertEqual(layer.fprop_dtype, jnp.float32)

    with self.subTest('override_dtype'):
      layer = base_layer.BaseLayer(dtype=jnp.float16)
      self.assertEqual(layer.dtype, jnp.float16)
      self.assertEqual(layer.fprop_dtype, jnp.float16)

    with self.subTest('override_fprop_dtype'):
      layer = base_layer.BaseLayer(fprop_dtype=jnp.float64)
      self.assertEqual(layer.fprop_dtype, jnp.float64)

    with self.subTest('override_both'):
      layer = base_layer.BaseLayer(dtype=jnp.float16, fprop_dtype=jnp.float64)
      self.assertEqual(layer.fprop_dtype, jnp.float64)

    with self.subTest('frozen_during_post_init'):
      # If a BaseLayer is created by an `@nn.compact` method, then
      # it will be already-frozen during __post_init__.  This test checks
      # that we can still set fprop_dtype, even though the instance is
      # frozen.

      class SomeFiddleLayer(base_layer.BaseLayer):

        def __call__(self, x):
          return x

      class SomeFlaxModel(nn.Module):

        @nn.compact
        def __call__(self, x):
          return SomeFiddleLayer()(x)

      with base_layer.JaxContext.new_context():
        SomeFlaxModel().init(jax.random.PRNGKey(1), jnp.ones((4, 4, 3)))

  def test_fiddle_base_layer_may_not_have_hparams(self):
    expected_err = (
        "For <class '.*Layer'>: PAX layers should no longer use nested HParams "
        'classes. Instead, add fields directly to the layer class.')
    with self.assertRaisesRegex(ValueError, expected_err):
      class Layer(base_layer.BaseLayer):  # pylint: disable=unused-variable

        class HParams:
          x: int = 0

  def test_check_template_has_do_not_build_tag(self):

    # pylint: disable=unused-variable
    with self.subTest('FiddleConfig'):
      with self.assertRaisesRegex(
          ValueError,
          'has a template type, but does not have the pax_fiddle.DoNotBuild.*'):

        class Layer2(base_layer.BaseLayer):
          child_tpl: pax_fiddle.Config = dataclasses.field(
              default_factory=lambda: pax_fiddle.Config(SimpleBaseLayer))

    with self.subTest('Optional_FiddleConfig'):
      if not hasattr(typing, 'get_origin'):
        self.skipTest('This version of Python has not typing.get_origin')
      with self.assertRaisesRegex(
          ValueError,
          'has a template type, but does not have the pax_fiddle.DoNotBuild.*'):

        class Layer4(base_layer.BaseLayer):
          child_tpl: Optional[pax_fiddle.Config] = None

    with self.subTest('Optional_Parameterized_FiddleConfig'):
      if not hasattr(typing, 'get_origin'):
        self.skipTest('This version of Python has not typing.get_origin')
      with self.assertRaisesRegex(
          ValueError,
          'has a template type, but does not have the pax_fiddle.DoNotBuild.*'):

        class Layer5(base_layer.BaseLayer):
          child_tpl: Optional[pax_fiddle.Config[TrivialFiddleLayer]] = None

    with self.subTest('Optional_List_FiddleConfig'):
      if not hasattr(typing, 'get_origin'):
        self.skipTest('This version of Python has not typing.get_origin')
      with self.assertRaisesRegex(
          ValueError,
          'has a template type, but does not have the pax_fiddle.DoNotBuild.*'):

        class Layer6(base_layer.BaseLayer):
          child_tpl: Optional[List[pax_fiddle.Config]] = None

    with self.subTest('tuple_int_int'):

      if (sys.version_info.major, sys.version_info.minor) < (3, 9):
        self.skipTest('tuple[int, int] not supported in this Python version')

      class Layer7(base_layer.BaseLayer):
        child: Tuple[int, int] = (1, 2)

  def testMissingDoNotFieldTagError(self):

    class Parent(base_layer.BaseLayer):

      child_tpl: Any = None

      def setup(self):
        self.create_child('child', self.child_tpl)

      def __call__(self):
        return None

    cfg = pax_fiddle.Config(
        Parent, child_tpl=pax_fiddle.Config(TrivialFiddleLayer))
    layer = pax_fiddle.build(cfg)
    with self.assertRaisesRegex(
        ValueError,
        'Expected .* to be Fiddle Configs.* This may be caused by '
        'a missing DoNotBuild tag on a field that contains a Fiddle Config.'):
      layer.init(jax.random.PRNGKey(0), 0)

  def testOverrideParamsInit(self):

    class Child(base_layer.BaseLayer):
      params_init: base_layer.WeightInit = (
          base_layer.WeightInit.UniformUnitScaling(scale=0.5))

    class Parent(base_layer.BaseLayer):

      child_tpl: pax_fiddle.Config = base_layer.template_field(Child)

      def setup(self):
        self.create_child('child', self.child_tpl)

      def __call__(self):
        return None

    cfg = pax_fiddle.Config(Parent)
    layer = pax_fiddle.build(cfg)
    layer.init(jax.random.PRNGKey(0))
    prng_key = jax.random.PRNGKey(seed=123)

    def gen_post_init_hparams(prng_key):
      return layer.apply({},
                         rngs={base_layer.PARAMS: prng_key},
                         method=layer.post_init_hparams,
                         mutable=True)[1]

    variables_abstract = jax.eval_shape(gen_post_init_hparams, prng_key)
    assert base_layer.HYPER_PARAMS in variables_abstract
    hyper_params = jax.tree_map(
        lambda x: x.meta,
        variables_abstract[base_layer.HYPER_PARAMS],
        is_leaf=lambda x: isinstance(x, base_layer.WrappedHParams))

    self.assertEqual(0.5, hyper_params['child']['_hparams'].params_init.scale)
    self.assertEqual('uniform_unit_scaling',
                     hyper_params['child']['_hparams'].params_init.method)

  def testTypeCheckingForDtype(self):
    layer_p = pax_fiddle.Config(SimpleBaseLayer)
    with self.assertRaisesRegex(
        TypeError, r'Please use `layer_p\.Instantiate\(\)` instead'):
      SimpleBaseLayer(layer_p)

    with self.assertRaisesRegex(
        TypeError, r'Please use `layer_p\.Instantiate\(\)` instead'):
      SimpleBaseLayer('foo')

  def test_get_fan_in_fan_out(self):
    self.assertEqual((None, None), base_layer.get_fan_in_fan_out(shape=[]))
    self.assertEqual((1024, 1024), base_layer.get_fan_in_fan_out(shape=[1024]))
    self.assertEqual((1024, 8192), base_layer.get_fan_in_fan_out([1024, 8192]))
    self.assertEqual((64 * 1024, 64 * 8192),
                     base_layer.get_fan_in_fan_out([64, 1024, 8192]))
    # With explicit fan_in_axes/fan_out_axes the factor of 64 disappears.
    self.assertEqual((1024, 8192),
                     base_layer.get_fan_in_fan_out([64, 1024, 8192],
                                                   fan_in_axes=[-2],
                                                   fan_out_axes=[-1]))
    self.assertEqual((3 * 1024 * 32, 3 * 1024 * 128),
                     base_layer.get_fan_in_fan_out(shape=[3, 1024, 32, 128]))
    # With explicit fan_in_axes/fan_out_axes the factor of 3 disappears,
    # and fan out is computed as product of last 2 dims.
    self.assertEqual((1024, 32 * 128),
                     base_layer.get_fan_in_fan_out(
                         shape=[3, 1024, 32, 128],
                         fan_in_axes=[-3],
                         fan_out_axes=[-2, -1],
                     ))


if __name__ == '__main__':
  absltest.main()
