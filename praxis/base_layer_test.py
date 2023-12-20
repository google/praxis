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
from typing import Any, Callable

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


class Linear(base_layer.BaseLayer):
  input_dims: int = 2
  output_dims: int = 2

  def setup(self):
    self.create_variable(
        'w',
        base_layer.WeightHParams(
            shape=[self.input_dims, self.output_dims],
            init=self.params_init,
            dtype=self.dtype,
        ),
    )

  def __call__(self, inputs):
    return jnp.matmul(inputs, self.theta.w)

  def quantized_partition_specs(self) -> Any:
    return {base_layer.PARAMS: {'w': base_layer.BoxedPartitionSpec(meta=None)}}

  def quantize_weight(self) -> base_layer.NestedJTensor:
    return {base_layer.PARAMS: {'w': self.theta.w.astype(jnp.int8)}}


class MultipleLinearLayer(base_layer.BaseLayer):
  linear1: AddBias | None = pax_fiddle.instance_field(Linear)
  linear2_tpl: pax_fiddle.Config[Linear] = pax_fiddle.template_field(Linear)

  def setup(self):
    self.create_child('linear2', self.linear2_tpl)
    # A dangling layer that is not passed through in __call__, won't have any
    # variables.
    self.create_child('linear_dangling', self.linear2_tpl)

  def __call__(self, x: base_layer.JTensor) -> base_layer.JTensor:
    return self.linear2(self.linear1(x))


class QuantizedSubChannelFFN(base_layer.BaseLayer):
  # input_dim = sub_channels * block_size
  sub_channels: int = 2
  block_size: int = 2
  output_dims: int = 3

  def setup(self):
    self.create_quantized_variable(
        'w',
        base_layer.WeightHParams(
            shape=[self.sub_channels, self.block_size, self.output_dims],
            init=self.params_init,
        ),
        use_symmetric=False,
        scale_hparams=base_layer.WeightHParams(
            shape=[self.sub_channels, self.output_dims]
        ),
    )

  def __call__(self, inputs):
    w, s, zp = self.get_quantized_weight('w', False)
    inputs = jnp.reshape(
        inputs, [inputs.shape[0], self.sub_channels, self.block_size]
    )
    out = jnp.einsum('...sc,scz,sz->...z', inputs, w, s)
    return out - jnp.einsum('...sc,sz->...z', inputs, zp)

  def quantized_partition_specs(self) -> Any:
    scale_name = 'w' + base_layer.QUANTIZED_SCALE_NAME_POSTFIX
    zp_name = 'w' + base_layer.QUANTIZED_ZP_NAME_POSTFIX
    pspec = base_layer.BoxedPartitionSpec(meta=None)
    return {base_layer.PARAMS: {'w': pspec, scale_name: pspec, zp_name: pspec}}

  def quantize_weight(self) -> base_layer.NestedJTensor:
    w, s, zp = self.get_quantized_weight('w', False)
    scale_name = 'w' + base_layer.QUANTIZED_SCALE_NAME_POSTFIX
    zp_name = 'w' + base_layer.QUANTIZED_ZP_NAME_POSTFIX
    return {base_layer.PARAMS: {'w': w, scale_name: s, zp_name: zp}}


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

  def test_summary_type_from_key(self):
    for t in base_layer.SummaryType:
      name = 'summary'
      key = name + base_layer.get_summary_type_suffix(t)
      t_key = base_layer.get_summary_type_from_key(key)
      self.assertEqual(t, t_key)
      self.assertEqual(name, base_layer.trim_summary_type_from_key(key))

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

  def test_quantize_children(self):
    layer_p = pax_fiddle.Config(
        MultipleLinearLayer, name='test_multiple_linear'
    )
    layer = base_layer.instantiate(layer_p)

    x = jnp.array([1.0, 2.0], dtype=jnp.float32)
    init_vars = layer.init(jax.random.PRNGKey(0), x)
    qw, _ = layer.apply(init_vars, mutable=[], method=layer.quantize_weight)
    self.assertEqual(
        jax.tree_map(lambda x: x.dtype, qw),
        {
            'params': {
                'linear1': {'w': jnp.int8},
                'linear2': {'w': jnp.int8},
            }
        },
    )

    pspec, _ = layer.apply(
        init_vars, mutable=[], method=layer.quantized_partition_specs
    )
    dummy_pspec = base_layer.BoxedPartitionSpec(meta=None)
    self.assertEqual(
        pspec,
        {
            'params': {
                'linear1': {'w': dummy_pspec},
                'linear2': {'w': dummy_pspec},
            }
        },
    )

  def test_quantized_sub_channels(self):
    layer_p = pax_fiddle.Config(QuantizedSubChannelFFN, name='test_sub_channel')
    layer = base_layer.instantiate(layer_p)
    x = jnp.arange(8).reshape(2, 4).astype(jnp.float32)
    init_vars = layer.init(jax.random.PRNGKey(0), x)
    quantized_types = {
        'params': {
            'w': jnp.int8,
            'w_quantized_scale': jnp.float32,
            'w_quantized_zp': jnp.float32,
        }
    }
    dummy_pspec = base_layer.BoxedPartitionSpec(meta=None)
    self.assertEqual(
        jax.tree_map(lambda x: x.dtype, init_vars), quantized_types
    )
    qw, _ = layer.apply(init_vars, mutable=[], method=layer.quantize_weight)
    self.assertEqual(jax.tree_map(lambda x: x.dtype, qw), quantized_types)
    quantized_pspec, _ = layer.apply(
        init_vars, mutable=[], method=layer.quantized_partition_specs
    )
    self.assertEqual(
        quantized_pspec,
        {
            'params': {
                'w': dummy_pspec,
                'w_quantized_scale': dummy_pspec,
                'w_quantized_zp': dummy_pspec,
            }
        },
    )

  @parameterized.parameters((0, 2), (3, 0), (1, 4))
  def test_layer_building_nn_compact(self, num_child: int, num_children: int):
    x = jnp.array([[0.0, 1.0], [2.0, 3.0]], dtype=jnp.float32)

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
        contiguous_submeshes=True,
    )

    with self.subTest('fiddle_to_fiddle'):
      target = pax_fiddle.Config(
          base_layer.BaseLayer, dtype=jnp.float16, ici_mesh_shape=None
      )
      base_layer.BaseLayerApi.copy_base_hparams(source, target)
      self.assertEqual(target.dtype, jnp.float16)
      self.assertEqual(target.ici_mesh_shape, [2, 3, 4])
      self.assertEqual(target.dcn_mesh_shape, [5, 6, 7])
      self.assertTrue(target.contiguous_submeshes)

  def test_copy_base_hparams_instantiate(self):
    class ChildLayer(base_layer.BaseLayer):
      pass

      def __call__(self):
        return 0.0

    class ParentLayer(base_layer.BaseLayer):
      # instance fields:
      a: base_layer.BaseLayer = base_layer.instance_field(ChildLayer)
      bs: list[base_layer.BaseLayer] = base_layer.instance_field(list)
      # template fields:
      x_tpl: LayerTpl = base_layer.template_field(ChildLayer)
      y_tpls: list[LayerTpl] = base_layer.template_field(list)

      def setup(self):
        self.create_child('x', self.x_tpl)
        self.create_children('ys', self.y_tpls)

      def __call__(self):
        self.a()
        for x in self.bs:
          x()
        self.x()
        for y in self.ys:
          y()
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

    def validate(model):
      # model
      hparams = model['_hparams']
      self.assertEqual(hparams.cls, ParentLayer)
      self.assertEqual(hparams.dtype, jnp.int64)
      self.assertEqual(hparams.ici_mesh_shape, (1,))
      self.assertEqual(hparams.params_init.scale, 2.0)

      # model.a
      a_hparams = model['a']['_hparams']
      self.assertEqual(a_hparams.cls, ParentLayer)
      self.assertEqual(a_hparams.dtype, jnp.int64)
      self.assertEqual(a_hparams.ici_mesh_shape, (3,))
      self.assertEqual(a_hparams.dcn_mesh_shape, (2,))
      self.assertEqual(a_hparams.params_init.scale, 4.0)

      aa_hparams = model['a']['a']['_hparams']
      self.assertEqual(aa_hparams.cls, ChildLayer)
      self.assertEqual(aa_hparams.dtype, jnp.int64)
      self.assertEqual(aa_hparams.ici_mesh_shape, (3,))
      self.assertEqual(aa_hparams.dcn_mesh_shape, (2,))

      a_bs_0 = model['a']['bs_0']['_hparams']
      self.assertEqual(a_bs_0.cls, ChildLayer)
      self.assertEqual(a_bs_0.dtype, jnp.int64)
      self.assertEqual(a_bs_0.ici_mesh_shape, (3,))
      self.assertEqual(a_bs_0.dcn_mesh_shape, (4,))
      self.assertEqual(a_bs_0.params_init.scale, 4.0)

      a_x = model['a']['x']['_hparams']
      self.assertEqual(a_x.dtype, jnp.int64)
      self.assertEqual(a_x.ici_mesh_shape, (3,))
      self.assertEqual(a_x.dcn_mesh_shape, (2,))

      # model.bs
      bs_0 = model['bs_0']['_hparams']
      self.assertEqual(bs_0.cls, ChildLayer)
      self.assertEqual(bs_0.dtype, jnp.int64)
      self.assertEqual(bs_0.ici_mesh_shape, (5,))
      self.assertEqual(bs_0.dcn_mesh_shape, (2,))
      self.assertEqual(bs_0.params_init.scale, 2.0)
      # model.x

      x = model['x']['_hparams']
      self.assertEqual(x.cls, ChildLayer)
      self.assertEqual(x.dtype, jnp.int32)
      self.assertEqual(x.ici_mesh_shape, (1,))
      self.assertEqual(x.dcn_mesh_shape, (2,))
      self.assertEqual(x.params_init.scale, 2.0)

      # model.ys
      ys_0 = model['ys_0']['_hparams']
      self.assertEqual(ys_0.cls, ParentLayer)
      self.assertEqual(ys_0.dtype, jnp.int64)
      self.assertEqual(ys_0.ici_mesh_shape, (1,))
      self.assertEqual(ys_0.dcn_mesh_shape, (6,))
      self.assertEqual(ys_0.params_init.scale, 2.0)

      ys_0_a = model['ys_0']['a']['_hparams']
      self.assertEqual(ys_0_a.cls, ChildLayer)
      self.assertEqual(ys_0_a.dtype, jnp.float16)
      self.assertEqual(ys_0_a.ici_mesh_shape, (1,))
      self.assertEqual(ys_0_a.dcn_mesh_shape, (6,))

      ys_0_x = model['ys_0']['x']['_hparams']
      self.assertEqual(ys_0_x.cls, ChildLayer)
      self.assertEqual(ys_0_x.dtype, jnp.int64)
      self.assertEqual(ys_0_x.ici_mesh_shape, (1,))
      self.assertEqual(ys_0_x.dcn_mesh_shape, (6,))

      ys_0_ys_0 = model['ys_0']['ys_0']['_hparams']
      self.assertEqual(ys_0_ys_0.cls, ChildLayer)
      self.assertEqual(ys_0_ys_0.dtype, jnp.float16)
      self.assertEqual(ys_0_ys_0.ici_mesh_shape, (1,))
      self.assertEqual(ys_0_ys_0.dcn_mesh_shape, (7,))
      self.assertEqual(ys_0_ys_0.params_init.scale, 2.0)

    model = make_model.as_buildable().Instantiate()
    configs = model.abstract_init_with_mdl_config()
    print(configs)
    validate(configs)

  def test_post_init_hparams(self):

    class FiddleChild(base_layer.BaseLayer):
      x: int = 0

      def __call__(self):
        return 0.0

    class FiddleParent(base_layer.BaseLayer):

      child_tpl: pax_fiddle.Config = base_layer.template_field(FiddleChild)
      child_tpl_list: list[pax_fiddle.Config] = base_layer.template_field(None)
      child_tpl_dict: dict[str, pax_fiddle.Config] = base_layer.template_field(
          None
      )
      child_instance_list: list[base_layer.BaseLayer] | None = None
      child_instance_dict: list[base_layer.BaseLayer] | None = None

      def setup(self):
        child_tpl = self.child_tpl.clone()
        child_tpl.x += 2
        self.create_child('child', child_tpl)

      def __call__(self):
        # Really trigger child to be setup.
        self.child()
        return 0.0

    p = pax_fiddle.Config(FiddleParent, name='test')
    p.child_tpl = pax_fiddle.Config(FiddleChild, x=5)
    p.child_tpl_list = [
        pax_fiddle.Config(FiddleChild, x=7),
        pax_fiddle.Config(FiddleChild, x=12),
    ]
    p.child_tpl_dict = {'x': pax_fiddle.Config(FiddleChild, x=12)}
    p.child_instance_list = p.child_tpl_list  # pytype: disable=annotation-type-mismatch  # use-fiddle-overlay
    p.child_instance_dict = p.child_tpl_dict  # pytype: disable=annotation-type-mismatch  # use-fiddle-overlay
    layer = p.Instantiate()

    hyper_params = layer.abstract_init_with_mdl_config()
    self.assertEqual(hyper_params['_hparams'].dtype, jnp.float32)
    self.assertIsNone(hyper_params['_hparams'].child_tpl)
    self.assertIsNone(hyper_params['_hparams'].child_tpl_list)
    self.assertIsNone(hyper_params['_hparams'].child_tpl_dict)
    self.assertIsNone(hyper_params['_hparams'].child_instance_list)
    self.assertIsNone(hyper_params['_hparams'].child_instance_dict)
    self.assertEqual(hyper_params['child']['_hparams'].dtype, jnp.float32)
    self.assertEqual(hyper_params['child']['_hparams'].x, 7)

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
      # TODO(pax-team): Preserve only tuple option, once Flax 0.6.11 is
      # released.
      expected_to_text_options = []
      for t in (tuple, list):
        expected_to_text_options.append(
            '\n'.join([
                'activation_split_dims_mapping.out : NoneType',
                f'cls : type/__main__/{Layer.__qualname__}',
                'contiguous_submeshes : NoneType',
                f'dcn_mesh_shape : {t((3, 4))}',
                'dtype : type/jax.numpy/float32',
                'fprop_dtype : type/jax.numpy/float16',
                f'ici_mesh_shape : {t((1, 2))}',
                f"mesh_axis_names : {t(('a', 'b'))}",
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
      self.assertIn(actual_to_text, expected_to_text_options)

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

  def test_unbox_meta(self):
    meta = base_layer.WeightHParams([10, 10])
    tree = {
        'params': {
            'whp': base_layer.BoxedParam(value=jnp.zeros([10, 10]), meta=meta),
            'jarr': jnp.zeros([1, 1, 1]),
        }
    }
    unboxed = base_layer.unbox_meta(tree)
    self.assertEqual(unboxed['params']['whp'], meta)
    self.assertIsInstance(unboxed['params']['jarr'], base_layer.WeightHParams)
    self.assertSequenceEqual(unboxed['params']['jarr'].shape, [1, 1, 1])
    self.assertEqual(unboxed['params']['jarr'].dtype, jnp.float32)
    self.assertEmpty(unboxed['params']['jarr'].collections)

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
        (
            'Expected .* to be Fiddle Configs.* This may be caused by an'
            ' incorrect type annotation on a field that contains a Fiddle'
            ' Config.'
        ),
    ):
      layer.init(jax.random.PRNGKey(0), 0)

  def testTypeCheckingForDtype(self):
    layer_p = pax_fiddle.Config(SimpleBaseLayer)
    with self.assertRaisesRegex(
        TypeError, r'Please use `layer_p\.Instantiate\(\)` instead'):
      SimpleBaseLayer(layer_p)  # pytype: disable=wrong-arg-types  # jnp-type

    with self.assertRaisesRegex(
        TypeError, r'Please use `layer_p\.Instantiate\(\)` instead'):
      SimpleBaseLayer('foo')  # pytype: disable=wrong-arg-types  # jnp-type

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
