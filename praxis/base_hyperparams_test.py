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

"""Tests for base_hyperparams."""

import dataclasses
import functools
import inspect
import pickle
import textwrap
from typing import Optional

from absl.testing import absltest
import fiddle as fdl
from praxis import base_hyperparams
from praxis import base_layer


class SimpleTestClass(base_hyperparams.BaseParameterizable):

  class HParams(base_hyperparams.BaseHyperParams):
    a: int
    b: str = 'b'


class SimpleTestChild(SimpleTestClass):

  class HParams(SimpleTestClass.HParams):
    c: float = 2.0

  params: base_hyperparams.BaseHyperParams
  child: SimpleTestClass

  def __init__(self, hparams: base_hyperparams.BaseHyperParams,
               child: SimpleTestClass):
    super().__init__(hparams)
    self.child = child


class NestedTestClass(base_hyperparams.BaseParameterizable):

  class HParams(base_hyperparams.BaseHyperParams):
    # Note: This is now no longer recommended; only Params should be fields of
    # Params.
    d: Optional[SimpleTestChild] = None
    e: float = 3.0


class HyperParamsTest(absltest.TestCase):

  def test_params_dataclass(self):
    x = SimpleTestClass.HParams(a=1)

    self.assertTrue(dataclasses.is_dataclass(x))
    self.assertEqual(1, x.a)
    self.assertEqual('b', x.b)
    # TODO(b/225403281): Enable the following once dataclasses are frozen.
    # _ = hash(x)  # Hash should work.
    cls_repr = ("SimpleTestClass.HParams(a=1, b='b', "
                "cls=<class '__main__.SimpleTestClass'>)")
    self.assertEqual(cls_repr, repr(x))

  def test_params_child(self):
    x = SimpleTestChild.HParams(a=1)

    self.assertTrue(dataclasses.is_dataclass(x))
    self.assertEqual(1, x.a)
    self.assertEqual('b', x.b)
    self.assertEqual(2.0, x.c)
    # TODO(b/225403281): Enable the following once dataclasses are frozen.
    # _ = hash(x)  # Hash should work.
    self.assertEqual(
        "SimpleTestChild.HParams(a=1, b='b', "
        "cls=<class '__main__.SimpleTestChild'>, c=2.0)", repr(x))

  def test_overriding_param_without_listing(self):
    with self.assertRaisesRegex(AttributeError, 'Attribute b was overridden'):

      class Broken(SimpleTestClass):

        class HParams(SimpleTestClass.HParams):
          b: bool

      _ = Broken.HParams(a=1)

  def test_overriding_param_with_listing(self):

    class Works(SimpleTestClass):

      class HParams(SimpleTestClass.HParams):
        _attribute_overrides = ('b',)
        b: bool

    x = Works.HParams(a=1, b=False)
    expected = ('Works.HParams(a=1, b=False, cls=<class '
                "'__main__.HyperParamsTest.test_overriding_param_with_listing."
                "<locals>.Works'>)")
    got = repr(x)[-len(expected):]
    self.assertEqual(got, expected, f'full repr: {repr(x)}')

  def test_to_text(self):
    x = NestedTestClass.HParams(
        d=SimpleTestChild.HParams(a=456, b='hello'), e=37)
    self.assertEqual(
        x.to_text(),
        textwrap.dedent("""\
        cls : type/__main__/NestedTestClass
        d.a : 456
        d.b : 'hello'
        d.c : 2.0
        d.cls : type/__main__/SimpleTestChild
        e : 37
        """))

  def test_freeze_params(self):
    # pylint: disable=protected-access
    x = NestedTestClass.HParams(
        d=SimpleTestChild.HParams(a=456, b='hello'), e=37)
    x.freeze()
    with self.assertRaises(AttributeError):
      x.d.a = 100
    x.unfreeze()
    x.d.a = 200
    self.assertEqual(200, x.d.a)
    x.freeze()
    self.assertEqual(True, x._internal_frozen)
    self.assertEqual(True, x.d._internal_frozen)
    x_clone = x.clone()
    self.assertEqual(200, x_clone.d.a)
    self.assertEqual(False, x_clone._internal_frozen)
    x_clone.d.a = 300
    self.assertEqual(300, x_clone.d.a)
    # pylint: enable=protected-access

  def test_fiddle_params_config(self):
    config = SimpleTestClass.HParams.config(a=1)
    config.a = 2
    params = base_hyperparams.instantiate(config)
    self.assertIsInstance(params, SimpleTestClass.HParams)
    self.assertEqual(2, params.a)

  def test_fiddle_params_partial(self):
    config = SimpleTestClass.HParams.partial(a=1)
    config.a = 2
    params_fn = base_hyperparams.instantiate(config)
    self.assertIsInstance(params_fn, functools.partial)
    self.assertEqual(2, params_fn.keywords['a'])

    # Instantiate the partial.
    params = params_fn()
    self.assertIsInstance(params, SimpleTestClass.HParams)
    self.assertEqual(2, params.a)

    # Should allow overrides like any functools.partial object.
    params = params_fn(a=3)
    self.assertEqual(3, params.a)

  def test_fiddle_allowed_parameters(self):
    p = SimpleTestClass.config()
    p.a = 5
    obj = base_hyperparams.instantiate(p)
    self.assertIsInstance(obj.hparams, SimpleTestClass.HParams)
    self.assertEqual(5, obj.hparams.a)
    self.assertEqual('b', obj.hparams.b)

  def test_fiddle_overrides_defaults(self):
    p = SimpleTestClass.config(a=42, b='c')
    obj = base_hyperparams.instantiate(p)
    self.assertIsInstance(obj, SimpleTestClass)
    self.assertIsInstance(obj.hparams, SimpleTestClass.HParams)
    self.assertEqual(42, obj.hparams.a)
    self.assertEqual('c', obj.hparams.b)

  def test_fiddle_eager_error_checking(self):
    p = SimpleTestClass.config()
    with self.assertRaisesRegex(
        TypeError,
        "No parameter named 'not_there' exists.*valid parameter names: a, b"):
      p.not_there = 5

    with self.assertRaisesRegex(TypeError, 'invalid_name'):
      _ = SimpleTestClass.config(invalid_name=5)

  def test_fiddle_in_plain_dataclass_errors(self):
    cfg = SimpleTestClass.config()
    with self.assertRaisesRegex(TypeError,
                                r'.*Fiddle config.*inside dataclasses'):
      NestedTestClass.HParams(d=cfg)

    nested_params = NestedTestClass.HParams()
    with self.assertRaisesRegex(TypeError,
                                r'.*Fiddle config.*inside dataclasses'):
      nested_params.d = cfg

  def test_fiddle_nested(self):
    p = NestedTestClass.config()
    p.d = SimpleTestChild.config(a=40, c=-1.3)
    p.d.child = SimpleTestClass.config()  # TODO(saeta): use __fiddle_init__?
    p.d.child.a = 42
    p.d.child.b = 'very_nested_b'

    obj = base_hyperparams.instantiate(p)

    self.assertIsInstance(obj.hparams, NestedTestClass.HParams)
    self.assertIsInstance(obj.hparams.d, SimpleTestChild)
    self.assertIsInstance(obj.hparams.d.child, SimpleTestClass)
    self.assertEqual(obj.hparams.e, 3.0)
    self.assertEqual(obj.hparams.d.hparams.a, 40)
    self.assertEqual(obj.hparams.d.hparams.b, 'b')
    self.assertEqual(obj.hparams.d.hparams.c, -1.3)
    self.assertEqual(obj.hparams.d.child.hparams.a, 42)
    self.assertEqual(obj.hparams.d.child.hparams.b, 'very_nested_b')

  def test_fiddle_serialization(self):
    p = SimpleTestClass.config(a=24)
    reloaded = pickle.loads(pickle.dumps(p))
    obj = base_hyperparams.instantiate(reloaded)

    self.assertIsInstance(obj, SimpleTestClass)
    self.assertIsInstance(obj.hparams, SimpleTestClass.HParams)
    self.assertEqual(24, obj.hparams.a)
    self.assertEqual('b', obj.hparams.b)

  def test_fiddle_partial(self):
    p = SimpleTestClass.partial()
    partial = base_hyperparams.instantiate(p)
    obj = partial(a=10, b='my_b')

    self.assertIsInstance(obj, SimpleTestClass)
    self.assertIsInstance(obj.hparams, SimpleTestClass.HParams)
    self.assertEqual(10, obj.hparams.a)
    self.assertEqual('my_b', obj.hparams.b)

  def test_no_extra_init_arguments(self):
    simple_init_signature = inspect.signature(SimpleTestClass)
    self.assertNotIn('_config_init_name_for_params_object',
                     simple_init_signature.parameters)
    self.assertNotIn('_nonconfigurable_init_args',
                     simple_init_signature.parameters)

    child_init_signature = inspect.signature(SimpleTestChild)
    self.assertNotIn('_config_init_name_for_params_object',
                     child_init_signature.parameters)
    self.assertNotIn('_nonconfigurable_init_args',
                     child_init_signature.parameters)

    layer_init_signature = inspect.signature(base_layer.BaseLayer)
    self.assertNotIn('_config_init_name_for_params_object',
                     layer_init_signature.parameters)
    self.assertNotIn('_nonconfigurable_init_args',
                     layer_init_signature.parameters)

  def test_duplicate_parameters(self):
    with self.assertRaisesRegex(TypeError, r'Duplicated parameter.*foo'):

      class DuplicatedParameter(base_hyperparams.BaseParameterizable):  # pylint: disable=unused-variable

        class HParams(base_hyperparams.BaseHyperParams):
          foo: int = 0

        def __init__(self,
                     hparams: base_hyperparams.BaseParameterizable.HParams,
                     foo: int = 7):
          pass

  def test_improper_init_arg_name(self):
    with self.assertRaisesRegex(
        TypeError, r'WrongInit.__init__ must have a parameter '
        r'named hparams'):

      class WrongInit(base_hyperparams.BaseParameterizable):  # pylint: disable=unused-variable

        class HParams(base_hyperparams.BaseHyperParams):
          something: int = 42

        def __init__(self, p: base_hyperparams.BaseParameterizable.HParams):
          pass

  def test_make_factories(self):

    class DefaultFactoryTestClass(base_hyperparams.BaseParameterizable):

      class HParams(base_hyperparams.BaseHyperParams):
        a: list[str] = dataclasses.field(default_factory=lambda: [1, 2, 3])

    instance_1 = DefaultFactoryTestClass.make()
    instance_2 = DefaultFactoryTestClass.make()
    self.assertEqual(instance_1.hparams.a, [1, 2, 3])
    instance_1.hparams.a.append(4)
    self.assertEqual(instance_1.hparams.a, [1, 2, 3, 4])
    self.assertEqual(instance_2.hparams.a, [1, 2, 3])

  def test_fiddle_factory_integration(self):

    class Foo(base_hyperparams.BaseParameterizable):

      class HParams(base_hyperparams.BaseHyperParams):
        foo_a: int = 0

    class Bar(base_hyperparams.BaseParameterizable):

      class HParams(base_hyperparams.BaseHyperParams):
        foo_tpl: base_hyperparams.BaseHyperParams = (
            base_hyperparams.sub_config_field(Foo.HParams))

    field, = [
        field for field in dataclasses.fields(Bar.HParams)
        if field.name == 'foo_tpl'
    ]
    self.assertIsInstance(field.default_factory,
                          base_hyperparams.SubConfigFactory)
    cfg = Bar.HParams.config()
    cfg.foo_tpl.foo_a = 1
    bar_params = fdl.build(cfg)
    self.assertEqual(bar_params.foo_tpl.foo_a, 1)


if __name__ == '__main__':
  absltest.main()
