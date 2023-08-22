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

"""Tests for pax_fiddle."""

import copy
import dataclasses
import functools
import types
from typing import Any, Callable, NamedTuple, Sequence

from absl.testing import absltest
from absl.testing import parameterized
import fiddle as fdl
from fiddle import daglish
from fiddle import testing
from fiddle.experimental import serialization
from fiddle.experimental import visualize
from flax import core as flax_core
from flax import linen as nn
import jax
from jax import numpy as jnp
from lingvo.core import nested_map
from praxis import base_hyperparams
from praxis import base_layer
from praxis import pax_fiddle


@dataclasses.dataclass
class Wheel:
  radius: int = 5

  def setup(self):
    return self


@dataclasses.dataclass
class ColoredWheel(Wheel):
  color: str = "black"


@dataclasses.dataclass
class Person:
  name: str | None = None

  def setup(self):
    return self


@dataclasses.dataclass
class Vehicle:
  wheel_tpl: pax_fiddle.Config[Wheel] = pax_fiddle.template_field(Wheel)
  num_wheels: int = 4
  owner: Person = pax_fiddle.instance_field(Person)
  wheels: list[Wheel] | None = None  # Initialized by setup.

  def setup(self):
    assert self.wheels is None
    self.wheels = [
        pax_fiddle.build(self.wheel_tpl).setup() for _ in range(self.num_wheels)
    ]
    return self


@dataclasses.dataclass
class ColoredVehicle(Vehicle):
  color: str = "white"
  wheel_tpl: pax_fiddle.Config[Wheel] = pax_fiddle.template_field(
      lambda: ColoredWheel(color="red")  # override color
  )


@dataclasses.dataclass
class Fleet:
  vehicle_tpl: pax_fiddle.Config[Vehicle] = pax_fiddle.template_field(Vehicle)
  num_vehicles: int = 1
  manager: Person = pax_fiddle.instance_field(Person)
  vehicles: list[Vehicle] | None = None  # Initialized by setup.

  def setup(self):
    assert self.vehicles is None
    self.vehicles = [
        pax_fiddle.build(self.vehicle_tpl).setup()
        for _ in range(self.num_vehicles)
    ]
    return self


@dataclasses.dataclass
class BusStop:
  location: str  # required arg.
  times: list[int] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class HourlyBusStop(BusStop):
  times: list[int] = dataclasses.field(default_factory=lambda: list(range(24)))


@dataclasses.dataclass
class WheelFactory:
  wheel_tpl: list[pax_fiddle.Config[Wheel]] = dataclasses.field(
      default_factory=list
  )


class NonDataclassWheelFactory:

  def __init__(self, wheel_tpl: list[pax_fiddle.Config]):
    self.wheel_tpl = wheel_tpl


class NamedTupleWheelFactory(NamedTuple):
  wheel_tpl: list[pax_fiddle.Config[Wheel]]


class SubFieldAndTemplateFieldTest(testing.TestCase):

  def test_default_fleet_config(self):
    config = pax_fiddle.Config(Fleet)
    with self.subTest("expected_config"):
      self.assertDagEqual(
          config,
          pax_fiddle.Config(
              Fleet,
              vehicle_tpl=pax_fiddle.Config(
                  Vehicle,
                  wheel_tpl=pax_fiddle.Config(Wheel),
                  owner=pax_fiddle.Config(Person),
              ),
              manager=pax_fiddle.Config(Person),
          ),
      )

    with self.subTest("with_materialized_defaults"):
      fdl.materialize_defaults(config)
      self.assertDagEqual(
          config,
          pax_fiddle.Config(
              Fleet,
              vehicle_tpl=pax_fiddle.Config(
                  Vehicle,
                  wheel_tpl=pax_fiddle.Config(Wheel, radius=5),
                  owner=pax_fiddle.Config(Person, name=None),
                  num_wheels=4,
                  wheels=None,
              ),
              num_vehicles=1,
              manager=pax_fiddle.Config(Person, name=None),
              vehicles=None,
          ),
      )

  def test_build_default_fleet_config(self):
    config = pax_fiddle.Config(Fleet)
    config.manager.name = "Ben"  # Required arg
    fleet = pax_fiddle.build(config)
    self.assertEqual(
        fleet,
        Fleet(
            vehicle_tpl=pax_fiddle.Config(Vehicle),
            num_vehicles=1,
            manager=Person("Ben"),
        ),
    )

  def test_build_custom_fleet_config(self):
    config = pax_fiddle.Config(Fleet)
    config.manager.name = "Ben"
    config.num_vehicles = 3
    config.vehicle_tpl.wheel_tpl.radius *= 2
    config.vehicle_tpl.owner = config.manager
    fleet = pax_fiddle.build(config)
    self.assertEqual(
        fleet,
        Fleet(
            vehicle_tpl=pax_fiddle.Config(
                Vehicle,
                wheel_tpl=pax_fiddle.Config(Wheel, radius=10),
                owner=pax_fiddle.Config(Person, name="Ben"),
            ),
            num_vehicles=3,
            manager=Person("Ben"),
        ),
    )

  def test_build_and_setup_default_fleet_config(self):
    config = pax_fiddle.Config(Fleet)
    config.manager.name = "Ben"  # Required arg
    config.vehicle_tpl.owner.name = "Joe"  # Required arg
    fleet = pax_fiddle.build(config).setup()
    self.assertEqual(
        fleet,
        Fleet(
            vehicle_tpl=pax_fiddle.Config(
                Vehicle, owner=pax_fiddle.Config(Person, name="Joe")
            ),
            vehicles=[
                Vehicle(
                    wheel_tpl=pax_fiddle.Config(Wheel),
                    wheels=[Wheel(), Wheel(), Wheel(), Wheel()],
                    num_wheels=4,
                    owner=Person("Joe"),
                )
            ],
            num_vehicles=1,
            manager=Person("Ben"),
        ),
    )

  def test_build_and_setup_custom_fleet_config(self):
    config = pax_fiddle.Config(Fleet)

    config.manager.name = "Ben"
    config.num_vehicles = 3
    config.vehicle_tpl.wheel_tpl.radius *= 2
    config.vehicle_tpl.owner.name = "Joe"

    with self.subTest("got_expected_fleet"):
      fleet = pax_fiddle.build(config).setup()
      self.assertEqual(
          fleet,
          Fleet(
              vehicle_tpl=pax_fiddle.Config(
                  Vehicle,
                  owner=pax_fiddle.Config(Person, name="Joe"),
                  wheel_tpl=pax_fiddle.Config(Wheel, radius=10),
              ),
              vehicles=3
              * [
                  Vehicle(
                      wheel_tpl=pax_fiddle.Config(Wheel, radius=10),
                      wheels=4 * [Wheel(10)],
                      num_wheels=4,
                      owner=Person("Joe"),
                  )
              ],
              num_vehicles=3,
              manager=Person("Ben"),
          ),
      )

    # Check that no sub-objects are unintentionally shared.
    with self.subTest("no_accidental_sharing"):
      self.assertIsNot(fleet.vehicles[0], fleet.vehicles[1])
      self.assertIsNot(fleet.vehicles[0].owner, fleet.vehicles[1].owner)
      self.assertIsNot(fleet.vehicles[0].wheels[0], fleet.vehicles[0].wheels[1])
      self.assertIsNot(fleet.vehicles[0].wheels[0], fleet.vehicles[1].wheels[0])
      self.assertIsNot(fleet.vehicles[0].wheels[0], fleet.vehicles[1].wheels[1])
      self.assertIsNot(fleet.manager, fleet.vehicles[0].owner)

  def test_shared_vehicle_owners(self):
    config = pax_fiddle.Config(Fleet)

    config.manager.name = "Ben"
    config.num_vehicles = 3
    config.vehicle_tpl.owner = config.manager  # shared config object.

    fleet = pax_fiddle.build(config).setup()
    self.assertEqual(
        fleet,
        Fleet(
            vehicle_tpl=pax_fiddle.Config(
                Vehicle, owner=pax_fiddle.Config(Person, name="Ben")
            ),
            vehicles=3
            * [
                Vehicle(
                    wheel_tpl=pax_fiddle.Config(Wheel),
                    wheels=4 * [Wheel()],
                    num_wheels=4,
                    owner=Person("Ben"),
                )
            ],
            num_vehicles=3,
            manager=Person("Ben"),
        ),
    )

    # Note: there is no object sharing between fleet.manager and
    # fleet.vehicle[i].owner, or between fleet.vehicle[i].owner and
    # fleet.vehicle[j].owner, despite the fact that they were all constructed
    # from the same Config object (by identity).  This lack of sharing occurs
    # because they were all constructed during different calls to
    # pax_fiddle.build.  This will change when we transition to using factory
    # objects (partials) instead of config objects to store templates.
    self.assertIsNot(fleet.vehicles[0].owner, fleet.vehicles[1].owner)
    self.assertIsNot(fleet.vehicles[0].owner, fleet.vehicles[2].owner)
    self.assertIsNot(fleet.vehicles[1].owner, fleet.vehicles[2].owner)
    self.assertIsNot(fleet.manager, fleet.vehicles[0].owner)

  def test_use_custom_wheel(self):
    config = pax_fiddle.Config(Fleet)
    config.vehicle_tpl.wheel_tpl = pax_fiddle.Config(ColoredWheel, color="red")
    fleet = pax_fiddle.build(config).setup()
    self.assertIsInstance(fleet.vehicles[0].wheels[0], ColoredWheel)
    self.assertEqual(
        fleet.vehicles[0].wheels[0], ColoredWheel(radius=5, color="red")
    )

  def test_build_fleet_directly(self):
    fleet = Fleet()
    fleet = fleet.setup()
    self.assertEqual(
        fleet,
        Fleet(
            vehicle_tpl=pax_fiddle.Config(Vehicle),
            num_vehicles=1,
            vehicles=[
                Vehicle(
                    wheel_tpl=pax_fiddle.Config(Wheel),
                    num_wheels=4,
                    wheels=[Wheel(), Wheel(), Wheel(), Wheel()],
                    owner=Person(),
                )
            ],
            manager=Person(),
        ),
    )

    self.assertEqual(fleet.vehicle_tpl, pax_fiddle.Config(Vehicle))
    self.assertEqual(fleet.vehicles[0].wheel_tpl, pax_fiddle.Config(Wheel))

  def test_instance_field_empty_container_default_factory(self):
    @dataclasses.dataclass
    class TestCls:
      items: list[Any] = pax_fiddle.instance_field(list)
      tags: dict[str, Any] = pax_fiddle.instance_field(dict)

    cfg = pax_fiddle.Config(TestCls)
    self.assertDagEqual(cfg, pax_fiddle.Config(TestCls, items=[], tags={}))
    v1 = cfg.Instantiate()
    v2 = cfg.Instantiate()
    self.assertEqual(v1, TestCls(items=[], tags={}))
    self.assertEqual(v2, TestCls(items=[], tags={}))
    self.assertIsNot(v1.items, v2.items)
    self.assertIsNot(v1.tags, v2.tags)

    cfg.items.append(pax_fiddle.Config(Wheel))
    v3 = cfg.Instantiate()
    self.assertEqual(v3, TestCls(items=[Wheel()]))


class SampleTag(fdl.Tag):
  """A tag used in testing."""


class AdditionalTag(fdl.Tag):
  """A second tag to test multiple tags."""


@dataclasses.dataclass
class ATaggedType:
  untagged: str
  tagged: str = pax_fiddle.field(tags=SampleTag, default="tagged")
  double_tagged: str = pax_fiddle.field(
      tags=(AdditionalTag, SampleTag), default_factory=lambda: "other_field"
  )

  @pax_fiddle.auto_config
  @classmethod
  def default(cls):
    return cls(untagged="untagged_default")


@dataclasses.dataclass
class AnotherTaggedType:
  tagged: str = pax_fiddle.field(tags=AdditionalTag, default="tagged")


def sample_fn():
  return 1


@pax_fiddle.auto_config
def nested_structure():
  return {"foo": [sample_fn(), (2, 3)]}


@dataclasses.dataclass
class AnAutoconfigType:
  tagged_type: ATaggedType = pax_fiddle.field(
      default_factory=ATaggedType.default
  )
  another_default: dict[str, Any] = pax_fiddle.field(
      default_factory=nested_structure
  )

  # We need this for `AncestorType` below, but we might be able to make
  # `pax_fiddle.auto_config(AnAutoconfigType)` work in the future.
  @pax_fiddle.auto_config
  @classmethod
  def default(cls):
    return cls()


@dataclasses.dataclass
class AncestorType:
  # We might want to make this more compact.
  child: AnAutoconfigType = pax_fiddle.field(
      default_factory=AnAutoconfigType.default
  )


@dataclasses.dataclass
class Parent:
  """A class w/ a field that uses configurable_factory=True."""

  child: AnAutoconfigType = pax_fiddle.field(
      default_factory=AnAutoconfigType, configurable_factory=True
  )
  y: int = 0


@dataclasses.dataclass
class ParentPair:
  first: Parent = pax_fiddle.field(
      default_factory=Parent, configurable_factory=True
  )
  second: Parent = pax_fiddle.field(
      default_factory=Parent, configurable_factory=True
  )


@dataclasses.dataclass
class ParentWithOptionalChild:
  child: Any = None


@dataclasses.dataclass
class ParentWithATaggedTypeChild:
  child: Any = pax_fiddle.field(
      default_factory=ATaggedType, configurable_factory=True
  )


@dataclasses.dataclass
class A:
  x: int = 0


@dataclasses.dataclass
class B:
  a: A = pax_fiddle.field(default_factory=A, configurable_factory=True)


@dataclasses.dataclass
class C:
  b: B = pax_fiddle.field(default_factory=B, configurable_factory=True)

  @pax_fiddle.auto_config
  @classmethod
  def factory(cls):
    return functools.partial(cls)

  @pax_fiddle.auto_config
  @classmethod
  def factory2(cls):
    return functools.partial(cls, b=B())


@dataclasses.dataclass
class D:
  c_factory: Callable[..., C] = pax_fiddle.field(default_factory=C.factory)

  @pax_fiddle.auto_config
  @classmethod
  def factory(cls):
    return functools.partial(cls)


@dataclasses.dataclass
class D2:
  c_factory: Callable[..., C] = pax_fiddle.field(default_factory=C.factory2)

  @pax_fiddle.auto_config
  @classmethod
  def factory(cls):
    return functools.partial(cls)


@dataclasses.dataclass
class E:
  d_factory: Callable[..., D2] = pax_fiddle.field(default_factory=D2.factory)


class DataclassFieldExpansionTest(testing.TestCase):

  def test_dataclass_tagging(self):
    config = pax_fiddle.Config(ATaggedType)

    self.assertEqual({SampleTag}, fdl.get_tags(config, "tagged"))
    self.assertEqual(
        {SampleTag, AdditionalTag}, fdl.get_tags(config, "double_tagged")
    )

    fdl.set_tagged(config, tag=AdditionalTag, value="set_correctly")

    self.assertEqual(config.double_tagged, "set_correctly")

  def test_metadata_passthrough(self):
    other_metadata = types.MappingProxyType({"something": 4})
    constructed_field = pax_fiddle.field(metadata=other_metadata)

    self.assertIn("something", constructed_field.metadata)
    self.assertEqual(4, constructed_field.metadata["something"])

  def test_auto_config_basic_equality(self):
    self.assertEqual(
        pax_fiddle.build(pax_fiddle.Config(AnAutoconfigType)),
        AnAutoconfigType(),
    )
    self.assertEqual(
        pax_fiddle.build(pax_fiddle.Config(AncestorType)), AncestorType()
    )

  def test_name_set_in_both_cases(self):
    # A slightly more concrete test of the above.
    self.assertEqual(
        pax_fiddle.build(
            pax_fiddle.Config(AnAutoconfigType)
        ).tagged_type.untagged,
        "untagged_default",
    )
    self.assertEqual(
        AnAutoconfigType().tagged_type.untagged, "untagged_default"
    )

  def test_auto_config_override_equality(self):
    self.assertEqual(
        AnAutoconfigType(another_default={"3": 4}).another_default, {"3": 4}
    )
    self.assertEqual(
        pax_fiddle.build(
            pax_fiddle.Config(AnAutoconfigType, another_default={"3": 4})
        ),
        AnAutoconfigType(another_default={"3": 4}),
    )

  def test_auto_config_field_init(self):
    config = pax_fiddle.Config(AnAutoconfigType)
    config.another_default["foo"][1] += (4,)
    obj = pax_fiddle.build(config)
    self.assertEqual(obj.another_default, {"foo": [1, (2, 3, 4)]})

  def test_mandatory_fields(self):
    @dataclasses.dataclass
    class TwoMandatoryFieldsDataclass:
      foo: int = pax_fiddle.field(tags=SampleTag)
      bar: int

    instance = TwoMandatoryFieldsDataclass(3, 4)
    self.assertEqual(instance.foo, 3)
    self.assertEqual(instance.bar, 4)

  def test_invalid_definition_with_defaults(self):
    with self.assertRaisesRegex(
        ValueError, "cannot specify both default and default_factory"
    ):
      pax_fiddle.field(default_factory=nested_structure, default=4)

  def test_configurable_factory(self):
    config = pax_fiddle.Config(ParentPair)
    expected_config = pax_fiddle.Config(
        ParentPair,
        pax_fiddle.Config(Parent, child=pax_fiddle.Config(AnAutoconfigType)),
        pax_fiddle.Config(Parent, child=pax_fiddle.Config(AnAutoconfigType)),
    )
    self.assertDagEqual(config, expected_config)
    self.assertEqual(pax_fiddle.build(config), ParentPair())

  def test_configurable_factory_can_be_configured(self):
    # Create a config and make some changes to it.
    config = pax_fiddle.Config(ParentPair)
    config.first.y = 100
    config.second.child.another_default = {"x": 1}
    fdl.set_tagged(config, tag=SampleTag, value="changed")

    # Create a ParentPair object and make the same changes.
    expected_result = ParentPair()
    expected_result.first.y = 100
    expected_result.second.child.another_default = {"x": 1}
    expected_result.first.child.tagged_type.tagged = "changed"
    expected_result.first.child.tagged_type.double_tagged = "changed"
    expected_result.second.child.tagged_type.tagged = "changed"
    expected_result.second.child.tagged_type.double_tagged = "changed"

    self.assertEqual(pax_fiddle.build(config), expected_result)

  def test_configurable_factory_no_unintentional_aliasing(self):
    config = pax_fiddle.Config(ParentPair)
    self.assertIsNot(config.first, config.second)
    self.assertIsNot(config.first.child, config.second.child)
    self.assertIsNot(
        config.first.child.tagged_type, config.second.child.tagged_type
    )
    self.assertIsNot(
        config.first.child.another_default, config.second.child.another_default
    )

    val = pax_fiddle.build(config)
    self.assertIsNot(val.first, val.second)
    self.assertIsNot(val.first.child, val.second.child)
    self.assertIsNot(val.first.child.tagged_type, val.second.child.tagged_type)
    self.assertIsNot(
        val.first.child.another_default, val.second.child.another_default
    )

  def test_configurable_factory_autoconfig_error(self):
    with self.assertRaisesRegex(
        ValueError,
        "configurable_factory should not be used with auto_config'ed functions",
    ):
      pax_fiddle.field(
          default_factory=AnAutoconfigType.default, configurable_factory=True
      )

  def test_nested_dataclass_default_factories(self):
    with self.subTest("config_value"):
      cfg = pax_fiddle.Config(D)
      expected = pax_fiddle.Config(
          D,
          c_factory=pax_fiddle.Partial(
              C, pax_fiddle.ArgFactory(B, pax_fiddle.ArgFactory(A))
          ),
      )
      self.assertDagEqual(cfg, expected)

    with self.subTest("built_value_identity"):
      for d in [D(), pax_fiddle.build(pax_fiddle.Config(D))]:
        c1 = d.c_factory()
        c2 = d.c_factory()
        self.assertIsNot(c1, c2)
        self.assertIsNot(c1.b, c2.b)
        self.assertIsNot(c1.b.a, c2.b.a)

    with self.subTest("change_arg_factory_to_config"):
      cfg = pax_fiddle.Config(D)
      cfg.c_factory.b = pax_fiddle.Config(B)  # Now this will be shared.
      d = pax_fiddle.build(cfg)
      c1 = d.c_factory()
      c2 = d.c_factory()
      self.assertIsNot(c1, c2)
      self.assertIs(c1.b, c2.b)
      self.assertIs(c1.b.a, c2.b.a)

    with self.subTest("double_partial"):
      with self.assertRaisesRegex(ValueError, "Unable to safely replace"):
        pax_fiddle.Config(E)

    with self.subTest("expand_dataclass_default_factories_docstring"):
      f = lambda x: x
      g = lambda v=0: [v]
      make_fn = pax_fiddle.auto_config(lambda: functools.partial(f, x=g()))

      @dataclasses.dataclass
      class Test:
        fn: Callable[[], object] = pax_fiddle.field(default_factory=make_fn)

      with self.assertRaisesRegex(ValueError, "Unable to safely replace"):
        pax_fiddle.Partial(Test)

  def test_field_has_tag(self):
    self.assertTrue(
        pax_fiddle.field_has_tag(pax_fiddle.field(tags=SampleTag), SampleTag)
    )
    self.assertTrue(
        pax_fiddle.field_has_tag(
            pax_fiddle.field(tags=(SampleTag, AdditionalTag)), SampleTag
        )
    )
    self.assertFalse(
        pax_fiddle.field_has_tag(
            pax_fiddle.field(tags=AdditionalTag), SampleTag
        )
    )
    self.assertFalse(pax_fiddle.field_has_tag(pax_fiddle.field(), SampleTag))
    self.assertFalse(pax_fiddle.field_has_tag(dataclasses.field(), SampleTag))

  def test_update_callable_for_tagged_fields(self):
    cfg = pax_fiddle.Config(ATaggedType)
    self.assertEqual(fdl.get_tags(cfg, "tagged"), {SampleTag})

    # When we switch to a new dataclass callable, any tags associated with
    # fields get added.
    pax_fiddle.update_callable(cfg, AnotherTaggedType)
    self.assertEqual(fdl.get_tags(cfg, "tagged"), {SampleTag, AdditionalTag})

    # Even if we've manually adjusted the tags, they will get added.
    fdl.clear_tags(cfg, "tagged")
    pax_fiddle.update_callable(cfg, ATaggedType)
    self.assertEqual(fdl.get_tags(cfg, "tagged"), {SampleTag})

  def test_update_callable_for_configurable_factories(self):
    with self.subTest("add_configurable_factory"):
      # update_callable will add configurable factories for any fields that
      # do not have any (explicit) value.
      cfg = pax_fiddle.Config(ParentWithOptionalChild)
      self.assertIsNone(cfg.child)
      pax_fiddle.update_callable(cfg, Parent)
      self.assertEqual(fdl.get_callable(cfg.child), AnAutoconfigType)

    with self.subTest("do_not_overwrite_explicit_value"):
      # This example differs from the one above in that child is *explicitly*
      # set to `None`, so it won't get overwritten.
      cfg = pax_fiddle.Config(ParentWithOptionalChild, child=None)
      pax_fiddle.update_callable(cfg, Parent)
      self.assertIsNone(cfg.child)

    with self.subTest("do_not_overwrite_previous_configurable_factory"):
      cfg = pax_fiddle.Config(ParentWithATaggedTypeChild)
      self.assertEqual(fdl.get_callable(cfg.child), ATaggedType)
      pax_fiddle.update_callable(cfg, Parent)
      self.assertEqual(fdl.get_callable(cfg.child), ATaggedType)

    with self.subTest("do_not_delete_configurable_factory"):
      # In this test, we change to a class whose default value for `child` is
      # None; but we leave the Config built with the configurable factory.
      cfg = pax_fiddle.Config(ParentWithATaggedTypeChild)
      pax_fiddle.update_callable(cfg, ParentWithOptionalChild)
      self.assertEqual(fdl.get_callable(cfg.child), ATaggedType)


class PaxConfigTest(testing.TestCase, parameterized.TestCase):

  def test_cls_property(self):
    cfg = pax_fiddle.Config(
        Vehicle, wheel_tpl=pax_fiddle.Config(Wheel), num_wheels=3
    )
    with self.subTest("read"):
      self.assertEqual(cfg.cls, Vehicle)
      self.assertEqual(cfg.wheel_tpl.cls, Wheel)
      self.assertEqual(cfg.owner.cls, Person)

    with self.subTest("write"):
      cfg.cls = ColoredVehicle
      cfg.wheel_tpl.cls = ColoredWheel
      self.assertEqual(cfg.cls, ColoredVehicle)
      self.assertEqual(cfg.wheel_tpl.cls, ColoredWheel)

  def test_sub_template_field_with_lambda(self):
    cfg = pax_fiddle.Config(ColoredVehicle)
    self.assertIsInstance(cfg.wheel_tpl, pax_fiddle.Config)
    self.assertEqual(cfg.wheel_tpl.cls, ColoredWheel)
    self.assertEqual(cfg.wheel_tpl.color, "red")

  def test_clone(self):
    cfg = pax_fiddle.Config(
        Vehicle, wheel_tpl=pax_fiddle.Config(Wheel), num_wheels=3
    )
    clone = cfg.clone()
    self.assertEqual(cfg, clone)
    self.assertIsNot(cfg, clone)
    self.assertIsNot(cfg.wheel_tpl, clone.wheel_tpl)

  def test_set(self):
    cfg = pax_fiddle.Config(Vehicle)
    cfg.set(num_wheels=2)
    cfg.wheel_tpl.set(radius=20)
    cfg.owner.set(name="Grug")
    self.assertDagEqual(
        cfg,
        pax_fiddle.Config(
            Vehicle,
            num_wheels=2,
            owner=pax_fiddle.Config(Person, "Grug"),
            wheel_tpl=pax_fiddle.Config(Wheel, radius=20),
        ),
    )

  @parameterized.parameters([
      pax_fiddle.build,
      pax_fiddle.instantiate,
      base_hyperparams.instantiate,
      pax_fiddle.PaxConfig.Instantiate,
  ])
  def test_instantiate(self, build_func):
    cfg = pax_fiddle.Config(Vehicle)
    cfg.set(num_wheels=2)
    cfg.wheel_tpl.set(radius=20)

    vehicle = build_func(cfg).setup()
    self.assertEqual(
        vehicle,
        Vehicle(
            wheel_tpl=pax_fiddle.Config(Wheel, radius=20),
            num_wheels=2,
            wheels=[Wheel(20), Wheel(20)],
            owner=Person(),
        ),
    )

  @parameterized.parameters([
      pax_fiddle.instantiate,
      base_hyperparams.instantiate,
      pax_fiddle.PaxConfig.Instantiate,
  ])
  def test_instantiate_with_override(self, build_func):
    cfg = pax_fiddle.Config(Vehicle)
    cfg.set(num_wheels=2)
    cfg.wheel_tpl.set(radius=20)
    vehicle = build_func(cfg, owner=Person("Mo"), num_wheels=3).setup()
    self.assertEqual(
        vehicle,
        Vehicle(
            wheel_tpl=pax_fiddle.Config(Wheel, radius=20),
            num_wheels=3,
            wheels=[Wheel(20), Wheel(20), Wheel(20)],
            owner=Person("Mo"),
        ),
    )

  def test_copy_fields_from(self):
    source = pax_fiddle.Config(Vehicle, num_wheels=2)
    source.wheel_tpl.set(radius=20)
    target = pax_fiddle.Config(Vehicle)
    expected = copy.deepcopy(source)
    target.copy_fields_from(source)
    self.assertEqual(target, expected)

  def test_copy_fields_from_does_copy_name(self):
    source = pax_fiddle.Config(Person, "A")
    target = pax_fiddle.Config(Person, "B")
    target.copy_fields_from(source)
    self.assertEqual(target, pax_fiddle.Config(Person, "A"))

  def test_copy_fields_from_does_not_copy_parent(self):
    @dataclasses.dataclass
    class TestCls:
      parent: str

    source = pax_fiddle.Config(TestCls, "A")
    target = pax_fiddle.Config(TestCls, "B")
    target.copy_fields_from(source)
    self.assertEqual(target, pax_fiddle.Config(TestCls, "B"))

  def test_copy_fields_from_missing_fields_in_source(self):
    source = pax_fiddle.Config(Wheel, radius=10)
    target = pax_fiddle.Config(ColoredWheel, radius=3, color="red")
    target.copy_fields_from(source)
    self.assertEqual(
        target, pax_fiddle.Config(ColoredWheel, radius=10, color="red")
    )

  def test_copy_fields_from_missing_fields_in_self(self):
    source = pax_fiddle.Config(ColoredWheel, radius=3, color="red")
    target = pax_fiddle.Config(Wheel, radius=10)

    with self.assertRaisesRegex(
        ValueError, "Copying incompatible HParams: 'color' not in self"
    ):
      target.copy_fields_from(source)

    target.copy_fields_from(source, missing_fields_in_self=["color"])
    self.assertEqual(target, pax_fiddle.Config(Wheel, radius=3))

  def test_copy_fields_from_missing_required_value(self):
    source = pax_fiddle.Config(BusStop)
    target = pax_fiddle.Config(BusStop)
    with self.assertRaisesRegex(
        ValueError, "Can't copy from missing required .*BusStop.location"
    ):
      target.copy_fields_from(source)

  def test_copy_fields_from_compatible_default_factory(self):
    source = pax_fiddle.Config(BusStop, "Oak Town")
    target = pax_fiddle.Config(BusStop, times=[5, 8])
    target.copy_fields_from(source)
    self.assertEqual(target, pax_fiddle.Config(BusStop, "Oak Town"))

  def test_copy_fields_from_no_unintentional_sharing(self):
    source = pax_fiddle.Config(BusStop, "Oak Town", times=[5, 8])
    target = pax_fiddle.Config(BusStop)
    target.copy_fields_from(source)
    self.assertEqual(target, pax_fiddle.Config(BusStop, "Oak Town", [5, 8]))
    self.assertEqual(target.times, source.times)
    self.assertIsNot(target.times, source.times)

  def test_copy_fields_from_incompatible_default_factory(self):
    source = pax_fiddle.Config(BusStop, "Oak Town")
    target = pax_fiddle.Config(HourlyBusStop)
    with self.assertRaisesRegex(
        ValueError, "Can't copy from default_factory .*BusStop.times"
    ):
      target.copy_fields_from(source)

  def test_copy_fields_from_invalid_source(self):
    with self.subTest("source_field_not_in_self"):
      source = pax_fiddle.Config(Wheel)
      target = pax_fiddle.Config(Vehicle)
      with self.assertRaisesRegex(
          ValueError, "Copying incompatible HParams: 'radius' not in self"
      ):
        target.copy_fields_from(source)

  def test_mesh_shape(self):
    cfg = pax_fiddle.Config(base_layer.BaseLayer)
    self.assertIsNone(cfg.mesh_shape)
    cfg.mesh_axis_names = ["a", "b"]
    cfg.ici_mesh_shape = [1, 2]
    self.assertEqual(cfg.mesh_shape, [1, 2])
    cfg.dcn_mesh_shape = [3, 4]
    self.assertEqual(cfg.mesh_shape, [1 * 3, 2 * 4])

  def test_nested_map_serialization(self):
    cfg = pax_fiddle.Config(nested_map.NestedMap, x=1, y="y")
    serialized_cfg = serialization.dump_json(cfg)
    deserialized_cfg = serialization.load_json(serialized_cfg)
    self.assertEqual(cfg, deserialized_cfg)


class LayerA(nn.Module):
  x: int = 0

  def __call__(self):
    return self.x


class LayerB(nn.Module):
  a: LayerA = pax_fiddle.instance_field(LayerA)

  def __call__(self):
    return self.a()


class LayerC(nn.Module):
  b_tpl: pax_fiddle.Config = pax_fiddle.template_field(LayerB)

  def setup(self):
    self.b = pax_fiddle.build(self.b_tpl)
    if self.b.parent is not self:
      raise AssertionError(
          "Expected self.b.parent to be self (inside self.setup)."
      )

  def __call__(self, x):
    if self.b.parent is not self:
      raise AssertionError(
          "Expected self.b.parent to be self (before b.setup)."
      )
    self.b()  # Causes b.setup() to be called.
    if self.b.a.parent is not self.b:
      raise AssertionError(
          "Expected self.b.a.parent to be self.b (after b.setup)."
      )
    return 0


class LayerD(base_layer.BaseLayer):

  def setup(self):
    self.create_variable(
        "v",
        base_layer.WeightHParams(
            shape=[], init=base_layer.WeightInit.Constant(3)
        ),
    )

  def __call__(self, x):
    return self.theta.v * x


class BuildTest(testing.TestCase, parameterized.TestCase):
  """Tests for pax_fiddle.build."""

  @parameterized.parameters([
      pax_fiddle.build,
      pax_fiddle.instantiate,
      base_hyperparams.instantiate,
      pax_fiddle.PaxConfig.Instantiate,
  ])
  def test_parent_links_are_not_set_to_outer_scope(self, build_func):
    # This test checks that using the `empty_flax_module_stack` decorator is
    # effective at preventing modules from being built with the wrong parent.
    with self.subTest(build_func.__qualname__):
      cfg = pax_fiddle.Config(LayerC)
      cfg.parent = flax_core.Scope({})
      c = build_func(cfg)
      self.assertIs(c.b.parent, c)
      self.assertEqual(c(5), 0)  # Causes c.setup() and c.__call__() to be run.
      self.assertIs(c.b.parent, c)
      self.assertIs(c.b.a.parent, c.b)

  def test_build_works_with_nn_compact(self):
    class SomeFlaxModel(nn.Module):
      tpl: pax_fiddle.Config

      @nn.compact
      def __call__(self, x):
        layer = self.tpl.Instantiate()
        assert layer.parent is self
        # Calling `layer(x)` would raise an exception (can't read unbound
        # variables) if the parent of `layer` were `None`.
        return layer(x)

    m = SomeFlaxModel(pax_fiddle.Config(LayerD))
    inputs = jnp.array(22)
    with base_layer.JaxContext.new_context():
      m.init(jax.random.PRNGKey(1), inputs)

  def test_do_not_build_if_type_is_pax_config(self):
    def f1(x: pax_fiddle.Config):
      return x

    def f2(x: pax_fiddle.Config[Wheel]):
      return x

    def f3(x: pax_fiddle.Config[Wheel] | None):
      return x

    def f4(x: pax_fiddle.Config | Sequence[pax_fiddle.Config] | None):
      return x

    for fn in [f1, f2, f3, f4]:
      cfg = pax_fiddle.Config(fn, x=pax_fiddle.Config(Wheel))
      result = pax_fiddle.build(cfg)
      self.assertDagEqual(result, pax_fiddle.Config(Wheel))

  def test_do_not_build_function_args_if_arg_is_pax_config_container(self):

    def f1(x: list[pax_fiddle.Config]):
      return x

    def f2(x: list[pax_fiddle.Config[Wheel]]):
      return x

    def f3(x: list[pax_fiddle.Config[Wheel]] | None):
      return x

    def f4(x: Sequence[pax_fiddle.Config[Wheel]] | pax_fiddle.Config):
      return x

    for fn in [f1, f2, f3, f4]:
      cfg = pax_fiddle.Config(fn, x=[pax_fiddle.Config(Wheel)])
      result = pax_fiddle.build(cfg)
      self.assertDagEqual(result, [pax_fiddle.Config(Wheel)])

  @parameterized.named_parameters([
      ("_dataclass", WheelFactory),
      ("_regular_class", NonDataclassWheelFactory),
      ("_named_tuple", NamedTupleWheelFactory),
  ])
  def test_do_not_build_type_args_if_arg_is_pax_config_container(self, cls):
    wheel_tpl = [pax_fiddle.Config(Wheel), pax_fiddle.Config(Wheel, radius=8)]
    cfg = pax_fiddle.Config(cls, wheel_tpl=wheel_tpl)
    instance = pax_fiddle.build(cfg)
    self.assertDagEqual(instance.wheel_tpl, wheel_tpl)

  def test_do_build_default_factory_list(self):
    cfg = pax_fiddle.Config(WheelFactory)
    factory = pax_fiddle.build(cfg)
    self.assertEqual(factory.wheel_tpl, [])


class LayerE(base_layer.BaseLayer):
  tpl: pax_fiddle.Config[LayerD] = base_layer.template_field(LayerD)


class DaglishTest(testing.TestCase, parameterized.TestCase):

  def test_hparams_with_fiddle_subconfig(self):
    config = pax_fiddle.Config(LayerE)

    all_sub_values = {
        daglish.path_str(path): value
        for value, path in pax_fiddle.iterate(config)
    }
    self.assertDictEqual(
        all_sub_values,
        {
            "": config,
            ".activation_split_dims_mapping": pax_fiddle.Config(
                base_layer.BaseLayer.ActivationSharding, out=None
            ),
            ".params_init": pax_fiddle.Config(
                base_layer.WeightInit, method="xavier", scale=1.000001
            ),
            ".params_init.method": "xavier",
            ".params_init.scale": 1.000001,
            ".tpl": config.tpl,
            ".tpl.activation_split_dims_mapping": (
                config.tpl.activation_split_dims_mapping
            ),
            ".tpl.params_init": config.tpl.params_init,
            ".tpl.weight_split_dims_mapping": (
                config.tpl.weight_split_dims_mapping
            ),
            ".weight_split_dims_mapping": pax_fiddle.Config(
                base_layer.BaseLayer.WeightSharding, wt=None
            ),
        },
    )

  def test_non_memoized_iterate(self):
    config = pax_fiddle.Config(LayerE)
    fdl.materialize_defaults(config)
    paths = [
        daglish.path_str(path)
        for _, path in pax_fiddle.iterate(config, memoized=False)
    ]

    # These were absent above because their values were memoized.
    self.assertIn(".weight_split_dims_mapping.wt", paths)
    self.assertIn(".tpl.params_init.scale", paths)

  def test_doesnt_recurse_generic_dataclass(self):
    @dataclasses.dataclass(frozen=True)
    class MyDataclass:
      a: int = 4
      b: int = 5

    value = MyDataclass(6, 7)
    self.assertLen(list(pax_fiddle.iterate(value)), 1)

  @parameterized.parameters(
      pax_fiddle.BasicTraversal, pax_fiddle.MemoizedTraversal
  )
  def test_replacement(self, traversal_cls):
    config = pax_fiddle.Config(LayerE)
    fdl.materialize_defaults(config)

    def traverse(value, state: daglish.State):
      if value == jnp.float32:
        return jnp.bfloat16
      else:
        return state.map_children(value)

    replaced_config = traversal_cls.run(traverse, config)
    self.assertEqual(replaced_config.dtype, jnp.bfloat16)


@dataclasses.dataclass
class SubclassWithDefaultFactory:
  w: Any = pax_fiddle.field(default_factory=nested_structure)


@pax_fiddle.auto_config
def subclass_fn():
  return SubclassWithDefaultFactory()


@dataclasses.dataclass
class ClassWithDefaultFactory:
  x: Any = pax_fiddle.field(default_factory=subclass_fn)
  z: int = 4


@dataclasses.dataclass(frozen=True)
class FakeLayer:
  name: str


@dataclasses.dataclass(frozen=True)
class FakeEncoderDecoder:
  encoder: FakeLayer
  decoder: FakeLayer


@pax_fiddle.auto_config
def fake_encoder_decoder_fixture():
  return FakeEncoderDecoder(FakeLayer("encoder"), FakeLayer("decoder"))


@dataclasses.dataclass
class ClassWithEncoderDecoder:
  other: Any
  encoder_decoder: FakeEncoderDecoder = pax_fiddle.field(
      default_factory=fake_encoder_decoder_fixture
  )


@dataclasses.dataclass
class ClassWithTwoEncoderDecoders:
  encoder_decoder_a: FakeEncoderDecoder = pax_fiddle.field(
      default_factory=fake_encoder_decoder_fixture
  )
  encoder_decoder_b: FakeEncoderDecoder = pax_fiddle.field(
      default_factory=fake_encoder_decoder_fixture
  )


class TrimDefaultsTest(testing.TestCase):

  def test_trim_deep_defaults(self):
    config = pax_fiddle.Config(ClassWithDefaultFactory)
    config.z = 10
    self.assertIn("x", config.__arguments__)
    self.assertIn("z", config.__arguments__)

    trimmed_without_removing_deep_defaults = visualize.with_defaults_trimmed(
        config
    )
    self.assertIn("x", trimmed_without_removing_deep_defaults.__arguments__)
    self.assertIn("z", trimmed_without_removing_deep_defaults.__arguments__)

    trimmed_removing_deep_defaults = visualize.with_defaults_trimmed(
        config, remove_deep_defaults=True
    )
    self.assertNotIn("x", trimmed_removing_deep_defaults.__arguments__)
    self.assertIn("z", trimmed_removing_deep_defaults.__arguments__)

  def test_trim_deep_defaults_complex_sub_configs(self):
    config = pax_fiddle.Config(ClassWithEncoderDecoder)
    self.assertIn("encoder_decoder", config.__arguments__)
    trimmed_removing_deep_defaults = visualize.with_defaults_trimmed(
        config, remove_deep_defaults=True
    )
    self.assertNotIn(
        "encoder_decoder", trimmed_removing_deep_defaults.__arguments__
    )

  def test_trim_deep_defaults_doesnt_trim_shared(self):
    config = pax_fiddle.Config(ClassWithEncoderDecoder)
    self.assertIn("encoder_decoder", config.__arguments__)
    config.other = config.encoder_decoder.encoder

    trimmed_removing_deep_defaults = visualize.with_defaults_trimmed(
        config, remove_deep_defaults=True
    )
    self.assertIn(
        "encoder_decoder", trimmed_removing_deep_defaults.__arguments__
    )

  def test_trim_deep_defaults_doesnt_trim_shared_2(self):
    """Another shared example, double-referencing the default attr."""
    config = pax_fiddle.Config(ClassWithEncoderDecoder)
    self.assertIn("encoder_decoder", config.__arguments__)
    config.other = config.encoder_decoder

    trimmed_removing_deep_defaults = visualize.with_defaults_trimmed(
        config, remove_deep_defaults=True
    )
    self.assertIn(
        "encoder_decoder", trimmed_removing_deep_defaults.__arguments__
    )

  def test_trim_deep_defaults_doesnt_trim_shared_3(self):
    """Another shared example, with default attrs force-shared."""
    config = pax_fiddle.Config(ClassWithTwoEncoderDecoders)
    self.assertIn("encoder_decoder_a", config.__arguments__)
    self.assertIn("encoder_decoder_b", config.__arguments__)
    config.encoder_decoder_b = config.encoder_decoder_a

    trimmed_removing_deep_defaults = visualize.with_defaults_trimmed(
        config, remove_deep_defaults=True
    )
    self.assertIn(
        "encoder_decoder_a", trimmed_removing_deep_defaults.__arguments__
    )
    self.assertIn(
        "encoder_decoder_b", trimmed_removing_deep_defaults.__arguments__
    )

  def test_trim_deep_defaults_trims_two_encoder_decoders(self):
    """Mostly a check that we set up the above test correctly."""
    config = pax_fiddle.Config(ClassWithTwoEncoderDecoders)
    self.assertIn("encoder_decoder_a", config.__arguments__)
    self.assertIn("encoder_decoder_b", config.__arguments__)

    trimmed_removing_deep_defaults = visualize.with_defaults_trimmed(
        config, remove_deep_defaults=True
    )
    self.assertNotIn(
        "encoder_decoder_a", trimmed_removing_deep_defaults.__arguments__
    )
    self.assertNotIn(
        "encoder_decoder_b", trimmed_removing_deep_defaults.__arguments__
    )

  def test_trim_deep_defaults_ok_with_immutable(self):
    config = pax_fiddle.Config(ClassWithDefaultFactory)
    # References the (2, 3) tuple in `nested_structure()`.
    config.z = fdl.Config(lambda item: item[0], config.x.w["foo"][1])
    self.assertIn("x", config.__arguments__)
    trimmed_removing_deep_defaults = visualize.with_defaults_trimmed(
        config, remove_deep_defaults=True
    )
    self.assertNotIn("x", trimmed_removing_deep_defaults.__arguments__)


if __name__ == "__main__":
  absltest.main()
