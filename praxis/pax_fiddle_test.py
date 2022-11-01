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

"""Tests for pax_fiddle."""

import copy
import dataclasses
from typing import Optional, List
from absl.testing import absltest
from absl.testing import parameterized

import fiddle as fdl
from fiddle import testing
from flax import core as flax_core
from flax import linen as nn
import jax
from jax import numpy as jnp
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
  name: Optional[str] = None

  def setup(self):
    return self


@dataclasses.dataclass
class Vehicle:
  wheel_tpl: pax_fiddle.Config[Wheel] = pax_fiddle.template_field(Wheel)
  num_wheels: int = 4
  owner: Person = pax_fiddle.sub_field(Person)
  wheels: Optional[List[Wheel]] = None  # Initialized by setup.

  def setup(self):
    assert self.wheels is None
    self.wheels = [
        pax_fiddle.build(self.wheel_tpl).setup() for i in range(self.num_wheels)
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
  manager: Person = pax_fiddle.sub_field(Person)
  vehicles: Optional[List[Vehicle]] = None  # Initialized by setup.

  def setup(self):
    assert self.vehicles is None
    self.vehicles = [
        pax_fiddle.build(self.vehicle_tpl).setup()
        for i in range(self.num_vehicles)
    ]
    return self


@dataclasses.dataclass
class BusStop:
  location: str  # required arg.
  times: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class HourlyBusStop(BusStop):
  times: list = dataclasses.field(default_factory=lambda: list(range(24)))


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
                  owner=pax_fiddle.Config(Person)),
              manager=pax_fiddle.Config(Person)))

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
                  wheels=None),
              num_vehicles=1,
              manager=pax_fiddle.Config(Person, name=None),
              vehicles=None))

  def test_build_default_fleet_config(self):
    config = pax_fiddle.Config(Fleet)
    config.manager.name = "Ben"  # Required arg
    fleet = pax_fiddle.build(config)
    self.assertEqual(
        fleet,
        Fleet(
            vehicle_tpl=pax_fiddle.Config(Vehicle),
            num_vehicles=1,
            manager=Person("Ben")))

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
                owner=pax_fiddle.Config(Person, name="Ben")),
            num_vehicles=3,
            manager=Person("Ben")))

  def test_build_and_setup_default_fleet_config(self):
    config = pax_fiddle.Config(Fleet)
    config.manager.name = "Ben"  # Required arg
    config.vehicle_tpl.owner.name = "Joe"  # Required arg
    fleet = pax_fiddle.build(config).setup()
    self.assertEqual(
        fleet,
        Fleet(
            vehicle_tpl=pax_fiddle.Config(
                Vehicle, owner=pax_fiddle.Config(Person, name="Joe")),
            vehicles=[
                Vehicle(
                    wheel_tpl=pax_fiddle.Config(Wheel),
                    wheels=[Wheel(), Wheel(),
                            Wheel(), Wheel()],
                    num_wheels=4,
                    owner=Person("Joe"))
            ],
            num_vehicles=1,
            manager=Person("Ben")))

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
                  wheel_tpl=pax_fiddle.Config(Wheel, radius=10)),
              vehicles=3 * [
                  Vehicle(
                      wheel_tpl=pax_fiddle.Config(Wheel, radius=10),
                      wheels=4 * [Wheel(10)],
                      num_wheels=4,
                      owner=Person("Joe"))
              ],
              num_vehicles=3,
              manager=Person("Ben")))

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
                Vehicle, owner=pax_fiddle.Config(Person, name="Ben")),
            vehicles=3 * [
                Vehicle(
                    wheel_tpl=pax_fiddle.Config(Wheel),
                    wheels=4 * [Wheel()],
                    num_wheels=4,
                    owner=Person("Ben"))
            ],
            num_vehicles=3,
            manager=Person("Ben")))

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
    self.assertEqual(fleet.vehicles[0].wheels[0],
                     ColoredWheel(radius=5, color="red"))

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
                    wheels=[Wheel(), Wheel(),
                            Wheel(), Wheel()],
                    owner=Person())
            ],
            manager=Person()))

    self.assertEqual(fleet.vehicle_tpl, pax_fiddle.Config(Vehicle))
    self.assertEqual(fleet.vehicles[0].wheel_tpl, pax_fiddle.Config(Wheel))


class PaxConfigTest(testing.TestCase, parameterized.TestCase):

  def test_cls_property(self):
    cfg = pax_fiddle.Config(
        Vehicle, wheel_tpl=pax_fiddle.Config(Wheel), num_wheels=3)
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
        Vehicle, wheel_tpl=pax_fiddle.Config(Wheel), num_wheels=3)
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
            wheel_tpl=pax_fiddle.Config(Wheel, radius=20)))

  @parameterized.parameters([
      pax_fiddle.build, pax_fiddle.instantiate, base_hyperparams.instantiate,
      pax_fiddle.PaxConfig.Instantiate
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
            owner=Person()))

  @parameterized.parameters([
      pax_fiddle.instantiate, base_hyperparams.instantiate,
      pax_fiddle.PaxConfig.Instantiate
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
            owner=Person("Mo")))

  def test_copy_fields_from(self):
    source = pax_fiddle.Config(Vehicle, num_wheels=2)
    source.wheel_tpl.set(radius=20)
    target = pax_fiddle.Config(Vehicle)
    expected = copy.deepcopy(source)
    target.copy_fields_from(source)
    self.assertEqual(target, expected)

  def test_copy_fields_from_does_not_copy_name(self):
    source = pax_fiddle.Config(Person, "A")
    target = pax_fiddle.Config(Person, "B")
    target.copy_fields_from(source)
    self.assertEqual(target, pax_fiddle.Config(Person, "B"))

  def test_copy_fields_from_missing_fields_in_source(self):
    source = pax_fiddle.Config(Wheel, radius=10)
    target = pax_fiddle.Config(ColoredWheel, radius=3, color="red")
    target.copy_fields_from(source)
    self.assertEqual(target, pax_fiddle.Config(ColoredWheel,
                                               radius=10, color="red"))

  def test_copy_fields_from_missing_fields_in_self(self):
    source = pax_fiddle.Config(ColoredWheel, radius=3, color="red")
    target = pax_fiddle.Config(Wheel, radius=10)

    with self.assertRaisesRegex(
        ValueError, "Copying incompatible HParams: 'color' not in self"):
      target.copy_fields_from(source)

    target.copy_fields_from(source, missing_fields_in_self=["color"])
    self.assertEqual(target, pax_fiddle.Config(Wheel, radius=3))

  def test_copy_fields_from_missing_required_value(self):
    source = pax_fiddle.Config(BusStop)
    target = pax_fiddle.Config(BusStop)
    with self.assertRaisesRegex(
        ValueError, "Can't copy from missing required .*BusStop.location"):
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
        ValueError, "Can't copy from default_factory .*BusStop.times"):
      target.copy_fields_from(source)

  def test_copy_fields_from_invalid_source(self):
    with self.subTest("source_field_not_in_self"):
      source = pax_fiddle.Config(Wheel)
      target = pax_fiddle.Config(Vehicle)
      with self.assertRaisesRegex(
          ValueError, "Copying incompatible HParams: 'radius' not in self"):
        target.copy_fields_from(source)

  def test_mesh_shape(self):
    cfg = pax_fiddle.Config(base_layer.FiddleBaseLayer)
    self.assertIsNone(cfg.mesh_shape)
    cfg.mesh_axis_names = ["a", "b"]
    cfg.ici_mesh_shape = [1, 2]
    self.assertEqual(cfg.mesh_shape, [1, 2])
    cfg.dcn_mesh_shape = [3, 4]
    self.assertEqual(cfg.mesh_shape, [1 * 3, 2 * 4])


class LayerA(nn.Module):
  x: int = 0

  def __call__(self):
    return self.x


class LayerB(nn.Module):
  a: LayerA = pax_fiddle.sub_field(LayerA)

  def __call__(self):
    return self.a()


class LayerC(nn.Module):
  b_tpl: pax_fiddle.Config = pax_fiddle.template_field(LayerB)

  def setup(self):
    self.b = pax_fiddle.build(self.b_tpl)
    if self.b.parent is not self:
      raise AssertionError(
          "Expected self.b.parent to be self (inside self.setup).")
    if self.b.a.parent is not None:
      raise AssertionError(
          "Expected self.b.a.parent to be None (inside self.setup).")

  def __call__(self, x):
    if self.b.parent is not self:
      raise AssertionError(
          "Expected self.b.parent to be self (before b.setup).")
    if self.b.a.parent is not None:
      raise AssertionError(
          "Expected self.b.a.parent to be None (before b.setup).")
    self.b()  # Causes b.setup() to be called.
    if self.b.a.parent is not self.b:
      raise AssertionError(
          "Expected self.b.a.parent to be self.b (after b.setup).")
    return 0


class LayerD(base_layer.FiddleBaseLayer):

  def setup(self):
    self.create_variable(
        "v",
        base_layer.WeightHParams(
            shape=[], init=base_layer.WeightInit.Constant(3)))

  def __call__(self, x):
    return self.theta.v * x


class BuildTest(testing.TestCase, parameterized.TestCase):
  """Tests for pax_fiddle.build."""

  @parameterized.parameters([
      pax_fiddle.build, pax_fiddle.instantiate, base_hyperparams.instantiate,
      pax_fiddle.PaxConfig.Instantiate
  ])
  def test_parent_links_are_not_set_to_outer_scope(self, build_func):
    # This test checks that using the `empty_flax_module_stack` decorator is
    # effective at preventing modules from being built with the wrong parent.
    with self.subTest(build_func.__qualname__):
      cfg = pax_fiddle.Config(LayerC)
      cfg.parent = flax_core.Scope({})
      c = build_func(cfg)
      self.assertIs(c.b.parent, c)
      self.assertIsNone(c.b.a.parent)
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


if __name__ == "__main__":
  absltest.main()
