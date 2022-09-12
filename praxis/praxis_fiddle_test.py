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

"""Tests for praxis_fiddle."""

import dataclasses
from typing import Optional, List
from absl.testing import absltest

import fiddle as fdl
from fiddle import testing
from praxis import praxis_fiddle


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
  wheel_tpl: fdl.Config[Wheel] = praxis_fiddle.template_field(Wheel)
  num_wheels: int = 4
  owner: Person = praxis_fiddle.sub_field(Person)
  wheels: Optional[List[Wheel]] = None  # Initialized by setup.

  def setup(self):
    assert self.wheels is None
    self.wheels = [
        praxis_fiddle.build(self.wheel_tpl).setup()
        for i in range(self.num_wheels)
    ]
    return self


@dataclasses.dataclass
class Fleet:
  vehicle_tpl: fdl.Config[Vehicle] = praxis_fiddle.template_field(Vehicle)
  num_vehicles: int = 1
  manager: Person = praxis_fiddle.sub_field(Person)
  vehicles: Optional[List[Vehicle]] = None  # Initialized by setup.

  def setup(self):
    assert self.vehicles is None
    self.vehicles = [
        praxis_fiddle.build(self.vehicle_tpl).setup()
        for i in range(self.num_vehicles)
    ]
    return self


class SubFieldAndTemplateFieldTest(testing.TestCase):

  def test_default_fleet_config(self):
    config = fdl.Config(Fleet)
    with self.subTest("expected_config"):
      self.assertDagEqual(
          config,
          fdl.Config(
              Fleet,
              vehicle_tpl=fdl.Config(
                  Vehicle,
                  wheel_tpl=fdl.Config(Wheel),
                  owner=fdl.Config(Person)),
              manager=fdl.Config(Person)))

    with self.subTest("with_materialized_defaults"):
      fdl.materialize_defaults(config)
      self.assertDagEqual(
          config,
          fdl.Config(
              Fleet,
              vehicle_tpl=fdl.Config(
                  Vehicle,
                  wheel_tpl=fdl.Config(Wheel, radius=5),
                  owner=fdl.Config(Person, name=None),
                  num_wheels=4,
                  wheels=None),
              num_vehicles=1,
              manager=fdl.Config(Person, name=None),
              vehicles=None))

  def test_build_default_fleet_config(self):
    config = fdl.Config(Fleet)
    config.manager.name = "Ben"  # Required arg
    fleet = praxis_fiddle.build(config)
    self.assertEqual(
        fleet,
        Fleet(
            vehicle_tpl=fdl.Config(Vehicle),
            num_vehicles=1,
            manager=Person("Ben")))

  def test_build_custom_fleet_config(self):
    config = fdl.Config(Fleet)
    config.manager.name = "Ben"
    config.num_vehicles = 3
    config.vehicle_tpl.wheel_tpl.radius *= 2
    config.vehicle_tpl.owner = config.manager
    fleet = praxis_fiddle.build(config)
    self.assertEqual(
        fleet,
        Fleet(
            vehicle_tpl=fdl.Config(
                Vehicle,
                wheel_tpl=fdl.Config(Wheel, radius=10),
                owner=fdl.Config(Person, name="Ben")),
            num_vehicles=3,
            manager=Person("Ben")))

  def test_build_and_setup_default_fleet_config(self):
    config = fdl.Config(Fleet)
    config.manager.name = "Ben"  # Required arg
    config.vehicle_tpl.owner.name = "Joe"  # Required arg
    fleet = praxis_fiddle.build(config).setup()
    self.assertEqual(
        fleet,
        Fleet(
            vehicle_tpl=fdl.Config(
                Vehicle, owner=fdl.Config(Person, name="Joe")),
            vehicles=[
                Vehicle(
                    wheel_tpl=fdl.Config(Wheel),
                    wheels=[Wheel(), Wheel(),
                            Wheel(), Wheel()],
                    num_wheels=4,
                    owner=Person("Joe"))
            ],
            num_vehicles=1,
            manager=Person("Ben")))

  def test_build_and_setup_custom_fleet_config(self):
    config = fdl.Config(Fleet)

    config.manager.name = "Ben"
    config.num_vehicles = 3
    config.vehicle_tpl.wheel_tpl.radius *= 2
    config.vehicle_tpl.owner.name = "Joe"

    with self.subTest("got_expected_fleet"):
      fleet = praxis_fiddle.build(config).setup()
      self.assertEqual(
          fleet,
          Fleet(
              vehicle_tpl=fdl.Config(
                  Vehicle,
                  owner=fdl.Config(Person, name="Joe"),
                  wheel_tpl=fdl.Config(Wheel, radius=10)),
              vehicles=3 * [
                  Vehicle(
                      wheel_tpl=fdl.Config(Wheel, radius=10),
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
    config = fdl.Config(Fleet)

    config.manager.name = "Ben"
    config.num_vehicles = 3
    config.vehicle_tpl.owner = config.manager  # shared config object.

    fleet = praxis_fiddle.build(config).setup()
    self.assertEqual(
        fleet,
        Fleet(
            vehicle_tpl=fdl.Config(
                Vehicle, owner=fdl.Config(Person, name="Ben")),
            vehicles=3 * [
                Vehicle(
                    wheel_tpl=fdl.Config(Wheel),
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
    # praxis_fiddle.build.  This will change when we transition to using factory
    # objects (partials) instead of config objects to store templates.
    self.assertIsNot(fleet.vehicles[0].owner, fleet.vehicles[1].owner)
    self.assertIsNot(fleet.vehicles[0].owner, fleet.vehicles[2].owner)
    self.assertIsNot(fleet.vehicles[1].owner, fleet.vehicles[2].owner)
    self.assertIsNot(fleet.manager, fleet.vehicles[0].owner)

  def test_use_custom_wheel(self):
    config = fdl.Config(Fleet)
    config.vehicle_tpl.wheel_tpl = fdl.Config(ColoredWheel, color="red")
    fleet = fdl.build(config).setup()
    print("FFF", fleet)
    self.assertIsInstance(fleet.vehicles[0].wheels[0], ColoredWheel)
    self.assertEqual(fleet.vehicles[0].wheels[0],
                     ColoredWheel(radius=5, color="red"))

  def test_build_fleet_directly(self):
    fleet = Fleet()
    fleet = fleet.setup()
    self.assertEqual(
        fleet,
        Fleet(
            vehicle_tpl=fdl.Config(Vehicle),
            num_vehicles=1,
            vehicles=[
                Vehicle(
                    wheel_tpl=fdl.Config(Wheel),
                    num_wheels=4,
                    wheels=[Wheel(), Wheel(),
                            Wheel(), Wheel()],
                    owner=Person())
            ],
            manager=Person()))

    self.assertEqual(fleet.vehicle_tpl, fdl.Config(Vehicle))
    self.assertEqual(fleet.vehicles[0].wheel_tpl, fdl.Config(Wheel))


if __name__ == "__main__":
  absltest.main()
