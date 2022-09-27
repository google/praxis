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

"""Declares Fiddle tags for Praxis.

Fiddle tags are essentially named collections of configuration values. They can
diverge, but can also all be set at once to the same value.
"""

import fiddle as fdl


class BaseDType(fdl.Tag):
  """Base JAX DType.

  Usually you want ParamsDType or ActivationDType, but we provide a base class
  here in case the user wants to quickly set all DTypes.
  """


class ParamsDType(BaseDType):
  """DType for parameters."""


class ActivationDType(BaseDType):
  """DType for parameters."""


class WeightInit(fdl.Tag):
  """Weight initializer class.

  Tagged values should generally be base_layer.WeightInit.
  """


class ParamsSplitDimsMapping(fdl.Tag):
  """SplitDimsMapping for parameters."""


class ActivationSplitDimsMapping(fdl.Tag):
  """SplitDimsMapping for activations."""


class DropoutRate(fdl.Tag):
  """Tag for dropout rates."""
