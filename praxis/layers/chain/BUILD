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

# Description: Chain and utilities.
# The public API is defined in __init__.py.

load("//praxis:praxis.bzl", "py_strict_test")
load("//praxis:praxis.bzl", "pytype_strict_library")
load("//praxis:build-visibility.bzl", "JAX_VISIBILITY")

package(default_visibility = JAX_VISIBILITY)

pytype_strict_library(
    name = "chain",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":chain_extensions",
        ":chain_lib",
    ],
)

pytype_strict_library(
    name = "chain_lib",
    srcs = ["chain.py"],
    srcs_version = "PY3",
    deps = ["//praxis:base_layer"],
)

pytype_strict_library(
    name = "chain_extensions",
    srcs = ["chain_extensions.py"],
    deps = [
        ":chain_lib",
        # Implicit absl.logging dependency.
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis/layers:activations",
        "//praxis/layers:linears",
        "//praxis/layers:repeats",
    ],
)

py_strict_test(
    name = "chain_test",
    srcs = ["chain_test.py"],
    deps = [
        ":chain",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_hyperparams",
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:test_utils",
        "//praxis/layers:activations",
    ],
)
