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

# Description:
#   Streaming related layers. The public API is defined in __init__.py.

load("//praxis:praxis.bzl", "py_strict_test")
load("//praxis:praxis.bzl", "pytype_strict_library")
load("//praxis:build-visibility.bzl", "JAX_VISIBILITY")

licenses(["notice"])

package(default_visibility = JAX_VISIBILITY)

pytype_strict_library(
    name = "streaming",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":attentions",
        ":convolutions",
        ":streaming_base",
    ],
)

pytype_strict_library(
    name = "streaming_base",
    srcs = ["streaming_base.py"],
    srcs_version = "PY3",
    deps = [
        # Implicit flax.core dependency.
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "operations",
    srcs = ["operations.py"],
    srcs_version = "PY3",
    deps = [
        ":streaming_base",
        # Implicit flax.core dependency.
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "convolutions",
    srcs = ["convolutions.py"],
    srcs_version = "PY3",
    deps = [
        ":streaming_base",
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:pytypes",
        "//praxis/layers:activations",
        "//praxis/layers:convolutions",
        "//praxis/layers:linears",
        "//praxis/layers:normalizations",
    ],
)

py_strict_test(
    name = "convolutions_test",
    srcs = ["convolutions_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":convolutions",
        ":operations",
        ":streaming",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:test_utils",
        "//praxis/layers:convolutions",
    ],
)

pytype_strict_library(
    name = "attentions",
    srcs = ["attentions.py"],
    srcs_version = "PY3",
    deps = [
        ":streaming_base",
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:pytypes",
        "//praxis/layers:attentions",
    ],
)

py_strict_test(
    name = "attentions_test",
    srcs = ["attentions_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":attentions",
        ":operations",
        ":streaming",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:test_utils",
        "//praxis/layers:attentions",
    ],
)
