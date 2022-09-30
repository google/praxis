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
        ":conformers",
        ":convolutions",
        ":embedding_softmax",
        ":normalizations",
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
    name = "test_utils",
    testonly = True,
    srcs = ["test_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":streaming_base",
        # Implicit flax.core dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:pytypes",
        "//praxis:test_utils",
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
    shard_count = 8,
    srcs_version = "PY3",
    deps = [
        ":convolutions",
        ":streaming",
        ":test_utils",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:py_utils",
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
        ":streaming",
        ":test_utils",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis/layers:attentions",
    ],
)

pytype_strict_library(
    name = "normalizations",
    srcs = ["normalizations.py"],
    srcs_version = "PY3",
    deps = [
        ":streaming_base",
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:pytypes",
        "//praxis/layers:normalizations",
    ],
)

py_strict_test(
    name = "normalizations_test",
    srcs = ["normalizations_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":normalizations",
        ":streaming",
        ":test_utils",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis/layers:normalizations",
    ],
)

pytype_strict_library(
    name = "conformers",
    srcs = ["conformers.py"],
    srcs_version = "PY3",
    deps = [
        ":attentions",
        ":convolutions",
        ":streaming_base",
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:pytypes",
        "//praxis/layers:attentions",
        "//praxis/layers:conformers",
        "//praxis/layers:convolutions",
    ],
)

py_strict_test(
    name = "conformers_test",
    srcs = ["conformers_test.py"],
    python_version = "PY3",
    shard_count = 6,
    srcs_version = "PY3",
    deps = [
        ":conformers",
        ":streaming",
        ":test_utils",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis/layers:attentions",
        "//praxis/layers:conformers",
        "//praxis/layers:convolutions",
    ],
)

pytype_strict_library(
    name = "embedding_softmax",
    srcs = ["embedding_softmax.py"],
    srcs_version = "PY3",
    deps = [
        ":streaming_base",
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:pytypes",
        "//praxis/layers:embedding_softmax",
    ],
)

py_strict_test(
    name = "embedding_softmax_test",
    srcs = ["embedding_softmax_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":embedding_softmax",
        ":streaming",
        ":test_utils",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis/layers:embedding_softmax",
    ],
)
