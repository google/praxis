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
#   Quantization related layers. The public API is defined in __init__.py.

load("//praxis:praxis.bzl", "py_strict_test")
load("//praxis:praxis.bzl", "pytype_strict_library")
load("//praxis:build-visibility.bzl", "JAX_VISIBILITY")

licenses(["notice"])

package(default_visibility = JAX_VISIBILITY)

pytype_strict_library(
    name = "layers",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":attentions",
        ":linears",
        ":operations",
    ],
)

pytype_strict_library(
    name = "attentions",
    srcs = ["attentions.py"],
    srcs_version = "PY3",
    deps = [
        ":operations",
        # Implicit flax.core dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:asserts",
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
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit lingvo.core.batch_major_attention dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:decoder_utils",
        "//praxis:py_utils",
        "//praxis:test_utils",
        "//praxis/layers:attentions",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_library(
    name = "linears",
    srcs = ["linears.py"],
    srcs_version = "PY3",
    deps = [
        ":aqt",
        ":operations",
        # Implicit flax.core dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:asserts",
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:pytypes",
        "//praxis/layers:linears",
    ],
)

py_strict_test(
    name = "linears_test",
    srcs = ["linears_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":linears",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit lingvo.core.batch_major_attention dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:decoder_utils",
        "//praxis:py_utils",
        "//praxis:test_utils",
        "//praxis/layers:linears",
        "//praxis/layers:repeats",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_library(
    name = "operations",
    srcs = ["operations.py"],
    srcs_version = "PY3",
    deps = [
        ":aqt",
        # Implicit flax.core dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:asserts",
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

py_strict_test(
    name = "operations_test",
    srcs = ["operations_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":aqt",
        ":attentions",
        ":linears",
        ":operations",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:test_utils",
    ],
)

py_strict_test(
    name = "quantization_test",
    srcs = ["quantization_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":linears",
        ":operations",
        # Implicit absl.testing.absltest dependency.
        # Implicit jax dependency.
        # Implicit lingvo.core.batch_major_attention dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:decoder_utils",
        "//praxis:py_utils",
        "//praxis:test_utils",
        "//praxis/layers:linears",
        "//praxis/layers:repeats",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_library(
    name = "aqt",
    srcs = ["aqt.py"],
    srcs_version = "PY3",
    deps = [
        # Implicit jax dependency.
        "//praxis:base_hyperparams",
        "//praxis:base_layer",
        "//praxis:pytypes",
    ],
)

py_strict_test(
    name = "aqt_test",
    srcs = ["aqt_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":aqt",
        # Implicit absl.testing.absltest dependency.
        # Implicit jax dependency.
        "//praxis:pytypes",
        "//praxis:test_utils",
    ],
)
