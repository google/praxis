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

# Description:
#   Sparsity related layers. The public API is defined in __init__.py.

load("//praxis:praxis.bzl", "pytype_strict_library")
load("//praxis:praxis.bzl", "py_strict_test")
load("//praxis:build-visibility.bzl", "JAX_VISIBILITY")

package(default_visibility = JAX_VISIBILITY)

pytype_strict_library(
    name = "layers",
    srcs = ["__init__.py"],
    deps = [
        ":attentions",
        ":linears",
    ],
)

pytype_strict_library(
    name = "sparsity_hparams",
    srcs = ["sparsity_hparams.py"],
)

pytype_strict_library(
    name = "linears",
    srcs = ["linears.py"],
    deps = [
        ":sparse_base_layer",
        ":sparsity",
        ":sparsity_hparams",
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:pytypes",
        "//praxis/layers:linears",
    ],
)

pytype_strict_library(
    name = "sparsity",
    srcs = ["sparsity.py"],
    deps = [
        ":sparsity_hparams",
        # Implicit absl.logging dependency.
        # Implicit flax.core dependency.
        # Implicit jax dependency.
    ],
)

py_strict_test(
    name = "sparsity_test",
    srcs = ["sparsity_test.py"],
    deps = [
        ":sparsity",
        ":sparsity_hparams",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit flax.core dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
    ],
)

py_strict_test(
    name = "linears_test",
    srcs = ["linears_test.py"],
    deps = [
        ":linears",
        ":sparsity_hparams",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:test_utils",
        "//praxis/layers:linears",
    ],
)

pytype_strict_library(
    name = "attentions",
    srcs = ["attentions.py"],
    deps = [
        ":sparse_base_layer",
        ":sparsity",
        ":sparsity_hparams",
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:pytypes",
        "//praxis/layers:attentions",
    ],
)

py_strict_test(
    name = "attentions_test",
    srcs = ["attentions_test.py"],
    deps = [
        ":attentions",
        ":sparsity_hparams",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:test_utils",
        "//praxis/layers:attentions",
    ],
)

pytype_strict_library(
    name = "sparse_base_layer",
    srcs = ["sparse_base_layer.py"],
    deps = [
        ":sparsity",
        ":sparsity_hparams",
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:pytypes",
    ],
)
