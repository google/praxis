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

load("//praxis:build-visibility.bzl", "JAX_VISIBILITY")

# Description:
#   video related layers. The public API is defined in __init__.py.
load("//praxis:praxis.bzl", "pytype_strict_library", "pytype_strict_test")

package(default_visibility = JAX_VISIBILITY)

pytype_strict_library(
    name = "enc_dec_3dcnn",
    srcs = [
        "enc_dec_3dcnn.py",
    ],
    srcs_version = "PY3",
    deps = [
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:pytypes",
        "//praxis/layers:activations",
        "//praxis/layers:convolutions",
        "//praxis/layers:linears",
        "//praxis/layers:normalizations",
    ],
)

pytype_strict_test(
    name = "enc_dec_3dcnn_test",
    srcs = [
        "enc_dec_3dcnn_test.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":enc_dec_3dcnn",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:test_utils",
    ],
)

pytype_strict_library(
    name = "quantizer",
    srcs = [
        "quantizer.py",
    ],
    srcs_version = "PY3",
    deps = [
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "vqvae",
    srcs = [
        "vqvae.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":enc_dec_3dcnn",
        ":quantizer",
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:base_model",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:pytypes",
        "//praxis/layers:activations",
    ],
)
