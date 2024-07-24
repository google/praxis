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
#   Praxis layers. The public API is defined in __init__.py.

load("//praxis:build-visibility.bzl", "JAX_VISIBILITY")
load("//praxis:praxis.bzl", "pytype_strict_library", "pytype_strict_test")

package(default_visibility = JAX_VISIBILITY)

licenses(["notice"])

pytype_strict_library(
    name = "fp8_nvidia_gpu",
    srcs = ["fp8_nvidia_gpu.py"],
    deps = [
        # Implicit flax.core dependency.
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:pytypes",
        "//praxis/layers",
    ],
)

pytype_strict_test(
    name = "fp8_nvidia_gpu_test",
    srcs = ["fp8_nvidia_gpu_test.py"],
    deps = [
        ":fp8_nvidia_gpu",
        # Implicit absl.testing.absltest dependency.
        # Implicit flax.core dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:test_utils",
        "//praxis/layers:linears",
        "//praxis/layers:pipeline",
    ],
)
