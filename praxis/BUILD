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
#   Praxis: a set of modules for training machine learning models in Jax.

# Internal license-related bzl load
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("//praxis:build-visibility.bzl", "JAX_VISIBILITY")
load("//praxis:praxis.bzl", "pytype_strict_test")
load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")

package(default_visibility = JAX_VISIBILITY)

licenses(["notice"])

exports_files(["LICENSE"])

py_library(
    name = "asserts",
    srcs = ["asserts.py"],
    type_checking = True,
    deps = [
        ":py_utils",
        # Implicit jax dependency.
    ],
)

py_library(
    name = "base_hyperparams",
    srcs = ["base_hyperparams.py"],
    type_checking = True,
    deps = [
        ":lazy_loader",
        ":pax_fiddle",
        ":py_utils",
        # Implicit python proto dependency.
        # Implicit absl.logging dependency.
        # Implicit fiddle dependency.
        # Implicit flax.core dependency.
        # Implicit ml_collections config_dict dependency.
        # Implicit numpy dependency.
        # Implicit typing_extensions dependency.
    ],
)

py_library(
    name = "base_input",
    srcs = ["base_input.py"],
    type_checking = True,
    deps = [
        ":base_hyperparams",
        ":lazy_loader",
        ":lingvo_lib",
        ":pax_fiddle",
        ":py_utils",
        ":pytypes",
        # Implicit absl.logging dependency.
        # Implicit etils dependency.
        # Implicit fiddle dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        # Implicit tensorflow_no_contrib dependency.
    ],
)

py_library(
    name = "base_layer",
    srcs = ["base_layer.py"],
    type_checking = True,
    deps = [
        ":asserts",
        ":base_hyperparams",
        ":pax_fiddle",
        ":py_utils",
        ":pytypes",
        # Implicit absl.flags dependency.
        # Implicit absl.logging dependency.
        # Implicit fiddle dependency.
        # Implicit fiddle.daglish dependency.
        # Implicit flax.core dependency.
        # Implicit jax dependency.
        "//third_party/py/jax/extend:core",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "base_model",
    srcs = ["base_model.py"],
    type_checking = True,
    deps = [
        ":base_input",
        ":base_layer",
        ":pax_fiddle",
        ":py_utils",
        ":pytypes",
    ],
)

py_library(
    name = "beam_search",
    srcs = ["beam_search.py"],
    type_checking = True,
    deps = [
        ":base_layer",
        ":decoder_hparams",
        ":decoder_utils",
        ":py_utils",
        # Implicit flax.core dependency.
        # Implicit jax dependency.
    ],
)

py_library(
    name = "decoder_hparams",
    srcs = ["decoder_hparams.py"],
    type_checking = True,
    deps = [
        ":base_hyperparams",
        ":decoder_utils",
        ":pax_fiddle",
        ":pytypes",
        ":sample_decode",
    ],
)

py_library(
    name = "decoder_utils",
    srcs = ["decoder_utils.py"],
    type_checking = True,
    deps = [
        ":base_layer",
        ":py_utils",
        ":pytypes",
        # Implicit flax.core dependency.
        # Implicit jax dependency.
    ],
)

py_library(
    name = "flat_beam_search",
    srcs = ["flat_beam_search.py"],
    type_checking = True,
    deps = [
        ":base_layer",
        ":decoder_utils",
        ":py_utils",
        # Implicit flax.core dependency.
        # Implicit jax dependency.
    ],
)

py_library(
    name = "flax_utils",
    srcs = ["flax_utils.py"],
    type_checking = True,
    deps = [
        ":base_layer",
        ":pytypes",
        # Implicit flax.core dependency.
        # Implicit jax dependency.
    ],
)

py_library(
    name = "gshard_utils",
    srcs = ["gshard_utils.py"],
    type_checking = True,
    deps = [
        ":pytypes",
        # Implicit absl.logging dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "lazy_loader",
    srcs = ["lazy_loader.py"],
    type_checking = True,
    deps = [
        # Implicit absl.logging dependency.
    ],
)

py_library(
    name = "lingvo_lib",
    srcs = ["lingvo_lib.py"],
    type_checking = True,
    deps = [
        ":lazy_loader",
        # Implicit lingvo.core.cluster dependency.
        # Implicit lingvo.core.cluster_factory dependency.
        # Implicit lingvo.core.datasource dependency.
        # Implicit lingvo.core.hyperparams dependency.
        # Implicit lingvo.core.nested_map dependency.
    ],
)

py_library(
    name = "metric_utils",
    srcs = ["metric_utils.py"],
    type_checking = True,
    deps = [
        # Implicit jax dependency.
    ],
)

py_library(
    name = "optimizer_prefix_vectorization",
    srcs = ["optimizer_prefix_vectorization.py"],
    type_checking = True,
    deps = [
        ":base_layer",
        ":optimizers",
        ":py_utils",
        ":pytypes",
        # Implicit jax dependency.
        # Implicit optax dependency.
    ],
)

pytype_strict_test(
    name = "optimizer_prefix_vectorization_test",
    srcs = ["optimizer_prefix_vectorization_test.py"],
    deps = [
        ":base_layer",
        ":optimizer_prefix_vectorization",
        ":test_utils",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit optax dependency.
    ],
)

py_library(
    name = "optimizers",
    srcs = ["optimizers.py"],
    type_checking = True,
    deps = [
        ":asserts",
        ":base_hyperparams",
        ":base_layer",
        ":gshard_utils",
        ":pax_fiddle",
        ":py_utils",
        ":pytypes",
        ":schedules",
        # Implicit optax_shampoo dependency.
        # Implicit absl.logging dependency.
        # Implicit jax dependency.
        # Implicit optax dependency.
    ],
)

py_library(
    name = "py_utils",
    srcs = ["py_utils.py"],
    type_checking = True,
    deps = [
        ":lingvo_lib",
        ":pytypes",
        ":trees",
        # Implicit absl.flags dependency.
        # Implicit absl.logging dependency.
        # Implicit flax.core dependency.
        # Implicit jax dependency.
        # Implicit jax.mesh_utils dependency.
        # Implicit numpy dependency.
        # Implicit optax dependency.
    ],
)

py_library(
    name = "pytypes",
    srcs = ["pytypes.py"],
    type_checking = True,
    deps = [
        ":lingvo_lib",
        # Implicit absl.logging dependency.
        # Implicit clu.metrics dependency.
        # Implicit jax dependency.
        # Implicit jaxtyping dependency.
        # Implicit numpy dependency.
        # Implicit typeguard dependency.
    ],
)

pytype_strict_test(
    name = "pytypes_test",
    srcs = ["pytypes_test.py"],
    deps = [
        ":base_layer",
        ":pytypes",
        ":test_utils",
        ":trees",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit flax.core dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "sample_decode",
    srcs = ["sample_decode.py"],
    type_checking = True,
    deps = [
        ":asserts",
        ":base_hyperparams",
        ":base_layer",
        ":decoder_utils",
        ":pax_fiddle",
        ":py_utils",
        ":token_samplers",
        # Implicit flax.core dependency.
        # Implicit jax dependency.
    ],
)

py_library(
    name = "schedules",
    srcs = ["schedules.py"],
    type_checking = True,
    deps = [
        ":base_hyperparams",
        ":pax_fiddle",
        # Implicit jax dependency.
        # Implicit optax dependency.
    ],
)

py_library(
    name = "token_samplers",
    srcs = ["token_samplers.py"],
    type_checking = True,
    deps = [
        ":base_hyperparams",
        ":base_layer",
        ":decoder_utils",
        ":pax_fiddle",
        ":py_utils",
        ":pytypes",
        # Implicit jax dependency.
    ],
)

py_library(
    name = "test_utils",
    testonly = True,
    srcs = ["test_utils.py"],
    type_checking = True,
    deps = [
        ":base_layer",
        ":py_utils",
        ":pytypes",
        # Implicit absl.flags dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit etils dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        # Implicit tensorflow_no_contrib dependency.
    ],
)

py_library(
    name = "pax_fiddle",
    srcs = ["pax_fiddle.py"],
    type_checking = True,
    deps = [
        # Implicit fiddle dependency.
        # Implicit fiddle.building dependency.
        # Implicit fiddle.daglish dependency.
        # Implicit fiddle.history dependency.
        # Implicit fiddle.signatures dependency.
        # Implicit fiddle.experimental.auto_config dependency.
        # Implicit fiddle.extensions.jax dependency.
        # Implicit flax.core dependency.
        # Implicit lingvo.core.nested_map dependency.
        # Implicit typing_extensions dependency.
    ],
)

py_library(
    name = "fiddle_tags",
    srcs = ["fiddle_tags.py"],
    type_checking = True,
    deps = [
        # Implicit fiddle dependency.
    ],
)

pytype_strict_test(
    name = "asserts_test",
    srcs = ["asserts_test.py"],
    deps = [
        ":asserts",
        ":py_utils",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit upb python proto dependency.
        # Implicit jax.experimental.jax2tf dependency.
        # Implicit numpy dependency.
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_test(
    name = "base_hyperparams_test",
    srcs = ["base_hyperparams_test.py"],
    tags = ["fast_and_reliable_fiddle_integration_test"],
    deps = [
        ":base_hyperparams",
        ":base_layer",
        ":pax_fiddle",
        # Implicit absl.testing.absltest dependency.
        # Implicit fiddle dependency.
        # Implicit flax.core dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit ml_collections config_dict dependency.
        # Implicit numpy dependency.
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_test(
    name = "base_input_test",
    srcs = ["base_input_test.py"],
    deps = [
        ":base_hyperparams",
        ":base_input",
        ":pax_fiddle",
        ":py_utils",
        ":test_utils",
        # Implicit absl.flags dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit fiddle dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit lingvo.core.base_input_generator dependency.
        # Implicit lingvo.core.generic_input dependency.
        # Implicit lingvo.core.py_utils dependency.
        # Implicit numpy dependency.
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_test(
    name = "base_layer_test",
    srcs = ["base_layer_test.py"],
    tags = ["fast_and_reliable_fiddle_integration_test"],
    deps = [
        ":base_hyperparams",
        ":base_layer",
        ":pax_fiddle",
        ":test_utils",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit fiddle dependency.
        # Implicit flax.core dependency.
        # Implicit jax dependency.
    ],
)

pytype_strict_test(
    name = "beam_search_test",
    srcs = ["beam_search_test.py"],
    deps = [
        ":base_layer",
        ":beam_search",
        ":pax_fiddle",
        ":py_utils",
        ":test_utils",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis/layers:models",
        "//praxis/layers:transformer_models",
    ],
)

pytype_strict_test(
    name = "decoder_utils_test",
    srcs = ["decoder_utils_test.py"],
    deps = [
        ":base_model",
        ":decoder_utils",
        ":pytypes",
        ":test_utils",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
    ],
)

pytype_strict_test(
    name = "flat_beam_search_test",
    srcs = ["flat_beam_search_test.py"],
    deps = [
        ":flat_beam_search",
        ":test_utils",
        # Implicit absl.testing.absltest dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
    ],
)

pytype_strict_test(
    name = "optimizers_test",
    srcs = ["optimizers_test.py"],
    deps = [
        ":base_layer",
        ":optimizers",
        ":pax_fiddle",
        ":py_utils",
        ":schedules",
        ":test_utils",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        # Implicit optax dependency.
    ],
)

pytype_strict_test(
    name = "py_utils_test",
    srcs = ["py_utils_test.py"],
    deps = [
        ":base_layer",
        ":py_utils",
        ":test_utils",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit flax.core dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
    ],
)

pytype_strict_test(
    name = "sample_decode_test",
    srcs = ["sample_decode_test.py"],
    deps = [
        ":base_layer",
        ":base_model",
        ":pax_fiddle",
        ":py_utils",
        ":pytypes",
        ":sample_decode",
        ":test_utils",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit flax.core dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
    ],
)

pytype_strict_test(
    name = "schedules_test",
    srcs = ["schedules_test.py"],
    deps = [
        ":schedules",
        ":test_utils",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit lingvo.core.py_utils dependency.
        # Implicit lingvo.core.schedule dependency.
    ],
)

pytype_strict_test(
    name = "token_samplers_test",
    srcs = ["token_samplers_test.py"],
    deps = [
        ":py_utils",
        ":test_utils",
        ":token_samplers",
        # Implicit absl.testing.absltest dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
    ],
)

pytype_strict_test(
    name = "pax_fiddle_test",
    srcs = ["pax_fiddle_test.py"],
    tags = ["fast_and_reliable_fiddle_integration_test"],
    deps = [
        ":base_hyperparams",
        ":base_layer",
        ":pax_fiddle",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit fiddle dependency.
        # Implicit fiddle.daglish dependency.
        # Implicit fiddle.experimental.serialization dependency.
        # Implicit fiddle.experimental.visualize dependency.
        # Implicit fiddle.testing dependency.
        # Implicit flax.core dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit lingvo.core.nested_map dependency.
    ],
)

py_library(
    name = "trees",
    srcs = ["trees.py"],
    type_checking = True,
    deps = [
        ":pytypes",
        # Implicit jax dependency.
        # Implicit jaxtyping dependency.
    ],
)

pytype_strict_test(
    name = "trees_test",
    srcs = ["trees_test.py"],
    deps = [
        ":base_layer",
        ":pytypes",
        ":test_utils",
        ":trees",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit flax.core dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
    ],
)

bzl_library(
    name = "praxis_bzl",
    srcs = ["praxis.bzl"],
    # Implicit build flag
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "build_visibility_bzl",
    srcs = ["build-visibility.bzl"],
    # Implicit build flag
    visibility = ["//visibility:private"],
)
