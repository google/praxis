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

load("//praxis:praxis.bzl", "pytype_strict_contrib_test", "pytype_strict_library")
load("//praxis:praxis.bzl", "py_strict_test")
load("//praxis:build-visibility.bzl", "JAX_VISIBILITY")

package(default_visibility = JAX_VISIBILITY)

licenses(["notice"])

exports_files(["LICENSE"])

pytype_strict_library(
    name = "asserts",
    srcs = ["asserts.py"],
    deps = [
        ":py_utils",
        # Implicit jax dependency.
    ],
)

pytype_strict_library(
    name = "base_hyperparams",
    srcs = ["base_hyperparams.py"],
    srcs_version = "PY3",
    deps = [
        ":pax_fiddle",
        ":py_utils",
        # Implicit python proto dependency.
        # Implicit absl.logging dependency.
        # Implicit fiddle dependency.
        # Implicit flax.core dependency.
        # Implicit ml_collections config_dict dependency.
        # Implicit numpy dependency.
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_library(
    name = "base_input",
    srcs = ["base_input.py"],
    srcs_version = "PY3",
    deps = [
        ":base_hyperparams",
        ":pax_fiddle",
        ":py_utils",
        ":pytypes",
        # Implicit absl.logging dependency.
        # Implicit fiddle dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_library(
    name = "base_layer",
    srcs = ["base_layer.py"],
    srcs_version = "PY3",
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
        # Implicit numpy dependency.
    ],
)

pytype_strict_library(
    name = "base_model",
    srcs = ["base_model.py"],
    srcs_version = "PY3",
    deps = [
        ":base_input",
        ":base_layer",
        ":pax_fiddle",
        ":py_utils",
        ":pytypes",
    ],
)

pytype_strict_library(
    name = "beam_search",
    srcs = ["beam_search.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":decoder_hparams",
        ":decoder_utils",
        ":py_utils",
        # Implicit flax.core dependency.
        # Implicit jax dependency.
    ],
)

pytype_strict_library(
    name = "decoder_hparams",
    srcs = ["decoder_hparams.py"],
    srcs_version = "PY3",
    deps = [
        ":base_hyperparams",
        ":base_layer",
        ":decoder_utils",
        ":pax_fiddle",
        ":pytypes",
        ":sample_decode",
    ],
)

pytype_strict_library(
    name = "decoder_utils",
    srcs = ["decoder_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":base_hyperparams",
        ":base_layer",
        ":pytypes",
        # Implicit flax.core dependency.
        # Implicit jax dependency.
    ],
)

pytype_strict_library(
    name = "flat_beam_search",
    srcs = ["flat_beam_search.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":decoder_utils",
        ":py_utils",
        # Implicit flax.core dependency.
        # Implicit jax dependency.
    ],
)

pytype_strict_library(
    name = "flax_utils",
    srcs = ["flax_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":pytypes",
        # Implicit flax.core dependency.
        # Implicit jax dependency.
    ],
)

pytype_strict_library(
    name = "gshard_utils",
    srcs = ["gshard_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":pytypes",
        # Implicit absl.logging dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
    ],
)

pytype_strict_library(
    name = "lingvo_lib",
    srcs = ["lingvo_lib.py"],
    deps = [
        # Implicit lingvo.core.cluster dependency.
        # Implicit lingvo.core.cluster_factory dependency.
        # Implicit lingvo.core.datasource dependency.
        # Implicit lingvo.core.hyperparams dependency.
        # Implicit lingvo.core.py_utils dependency.
    ],
)

pytype_strict_library(
    name = "metric_utils",
    srcs = ["metric_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":py_utils",
        ":pytypes",
        # Implicit jax dependency.
    ],
)

pytype_strict_library(
    name = "optimizer_prefix_vectorization",
    srcs = ["optimizer_prefix_vectorization.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":optimizers",
        ":py_utils",
        ":pytypes",
        # Implicit jax dependency.
        # Implicit optax dependency.
    ],
)

pytype_strict_library(
    name = "optimizers",
    srcs = ["optimizers.py"],
    srcs_version = "PY3",
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

pytype_strict_library(
    name = "py_utils",
    srcs = ["py_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":lingvo_lib",
        # Implicit absl.flags dependency.
        # Implicit absl.logging dependency.
        # Implicit flax dependency.
        # Implicit jax dependency.
        # Implicit jax.mesh_utils dependency.
        # Implicit numpy dependency.
        # Implicit optax dependency.
    ],
)

pytype_strict_library(
    name = "pytypes",
    srcs = ["pytypes.py"],
    srcs_version = "PY3",
    deps = [
        ":py_utils",
        # Implicit clu.metrics dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
    ],
)

pytype_strict_library(
    name = "sample_decode",
    srcs = ["sample_decode.py"],
    srcs_version = "PY3",
    deps = [
        ":asserts",
        ":base_hyperparams",
        ":base_layer",
        ":decoder_utils",
        ":py_utils",
        ":pytypes",
        # Implicit flax.core dependency.
        # Implicit jax dependency.
    ],
)

pytype_strict_library(
    name = "schedules",
    srcs = ["schedules.py"],
    srcs_version = "PY3",
    deps = [
        ":base_hyperparams",
        ":pytypes",
        # Implicit jax dependency.
        # Implicit optax dependency.
    ],
)

pytype_strict_library(
    name = "test_utils",
    testonly = True,
    srcs = ["test_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":py_utils",
        # Implicit absl.flags dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_library(
    name = "pax_fiddle",
    srcs = ["pax_fiddle.py"],
    deps = [
        # Implicit fiddle dependency.
        # Implicit fiddle.building dependency.
        # Implicit fiddle.daglish dependency.
        # Implicit fiddle.history dependency.
        # Implicit fiddle.signatures dependency.
        # Implicit fiddle.experimental.auto_config dependency.
        # Implicit fiddle.experimental.dataclasses dependency.
        # Implicit fiddle.extensions.jax dependency.
        # Implicit flax.core dependency.
        # Implicit typing_extensions dependency.
    ],
)

pytype_strict_library(
    name = "fiddle_tags",
    srcs = ["fiddle_tags.py"],
    deps = [
        # Implicit fiddle dependency.
    ],
)

py_strict_test(
    name = "asserts_test",
    srcs = ["asserts_test.py"],
    deps = [
        ":asserts",
        ":py_utils",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax.experimental.jax2tf dependency.
        # Implicit numpy dependency.
    ],
)

py_strict_test(
    name = "base_hyperparams_test",
    srcs = ["base_hyperparams_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["fast_and_reliable_fiddle_integration_test"],
    deps = [
        ":base_hyperparams",
        ":base_layer",
        ":pax_fiddle",
        # Implicit absl.testing.absltest dependency.
        # Implicit fiddle dependency.
        # Implicit fiddle.experimental.serialization dependency.
        # Implicit flax.core dependency.
        # Implicit jax dependency.
        # Implicit ml_collections config_dict dependency.
        # Implicit numpy dependency.
        # Implicit tensorflow_no_contrib dependency.
    ],
)

py_strict_test(
    name = "base_input_test",
    srcs = ["base_input_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":base_hyperparams",
        ":base_input",
        ":py_utils",
        ":test_utils",
        # Implicit absl.flags dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit fiddle dependency.
        # Implicit lingvo.core.base_input_generator dependency.
        # Implicit lingvo.core.generic_input dependency.
        # Implicit lingvo.core.py_utils dependency.
        # Implicit numpy dependency.
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_contrib_test(
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

py_strict_test(
    name = "beam_search_test",
    srcs = ["beam_search_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":beam_search",
        ":pax_fiddle",
        ":py_utils",
        ":test_utils",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis/layers:models",
        "//praxis/layers:transformer_models",
    ],
)

py_strict_test(
    name = "decoder_utils_test",
    srcs = ["decoder_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":decoder_utils",
        ":test_utils",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
    ],
)

py_strict_test(
    name = "flat_beam_search_test",
    srcs = ["flat_beam_search_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":flat_beam_search",
        ":test_utils",
        # Implicit absl.testing.absltest dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
    ],
)

py_strict_test(
    name = "optimizers_test",
    srcs = ["optimizers_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":optimizers",
        ":schedules",
        ":test_utils",
        # Implicit absl.testing.absltest dependency.
        # Implicit jax dependency.
        # Implicit optax dependency.
    ],
)

py_strict_test(
    name = "py_utils_test",
    srcs = ["py_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":py_utils",
        ":test_utils",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit flax.core dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
    ],
)

py_strict_test(
    name = "sample_decode_test",
    srcs = ["sample_decode_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":base_model",
        ":pax_fiddle",
        ":py_utils",
        ":sample_decode",
        ":test_utils",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit flax.core dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
    ],
)

py_strict_test(
    name = "schedules_test",
    srcs = ["schedules_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":base_hyperparams",
        ":schedules",
        ":test_utils",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit lingvo.core.py_utils dependency.
        # Implicit lingvo.core.schedule dependency.
    ],
)

py_strict_test(
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
        # Implicit fiddle.testing dependency.
        # Implicit flax.core dependency.
        # Implicit jax dependency.
    ],
)
