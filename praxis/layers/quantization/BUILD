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
        ":conformers",
        ":embedding_softmax",
        ":linears",
        ":multi_query_attention",
        ":ngrammer",
        ":operations",
        ":searchable",
    ],
)

pytype_strict_library(
    name = "quantize",
    srcs = ["quantize.py"],
    srcs_version = "PY3",
    deps = [
        ":layers",
        ":quantization_hparams",
        # Implicit fiddle dependency.
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis/layers",
    ],
)

py_strict_test(
    name = "quantize_test",
    srcs = ["quantize_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":layers",
        ":quantization_hparams",
        ":quantize",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        "//praxis:pax_fiddle",
        "//praxis:test_utils",
        "//praxis/layers",
        "//praxis/layers:embedding_softmax",
    ],
)

pytype_strict_library(
    name = "attentions",
    srcs = ["attentions.py"],
    srcs_version = "PY3",
    deps = [
        ":operations",
        ":quantization_hparams",
        ":quantizer",
        ":utils",
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
        ":quantization_hparams",
        # Implicit absl.logging dependency.
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
    name = "quantization_test_utils",
    testonly = 1,
    srcs = ["tests/test_util.py"],
    srcs_version = "PY3",
    deps = [
        "//praxis:test_utils",
    ],
)

py_strict_test(
    name = "attention_projection_ptq_test",
    srcs = ["tests/attention_projection_ptq_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":attentions",
        ":quantization_hparams",
        ":quantization_test_utils",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis/layers:attentions",
    ],
)

py_strict_test(
    name = "attention_projection_fq_test",
    srcs = ["tests/attention_projection_fq_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":attentions",
        ":quantization_hparams",
        ":quantization_test_utils",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis/layers:attentions",
    ],
)

# TODO(dhchoi) simplify test
py_strict_test(
    name = "attention_projection_aqt_test",
    srcs = ["tests/attention_projection_aqt_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":attentions",
        ":quantization_hparams",
        ":quantization_test_utils",
        # Implicit absl.logging dependency.
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

py_strict_test(
    name = "combined_qkv_projection_ptq_test",
    srcs = ["tests/combined_qkv_projection_ptq_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":attentions",
        ":quantization_hparams",
        ":quantization_test_utils",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis/layers:attentions",
    ],
)

py_strict_test(
    name = "combined_qkv_projection_fq_test",
    srcs = ["tests/combined_qkv_projection_fq_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":attentions",
        ":quantization_hparams",
        ":quantization_test_utils",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis/layers:attentions",
    ],
)

py_strict_test(
    name = "combined_qkv_projection_aqt_test",
    srcs = ["tests/combined_qkv_projection_aqt_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":attentions",
        ":quantization_hparams",
        ":quantization_test_utils",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis/layers:attentions",
    ],
)

py_strict_test(
    name = "dotproduct_attention_aqt_test",
    srcs = ["tests/dotproduct_attention_aqt_test.py"],
    python_version = "PY3",
    shard_count = 30,
    srcs_version = "PY3",
    deps = [
        ":attentions",
        ":quantization_hparams",
        ":quantization_test_utils",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis/layers:attentions",
    ],
)

py_strict_test(
    name = "linear_ptq_test",
    srcs = ["tests/linear_ptq_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":linears",
        ":quantization_hparams",
        ":quantization_test_utils",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis/layers:linears",
    ],
)

pytype_strict_library(
    name = "conformers",
    srcs = ["conformers.py"],
    srcs_version = "PY3",
    deps = [
        ":attentions",
        # Implicit jax dependency.
        "//praxis:pytypes",
        "//praxis/layers:attentions",
        "//praxis/layers:conformers",
    ],
)

py_strict_test(
    name = "conformers_test",
    srcs = ["conformers_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":conformers",
        ":quantization_hparams",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:test_utils",
        "//praxis/layers:attentions",
        "//praxis/layers:conformers",
    ],
)

pytype_strict_library(
    name = "linears",
    srcs = ["linears.py"],
    srcs_version = "PY3",
    deps = [
        ":operations",
        ":quantization_hparams",
        ":quantizer",
        ":utils",
        # Implicit jax dependency.
        "//praxis:base_layer",
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
        ":quantization_hparams",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:schedules",
        "//praxis:test_utils",
        "//praxis/layers:linears",
    ],
)

pytype_strict_library(
    name = "multi_query_attention",
    srcs = ["multi_query_attention.py"],
    srcs_version = "PY3",
    deps = [
        ":operations",
        ":quantization_hparams",
        ":quantizer",
        ":utils",
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:pytypes",
        "//praxis/layers:multi_query_attention",
    ],
)

py_strict_test(
    name = "multi_query_attention_test",
    srcs = ["multi_query_attention_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":multi_query_attention",
        ":quantization_hparams",
        ":utils",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:test_utils",
        "//praxis/layers:multi_query_attention",
    ],
)

py_strict_test(
    name = "one_headed_attention_projection_aqt_test",
    srcs = ["tests/one_headed_attention_projection_aqt_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":multi_query_attention",
        ":quantization_hparams",
        ":quantization_test_utils",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:test_utils",
        "//praxis/layers:multi_query_attention",
    ],
)

pytype_strict_library(
    name = "embedding_softmax",
    srcs = ["embedding_softmax.py"],
    srcs_version = "PY3",
    deps = [
        ":linears",
        ":operations",
        ":quantization_hparams",
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:pytypes",
        "//praxis/layers:activations",
        "//praxis/layers:embedding_softmax",
        "//praxis/layers:linears",
    ],
)

py_strict_test(
    name = "embedding_softmax_test",
    srcs = ["embedding_softmax_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":layers",
        ":quantization_hparams",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:test_utils",
        "//praxis/layers:embedding_softmax",
    ],
)

pytype_strict_library(
    name = "ngrammer",
    srcs = ["ngrammer.py"],
    srcs_version = "PY3",
    deps = [
        ":embedding_softmax",
        ":quantization_hparams",
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis/layers:ngrammer",
        "//praxis/layers:normalizations",
    ],
)

py_strict_test(
    name = "ngrammer_test",
    srcs = ["ngrammer_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":layers",
        ":quantization_hparams",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:test_utils",
        "//praxis/layers:embedding_softmax",
        "//praxis/layers:ngrammer",
    ],
)

pytype_strict_library(
    name = "operations",
    srcs = ["operations.py"],
    srcs_version = "PY3",
    deps = [
        ":optimization",
        ":utils",
        # Implicit absl.logging dependency.
        # Implicit jax dependency.
        "//praxis:pytypes",
    ],
)

py_strict_test(
    name = "operations_test",
    srcs = ["operations_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":operations",
        ":quantizer",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
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
        ":quantization_hparams",
        # Implicit absl.testing.absltest dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:test_utils",
        "//praxis/layers:repeats",
    ],
)

pytype_strict_library(
    name = "optimization",
    srcs = ["optimization.py"],
    srcs_version = "PY3",
    deps = [
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:pytypes",
    ],
)

py_strict_test(
    name = "optimization_test",
    srcs = ["optimization_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":optimization",
        # Implicit absl.testing.absltest dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:test_utils",
    ],
)

pytype_strict_library(
    name = "quantizer",
    srcs = ["quantizer.py"],
    srcs_version = "PY3",
    deps = [
        ":operations",
        ":quantization_hparams",
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:pytypes",
    ],
)

py_strict_test(
    name = "quantizer_test",
    srcs = ["quantizer_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":quantizer",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:pytypes",
        "//praxis:test_utils",
    ],
)

pytype_strict_library(
    name = "quantization_hparams",
    srcs = ["quantization_hparams.py"],
    srcs_version = "PY3",
    deps = [
        # Implicit jax dependency.
        "//praxis:base_hyperparams",
    ],
)

pytype_strict_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY3",
    deps = [
        # Implicit jax dependency.
    ],
)

pytype_strict_library(
    name = "automl_select",
    srcs = ["automl_select.py"],
    deps = [
        # Implicit flax.core dependency.
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "searchable",
    srcs = ["searchable.py"],
    deps = [
        ":attentions",
        ":automl_select",
        ":linears",
        ":quantization_hparams",
        ":quantizer",
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
    ],
)

py_strict_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":utils",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:test_utils",
    ],
)

py_strict_test(
    name = "automl_select_test",
    srcs = ["automl_select_test.py"],
    deps = [
        ":automl_select",
        ":quantization_hparams",
        ":quantizer",
        # Implicit absl.testing.absltest dependency.
        # Implicit jax dependency.
        "//praxis:base_hyperparams",
        "//praxis:pax_fiddle",
        "//praxis:test_utils",
    ],
)

py_strict_test(
    name = "searchable_test",
    srcs = ["searchable_test.py"],
    deps = [
        ":quantization_hparams",
        ":searchable",
        # Implicit absl.testing.absltest dependency.
        # Implicit jax dependency.
        "//praxis:base_hyperparams",
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:test_utils",
    ],
)
