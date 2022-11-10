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
#   Praxis layers. The public API is defined in __init__.py.

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
        ":activations",
        ":adapters",
        ":attentions",
        ":augmentations",
        ":bregman",
        ":checkpoint_policy",
        ":conformers",
        ":convolutions",
        ":embedding_softmax",
        ":frnn",
        ":linears",
        ":losses",
        ":models",
        ":ngrammer",
        ":normalizations",
        ":pipeline",
        ":poolings",
        ":quantizer",
        ":repeats",
        ":resnets",
        ":rnn_cell",
        ":spectrum_augmenter",
        ":stochastics",
        ":transformer_models",
        ":transformers",
        ":vanillanets",
        ":vits",
    ],
)

pytype_strict_library(
    name = "activations",
    srcs = ["activations.py"],
    srcs_version = "PY3",
    deps = [
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:pytypes",
    ],
)

py_strict_test(
    name = "activations_test",
    srcs = ["activations_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":activations",
        # Implicit absl.testing.absltest dependency.
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:test_utils",
    ],
)

pytype_strict_library(
    name = "adapters",
    srcs = ["adapters.py"],
    srcs_version = "PY3",
    deps = [
        ":activations",
        ":normalizations",
        ":transformers",
        # Implicit jax dependency.
        "//praxis:asserts",
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "attentions",
    srcs = ["attentions.py"],
    srcs_version = "PY3",
    deps = [
        ":embedding_softmax",
        ":stochastics",
        # Implicit absl.logging dependency.
        # Implicit flax.core dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:asserts",
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "augmentations",
    srcs = ["augmentations.py"],
    srcs_version = "PY3",
    deps = [
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "bregman",
    srcs = ["bregman.py"],
    srcs_version = "PY3",
    deps = [
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "convolutions",
    srcs = ["convolutions.py"],
    srcs_version = "PY3",
    deps = [
        ":activations",
        ":linears",
        ":normalizations",
        ":stochastics",
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "embedding_softmax",
    srcs = ["embedding_softmax.py"],
    srcs_version = "PY3",
    deps = [
        ":activations",
        ":linears",
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "frnn",
    srcs = ["frnn.py"],
    srcs_version = "PY3",
    deps = [
        ":rnn_cell",
        # Implicit flax.core dependency.
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "glam",
    srcs = ["glam.py"],
    srcs_version = "PY3",
    deps = [
        ":activations",
        ":attentions",
        ":embedding_softmax",
        ":normalizations",
        ":transformer_models",
        ":transformers",
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
    ],
)

pytype_strict_library(
    name = "linears",
    srcs = ["linears.py"],
    srcs_version = "PY3",
    deps = [
        ":activations",
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "losses",
    srcs = ["losses.py"],
    srcs_version = "PY3",
    deps = [
        # Implicit jax_bitempered_loss dependency.
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "models",
    srcs = ["models.py"],
    srcs_version = "PY3",
    deps = [
        ":augmentations",
        ":embedding_softmax",
        ":linears",
        ":resnets",
        ":transformer_models",
        # Implicit absl.logging dependency.
        # Implicit clu.metrics dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:asserts",
        "//praxis:base_input",
        "//praxis:base_layer",
        "//praxis:base_model",
        "//praxis:beam_search",
        "//praxis:decoder_hparams",
        "//praxis:decoder_utils",
        "//praxis:flat_beam_search",
        "//praxis:metric_utils",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:pytypes",
        "//praxis:sample_decode",
    ],
)

pytype_strict_library(
    name = "ngrammer",
    srcs = ["ngrammer.py"],
    srcs_version = "PY3",
    deps = [
        ":bregman",
        ":embedding_softmax",
        ":normalizations",
        # Implicit jax dependency.
        "//praxis:asserts",
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:pytypes",
        # Implicit sympy dependency.
    ],
)

pytype_strict_library(
    name = "normalizations",
    srcs = ["normalizations.py"],
    srcs_version = "PY3",
    deps = [
        # Implicit jax dependency.
        "//praxis:asserts",
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "objectives",
    srcs = [
        "ctc_objectives.py",
        "quantizer_objectives.py",
    ],
    srcs_version = "PY3",
    deps = [
        # Implicit jax dependency.
        # Implicit numpy dependency.
        # Implicit optax dependency.
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "stats",
    srcs = ["stats.py"],
    srcs_version = "PY3",
    deps = [
        # Implicit jax dependency.
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

py_strict_test(
    name = "stats_test",
    srcs = ["stats_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":stats",
        # Implicit absl.testing.absltest dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:test_utils",
    ],
)

pytype_strict_library(
    name = "pipeline",
    srcs = ["pipeline.py"],
    srcs_version = "PY3",
    deps = [
        ":checkpoint_policy",
        # Implicit flax.core dependency.
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:flax_utils",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "poolings",
    srcs = ["poolings.py"],
    srcs_version = "PY3",
    deps = [
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "quantizer",
    srcs = ["quantizer.py"],
    srcs_version = "PY3",
    deps = [
        ":linears",
        ":objectives",
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "repeats",
    srcs = ["repeats.py"],
    srcs_version = "PY3",
    deps = [
        ":checkpoint_policy",
        # Implicit flax.core dependency.
        # Implicit jax dependency.
        "//praxis:asserts",
        "//praxis:base_layer",
        "//praxis:flax_utils",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

py_strict_test(
    name = "repeats_test",
    srcs = ["repeats_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":repeats",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:test_utils",
    ],
)

pytype_strict_library(
    name = "resnets",
    srcs = ["resnets.py"],
    srcs_version = "PY3",
    deps = [
        ":activations",
        ":convolutions",
        ":normalizations",
        ":poolings",
        ":stochastics",
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "spectrum_augmenter",
    srcs = ["spectrum_augmenter.py"],
    srcs_version = "PY3",
    deps = [
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

py_strict_test(
    name = "spectrum_augmenter_test",
    srcs = ["spectrum_augmenter_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":spectrum_augmenter",
        # Implicit absl.testing.absltest dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:test_utils",
    ],
)

pytype_strict_library(
    name = "rnn_cell",
    srcs = ["rnn_cell.py"],
    srcs_version = "PY3",
    deps = [
        # Implicit jax dependency.
        "//praxis:asserts",
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "vanillanets",
    srcs = ["vanillanets.py"],
    srcs_version = "PY3",
    deps = [
        ":convolutions",
        ":poolings",
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

py_strict_test(
    name = "vanillanets_test",
    srcs = ["vanillanets_test.py"],
    deps = [
        ":poolings",
        ":vanillanets",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
    ],
)

pytype_strict_library(
    name = "stochastics",
    srcs = ["stochastics.py"],
    srcs_version = "PY3",
    deps = [
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "test_layers",
    testonly = True,
    srcs = ["test_layers.py"],
    srcs_version = "PY3",
    deps = [
        ":linears",
        ":normalizations",
        ":transformers",
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:base_model",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

py_strict_test(
    name = "adapters_test",
    srcs = ["adapters_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":adapters",
        ":normalizations",
        ":transformers",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit lingvo.core.cluster_factory dependency.
        # Implicit lingvo.core.layers dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:test_utils",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

py_strict_test(
    name = "conformers_test",
    srcs = ["conformers_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":attentions",
        ":conformers",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit lingvo.core.cluster_factory dependency.
        # Implicit lingvo.core.conformer_layer dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:test_utils",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_library(
    name = "conformers",
    srcs = ["conformers.py"],
    srcs_version = "PY3",
    deps = [
        ":activations",
        ":attentions",
        ":convolutions",
        ":normalizations",
        ":stochastics",
        ":transformers",
        # Implicit jax dependency.
        "//praxis:asserts",
        "//praxis:base_hyperparams",
        "//praxis:base_layer",
        "//praxis:py_utils",
    ],
)

pytype_strict_library(
    name = "checkpoint_policy",
    srcs = ["checkpoint_policy.py"],
    srcs_version = "PY3",
    deps = [
        # Implicit jax dependency.
    ],
)

pytype_strict_library(
    name = "transformers",
    srcs = ["transformers.py"],
    srcs_version = "PY3",
    deps = [
        ":activations",
        ":attentions",
        ":checkpoint_policy",
        ":linears",
        ":normalizations",
        ":pipeline",
        ":repeats",
        ":stats",
        ":stochastics",
        # Implicit absl.logging dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:gshard_utils",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "transformer_models",
    srcs = ["transformer_models.py"],
    deps = [
        ":attentions",
        ":embedding_softmax",
        ":multi_query_attention",
        ":normalizations",
        ":transformers",
        # Implicit jax dependency.
        "//praxis:asserts",
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

py_strict_test(
    name = "vits_test",
    srcs = ["vits_test.py"],
    deps = [
        ":activations",
        ":embedding_softmax",
        ":transformers",
        ":vits",
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
    name = "attentions_test",
    srcs = ["attentions_test.py"],
    python_version = "PY3",
    shard_count = 2,
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
        # Implicit tensorflow_no_contrib dependency.
    ],
)

py_strict_test(
    name = "augmentations_test",
    srcs = ["augmentations_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":augmentations",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:test_utils",
    ],
)

py_strict_test(
    name = "bregman_test",
    srcs = ["bregman_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":bregman",
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
    name = "ctc_objectives_test",
    srcs = ["ctc_objectives_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":objectives",
        # Implicit absl.testing.absltest dependency.
        # Implicit jax dependency.
        # Implicit jax.experimental.jax2tf dependency.
        # Implicit numpy dependency.
        # Implicit optax dependency.
        "//praxis:pytypes",
        "//praxis:test_utils",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

py_strict_test(
    name = "embedding_softmax_test",
    srcs = ["embedding_softmax_test.py"],
    python_version = "PY3",
    shard_count = 2,
    srcs_version = "PY3",
    deps = [
        ":embedding_softmax",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit lingvo.core.layers dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:test_utils",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

py_strict_test(
    name = "frnn_test",
    srcs = ["frnn_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":frnn",
        ":rnn_cell",
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
    name = "linears_test",
    srcs = ["linears_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":activations",
        ":linears",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit lingvo.core.layers dependency.
        # Implicit numpy dependency.
        "//praxis:base_hyperparams",
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:test_utils",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

py_strict_test(
    name = "shared_layers_test",
    srcs = ["shared_layers_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":linears",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_hyperparams",
        "//praxis:base_layer",
        "//praxis:test_utils",
    ],
)

py_strict_test(
    name = "models_test",
    srcs = ["models_test.py"],
    python_version = "PY3",
    shard_count = 4,
    srcs_version = "PY3",
    deps = [
        ":models",
        ":resnets",
        ":transformer_models",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:decoder_utils",
        "//praxis:py_utils",
        "//praxis:pytypes",
        "//praxis:test_utils",
    ],
)

py_strict_test(
    name = "ngrammer_test",
    srcs = ["ngrammer_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":bregman",
        ":ngrammer",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit lingvo.core.attention_util dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:test_utils",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

py_strict_test(
    name = "normalizations_test",
    srcs = ["normalizations_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":normalizations",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit lingvo.core.bn_layers dependency.
        # Implicit lingvo.core.layers dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:test_utils",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

py_strict_test(
    name = "poolings_test",
    srcs = ["poolings_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":poolings",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit lingvo.core.layers dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:test_utils",
        # Implicit tensorflow_no_contrib dependency.
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
        "//praxis:test_utils",
    ],
)

py_strict_test(
    name = "quantizer_objectives_test",
    srcs = ["quantizer_objectives_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":objectives",
        # Implicit absl.testing.absltest dependency.
        # Implicit numpy dependency.
        "//praxis:test_utils",
    ],
)

py_strict_test(
    name = "rnn_cell_test",
    srcs = ["rnn_cell_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":rnn_cell",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit lingvo.core.py_utils dependency.
        # Implicit lingvo.core.rnn_cell dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:test_utils",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

py_strict_test(
    name = "stochastics_test",
    srcs = ["stochastics_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":stochastics",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:test_utils",
    ],
)

py_strict_test(
    name = "transformers_test",
    size = "large",
    srcs = ["transformers_test.py"],
    python_version = "PY3",
    shard_count = 20,
    srcs_version = "PY3",
    deps = [
        ":activations",
        ":attentions",
        ":transformers",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit lingvo.core.batch_major_attention dependency.
        # Implicit lingvo.core.layers_with_attention dependency.
        # Implicit numpy dependency.
        "//praxis:base_hyperparams",
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:test_utils",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

py_strict_test(
    name = "transformer_models_test",
    size = "large",
    srcs = ["transformer_models_test.py"],
    python_version = "PY3",
    shard_count = 27,
    srcs_version = "PY3",
    deps = [
        ":attentions",
        ":embedding_softmax",
        ":glam",
        ":ngrammer",
        ":transformer_models",
        ":transformers",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit lingvo.core.gshard_builder dependency.
        # Implicit numpy dependency.
        "//praxis:base_hyperparams",
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:test_utils",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_library(
    name = "flax_adapter",
    srcs = ["flax_adapter.py"],
    srcs_version = "PY3",
    deps = [
        # Implicit flax.core dependency.
        "//praxis:base_layer",
        "//praxis:flax_utils",
        "//praxis:pytypes",
    ],
)

py_strict_test(
    name = "flax_adapter_test",
    srcs = ["flax_adapter_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":flax_adapter",
        ":normalizations",
        # Implicit absl.testing.absltest dependency.
        # Implicit fiddle dependency.
        # Implicit flax.core dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pytypes",
        "//praxis:test_utils",
    ],
)

pytype_strict_library(
    name = "flaxformer_models",
    srcs = ["flaxformer_models.py"],
    srcs_version = "PY3",
    deps = [
        ":flax_adapter",
        # Implicit flaxformer.architectures.t5.parallel_fused_decoder dependency.
        # Implicit flaxformer.architectures.t5.t5_architecture dependency.
        # Implicit flaxformer.components.dense dependency.
        # Implicit flaxformer.components.embedding dependency.
        # Implicit flaxformer.components.layer_norm dependency.
        # Implicit flaxformer.components.relative_position_biases dependency.
        # Implicit flaxformer.components.attention.dense_attention dependency.
        # Implicit clu dependency.
        # Implicit flax.core dependency.
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:base_model",
        "//praxis:decoder_hparams",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:pytypes",
        # Implicit t5x.decoding dependency.
        # Implicit t5x.losses dependency.
    ],
)

pytype_strict_library(
    name = "vits",
    srcs = ["vits.py"],
    deps = [
        ":activations",
        ":embedding_softmax",
        ":linears",
        ":normalizations",
        ":poolings",
        ":stochastics",
        ":transformers",
        # Implicit einops dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_hyperparams",
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "multi_query_attention",
    srcs = ["multi_query_attention.py"],
    deps = [
        ":attentions",
        ":embedding_softmax",
        ":stochastics",
        # Implicit flax.core dependency.
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:pytypes",
    ],
)

py_strict_test(
    name = "flaxformer_models_test",
    srcs = ["flaxformer_models_test.py"],
    data = glob(["testdata/*.json"]),
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":flaxformer_models",
        # Implicit flaxformer.testing_utils dependency.
        # Implicit flaxformer.t5x.fiddle.architectures.t5_1_1 dependency.
        # Implicit flaxformer.flaxformer.t5x.fiddle.models.t5x_model dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit fiddle dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:test_utils",
        # Implicit t5x.models dependency.
        # Implicit t5x.partitioning dependency.
    ],
)

py_strict_test(
    name = "multi_query_attention_test",
    srcs = ["multi_query_attention_test.py"],
    deps = [
        ":attentions",
        ":multi_query_attention",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:test_utils",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

py_strict_test(
    name = "convolutions_test",
    srcs = ["convolutions_test.py"],
    python_version = "PY3",
    shard_count = 10,
    srcs_version = "PY3",
    deps = [
        ":convolutions",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit lingvo.core.cluster_factory dependency.
        # Implicit lingvo.core.conformer_layer dependency.
        # Implicit core.conv_layers_with_time_padding dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:test_utils",
        # Implicit tensorflow_no_contrib dependency.
    ],
)
