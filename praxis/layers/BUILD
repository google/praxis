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

load("//praxis:praxis.bzl", "pytype_strict_library", "pytype_strict_test")
load("//praxis:build-visibility.bzl", "JAX_VISIBILITY")

licenses(["notice"])

package(default_visibility = JAX_VISIBILITY)

pytype_strict_library(
    name = "layers",
    srcs = ["__init__.py"],
    deps = [
        ":activations",
        ":adapters",
        ":attentions",
        ":augmentations",
        ":base_ops",
        ":bregman",
        ":checkpoint_policy",
        ":conformers",
        ":convolutions",
        ":einsum",
        ":embedding_softmax",
        ":frnn",
        ":grouped_query_attention",
        ":linears",
        ":losses",
        ":mobilenet",
        ":models",
        ":ngrammer",
        ":normalizations",
        ":pipeline",
        ":poolings",
        ":quantizer",
        ":repeats",
        ":resnets",
        ":rnn_cell",
        ":sequential",
        ":spectrum_augmenter",
        ":ssm",
        ":ssm_transformers",
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
    deps = [
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:pytypes",
    ],
)

pytype_strict_test(
    name = "activations_test",
    srcs = ["activations_test.py"],
    deps = [
        ":activations",
        # Implicit absl.testing.absltest dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:test_utils",
    ],
)

pytype_strict_library(
    name = "adapters",
    srcs = ["adapters.py"],
    deps = [
        ":activations",
        ":normalizations",
        ":transformers",
        # Implicit fiddle dependency.
        # Implicit jax dependency.
        "//praxis:asserts",
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "gpu_fast_attention",
    srcs = ["gpu_fast_attention.py"],
    # Internal cuda visibility rule.
    deps = [
        ":attentions",
        ":grouped_query_attention",
        ":multi_query_attention",
        ":normalizations",
        # Implicit jax dependency.
        # Implicit Pallas GPU dependency.  # BC internal suffix.
        # Implicit Pallas GPU Ops dependency.  # BC internal suffix.
        # Implicit numpy dependency.
        "//praxis:asserts",
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "attentions",
    srcs = ["attentions.py"],
    deps = [
        ":base_ops",
        ":embedding_softmax",
        ":stochastics",
        # Implicit absl.logging dependency.
        # Implicit flax.core dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:asserts",
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "grouped_query_attention",
    srcs = ["grouped_query_attention.py"],
    deps = [
        ":einsum",
        ":sharding",
        # Implicit flax.core dependency.
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_test(
    name = "grouped_query_attention_test",
    srcs = ["grouped_query_attention_test.py"],
    env = {"XLA_FLAGS": "--xla_force_host_platform_device_count=4"},
    deps = [
        ":grouped_query_attention",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:test_utils",
    ],
)

pytype_strict_library(
    name = "augmentations",
    srcs = ["augmentations.py"],
    deps = [
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "base_ops",
    srcs = ["base_ops.py"],
    deps = [
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "bregman",
    srcs = ["bregman.py"],
    deps = [
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "chunk",
    srcs = ["chunk.py"],
    deps = [
        # Implicit jax dependency.
        "//praxis:pytypes",
    ],
)

pytype_strict_test(
    name = "chunk_test",
    srcs = ["chunk_test.py"],
    deps = [
        ":chunk",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit upb python proto dependency.
        # Implicit numpy dependency.
        "//praxis:test_utils",
    ],
)

pytype_strict_library(
    name = "convolutions",
    srcs = ["convolutions.py"],
    deps = [
        ":activations",
        ":linears",
        ":normalizations",
        ":stochastics",
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "einsum",
    srcs = ["einsum.py"],
    deps = [
        ":base_ops",
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:pytypes",
    ],
)

pytype_strict_test(
    name = "einsum_test",
    size = "large",
    srcs = ["einsum_test.py"],
    deps = [
        ":einsum",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:test_utils",
    ],
)

pytype_strict_library(
    name = "embedding_softmax",
    srcs = ["embedding_softmax.py"],
    deps = [
        ":activations",
        ":base_ops",
        ":chunk",
        ":linears",
        # Implicit flax.core dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "frnn",
    srcs = ["frnn.py"],
    deps = [
        ":rnn_cell",
        # Implicit flax.core dependency.
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "glam",
    srcs = ["glam.py"],
    deps = [
        ":activations",
        ":attentions",
        ":checkpoint_policy",
        ":embedding_softmax",
        ":normalizations",
        ":transformer_models",
        ":transformers",
        # Implicit fiddle dependency.
        "//praxis:pax_fiddle",
    ],
)

pytype_strict_library(
    name = "linears",
    srcs = ["linears.py"],
    deps = [
        ":activations",
        ":base_ops",
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "losses",
    srcs = ["losses.py"],
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
    deps = [
        ":augmentations",
        ":embedding_softmax",
        ":linears",
        ":mobilenet",
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
    deps = [
        ":bregman",
        ":embedding_softmax",
        ":normalizations",
        # Implicit jax dependency.
        "//praxis:asserts",
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:pytypes",
        # Implicit sympy dependency.
    ],
)

pytype_strict_library(
    name = "normalizations",
    srcs = ["normalizations.py"],
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
    deps = [
        # Implicit jax dependency.
        # Implicit numpy dependency.
        # Implicit optax dependency.
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "ssm",
    srcs = ["ssm.py"],
    deps = [
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "ssm_transformers",
    srcs = ["ssm_transformers.py"],
    deps = [
        ":activations",
        ":checkpoint_policy",
        ":linears",
        ":ssm",
        ":stats",
        ":stochastics",
        ":transformers",
        # Implicit absl.logging dependency.
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_test(
    name = "ssm_transformers_test",
    size = "large",
    srcs = ["ssm_transformers_test.py"],
    shard_count = 20,
    deps = [
        ":attentions",
        ":ssm_transformers",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_hyperparams",
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:test_utils",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_library(
    name = "stats",
    srcs = ["stats.py"],
    deps = [
        # Implicit jax dependency.
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_test(
    name = "stats_test",
    srcs = ["stats_test.py"],
    deps = [
        ":stats",
        # Implicit absl.testing.absltest dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:test_utils",
    ],
)

pytype_strict_library(
    name = "pipeline",
    srcs = ["pipeline.py"],
    deps = [
        ":checkpoint_policy",
        # Implicit flax.core dependency.
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "poolings",
    srcs = ["poolings.py"],
    deps = [
        # Implicit jax dependency.
        "//praxis:asserts",
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "quantizer",
    srcs = ["quantizer.py"],
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
    deps = [
        ":checkpoint_policy",
        # Implicit flax.core dependency.
        # Implicit jax dependency.
        "//praxis:asserts",
        "//praxis:base_layer",
        "//praxis:flax_utils",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_test(
    name = "repeats_test",
    srcs = ["repeats_test.py"],
    deps = [
        ":linears",
        ":repeats",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:test_utils",
        "//praxis/layers/quantization:linears",
        "//praxis/layers/quantization/sparsity:sparsity_hparams",
        "//praxis/layers/quantization/sparsity:sparsity_modes",
    ],
)

pytype_strict_library(
    name = "mobilenet",
    srcs = ["mobilenet.py"],
    deps = [
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:pytypes",
        "//praxis/layers:activations",
        "//praxis/layers:convolutions",
        "//praxis/layers:linears",
        "//praxis/layers:normalizations",
        "//praxis/layers:poolings",
        "//praxis/layers:sequential",
    ],
)

pytype_strict_library(
    name = "resnets",
    srcs = ["resnets.py"],
    deps = [
        ":activations",
        ":convolutions",
        ":normalizations",
        ":poolings",
        ":stochastics",
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "spectrum_augmenter",
    srcs = ["spectrum_augmenter.py"],
    deps = [
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_test(
    name = "spectrum_augmenter_test",
    srcs = ["spectrum_augmenter_test.py"],
    deps = [
        ":spectrum_augmenter",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:test_utils",
    ],
)

pytype_strict_library(
    name = "rnn_cell",
    srcs = ["rnn_cell.py"],
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
    deps = [
        ":convolutions",
        ":poolings",
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_test(
    name = "vanillanets_test",
    srcs = ["vanillanets_test.py"],
    deps = [
        ":poolings",
        ":vanillanets",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
    ],
)

pytype_strict_library(
    name = "stochastics",
    srcs = ["stochastics.py"],
    deps = [
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "sequential",
    srcs = ["sequential.py"],
    deps = [
        "//praxis:base_layer",
    ],
)

pytype_strict_library(
    name = "test_layers",
    testonly = True,
    srcs = ["test_layers.py"],
    deps = [
        ":linears",
        ":normalizations",
        ":transformers",
        # Implicit clu.metrics dependency.
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:base_model",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_test(
    name = "adapters_test",
    srcs = ["adapters_test.py"],
    deps = [
        ":adapters",
        ":normalizations",
        ":transformers",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit lingvo.core.cluster_factory dependency.
        # Implicit lingvo.core.layers dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:test_utils",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_test(
    name = "conformers_test",
    srcs = ["conformers_test.py"],
    deps = [
        ":attentions",
        ":conformers",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit lingvo.core.cluster_factory dependency.
        # Implicit lingvo.core.conformer_layer dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:test_utils",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_library(
    name = "conformers",
    srcs = ["conformers.py"],
    deps = [
        ":activations",
        ":attentions",
        ":convolutions",
        ":normalizations",
        ":stochastics",
        ":transformers",
        # Implicit fiddle dependency.
        # Implicit jax dependency.
        "//praxis:asserts",
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
    ],
)

pytype_strict_library(
    name = "checkpoint_policy",
    srcs = ["checkpoint_policy.py"],
    deps = [
        # Implicit jax dependency.
    ],
)

pytype_strict_library(
    name = "transformers",
    srcs = ["transformers.py"],
    deps = [
        ":activations",
        ":attentions",
        ":base_ops",
        ":checkpoint_policy",
        ":linears",
        ":normalizations",
        ":pipeline",
        ":repeats",
        ":stats",
        ":stochastics",
        # Implicit absl.logging dependency.
        # Implicit flax.core dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:gshard_utils",
        "//praxis:pax_fiddle",
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
        # Implicit fiddle dependency.
        # Implicit jax dependency.
        "//praxis:asserts",
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_test(
    name = "vits_test",
    srcs = ["vits_test.py"],
    deps = [
        ":activations",
        ":embedding_softmax",
        ":poolings",
        ":transformers",
        ":vits",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:test_utils",
    ],
)

pytype_strict_test(
    name = "attentions_test",
    srcs = ["attentions_test.py"],
    shard_count = 2,
    deps = [
        ":attentions",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit lingvo.core.batch_major_attention dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:decoder_utils",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:test_utils",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_test(
    name = "augmentations_test",
    srcs = ["augmentations_test.py"],
    deps = [
        ":augmentations",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:test_utils",
    ],
)

pytype_strict_test(
    name = "bregman_test",
    srcs = ["bregman_test.py"],
    deps = [
        ":bregman",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:test_utils",
    ],
)

pytype_strict_test(
    name = "ctc_objectives_test",
    srcs = ["ctc_objectives_test.py"],
    deps = [
        ":objectives",
        # Implicit absl.testing.absltest dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit jax.experimental.jax2tf dependency.
        # Implicit numpy dependency.
        # Implicit optax dependency.
        "//praxis:py_utils",
        "//praxis:pytypes",
        "//praxis:test_utils",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_test(
    name = "embedding_softmax_test",
    srcs = ["embedding_softmax_test.py"],
    shard_count = 2,
    deps = [
        ":embedding_softmax",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit lingvo.core.layers dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:test_utils",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_test(
    name = "frnn_test",
    srcs = ["frnn_test.py"],
    deps = [
        ":frnn",
        ":rnn_cell",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:test_utils",
    ],
)

pytype_strict_test(
    name = "linears_test",
    srcs = ["linears_test.py"],
    deps = [
        ":activations",
        ":linears",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit flax.core dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit lingvo.core.layers dependency.
        # Implicit numpy dependency.
        "//praxis:base_hyperparams",
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:pytypes",
        "//praxis:test_utils",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_test(
    name = "shared_layers_test",
    srcs = ["shared_layers_test.py"],
    tags = ["fast_and_reliable_fiddle_integration_test"],
    deps = [
        ":linears",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_hyperparams",
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:test_utils",
    ],
)

pytype_strict_test(
    name = "models_test",
    srcs = ["models_test.py"],
    shard_count = 4,
    deps = [
        ":attentions",
        ":embedding_softmax",
        ":mobilenet",
        ":models",
        ":ngrammer",
        ":resnets",
        ":transformer_models",
        ":transformers",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:decoder_utils",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:pytypes",
        "//praxis:sample_decode",
        "//praxis:test_utils",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_test(
    name = "ngrammer_test",
    srcs = ["ngrammer_test.py"],
    deps = [
        ":bregman",
        ":ngrammer",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit lingvo.core.attention_util dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:test_utils",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_test(
    name = "normalizations_test",
    srcs = ["normalizations_test.py"],
    deps = [
        ":convolutions",
        ":normalizations",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit lingvo.core.bn_layers dependency.
        # Implicit core.conv_layers_with_time_padding dependency.
        # Implicit lingvo.core.layers dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:test_utils",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_test(
    name = "pipeline_gpu_test",
    srcs = ["pipeline_test.py"],
    main = "pipeline_test.py",
    shard_count = 2,
    tags = ["manual"],
    # Internal pipeline_gpu_test tags.
    deps = [
        ":attentions",
        ":normalizations",
        ":pipeline",
        ":transformers",
        # Internal GPU dependency.
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:asserts",
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:test_utils",
    ],
)

# Internal pipeline_test.

pytype_strict_test(
    name = "poolings_test",
    srcs = ["poolings_test.py"],
    deps = [
        ":poolings",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit lingvo.core.batch_major_attention dependency.
        # Implicit lingvo.core.layers dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:test_utils",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_test(
    name = "quantizer_test",
    srcs = ["quantizer_test.py"],
    deps = [
        ":quantizer",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:test_utils",
    ],
)

pytype_strict_test(
    name = "quantizer_objectives_test",
    srcs = ["quantizer_objectives_test.py"],
    deps = [
        ":objectives",
        # Implicit absl.testing.absltest dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:test_utils",
    ],
)

pytype_strict_test(
    name = "rnn_cell_test",
    srcs = ["rnn_cell_test.py"],
    deps = [
        ":rnn_cell",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit lingvo.core.py_utils dependency.
        # Implicit lingvo.core.rnn_cell dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:test_utils",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_test(
    name = "ssm_test",
    srcs = ["ssm_test.py"],
    deps = [
        ":ssm",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:test_utils",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_test(
    name = "stochastics_test",
    srcs = ["stochastics_test.py"],
    deps = [
        ":stochastics",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:test_utils",
    ],
)

pytype_strict_test(
    name = "sequential_test",
    srcs = ["sequential_test.py"],
    deps = [
        ":sequential",
        # Implicit absl.testing.absltest dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:pytypes",
        "//praxis:test_utils",
    ],
)

pytype_strict_test(
    name = "transformers_test",
    size = "large",
    srcs = ["transformers_test.py"],
    shard_count = 20,
    deps = [
        ":activations",
        ":attentions",
        ":transformers",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit lingvo.core.batch_major_attention dependency.
        # Implicit lingvo.core.layers_with_attention dependency.
        # Implicit numpy dependency.
        "//praxis:base_hyperparams",
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:test_utils",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_test(
    name = "transformer_models_test",
    size = "large",
    srcs = ["transformer_models_test.py"],
    shard_count = 8,
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
        # Implicit fiddle dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit lingvo.core.gshard_builder dependency.
        # Implicit numpy dependency.
        "//praxis:base_hyperparams",
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:test_utils",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_test(
    name = "transformer_models_encoder_decoder_test",
    size = "large",
    srcs = ["transformer_models_encoder_decoder_test.py"],
    shard_count = 48,
    deps = [
        ":embedding_softmax",
        ":ngrammer",
        ":transformer_models",
        ":transformers",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit fiddle dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:test_utils",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_library(
    name = "flax_adapter",
    srcs = ["flax_adapter.py"],
    deps = [
        # Implicit flax.core dependency.
        "//praxis:base_layer",
        "//praxis:flax_utils",
        "//praxis:pax_fiddle",
        "//praxis:pytypes",
    ],
)

pytype_strict_test(
    name = "flax_adapter_test",
    srcs = ["flax_adapter_test.py"],
    deps = [
        ":flax_adapter",
        ":normalizations",
        # Implicit absl.testing.absltest dependency.
        # Implicit fiddle dependency.
        # Implicit flax.core dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:pytypes",
        "//praxis:test_utils",
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
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "multi_query_attention",
    srcs = ["multi_query_attention.py"],
    deps = [
        ":attentions",
        ":base_ops",
        ":embedding_softmax",
        ":stochastics",
        # Implicit flax.core dependency.
        # Implicit jax dependency.
        "//praxis:asserts",
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_test(
    name = "multi_query_attention_test",
    srcs = ["multi_query_attention_test.py"],
    deps = [
        ":attentions",
        ":multi_query_attention",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:test_utils",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_test(
    name = "convolutions_test",
    srcs = ["convolutions_test.py"],
    shard_count = 10,
    deps = [
        ":activations",
        ":convolutions",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        # Implicit lingvo.core.cluster_factory dependency.
        # Implicit lingvo.core.conformer_layer dependency.
        # Implicit core.conv_layers_with_time_padding dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:test_utils",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_library(
    name = "searchable",
    srcs = ["searchable.py"],
    deps = [
        # Implicit flax.core dependency.
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:pytypes",
    ],
)

pytype_strict_test(
    name = "searchable_test",
    srcs = ["searchable_test.py"],
    deps = [
        ":linears",
        ":searchable",
        # Implicit absl.testing.absltest dependency.
        # Implicit upb python proto dependency.
        # Implicit jax dependency.
        "//praxis:base_hyperparams",
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:test_utils",
    ],
)

pytype_strict_library(
    name = "sharding",
    srcs = ["sharding.py"],
    deps = [
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:py_utils",
    ],
)

pytype_strict_library(
    name = "grok",
    srcs = ["grok.py"],
    deps = [
        ":activations",
        ":attentions",
        ":checkpoint_policy",
        ":embedding_softmax",
        ":multi_query_attention",
        ":normalizations",
        ":transformer_models",
        ":transformers",
        # Implicit fiddle dependency.
        "//praxis:pax_fiddle",
        "//praxis/layers/injection:fp8_nvidia_gpu",
    ],
)
