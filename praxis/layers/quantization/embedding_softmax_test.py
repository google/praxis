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

"""Test for quantized Embedding and softmax layers."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import pax_fiddle
from praxis import test_utils
from praxis.layers import embedding_softmax
from praxis.layers import quantization
from praxis.layers.quantization import quantization_hparams
from praxis.layers.quantization import utils

instantiate = base_layer.instantiate
WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams
QuantizationParams = quantization_hparams.QuantizationParams
QuantizationMode = quantization_hparams.QuantizationMode
QuantizationType = quantization_hparams.QuantizationType
WeightQuantizationParams = quantization_hparams.WeightQuantizationParams


# TODO(b/283326926): add sync test for all three class of Embedding
# quantizations.


class EmbeddingTest(test_utils.TestCase):
  INPUT_DIMS = 2
  NUM_CLASSES = 3
  WEIGHTS = [[-2, 1], [-1, 2], [0, 3]]
  QUANTIZED_WEIGHTS_SYMMETRIC = {
      QuantizationType.PTQ: [[-127, 64], [-64, 127], [0, 127]],
      QuantizationType.FQ: [[-127, 64], [-64, 127], [0, 127]],
      QuantizationType.AQT: [[-127, 63], [-63, 127], [0, 127]],
      }
  SCALES_SYMMETRIC = [0.01574803, 0.01574803, 0.02362205]
  QUANTIZED_WEIGHTS_ASYMMETRIC = [[-128, 127], [-128, 127], [-128, 127]]
  SCALES_ASYMMETRIC = [0.01176471, 0.01176471, 0.01176471]
  ZPS = {
      QuantizationType.PTQ: [0.494118, -0.505882, -1.505882],
      QuantizationType.FQ: [0.494118, -0.505882, -1.505882],
      QuantizationType.AQT: [0.494102, -0.505898, -1.505898],
      }

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  @parameterized.product(
      use_symmetric=[True, False],
      quantization_type=[
          QuantizationType.PTQ,
          QuantizationType.FQ,
          QuantizationType.AQT,
      ],
      quant_mode=[QuantizationMode.MATERIALIZE, QuantizationMode.TRAINING],
  )
  def test_quantize(self, use_symmetric, quantization_type, quant_mode):
    quantization_option = QuantizationParams(
        quantization_type=quantization_type,
        mode=quant_mode,
        weight_params=WeightQuantizationParams(use_symmetric=use_symmetric),
    )

    p = pax_fiddle.Config(
        quantization.Embedding,
        name='_embedding_q',
        mesh_axis_names=['replica', 'mdl', 'data'],
        weight_split_dims_mapping=base_layer.BaseLayer.WeightSharding(
            wt=['mdl', 'data']
        ),
        quantization=quantization_option,
        input_dims=self.INPUT_DIMS,
        num_classes=self.NUM_CLASSES,
    )

    layer = instantiate(p)
    inputs = np.random.randint(1, p.num_classes, [2, 1])

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = layer.init(prng_key, inputs)
      initial_vars[base_layer.PARAMS]['emb_var'] = np.array(
          self.WEIGHTS, dtype=p.dtype
      )

      res, _ = layer.apply(
          initial_vars, mutable=[], method=layer.quantize_weight
      )
      pspec, _ = layer.apply(
          initial_vars, mutable=[], method=layer.quantized_partition_specs
      )

    # Test quantized weights.
    q_params_weights = res[base_layer.PARAMS]
    weight = q_params_weights.get('emb_var', None)
    weight_scale = q_params_weights.get('emb_var_quantized_scale', None)
    if use_symmetric:
      self.assertArraysEqual(
          weight,
          np.array(
              self.QUANTIZED_WEIGHTS_SYMMETRIC[quantization_type], dtype=np.int8
          ),
      )
      self.assertAllClose(
          weight_scale,
          np.array(self.SCALES_SYMMETRIC, dtype=p.dtype),
      )
    else:
      self.assertArraysEqual(
          weight,
          np.array(self.QUANTIZED_WEIGHTS_ASYMMETRIC, dtype=np.int8),
      )
      self.assertAllClose(
          weight_scale,
          np.array(self.SCALES_ASYMMETRIC, dtype=p.dtype),
      )
      weight_zp = q_params_weights.get('emb_var_quantized_zp', None)
      self.assertAllClose(
          weight_zp,
          np.array(self.ZPS[quantization_type], dtype=p.dtype),
      )

    # Test quantized pspec.
    q_params_pspecs = pspec[base_layer.PARAMS]
    weight_pspec = q_params_pspecs.get('emb_var', None)
    weight_scale_pspec = q_params_pspecs.get('emb_var_quantized_scale', None)
    self.assertEqual(
        weight_pspec,
        base_layer.BoxedPartitionSpec(
            meta=jax.sharding.PartitionSpec('mdl', 'data')
        ),
    )
    self.assertEqual(
        weight_scale_pspec,
        base_layer.BoxedPartitionSpec(meta=jax.sharding.PartitionSpec('mdl')),
    )
    if not use_symmetric:
      weight_zp_pspec = q_params_pspecs.get('emb_var_quantized_zp', None)
      self.assertEqual(
          weight_zp_pspec,
          base_layer.BoxedPartitionSpec(meta=jax.sharding.PartitionSpec('mdl')),
      )

  @parameterized.product(
      quantization_type=[
          QuantizationType.PTQ,
          QuantizationType.AQT,
          QuantizationType.FQ,
      ],
      use_symmetric=[True, False],
      lookup_style=['index', 'matmul'],
  )
  def test_quantized_inference(
      self, quantization_type, use_symmetric, lookup_style
  ):
    quantization_option = QuantizationParams(
        quantization_type=quantization_type,
        mode=QuantizationMode.INFERENCE,
        weight_params=WeightQuantizationParams(use_symmetric=use_symmetric),
    )
    f_p = pax_fiddle.Config(
        embedding_softmax.Embedding,
        name='_embedding',
        mesh_axis_names=['replica', 'mdl', 'data'],
        weight_split_dims_mapping=base_layer.BaseLayer.WeightSharding(
            wt=['mdl', 'data']
        ),
        input_dims=self.INPUT_DIMS,
        num_classes=self.NUM_CLASSES,
        lookup_style=lookup_style,
    )
    q_p = pax_fiddle.Config(
        quantization.Embedding,
        name='_embedding_q',
        mesh_axis_names=['replica', 'mdl', 'data'],
        weight_split_dims_mapping=base_layer.BaseLayer.WeightSharding(
            wt=['mdl', 'data']
        ),
        quantization=quantization_option,
        input_dims=self.INPUT_DIMS,
        num_classes=self.NUM_CLASSES,
        lookup_style=lookup_style,
    )

    q_layer = instantiate(q_p)
    f_layer = instantiate(f_p)

    inputs = np.random.randint(1, q_p.num_classes, [2, 1])

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      q_initial_vars = q_layer.init(prng_key, inputs)
      q_initial_vars[base_layer.PARAMS]['emb_var'] = np.array(
          self.QUANTIZED_WEIGHTS_SYMMETRIC[quantization_type]
          if use_symmetric
          else self.QUANTIZED_WEIGHTS_ASYMMETRIC,
          dtype=np.int8,
      )
      q_initial_vars[base_layer.PARAMS]['emb_var_quantized_scale'] = np.array(
          self.SCALES_SYMMETRIC if use_symmetric else self.SCALES_ASYMMETRIC,
          dtype=q_p.dtype,
      )
      if not use_symmetric:
        q_initial_vars[base_layer.PARAMS]['emb_var_quantized_zp'] = np.array(
            self.ZPS[quantization_type],
            dtype=q_p.dtype,
        )
      q_lookup = q_layer.apply(q_initial_vars, inputs)

      f_initial_vars = f_layer.init(prng_key, inputs)
      f_initial_vars[base_layer.PARAMS]['emb_var'] = np.array(
          self.WEIGHTS, dtype=f_p.dtype
      )
      f_lookup = f_layer.apply(f_initial_vars, inputs)
      self.assertAllClose(q_lookup, f_lookup, rtol=1e-2, atol=1e-2)


class SharedEmbeddingSoftmaxTest(test_utils.TestCase):
  INPUT_DIMS = 2
  NUM_CLASSES = 3
  WEIGHTS = [[-2, -1, 0], [1, 2, 3]]
  QUANTIZED_WEIGHTS_SYMMETRIC = [[-127, -64, 0], [64, 127, 127]]
  SCALES_SYMMETRIC = [0.01574803, 0.01574803, 0.02362205]
  QUANTIZED_WEIGHTS_ASYMMETRIC = [[-128, -128, -128], [127, 127, 127]]
  SCALES_ASYMMETRIC = [0.01176471, 0.01176471, 0.01176471]
  ZPS = [0.494118, -0.505882, -1.505882]

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  @parameterized.product(
      precision=[4, 8],
      quantization_type=[
          QuantizationType.PTQ,
          QuantizationType.AQT,
          QuantizationType.FQ,
      ],
      use_symmetric=[True, False],
      lookup_style=['index', 'matmul'],
      use_native_types=[True, False],
  )
  def test_quantized_inference(
      self,
      precision,
      quantization_type,
      use_symmetric,
      lookup_style,
      use_native_types,
  ):
    dtype = jnp.int4 if use_native_types and precision == 4 else jnp.int8
    quantization_option = QuantizationParams(
        quantization_type=quantization_type,
        mode=QuantizationMode.INFERENCE,
        weight_params=WeightQuantizationParams(
            precision=precision,
            use_symmetric=use_symmetric,
            use_int4_packed_weights=not use_native_types,
            dtype=dtype,
            int4_packed_weights_container_dtype=jnp.int8,
        ),
    )
    f_p = pax_fiddle.Config(
        embedding_softmax.SharedEmbeddingSoftmax,
        name='_shared_embedding_softmax',
        mesh_axis_names=['replica', 'mdl', 'data'],
        weight_split_dims_mapping=base_layer.BaseLayer.WeightSharding(
            wt=['mdl', 'data']
        ),
        input_dims=self.INPUT_DIMS,
        num_classes=self.NUM_CLASSES,
        lookup_style=lookup_style,
    )
    q_p = pax_fiddle.Config(
        quantization.SharedEmbeddingSoftmax,
        name='_shared_embedding_softmax_q',
        mesh_axis_names=['replica', 'mdl', 'data'],
        weight_split_dims_mapping=base_layer.BaseLayer.WeightSharding(
            wt=['mdl', 'data']
        ),
        quantization=quantization_option,
        input_dims=self.INPUT_DIMS,
        num_classes=self.NUM_CLASSES,
    )
    q_layer = instantiate(q_p)
    f_layer = instantiate(f_p)

    inputs = np.random.normal(1.5, 2.0, [2, q_p.input_dims]).astype(np.float32)
    class_weights = np.random.normal(1.5, 2.0, [2, 1])
    class_ids = np.random.randint(1, q_p.num_classes, [2, 1])

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      q_initial_vars = q_layer.init(
          prng_key, inputs, class_weights, class_ids=class_ids
      )
      q_linear_params = q_initial_vars[base_layer.PARAMS]['logits_ffn'][
          'linear'
      ]
      q_linear_params['w'] = np.array(
          self.QUANTIZED_WEIGHTS_SYMMETRIC
          if use_symmetric
          else self.QUANTIZED_WEIGHTS_ASYMMETRIC,
          dtype=np.int8,
      )
      if precision == 4:
        if use_native_types:
          q_linear_params['w'] = jnp.array(q_linear_params['w'], dtype)
        else:
          q_linear_params['w'] = utils.pack_4bit(q_linear_params['w'], 0, dtype)
      q_linear_params['w_quantized_scale'] = np.array(
          self.SCALES_SYMMETRIC if use_symmetric else self.SCALES_ASYMMETRIC,
          dtype=q_p.dtype,
      )
      if not use_symmetric:
        q_linear_params['w_quantized_zp'] = np.array(
            self.ZPS,
            dtype=q_p.dtype,
        )
      q_logits = q_layer.apply(
          q_initial_vars, inputs, method=q_layer.get_logits
      )
      q_lookup = q_layer.apply(
          q_initial_vars, class_ids, method=q_layer.emb_lookup
      )

      f_initial_vars = f_layer.init(
          prng_key, inputs, class_weights, class_ids=class_ids
      )
      f_initial_vars[base_layer.PARAMS]['logits_ffn']['linear']['w'] = np.array(
          self.WEIGHTS, dtype=f_p.dtype
      )
      f_logits = f_layer.apply(
          f_initial_vars, inputs, method=f_layer.get_logits
      )
      f_lookup = f_layer.apply(
          f_initial_vars, class_ids, method=f_layer.emb_lookup
      )
      self.assertEqual(q_logits.shape, f_logits.shape)
      self.assertEqual(q_lookup.shape, f_lookup.shape)
      if precision == 8:
        self.assertAllClose(q_logits, f_logits, rtol=5e-1, atol=5e-2)
        self.assertAllClose(q_lookup, f_lookup, rtol=1e-2, atol=1e-2)


class NClassMajorSharedEmbeddingSoftmaxTest(test_utils.TestCase):
  INPUT_DIMS = 2
  NUM_CLASSES = 3
  WEIGHTS = [[-2, -1, 0], [1, 2, 3]]
  QUANTIZED_WEIGHTS_SYMMETRIC = {
      QuantizationType.PTQ: [[-127, 64], [-64, 127], [0, 127]],
      QuantizationType.FQ: [[-127, 64], [-64, 127], [0, 127]],
      QuantizationType.AQT: [[-127, 63], [-63, 127], [0, 127]],
      }
  SCALES_SYMMETRIC = [0.01574803, 0.01574803, 0.02362205]
  QUANTIZED_WEIGHTS_ASYMMETRIC = [[-128, 127], [-128, 127], [-128, 127]]
  SCALES_ASYMMETRIC = [0.01176471, 0.01176471, 0.01176471]
  ZPS = {
      QuantizationType.PTQ: [0.494118, -0.505882, -1.505882],
      QuantizationType.FQ: [0.494118, -0.505882, -1.505882],
      QuantizationType.AQT: [0.494102, -0.505898, -1.505898],
      }

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  @parameterized.product(
      use_symmetric=[True, False],
      quantization_type=[
          QuantizationType.PTQ,
          QuantizationType.FQ,
          QuantizationType.AQT,
      ],
  )
  def test_quantize(self, use_symmetric, quantization_type):
    quantization_option = QuantizationParams(
        quantization_type=quantization_type,
        mode=QuantizationMode.MATERIALIZE,
        weight_params=WeightQuantizationParams(use_symmetric=use_symmetric),
    )

    p = pax_fiddle.Config(
        quantization.NClassMajorSharedEmbeddingSoftmax,
        name='_nclass_major_shared_embedding_softmax_q',
        mesh_axis_names=['replica', 'mdl', 'data'],
        weight_split_dims_mapping=base_layer.BaseLayer.WeightSharding(
            wt=['mdl', 'data']
        ),
        quantization=quantization_option,
        input_dims=self.INPUT_DIMS,
        num_classes=self.NUM_CLASSES,
    )

    layer = instantiate(p)

    inputs = np.random.normal(1.5, 2.0, [2, p.input_dims]).astype(np.float32)
    class_weights = np.random.normal(1.5, 2.0, [2, 1])
    class_ids = np.random.randint(1, p.num_classes, [2, 1])

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = layer.init(
          prng_key, inputs, class_weights, class_ids=class_ids
      )
      initial_vars[base_layer.PARAMS]['w'] = np.array(
          self.WEIGHTS, dtype=p.dtype
      )

      res, _ = layer.apply(
          initial_vars, mutable=[], method=layer.quantize_weight
      )
      pspec, _ = layer.apply(
          initial_vars, mutable=[], method=layer.quantized_partition_specs
      )

    # Test quantized weights.
    q_params_weights = res[base_layer.PARAMS]
    weight = q_params_weights.get('w', None)
    weight_scale = q_params_weights.get('w_quantized_scale', None)
    if use_symmetric:
      self.assertArraysEqual(
          weight,
          np.array(
              self.QUANTIZED_WEIGHTS_SYMMETRIC[quantization_type], dtype=np.int8
          ),
      )
      self.assertAllClose(
          weight_scale,
          np.array(self.SCALES_SYMMETRIC, dtype=p.dtype),
      )
    else:
      self.assertArraysEqual(
          weight,
          np.array(self.QUANTIZED_WEIGHTS_ASYMMETRIC, dtype=np.int8),
      )
      self.assertAllClose(
          weight_scale,
          np.array(self.SCALES_ASYMMETRIC, dtype=p.dtype),
      )
      weight_zp = q_params_weights.get('w_quantized_zp', None)
      self.assertAllClose(
          weight_zp,
          np.array(self.ZPS[quantization_type], dtype=p.dtype),
      )

    # Test quantized pspec.
    q_params_pspecs = pspec[base_layer.PARAMS]
    weight_pspec = q_params_pspecs.get('w', None)
    weight_scale_pspec = q_params_pspecs.get('w_quantized_scale', None)
    self.assertEqual(
        weight_pspec,
        base_layer.BoxedPartitionSpec(
            meta=jax.sharding.PartitionSpec('data', 'mdl')
        ),
    )
    self.assertEqual(
        weight_scale_pspec,
        base_layer.BoxedPartitionSpec(meta=jax.sharding.PartitionSpec('data')),
    )
    if not use_symmetric:
      weight_zp_pspec = q_params_pspecs.get('w_quantized_zp', None)
      self.assertEqual(
          weight_zp_pspec,
          base_layer.BoxedPartitionSpec(
              meta=jax.sharding.PartitionSpec('data')
          ),
      )

  # TODO(b/283326926): raise training mode error for
  # NClassMajorSharedEmbeddingSoftmax.

  @parameterized.product(
      quantization_type=[
          QuantizationType.PTQ,
          QuantizationType.AQT,
          QuantizationType.FQ,
      ],
      use_symmetric=[True, False],
      lookup_style=['index', 'matmul'],
  )
  def test_quantized_inference(
      self, quantization_type, use_symmetric, lookup_style
  ):
    quantization_option = QuantizationParams(
        quantization_type=quantization_type,
        mode=QuantizationMode.INFERENCE,
        weight_params=WeightQuantizationParams(use_symmetric=use_symmetric),
    )
    f_p = pax_fiddle.Config(
        embedding_softmax.SharedEmbeddingSoftmax,
        name='_shared_embedding_softmax',
        mesh_axis_names=['replica', 'mdl', 'data'],
        weight_split_dims_mapping=base_layer.BaseLayer.WeightSharding(
            wt=['mdl', 'data']
        ),
        input_dims=self.INPUT_DIMS,
        num_classes=self.NUM_CLASSES,
        lookup_style=lookup_style,
    )
    q_p = pax_fiddle.Config(
        quantization.NClassMajorSharedEmbeddingSoftmax,
        name='_nclass_major_shared_embedding_softmax_q',
        mesh_axis_names=['replica', 'mdl', 'data'],
        weight_split_dims_mapping=base_layer.BaseLayer.WeightSharding(
            wt=['mdl', 'data']
        ),
        quantization=quantization_option,
        input_dims=self.INPUT_DIMS,
        num_classes=self.NUM_CLASSES,
        lookup_style=lookup_style,
    )

    q_layer = instantiate(q_p)
    f_layer = instantiate(f_p)

    inputs = np.random.normal(1.5, 2.0, [2, q_p.input_dims]).astype(np.float32)
    class_weights = np.random.normal(1.5, 2.0, [2, 1])
    class_ids = np.random.randint(1, q_p.num_classes, [2, 1])

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      q_initial_vars = q_layer.init(
          prng_key, inputs, class_weights, class_ids=class_ids
      )
      q_initial_vars[base_layer.PARAMS]['w'] = np.array(
          self.QUANTIZED_WEIGHTS_SYMMETRIC[quantization_type]
          if use_symmetric
          else self.QUANTIZED_WEIGHTS_ASYMMETRIC,
          dtype=np.int8,
      )
      q_initial_vars[base_layer.PARAMS]['w_quantized_scale'] = np.array(
          self.SCALES_SYMMETRIC if use_symmetric else self.SCALES_ASYMMETRIC,
          dtype=q_p.dtype,
      )
      if not use_symmetric:
        q_initial_vars[base_layer.PARAMS]['w_quantized_zp'] = np.array(
            self.ZPS[quantization_type],
            dtype=q_p.dtype,
        )
      q_logits = q_layer.apply(
          q_initial_vars, inputs, method=q_layer.get_logits
      )
      q_lookup = q_layer.apply(
          q_initial_vars, class_ids, method=q_layer.emb_lookup
      )

      f_initial_vars = f_layer.init(
          prng_key, inputs, class_weights, class_ids=class_ids
      )
      f_initial_vars[base_layer.PARAMS]['logits_ffn']['linear']['w'] = np.array(
          self.WEIGHTS, dtype=f_p.dtype
      )
      f_logits = f_layer.apply(
          f_initial_vars, inputs, method=f_layer.get_logits
      )
      f_lookup = f_layer.apply(
          f_initial_vars, class_ids, method=f_layer.emb_lookup
      )
      self.assertAllClose(q_logits, f_logits, rtol=5e-1, atol=5e-2)
      self.assertAllClose(q_lookup, f_lookup, rtol=1e-2, atol=1e-2)


if __name__ == '__main__':
  absltest.main()
