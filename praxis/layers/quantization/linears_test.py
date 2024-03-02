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

"""Tests for quantized linears."""

import copy
import itertools
from typing import Any, Sequence

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import pax_fiddle
from praxis import test_utils
from praxis.layers import linears
from praxis.layers.quantization import linears as qlinears
from praxis.layers.quantization import operations as qoperations
from praxis.layers.quantization import quantization_hparams
from praxis.layers.quantization import utils as qutils


instantiate = base_layer.instantiate
WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams
QuantizationParams = quantization_hparams.QuantizationParams
QuantizationMode = quantization_hparams.QuantizationMode
QuantizationType = quantization_hparams.QuantizationType

PARAMS = base_layer.PARAMS
NON_TRAINABLE = base_layer.NON_TRAINABLE
SUMMARIES = base_layer.SUMMARIES


def _generate_quantization_types_modes() -> Sequence[dict[str, Any]]:
  keys = ['testcase_name', 'quantization_type', 'mode', 'dtype', 'precision']
  types = [QuantizationType.PTQ, QuantizationType.FQ, QuantizationType.AQT]
  modes = [QuantizationMode.INFERENCE, QuantizationMode.TRAINING]
  dtypes = [jnp.int8, jnp.uint8]
  precisions = [8, 4]

  cases = []
  for case in itertools.product(types, modes, dtypes, precisions):
    name = (
        case[0].value
        + '_'
        + case[1].value
        + '_'
        + str(case[2])
        + '_'
        + str(case[3])
    )
    cases.append([name] + list(case))

  return [dict(zip(keys, case)) for case in cases]


def _generate_quantization_types_symmetric() -> Sequence[dict[str, Any]]:
  keys = [
      'testcase_name',
      'quantization_type',
      'use_symmetric',
      'precision',
      'use_int4_packed_weights',
      'int4_packed_weights_container_dtype',
  ]
  quantization_type = [
      QuantizationType.PTQ,
      QuantizationType.FQ,
      QuantizationType.AQT,
  ]
  precision = [4, 8]
  use_symmetric = [True, False]
  use_int4_packed_weights = [True, False]
  int4_packed_weights_container_dtype = [jnp.int32, jnp.int8]

  cases = []
  for case in itertools.product(
      quantization_type,
      use_symmetric,
      precision,
      use_int4_packed_weights,
      int4_packed_weights_container_dtype,
  ):
    quant_type = case[0].value
    is_symmetric = 'symmetric' if case[1] else 'asymmetric'
    prec = str(case[2])
    pack = 'pack' if case[3] else 'no_pack'
    dtype = str(case[4])
    name = '_'.join([quant_type, is_symmetric, prec, pack, dtype])

    if not (prec == '8' and pack):  # Packing of int8 is not supported.
      cases.append([name] + list(case))

  return [dict(zip(keys, case)) for case in cases]


class QuantizedLinearTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  @parameterized.named_parameters(_generate_quantization_types_modes())
  def test_linear_quantized(self, quantization_type, mode, dtype, precision):
    p = pax_fiddle.Config(
        qlinears.Linear,
        name='_linear',
        input_dims=8,
        output_dims=4,
        quantization=QuantizationParams(
            quantization_type=quantization_type,
            mode=mode,
            weight_params=quantization_hparams.WeightQuantizationParams(
                dtype=dtype,
                precision=precision,
            ),
        ),
    )
    linear = instantiate(p)
    inputs = jnp.array(
        [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]],
        dtype=p.dtype,
    )
    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = linear.init(prng_key, inputs)
      outputs = linear.apply(initial_vars, inputs)
    self.assertEqual(outputs.shape, (2, 4))
    if mode == QuantizationMode.INFERENCE:
      self.assertAllClose(jnp.full((2, 4), 0.0), outputs)
    else:
      self.assertRaises(AssertionError, self.assertAllClose,
                        jnp.full((2, 4), 0.0, dtype=p.dtype), outputs)

  def test_linear_aqt_quantized(self):
    p_q = pax_fiddle.Config(
        qlinears.Linear,
        name='_linear_q',
        quantization=QuantizationParams(
            quantization_type=QuantizationType.AQT,
            mode=QuantizationMode.TRAINING,
            act_params=quantization_hparams.ActQuantizationParams(precision=3),
            weight_params=quantization_hparams.WeightQuantizationParams(
                precision=2,
                add_scale_eps=False,
            ),
        ),
    )
    p_q.input_dims = 3
    p_q.output_dims = 2

    inputs = np.array(
        [
            [-7.0, 4.01, 4.01],
            [-7.0, 0.01, -4.01],
        ],)

    weight = np.array(
        [
            [-1.5, 0.99],
            [-0.99, 0],
            [-0.01, 1.5]
        ],)
    expected_output = np.array(
        [
            [3.5, -3.5],
            [10.5, -17.5]
        ])

    linear_q = instantiate(p_q)
    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars_q = linear_q.init(prng_key, inputs)
      initial_vars_q['params']['w'] = weight
      outputs_q = linear_q.apply(initial_vars_q, inputs)
    self.assertAllClose(expected_output, outputs_q)

  def _get_layer_and_inputs(self):
    p_q = pax_fiddle.Config(
        qlinears.Linear,
        name='_linear_q',
        quantization=QuantizationParams(
            quantization_type=QuantizationType.AQT,
            mode=QuantizationMode.TRAINING,
            act_params=quantization_hparams.ActQuantizationParams(precision=3),
            weight_params=quantization_hparams.WeightQuantizationParams(
                precision=2,
                use_step_count=True,
            ),
        ),
    )
    p_q.input_dims = 4
    p_q.output_dims = 4

    inputs = np.random.normal(size=[4, 4, p_q.input_dims]).astype(np.float32)

    linear_q = instantiate(p_q)
    return linear_q, inputs

  def test_linear_step_count_in_eval(self):
    linear_q, inputs = self._get_layer_and_inputs()
    context_params = base_layer.JaxContext.HParams(do_eval=True)
    with base_layer.JaxContext.new_context(hparams=context_params):
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars_q = linear_q.init(prng_key, inputs)
      _, updated_variables = linear_q.apply(
          initial_vars_q, inputs, mutable=[NON_TRAINABLE])
    # In eval mode step_count is not changed
    self.assertArraysEqual(updated_variables[NON_TRAINABLE]['step_count'],
                           np.array([0]))

  def test_linear_step_count_in_train(self):
    linear_q, inputs = self._get_layer_and_inputs()
    context_params = base_layer.JaxContext.HParams(do_eval=False)
    with base_layer.JaxContext.new_context(hparams=context_params):
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars_q = linear_q.init(prng_key, inputs)
      _, updated_vars = linear_q.apply(
          initial_vars_q, inputs, mutable=[PARAMS, NON_TRAINABLE, SUMMARIES]
      )
    self.assertArraysEqual(
        updated_vars[NON_TRAINABLE]['step_count'], np.array([1])
    )
    self.assertArraysEqual(
        updated_vars[SUMMARIES]['step_count_scalar'], np.array([0])
    )

    with base_layer.JaxContext.new_context(hparams=context_params):
      _, updated_vars = linear_q.apply(
          updated_vars, inputs, mutable=[PARAMS, NON_TRAINABLE, SUMMARIES]
      )
    self.assertArraysEqual(
        updated_vars[NON_TRAINABLE]['step_count'], np.array([2])
    )
    self.assertArraysEqual(
        updated_vars[SUMMARIES]['step_count_scalar'], np.array([1])
    )

  def test_linear_calibration(self):
    p_q = pax_fiddle.Config(
        qlinears.Linear,
        name='_linear_q',
        quantization=QuantizationParams(
            mode=QuantizationMode.CALIB,
            quantization_type=QuantizationType.FR,
        ),
        input_dims=3,
        output_dims=3,
    )
    linear_q = instantiate(p_q)
    inputs = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=jnp.float32)

    context_params = base_layer.JaxContext.HParams(do_eval=False)
    with base_layer.JaxContext.new_context(hparams=context_params):
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars_q = linear_q.init(prng_key, inputs)
      self.assertArraysEqual(
          initial_vars_q[NON_TRAINABLE]['framestat'],
          jnp.array([0], dtype=jnp.bfloat16),
      )
      _, updated_vars = linear_q.apply(
          initial_vars_q, inputs, mutable=[PARAMS, NON_TRAINABLE, SUMMARIES]
      )
      self.assertArraysEqual(
          updated_vars[NON_TRAINABLE]['framestat'],
          jnp.array([9], dtype=jnp.bfloat16),
      )

      # No grad.
      def loss(params, inputs):
        return jnp.sum(linear_q.apply(params, inputs)[0])

      grad = jax.grad(loss)(initial_vars_q, inputs)
      self.assertArraysEqual(
          grad['params']['w'],
          jnp.zeros_like(initial_vars_q['params']['w'], dtype=jnp.float32),
      )

  def test_int4_weight_init(self):
    p = pax_fiddle.Config(
        qlinears.Linear,
        name='linear',
        input_dims=16,
        output_dims=32,
        quantization=QuantizationParams(
            mode=QuantizationMode.INFERENCE,
            weight_params=quantization_hparams.WeightQuantizationParams(
                precision=4,
                dtype=jnp.int4,
                use_int4_packed_weights=False,
            ),
            act_params=quantization_hparams.ActQuantizationParams(precision=4),
        ),
    )
    linear = instantiate(p)
    with base_layer.JaxContext.new_context():
      inputs = jnp.zeros([1, p.input_dims], dtype=jnp.float32)
      linear_vars = linear.init(jax.random.PRNGKey(123), inputs)
      self.assertEqual(linear_vars['params']['w'].dtype, jnp.int4)
      linear.apply(linear_vars, inputs)

  @parameterized.product(
      input_dim=[64, 256, 1024],
      apply_jit=[False, True],
      jit_backend=['cpu', None],
  )
  def test_int4_packed_weight_equality(self, input_dim, apply_jit, jit_backend):
    p_int32_packed = pax_fiddle.Config(
        qlinears.Linear,
        name='_linear',
        input_dims=input_dim,
        output_dims=128 * 8,
        quantization=QuantizationParams(
            quantization_type=QuantizationType.PTQ,
            mode=QuantizationMode.INFERENCE,
            weight_params=quantization_hparams.WeightQuantizationParams(
                precision=4,
                use_int4_packed_weights=True,
                dtype=jnp.int8,
                int4_packed_weights_container_dtype=jnp.int32,
            ),
        ),
    )
    p_unpacked = p_int32_packed.clone()
    p_unpacked.quantization.weight_params.use_int4_packed_weights = False
    p_int8_packed = p_int32_packed.clone()
    p_int8_packed.quantization.weight_params.int4_packed_weights_container_dtype = (
        jnp.int8
    )
    linear_int32_packed = instantiate(p_int32_packed)
    linear_int8_packed = instantiate(p_int8_packed)
    linear_unpacked = instantiate(p_unpacked)
    inputs = np.random.normal(size=[1, 1, p_int32_packed.input_dims]).astype(
        np.float32
    )

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      w = jax.random.randint(
          prng_key,
          shape=[p_int32_packed.input_dims, p_int32_packed.output_dims],
          minval=qoperations.get_min_max(4)[0],
          maxval=qoperations.get_min_max(4)[1] + 1,
          dtype=jnp.int8,
      )
      s = jax.random.uniform(
          prng_key, shape=[p_int32_packed.output_dims], dtype=jnp.float32
      )
      packed_4bit_in_int32 = qutils.pack_4bit(w, 0, packed_dtype=jnp.int32)
      packed_4bit_in_int8 = qutils.pack_4bit(w, 0, packed_dtype=jnp.int8)
      self.assertArraysEqual(
          w,
          qutils.unpack_4bit(packed_4bit_in_int32, 0, jnp.int8),
      )
      self.assertArraysEqual(
          w, qutils.unpack_4bit(packed_4bit_in_int8, 0, jnp.int8)
      )

      # Same weights packed in different format
      packed_int32_vars = {
          'params': {
              'w': packed_4bit_in_int32,
              'w_quantized_scale': s,
          }
      }
      packed_int8_vars = {
          'params': {
              'w': packed_4bit_in_int8,
              'w_quantized_scale': s,
          }
      }
      unpacked_vars = {'params': {'w': w, 'w_quantized_scale': s}}
      packed_int32_apply = (
          jax.jit(linear_int32_packed.apply, backend=jit_backend)
          if apply_jit
          else linear_int32_packed.apply
      )
      packed_int8_apply = (
          jax.jit(linear_int8_packed.apply, backend=jit_backend)
          if apply_jit
          else linear_int8_packed.apply
      )
      unpacked_apply = (
          jax.jit(linear_unpacked.apply, backend=jit_backend)
          if apply_jit
          else linear_unpacked.apply
      )
      packed_int32_otuput = packed_int32_apply(packed_int32_vars, inputs)
      packed_int8_otuput = packed_int8_apply(packed_int8_vars, inputs)
      unpacked_output = unpacked_apply(unpacked_vars, inputs)
      bf16_epsilon = float(jnp.finfo(jnp.bfloat16).eps)
      self.assertAllClose(
          unpacked_output,
          packed_int32_otuput,
          rtol=bf16_epsilon,
          atol=bf16_epsilon,
      )
      self.assertAllClose(
          unpacked_output,
          packed_int8_otuput,
          rtol=bf16_epsilon,
          atol=bf16_epsilon,
      )


class QuantizedLinearsSyncTest(test_utils.TestCase):
  """Sync tests between quantized Linear and regular Linear.

  Quantized Linear is expected to be identical to regular linear when running
  with mode=TRAINING.
  """

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  def run_and_compare(self, p_f, p_q, inputs):
    linear_f = instantiate(p_f)
    linear_q = instantiate(p_q)
    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars_f = linear_f.init(prng_key, inputs)
      initial_vars_q = linear_q.init(prng_key, inputs)
      outputs_f = linear_f.apply(initial_vars_f, inputs)
      outputs_q = linear_q.apply(initial_vars_q, inputs)
    self.assertAllClose(outputs_f, outputs_q)

  @parameterized.parameters(
      QuantizationParams(mode=QuantizationMode.TRAINING), None
  )
  def test_linear_ptq_quantized(self, quantization):
    p_f = pax_fiddle.Config(linears.Linear, name='_linear_f')
    p_q = pax_fiddle.Config(
        qlinears.Linear,
        name='_linear_q',
        quantization=quantization,
    )
    for p in [p_f, p_q]:
      p.input_dims = 16
      p.output_dims = 24

    inputs = np.random.normal(1.5, 2.0, [5, 16]).astype(np.float32)
    self.run_and_compare(p_f, p_q, inputs)

  def test_linear_quantized_in_inference_mode(self):
    p_f = pax_fiddle.Config(linears.Linear, name='_linear_f')
    p_q = pax_fiddle.Config(
        qlinears.Linear,
        name='_linear_q',
        quantization=QuantizationParams(mode=QuantizationMode.INFERENCE),
    )
    for p in [p_f, p_q]:
      p.input_dims = 4
      p.output_dims = 2

    inputs = jax.random.normal(jax.random.PRNGKey(0), (3, 4)).astype(
        jnp.float32
    )
    quantized_weight = jax.random.randint(
        jax.random.PRNGKey(0), (4, 2), minval=-128, maxval=127, dtype=jnp.int8
    )
    w_scale = jnp.array([0.5, 2.0], dtype=jnp.float32)
    weight_rescaled = quantized_weight * w_scale

    linear_f = instantiate(p_f)
    linear_q = instantiate(p_q)
    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars_f = linear_f.init(prng_key, inputs)
      initial_vars_q = linear_q.init(prng_key, inputs)
      initial_vars_f['params']['w'] = weight_rescaled
      initial_vars_q['params']['w'] = quantized_weight
      initial_vars_q['params']['w_quantized_scale'] = w_scale
      outputs_f = linear_f.apply(initial_vars_f, inputs)
      outputs_q = linear_q.apply(initial_vars_q, inputs)
    self.assertAllClose(outputs_f, outputs_q)


class QuantizeLinearTest(test_utils.TestCase):
  """Quantize Linear."""

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  @parameterized.named_parameters(_generate_quantization_types_symmetric())
  def test_quantize_linear(
      self,
      quantization_type,
      use_symmetric,
      precision,
      use_int4_packed_weights,
      int4_packed_weights_container_dtype,
  ):
    p = pax_fiddle.Config(
        qlinears.Linear,
        name='_linear_q',
        mesh_axis_names=['replica', 'mdl', 'data'],
        weight_split_dims_mapping=base_layer.BaseLayer.WeightSharding(
            wt=['mdl', 'data']
        ),
        quantization=QuantizationParams(
            quantization_type=quantization_type,
            mode=QuantizationMode.TRAINING,
            weight_params=quantization_hparams.WeightQuantizationParams(
                precision=precision,
                use_symmetric=use_symmetric,
                use_int4_packed_weights=use_int4_packed_weights,
                dtype=jnp.int8,
                int4_packed_weights_container_dtype=int4_packed_weights_container_dtype,
            ),
        ),
    )
    p.input_dims = 8
    p.output_dims = 8
    layer = instantiate(p)

    inputs = np.random.normal(1.5, 2.0, [3, p.input_dims]).astype(np.float32)

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = layer.init(prng_key, inputs)

      res, _ = layer.apply(
          initial_vars, mutable=[], method=layer.quantize_weight)
      # Check ParititionSpecs.
      pspec, _ = layer.apply(
          initial_vars, mutable=[], method=layer.quantized_partition_specs
      )

    shapes = jax.tree_map(lambda x: x.shape, res)
    types = jax.tree_map(lambda x: x.dtype, res)

    pack_reduction = 1
    if use_int4_packed_weights:
      if int4_packed_weights_container_dtype == jnp.int32 and precision == 4:
        pack_reduction = 8
      elif int4_packed_weights_container_dtype == jnp.int8 and precision == 4:
        pack_reduction = 2

    expected_shape = {
        base_layer.PARAMS: {
            'w': (p.input_dims // pack_reduction, p.output_dims),
            'w_quantized_scale': (p.output_dims,),
        }
    }
    dtype = jnp.int8
    if use_int4_packed_weights:
      dtype = int4_packed_weights_container_dtype
    expected_types = {
        base_layer.PARAMS: {'w': dtype, 'w_quantized_scale': p.dtype}
    }
    expected_pspec = {
        'params': {
            'w': base_layer.BoxedPartitionSpec(
                meta=jax.sharding.PartitionSpec('mdl', 'data')
            ),
            # TODO(pax): Replicated scale runs faster on large models.
            'w_quantized_scale': base_layer.BoxedPartitionSpec(
                meta=jax.sharding.PartitionSpec('data')
            ),
        }
    }

    if not use_symmetric:
      expected_shape[base_layer.PARAMS]['w_quantized_zp'] = (p.output_dims,)
      expected_types[base_layer.PARAMS]['w_quantized_zp'] = p.dtype
      # TODO(pax): Replicated zp runs faster on large models.
      expected_pspec['params']['w_quantized_zp'] = (
          base_layer.BoxedPartitionSpec(meta=jax.sharding.PartitionSpec('data'))
      )

    self.assertEqual(shapes, expected_shape)
    self.assertEqual(types, expected_types)
    self.assertEqual(pspec, expected_pspec)

  def test_aqt_quantize_weight(self):
    p = pax_fiddle.Config(
        qlinears.Linear,
        name='_linear_q',
        quantization=QuantizationParams(
            quantization_type=QuantizationType.AQT,
            mode=QuantizationMode.TRAINING,
            act_params=None,
            weight_params=quantization_hparams.WeightQuantizationParams(
                precision=3,
                stop_scale_gradient=True,
                add_scale_eps=False,
            ),
        ),
    )

    p.input_dims = 3
    p.output_dims = 3
    layer = instantiate(p)

    inputs = np.random.normal(1.5, 2.0, [2, 3]).astype(np.float32)
    weight = np.array(
        [
            [-7.0, -1.01, 1.01],
            [-4.01, 3.50, 0.99],
            [-1.01, 1.99, -1.75],
        ],)
    q_weight = np.array(
        [
            [-3, -1, 2],
            [-2, 3, 2],
            [0, 2, -3]
        ], dtype=np.int8)
    expected_scale = jnp.array([2.333333, 1.166667, 0.583333], dtype=p.dtype)

    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = layer.init(prng_key, inputs)
      initial_vars['params']['w'] = weight

      res, _ = layer.apply(
          initial_vars, mutable=[], method=layer.quantize_weight
      )

    self.assertArraysEqual(res['params']['w'], q_weight)
    self.assertAllClose(
        res['params']['w_quantized_scale'], expected_scale, atol=1e-6
    )


class FactorizedLinearTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  @parameterized.named_parameters(
      ('1', 1),
      ('2', 2),
  )
  def test_linear_factorized(self, rank):
    p = pax_fiddle.Config(
        qlinears.Linear,
        name='_linear',
        input_dims=8,
        output_dims=4,
        rank=rank,
    )
    linear = instantiate(p)
    inputs = jnp.array(
        [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]],
        dtype=p.dtype,
    )
    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = linear.init(prng_key, inputs)
      outputs = linear.apply(initial_vars, inputs)
    self.assertEqual(outputs.shape, (2, 4))


class SubChannelLinearTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(12345)

  @parameterized.product(
      quantization_type=[
          QuantizationType.PTQ,
          QuantizationType.FQ,
      ],
      num_bits=[4, 8],
      use_symmetric=[True, False],
  )
  def test_quantize(self, quantization_type, num_bits, use_symmetric):
    batch_dims = 8
    input_dims = 256
    block_size = 16
    output_dims = 128
    tol = 0.01
    if num_bits == 4:
      tol = 0.15
    # postfix sc := sub-channel
    # postfix pc := per-channel
    p_training_sc = pax_fiddle.Config(
        qlinears.Linear,
        name='_linear',
        input_dims=input_dims,
        output_dims=output_dims,
        mesh_axis_names=['replica', 'mdl', 'data'],
        weight_split_dims_mapping=base_layer.BaseLayer.WeightSharding(
            wt=['mdl', None],
        ),
        quantization=QuantizationParams(
            quantization_type=quantization_type,
            mode=QuantizationMode.TRAINING,
            weight_params=quantization_hparams.WeightQuantizationParams(
                precision=num_bits,
                use_int4_packed_weights=True,
                int4_packed_weights_container_dtype=jnp.int32,
                block_size=block_size,
                use_symmetric=use_symmetric,
            ),
        ),
    )
    p_training_pc = copy.deepcopy(p_training_sc)
    p_inference_sc = copy.deepcopy(p_training_sc)
    p_inference_pc = copy.deepcopy(p_training_sc)
    p_training_pc.quantization.weight_params.block_size = 0
    p_inference_pc.quantization.weight_params.block_size = 0
    p_inference_sc.quantization.mode = QuantizationMode.INFERENCE
    p_inference_pc.quantization.mode = QuantizationMode.INFERENCE
    training_sc = instantiate(p_training_sc)
    training_pc = instantiate(p_training_pc)
    inference_pc = instantiate(p_inference_pc)
    inference_sc = instantiate(p_inference_sc)
    inputs = np.random.rand(batch_dims, input_dims).astype(jnp.float32)
    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=42)
      training_vars = training_sc.init(prng_key, inputs)
      inference_vars_pc, _ = training_pc.apply(
          training_vars, mutable=[], method=training_pc.quantize_weight
      )
      inference_vars_sc, _ = training_sc.apply(
          training_vars, mutable=[], method=training_sc.quantize_weight
      )
      expected_output = training_sc.apply(training_vars, inputs)
      quantized_output_pc = inference_pc.apply(inference_vars_pc, inputs)
      quantized_output_sc = inference_sc.apply(inference_vars_sc, inputs)
      distortion_pc = jnp.sum(jnp.square(quantized_output_pc - expected_output))
      distortion_sc = jnp.sum(jnp.square(quantized_output_sc - expected_output))
      training_pspec, _ = training_sc.apply(
          training_vars,
          mutable=[],
          method=training_sc.quantized_partition_specs,
      )
      inference_pspec, _ = inference_sc.apply(
          inference_vars_sc,
          mutable=[],
          method=inference_sc.quantized_partition_specs,
      )

    # Sub-channel should be more accurate than per-channel.
    self.assertLess(distortion_sc, distortion_pc)
    self.assertAllClose(
        expected_output, quantized_output_sc, rtol=tol, atol=tol
    )
    expected_pspec = {
        'params': {
            'w': base_layer.BoxedPartitionSpec(
                meta=jax.sharding.PartitionSpec('mdl', None, None)
            ),
            'w_quantized_scale': base_layer.BoxedPartitionSpec(
                meta=jax.sharding.PartitionSpec('mdl', None)
            ),
        }
    }
    if not use_symmetric:
      expected_pspec['params']['w_quantized_zp'] = (
          base_layer.BoxedPartitionSpec(
              meta=jax.sharding.PartitionSpec('mdl', None)
          )
      )
    self.assertEqual(expected_pspec, training_pspec)
    self.assertEqual(expected_pspec, inference_pspec)


class LinearLoRATest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  @parameterized.parameters(None, 'pre', 'mid', 'post')
  def test_linear_lora(self, norm_order):
    p = pax_fiddle.Config(
        qlinears.LinearLoRA,
        name='_linear',
        input_dims=8,
        output_dims=4,
        lora_rank=2,
        norm_order=norm_order,
    )
    linear = instantiate(p)
    inputs = jnp.array(
        [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]],
        dtype=p.dtype,
    )
    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = linear.init(prng_key, inputs)
      outputs = linear.apply(initial_vars, inputs)
    self.assertEqual(outputs.shape, (2, 4))
    self.assertEqual(initial_vars['params']['w_left'].shape, (8, 2))
    self.assertEqual(initial_vars['params']['w_right'].shape, (2, 4))


if __name__ == '__main__':
  absltest.main()
