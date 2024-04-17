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

"""Tests for Pax pipeline layers.

Note: This test requires 4 GPU cores.
"""

import functools
import itertools

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from jax.experimental.pjit import pjit
import numpy as np
from praxis import asserts
from praxis import base_layer
from praxis import pax_fiddle
from praxis import py_utils
from praxis import test_utils
from praxis.layers import attentions
from praxis.layers import normalizations
from praxis.layers import pipeline
from praxis.layers import transformers

NestedMap = py_utils.NestedMap
WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams
SplitDimsMapping = base_layer.SplitDimsMapping

PARAMS = base_layer.PARAMS

instantiate = base_layer.instantiate


class SingleStageLayer(base_layer.BaseLayer):
  """Stage-parallel dense-relu-dense.

  Attributes:
    model_dim: Model dimension size.
    hidden_dim: Hidden dimension size.
    w_in_mh_sharding: w_in_mh_sharding.
    w_out_hm_sharding: w_out_hm_sharding.
    bsm_sharding: bsm_sharding.
    bsh_sharding: bsh_sharding.
  """
  model_dim: int = 0
  hidden_dim: int = 0
  w_in_mh_sharding: SplitDimsMapping = None
  w_out_hm_sharding: SplitDimsMapping = None
  bsm_sharding: SplitDimsMapping = None
  bsh_sharding: SplitDimsMapping = None

  def setup(self):
    assert self.name
    assert self.model_dim > 0
    assert self.hidden_dim > 0

    self.create_variable(
        'w_in',
        WeightHParams(
            shape=[self.model_dim, self.hidden_dim],
            init=WeightInit.Gaussian(1.0),
            mesh_shape=self.mesh_shape,
            tensor_split_dims_mapping=self.w_in_mh_sharding))
    self.create_variable(
        'w_out',
        WeightHParams(
            shape=[self.hidden_dim, self.model_dim],
            init=WeightInit.Gaussian(1.0),
            mesh_shape=self.mesh_shape,
            tensor_split_dims_mapping=self.w_out_hm_sharding))
    # A counter keeping track of how many times fprop is invoked.
    self.create_variable(
        'counter',
        WeightHParams(
            shape=[],
            dtype=jnp.int32,
            init=WeightInit.Constant(0),
            tensor_split_dims_mapping=()),
        trainable=False)

  def __call__(self, inputs):
    theta = self.theta
    w_in = base_layer.maybe_shard(theta.w_in, self.w_in_mh_sharding,
                                  self.mesh_axis_names)
    w_out = base_layer.maybe_shard(theta.w_out, self.w_out_hm_sharding,
                                   self.mesh_axis_names)
    inputs = base_layer.maybe_shard(inputs, self.bsm_sharding,
                                    self.mesh_axis_names)
    h = jnp.einsum(
        'bsm,mh->bsh', inputs, w_in, precision=jax.lax.Precision.HIGHEST)
    h = jax.nn.relu(h)
    h = base_layer.maybe_shard(h, self.bsh_sharding, self.mesh_axis_names)
    outp = jnp.einsum(
        'bsh,hm->bsm', h, w_out, precision=jax.lax.Precision.HIGHEST)
    # This constant is summed across stages, but averaged across microbatches.
    self.add_aux_loss('one', jnp.array(1.0), 0.5)
    # This summary is averaged across microbatches, and kept as per-stage.
    self.add_summary('one', jnp.array(1.0))
    self.update_var('counter', self.get_var('counter') + 1)
    return base_layer.maybe_shard(outp, self.bsm_sharding, self.mesh_axis_names)


class PipelineTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  @parameterized.parameters([(True, 1, 16, False), (False, 1, 16, False),
                             (True, 2, 16, False), (False, 2, 4, False),
                             (False, 2, 4, True)])
  def test_vmap_single_stage_body(self, stream_io, circular_repeat,
                                  microbatches, circular_share_weights):
    n_devs = jax.device_count()
    if n_devs % 4 != 0:
      stages = n_devs
    else:
      stages = 4
    mesh_shape = [stages, n_devs // stages]
    device_mesh = np.array(jax.local_devices()).reshape(mesh_shape)
    stage_axis = 'stage'
    mdl_axis = 'mdl'
    mesh_axis_names = [stage_axis, mdl_axis]
    w_in_mh_sharding = [None, mdl_axis]
    w_out_hm_sharding = [mdl_axis, None]
    bsm_sharding = [None, None, mdl_axis]
    bsh_sharding = [None, None, mdl_axis]

    model_dim = 128
    hidden_dim = 256
    seq_len = 32
    microbatch_size = 1

    inner_params = pax_fiddle.Config(
        SingleStageLayer,
        model_dim=model_dim,
        hidden_dim=hidden_dim,
        w_in_mh_sharding=w_in_mh_sharding,
        w_out_hm_sharding=w_out_hm_sharding,
        bsm_sharding=bsm_sharding,
        bsh_sharding=bsh_sharding)

    test_inputs = np.ones((microbatches, microbatch_size, seq_len, model_dim))

    is_circular_schedule = circular_repeat != 1

    if is_circular_schedule:
      pipelined_layer_p = pax_fiddle.Config(
          pipeline.CircularLayerwiseShardablePipelined,
          name='pipeline',
          num_stages=stages,
          mesh_axis_names=mesh_axis_names,
          single_stage_body=inner_params,
          stream_io=stream_io,
          circular_repeat=circular_repeat,
          share_weights=circular_share_weights,
          polluting_bubbles_with_nan=True)
    else:
      pipelined_layer_p = pax_fiddle.Config(
          pipeline.LayerwiseShardablePipelined,
          name='pipeline',
          num_stages=stages,
          mesh_axis_names=mesh_axis_names,
          single_stage_body=inner_params,
          stream_io=stream_io,
          polluting_bubbles_with_nan=True)

    pipelined_layer_p.weight_split_dims_mapping.stages = [stage_axis]
    pipelined_layer = instantiate(pipelined_layer_p)
    with jax.sharding.Mesh(device_mesh, mesh_axis_names):

      def init(key, inp):
        return pipelined_layer.init(key, inp)

      pjit_init = pjit(init, in_shardings=None, out_shardings=None)

      prng_key = jax.random.PRNGKey(seed=123)
      weight_hparams = pipelined_layer.abstract_init_with_metadata(test_inputs)
      print('## weight_hparams=', weight_hparams)
      self.assertEqual(set(weight_hparams), {'params', 'non_trainable'})
      w_in_metadata = weight_hparams['params']['body']['w_in']
      self.assertEqual(w_in_metadata.shape, [model_dim, hidden_dim])
      if not is_circular_schedule or circular_share_weights:
        self.assertEqual(w_in_metadata.repeat_prefix, [stages])
        self.assertEqual(w_in_metadata.repeat_prefix_split_dims_mapping,
                         ('stage',))
      else:
        self.assertEqual(w_in_metadata.repeat_prefix, [circular_repeat, stages])
        self.assertEqual(w_in_metadata.repeat_prefix_split_dims_mapping,
                         (None, 'stage'))

      # gpipe:
      #   {'body': {'w_in': (4, 128, 256), 'w_out': (4, 256, 128)}}
      # circular:
      #   {'body': {'w_in': (2, 4, 128, 256), 'w_out': (2, 4, 256, 128)}}
      pipelined_layer_vars = pjit_init(prng_key, test_inputs)
      print('## pipelined_layer_vars=',
            jax.tree.map(lambda x: x.shape, pipelined_layer_vars))

      def loss(v, inp):
        # Need to allow
        result, _ = pipelined_layer.apply(
            v, inp, mutable=[base_layer.NON_TRAINABLE])
        assert result.shape == inp.shape
        return jnp.sum(result**2)

      def get_non_weight_vars(v, inp):
        mutables = [
            base_layer.AUX_LOSS, base_layer.SUMMARIES, base_layer.NON_TRAINABLE
        ]
        _, updated_vars = pipelined_layer.apply(v, inp, mutable=mutables)
        return (updated_vars[base_layer.AUX_LOSS]['body']['one'],
                updated_vars[base_layer.SUMMARIES]['body'],
                updated_vars[base_layer.NON_TRAINABLE]['body']['counter'])

      # Set pjit input/output to be replicated.
      # Rely on sharding annotations within StageParallelLayer.fprop.
      pjit_fprop = pjit(
          jax.grad(loss, allow_int=True), in_shardings=None, out_shardings=None
      )
      result = pjit_fprop(pipelined_layer_vars, test_inputs)
      # Do not care about grad of integer-valued counter.
      del result[base_layer.NON_TRAINABLE]
      aux_loss, summaries, counter = pjit(
          get_non_weight_vars, in_shardings=None, out_shardings=None
      )(pipelined_layer_vars, test_inputs)

    def ref_fn(weights, inputs):
      data = inputs
      w_lmh = weights['params']['body']['w_in']
      w_lhm = weights['params']['body']['w_out']
      for r in range(circular_repeat):
        for i in range(stages):
          h = jnp.einsum(
              'lbsm,mh->lbsh',
              data,
              w_lmh[r][i] if circular_repeat > 1 and not circular_share_weights
              else w_lmh[i],
              precision=jax.lax.Precision.HIGHEST)
          h = jax.nn.relu(h)
          data = jnp.einsum(
              'lbsh,hm->lbsm',
              h,
              w_lhm[r][i] if circular_repeat > 1 and not circular_share_weights
              else w_lhm[i],
              precision=jax.lax.Precision.HIGHEST)

      return jnp.sum(data**2)

    expected = jax.grad(
        ref_fn, allow_int=True)(pipelined_layer_vars, test_inputs)
    # Do not care about grad of integer-valued counter.
    del expected[base_layer.NON_TRAINABLE]
    # TODO(zhangqiaorjc): Improve the numerical stability.
    jax.tree.map(
        functools.partial(self.assertAllClose, atol=2e-2, rtol=2e-2), result,
        expected)
    # Aux loss is summed across stages and circular_repeat layers.
    self.assertArraysEqual(aux_loss.value, float(stages * circular_repeat))
    self.assertArraysEqual(aux_loss.weight,
                           float(stages * circular_repeat / 2.0))
    var_repeats = circular_repeat if not circular_share_weights else 1
    for i in range(var_repeats):
      for j in range(stages):
        if var_repeats > 1:
          key = f'one.circular_layer{i}.stage{j}_scalar'
        else:
          key = f'one.stage{j}_scalar'
        self.assertArraysEqual(summaries[key], 1.0)
    if var_repeats == 1:
      self.assertArraysEqual(
          counter, [microbatches * circular_repeat // var_repeats] * stages)
    else:
      self.assertArraysEqual(counter,
                             [[microbatches] * stages] * circular_repeat)


class PipelinedTransformerTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  @parameterized.parameters([(2, 1), (2, 4), (4, 1), (4, 8)])
  def test_moe_transformer_layer(self, stages, microbatches):
    if jax.device_count() % stages != 0:
      logging.info('Skipping test due to unsupported number of stages.')
      return
    mesh_shape = [stages, jax.device_count() // stages]
    device_mesh = np.array(jax.local_devices()).reshape(mesh_shape)
    stage_axis = 'stage'
    mdl_axis = 'mdl'
    mesh_axis_names = [stage_axis, mdl_axis]

    model_dims = 8
    seq_len = 16
    micro_batch_size = 2
    batch_size = microbatches * micro_batch_size

    e_dim = 4
    num_groups = 4

    stage_body_param = pax_fiddle.Config(
        transformers.StackedTransformer,
        name='moe_body',
        model_dims=model_dims,
        hidden_dims=model_dims * 4,
        num_heads=2,
        mask_self_attention=True,
        num_layers=1,
        moe_layers=[0],
        num_experts=e_dim,
        num_groups=num_groups,
        packed_input=True,
        use_cross_attention=False)
    moe_p = stage_body_param.moe_layer_tpl
    moe_p.input_dims = model_dims
    moe_p.hidden_dims = model_dims * 4
    moe_p.ln_tpl = pax_fiddle.Config(normalizations.RmsNorm)
    moe_p.ln_tpl.direct_scale = True
    moe_p.num_experts = e_dim
    moe_p.num_groups = num_groups
    moe_p.expert_capacity_dim = 0
    moe_p.unadjusted_expert_capacity_factor = 2
    moe_p.internal_gshard_variance_scaling_fan_in_init = True

    pipeline_param = pax_fiddle.Config(
        transformers.PipelinedTransformer,
        name='pipelined_transformer',
        pipeline_stage=stage_body_param,
        num_pipeline_stages=stages,
        num_pipeline_microbatches=None,
        pipeline_microbatch_size=micro_batch_size,
        mesh_axis_names=mesh_axis_names,
    )
    pipeline_param.weight_split_dims_mapping.stages = [stage_axis]
    repeat_body_param = stage_body_param.clone()
    repeat_body_param.num_groups = num_groups * microbatches
    repeat_body_param.moe_layer_tpl.num_groups = num_groups * microbatches

    pipelined_transformer_layer = instantiate(pipeline_param)

    repeat_xformer_param = pax_fiddle.Config(
        transformers.StackedTransformerRepeated,
        name='jax_stacked_transformer_layer_repeated',
        block=repeat_body_param,
        x_times=stages)
    repeat_xformer_layer = instantiate(repeat_xformer_param)

    npy_inputs = np.random.normal(
        1.0, 0.5, [batch_size, seq_len, model_dims]).astype('float32')
    inputs = jnp.asarray(npy_inputs)
    npy_paddings = np.random.randint(0, 1,
                                     [batch_size, seq_len]).astype('float32')
    paddings = jnp.asarray(npy_paddings)
    segment_mask = None
    segment_ids = np.random.randint(0, 3, [batch_size, seq_len])
    segment_mask = attentions.segment_mask(segment_ids, dtype=jnp.float32)

    cross_inputs = None
    cross_paddings = None
    cross_segment_mask = None

    prng_key = jax.random.PRNGKey(seed=123)

    def init(key, *args, **kwargs):
      with base_layer.JaxContext.new_context():
        return pipelined_transformer_layer.init(key, *args, **kwargs)

    def wrapped_fn(v, *args, **kwargs):
      with base_layer.JaxContext.new_context():
        res = pipelined_transformer_layer.apply(v, *args, **kwargs)
      return res

    pjit_fprop = pjit(wrapped_fn, in_shardings=None, out_shardings=None)

    with jax.sharding.Mesh(device_mesh, mesh_axis_names):
      with base_layer.JaxContext.new_context():
        weight_hparams = pipelined_transformer_layer.abstract_init_with_metadata(
            inputs, paddings, segment_mask, cross_inputs, cross_paddings,
            cross_segment_mask)
        print('## weight_hparams=', weight_hparams)
        pjit_init = pjit(init, in_shardings=None, out_shardings=None)
        pipeline_vars = pjit_init(prng_key, inputs, paddings, segment_mask,
                                  cross_inputs, cross_paddings,
                                  cross_segment_mask)
        repeated_vars = py_utils.NestedMap(
            params=py_utils.NestedMap(
                repeat=py_utils.NestedMap(
                    sub=py_utils.NestedMap(
                        x_layers_0=pipeline_vars['params']['pipeline']['body']
                        ['x_layers_0']))))

        asserts.assert_same_structure(
            repeated_vars,
            repeat_xformer_layer.init(prng_key, inputs, paddings, segment_mask,
                                      cross_inputs, cross_paddings,
                                      cross_segment_mask))
        outputs = pjit_fprop(pipeline_vars, inputs, paddings, segment_mask,
                             cross_inputs, cross_paddings, cross_segment_mask)
        outputs = outputs.reshape(batch_size, seq_len, model_dims)
        outputs_repeated = repeat_xformer_layer.apply(
            repeated_vars.ToNestedDict(),
            inputs,
            paddings,
            segment_mask=segment_mask,
            cross_inputs=cross_inputs,
            cross_paddings=cross_paddings,
            cross_segment_mask=cross_segment_mask)

    print('pipelined outputs')
    print(outputs)
    print('repeated outputs')
    print(outputs_repeated)
    self.assertAllClose(outputs, outputs_repeated, atol=1e-5)

  @parameterized.parameters(*list(itertools.product([True, False], repeat=3)))
  def test_pipelined_transformer_layer(self, mask_self_attention, packed_input,
                                       use_cross_attention):
    n_devs = jax.device_count()
    if n_devs % 4 != 0:
      stages = n_devs
    else:
      stages = 4
    mesh_shape = [stages, n_devs // stages]
    device_mesh = np.array(jax.local_devices()).reshape(mesh_shape)
    stage_axis = 'stage'
    mdl_axis = 'mdl'
    mesh_axis_names = [stage_axis, mdl_axis]

    model_dims = 16
    seq_len = np.random.randint(12, 32)
    microbatches = stages * 2
    batch_size = microbatches * 2

    stage_body_param = pax_fiddle.Config(
        transformers.StackedTransformer,
        name='jax_stacked_transformer_layer_repeated',
        model_dims=model_dims,
        hidden_dims=64,
        num_heads=8,
        mask_self_attention=mask_self_attention,
        num_layers=1,
        packed_input=packed_input,
        use_cross_attention=use_cross_attention)

    pipeline_param = pax_fiddle.Config(
        transformers.PipelinedTransformer,
        name='pipelined_transformer',
        pipeline_stage=stage_body_param,
        num_pipeline_stages=stages,
        num_pipeline_microbatches=microbatches,
        mesh_axis_names=mesh_axis_names,
    )
    pipeline_param.weight_split_dims_mapping.stages = [stage_axis]
    pipelined_transformer_layer = instantiate(pipeline_param)

    repeat_xformer_param = pax_fiddle.Config(
        transformers.StackedTransformerRepeated,
        name='jax_stacked_transformer_layer_repeated',
        block=stage_body_param,
        x_times=stages)
    repeat_xformer_layer = instantiate(repeat_xformer_param)

    npy_inputs = np.random.normal(
        1.0, 0.5, [batch_size, seq_len, model_dims]).astype('float32')
    inputs = jnp.asarray(npy_inputs)
    npy_paddings = np.random.randint(0, 1,
                                     [batch_size, seq_len]).astype('float32')
    paddings = jnp.asarray(npy_paddings)
    segment_mask = None
    if packed_input:
      segment_ids = np.random.randint(0, 3, [batch_size, seq_len])
      segment_mask = attentions.segment_mask(segment_ids, dtype=jnp.float32)

    cross_inputs = None
    cross_paddings = None
    cross_segment_mask = None
    if use_cross_attention:
      cross_seq_len = np.random.randint(10, 64)
      npy_cross_inputs = np.random.normal(
          1.0, 0.5, [batch_size, cross_seq_len, model_dims]).astype('float32')
      cross_inputs = jnp.asarray(npy_cross_inputs)
      npy_cross_paddings = np.random.randint(
          0, 1, [batch_size, cross_seq_len]).astype('float32')
      cross_paddings = jnp.asarray(npy_cross_paddings)
      if packed_input:
        source_segment_ids = np.random.randint(0, 3,
                                               [batch_size, cross_seq_len])
        cross_segment_mask = attentions.segment_mask(
            segment_ids, source_segment_ids, dtype=jnp.float32
        )

    def init(key, *args, **kwargs):
      with base_layer.JaxContext.new_context():
        return pipelined_transformer_layer.init(key, *args, **kwargs)

    pjit_init = pjit(init, in_shardings=None, out_shardings=None)

    def wrapped_fn(v, *args, **kwargs):
      with base_layer.JaxContext.new_context():
        res = pipelined_transformer_layer.apply(v, *args, **kwargs)
      return res

    pjit_fprop = pjit(wrapped_fn, in_shardings=None, out_shardings=None)

    with jax.sharding.Mesh(device_mesh, mesh_axis_names):
      with base_layer.JaxContext.new_context():
        prng_key = jax.random.PRNGKey(seed=123)
        pipeline_vars = pjit_init(prng_key, inputs, paddings, segment_mask,
                                  cross_inputs, cross_paddings,
                                  cross_segment_mask)
        repeated_vars = py_utils.NestedMap(
            params=py_utils.NestedMap(
                repeat=py_utils.NestedMap(
                    sub=py_utils.NestedMap(
                        x_layers_0=pipeline_vars['params']['pipeline']['body']
                        ['x_layers_0']))))

        asserts.assert_same_structure(
            repeated_vars,
            repeat_xformer_layer.init(
                prng_key,
                inputs,
                paddings,
                segment_mask=segment_mask,
                cross_inputs=cross_inputs,
                cross_paddings=cross_paddings,
                cross_segment_mask=cross_segment_mask))
        # TODO(zhangqiaorjc): pjit does not yet support kwargs.
        outputs = pjit_fprop(pipeline_vars, inputs, paddings, segment_mask,
                             cross_inputs, cross_paddings, cross_segment_mask)

        outputs_repeated = repeat_xformer_layer.apply(
            repeated_vars.ToNestedDict(),
            inputs,
            paddings,
            segment_mask=segment_mask,
            cross_inputs=cross_inputs,
            cross_paddings=cross_paddings,
            cross_segment_mask=cross_segment_mask)
        self.assertAllClose(outputs, outputs_repeated, atol=1e-2, rtol=1e-2)


if __name__ == '__main__':
  absltest.main()
