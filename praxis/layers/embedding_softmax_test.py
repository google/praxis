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

"""Tests for Praxis embedding and softmax layers."""

import itertools

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from lingvo.core import layers as lingvo_layers
import numpy as np
from praxis import base_layer
from praxis import py_utils
from praxis import test_utils
from praxis.layers import embedding_softmax
import tensorflow.compat.v2 as tf

instantiate = base_layer.instantiate
to_np = test_utils.to_np

NON_TRAINABLE = base_layer.NON_TRAINABLE
SUMMARIES = base_layer.SUMMARIES


class TokenCounterTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)
    tf.random.set_seed(123)

  def test_token_counter(self):
    test_layer_p = embedding_softmax.TokenCounter.HParams(name='tc')
    layer = instantiate(test_layer_p)

    prng_key = jax.random.PRNGKey(seed=1234)
    prng_key, init_key = jax.random.split(prng_key)
    inputs = np.zeros([100, 100], dtype=jnp.int32)
    paddings = np.zeros([100, 100])

    with base_layer.JaxContext.new_context():
      initial_vars = layer.init(init_key, inputs, paddings)
      logging.info('initial_vars: %s', initial_vars)
      _, updated_variables = layer.apply(
          initial_vars, inputs, paddings, mutable=[NON_TRAINABLE, SUMMARIES])
    new_vars = updated_variables
    logging.info('new_vars: %s', new_vars)

    non_trainable_vars = new_vars[NON_TRAINABLE]
    logging.info(non_trainable_vars)
    tf.nest.assert_same_structure(non_trainable_vars, {
        'approx_total_tokens_mm': None,
    })
    self.assertAllClose(0.01, non_trainable_vars['approx_total_tokens_mm'])

    with base_layer.JaxContext.new_context():
      _, updated_vars = layer.apply(
          new_vars, inputs, paddings, mutable=[NON_TRAINABLE, SUMMARIES])
    non_trainable_vars = updated_vars[NON_TRAINABLE]
    summaries = updated_vars[SUMMARIES]
    tf.nest.assert_same_structure(summaries, {
        'approx_total_tokens_mm_scalar': None,
    })
    logging.info(summaries)
    self.assertAllClose(0.01, summaries['approx_total_tokens_mm_scalar'])
    self.assertAllClose(0.02, non_trainable_vars['approx_total_tokens_mm'])


class EmbeddingSoftmaxTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)
    tf.random.set_seed(123)

  @parameterized.parameters(('index', True), ('index', False), ('matmul', True),
                            ('matmul', False))
  def test_single_sharded_embedding_layer(self, lookup_style, scale_sqrt_depth):
    p = embedding_softmax.Embedding.HParams(
        name='jax_emb_lookup',
        num_classes=10,
        input_dims=40,
        lookup_style=lookup_style,
        scale_sqrt_depth=scale_sqrt_depth)
    emb_layer = instantiate(p)
    npy_input = np.random.randint(0, p.num_classes, [10, 20]).astype('int32')
    inputs = jnp.asarray(npy_input)
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = emb_layer.init(prng_key, inputs, method=emb_layer.emb_lookup)
    outputs = emb_layer.apply(initial_vars, inputs, method=emb_layer.emb_lookup)
    # Test whether tf Embedding layer returns same output
    # Modify initial_vars to use TF compatible params
    tf_initial_vars = py_utils.NestedMap.FromNestedDict(initial_vars['params'])
    tf_p = lingvo_layers.SingleShardEmbeddingLayer.Params().Set(
        name='tf_emb_lookup',
        vocab_size=p.num_classes,
        embedding_dim=p.input_dims,
        scale_sqrt_depth=scale_sqrt_depth)
    tf_emb_layer = tf_p.Instantiate()
    tf_output = tf_emb_layer.FProp(tf_initial_vars,
                                   tf.constant(inputs, dtype=tf.int32))
    np_outputs = to_np(outputs)
    tf_np_outputs = to_np(tf_output)
    self.assertAllClose(tf_np_outputs, np_outputs, atol=1e-6)

  @parameterized.parameters((0., True, False, 0), (0., False, True, 0),
                            (1.0, True, False, 0.1), (1.0, False, True, 0.1))
  def test_single_sharded_softmax_layer(self, soft_cap_logits, use_class_ids,
                                        use_class_probabilities,
                                        label_smoothing_prob):
    if use_class_ids:
      class_ids = np.random.randint(0, 50, [8, 10, 1])
    else:
      class_ids = None
    if use_class_probabilities:
      class_probabilities = np.random.normal(1.5, 2.0, [8, 10, 50])
    else:
      class_probabilities = None
    p = embedding_softmax.FullSoftmax.HParams(
        name='jax_softmax',
        num_classes=50,
        input_dims=40,
        soft_cap_logits=soft_cap_logits,
        label_smoothing_prob=label_smoothing_prob)
    softmax_layer = instantiate(p)
    npy_input = np.random.normal(1.5, 2.0, [8, 10, p.input_dims])
    inputs = jnp.asarray(npy_input)
    class_weights = np.random.normal(1.5, 2.0, [8, 10, 1])
    if class_probabilities is not None:
      class_probabilities /= np.sum(class_probabilities, axis=-1, keepdims=True)
    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=1234)
      initial_vars = softmax_layer.init(
          prng_key, inputs, method=softmax_layer.get_logits)
      logits = softmax_layer.apply(
          initial_vars, inputs, method=softmax_layer.get_logits)
      outputs = softmax_layer.apply(
          initial_vars,
          inputs,
          class_weights,
          class_ids=class_ids,
          class_probabilities=class_probabilities)
    # Test whether tf Softmax layer returns same output
    # Modify initial_vars to use TF compatible params
    tf_initial_vars = py_utils.NestedMap.FromNestedDict(initial_vars['params'])
    tf_initial_vars.linear = py_utils.NestedMap()
    tf_initial_vars.linear.w = tf_initial_vars.logits_ffn.linear.w
    tf_initial_vars.bias = py_utils.NestedMap()
    tf_initial_vars.bias.b = tf_initial_vars.logits_ffn.bias.b
    tf_p = lingvo_layers.SingleShardFullSoftmax.Params().Set(
        name='tf_softmax',
        num_classes=p.num_classes,
        input_dim=p.input_dims,
        logits_soft_max=soft_cap_logits)
    tf_softmax_layer = tf_p.Instantiate()
    tf_logits = tf_softmax_layer.Logits(tf_initial_vars,
                                        tf.constant(inputs, dtype=tf.float32))
    if use_class_ids and label_smoothing_prob > 0:
      class_probabilities = np.zeros([8, 10, 50])
      index = np.indices([8, 10])
      class_probabilities[index[0], index[1], np.squeeze(class_ids, 2)] = 1
      class_probabilities = (
          class_probabilities * (1 - label_smoothing_prob) +
          (1 - class_probabilities) * label_smoothing_prob /
          (p.num_classes - 1))
      class_ids = None
    tf_output = tf_softmax_layer.FProp(
        tf_initial_vars,
        tf.constant(inputs, dtype=tf.float32),
        class_weights,
        class_ids=class_ids,
        class_probabilities=class_probabilities)
    # Check all entries in the NestedMap and ensure it matches TF
    np_get_logits = to_np(logits)
    tf_np_get_logits = to_np(tf_logits)
    self.assertAllClose(np_get_logits, tf_np_get_logits, atol=1e-6)
    # Note: The argmax-related values are very sensitive to numerical errors.
    for k in outputs.keys():
      self.assertAllClose(to_np(outputs[k]), to_np(tf_output[k]), atol=1e-6)

  def test_simple_softmax_layer_class_ids(self):
    batch_size = 8
    num_classes = 50
    class_ids = np.random.randint(0, 50, [8, 1])
    p = embedding_softmax.FullSoftmax.HParams(
        name='jax_softmax', num_classes=num_classes, input_dims=40)
    softmax_layer = instantiate(p)
    npy_input = np.random.normal(1.5, 2.0, [batch_size, p.input_dims])
    inputs = jnp.asarray(npy_input)
    class_weights = np.random.normal(1.5, 2.0, [batch_size, 1])
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = softmax_layer.init(
        prng_key, inputs, method=softmax_layer.get_logits)
    logits = softmax_layer.apply(
        initial_vars, inputs, method=softmax_layer.get_logits)
    with base_layer.JaxContext.new_context():
      outputs = softmax_layer.apply(
          initial_vars,
          inputs,
          class_weights,
          class_ids=class_ids,
          class_probabilities=None)
    # Test whether tf Softmax layer returns same output.
    # Modify initial_vars to use TF compatible params.
    tf_initial_vars = py_utils.NestedMap.FromNestedDict(initial_vars['params'])
    tf_initial_vars = test_utils.replace_jax_simple_full_softmax_vars_to_tf(
        tf_initial_vars)
    # Convert all the values to TF tensor.
    tf_initial_vars = tf.nest.map_structure(tf.convert_to_tensor,
                                            tf_initial_vars)

    tf_p = lingvo_layers.SimpleFullSoftmax.Params().Set(
        name='tf_softmax', num_classes=p.num_classes, input_dim=p.input_dims)
    tf_softmax_layer = tf_p.Instantiate()
    tf_logits = tf_softmax_layer.Logits(tf_initial_vars,
                                        tf.constant(inputs, dtype=tf.float32))
    tf_output = tf_softmax_layer.FProp(
        tf_initial_vars,
        tf.constant(inputs, dtype=tf.float32),
        class_weights,
        class_ids=class_ids,
        class_probabilities=None)
    # Check all entries in the NestedMap and ensure it matches TF.
    np_get_logits = to_np(logits)
    tf_np_get_logits = to_np(tf_logits)
    self.assertAllClose(np_get_logits, tf_np_get_logits)
    for k in outputs.keys():
      self.assertAllClose(to_np(outputs[k]), to_np(tf_output[k]))

  def test_simple_softmax_label_smoothing(self):
    batch_size = 2
    num_classes = 10
    class_ids = np.random.randint(0, 50, [batch_size, 1])
    p = embedding_softmax.FullSoftmax.HParams(
        name='jax_softmax',
        num_classes=num_classes,
        input_dims=40,
        label_smoothing_prob=0.5)  # build softmax with label smoothing.
    # Boiler-plate stuff, initialize weights and inputs/ class ids.
    softmax_layer = instantiate(p)
    npy_input = np.random.normal(1.5, 2.0, [batch_size, p.input_dims])
    inputs = jnp.asarray(npy_input)
    class_weights = np.random.normal(1.5, 2.0, [batch_size, 1])

    # check label smoothing for 4 setups.
    # 1. Train time + label_smoothing_apply_for_eval=True
    # 2. Train time + label_smoothing_apply_for_eval=False
    # 3. Eval time + label_smoothing_apply_for_eval=True
    # 4. Eval time + label_smoothing_apply_for_eval=False
    # First three setups should have identical results, since label smoothing
    # is always applied at training time.
    per_example_xent_for_setups = []

    for (is_training_mode,
         label_smoothing_apply_for_eval) in itertools.product([True, False],
                                                              [True, False]):
      context_p = base_layer.JaxContext.HParams(do_eval=not is_training_mode)
      p.label_smoothing_apply_for_eval = label_smoothing_apply_for_eval
      softmax_layer = instantiate(p)
      with base_layer.JaxContext.new_context(hparams=context_p):
        prng_key = jax.random.PRNGKey(seed=123)
        initial_vars = softmax_layer.init(
            prng_key,
            inputs,
            class_weights,
            class_ids=class_ids,
            class_probabilities=None)
        outputs = softmax_layer.apply(
            initial_vars,
            inputs,
            class_weights,
            class_ids=class_ids,
            class_probabilities=None)
        per_example_xent_for_setups.append(outputs.per_example_xent)
    # cross entropy must be same for first three setups.
    self.assertAllClose(
        to_np(per_example_xent_for_setups[0]),
        to_np(per_example_xent_for_setups[1]))
    self.assertAllClose(
        to_np(per_example_xent_for_setups[1]),
        to_np(per_example_xent_for_setups[2]))
    # cross entropy for setup 4 (without ls) must be less than other setups.
    np.testing.assert_array_less(
        to_np(per_example_xent_for_setups[3]),
        to_np(per_example_xent_for_setups[0]))

  @parameterized.parameters((8, 1001), (16, 1024), (32, 30000))
  def test_simple_softmax_layer_class_probs(self, batch_size, num_classes):
    class_probabilities = np.random.normal(1.5, 2.0, [batch_size, num_classes])
    # Normalize class probabilities to be a probability distribution.
    class_probabilities /= np.sum(class_probabilities, axis=-1, keepdims=True)
    p = embedding_softmax.FullSoftmax.HParams(
        name='jax_softmax', num_classes=num_classes, input_dims=40)
    softmax_layer = instantiate(p)
    npy_input = np.random.normal(1.5, 2.0, [batch_size, p.input_dims])
    inputs = jnp.asarray(npy_input)
    class_weights = np.random.normal(1.5, 2.0, [batch_size, 1])
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = softmax_layer.init(
        prng_key, inputs, method=softmax_layer.get_logits)
    logits = softmax_layer.apply(
        initial_vars, inputs, method=softmax_layer.get_logits)
    with base_layer.JaxContext.new_context():
      outputs = softmax_layer.apply(
          initial_vars,
          inputs,
          class_weights,
          class_ids=None,
          class_probabilities=class_probabilities)
    # Test whether tf Softmax layer returns same output.
    # Modify initial_vars to use TF compatible params.
    tf_initial_vars = py_utils.NestedMap.FromNestedDict(initial_vars['params'])
    tf_initial_vars = test_utils.replace_jax_simple_full_softmax_vars_to_tf(
        tf_initial_vars)
    # Convert all the values to TF tensor.
    tf_initial_vars = tf.nest.map_structure(tf.convert_to_tensor,
                                            tf_initial_vars)

    tf_p = lingvo_layers.SimpleFullSoftmax.Params().Set(
        name='tf_softmax', num_classes=p.num_classes, input_dim=p.input_dims)
    tf_softmax_layer = tf_p.Instantiate()
    tf_logits = tf_softmax_layer.Logits(tf_initial_vars,
                                        tf.constant(inputs, dtype=tf.float32))
    tf_output = tf_softmax_layer.FProp(
        tf_initial_vars,
        tf.constant(inputs, dtype=tf.float32),
        class_weights,
        class_ids=None,
        class_probabilities=class_probabilities)
    # Check all entries in the NestedMap and ensure it matches TF.
    np_get_logits = to_np(logits)
    tf_np_get_logits = to_np(tf_logits)
    self.assertAllClose(np_get_logits, tf_np_get_logits)
    for k in outputs.keys():
      self.assertAllClose(to_np(outputs[k]), to_np(tf_output[k]))

  def test_simple_softmax_layer_value_error(self):
    batch_size = 8
    num_classes = 50
    class_ids = None
    class_probabilities = None
    p = embedding_softmax.FullSoftmax.HParams(
        name='jax_softmax', num_classes=num_classes, input_dims=40)
    softmax_layer = instantiate(p)
    npy_input = np.random.normal(1.5, 2.0, [batch_size, p.input_dims])
    inputs = jnp.asarray(npy_input)
    class_weights = np.random.normal(1.5, 2.0, [batch_size, 1])
    with self.assertRaises(ValueError):
      with base_layer.JaxContext.new_context():
        prng_key = jax.random.PRNGKey(seed=123)
        initial_vars = softmax_layer.init(
            prng_key,
            inputs,
            class_weights,
            class_ids=class_ids,
            class_probabilities=class_probabilities)
        _ = softmax_layer.apply(
            initial_vars,
            inputs,
            class_weights,
            class_ids=class_ids,
            class_probabilities=class_probabilities)

  @parameterized.parameters((0., 'index', True), (0., 'matmul', True),
                            (1.0, 'index', False), (1.0, 'matmul', False))
  def test_single_sharded_shared_embedding_softmax_layer(
      self, soft_cap_logits, lookup_style, scale_sqrt_depth):
    class_ids = np.random.randint(1, 50, [8, 10, 1])
    p = embedding_softmax.SharedEmbeddingSoftmax.HParams(
        name='jax_softmax',
        num_classes=50,
        input_dims=40,
        soft_cap_logits=soft_cap_logits,
        lookup_style=lookup_style,
        scale_sqrt_depth=scale_sqrt_depth)
    softmax_layer = instantiate(p)
    npy_input = np.random.normal(1.5, 2.0, [8, 10, p.input_dims])
    inputs = jnp.asarray(npy_input)
    class_weights = np.random.normal(1.5, 2.0, [8, 10, 1])
    with base_layer.JaxContext.new_context():
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = softmax_layer.init(
          prng_key, inputs, class_weights, class_ids=class_ids)
      outputs = softmax_layer.apply(
          initial_vars, inputs, class_weights, class_ids=class_ids)
    ids = np.squeeze(class_ids, axis=-1)
    emb_lookup_outputs = softmax_layer.apply(
        initial_vars, ids=jnp.asarray(ids), method=softmax_layer.emb_lookup)
    # Test whether tf Softmax layer returns same output
    # Modify initial_vars to use TF compatible params
    tf_initial_vars = py_utils.NestedMap.FromNestedDict(initial_vars['params'])
    tf_initial_vars.linear = py_utils.NestedMap()
    tf_initial_vars.linear.w = tf_initial_vars.logits_ffn.linear.w
    tf_initial_vars.bias = py_utils.NestedMap()
    tf_initial_vars.bias.b = tf_initial_vars.logits_ffn.bias.b
    tf_p = lingvo_layers.SingleShardSharedEmbeddingSoftmax.Params().Set(
        name='tf_softmax',
        num_classes=p.num_classes,
        input_dim=p.input_dims,
        vocab_size=p.num_classes,
        embedding_dim=p.input_dims,
        logits_soft_max=soft_cap_logits,
        scale_sqrt_depth=scale_sqrt_depth)
    tf_softmax_layer = tf_p.Instantiate()
    tf_output = tf_softmax_layer.FProp(
        tf_initial_vars,
        tf.constant(inputs, dtype=tf.float32),
        class_weights,
        class_ids=class_ids)
    tf_emb_lookup_output = tf_softmax_layer.EmbLookup(
        tf_initial_vars, ids=tf.constant(ids))

    # Check all entries in the NestedMap and ensure it matches TF
    np_logits = to_np(outputs.logits)
    tf_np_logits = to_np(tf_output.logits)
    self.assertAllClose(np_logits, tf_np_logits, atol=1e-6)
    for k in outputs.keys():
      self.assertAllClose(to_np(outputs[k]), to_np(tf_output[k]), atol=1e-6)
    np_emb_lookup_output = to_np(emb_lookup_outputs)
    tf_np_emb_lookup_output = to_np(tf_emb_lookup_output)
    self.assertAllClose(
        tf_np_emb_lookup_output, np_emb_lookup_output, atol=1e-6)

  @parameterized.parameters((8, 1001), (16, 1024), (32, 30000))
  def test_sigmoid_cross_entropy_class_probs(self, batch_size, num_classes):
    class_probabilities = np.random.normal(1.5, 2.0, [batch_size, num_classes])
    # Normalize class probabilities to be a probability distribution.
    class_probabilities /= np.sum(class_probabilities, axis=-1, keepdims=True)
    p = embedding_softmax.SigmoidCrossEntropy.HParams(
        name='jax_softmax', num_classes=num_classes, input_dims=40)
    sigmoid_xent_layer = instantiate(p)
    npy_input = np.random.normal(1.5, 2.0, [batch_size, p.input_dims])
    inputs = jnp.asarray(npy_input)
    class_weights = np.random.normal(1.5, 2.0, [batch_size, 1])
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = sigmoid_xent_layer.init(
        prng_key, inputs, method=sigmoid_xent_layer.get_logits)
    logits = sigmoid_xent_layer.apply(
        initial_vars, inputs, method=sigmoid_xent_layer.get_logits)
    with base_layer.JaxContext.new_context():
      outputs = sigmoid_xent_layer.apply(
          initial_vars,
          inputs,
          class_weights,
          class_ids=None,
          class_probabilities=class_probabilities)
    # Test whether tf sigmoid-cross-entropy layer returns same output.
    # Modify initial_vars to use TF compatible params.
    tf_initial_vars = py_utils.NestedMap.FromNestedDict(initial_vars['params'])
    tf_initial_vars = test_utils.replace_jax_simple_full_softmax_vars_to_tf(
        tf_initial_vars)
    # Convert all the values to TF tensor.
    tf_initial_vars = tf.nest.map_structure(tf.convert_to_tensor,
                                            tf_initial_vars)

    tf_p = lingvo_layers.SimpleFullSigmoidCrossEntropy.Params().Set(
        name='tf_softmax',
        num_classes=p.num_classes,
        input_dim=p.input_dims,
        bias_init=-10.0)
    tf_sigmoid_xent_layer = tf_p.Instantiate()
    tf_logits = tf_sigmoid_xent_layer.Logits(
        tf_initial_vars, tf.constant(inputs, dtype=tf.float32))
    tf_output = tf_sigmoid_xent_layer.FProp(
        tf_initial_vars,
        tf.constant(inputs, dtype=tf.float32),
        class_weights,
        class_ids=None,
        class_probabilities=tf.constant(class_probabilities, dtype=tf.float32))
    # Check all entries in the NestedMap and ensure it matches TF.
    np_get_logits = to_np(logits)
    tf_np_get_logits = to_np(tf_logits)
    self.assertAllClose(np_get_logits, tf_np_get_logits)
    for k in outputs.keys():
      self.assertAllClose(to_np(outputs[k]), to_np(tf_output[k]))

  @parameterized.parameters(8, 1)
  def test_sigmoid_cross_entropy_class_weights(self, num_classes):
    batch_size = 32
    class_probabilities = np.random.normal(1.5, 2.0, [batch_size, num_classes])
    # Normalize class probabilities to be a probability distribution.
    class_probabilities /= np.sum(class_probabilities, axis=-1, keepdims=True)
    p = embedding_softmax.SigmoidCrossEntropy.HParams(
        name='jax_softmax', num_classes=num_classes, input_dims=40)
    sigmoid_xent_layer = instantiate(p)
    npy_input = np.random.normal(1.5, 2.0, [batch_size, p.input_dims])
    inputs = jnp.asarray(npy_input)
    class_weights = np.random.normal(1.5, 2.0, [batch_size, num_classes])
    target_weights = np.sum(class_weights, keepdims=True)
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = sigmoid_xent_layer.init(
        prng_key, inputs, method=sigmoid_xent_layer.get_logits)
    logits = sigmoid_xent_layer.apply(
        initial_vars, inputs, method=sigmoid_xent_layer.get_logits)
    with base_layer.JaxContext.new_context():
      outputs = sigmoid_xent_layer.apply(
          initial_vars,
          inputs,
          class_weights,
          class_ids=None,
          class_probabilities=class_probabilities)
    # Test whether tf sigmoid-cross-entropy layer returns same output.
    # Modify initial_vars to use TF compatible params.
    tf_initial_vars = py_utils.NestedMap.FromNestedDict(initial_vars['params'])
    tf_initial_vars = test_utils.replace_jax_simple_full_softmax_vars_to_tf(
        tf_initial_vars)
    # Convert all the values to TF tensor.
    tf_initial_vars = tf.nest.map_structure(tf.convert_to_tensor,
                                            tf_initial_vars)

    tf_p = lingvo_layers.SimpleFullSigmoidCrossEntropy.Params().Set(
        name='tf_softmax',
        num_classes=p.num_classes,
        input_dim=p.input_dims,
        bias_init=-10.0)
    tf_sigmoid_xent_layer = tf_p.Instantiate()
    tf_logits = tf_sigmoid_xent_layer.Logits(
        tf_initial_vars, tf.constant(inputs, dtype=tf.float32))
    tf_output = tf_sigmoid_xent_layer.FProp(
        tf_initial_vars,
        tf.constant(inputs, dtype=tf.float32),
        target_weights,
        class_ids=None,
        class_probabilities=tf.constant(class_probabilities, dtype=tf.float32))
    # Check all entries in the NestedMap and ensure it matches TF.
    np_get_logits = to_np(logits)
    tf_np_get_logits = to_np(tf_logits)
    self.assertAllClose(np_get_logits, tf_np_get_logits)
    self.assertAllClose(
        to_np(outputs['total_weight']), to_np(tf_output['total_weight']))
    self.assertAllClose(
        to_np(outputs['per_example_argmax']),
        to_np(tf_output['per_example_argmax']))

  def test_sigmoid_cross_entropy_class_weights_no_ffn(self):
    batch_size = 32
    num_classes = 40
    class_probabilities = np.random.normal(1.5, 2.0, [batch_size, num_classes])
    # Normalize class probabilities to be a probability distribution.
    class_probabilities /= np.sum(class_probabilities, axis=-1, keepdims=True)
    p = embedding_softmax.SigmoidCrossEntropy.HParams(
        name='jax_softmax',
        num_classes=num_classes,
        input_dims=40,
        feed_forward_tpl=None)
    sigmoid_xent_layer = instantiate(p)
    npy_input = np.random.normal(1.5, 2.0, [batch_size, p.input_dims])
    inputs = jnp.asarray(npy_input)
    class_weights = np.random.normal(1.5, 2.0, [batch_size, num_classes])
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = sigmoid_xent_layer.init(
        prng_key, inputs, method=sigmoid_xent_layer.get_logits)
    with base_layer.JaxContext.new_context():
      outputs = sigmoid_xent_layer.apply(
          initial_vars,
          inputs,
          class_weights,
          class_ids=None,
          class_probabilities=class_probabilities)
      self.assertAllClose(to_np(outputs['logits']), inputs)

  @parameterized.parameters((1, 10), (1, 1e5), (10, 20), (10, 1e5))
  def test_position_embedding_layer(self, min_timescale, max_timescale):
    p = embedding_softmax.PositionalEmbedding.HParams(
        name='jax_pos',
        embedding_dims=50,
        min_timescale=min_timescale,
        max_timescale=max_timescale)
    pos_layer = instantiate(p)
    seq_length = np.random.randint(100, 1000)
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = pos_layer.init(prng_key, seq_length)
    output = pos_layer.apply(initial_vars, seq_length)

    output = jnp.squeeze(output, axis=0)
    # Test whether tf PositionalEmbedding layer returns same output
    # Modify initial_vars to use TF compatible params
    tf_initial_vars = initial_vars
    tf_p = lingvo_layers.PositionalEmbeddingLayer.Params().Set(
        name='tf_pos',
        embedding_dim=p.embedding_dims,
        min_timescale=min_timescale,
        max_timescale=max_timescale)
    tf_pos_layer = tf_p.Instantiate()
    tf_output = tf_pos_layer.FProp(tf_initial_vars, seq_length)
    np_pos = to_np(output)
    tf_np_pos = to_np(tf_output)
    self.assertAllClose(tf_np_pos, np_pos, atol=1e-3)

  @parameterized.parameters((1, 10), (1, 1e5), (10, 20), (10, 1e5))
  def test_position_embedding_layer_with_position(self, min_timescale,
                                                  max_timescale):
    p = embedding_softmax.PositionalEmbedding.HParams(
        name='jax_pos',
        embedding_dims=50,
        min_timescale=min_timescale,
        max_timescale=max_timescale)
    pos_layer = instantiate(p)
    position = np.array([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
                         [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
                         [0, 1, 2, 3, 4, 5, 6, 0, 1, 2],
                         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = pos_layer.init(prng_key, position=position)
    output = pos_layer.apply(initial_vars, position=position)
    # Test whether tf PositionalEmbedding layer returns same output
    # Modify initial_vars to use TF compatible params
    tf_initial_vars = initial_vars
    tf_p = lingvo_layers.PositionalEmbeddingLayer.Params().Set(
        name='tf_pos',
        embedding_dim=p.embedding_dims,
        min_timescale=min_timescale,
        max_timescale=max_timescale)
    tf_pos_layer = tf_p.Instantiate()
    tf_output = tf_pos_layer.FPropWithPosition(tf_initial_vars, position)
    np_pos = to_np(output)
    tf_np_pos = to_np(tf_output)
    self.assertAllClose(tf_np_pos, np_pos, atol=1e-3)

  @parameterized.parameters((1, 10, 1), (1, 1e5, 3), (10, 20, 4), (10, 1e5, 5))
  def test_rotary_position_embedding_layer_prefix(self, min_timescale,
                                                  max_timescale, window_size):
    embedding_dims = 32
    p = embedding_softmax.RotaryPositionalEmbedding.HParams(
        name='jax_pos',
        embedding_dims=embedding_dims,
        min_timescale=min_timescale,
        max_timescale=max_timescale)
    pos_layer = instantiate(p)
    inputs = np.random.normal(1.5, 2.5, (2, 8, 4, embedding_dims))
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = pos_layer.init(prng_key, inputs)
    output = pos_layer.apply(initial_vars, inputs)
    # Test whether extend_step returns same output.
    for i in range(inputs.shape[1]):
      start = max(0, i + 1 - window_size)
      end = i + 1
      inputs_prefix = inputs[:, start:end, :, :]
      pad_width = window_size - end + start
      paddings = [(0, 0), (pad_width, 0), (0, 0), (0, 0)]
      inputs_prefix = jnp.pad(inputs_prefix, paddings)
      jax_extend_step_out = pos_layer.apply(
          initial_vars, inputs_prefix, position=i, method=pos_layer.extend_step)
      jax_extend_step_out = jax.lax.dynamic_slice_in_dim(
          jax_extend_step_out,
          start_index=window_size - 1,
          slice_size=1,
          axis=1)
      jax_np_extend_step_out = test_utils.to_np(jax_extend_step_out)
      jax_fprop_slice = jax.lax.dynamic_slice_in_dim(
          output, start_index=i, slice_size=1, axis=1)
      self.assertArraysEqual(jax_fprop_slice, jax_np_extend_step_out)

  @parameterized.parameters((1, 10), (1, 1e5), (10, 20), (10, 1e5))
  def test_rotary_position_embedding_layer_no_prefix(self, min_timescale,
                                                     max_timescale):
    embedding_dims = 32
    p = embedding_softmax.RotaryPositionalEmbedding.HParams(
        name='jax_pos',
        embedding_dims=embedding_dims,
        min_timescale=min_timescale,
        max_timescale=max_timescale)
    pos_layer = instantiate(p)
    inputs = np.random.normal(1.5, 2.5, (2, 8, 4, embedding_dims))
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = pos_layer.init(prng_key, inputs=inputs)
    output = pos_layer.apply(initial_vars, inputs=inputs)
    # Test whether extend_step returns same output.
    for i in range(inputs.shape[1]):
      jax_extend_step_out = pos_layer.apply(
          initial_vars,
          inputs[:, i, :, :],
          position=i,
          method=pos_layer.extend_step)
      jax_np_extend_step_out = test_utils.to_np(jax_extend_step_out)
      jax_fprop_slice = output[:, i, :, :]
      self.assertArraysEqual(jax_fprop_slice, jax_np_extend_step_out)

  @parameterized.parameters(
      ([0, 1, 0, 1],),
      ([0, 1, 2, 3],),
      ([0, 1, 2, 0],),
      ([0, 0, 1, 2],),
      (None),
  )
  def test_rotary_position_embedding_layer_2d(self, position):
    embedding_dims = 2
    min_timescale = 1
    max_timescale = 1e4
    p = embedding_softmax.RotaryPositionalEmbedding.HParams(
        name='jax_pos',
        embedding_dims=embedding_dims,
        min_timescale=min_timescale,
        max_timescale=max_timescale)
    pos_layer = instantiate(p)
    inputs = np.random.normal(1.5, 2.5, (1, 4, 1, embedding_dims))
    if position is None:
      position = jnp.arange(4, dtype=jnp.float32)
    position = jnp.array(position)
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = pos_layer.init(
        prng_key, inputs=inputs, position=position[jnp.newaxis, :])
    output = pos_layer.apply(
        initial_vars, inputs=inputs, position=position[jnp.newaxis, :])
    np_output = test_utils.to_np(output)
    sinusoid_inp = position
    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)
    first_part = inputs[0, :, 0, 0] * cos - inputs[0, :, 0, 1] * sin
    second_part = inputs[0, :, 0, 1] * cos + inputs[0, :, 0, 0] * sin
    expected_output = np.stack([first_part, second_part], axis=-1)
    self.assertArraysEqual(np_output[0, :, 0, :], expected_output)

  @parameterized.parameters('index', 'matmul')
  def test_trainable_positional_embedding_layer(self, lookup_style):
    p = embedding_softmax.TrainablePositionalEmbedding.HParams(
        name='jax_pos_emb',
        max_seq_length=10,
        embedding_dims=40,
        lookup_style=lookup_style)
    emb_layer = instantiate(p)
    npy_input = np.random.randint(0, p.max_seq_length,
                                  [10, p.max_seq_length]).astype('int32')
    inputs = jnp.asarray(npy_input)
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = emb_layer.init(prng_key, p.max_seq_length, inputs)
    outputs = emb_layer.apply(initial_vars, p.max_seq_length, inputs)
    # Test whether tf Embedding layer returns same output
    # Modify initial_vars to use TF compatible params
    tf_initial_vars = py_utils.NestedMap.FromNestedDict(initial_vars['params'])
    tf_p = lingvo_layers.SingleShardEmbeddingLayer.Params().Set(
        name='tf_pos_emb',
        vocab_size=p.max_seq_length,
        embedding_dim=p.embedding_dims)
    tf_emb_layer = tf_p.Instantiate()
    tf_output = tf_emb_layer.FProp(tf_initial_vars,
                                   tf.constant(inputs, dtype=tf.int32))
    np_outputs = to_np(outputs)
    tf_np_outputs = to_np(tf_output)
    self.assertAllClose(tf_np_outputs, np_outputs)


if __name__ == '__main__':
  absltest.main()
