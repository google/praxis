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

"""Base classes for the Praxis input layers."""

from __future__ import annotations

import abc
import copy
import re
from typing import Dict, Optional, Sequence

from absl import logging
import jax
from jax._src.lib import xla_client as xc
from jax.experimental import maps
from lingvo.core import cluster_factory
from lingvo.core import datasource
import numpy as np
from praxis import base_hyperparams
from praxis import py_utils
from praxis import pytypes
import tensorflow.compat.v1 as tf_v1
import tensorflow.compat.v2 as tf

NestedMap = py_utils.NestedMap
NestedJTensor = pytypes.NestedJTensor
Nested = pytypes.Nested
NestedShapeDtypeStruct = pytypes.NestedShapeDtypeStruct
NestedPartitionSpec = pytypes.NestedPartitionSpec
instantiate = base_hyperparams.instantiate


class BaseInput(base_hyperparams.BaseParameterizable):
  """Base class for Praxis input pipelines.

  During paxml's train, on each host an input instance will be
  created (instantiate(input_p)), and then get_next() is iteratively
  called in eager mode to generate one batch of data for each step
  of train/eval/etc.

  If supported, for eval, reset() is called after each eval step.
  See p.reset_for_eval below.

  A tf.data based input should inherit this class directly and implement
  get_next() and reset(). For an example of how to handle sharding for both
  training and eval data, please refer to the implementation of
  TFRecordBertInput at tasks/lm/input_generator.py.

  If there is already a Lingvo TF input generator that one would like to
  use directly, please use LingvoInputAdaptor below.
  """
  _VALIDATE_BATCH_SIZE_NOT_NONE = True

  class HParams(base_hyperparams.BaseParameterizable.HParams):
    """Hyper-parameters associated with this Input class.

    Attributes:
      batch_size: The (Jax per process) Batch size. Each call to get_next()
        returns a batch with this batch size.
      num_infeed_hosts: Usually set to jax.process_count(). Implementation must
        ensure that the data is sharded into this many shard.
      infeed_host_index: Usually set to jax.process_index(). Implementation must
        ensure that each instance returns a shard with this index.
      input_random_seed: If set, implementation must ensure that this is used to
        seed randomness, e.g. when shuffling in a deterministic manner.
      reset_for_eval: If set, eval will continue until tf.errors.OutOfRange is
        raised, and reset() will called for each eval. Implementation must
        ensure that all variant p.infeed_host_index instances raise after the
        same numberof calls to get_next() to ensure synchronization across
        hosts. If notset, get_next() must never raise.
      eval_loop_num_batches: Num of batches to process per eval loop. Must be >=
        1. This valueis ignored if reset_for_eval is set True, in which case,
        this value is dynamically determined by the number of available batches.
        If reset_for_eval is set to False, then each eval loop will process this
        many batches. Metrics over those batches will be aggregated and then
        reported.
      is_training: Whether or not this dataset is used for model traning.
      experimental_remote_input: whether to process inputs on remote hosts, when
        there is a single controller.
      batch_padding_size: The amount of right-padding applied to each invocation
        of get_next_padded(). Useful when the batch_size is smaller than the
        number of devices.
      custom_device_order: Custom order of devices in GSPMD sharding for the
        inputs. This is needed when there are data paddings on some devices in
        a multi-process environment. Values in the list are logical partition
        IDs (offsets in global_mesh.devices.flat) in the global mesh.
    """
    batch_size: Optional[int] = None
    # Sharding behavior. If num_infeed_hosts is 0, it will be given a default
    # value by PAX trainer; if it is still not set during __init__, 1 will be
    # used.
    num_infeed_hosts: int = 0
    infeed_host_index: int = 0
    # Deterministic randomness.
    input_random_seed: Optional[int] = None
    reset_for_eval: bool = False
    eval_loop_num_batches: int = 1
    is_training: bool = False
    experimental_remote_input: bool = False
    batch_padding_size: int = 0
    custom_device_order: Optional[Sequence[int]] = None

  @classmethod
  def get_batch_size(cls, hparams: BaseInput.HParams) -> int:
    assert hparams.batch_size is not None
    return hparams.batch_size

  def __init__(self, hparams: BaseInput.HParams) -> None:
    if self._VALIDATE_BATCH_SIZE_NOT_NONE and (hparams.batch_size is None):
      raise ValueError('Must specify p.batch_size.')
    if not hparams.name:
      hparams.name = 'train' if hparams.is_training else 'input'
    if hparams.num_infeed_hosts == 0:
      hparams.num_infeed_hosts = 1
    super().__init__(hparams)
    name = self.hparams.name
    if re.fullmatch(r'[a-zA-Z0-9.:_-]+', name) is None:
      raise ValueError(f'Input hparams p.name string invalid: "{name}" '
                       'does not fully match "[a-zA-Z0-9._-]+".')
    if hparams.experimental_remote_input and jax.process_count() > 1:
      raise NotImplementedError(
          'Remote input is not supported when there are multiple controllers.')

  def get_next(self) -> NestedJTensor:
    raise NotImplementedError

  def get_child(self, input_name: str) -> NestedJTensor:
    raise NotImplementedError

  def get_next_padded(self) -> NestedJTensor:
    unpadded = self.get_next()
    pad_size = self.hparams.batch_padding_size
    if pad_size == 0:
      return unpadded
    return jax.tree_util.tree_map(
        lambda x: np.pad(x, [[0, pad_size]] + [[0, 0]] * (x.ndim - 1)),
        unpadded)

  def reset(self) -> None:
    pass

  def ids_to_strings(self,
                     ids: pytypes.NpTensor,
                     lengths: pytypes.NpTensor,
                     key: Optional[str] = None) -> Sequence[str]:
    """Converts int ids into strings.

    Args:
      ids: A matrix of shape [batch, seqlen], each row is a sequence to be
        converted.
      lengths: A vector of shape [batch]. lens[i] is the sequence length of the
        i-th row. Only the first lens[i] tokens in ids[i, :] are valid tokens.
      key: Optional argument to specify which tokenizer to use. This is useful
        for example in a sequence model where the source and targets have
        different tokenizers.

    Returns:
      A list strings of shape [batch]. The converted texts.
    """
    raise NotImplementedError

  def reshard_for_pmap(self, arrays: NestedJTensor) -> NestedJTensor:
    """Reshards inputs for pmap.

    This function reshards `arrays`, inputs returned by this input class, to be
    suitable for pmap computation. The returned arrays are expected to have the
    pmap device dimension as the 0th dimension and per-device batch dimension as
    the 1st dimension. Input subclasses may override this to customize the
    resharding implementations.

    Args:
      arrays: Inputs returned by this input class.

    Returns:
      Resharded inputs.
    """
    assert self.hparams.custom_device_order is None
    return jax.tree_util.tree_map(py_utils.reshard, arrays)

  def reshard_for_spmd(self, arrays: NestedJTensor,
                       global_shapes: NestedShapeDtypeStruct,
                       global_mesh: maps.Mesh,
                       pspecs: NestedPartitionSpec) -> NestedJTensor:
    """Reshards inputs for pjit.

    This function reshards `arrays`, inputs returned by this input class, to be
    suitable for pjit computation on the given mesh. The caller also provides
    the expected global shapes and partition specs of the arrays to be returned.
    The returned arrays are expected to have the global batch dimension as the
    0th dimension and be sharded in the way specified by `pspecs`.Input
    subclasses may override this to customize the resharding implementations.

    Args:
      arrays: Inputs returned by this input class.
      global_shapes: Expected global shapes of `arrays` after resharding.
      global_mesh: Global mesh for pjit computation.
      pspecs: Expected partition specs for `arrays` after resharding.

    Returns:
      Resharded inputs.
    """
    py_utils.assert_same_shape_and_dtype(
        global_shapes,
        jax.tree_util.tree_map(py_utils.get_global_input_shape_dtype, arrays))
    device_order = self.hparams.custom_device_order
    if device_order is None:
      return py_utils.create_gda(arrays, global_shapes, global_mesh, pspecs)
    assert jax.config.jax_array
    assert len(device_order) == jax.device_count()

    # Use custom device order to create OpSharding in jax.Array.
    def _create_array(x, global_shape):
      op_sharding = xc.OpSharding()
      op_sharding.type = xc.OpSharding.Type.OTHER
      # Fully sharded on the batch dim.
      op_sharding.tile_assignment_dimensions = [len(device_order)] + [1] * (
          len(global_shape.shape) - 1)
      # Custom device order.
      op_sharding.tile_assignment_devices = device_order
      dbs = py_utils.put_to_devices(x, global_mesh.local_devices)
      sharding = jax.sharding.OpShardingSharding(
          list(global_mesh.devices.flat), op_sharding)
      return jax.make_array_from_single_device_arrays(global_shape.shape,
                                                      sharding, dbs)

    return jax.tree_util.tree_map(_create_array, arrays, global_shapes)


class LingvoInputAdaptor(BaseInput):
  """Syntactic sugar for adapting a Lingvo style input for Pax.

  This should be able to wrap any Lingvo TF input generator to be used in
  Pax. Remember to set `p.is_training=True` on the training dataset.

  Some usage caveats below.

  For eval, `p.num_samples` or other similar params like samples_per_summary are
  completely ignored by Pax. Caller should instead set `p.num_batches` to
  (p.num_samples // batch_size) with `p.reset_for_eval=True` so that each eval
  step reads (approximately) one epoch of eval data. This might not be needed if
  the input already is finite (e.g. with p.repeat_count=1).

  When multiple infeed hosts are used, one must take care to ensure that the
  Lingvo input either already uses InfeedContextScope for proper sharding, or
  alternatively do not use the same random seed on all hosts. In other words,
  one must avoid the failure case where each host emits identical training data.
  See also p.allow_fixed_file_random_seed below.
  """
  _VALIDATE_BATCH_SIZE_NOT_NONE = False
  _VALIDATE_BATCH_SIZE_NONE = True

  class HParams(BaseInput.HParams):
    """Associated hyperparams for this BaseInput.

    Attributes:
      input: Params of a Lingvo input generator.
      num_batches: If specified and positive, raises tf.errors.OutOfRange after
        this manybatches have been produced. This forces a raise after
        get_next() is called this many times, to support p.reset_for_eval=True.
      allow_fixed_file_random_seed: If not set, disallows a fixed, non-zero
        p.input.file_random_seed. We disallow by default to avoid having
        identical input batches across different infeed hosts. If set, random
        seeds are adjusted by p.infeed_host_index to ensure different random
        seeds.
      cluster_do_eval: Whether to set cluster.do_eval to True for non-training
        data. Note that if set to True, this will change
        cluster.require_sequential_input_order to True as a result. Ignored when
        p.is_training is True.
    """
    input: Optional[py_utils.InstantiableParams] = None
    num_batches: Optional[int] = None
    allow_fixed_file_random_seed: bool = False
    cluster_do_eval: bool = False

  @classmethod
  def get_batch_size(cls, hparams: LingvoInputAdaptor.HParams) -> int:
    assert hparams.input is not None
    if hasattr(hparams.input, 'bucket_batch_limit'):
      return hparams.input.bucket_batch_limit[0]
    elif hasattr(hparams.input, 'batch_size'):
      return hparams.input.batch_size
    else:
      raise ValueError(
          'hparams.input has no attribute of bucket_batch_limit or batch_size.')

  def __init__(self, hparams: LingvoInputAdaptor.HParams) -> None:
    if self._VALIDATE_BATCH_SIZE_NONE and hparams.batch_size is not None:
      raise ValueError('LingvoInputAdaptor does not support p.batch_size. '
                       'Please specify batch size on p.input, e.g. with '
                       'p.input.bucket_batch_limit = [4] or '
                       'p.input.args.batch=4, depeding the Lingvo input '
                       f'used. Currently: p.batch_size={hparams.batch_size}, '
                       'it must be None.')
    if not hparams.name:
      hparams.name = hparams.input.name
    if hparams.input is None:
      raise ValueError('Params of a Lingvo input generator is not set.')
    if hasattr(hparams.input,
               'file_random_seed') and (hparams.input.file_random_seed) and (
                   not hparams.allow_fixed_file_random_seed):
      raise ValueError(
          'Input data using fixed non-zero file_random_seed: '
          f'hparams.input.file_random_seed={hparams.input.file_random_seed}. '
          'This means each host *might* infeed identical batches. You can set '
          'hparams.input.file_random_seed = 0, or if certain this is intended, '
          'suppress this error by setting hparams.allow_fixed_file_random_seed '
          '= True.')
    super().__init__(hparams)
    self._cluster = copy.deepcopy(cluster_factory.Current())
    # For Lingvo's Cluster context that may impact the behavior of this input
    # generator, we always set use_tpu to True, and optionally set do_eval
    # for non-training data when configured to do so. All other Cluster params
    # use the default value.
    self._cluster.params.xla_device = 'tpu'
    self._cluster.params.enable_asserts = False
    # This indirectly sets cluster.require_sequential_input_order as well.
    self._cluster.params.do_eval = (not hparams.is_training and
                                    hparams.cluster_do_eval)
    self._initialize()

  def _update_file_random_seed(self) -> None:
    """Updates file random seed to use different seeds for different hosts."""
    p = self.hparams
    if hasattr(p.input, 'file_random_seed') and p.input.file_random_seed:
      # Make sure each host uses a different random seed.
      p.input.file_random_seed += p.infeed_host_index

  def _initialize(self) -> None:
    """Initializes the relevant fields of this adaptor input."""
    p = self.hparams
    self._update_file_random_seed()
    # We make self.input public so that users can access its methods like
    # IdsToStrings if needed.
    with py_utils.infeed_context_scope(
        infeed_host_index=p.infeed_host_index,
        num_infeed_hosts=p.num_infeed_hosts), self._cluster:
      self.input = p.input.Instantiate()

    if hasattr(self.input, 'datasource') and isinstance(
        self.input.datasource, datasource.TFDatasetSource):
      # For the special case when the input is implemented by a tf.data.Dataset,
      # call eagerly. Using tf.function may result in returning duplicate
      # batches.
      self._get_next_fn = self._get_batch
    else:
      self._get_next_fn = tf.function(self._get_batch)
    self._num_batches_produced = 0

  def _get_batch(self) -> NestedMap:
    p = self.hparams
    with py_utils.infeed_context_scope(
        infeed_host_index=p.infeed_host_index,
        num_infeed_hosts=p.num_infeed_hosts), self._cluster:
      ret = self.input.GetPreprocessedInputBatch()
    # Remove unsupported string (byte) array from input.
    return ret.Filter(lambda v: v.dtype != tf.string)

  def get_next(self) -> NestedJTensor:
    p = self.hparams
    if p.num_batches is not None and p.num_batches > 0:
      if self._num_batches_produced >= p.num_batches:
        raise tf.errors.OutOfRangeError(
            node_def=None,
            op=None,
            message=f'num_batches exceeding {self._num_batches_produced}')
      self._num_batches_produced += 1
    ret = self._get_next_fn()
    ret = jax.tree_util.tree_map(lambda x: x.numpy(), ret)
    batch_size = jax.tree_util.tree_leaves(ret)[0].shape[0]
    ret.eval_sample_weights = np.ones([batch_size], np.float32)
    return ret

  def reset(self) -> None:
    if hasattr(self.input, 'datasource') and isinstance(
        self.input.datasource, datasource.TFDatasetSource):
      self.input.datasource.Reset()
      # reset counter to 0.
      self._num_batches_produced = 0
      return
    # reinstantiate the input and retrace self._get_batch.
    self._initialize()

  def ids_to_strings(self,
                     ids: pytypes.NpTensor,
                     lengths: pytypes.NpTensor,
                     key: Optional[str] = None) -> Sequence[str]:
    """Converts int ids into strings."""
    bytes_list = self.input.IdsToStrings(ids, lengths, key=key)
    if isinstance(bytes_list, tf.Tensor):
      if tf.executing_eagerly():
        bytes_list = bytes_list.numpy()
      else:
        with tf_v1.Session().as_default():
          bytes_list = bytes_list.eval()
    return [b.decode('utf-8') for b in bytes_list]


class LingvoInputAdaptorNewBatchSize(LingvoInputAdaptor):
  """A similar adapter as LingvoInputAdaptor supporting a new batch size.

  LingvoInputAdaptor uses the batch size specified by the underlying Lingvo
  input. This class, however, allows specifying a smaller p.batch_size.
  This can be useful when the Lingvo input expects a large batch size,
  but the user wants a smaller batch size, e.g. when the Lingvo input uses
  a fixed packing factor to do packing, which can more efficiently pack with
  more data.

  We require that the batch size of the underlying Lingvo input must divide
  p.batch_size. Internally this class acts as a cache, retrieving the large
  batches from the parent class size, and consuming it by slicing it to the
  smaller batch size specified by the user.

  Example usage:
      p = ChangeBatchSizeInput.HParams(...)
      p.input.packing_factor = 3.5
      p.input.bucket_batch_limit = [4096]
      p.batch_size = 4
  """
  _VALIDATE_BATCH_SIZE_NOT_NONE = True
  _VALIDATE_BATCH_SIZE_NONE = False

  @classmethod
  def get_batch_size(cls, hparams: BaseInput.HParams) -> int:
    assert hparams.batch_size is not None
    return hparams.batch_size

  def __init__(self, hparams: LingvoInputAdaptor.HParams):
    super().__init__(hparams)
    self._current_batch = super().get_next()
    self._inner_batch_size = jax.tree_util.tree_leaves(
        self._current_batch)[0].shape[0]
    logging.info(
        'The wrapped Lingvo input has batch size %d, the actual input '
        'has batch size %d.', self._inner_batch_size, hparams.batch_size)
    if self._inner_batch_size % hparams.batch_size != 0:
      raise ValueError(f'Lingvo input batch size {self._inner_batch_size} '
                       f'must be a multiple of p.batch_size='
                       f'{hparams.batch_size}.')
    self._current_batch_index = 0

  def get_next(self) -> py_utils.NestedMap:
    p = self.hparams
    if self._current_batch_index >= self._inner_batch_size:
      self._current_batch = super().get_next()
      self._current_batch_index = 0

    def _get_subrows(b):
      start = self._current_batch_index
      return b[start:start + p.batch_size]

    ret = jax.tree_util.tree_map(_get_subrows, self._current_batch)
    self._current_batch_index += p.batch_size
    return ret

  def reset(self) -> None:
    super().reset()
    self._current_batch = super().get_next()
    self._current_batch_index = 0


class LingvoEvalAdaptor(LingvoInputAdaptor):
  """A similar adapter as LingvoInputAdaptor, but specifically for eval data.

  LingvoEvalAdaptor automatically pads eval data so that it is easy to run eval
  on one epoch. This adaptor handles multihost infeed and padding the data to
  multiples of global batch size.

  We make the following assumptions on the underlying Lingvo input p.input:
  * it returns the entire data in a deterministic manner;
  * it is finite (does not repeat).

  To avoid dropping remainder issues and to guarantee the eval dataset is
  complete, we recommend that the underlying Lingvo input p.input uses a batch
  size of 1. (The actual batch size used on this input is specified on
  `p.batch_size`.)

  Example usage:
      p = LingvoEvalAdaptor.HParams(...)
      p.input.bucket_batch_limit = [1]
      p.batch_size = 4
  """
  _VALIDATE_BATCH_SIZE_NOT_NONE = True
  _VALIDATE_BATCH_SIZE_NONE = False

  @classmethod
  def get_batch_size(cls, hparams: LingvoInputAdaptor.HParams) -> int:
    return hparams.batch_size

  def __init__(self, hparams: LingvoInputAdaptor.HParams):
    super().__init__(hparams)
    if hparams.is_training:
      raise ValueError('LingvoEvalAdaptor requires p.is_traing=False.')
    if not hparams.reset_for_eval:
      raise ValueError('LingvoEvalAdaptor requires p.reset_for_eval=True.')
    self._num_samples = None
    self._dataset = self._get_dataset()
    self._iter = self._dataset.as_numpy_iterator()

  def _update_file_random_seed(self) -> None:
    """Updates file random seed.

    This overrides LingvoInputAdaptor._update_file_random_seed where each host
    is assigned a different file random seed. It does nothing to make sure every
    host uses the same file random seed in hparams.input.
    """
    pass

  @property
  def num_samples(self) -> int:
    return self._num_samples

  def _pad(self, ds: tf.data.Dataset) -> tf.data.Dataset:
    """Pads the dataset to be multiples of global batch size."""

    def _add_weight(b):
      b.eval_sample_weights = 1.0
      return b

    def _add_pad(b):
      b.eval_sample_weights = 0.0
      return b

    ds = ds.map(_add_weight)
    p = self.hparams
    eval_set_size = len(list(ds.as_numpy_iterator()))
    self._num_samples = eval_set_size
    logging.info(
        'LingvoEvalAdaptor p.name=%s contains %d examples before '
        'padding.', p.name, eval_set_size)
    global_batch_size = p.batch_size * p.num_infeed_hosts
    if eval_set_size % global_batch_size == 0:
      return ds
    total_size = (eval_set_size // global_batch_size + 1) * global_batch_size
    logging.info(
        'LingvoEvalAdaptor p.name=%s is expanded to contain %d '
        'examples globally after padding.', p.name, total_size)
    pad_ds = ds.take(1).map(_add_pad).repeat(total_size - eval_set_size)
    return ds.concatenate(pad_ds)

  def _get_dataset(self) -> tf.data.Dataset:
    """Returns the eval dataset."""
    data_tensor = super().get_next()
    while True:
      try:
        new_data = super().get_next()
      except (tf.errors.OutOfRangeError, StopIteration):
        break
      data_tensor = jax.tree_util.tree_map(
          lambda x, y: tf.concat([x, y], axis=0), data_tensor, new_data)
    ds = tf.data.Dataset.from_tensor_slices(data_tensor)
    ds = self._pad(ds)
    p = self.hparams
    ds = ds.batch(p.batch_size, drop_remainder=False)
    return ds.shard(num_shards=p.num_infeed_hosts, index=p.infeed_host_index)

  def get_next(self) -> py_utils.NestedMap:
    return next(self._iter)

  def reset(self) -> None:
    self._iter = self._dataset.as_numpy_iterator()


class LingvoLazyEvalAdaptor(LingvoInputAdaptor):
  """A similar adapter as LingvoEvalAdaptor, but not load all data to memory.

  LingvoLazyEvalAdaptor helps multi-host evaluation to get the same set of
  examples as the single-host evaluation gets, by returning data in a sharded
  manner. The main difference from `LingvoEvalAdaptor` is that this class
  (`LingvoLazyEvalAdaptor`) does NOT read the whole dataset into memory. This
  can be helpful if the eval set is quite large.

  We make the following assumptions on the underlying Lingvo input p.input:
  * it returns the entire data in a deterministic manner;
  * it knows the number of samples in p.input.num_samples;
  * it has a feature named `eval_sample_weights` to represent the validity of
      each sample, 1 for valid and 0 for invalid;
  * it repeats after all samples are exhausted (this requirement can be easily
      lifted in the future if necessary).

  The batch_size is handled by the underlying Lingvo input: p.input.batch_size.

  Example usage:
      input_p.batch_size = 4
      p = LingvoLazyEvalAdaptor.HParams(input_p)
  """
  _VALIDATE_BATCH_SIZE_NOT_NONE = False
  _VALIDATE_BATCH_SIZE_NONE = True

  def __init__(self, hparams: LingvoInputAdaptor.HParams):
    super().__init__(hparams)
    if hparams.is_training:
      raise ValueError('LingvoLazyEvalAdaptor requires p.is_traing=False.')
    if not hparams.reset_for_eval:
      raise ValueError('LingvoLazyEvalAdaptor requires p.reset_for_eval=True.')
    if hparams.infeed_host_index >= hparams.num_infeed_hosts:
      raise ValueError('Must have infeed_host_index < num_infeed_hosts')
    if (not isinstance(hparams.input.batch_size, int) or
        hparams.input.batch_size <= 0):
      raise ValueError('Must have positive batch_size in the underlying input: '
                       f'get {hparams.input.batch_size} instead.')
    self.batch_size = self.get_batch_size(hparams)
    # Global batch size across all hosts
    global_batch_size = hparams.num_infeed_hosts * self.batch_size
    # Global number of samples across all hosts
    global_num_samples = hparams.input.num_samples
    # Number of batches each host should at least have
    num_batches = hparams.input.num_samples // global_batch_size
    # Number of samples each host should at least have
    num_samples = num_batches * self.batch_size
    # The remaining samples after distributing evenly across hosts
    num_samples_remainder = global_num_samples - num_batches * global_batch_size
    # One more batch to handle the remaining samples.
    if num_samples_remainder > 0:
      num_batches += 1
    # Number of hosts which handle a full batch for the remaining samples.
    num_full_batches_remainder = num_samples_remainder // self.batch_size
    # Number of samples less than a full batch, assigned to the last host.
    num_samples_remainder -= num_full_batches_remainder * self.batch_size
    if hparams.infeed_host_index < num_full_batches_remainder:
      # These hosts need to handle a full batch for the remaining samples.
      num_samples += self.batch_size
    elif hparams.infeed_host_index == num_full_batches_remainder:
      # This host needs to handle a partial (possibly empty) batch for the
      # remaining samples.
      num_samples += num_samples_remainder
    self._num_samples = num_samples
    self.num_batches = num_batches
    self.reset()

  def _update_file_random_seed(self) -> None:
    """Updates file random seed.

    This overrides LingvoInputAdaptor._update_file_random_seed where each host
    is assigned a different file random seed. It does nothing to make sure every
    host uses the same file random seed in hparams.input.
    """
    pass

  @property
  def num_samples(self) -> int:
    return self._num_samples

  def get_next(self):
    # If this host has emitted enough number of batches, it should stop. Please
    # note that a host may not have emitted enough batches even if it has
    # emitted all its samples. For example, there are 2 hosts with batch size 1,
    # and there are totally only 1 sample. In this case, the first host has 1
    # sample and the second host has 0 samples. But the second host should still
    # emit 1 batch, with all invalid samples.
    if self._num_batches_emitted == self.num_batches:
      raise tf.errors.OutOfRangeError(
          node_def=None,
          op=None,
          message=f'{self.num_batches} batches have been exhausted.')
    # Number of remaining samples this host has.
    remaining_samples = self._num_samples - self._num_examples_emitted
    ret = super().get_next()
    self._num_batches_emitted += 1
    # Number of valid samples this batch has.
    num_valid_samples = min(self.batch_size, remaining_samples)
    if 'eval_sample_weights' not in ret:
      raise ValueError('eval_sample_weights must be included in the data')
    # Sets the weight of invalid samples to 0.
    ret.eval_sample_weights[num_valid_samples:] = 0.
    self._num_examples_emitted += num_valid_samples
    remaining_samples -= num_valid_samples
    # If there are still remaining samples in this host, skip n-1 batches which
    # belong to other hosts.
    if remaining_samples > 0:
      for _ in range(self.hparams.num_infeed_hosts - 1):
        super().get_next()
    return ret

  def reset(self):
    super().reset()
    # Skips k batches which belong to other hosts.
    for _ in range(self.hparams.infeed_host_index):
      super().get_next()
    self._num_examples_emitted = 0
    self._num_batches_emitted = 0


class MultiInput(BaseInput):
  """Wraps children inputs and outputs a combined batch at each step.

  During Pax's train, on each host input instances for all children
  inputs will be created (instantiate(input_p)), and then get_next() is
  iteratively called for each child in eager mode to generate one batch of
  data for each step. Each batch will contain a batch from all children
  input generators nested into a NestedMap.
  """
  _VALIDATE_BATCH_SIZE_NOT_NONE = False  # Validated separately for children.
  _VALIDATE_BATCH_SIZE_NONE = True  # Can't set batch size for wrapper.

  class HParams(BaseInput.HParams):
    """Hyper-parameters associated with this Input class.

    Attributes:
      input_to_params: Dict from input names to input generator parameter
        definitions for each input. Input generators need to implement
        BaseInput.
      default_input: Default input to use for ids_to_strings or other
        input generator methods.
    """
    input_to_params: Dict[str, BaseInput.HParams] = None
    default_input: str = None

  @classmethod
  def get_batch_size(cls, hparams: MultiInput.HParams) -> int:
    assert hparams.input_to_params
    logging.warning(
        'get_batch_size for MultiInput only returns batch size for the first '
        'input. This might be different from batch sizes for other inputs.'
    )
    first = list(hparams.input_to_params.values())[0]
    return first.cls.get_batch_size(first)

  def __init__(self, hparams: MultiInput.HParams) -> None:
    if self._VALIDATE_BATCH_SIZE_NONE and hparams.batch_size is not None:
      raise ValueError('MultiInput does not support p.batch_size. '
                       'Please specify batch size on each child input '
                       'separately.')
    if not hparams.name:
      hparams.name = 'train' if hparams.is_training else 'input'
    super().__init__(hparams)
    name = self.hparams.name
    if re.fullmatch(r'[a-zA-Z0-9.:_-]+', name) is None:
      raise ValueError(f'Input hparams p.name string invalid: "{name}" '
                       'does not fully match "[a-zA-Z0-9._-]+".')
    if hparams.input_to_params is None:
      raise ValueError('Need to define inputs.')

    if hparams.reset_for_eval and len(hparams.input_to_params) > 1:
      raise ValueError(
          'Only 1 input can be specified when using reset_for_eval.')

    self._inputs = {}
    for input_name, input_params in hparams.input_to_params.items():
      # Overriding params for children to match parent.
      input_params.num_infeed_hosts = hparams.num_infeed_hosts
      input_params.infeed_host_index = hparams.infeed_host_index
      input_params.is_training = hparams.is_training
      input_params.reset_for_eval = hparams.reset_for_eval
      input_params.eval_loop_num_batches = hparams.eval_loop_num_batches
      input_params.name = hparams.name + '_' + input_name

      self._inputs[input_name] = instantiate(input_params)

  def get_next(self) -> NestedJTensor:
    input_batches = {}
    for input_name, input_gen in self._inputs.items():
      input_batches[input_name] = input_gen.get_next()
    return NestedMap(input_batches)

  def get_child(self, input_name: str) -> NestedJTensor:
    return self._inputs[input_name]

  def reset(self) -> None:
    for _, input_gen in self._inputs.items():
      input_gen.reset()

  def ids_to_strings(self,
                     ids: pytypes.NpTensor,
                     lengths: pytypes.NpTensor,
                     key: Optional[str] = None,
                     input_name: Optional[str] = None) -> Sequence[str]:
    """Converts int ids into strings using a particular input.

    Args:
      ids: A matrix of shape [batch, seqlen], each row is a sequence to be
        converted.
      lengths: A vector of shape [batch]. lens[i] is the sequence length of the
        i-th row. Only the first lens[i] tokens in ids[i, :] are valid tokens.
      key: Optional argument to specify whether a tokenizer to use is the source
        or target. This is useful for example in a sequence model where the
        source and targets have different tokenizers. For the source corpus the
        key should be `src` while for the target corpus the key should be `tgt`.
      input_name: Argument specifying which input's ids_to_strings to call.

    Returns:
      A list strings of shape [batch]. The converted texts.
    """
    if input_name is None:
      input_name = self.hparams.default_input
    return self._inputs[input_name].ids_to_strings(ids, lengths, key)


class BaseInputSpecsProvider(
    base_hyperparams.BaseParameterizable, metaclass=abc.ABCMeta):
  """Base class to provide input specs for model initialization.

  This helper class is added for shape inference support.
  """

  @abc.abstractmethod
  def get_input_specs(self) -> NestedShapeDtypeStruct:
    """Returns example input specs for model initialization."""


class DatasetInputSpecsProvider(BaseInputSpecsProvider):
  """Class to provide input specs from a dataset for model initialization."""

  class HParams(BaseInputSpecsProvider.HParams):
    """Hyper-parameters for this parameterizable component."""
    input_p: Optional[BaseInput.HParams] = None

  def get_input_specs(self) -> NestedShapeDtypeStruct:
    """Returns example input specs from the input pipeline for model init."""
    # Note that this re-instantiate the input pipeline every time
    # `.get_input_specs()` is called. In practice, we typically call this
    # method only once at model initialization time.
    # Few inputs (e.g. BaseInput) may try to still mutate this hparams.
    # Clone it to make it mutable for now before calliing instantiate.
    input_pipeline = instantiate(self.hparams.input_p.clone())
    return jax.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
                        input_pipeline.get_next_padded())
