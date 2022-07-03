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

import copy
import re
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

from absl import logging
from lingvo.core import cluster_factory
from lingvo.core import datasource
import numpy as np
from praxis import base_hyperparams
from praxis import py_utils
from praxis import pytypes
import tensorflow.compat.v2 as tf

NestedMap = py_utils.NestedMap
NestedJTensor = pytypes.NestedJTensor
Nested = pytypes.Nested
instantiate = base_hyperparams.instantiate


class BaseInput(base_hyperparams.BaseParameterizable):
  """Base class for Jax input classes.

  During Lingvo Jax's train, on each host an input instance will be
  created (instantiate(input_p)), and then get_next() is iteratively
  called in eager mode to generate one batch of data for each step
  of train/eval/etc.

  If supported, for eval, reset() is called after each eval step.
  See p.reset_for_eval below.

  A tf.data based input should inherit this class directly and implement
  get_next() and reset(). For an example of how to handle sharding for both
  training and eval data, please refer to the implementation of
  TFRecordBertInput at tasks/lm/input_generator.py.

  If there is already an Lingvo TF input generator that one would like to
  use directly, please use LingvoInputAdaptor below.

  tf_graph_get_next() is an optional TF graph mode implementation of get_next
  and returns output tensors and init ops. It's used when
  experimental_remote_input is set.
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
      experimental_remote_input: How to process inputs on remote hosts, when
        there is a single controller. If set, it requires an implementation of
        tf_graph_get_next().
      batch_padding_size: The amount of right-padding applied to each invocation
        of get_next_padded(). Useful when the batch_size is smaller than the
        number of devices.
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
    experimental_remote_input: Optional[RemoteInput] = None
    batch_padding_size: int = 0

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
    if hparams.experimental_remote_input:
      if hparams.batch_padding_size > 0:
        raise NotImplementedError('Remote input padding not implemented')
      self._remote_input = hparams.experimental_remote_input.Instantiate(
          input_fn=self.tf_graph_get_next)

  def tf_graph_get_next(
      self, host_id: int) -> tuple[Nested[tf.Tensor], Iterable[tf.Operation]]:
    """TF graph mode implementation of get next.

    Only need to implement this method when the subclass wants to support
    experimental_remote_input.

    Args:
      host_id: An integer. The index of the host where to get the data. When
        experimental_remote_input is true, hparams.infeed_host_index is the
        index of the local device. Use host_id as infeed host index to get data
        from a remote host.

    Returns:
      A tuple of nested output tensors and a list of initialization ops.
    """
    raise NotImplementedError

  def get_next(self) -> NestedJTensor:
    raise NotImplementedError

  def get_next_padded(self) -> NestedJTensor:
    unpadded = self.get_next()
    pad_size = self.hparams.batch_padding_size
    if pad_size == 0:
      return unpadded
    return tf.nest.map_structure(
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
      key: Optional argument to specify whether a tokenizer to use is the source
        or target. This is useful for example in a sequence model where the
        source and targets have different tokenizers. For the source corpus the
        key should be `src` while for the target corpus the key should be `tgt`.

    Returns:
      A list strings of shape [batch]. The converted texts.
    """
    raise NotImplementedError


class RemoteInput(base_hyperparams.BaseParameterizable):
  """Base class for remote input processing."""

  class HParams(base_hyperparams.BaseParameterizable.HParams):
    """Hyper-parameters for RemoteInput."""

  def __init__(self, hparams: RemoteInput.HParams,
               input_fn: Callable[..., Any]) -> None:
    super().__init__(hparams)
    self._input_fn = input_fn

  def get_next(self) -> pytypes.NestedJTensor:
    raise NotImplementedError

  def reset(self) -> None:
    raise NotImplementedError


class LingvoInputAdaptor(BaseInput):
  """Syntactic sugar for adapting a Lingvo style input for Jax.

  This should be able to wrap any Lingvo TF input generator to be used in
  Lingvo Jax. Remember to set `p.is_training=True` on the training dataset.

  Some usage caveats below.

  For eval, `p.num_samples` or other similar params like samples_per_summary are
  completely ignored by Lingvo Jax. Caller should instead set `p.num_batches` to
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

  def _update_file_random_seed(self, infeed_host_index) -> None:
    """Updates file random seed to use different seeds for different hosts."""
    p = self.hparams
    if hasattr(p.input, 'file_random_seed') and p.input.file_random_seed:
      if not p.allow_fixed_file_random_seed:
        raise ValueError(
            'Training data using fixed non-zero file_random_seed: '
            f'p.input.file_random_seed={p.input.file_random_seed}. '
            'This means each host *might* infeed identical batches. You can set '
            'p.input.file_random_seed = 0, or if certain this is intended, '
            'suppress this error by setting p.allow_fixed_file_random_seed = '
            'True.')
      # Make sure each host uses a different random seed.
      p.input.file_random_seed += infeed_host_index

  def _initialize(self) -> None:
    """Initializes the relevant fields of this adaptor input."""
    p = self.hparams
    self._update_file_random_seed(p.infeed_host_index)
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
      if p.experimental_remote_input:
        raise NotImplementedError(
            'Distributed input processing for datasource is not supported yet.')
    else:
      self._get_next_fn = tf.function(self._get_batch)
    self._num_batches_produced = 0

  def _get_batch(self, host_index: Optional[int] = None) -> NestedMap:
    p = self.hparams
    if not p.experimental_remote_input and host_index is not None:
      raise ValueError(
          'Unexpected host index when experimental_remote_input is false.')
    infeed_host_index = (
        p.infeed_host_index if host_index is None else host_index)
    with py_utils.infeed_context_scope(
        infeed_host_index=infeed_host_index,
        num_infeed_hosts=p.num_infeed_hosts), self._cluster:
      ret = self.input.GetPreprocessedInputBatch()
    # Remove unsupported string (byte) array from input.
    return ret.Filter(lambda v: v.dtype != tf.string)

  def tf_graph_get_next(
      self, host_id: int) -> tuple[Nested[tf.Tensor], Iterable[tf.Operation]]:
    self._update_file_random_seed(host_id)
    ret = self._get_batch(host_id)
    batch_size = tf.nest.flatten(ret)[0].shape[0]
    ret.eval_sample_weights = tf.ones([batch_size], tf.float32)
    return ret, []

  def get_next(self) -> NestedJTensor:
    p = self.hparams
    if p.num_batches is not None and p.num_batches > 0:
      if self._num_batches_produced >= p.num_batches:
        raise tf.errors.OutOfRangeError(
            node_def=None,
            op=None,
            message=f'num_batches exceeding {self._num_batches_produced}')
      self._num_batches_produced += 1
    if p.experimental_remote_input:
      ret = self._remote_input.get_next()
    else:
      ret = self._get_next_fn()
      ret = tf.nest.map_structure(lambda x: x.numpy(), ret)
      batch_size = tf.nest.flatten(ret)[0].shape[0]
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
    if self.hparams.experimental_remote_input:
      self._remote_input.reset()

  def ids_to_strings(self,
                     ids: pytypes.NpTensor,
                     lengths: pytypes.NpTensor,
                     key: Optional[str] = None) -> Sequence[str]:
    """Converts int ids into strings."""
    bytes_list = self.input.IdsToStrings(ids, lengths, key=key)
    if isinstance(bytes_list, tf.Tensor):
      bytes_list = bytes_list.numpy()
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
    self._inner_batch_size = tf.nest.flatten(self._current_batch)[0].shape[0]
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

    ret = tf.nest.map_structure(_get_subrows, self._current_batch)
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

  def __init__(self, hparams: LingvoInputAdaptor.HParams):
    super().__init__(hparams)
    if hparams.is_training:
      raise ValueError('LingvoEvalAdaptor requires p.is_traing=False.')
    if not hparams.reset_for_eval:
      raise ValueError('LingvoEvalAdaptor requires p.reset_for_eval=True.')
    self._num_samples = None
    self._dataset = self._get_dataset()
    self._iter = self._dataset.as_numpy_iterator()

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
      data_tensor = tf.nest.map_structure(
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


class MultiStreamInput(BaseInput):
  """Wraps children inputs and outputs a combined batch at each step.

  During Lingvo Jax's train, on each host input instances for all children
  inputs will be created (instantiate(input_p)), and then get_next() is
  iteratively called for each child in eager mode to generate one batch of
  data for each step. Each batch will contain a batch from all children
  input generators nested into a NestedMap.

  NOTE: Batch sizes need to be equal across all input streams to work with pmap.
  """
  _VALIDATE_BATCH_SIZE_NOT_NONE = False  # Validated separately for children.
  _VALIDATE_BATCH_SIZE_NONE = True  # Can't set batch size for wrapper.

  class HParams(BaseInput.HParams):
    """Hyper-parameters associated with this Input class.

    Attributes:
      input_streams: Dict from stream names to input generator parameter
        definitions for each stream. Input generators need to implement
        BaseInput.
      default_stream: Default input stream to use for ids_to_strings or other
        input generator methods.
    """
    input_streams: Dict[str, BaseInput] = None
    default_stream: str = None

  @classmethod
  def get_batch_size(cls, hparams: MultiStreamInput.HParams) -> int:
    assert hparams.input_streams
    first = list(hparams.input_streams.values())[0]
    return first.cls.get_batch_size(first)

  def __init__(self, hparams: MultiStreamInput.HParams) -> None:
    if self._VALIDATE_BATCH_SIZE_NONE and hparams.batch_size is not None:
      raise ValueError('MultiStreamInput does not support p.batch_size. '
                       'Please specify batch size on each child input stream '
                       'separately.')
    if not hparams.name:
      hparams.name = 'train' if hparams.is_training else 'input'
    super().__init__(hparams)
    name = self.hparams.name
    if re.fullmatch(r'[a-zA-Z0-9.:_-]+', name) is None:
      raise ValueError(f'Input hparams p.name string invalid: "{name}" '
                       'does not fully match "[a-zA-Z0-9._-]+".')
    if hparams.input_streams is None:
      raise ValueError('Need to define input streams.')

    if hparams.reset_for_eval and len(hparams.input_streams) > 1:
      raise ValueError(
          'Only 1 input stream can be specified when using reset_for_eval.')

    self._input_streams = {}
    for stream_name, stream_params in hparams.input_streams.items():
      # Overriding params for children to match parent.
      stream_params.num_infeed_hosts = hparams.num_infeed_hosts
      stream_params.infeed_host_index = hparams.infeed_host_index
      stream_params.is_training = hparams.is_training
      stream_params.reset_for_eval = hparams.reset_for_eval
      stream_params.eval_loop_num_batches = hparams.eval_loop_num_batches
      stream_params.name = hparams.name + '_' + stream_name

      self._input_streams[stream_name] = instantiate(stream_params)

  def get_next(self) -> NestedJTensor:
    stream_batches = {}
    for stream_name, stream in self._input_streams.items():
      stream_batches[stream_name] = stream.get_next()
    return NestedMap(stream_batches)

  def reset(self) -> None:
    for _, stream in self._input_streams.items():
      stream.reset()

  def ids_to_strings(self,
                     ids: pytypes.NpTensor,
                     lengths: pytypes.NpTensor,
                     key: Optional[str] = None,
                     stream: Optional[str] = None) -> Sequence[str]:
    """Converts int ids into strings using a particular input stream.

    Args:
      ids: A matrix of shape [batch, seqlen], each row is a sequence to be
        converted.
      lengths: A vector of shape [batch]. lens[i] is the sequence length of the
        i-th row. Only the first lens[i] tokens in ids[i, :] are valid tokens.
      key: Optional argument to specify whether a tokenizer to use is the source
        or target. This is useful for example in a sequence model where the
        source and targets have different tokenizers. For the source corpus the
        key should be `src` while for the target corpus the key should be `tgt`.
      stream: Argument specifying which input stream's ids_to_strings to call.

    Returns:
      A list strings of shape [batch]. The converted texts.
    """
    if stream is None:
      stream = self.hparams.default_stream
    return self._input_streams[stream].ids_to_strings(ids, lengths, key)
