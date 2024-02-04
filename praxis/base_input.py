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

"""Base classes for the Praxis input layers."""

from __future__ import annotations

import abc
import copy
import dataclasses
import inspect
import math
import re
from typing import Any, Callable, Sequence

from absl import logging
from etils import epath
import fiddle as fdl
import jax
from jax.lib import xla_client as xc
import jax.tree_util
import numpy as np
from praxis import base_hyperparams
from praxis import lazy_loader
from praxis import lingvo_lib
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes

# TF is slow to import, so we do it lazily.
tf_v1 = lazy_loader.LazyLoader('tf_v1', globals(), 'tensorflow.compat.v1')
tf = lazy_loader.LazyLoader('tf', globals(), 'tensorflow.compat.v2')

NestedMap = py_utils.NestedMap
NestedJTensor = pytypes.NestedJTensor
Nested = pytypes.Nested
NestedShapeDtypeStruct = pytypes.NestedShapeDtypeStruct
NestedPartitionSpec = pytypes.NestedPartitionSpec
instantiate = base_hyperparams.instantiate

_NAME_REGEX = r'[a-zA-Z0-9.:_+-]+'


def _create_input(hparams: Any, state: bytes) -> Any:
  inp = instantiate(hparams)
  inp.set_state(state)
  return inp


def _to_numpy(x: tf.Tensor) -> np.ndarray:
  """Creates a Numpy array from an EagerTensor whithout copying.

  If the batch Tensors are extremely large, `.numpy()` (which creates
  a copy) will be too slow.

  Args:
    x: An EagerTensor.

  Returns:
    An immutable np.ndarray.
  """
  if hasattr(x, '_numpy'):
    numpy = x._numpy()  # pylint: disable=protected-access
  else:
    numpy = x.numpy()
  if isinstance(numpy, np.ndarray):
    # `numpy` shares the same underlying buffer as the `x` Tensor.
    # Tensors are expected to be immutable, so we disable writes.
    numpy.setflags(write=False)
  return numpy


class BaseInput(base_hyperparams.FiddleBaseParameterizable):
  """Base class for Praxis input pipelines.

  During paxml's train, on each host an input instance will be
  created (instantiate(input_p)), and then get_next() is iteratively
  called in eager mode to generate one batch of data for each step
  of train/eval/etc.

  If supported, for eval, reset() is called after each eval step.
  See self.reset_for_eval below.

  A tf.data based input should inherit this class directly and implement
  get_next() and reset(). For an example of how to handle sharding for both
  training and eval data, please refer to the implementation of
  TFRecordBertInput at tasks/lm/input_generator.py.

  Supporting checkpointing:
  Subclasses that support checkpointing of the input pipeline can enable
  preemption handling via 2 paths:
  1) Implement _get_state_internal() and _set_state_internal(). This will enable
     deterministic training. _get_state_internal() should be cheap as it might
     be executed more often than actual checkpointing. This is required to
     support `peek_padded()` correctly.
  2) Implement save() and restore(). This will allow Pax to handle preemptions
     but it might skip a batch each time the job restarts. Usually skipping
     a single batch doesn't impact model quality.

  If there is already a Lingvo TF input generator that one would like to
  use directly, please use LingvoInputAdaptor below.

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
      raised, and reset() will called for each eval. Implementation must ensure
      that all variant self.infeed_host_index instances raise after the same
      number of calls to get_next() to ensure synchronization across hosts. If
      not set, get_next() must never raise.
    eval_loop_num_batches: An integer representing the number of batches to
      process per eval loop. If >= 1, each eval loop will process this many
      batches. Metrics over those batches will be aggregated and then reported.
      If set to `-1`, each eval loop will use all batches.
    is_training: Whether or not this dataset is used for model traning.
    experimental_remote_input: whether to process inputs on remote hosts, when
      there is a single controller.
    batch_padding_size: The amount of right-padding applied to each invocation
      of get_next_padded(). Useful when the batch_size is smaller than the
      number of devices.
    custom_device_order: Custom order of devices in GSPMD sharding for the
      inputs. This is needed when there are data paddings on some devices in a
      multi-process environment. Values in the list are logical partition IDs
      (offsets in global_mesh.devices.flat) in the global mesh.
    input_checkpointing_enabled: overridden by
      task.train.enable_input_checkpointing; indicates whether training input
      should be checkpointed.
    tf_data_service_address: May be set to the address of the tf.data service
      dispatcher. This is usually set automatically by the trainer (not by the
      user). tf.data based input pipelines can use this to distribute input
      processing across many worker jobs.
    dataset: May be set to the underlying tf.data.Dataset, if exists. This is
      used by DatasetInputSpecsProvider to create input specs.
  """

  _VALIDATE_BATCH_SIZE_NOT_NONE = True
  batch_size: int | None = None
  # Sharding behavior. If num_infeed_hosts is 0, it will be given a default
  # value by PAX trainer; if it is still not set during __init__, 1 will be
  # used.
  num_infeed_hosts: int = 0
  infeed_host_index: int = 0
  # Deterministic randomness.
  input_random_seed: int | None = None
  reset_for_eval: bool = False
  eval_loop_num_batches: int = 1
  is_training: bool = False
  experimental_remote_input: bool = False
  batch_padding_size: int = 0
  custom_device_order: Sequence[int] | None = None
  input_checkpointing_enabled: bool = False
  tf_data_service_address: str | None = None
  dataset: tf.data.Dataset | None = dataclasses.field(
      default=None, init=False, repr=False
  )
  _peek: Any = dataclasses.field(init=False, repr=False)
  _state_before_peek: Any = dataclasses.field(init=False, repr=False)

  def __post_init__(self):
    if self._VALIDATE_BATCH_SIZE_NOT_NONE and (self.batch_size is None):
      raise ValueError('Must specify self.batch_size.')
    if not self.name:
      self.name = 'train' if self.is_training else 'input'
    if self.num_infeed_hosts == 0:
      self.num_infeed_hosts = 1
    name = self.name
    if re.fullmatch(_NAME_REGEX, name) is None:
      raise ValueError(
          f'Input params self.name string invalid: "{name}" '
          f'does not fully match "{_NAME_REGEX}".'
      )
    if (
        self.eval_loop_num_batches
        and self.eval_loop_num_batches <= 0
        and self.eval_loop_num_batches != -1
    ):
      raise ValueError(
          'eval_loop_num_batches must be positive, -1, found'
          f' {self.eval_loop_num_batches}.'
      )
    if self.experimental_remote_input and jax.process_count() > 1:
      raise NotImplementedError(
          'Remote input is not supported when there are multiple controllers.')
    # Allows a simple peek into the input, while maintaining correct iteration
    # sequence in get_next_padded() call.
    self._peek = None
    self._state_before_peek = None

  @classmethod
  def get_batch_size(
      cls, params: pax_fiddle.Config[BaseInput] | BaseInput
  ) -> int:
    assert params.batch_size is not None
    return params.batch_size

  @classmethod
  def get_global_batch_size(cls, params: pax_fiddle.Config[BaseInput]) -> int:
    assert params.num_infeed_hosts is not None
    return cls.get_batch_size(params) * params.num_infeed_hosts

  def save(self, checkpoint_path: epath.PathLike):
    state = self.get_state()
    epath.Path(checkpoint_path).write_bytes(state)

  def restore(self, checkpoint_path: epath.PathLike):
    state = epath.Path(checkpoint_path).read_bytes()
    self.set_state(state)

  def get_state(self) -> bytes:
    """Returns the serialized state as bytes object."""
    if self._peek is not None:
      if not self._state_before_peek:
        raise ValueError(
            'get_state() should only be used when input_checkpointing_enabled ='
            ' True. Otherwise, the peeked iterator state will not be captured.'
        )
      return self._state_before_peek
    return self._get_state_internal()

  def _get_state_internal(self) -> bytes:
    """Returns the serialized state as bytes object."""
    raise NotImplementedError

  def set_state(self, state: bytes) -> None:
    """Set the internal state from serialized bytes."""
    self._peek = None
    self._state_before_peek = None
    self._set_state_internal(state)

  def _set_state_internal(self, state: bytes) -> None:
    """Set the internal state from serialized bytes."""
    raise NotImplementedError

  def __reduce__(self) -> tuple[Callable[[Any, bytes], Any], tuple[Any, bytes]]:
    """Returns a callable to recreate the input object in its current state.

    Adheres to the contract of object.__reduce__().
    """
    hparams = self.hparams.clone()
    state = self.get_state()
    return (_create_input, (hparams, state))

  def get_next(self) -> NestedJTensor:
    raise NotImplementedError

  def get_child(self, input_name: str) -> NestedJTensor:
    raise NotImplementedError

  @py_utils.benchmark('[PAX STATUS]: ', first_n=2)
  def get_next_padded(self) -> NestedJTensor:
    """Gets next padded example from the input pipeline.

    If the example is `peeked` previously, returns the peeked example without
    actually calling into data pipeline so that we maintain the correct data
    iteration.

    Note that, if method is overridden in subclasses, it is user's duty to
    ensure peek behavior, or `peek_padded()` will lead to inconsistent
    states/results.

    Returns:
      The padded example from the data pipeline.
    """
    if self._peek is not None:
      output = self._peek
      self._peek = None
      self._state_before_peek = None
      return output
    unpadded = self.get_next()
    pad_size = self.batch_padding_size
    if pad_size == 0:
      return unpadded
    return jax.tree_util.tree_map(
        lambda x: np.pad(x, [[0, pad_size]] + [[0, 0]] * (x.ndim - 1)),
        unpadded,
    )

  def peek_padded(self) -> NestedJTensor | None:
    """Peeks into the current input data pipeline."""
    if self._peek is not None:
      return self._peek
    assert (
        self._state_before_peek is None
    ), "_peek was None, but _state_before_peek wasn't None"
    try:
      # Not all subclasses support _get_state_internal().
      try:
        self._state_before_peek = self._get_state_internal()
      except NotImplementedError:
        pass
      self._peek = self.get_next_padded()
    except (tf.errors.OutOfRangeError, StopIteration):
      logging.warning('Peek failed: input %s out of range.', self.name)
      self._peek = None
      self._state_before_peek = None
    return self._peek

  def reset(self) -> None:
    self._peek = None
    self._state_before_peek = None

  def ids_to_strings(
      self,
      ids: pytypes.NpTensor,
      lengths: pytypes.NpTensor,
      key: str | None = None,
  ) -> Sequence[str]:
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
    assert self.custom_device_order is None
    return jax.tree_util.tree_map(py_utils.reshard, arrays)

  def reshard_for_spmd(self, arrays: NestedJTensor,
                       global_mesh: jax.sharding.Mesh,
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
      global_mesh: Global mesh for pjit computation.
      pspecs: Expected partition specs for `arrays` after resharding.

    Returns:
      Resharded inputs.
    """
    global_shapes = jax.tree_util.tree_map(
        py_utils.get_global_input_shape_dtype, arrays
    )
    device_order = self.custom_device_order
    if device_order is None:
      return py_utils.make_array(arrays, global_shapes, global_mesh, pspecs)
    assert len(device_order) == jax.device_count()

    # Use custom device order to create OpSharding in jax.Array.
    def _make_array(x, global_shape):
      op_sharding = xc.OpSharding()
      op_sharding.type = xc.OpSharding.Type.OTHER
      # Fully sharded on the batch dim.
      op_sharding.tile_assignment_dimensions = [len(device_order)] + [1] * (
          len(global_shape.shape) - 1)
      # Custom device order.
      op_sharding.tile_assignment_devices = device_order
      dbs = py_utils.put_to_devices(x, global_mesh.local_devices)
      sharding = jax.sharding.GSPMDSharding(
          list(global_mesh.devices.flat), op_sharding)
      return jax.make_array_from_single_device_arrays(global_shape.shape,
                                                      sharding, dbs)

    return jax.tree_util.tree_map(_make_array, arrays, global_shapes)


class LingvoInputAdaptor(BaseInput):
  """Syntactic sugar for adapting a Lingvo style input for Pax.

  This should be able to wrap any Lingvo TF input generator to be used in
  Pax. Remember to set `self.is_training=True` on the training dataset.

  Some usage caveats below.

  For eval, `self.num_samples` or other similar params like samples_per_summary
  are
  completely ignored by Pax. Caller should instead set `self.num_batches` to
  (self.num_samples // batch_size) with `self.reset_for_eval=True` so that each
  eval
  step reads (approximately) one epoch of eval data. This might not be needed if
  the input already is finite (e.g. with self.repeat_count=1).

  When multiple infeed hosts are used, one must take care to ensure that the
  Lingvo input either already uses InfeedContextScope for proper sharding, or
  alternatively do not use the same random seed on all hosts. In other words,
  one must avoid the failure case where each host emits identical training data.
  See also self.allow_fixed_file_random_seed below.

  Attributes:
    input: Params of a Lingvo input generator.
    num_batches: If specified and positive, raises tf.errors.OutOfRange after
      this manybatches have been produced. This forces a raise after get_next()
      is called this many times, to support self.reset_for_eval=True.
    allow_fixed_file_random_seed: If not set, disallows a fixed, non-zero
      self.input.file_random_seed. We disallow by default to avoid having
      identical input batches across different infeed hosts. If set, random
      seeds are adjusted by self.infeed_host_index to ensure different random
      seeds.
    cluster_do_eval: Whether to set cluster.do_eval to True for non-training
      data. Note that if set to True, this will change
      cluster.require_sequential_input_order to True as a result. Ignored when
      self.is_training is True.
  """
  _VALIDATE_BATCH_SIZE_NOT_NONE = False
  _VALIDATE_BATCH_SIZE_NONE = True
  input: py_utils.InstantiableParams | None = None
  num_batches: int | None = None
  allow_fixed_file_random_seed: bool = False
  cluster_do_eval: bool = False
  _cluster: Any = dataclasses.field(init=False, repr=False)
  input_inst: Any = dataclasses.field(init=False, repr=False)
  _get_next_fn: Any = dataclasses.field(init=False, repr=False)
  _num_batches_produced: Any = dataclasses.field(init=False, repr=False)

  def __post_init__(self):
    assert self.input is not None
    if self._VALIDATE_BATCH_SIZE_NONE and self.batch_size is not None:
      raise ValueError(
          'LingvoInputAdaptor does not support self.batch_size. '
          'Please specify batch size on self.input, e.g. with '
          'self.input.bucket_batch_limit = [4] or '
          'self.input.args.batch=4, depeding the Lingvo input '
          f'used. Currently: self.batch_size={self.batch_size}, '
          'it must be None.'
      )
    if not self.name:
      self.name = self.input.name
    if self.input is None:
      raise ValueError('Params of a Lingvo input generator is not set.')
    if (
        hasattr(self.input, 'file_random_seed')
        and (self.input.file_random_seed)
        and (not self.allow_fixed_file_random_seed)
    ):
      raise ValueError(
          'Input data using fixed non-zero file_random_seed: '
          f'self.input.file_random_seed={self.input.file_random_seed}. '
          'This means each host *might* infeed identical batches. You can set '
          'self.input.file_random_seed = 0, or if certain this is intended, '
          'suppress this error by setting self.allow_fixed_file_random_seed '
          '= True.'
      )
    super().__post_init__()
    self._cluster = copy.deepcopy(py_utils.current_cluster())
    # For Lingvo's Cluster context that may impact the behavior of this input
    # generator, we always set use_tpu to True, and optionally set do_eval
    # for non-training data when configured to do so. All other Cluster params
    # use the default value.
    self._cluster.params.xla_device = 'tpu'
    self._cluster.params.enable_asserts = False
    # This indirectly sets cluster.require_sequential_input_order as well.
    self._cluster.params.do_eval = not self.is_training and self.cluster_do_eval
    self._cluster.params.tf_data_service_address = self.tf_data_service_address
    self._initialize()

  @classmethod
  def get_batch_size(
      cls,
      params: pax_fiddle.Config[LingvoInputAdaptor] | LingvoInputAdaptor,
  ) -> int:
    assert params.input is not None
    if hasattr(params.input, 'bucket_batch_limit'):
      return params.input.bucket_batch_limit[0]
    elif hasattr(params.input, 'batch_size'):
      return params.input.batch_size
    else:
      raise ValueError(
          'params.input has no attribute of bucket_batch_limit or batch_size.'
      )

  def _update_file_random_seed(self) -> None:
    """Updates file random seed to use different seeds for different hosts."""
    if hasattr(self.input, 'file_random_seed') and self.input.file_random_seed:
      # Make sure each host uses a different random seed.
      self.input.file_random_seed += self.infeed_host_index

  def _initialize(self) -> None:
    """Initializes the relevant fields of this adaptor input."""
    assert self.input is not None
    self._update_file_random_seed()
    # We make self.input public so that users can access its methods like
    # IdsToStrings if needed.
    with py_utils.infeed_context_scope(
        infeed_host_index=self.infeed_host_index,
        num_infeed_hosts=self.num_infeed_hosts,
    ), self._cluster:
      self.input_inst = self.input.Instantiate()

    if hasattr(self.input_inst, 'datasource') and isinstance(
        self.input_inst.datasource, lingvo_lib.datasource.TFDatasetSource
    ):
      # For the special case when the input is implemented by a tf.data.Dataset,
      # call eagerly. Using tf.function may result in returning duplicate
      # batches.
      self._get_next_fn = self._get_batch
    else:
      self._get_next_fn = tf.function(self._get_batch)
    self._num_batches_produced = 0

  def _get_batch(self) -> NestedMap:
    with py_utils.infeed_context_scope(
        infeed_host_index=self.infeed_host_index,
        num_infeed_hosts=self.num_infeed_hosts,
    ), self._cluster:
      ret = self.input_inst.GetPreprocessedInputBatch()
    # Remove unsupported string (byte) array from input if training.
    # Also remove unsupported string (byte) array from input if there are
    # multiple hosts for eval, since xla passthrough does not support multihost
    # eval b/279795947.
    if self.is_training or self.num_infeed_hosts > 1:
      return ret.Filter(lambda v: v.dtype != tf.string)
    else:
      return ret

  def get_next(self) -> NestedJTensor:
    if self.num_batches is not None and self.num_batches > 0:
      if self._num_batches_produced >= self.num_batches:
        raise tf.errors.OutOfRangeError(
            node_def=None,
            op=None,
            message=f'num_batches exceeding {self._num_batches_produced}')
      self._num_batches_produced += 1
    ret = self._get_next_fn()
    if tf.executing_eagerly():
      ret = jax.tree_util.tree_map(_to_numpy, ret)
    else:
      with tf_v1.Session() as sess:
        ret = sess.run(ret)
    batch_size = jax.tree_util.tree_leaves(ret)[0].shape[0]
    ret.eval_sample_weights = np.ones([batch_size], np.float32)
    return ret

  def reset(self) -> None:
    if hasattr(self.input_inst, 'datasource') and isinstance(
        self.input_inst.datasource, lingvo_lib.datasource.TFDatasetSource
    ):
      self.input_inst.datasource.Reset()
      # reset counter to 0.
      self._num_batches_produced = 0
      return
    # reinstantiate the input and retrace self._get_batch.
    self._initialize()

  def ids_to_strings(
      self,
      ids: pytypes.NpTensor,
      lengths: pytypes.NpTensor,
      key: str | None = None,
  ) -> Sequence[str]:
    """Converts int ids into strings."""
    bytes_list = self.input_inst.IdsToStrings(ids, lengths, key=key)
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
  input. This class, however, allows specifying a smaller self.batch_size.
  This can be useful when the Lingvo input expects a large batch size,
  but the user wants a smaller batch size, e.g. when the Lingvo input uses
  a fixed packing factor to do packing, which can more efficiently pack with
  more data.

  We require that the batch size of the underlying Lingvo input must divide
  self.batch_size. Internally this class acts as a cache, retrieving the large
  batches from the parent class size, and consuming it by slicing it to the
  smaller batch size specified by the user.

  Example usage:
      p = pax_fiddle.Config(ChangeBatchSizeInput, ...)
      self.input.packing_factor = 3.5
      self.input.bucket_batch_limit = [4096]
      self.batch_size = 4
  """
  _VALIDATE_BATCH_SIZE_NOT_NONE = True
  _VALIDATE_BATCH_SIZE_NONE = False
  _current_batch: Any = dataclasses.field(init=False, repr=False)
  _inner_batch_size: Any = dataclasses.field(init=False, repr=False)
  _current_batch_index: Any = dataclasses.field(init=False, repr=False)

  def __post_init__(self):
    super().__post_init__()
    self._current_batch = super().get_next()
    self._inner_batch_size = jax.tree_util.tree_leaves(
        self._current_batch)[0].shape[0]
    logging.info(
        (
            'The wrapped Lingvo input has batch size %d, the actual input '
            'has batch size %d.'
        ),
        self._inner_batch_size,
        self.batch_size,
    )
    if self._inner_batch_size % self.batch_size != 0:
      raise ValueError(
          f'Lingvo input batch size {self._inner_batch_size} '
          'must be a multiple of self.batch_size='
          f'{self.batch_size}.'
      )
    self._current_batch_index = 0

  @classmethod
  def get_batch_size(
      cls, params: pax_fiddle.Config[BaseInput] | BaseInput
  ) -> int:
    assert params.batch_size is not None
    return params.batch_size

  def get_next(self) -> py_utils.NestedMap:
    if self._current_batch_index >= self._inner_batch_size:
      self._current_batch = super().get_next()
      self._current_batch_index = 0

    def _get_subrows(b):
      start = self._current_batch_index
      return b[start : start + self.batch_size]

    ret = jax.tree_util.tree_map(_get_subrows, self._current_batch)
    self._current_batch_index += self.batch_size
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

  We make the following assumptions on the underlying Lingvo input self.input:
  * it returns the entire data in a deterministic manner;
  * it is finite (does not repeat).

  To avoid dropping remainder issues and to guarantee the eval dataset is
  complete, we recommend that the underlying Lingvo input self.input uses a
  batch
  size of 1. (The actual batch size used on this input is specified on
  `self.batch_size`.)

  Example usage:
      p = pax_fiddle.Config(LingvoEvalAdaptor, ...)
      self.input.bucket_batch_limit = [1]
      self.batch_size = 4
  """
  _VALIDATE_BATCH_SIZE_NOT_NONE = True
  _VALIDATE_BATCH_SIZE_NONE = False
  _num_samples: Any = dataclasses.field(init=False, repr=False)
  _dataset: Any = dataclasses.field(init=False, repr=False)
  _iter: Any = dataclasses.field(init=False, repr=False)

  def __post_init__(self):
    super().__post_init__()
    if self.is_training:
      raise ValueError('LingvoEvalAdaptor requires self.is_traing=False.')
    if not self.reset_for_eval:
      raise ValueError('LingvoEvalAdaptor requires self.reset_for_eval=True.')
    self._num_samples = None
    self._dataset = self._get_dataset()
    self._iter = self._dataset.as_numpy_iterator()

  @classmethod
  def get_batch_size(
      cls,
      params: pax_fiddle.Config[LingvoInputAdaptor] | LingvoInputAdaptor,
  ) -> int:
    assert params.batch_size is not None
    return params.batch_size

  def _update_file_random_seed(self) -> None:
    """Updates file random seed.

    This overrides LingvoInputAdaptor._update_file_random_seed where each host
    is assigned a different file random seed. It does nothing to make sure every
    host uses the same file random seed in params.input.
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
    eval_set_size = len(list(ds.as_numpy_iterator()))
    self._num_samples = eval_set_size
    logging.info(
        'LingvoEvalAdaptor self.name=%s contains %d examples before padding.',
        self.name,
        eval_set_size,
    )
    global_batch_size = self.batch_size * self.num_infeed_hosts
    if eval_set_size % global_batch_size == 0:
      return ds
    total_size = (eval_set_size // global_batch_size + 1) * global_batch_size
    logging.info(
        (
            'LingvoEvalAdaptor self.name=%s is expanded to contain %d '
            'examples globally after padding.'
        ),
        self.name,
        total_size,
    )
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
    ds = ds.batch(self.batch_size, drop_remainder=False)
    return ds.shard(
        num_shards=self.num_infeed_hosts, index=self.infeed_host_index
    )

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

  We make the following assumptions on the underlying Lingvo input self.input:
  * it returns the entire data in a deterministic manner;
  * it knows the number of samples in self.input.num_samples;
  * it has a feature named `eval_sample_weights` to represent the validity of
      each sample, 1 for valid and 0 for invalid;
  * it repeats after all samples are exhausted (this requirement can be easily
      lifted in the future if necessary).

  The batch_size is handled by the underlying Lingvo input:
  self.input.batch_size.

  Example usage:
      input_p.batch_size = 4
      p = pax_fiddle.Config(LingvoLazyEvalAdaptor, input_p)
  """
  _VALIDATE_BATCH_SIZE_NOT_NONE = False
  _VALIDATE_BATCH_SIZE_NONE = True
  batch_size: Any = dataclasses.field(init=False, repr=False)
  _num_samples: Any = dataclasses.field(init=False, repr=False)
  computed_num_batches: Any = dataclasses.field(init=False, repr=False)
  _num_examples_emitted: Any = dataclasses.field(init=False, repr=False)
  _num_batches_emitted: Any = dataclasses.field(init=False, repr=False)

  def __post_init__(self):
    super().__post_init__()
    assert self.input is not None
    if self.is_training:
      raise ValueError('LingvoLazyEvalAdaptor requires self.is_traing=False.')
    if not self.reset_for_eval:
      raise ValueError(
          'LingvoLazyEvalAdaptor requires self.reset_for_eval=True.'
      )
    if self.infeed_host_index >= self.num_infeed_hosts:
      raise ValueError('Must have infeed_host_index < num_infeed_hosts')
    if not isinstance(self.input.batch_size, int) or self.input.batch_size <= 0:
      raise ValueError(
          'Must have positive batch_size in the underlying input: '
          f'get {self.input.batch_size} instead.'
      )
    self.batch_size = self.get_batch_size(self)
    # Global batch size across all hosts
    global_batch_size = self.num_infeed_hosts * self.batch_size
    # Global number of samples across all hosts
    global_num_samples = self.input.num_samples
    # Number of batches each host should at least have
    num_batches = self.input.num_samples // global_batch_size
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
    if self.infeed_host_index < num_full_batches_remainder:
      # These hosts need to handle a full batch for the remaining samples.
      num_samples += self.batch_size
    elif self.infeed_host_index == num_full_batches_remainder:
      # This host needs to handle a partial (possibly empty) batch for the
      # remaining samples.
      num_samples += num_samples_remainder
    self._num_samples = num_samples
    self.computed_num_batches = num_batches
    if self.num_batches is not None:
      if self.num_batches <= 0:
        logging.warning(
            '`num_batches` is non-positive (i.e. %d) so ignored',
            self.num_batches,
        )
      elif self.num_batches > num_batches:
        logging.warning(
            '`num_batches` is greater than dataset capacity (%d>%d) so ignored',
            self.num_batches,
            num_batches,
        )
      else:
        self.computed_num_batches = self.num_batches
        logging.warning(
            '`num_batches` overridden to %d as requested by params',
            self.num_batches,
        )
    self.reset()

  def _update_file_random_seed(self) -> None:
    """Updates file random seed.

    This overrides LingvoInputAdaptor._update_file_random_seed where each host
    is assigned a different file random seed. It does nothing to make sure every
    host uses the same file random seed in params.input.
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
    if self._num_batches_emitted == self.computed_num_batches:
      raise tf.errors.OutOfRangeError(
          node_def=None,
          op=None,
          message=f'{self.computed_num_batches} batches have been exhausted.')
    # Number of remaining samples this host has.
    remaining_samples = self._num_samples - self._num_examples_emitted
    ret = super().get_next()
    self._num_batches_emitted += 1
    # Number of valid samples this batch has.
    num_valid_samples = min(self.batch_size, remaining_samples)
    if 'eval_sample_weights' not in ret:
      raise ValueError('eval_sample_weights must be included in the data')
    # Sets the weight of invalid samples to 0.
    ret.eval_sample_weights[num_valid_samples:] = 0.  # pytype: disable=attribute-error  # jax-ndarray
    self._num_examples_emitted += num_valid_samples
    remaining_samples -= num_valid_samples
    # If there are still remaining samples in this host, skip n-1 batches which
    # belong to other hosts.
    if remaining_samples > 0:
      for _ in range(self.num_infeed_hosts - 1):
        self._get_next_fn()
    return ret

  def reset(self):
    super().reset()
    # Skips k batches which belong to other hosts.
    for _ in range(self.infeed_host_index):
      self._get_next_fn()
    self._num_examples_emitted = 0
    self._num_batches_emitted = 0


class MultiInput(BaseInput):
  """Wraps children inputs and outputs a combined batch at each step.

  During Pax's train, on each host input instances for all children
  inputs will be created (instantiate(input_p)), and then get_next() is
  iteratively called for each child in eager mode to generate one batch of
  data for each step. Each batch will contain a batch from all children
  input generators nested into a NestedMap.

  Since Pax trainers with model sharding assume a global batch size for all
  input tensors, we reshape all tensors across different inputs to
  [global_batch_size, inner_input_batch_size, ...]. global_batch_size is
  automatically determined from children input generator batch sizes.

  The model code is responsible for collapsing the two batch_size dimensions.

  Attributes:
    input_to_params: Dict from input names to input generator parameter
      definitions for each input. Input generators need to implement BaseInput.
      Required.
    default_input: Default input to use for ids_to_strings or other input
      generator methods.
  """

  _VALIDATE_BATCH_SIZE_NOT_NONE = False  # Validated separately for children.
  _VALIDATE_BATCH_SIZE_NONE = True  # Can't set batch size for wrapper.

  input_to_params: dict[str, pax_fiddle.Config[BaseInput]] | None = None
  default_input: str | None = None
  _inputs: Any = dataclasses.field(init=False, repr=False)

  def __post_init__(self):
    if self._VALIDATE_BATCH_SIZE_NONE and self.batch_size is not None:
      raise ValueError(
          'MultiInput does not support self.batch_size. '
          'Please specify batch size on each child input '
          'separately.'
      )
    if not self.name:
      self.name = 'train' if self.is_training else 'input'
    super().__post_init__()
    name = self.name
    if re.fullmatch(_NAME_REGEX, name) is None:
      raise ValueError(
          f'Input params self.name string invalid: "{name}" '
          f'does not fully match "{_NAME_REGEX}".'
      )
    if self.input_to_params is None:
      raise ValueError('Need to define inputs.')

    if self.reset_for_eval and len(self.input_to_params) > 1:
      raise ValueError(
          'Only 1 input can be specified when using reset_for_eval.'
      )

    self._inputs = {}
    for input_name, input_params in self.input_to_params.items():
      # Overriding params for children to match parent.
      input_params.num_infeed_hosts = self.num_infeed_hosts
      input_params.infeed_host_index = self.infeed_host_index
      input_params.is_training = self.is_training
      input_params.reset_for_eval = self.reset_for_eval
      input_params.eval_loop_num_batches = self.eval_loop_num_batches
      input_params.name = self.name + '_' + input_name

      self._inputs[input_name] = instantiate(input_params)

  @classmethod
  def get_batch_size(
      cls, params: pax_fiddle.Config[MultiInput] | MultiInput
  ) -> int:
    assert params.input_to_params
    logging.warning(
        'get_batch_size for MultiInput only returns the outer batch size '
        'determined from the children input generators. This will be different '
        'from the actual batch sizes for children inputs.'
    )
    children_batch_sizes = []
    for child_ig_params in params.input_to_params.values():
      children_batch_sizes.append(
          child_ig_params.cls.get_batch_size(child_ig_params)
      )
    if len(children_batch_sizes) == 1:
      return children_batch_sizes[0]
    return math.gcd(*children_batch_sizes)

  def get_next(self) -> NestedJTensor:
    input_batches = {}
    for input_name, input_gen in self._inputs.items():
      input_batches[input_name] = input_gen.get_next()
    combined_batch = NestedMap(input_batches)
    outer_batch_size = self.get_batch_size(self)
    return combined_batch.Transform(
        lambda x: py_utils.reshape_with_outer_batch_size(x, outer_batch_size))

  def get_child(self, input_name: str) -> NestedJTensor:
    return self._inputs[input_name]

  def reset(self) -> None:
    for _, input_gen in self._inputs.items():
      input_gen.reset()

  def ids_to_strings(
      self,
      ids: pytypes.NpTensor,
      lengths: pytypes.NpTensor,
      key: str | None = None,
      input_name: str | None = None,
  ) -> Sequence[str]:
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
      if self.default_input is None:
        raise ValueError(
            '`input_name` is required when there is no `default_input`')
      input_name = self.default_input
    return self._inputs[input_name].ids_to_strings(ids, lengths, key)


class BaseInputSpecsProvider(
    base_hyperparams.FiddleBaseParameterizable, metaclass=abc.ABCMeta
):
  """Base class to provide input specs for model initialization.

  This helper class is added for shape inference support.
  """

  @abc.abstractmethod
  def get_input_specs(self) -> NestedShapeDtypeStruct:
    """Returns example per-process/device input specs for model initialization.

    Batch size is per-process for pjit, and per-device for pmap.

    In the case of pjit models, it is an *unpadded* batch size. Padding is used
    to ensure each device is fed identically-sized tensors within Pax.
    """


class DatasetInputSpecsProvider(BaseInputSpecsProvider):
  """Class to provide input specs from a dataset for model initialization."""
  input_p: pax_fiddle.Config[BaseInput] | None = None

  def __post_init__(self):
    # Instantiate the input pipeline and get the specs.
    # In practice, we typically use it only once at model
    # initialization time.
    input_pipeline: BaseInput = instantiate(self.input_p)
    dataset = input_pipeline.dataset
    if (
        dataset
        and isinstance(dataset, tf.data.Dataset)
        and
        # Only use dataset.element_spec to compute the input spec when all
        # dimensions are defined.
        jax.tree_util.tree_reduce(
            lambda c, x: c and x.shape.is_fully_defined(),
            dataset.element_spec,
            True,  # Initial value.
        )
    ):
      def tf_spec_to_jax(spec: tf.TensorSpec) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct(shape=spec.shape,
                                    dtype=spec.dtype.as_numpy_dtype())

      self._input_specs = jax.tree_map(tf_spec_to_jax, dataset.element_spec)
      return

    logging.warning(
        'b/292156360: The input specs is generated based on the first data '
        'batch. It is recommended to define an explicit input spec provider '
        'param in BaseExperiment.get_input_specs_provider_params(), which is '
        'more deterministic and efficient.'
    )
    self._input_specs = jax.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
        input_pipeline.get_next_padded(),
    )

  def get_input_specs(self) -> NestedShapeDtypeStruct:
    """Returns example input specs from the input pipeline for model init."""
    return self._input_specs
