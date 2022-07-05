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

"""Base class for the Praxis layers."""

from __future__ import annotations

import copy
import dataclasses
import enum
import functools
import itertools
import math
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, TypeVar

from absl import flags
from absl import logging
from flax import core as flax_core
from flax import linen as nn
from flax import struct
import jax
from jax import numpy as jnp
from jax import random as jrandom
from jax.experimental import pjit
import numpy as np
from praxis import asserts
from praxis import base_hyperparams
from praxis import py_utils
from praxis import pytypes
import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS

NestedMap = py_utils.NestedMap

ParamsT = pytypes.HParamsT
BaseLayerS = TypeVar('BaseLayerS', bound='BaseLayer')
BaseLayerT = TypeVar('BaseLayerT', bound='BaseLayer')
JTensor = pytypes.JTensor
PRNGKey = pytypes.PRNGKey
JTensorOrPartitionSpec = pytypes.JTensorOrPartitionSpec
NpTensor = pytypes.NpTensor
SummaryDict = pytypes.SummaryDict

BaseHyperParams = base_hyperparams.BaseHyperParams
BaseParameterizable = base_hyperparams.BaseParameterizable
InstantiableHyperParams = base_hyperparams.InstantiableHyperParams
sub_config_field = base_hyperparams.sub_config_field

NestedJTensor = pytypes.NestedJTensor
NestedBool = pytypes.NestedBool
NestedHParams = pytypes.NestedHParams
NestedJTensorOrPartitionSpec = pytypes.NestedJTensorOrPartitionSpec

SplitDimsMapping = pytypes.SplitDimsMapping

# Layer stack to establish parent child relationships.
_LAYER_STACK = py_utils.ThreadLocalStack()

# Global state that may impact how certain jax computation will be carried (e.g.
# whether or not to enable dropout).
_JaxContextStack = py_utils.ThreadLocalStack()

# A few special Flax Linen variable collection names.
PARAMS = 'params'
AUX_LOSS = 'aux_loss'
SUMMARIES = 'summaries'
NON_TRAINABLE = 'non_trainable'
DECODE_CACHE = 'decoder_cache'
PREFIX_DECODE_CACHE = 'prefix_decoder_cache'
# hyper-params used to construct a layer.
HYPER_PARAMS = 'hyper_params'

# Used for interoperability with Flax-based libraries and
# not for use within Pax' own layers.
# It will be handled as NON_TRAINABLE in train mode.
NON_PAX_VAR_COLLECTION = ['batch_stats']

# A few special Flax RNG stream names.
RANDOM = 'random'
NON_PAX_RNG_KEY = 'dropout'

# Public aliase of base_hyperparams.instantiate() for convenience.
instantiate = base_hyperparams.instantiate

# A function that transforms a decode state variable of a layer. It has three
# inputs: (state tensor, batch dim, time dim). It returns the transformed state.
# The transformation is done on the batch and/or time dimension; if a required
# dimension is missing (represented as -1), the function should not change the
# state.
DecodeStateTransformFn = Callable[[JTensor, int, int], JTensor]

# The axis name that is pmmaped over.
PMAP_PARALLEL_AXIS_NAME = 'batch'


# Whether caller is running under pmap(..., axis_name=PMAP_PARALLEL_AXIS_NAME).
# Note that if the pmap doesn't specify `axis_name=PMAP_PARALLEL_AXIS_NAME`,
# this function returns False.
# For example,
#   def f(x):
#     return is_running_under_pmap()
#
#   jax.pmap(f)(jnp.ones((1,)))
#     = ShardedDeviceArray([ False], dtype=bool)
#   jax.pmap(f, axis_name=PMAP_PARALLEL_AXIS_NAME)(jnp.ones((1,)))
#     = ShardedDeviceArray([ True], dtype=bool)
def is_running_under_pmap() -> bool:
  """Whether currently running under pmap with PMAP_PARALLEL_AXIS_NAME."""
  try:
    _ = jax.lax.axis_index(PMAP_PARALLEL_AXIS_NAME)
    return True
  except NameError:
    return False


class WeightHParamsCollection:
  """Weight hparams collection annotation.

  Weight hparams collections annotates variables with special properties,
  e.g. whether or not the variable is leanable, whether or not the variable is
  subject to lp regularization.
  """
  SKIP_LP_REGULARIZATION = '__lingvo_jax_skip_regularization'
  NON_TRAINABLE = '_lingvo_jax_non_trainable'
  REQUIRES_MEAN_SYNC = '_requires_mean_sync'


def var_not_trainable(var_hparams: ParamsT) -> bool:
  """Returns True if var_hparams is not a trainable variable."""
  return WeightHParamsCollection.NON_TRAINABLE in var_hparams.collections


def var_requires_mean_sync(var_hparams: ParamsT) -> bool:
  """Returns True if var_hparams requires synchronization across replicas."""
  return WeightHParamsCollection.REQUIRES_MEAN_SYNC in var_hparams.collections


def var_skip_lp_regularization(var_params: ParamsT) -> bool:
  return WeightHParamsCollection.SKIP_LP_REGULARIZATION in var_params.collections


def to_partition_spec(split_dims_mapping: SplitDimsMapping,
                      mesh_axis_names: Sequence[str]) -> pjit.PartitionSpec:
  """Converts split_dims_mapping to pjit.PartitionSpec.

  Args:
    split_dims_mapping: A (nested) tuple of mesh axis to split x over. Below are
      a few example sharding specifications. (0, 2)  - the first dim of x is
      split over the first axis of the mesh and the second dim over the third
      axis of the mesh. (1, -1) - the first dim of x is split over the second
      axis of the mesh and the second dim is replicated. (1, None) - the first
      dim is split over the second axis of the mesh, and the second dim
      replicated. ('data', 'mdl') - the first dim is split over the 'data' axis
      of the mesh and the second dim over the 'mdl' axis. (('replica', 'data'),
      'mdl'), the first dim is split over both the 'replica' and 'data' axes,
      while the second dim over the 'mdl' axis.
    mesh_axis_names: A tuple/list of strings of the name of the device mesh.

  Returns:
    A pjit.PartitionSpec.
  """

  def _parse_split_dims(dims_mapping):
    split_dims = []

    for s_i in dims_mapping:
      if isinstance(s_i, int):
        if s_i < 0:
          split_dims.append(None)
        else:
          assert s_i < len(mesh_axis_names), (
              f's_i: {s_i}, mesh_axis_names: {mesh_axis_names}')
          split_dims.append(mesh_axis_names[s_i])
      elif isinstance(s_i, str):
        assert s_i in mesh_axis_names
        split_dims.append(s_i)
      elif isinstance(s_i, (tuple, list)):
        split_dims.append(_parse_split_dims(s_i))
      else:
        assert s_i is None
        split_dims.append(None)

    return tuple(split_dims)

  partition_spec = _parse_split_dims(split_dims_mapping)
  return pjit.PartitionSpec(*partition_spec)


def var_partition_specs(
    var_specs: NestedHParams, mesh_shape: Sequence[int],
    device_axis_names: List[str]) -> NestedJTensorOrPartitionSpec:
  """Given variable specs (WeightHParams), returns pjit partition specs.

  Args:
    var_specs: A nested structure of WeightHParams.
    mesh_shape: Shape of logical mesh.
    device_axis_names: Axis name for each mesh axis.

  Returns:
    A nested structure of PartitionSpec.
  """

  assert len(device_axis_names) == len(mesh_shape)

  def _get_spec(var_p):
    v_shape = var_p.shape
    # v_split_dim_mapping may contain a mixture of -1, integers, str, or None.
    # -1 and None both indicates that the corresponding dim is not partitioned.
    v_split_dim_mapping = var_p.tensor_split_dims_mapping
    if v_split_dim_mapping is not None:
      assert len(v_split_dim_mapping) == len(v_shape)
    else:
      v_split_dim_mapping = [-1] * len(v_shape)

    if var_p.repeat_prefix is not None:
      repeat_prefix = var_p.repeat_prefix
      if var_p.repeat_prefix_split_dims_mapping is not None:
        prefix_split_dims_mapping = var_p.repeat_prefix_split_dims_mapping
        assert len(prefix_split_dims_mapping) == len(repeat_prefix)
      else:
        prefix_split_dims_mapping = [-1] * len(repeat_prefix)
      # Append sharding annotations for the prefix part.
      v_split_dim_mapping = (
          list(prefix_split_dims_mapping) + list(v_split_dim_mapping))

    return to_partition_spec(v_split_dim_mapping, device_axis_names)

  return jax.tree_map(_get_spec, var_specs)


def maybe_shard(x: JTensor,
                split_dims_mapping: SplitDimsMapping = None,
                mesh_axis_names: Optional[Sequence[str]] = None,
                unconstrained_dims: Optional[Sequence[int]] = None) -> JTensor:
  """Adds explicit xla sharding constraints.

  This is a wrap around jax.with_sharding_constraint to allow for adding
  explicit sharding annotations to an intermediate node in a jax function.

  No sharding annotation is added if either split_dims_mapping is None or
  mesh_axis_names is None.

  Args:
    x: the input tensor to be sharded.
    split_dims_mapping: A (nested) tuple of mesh axis to split x over. Below are
      a few example sharding specifications. (0, 2) - in this case, the first
      dim of x is split over the first axis of the mesh and the second dim over
      the third axis of the mesh. (1, -1) - in this case, the first dim of x is
      split over the second axis of the mesh and the second dim is replicated.
      (1, None) - First dim is split over the second dim of the mesh, and the
      second dim replicated. ('data', 'mdl') - in this case,  the first dim is
      split over the 'data' axis of the mesh and the second dim over the 'mdl'
      axis. (('replica', 'data'), 'mdl'), in this case the first dim is split
      over both the 'replica' and 'data' axes, while the second dim over the
      'mdl' axis.
    mesh_axis_names: A tuple/list of strings of the name of the device mesh.
    unconstrained_dims: A tuple/list of dimensions for which the sharding will
      be determined by XLA (sharding propagation). We allow this only for this
      internal annotation function, not for the program inputs/outputs.

  Returns:
    An annotated JTensor.
  """
  if split_dims_mapping is None or mesh_axis_names is None:
    return x

  assert len(x.shape) == len(split_dims_mapping), (
      f'Invalid split_dims_mapping. Expected len(split_dims_mapping) '
      f'is {len(x.shape)}, while it is {len(split_dims_mapping)}. '
      f'x.shape = {x.shape} and split_dims_mapping = {split_dims_mapping}')
  partition_spec = to_partition_spec(split_dims_mapping, mesh_axis_names)

  if unconstrained_dims is not None:
    partition_spec_list = list(partition_spec)
    for dim in unconstrained_dims:
      partition_spec_list[dim] = partition_spec.UNCONSTRAINED
    partition_spec = pjit.PartitionSpec(*partition_spec_list)

  return py_utils.with_sharding_constraint(x, partition_spec)


class WeightInit(BaseHyperParams):
  """Static class providing weight initialization config params.

  Attributes:
    method: Initialization method.
    scale: Initialization scale.
  """
  method: str
  scale: float

  @staticmethod
  def Gaussian(scale=1.0):
    """scale * jax.random.normal(0, 1.0)."""
    return WeightInit('gaussian', scale)

  @staticmethod
  def Uniform(scale=1.0):
    """scale * jax.random.uniform(-1.0, 1.0)."""
    return WeightInit('uniform', scale)

  @staticmethod
  def Xavier(scale=1.0):
    """Xavier initialization (x = sqrt(6. / (in + out)); [-x, x])."""
    return WeightInit('xavier', scale)

  @staticmethod
  def XavierWithFixupParams(scale=1.0,
                            depth=1.0,
                            layers_per_residual_block=1.0):
    """Xavier initialization with Fixup."""
    scale = scale * math.pow(depth, (-1.0 / (2 * layers_per_residual_block)))
    return WeightInit('xavier', scale)

  @staticmethod
  def Constant(scale=1.0):
    """scale."""
    return WeightInit('constant', scale)

  @staticmethod
  def TruncatedGaussian(scale=1.0):
    """scale * jax.random.truncated_normal(-2.0, 2.0)."""
    return WeightInit('truncated_gaussian', scale)

  @staticmethod
  def GaussianSqrtDim(scale=1.0):
    """scale * jax.random.normal(0, 1 / sqrt(dim0))."""
    return WeightInit('gaussian_sqrt_dim', scale)

  @staticmethod
  def GaussianSqrtFanIn(scale=1.0):
    """scale * jax.random.normal(0, 1 / sqrt(fan_in))."""
    return WeightInit('gaussian_sqrt_fanin', scale)

  @staticmethod
  def GaussianSqrtFanOut(scale=1.0):
    """scale * jax.random.normal(0, 1 / sqrt(fan_out))."""
    return WeightInit('gaussian_sqrt_fanout', scale)

  @staticmethod
  def GaussianSqrtFanAvg(scale=1.0):
    """jax.random.normal(0, sqrt(2.0 / (in + out)))."""
    return WeightInit('gaussian_sqrt_fanavg', scale)

  @staticmethod
  def UniformSqrtDim(scale=1.0):
    """scale * jax.random.uniform(-1 / sqrt(dim0), 1 / sqrt(dim0))."""
    return WeightInit('uniform_sqrt_dim', scale)

  @staticmethod
  def UniformUnitScaling(scale=1.0):
    """scale * sqrt(3) / sqrt(dim0) * jax.random.uniform(-1, 1)."""
    return WeightInit('uniform_unit_scaling', scale)

  @staticmethod
  def TruncatedGaussianSqrtDim(scale=1.0):
    """scale * jax.random.truncated_normal(0, 1 / sqrt(dim0))."""
    return WeightInit('truncated_gaussian_sqrt_dim', scale)

  @staticmethod
  def TruncatedGaussianSqrtFanIn(scale=1.0):
    """scale * jax.random.truncated_normal(0, 1 / sqrt(fan_in))."""
    return WeightInit('truncated_gaussian_sqrt_fanin', scale)

  @staticmethod
  def TruncatedGaussianSqrtFanOut(scale=1.0):
    """scale * jax.random.truncated_normal(0, 1 / sqrt(fan_out))."""
    return WeightInit('truncated_gaussian_sqrt_fanout', scale)

  @staticmethod
  def ScaledDeltaOrthogonal(scale=1.0):
    return WeightInit('delta_orthogonal', scale)


_DEFAULT_XAVIER_INIT = 1.000001


def default_param_init():
  # Here we use 1.000001 as a signature for user picking up the
  # default param initializer.
  return WeightInit.Xavier(_DEFAULT_XAVIER_INIT)


def is_default_param_init(p):
  return p.method == 'xavier' and abs(p.scale - _DEFAULT_XAVIER_INIT) < 1e-7


class WeightHParams(BaseHyperParams):
  """Hyperparams for a weight variable specifying shape/init/dtype etc.

  Attributes:
    shape: The weight shape.
    init: The initialization method.
    dtype: The weight data type.
    collections: Variable collections this weight belongs to.
    mesh_shape: Shape of logical mesh. mesh_shape and tensor_split_dims_mapping
      below together specifies how this weight tensor should be sharded across
      different tpu cores. If None, this variable is not sharded. Here are
      examples of mesh shape: [2, 3, 4] for 2-way replica parallelism, 3-way
      data parallelism and 4-way model parallelism.
    tensor_split_dims_mapping: A list of integers that map each tensor axis to
      the device mesh axis along which it is sharded. Its length is the tensor
      rank, and split_dims_mapping[i] is device mesh axis for tensor dimension
      i. Use -1 for tensor dimensions that are not sharded. If the list is set
      to None and a mesh_shape is specified, the sharding will be treated as
      replicated. Here is a concrete examples: mesh_shape=[2, 4] and shape=[x,
      y, z], so this is a 3d variable. tensor_split_dims_mapping=[-1, -1, 1], in
      this case, the third dim of the variable is split along the second dim of
      the mesh. Each split of the variable is of the shape [x, y, z/4].
    repeat_prefix: If not None, the full shape of this var is
      repeat_prefix+shape. For example, if repeat_prefix=[16, 2], and
      shape=[512, 1024], then real shape of variable is [16, 2, 512, 1024].
      "repeat_prefix" is often used if a layer is to be used in a recurrent
      loop, where logically there are n sub-layers, but for performance/hbm
      usage reasons we stack all the variables in creating those n-layers.
    repeat_prefix_split_dims_mapping: Tensor split dims mapping for the
      repeat_prefix dims.
  """
  shape: Sequence[int]
  init: Optional[WeightInit] = None
  dtype: Optional[jnp.dtype] = None
  collections: Optional[Sequence[str]] = None
  mesh_shape: Optional[Sequence[int]] = None
  tensor_split_dims_mapping: SplitDimsMapping = None
  repeat_prefix: Optional[Sequence[int]] = None
  repeat_prefix_split_dims_mapping: SplitDimsMapping = None

  # If any kwargs are None, they are given defaults from BaseLayer.hparams
  # in self.create_variable.
  def __post_init__(self):
    if self.collections is None:
      self.collections = []
    if self.mesh_shape is not None:
      if self.tensor_split_dims_mapping is None:
        self.tensor_split_dims_mapping = (-1,) * len(self.shape)
        logging.info(
            'Sets tensor_split_dims_mapping of a param of shape %s to %s',
            self.shape, self.tensor_split_dims_mapping)
      assert len(self.tensor_split_dims_mapping) == len(self.shape)


def get_fan_in_fan_out(
    shape: Sequence[int]) -> Tuple[Optional[int], Optional[int]]:
  """Returns (fan_in, fan_out) of a weight variable of the given shape."""
  if not shape:
    return None, None
  if len(shape) < 1:
    return 1, 1
  elif len(shape) == 1:
    # Following _compute_fans() from TF's init_ops.py.
    return shape[0], shape[0]
  else:
    receptive_field_size = 1
    for s in shape[:-2]:
      receptive_field_size *= s
    fan_in = shape[-2] * receptive_field_size
    fan_out = shape[-1] * receptive_field_size
    return fan_in, fan_out


def scaled_orthogonal(key: JTensor,
                      shape: Sequence[int],
                      dtype: jnp.dtype = jnp.float32):
  """Scaled orthogonal initialization."""
  scale = max(np.sqrt(float(shape[-2]) / shape[-1]), 1)
  ortho_init = jax.nn.initializers.orthogonal(
      scale=scale, column_axis=-1, dtype=dtype)
  return ortho_init(key, shape)


def scaled_delta_orthogonal(key: JTensor,
                            shape: Sequence[int],
                            dtype: jnp.dtype = jnp.float32):
  """Delta orthogonal kernels; see arXiv:1806.05393 / arxiv:2110.01765."""
  if len(shape) not in [3, 4, 5]:
    raise ValueError(
        'Delta orthogonal initializer requires a 3D, 4D or 5D shape.')
  ortho_matrix = scaled_orthogonal(key, shape[-2:], dtype=dtype)
  w = jnp.zeros(shape, dtype=dtype)
  if len(shape) == 3:
    k = shape[0]
    return w.at[(k - 1) // 2, ...].set(ortho_matrix)
  elif len(shape) == 4:
    k1, k2 = shape[:2]
    return w.at[(k1 - 1) // 2, (k2 - 1) // 2, ...].set(ortho_matrix)
  else:
    k1, k2, k3 = shape[:3]
    return w.at[(k1 - 1) // 2, (k2 - 1) // 2, (k3 - 1) // 2,
                ...].set(ortho_matrix)


# Caller ensures that `prng_key` is different for different init_var calls.
def init_var(var_full_name: str, var_p: WeightHParams,
             prng_key: PRNGKey) -> JTensor:
  """Creates an initial value of a var."""
  method = var_p.init.method
  scale = var_p.init.scale
  assert isinstance(scale, (int, float))
  shape = var_p.shape
  init_dtype = var_p.dtype
  logging.info(
      'Creating var %s with shape=%s, dtype=%s, init method=%s and scale=%s',
      var_full_name, shape, init_dtype.dtype, var_p.init.method,
      var_p.init.scale)
  # We rely on nn.scan to transform vars, hence init_var shouldn't expect a
  # repeat_prefix or repeat_prefix_split_dims_mapping.
  assert not var_p.repeat_prefix
  assert not var_p.repeat_prefix_split_dims_mapping

  if shape:
    assert all([dim_size > 0 for dim_size in shape
               ]), 'shape of %s is %s' % (var_full_name, shape)
    dim0 = shape[0]
  else:
    dim0 = 1

  if is_default_param_init(var_p.init):
    logging.debug(
        'WARNING!!! var %s is using the default xavier initializer.'
        ' Make sure this is intended.', var_full_name)

  if method in ['delta_orthogonal']:
    if len(shape) < 2:
      logging.warning('WARNING!! Delta orthogonal applied to 0/1D vars.')

  if (method in [
      'gaussian_sqrt_dim', 'uniform_sqrt_dim', 'truncated_gaussian_sqrt_dim'
  ]):
    if len(shape) > 2:
      # This is probably not the right method to use when len(shape) > 2,
      # e.g. dim0 will be 3 with a 3x3 conv2d kernel.
      logging.warning(
          'Initializing %s of shape %s with method %s: dim0=%s. '
          'Make sure that it is intended.', var_full_name, shape, method, dim0)
    scale *= 1.0 / math.sqrt(dim0)
  if method in ['gaussian_sqrt_fanin', 'truncated_gaussian_sqrt_fanin']:
    fan_in, _ = get_fan_in_fan_out(shape)
    if fan_in is not None:
      scale *= 1.0 / math.sqrt(fan_in)
  if method in ['gaussian_sqrt_fanout', 'truncated_gaussian_sqrt_fanout']:
    _, fan_out = get_fan_in_fan_out(shape)
    if fan_out is not None:
      scale *= 1.0 / math.sqrt(fan_out)
  if method in ['gaussian_sqrt_fanavg']:
    fan_in, fan_out = get_fan_in_fan_out(shape)
    if fan_in is not None and fan_out is not None:
      scale *= math.sqrt(2.0 / (fan_in + fan_out))

  if method in ['delta_orthogonal']:
    if len(shape) <= 2:
      return scale * jrandom.normal(prng_key, shape, init_dtype)
    elif len(shape) == 2:
      return scaled_orthogonal(prng_key, shape, init_dtype)
    else:
      return scaled_delta_orthogonal(prng_key, shape, init_dtype)
  if method in [
      'gaussian', 'gaussian_sqrt_dim', 'gaussian_sqrt_fanin',
      'gaussian_sqrt_fanout', 'gaussian_sqrt_fanavg'
  ]:
    return scale * jrandom.normal(prng_key, shape, init_dtype)
  elif method in ['uniform', 'uniform_sqrt_dim']:
    return scale * jrandom.uniform(
        prng_key, shape, init_dtype, minval=-1.0, maxval=1.0)
  elif method in [
      'truncated_gaussian', 'truncated_gaussian_sqrt_dim',
      'truncated_gaussian_sqrt_fanin', 'truncated_gaussian_sqrt_fanout'
  ]:
    return scale * jrandom.truncated_normal(
        prng_key, lower=-2.0, upper=2.0, shape=shape, dtype=init_dtype)
  elif method in ['constant']:
    return scale + jnp.zeros(shape=shape, dtype=init_dtype)
  elif method in ['xavier']:
    fan_in, fan_out = get_fan_in_fan_out(shape)
    limit = scale * math.sqrt(6. / (fan_in + fan_out))
    return limit * jrandom.uniform(
        prng_key, shape, init_dtype, minval=-1.0, maxval=1.0)
  elif method in ['uniform_unit_scaling']:
    input_size = 1.0
    for dim in shape[:-1]:
      input_size *= float(dim)
    # Avoid errors when initializing zero-size tensors.
    input_size = max(input_size, 1.0)
    max_val = math.sqrt(3 / input_size) * scale
    return max_val * jrandom.uniform(
        prng_key, shape, init_dtype, minval=-1.0, maxval=1.0)
  else:
    assert False, 'init_type %s not supported.' % method


# BoxedParam allows us to treat the actual variable jnp.array and its associated
# metadata, i.e. WeightParam as a single Flax variable collection.
@struct.dataclass
class BoxedParam:
  # `value` is the jnp.array of the variable.
  value: Any
  # `meta` is the WeightParam declared for the variable.
  # We do not want to transform the variable weight param so we mark the field
  # pytree_node=False to prevent JAX transforms from touching it.
  meta: Any = struct.field(pytree_node=False)


@struct.dataclass
class WrappedHParams:
  # We do not want to transform hyper-params so we mark the field
  # pytree_node=False to prevent JAX transforms from touching it.
  meta: BaseHyperParams = struct.field(pytree_node=False)


@struct.dataclass
class AuxLossStruct:
  value: JTensor
  weight: JTensor


def maybe_unbox_value(tree):
  """Return the `value` leaf component of the pytree if it is a BoxedParam."""
  return jax.tree_map(
      lambda bp: bp.value if isinstance(bp, BoxedParam) else bp,
      tree,
      is_leaf=lambda x: isinstance(x, BoxedParam))


def unbox_meta(tree):
  """Return the `meta` leaf component of the pytree of BoxedParam."""
  return jax.tree_map(
      lambda bp: bp.meta, tree, is_leaf=lambda x: isinstance(x, BoxedParam))


class SummaryType(enum.Enum):
  """Types of summary tensors."""
  SCALAR = 1
  IMAGE = 2
  TEXT = 5

  # Like SCALAR, but this type indicates that this data is suitable for use
  # with sensitive data.
  AGGREGATE_SCALAR = 3

  # Like IMAGE, but this type indicates that the image data was sufficiently
  # aggregated such that this is safe to use with sensitive data.
  AGGREGATE_IMAGE = 4


def get_summary_base_type(summary_type: SummaryType) -> SummaryType:
  if summary_type == SummaryType.AGGREGATE_SCALAR:
    return SummaryType.SCALAR
  if summary_type == SummaryType.AGGREGATE_IMAGE:
    return SummaryType.IMAGE
  return summary_type


def get_summary_type_suffix(summary_type: SummaryType) -> str:
  return '_' + summary_type.name.lower()


def get_summary_type_from_key(key: str) -> SummaryType:
  for t in SummaryType:
    if key.endswith('_' + t.name.lower()):
      return t
  raise ValueError('Cannot parse summary type from key: ' + key)


def trim_summary_type_from_key(key: str) -> str:
  for t in SummaryType:
    suffix = '_' + t.name.lower()
    if key.endswith(suffix):
      return key[:-len(suffix)]
  raise ValueError('Cannot parse summary type from key: ' + key)


class _SummaryDict:
  """A dict holding summaries generated during forward computation.

  Currently it supports 5 types: SCALAR, AGGREGATE_SCALAR, IMAGE,
  AGGREGATE_IMAGE, TEXT. Keys will be appended with a type suffix.
  """

  def __init__(self) -> None:
    self.dict = {}

  def add_summary(self, name: str, tensor: JTensor,
                  summary_type: SummaryType) -> None:
    """Adds named summary to the thread local dict.

    Args:
      name: name of the summary.
      tensor: value of the summary.
      summary_type: type of the summary.
    """
    summary_base_name = name
    summary_suffix = get_summary_type_suffix(summary_type)
    full_name = summary_base_name + summary_suffix
    next_iter = 0
    while full_name in self.dict:
      next_iter += 1
      full_name = summary_base_name + str(next_iter) + summary_suffix
    if summary_type == SummaryType.IMAGE or summary_type == SummaryType.AGGREGATE_IMAGE:
      if tensor.ndim == 3:
        # Add a batch dim.
        tensor = jnp.expand_dims(tensor, 0)
      assert tensor.ndim == 4
    self.dict[full_name] = tensor

  def clear(self) -> None:
    """Clears all summaries."""
    self.dict = {}


class JaxContext:
  """Global context under which jax computations are carried out."""

  class HParams(base_hyperparams.BaseHyperParams):
    """HParams for `JaxContent`.

    Attributes:
      do_eval: Whether to do eval.
      in_unit_test: Whether this is running in a unit test.
    """
    do_eval: Optional[bool] = None
    in_unit_test: Optional[bool] = None

  def __init__(self, hparams: JaxContext.HParams) -> None:
    self._hparams = hparams.clone()
    self._summary_dict = _SummaryDict()

  @property
  def summary_dict(self) -> _SummaryDict:
    return self._summary_dict

  @property
  def hparams(self) -> JaxContext.HParams:
    return self._hparams

  @property
  def do_eval(self) -> bool:
    return self.hparams.do_eval

  def __enter__(self) -> JaxContext:
    _JaxContextStack.stack.append(self)
    return self

  def __exit__(self, type_arg, value_arg, traceback_arg):
    assert _JaxContextStack.stack
    assert _JaxContextStack.stack[-1] is self
    _JaxContextStack.stack.pop()

  @staticmethod
  def top() -> Optional[JaxContext]:
    return _JaxContextStack.stack[-1] if _JaxContextStack.stack else None

  @staticmethod
  def new_context(*,
                  hparams: Optional[JaxContext.HParams] = None) -> JaxContext:
    """Returns a new empty JaxContext.

    Args:
      hparams: if not None, and instance of JaxContext.HParams(). If it is None,
        the newly constructed JaxContext will assume the same params as the
        current context if it is not None, or the default one.

    Returns:
      A new JaxContext.
    """
    if hparams is None:
      current = JaxContext.top()
      if current is None:
        new_hparams = JaxContext.HParams()
      else:
        new_hparams = current.hparams.clone()
    else:
      new_hparams = hparams.clone()
    context = JaxContext(new_hparams)
    return context


def cur_jax_context() -> JaxContext:
  current = JaxContext.top()
  assert current is not None
  return current


def add_global_summary(name: str,
                       tensor: JTensor,
                       summary_type: SummaryType = SummaryType.SCALAR) -> None:
  """Adds a global summary tensor.

  This summary is not associated with any particular layer and is added to the
  global JaxContext collection.

  Args:
    name: name of the summary.
    tensor: value of the summary.
    summary_type: type of the summary. Currently it supports 2 types: SCALAR,
      IMAGE. Keys will be appended with a type suffix. Image tensors must be
      either [batch, height, width, channels] or [height, width, channels].
  """
  context = cur_jax_context()
  global_namespace_name = '/' + name
  if jnp.issubdtype(tensor.dtype, jnp.floating):
    tensor = tensor.astype(jnp.float32)
  context.summary_dict.add_summary(global_namespace_name, tensor, summary_type)


def clear_global_summary() -> None:
  context = cur_jax_context()
  context.summary_dict.clear()


def all_global_summaries() -> SummaryDict:
  context = cur_jax_context()
  return context.summary_dict.dict


class Theta:
  """Dot syntax accession helper to be used inside a descriptor."""

  def __init__(self, module):
    self.module = module

  def __getattr__(self, k):
    self.module._try_setup()
    if not self.module.has_variable('params', k):
      raise ValueError(f'Module {self.module} has no theta.{k} defined.')
    return self.module.get_variable('params', k)

  def __getitem__(self, k):
    self.module._try_setup()
    if not self.module.has_variable('params', k):
      raise ValueError(f'Module {self.module} has no theta[{k}] defined.')
    return self.module.get_variable('params', k)


class ThetaDescriptor:
  """Dot syntax accession descriptor."""

  def __get__(self, obj, objtype=None):
    return Theta(obj)


_BaseLayerRecursionDictKeysToIgnore = [
    'parent', '_theta', '_state', 'scope', '_private_hparams', 'hparams',
    '_private_children'
]


# Inherit from Flax Linen Module.
class BaseLayer(
    BaseParameterizable,
    nn.Module,
    init_params_arg_name='_hparams',
    nonconfigurable_init_arg_names=('name', 'parent')):
  r"""Base class for all the layer object.

  Subclasses are expected to override the following functions:

  HParams(): Returns a configuration HParams for this layer.
  setup(): To setup this instance, which includes create all the sub-layers, as
    well as immediate layer variables.
  fprop(): The main method that carries out ML computation.

  TODO(pax-team): Add more doc-string and example.
  """
  # dataclass takes a single HParams object. This should not change during the
  # lifetime of this layer.
  _hparams: base_hyperparams.InstantiableHyperParams

  # Fetches variables from flax 'params' class via theta "dot" syntax.
  theta = ThetaDescriptor()

  class WeightShardingHParams(BaseHyperParams):
    """Represents how layer's learned parameters are partitioned across a mesh.

    This usually refers to the primary model weight. Sub-layers can define
    additional params for more weights.

    Attributes:
      wt: Sharding annotations for the primary model weight.
    """
    wt: SplitDimsMapping = None

  class ActivationShardingHParams(BaseHyperParams):
    """Represents how intermediate values should be partitioned across a mesh.

    This usually refers to the primary layer output. Sub-layers can define
    additional params for more activations.

    Attributes:
      out: Sharding annotations for the primary layer output.
    """
    out: SplitDimsMapping = None

  class HParams(InstantiableHyperParams):
    """Hyperparameters for this layer.

    Attributes:
      dtype: Default dtype for all variables.
      fprop_dtype: Activations datatype to use.
      params_init: How model weights should be initialized.
      skip_lp_regularization:  If True, all variables in this layer will skip Lp
        regularization. If None/False, only variables explicitly in the
        SKIP_LP_REGULARIZATION collection will skip Lp regularization. Also
        propagated to child layers with default settings (None).
      ici_mesh_shape: Shape of the logical mesh used for SPMD parallelism in
        each slice. The meaning of each mesh axis is defined by mesh_axis_names,
        so these two params must be the same length. If dcn_mesh_shape is
        present, the overall mesh is the product of ici_mesh_shape and
        dcn_mesh_shape. For example, an ici_mesh_shape of [2, 3, 4] with
        mesh_axis_names ['replica', 'data', 'mdl'] indicates 2-way replica
        parallelism, 3-way data parallelism, and 4-way model parallelism over 24
        devices. None, the default, is equivalent to a sequence of ones and
        means that the model is placed on a single device.
      dcn_mesh_shape: Shape of the logical mesh used for SPMD parallelism over
        multiple slices. The overall mesh is the product of ici_mesh_shape and
        dcn_mesh_shape, and the meaning of each mesh axis is defined by
        mesh_axis_names, so these three params must be the same length. For
        example, a dcn_mesh_shape of [2, 2, 1, 1] with mesh_axis_names ['stage',
        'replica', 'data', 'mdl'] indicates 2-way pipeline parallelism and 2-way
        replica parallelism over 4 slices. None, the default, is equivalent to a
        sequence of ones and means that the model is placed on a single slice.
      mesh_axis_names: Names for each mesh axis in ici_mesh_shape and/or
        dcn_mesh_shape. Common mesh axes include 'replica' for replica
        parallelism, 'data' for data parallelism, 'mdl' for model parallelism,
        and 'stage' for pipeline parallelism.
      weight_split_dims_mapping: Relevant only if the mesh shape params above
        are not None. It specifies how weight of this layer or those of the
        sublayers should be sharded over the overall device mesh. This field
        will be dynamically bound to WeightShardingHParams dataclass above.
      activation_split_dims_mapping: Relevant only if the mesh shape params
        above are not None. It specifies how activation of this layer or those
        of the sublayers should be sharded over the overall device mesh. This
        field will be dynamically bound to the ActivationShardingHParams
        dataclass above.
    """
    dtype: jnp.dtype = jnp.float32
    fprop_dtype: Optional[Any] = None
    params_init: WeightInit = dataclasses.field(
        default_factory=default_param_init)
    skip_lp_regularization: Optional[bool] = None
    ici_mesh_shape: Optional[Sequence[int]] = None
    dcn_mesh_shape: Optional[Sequence[int]] = None
    mesh_axis_names: Optional[Sequence[str]] = None
    weight_split_dims_mapping: Optional[BaseHyperParams] = None
    activation_split_dims_mapping: Optional[BaseHyperParams] = None

    @property
    def mesh_shape(self):
      if self.ici_mesh_shape is not None:
        assert len(self.ici_mesh_shape) == len(self.mesh_axis_names)
      if self.dcn_mesh_shape is None:
        return self.ici_mesh_shape
      else:
        assert len(self.ici_mesh_shape) == len(self.dcn_mesh_shape)
        return [i * d for i, d in zip(self.ici_mesh_shape, self.dcn_mesh_shape)]

  @staticmethod
  def copy_base_hparams(from_hparams: BaseLayer.HParams,
                        to_hparams: BaseLayer.HParams) -> BaseLayer.HParams:
    """Copies BaseLayer hparams from `from_hparams` to `to_hparams`."""
    assert issubclass(from_hparams.cls, BaseLayer)
    assert issubclass(to_hparams.cls, BaseLayer)
    # Copy-over the BaseLayer params.
    if to_hparams.dtype == jnp.float32:
      to_hparams.dtype = from_hparams.dtype
    if to_hparams.fprop_dtype is None:
      to_hparams.fprop_dtype = from_hparams.fprop_dtype
    if to_hparams.skip_lp_regularization is None:
      to_hparams.skip_lp_regularization = from_hparams.skip_lp_regularization
    if to_hparams.ici_mesh_shape is None:
      to_hparams.ici_mesh_shape = copy.deepcopy(from_hparams.ici_mesh_shape)
    if to_hparams.dcn_mesh_shape is None:
      to_hparams.dcn_mesh_shape = copy.deepcopy(from_hparams.dcn_mesh_shape)
    if to_hparams.mesh_axis_names is None:
      to_hparams.mesh_axis_names = copy.deepcopy(from_hparams.mesh_axis_names)
    if is_default_param_init(to_hparams.params_init):
      # Copy params_init as well. Both to_hparams.params_init and
      # from_hparams.params_init are hyperparams.HParams.
      # The only exception is when layer.setup override params_init with
      # Params().Set syntax in which case, from_hparams.params_init is a
      # WeightInit, copy.deepcopy(from_hparams.params_init) works in both cases.
      to_hparams.params_init = copy.deepcopy(from_hparams.params_init)
    return to_hparams

  @classmethod
  def __init_subclass__(cls, **kwargs: Any) -> None:
    """Automatically initializes all subclasses as custom dataclasses."""
    super().__init_subclass__(**kwargs)

    # Update the Params to dynamically bind a few fields.
    fields = [
        ('_attribute_overrides', Tuple[str, ...],
         ('cls', 'weight_split_dims_mapping', 'activation_split_dims_mapping')),
        ('cls', Type[Any], cls),
        ('weight_split_dims_mapping', BaseHyperParams,
         cls.WeightShardingHParams()),
        ('activation_split_dims_mapping', BaseHyperParams,
         cls.ActivationShardingHParams()),
    ]
    cls.HParams = dataclasses.make_dataclass(
        'HParams', fields=fields, bases=(cls.HParams,))  # pytype: disable=wrong-arg-types

  def post_init_hparams(self, *args):
    """Recursively populates the HYPER_PARAMS collection with hyper-params ...

    of self and all its children.

    The difference from self.hparams is that params here are post initialization
    tweaks and reflect the actual sub-layers being created.

    Args:
      *args: used for scan's rigid signature requirements.
    """
    hparams = self.hparams.clone()
    for p in dataclasses.fields(hparams):
      p_name = p.name
      p_value = getattr(hparams, p_name)
      if (isinstance(p_value, base_hyperparams.InstantiableHyperParams) and
          issubclass(p_value.cls, BaseLayer)):
        # No need to include sub-layer params, which will show up in its own
        # collection anyways.
        setattr(hparams, p_name, None)
    self.put_variable(HYPER_PARAMS, '_hparams', WrappedHParams(hparams))
    # walk through all the attributes on self and recursively apply
    # post_init_hparams on submodules:
    for key, val in self.__dict__.items():
      if key in _BaseLayerRecursionDictKeysToIgnore:
        continue  # don't create recursion loop!

      def force(v):
        if isinstance(v, BaseLayer):
          # pass dummy args through - again only needed for scan.
          v.post_init_hparams(*args)

      jax.tree_map(force, val)
    return None

  def __post_init__(self):
    assert self._hparams.name, (
        f'{type(self).__name__} HParams must define the layer\'s "name"')
    object.__setattr__(self, 'name', self._hparams.name)
    # We make a copy of the `_hparams` passed to __init__ the very first time in
    # case `_hparams` refers to a shared params object that gets mutated by
    # something outside this class.
    # Note: self.hparams is a property defined on BaseParameterizable that
    # returns self._hparams, which is why we set it like this.
    object.__setattr__(self, '_hparams', self._hparams.clone())
    # Freeze the layer hparams. This is to prevent accidental config mutations
    # that may lead to subtle bugs.
    self._hparams.freeze()
    object.__setattr__(self, '_theta', set())
    object.__setattr__(self, '_private_children', {})
    super().__post_init__()

  # linen.Module `adopt_attr_modules` sets `name` to None and clones the
  # BaseLayer object expecting the `name` attribute to be None still. Yet
  # BaesLayer.__post_init__ always overrides `name` attribute. Here we override
  # `clone` to explicitly retain the original `name` attribute.
  @nn.nowrap
  def clone(self, *, parent=None, **updates):
    oldname = self.name
    retval = super().clone(parent=parent, **updates)
    object.__setattr__(retval, 'name', oldname)
    return retval

  @nn.nowrap
  def _try_setup(self, shallow=False):
    setup_status_before = self._state.setup_called
    super()._try_setup(shallow=shallow)
    setup_status_after = self._state.setup_called
    if setup_status_before != setup_status_after:
      # setup() is being called. Let's perform some sanity checks.
      for k, v in self._state.children.items():
        if isinstance(v, BaseLayer):
          if k not in self._private_children:
            logging.warning(
                'Child %s is not created via self.create_child helper, '
                'possibly a shared layer.', k)
        elif v == 'param':
          assert k in self._theta, (
              f'Learnable param {k} is not created via create_variable helper.')
        else:
          pass

  # Recursively walk through the submodule tree to force lazy module
  # initialization to happen eagerly during the call. It effectively calls
  # `setup` on each submodule. It's important to note that we do not care about
  # the returned value of force_init but rather relies on the side effect of
  # mutating the self.scope object -- for the module tree structure and also
  # the variable collections that get added to self.scope.
  # In `setup`, `self.create_variable` ultimately calls `self.param` and
  # `self.variable`. They put BoxedParams into self.scope.
  # In `setup`, `self.create_child` builds the module tree structure in
  # self.scope.
  def force_init(self, *args):
    """Recursively forces setup() to be called."""
    # Dummy `*args` signature is needed for scan's rigid signature requirements.
    for key, val in self.__dict__.items():
      if key in _BaseLayerRecursionDictKeysToIgnore:
        continue  # don't create recursion loop!

      def force(v):
        if isinstance(v, nn.Module):
          # pass dummy args through - again only needed for scan.
          v.force_init(*args)

      jax.tree_map(force, val)
    return None

  # This top level `init` call does not use the returned value of `force_init`
  # but rather relies on the side effects of self.param/self.variable for
  # populating self.scope with the variable collections. What gets put into
  # self.scope is BoxedParams and thus the need for unboxing for super().init().
  # We intentionally unbox BoxedParams and only return jnp.array values to
  # callers because we expect users to do
  #   initial_vars = layer.init(k)
  #   outputs = layer.apply(initial_vars, method=layer.fprop)
  # where `initial_vars` that users see are unboxed jnp.arrays, and
  # also the code in fprop never sees BoxedParams but always jnp.arrays.
  def init(self, rngs, *args, **kwargs):
    # TODO(zhangqiaorjc): Pass args and kwargs to init when Praxis switches to
    # running forward computation for layer initialization, similar to what
    # Flax does.
    del args, kwargs
    variables = super().init(rngs, method=self.force_init, mutable=True)
    return flax_core.unfreeze(maybe_unbox_value(variables))

  # See comments in `init` above.
  # `variables` are unboxed variables, and not BoxedParams.
  def apply(self,
            variables,
            *args,
            rngs=None,
            method=None,
            mutable=False,
            capture_intermediates=False,
            **kwargs):
    # Default to self.__call__ to make callsite cleaner.
    if method is None:
      method = self.__call__
    result = super().apply(
        variables,
        *args,
        rngs=rngs,
        method=method,
        mutable=mutable,
        capture_intermediates=capture_intermediates,
        **kwargs)
    if mutable:
      outputs, updated_variables = result
      return outputs, flax_core.unfreeze(updated_variables)
    else:
      return result

  # The top level `init_fn` call does not use the returned value of `force_init`
  # but rather relies on the side effects of self.param/self.variable for
  # populating self.scope with the variable collections. What gets put into
  # self.scope is BoxedParams and thus init_fn returns the variable collections
  # with BoxedParams. We unboxed it and return WeightHParams metadata.
  def abstract_init_with_metadata(self, rngs, *args, **kwargs):
    # TODO(zhangqiaorjc): Pass args and kwargs to init when Praxis switches to
    # running forward computation for layer initialization, similar to what
    # Flax does.
    del args, kwargs
    init_fn = functools.partial(
        super().init, method=self.force_init, mutable=True)
    # Disable logging to reduce logspam.
    with py_utils.logging_verbosity_level('FATAL'):
      variables_abstract = jax.eval_shape(init_fn, rngs)
    if 'params_axes' in variables_abstract:
      del variables_abstract['params_axes']
    return flax_core.unfreeze(unbox_meta(variables_abstract))

  # Notes on Flax interoperability:
  #
  # If a Flax module contains a BaseLayer submodule, say
  #
  #   class FlaxModule(nn.Module):
  #     base_layer: BaseLayer
  #
  #     @@nn.compact
  #     def __call__(self, x):
  #       y = base_layer(x)
  #       ...
  #
  #  == How BaseLayer puts variables inside self.scope:
  #
  #  `FlaxModule.init` or `FlaxModule.apply` first call `FlaxModule.setup`,
  #  which then calls `base_layer.setup()` because `base_layer` is a submodule.
  #  `base_layer.setup()` calls `self.create_variable` which ultimately calls
  #  `self.param` and `self.variable` and put BoxedParams into self.scope.
  #
  #  == How FlaxModule.__call__ retrieves variables from self.scope
  #
  # `base_layer.fprop` internally may call `base_layer.theta` or
  # `base_layer.get_var` which ultimately calls `self.get_variable` to retrieve
  # the variables.
  # Since these variables are BoxedParams inside self.scope and FlaxModule
  # cannot handle BoxedParams, we need to unbox the return value of
  # `self.get_variable`.
  #
  # For Flax interop only, BaseLayer users should not use self.get_variable
  # directly. Note that this optional unboxing is a no-op for pure Pax case:
  #   initial_vars = layer.init(k)
  #   outputs = layer.apply(initial_vars, method=layer.fprop)
  # Here the unboxing is done at the top level `layer.init`, so self.scope
  # inside `layer.apply` sees unboxed variables, which means `self.get_variable`
  # also only sees already unboxed variables.
  def get_variable(self, col: str, name: str, default=None):
    retval = super().get_variable(col, name, default)
    # Unbox returned value in case Flax calls Pax.
    return maybe_unbox_value(retval)

  @nn.nowrap
  def add_summary(self,
                  name: str,
                  tensor: JTensor,
                  summary_type: SummaryType = SummaryType.SCALAR):

    next_iter = 0
    summary_name = name
    full_name = summary_name
    while self.has_variable(SUMMARIES, full_name):
      next_iter += 1
      full_name = summary_name + str(next_iter)
    full_name = full_name + get_summary_type_suffix(summary_type)
    if summary_type == SummaryType.IMAGE:
      if tensor.ndim == 3:
        # Add a batch dim.
        tensor = jnp.expand_dims(tensor, 0)
      assert tensor.ndim == 4
    # full_name is ensured to be unique.
    # reduction function is "overwrite" if layer is called multiple times.
    self.sow(SUMMARIES, full_name, tensor, reduce_fn=lambda x, y: y)

  @nn.nowrap
  def add_aux_loss(self, name: str, value: JTensor, weight=None):
    # Accumulate by summing aux_loss.
    if weight is None:
      weight = jnp.ones_like(value)

    def reduce_fn(x, y):
      assert isinstance(x, AuxLossStruct)
      assert isinstance(y, AuxLossStruct)
      return AuxLossStruct(value=x.value + y.value, weight=x.weight + y.weight)

    self.sow(
        AUX_LOSS,
        name,
        AuxLossStruct(value, weight),
        init_fn=lambda: AuxLossStruct(0.0, 0.0),
        reduce_fn=reduce_fn)

  @nn.nowrap
  def next_prng_key(self, name=RANDOM):
    return self.make_rng(name)

  @property
  def name(self) -> str:
    return self.hparams.name

  @property
  def jax_context(self) -> JaxContext:
    return cur_jax_context()

  @property
  def do_eval(self) -> bool:
    return self.jax_context.do_eval

  @property
  def fprop_dtype(self) -> Any:
    if self.hparams.fprop_dtype is not None:
      return self.hparams.fprop_dtype
    else:
      return self.hparams.dtype

  @nn.nowrap
  def get_var(self, name: str) -> Any:
    assert self.has_variable(NON_TRAINABLE, name)
    return self.get_variable(NON_TRAINABLE, name)

  @nn.nowrap
  def update_var(self, name: str, new_val: JTensor) -> None:
    """Update var 'name' in the forward pass."""
    old_val = self.get_var(name)
    if self.is_mutable_collection(NON_TRAINABLE) and not self.is_initializing():
      asserts.eq(old_val.shape, new_val.shape)
      self.put_variable(NON_TRAINABLE, name, new_val)

  @nn.nowrap
  def get_decode_state(self, name: str) -> JTensor:
    """Looks up decode state with given name.

    The decode state is batch major.
    Args:
      name: Variable name in decoder cache.

    Returns:
      Decode state with the given name.
    """
    assert self.has_variable(DECODE_CACHE, name)
    return self.get_variable(DECODE_CACHE, name)

  @nn.nowrap
  def update_decode_state(self, name: str, new_state: JTensor) -> None:
    """Updates decode state with the new value.

    This function can be used to initialize decode state as well. When
    DECODE_CACHE
    is immutable, this is a no-op.

    Args:
      name: Variable name in decoder cache.
      new_state: New state to update.
    """
    if self.is_mutable_collection(DECODE_CACHE):
      self.put_variable(DECODE_CACHE, name, new_state)

  @nn.nowrap
  def create_variable(self,
                      name: str,
                      var_hparams: WeightHParams,
                      trainable: bool = True) -> None:
    """Create a variable of this layer according to the parameter `var_hparams`.

    E.g.::

        def create_layer_variables(self):
          self.create_variable(
              'weight', WeightHParams(shape=[100, 100]))

    Args:
      name: Variable name which is used as the key into vars/theta.
      var_hparams: WeightHParams used to create the variable.
      trainable: whether or not this param is trainable.
    """
    p = self.hparams
    var_hparams = var_hparams.clone()

    # If users did not specify init and dtype for var_hparams, fill in from
    # self.hparams.
    if var_hparams.init is None:
      var_hparams.init = p.params_init.clone()
    if var_hparams.dtype is None:
      var_hparams.dtype = p.dtype

    if p.mesh_shape is not None:
      if (len([dim for dim in var_hparams.shape if dim > 1]) > 1 and
          var_hparams.tensor_split_dims_mapping is None):
        logging.warning('tensor_split_dims_mapping missing for %s: shape=%s',
                        self.scope.path_text + '/' + name, var_hparams.shape)
      if var_hparams.tensor_split_dims_mapping is not None:
        assert len(var_hparams.tensor_split_dims_mapping) == len(
            var_hparams.shape)

    if var_hparams.collections is None:
      var_hparams.collections = []

    if (p.skip_lp_regularization and
        WeightHParamsCollection.SKIP_LP_REGULARIZATION
        not in var_hparams.collections):
      var_hparams.collections = var_hparams.collections + [
          WeightHParamsCollection.SKIP_LP_REGULARIZATION
      ]

    if (not trainable) and (WeightHParamsCollection.NON_TRAINABLE
                            not in var_hparams.collections):
      var_hparams.collections = var_hparams.collections + [
          WeightHParamsCollection.NON_TRAINABLE
      ]

    if trainable:
      # This is a param in Flax terminology.
      def _initializer_fn(prng_key: PRNGKey):
        value = init_var(name, var_hparams, prng_key)
        return BoxedParam(value=value, meta=var_hparams)

      self.param(name, _initializer_fn)
      # Add var to the private theta name set for checks.
      self._theta.add(name)
    else:

      def _initializer_fn():
        # Use params rng stream to avoid having caller to provide one for
        # non-trainable variables.
        prng_key = self.make_rng(PARAMS)
        value = init_var(name, var_hparams, prng_key)
        return BoxedParam(value=value, meta=var_hparams)

      # Non-trainable variables go into Flax nontrainable var collection.
      self.variable(NON_TRAINABLE, name, _initializer_fn)

  @nn.nowrap
  def create_child(self, name: str, params: BaseLayer.HParams) -> None:
    """Creates a sub layer.

    The created sub layer can be accessed by `name`. E.g.::

        self.create_child('foo', foo_params)
        self.foo.fprop...

    or:

        self.children['foo'].Fprop...
        self.children.foo.Fprop...

    If the layer does not have a name set, i.e. foo_params.name is None, then
    its name will be set to `name`.

    Args:
      name: Sub layer name which is used as the key into vars/theta.
      params: `Hyperparams` object to instantiate a layer.
    """
    p = self.copy_base_hparams(self.hparams, params.clone())
    p.name = name
    assert p.name not in self._private_children
    child = instantiate(p)
    self._private_children[p.name] = child
    setattr(self, p.name, child)

  @nn.nowrap
  def create_children(self, name: str,
                      params: Sequence[BaseLayer.HParams]) -> None:
    """Creates a list of sub layers.

    The created sub layer list can be accessed by `name`. E.g.::

        self.create_children('foo', ...)
        self.foo[10].fprop...

    Args:
      name: The name for the sub layers, which is used as the key into
        vars/theta.
      params: a list of `Hyperparams` objects to create.
    """
    params = NestedMap.FromNestedDict(params)

    uid = itertools.count()

    def _instantiate(p: InstantiableHyperParams) -> BaseLayerT:
      p = self.copy_base_hparams(self.hparams, p.clone())
      p.name = '%s_%d' % (name, next(uid))
      child = instantiate(p)
      assert p.name not in self._private_children
      self._private_children[p.name] = child
      return child

    setattr(self, name, NestedMap(sub=params).Transform(_instantiate).sub)

  @nn.nowrap
  def _cast_to_fprop_dtype(self, value: Any) -> Any:
    """Casts values to the desired dtype."""

    def _cast(x):
      if x is None:
        return None
      if self.fprop_dtype != x.dtype:
        if jnp.issubdtype(x.dtype, jnp.floating):
          return x.astype(self.fprop_dtype)
      return x

    return tf.nest.map_structure(_cast, value)


def assert_has_shape(t: JTensor, shape: Sequence[int]) -> None:
  asserts.eq(t.ndim, len(shape))
  for i in range(t.ndim):
    if shape[i] != -1:
      asserts.eq(t.shape[i], shape[i])
