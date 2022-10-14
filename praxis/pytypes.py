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

from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import clu.metrics as clu_metrics
import jax
from jax import numpy as jnp
from jax.experimental import pjit
import numpy as np
from praxis import py_utils

HParamsT = py_utils.HParams
JTensor = jnp.ndarray
PRNGKey = JTensor
JTensorOrPartitionSpec = Union[JTensor, pjit.PartitionSpec]
NpTensor = np.ndarray
SummaryDict = Union[py_utils.NestedMap, Dict[str, JTensor]]
PyTreeDef = type(jax.tree_util.tree_structure(None))

T = TypeVar('T')
Nested = Union[T, Tuple[Any, ...], List[Any], Dict[str, Any],
               py_utils.NestedMap]
NestedJTensor = Nested[JTensor]
NestedNpTensor = Nested[NpTensor]
NestedBool = Nested[bool]
NestedInt = Nested[int]
NestedHParams = Nested[HParamsT]
NestedPartitionSpec = Nested[pjit.PartitionSpec]
NestedJTensorOrPartitionSpec = Nested[JTensorOrPartitionSpec]
NestedShapeDtypeStruct = Nested[jax.ShapeDtypeStruct]
NestedShapedArray = Nested[jax.ShapedArray]
NestedShapeDtypeLike = Union[NestedJTensor, NestedNpTensor,
                             NestedShapeDtypeStruct, NestedShapedArray]

# Sharding annotation for a dim can be a single int, or a str, or a sequence of
# (int, str), or None. For example "1", "-1", "None", "data", "(data, replia)"
# are all valid sharding annoations for a particular tensor axis.
DimShardingAnnotation = Optional[Union[Sequence[Union[int, str]], int, str]]
SplitDimsMapping = Optional[Sequence[DimShardingAnnotation]]

# Note(b/238657605): pytypes Metrics were renamed to WeightedScalars
# and Metrics are now true metric objects using clu.metrics
WeightedScalar = Tuple[JTensor, JTensor]
WeightedScalars = Union[Dict[str, WeightedScalar], py_utils.NestedMap]
WeightedScalarsList = Union[Dict[str, Sequence[WeightedScalar]],
                            py_utils.NestedMap]
Metrics = Union[py_utils.NestedMap, Dict[str, clu_metrics.Metric]]

LogicalAxisRules = Sequence[Tuple[str, Optional[str]]]
