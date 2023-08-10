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

"""Exposes public Lingvo symbols used by Praxis.

Lingvo symbols must *not* be imported elsewhere in Praxis library code (except
in test-related modules).
"""

from lingvo.core import cluster
from lingvo.core import cluster_factory
from lingvo.core import hyperparams
from lingvo.core import nested_map
from praxis import lazy_loader

# datasource is slow to import (because it imports TF), so we do it lazily.
datasource = lazy_loader.LazyLoader(
    'datasource', globals(), 'lingvo.core.datasource'
)

# Note: These are only used by LingvoInputAdaptor, and may possibly be moved
# outside of Praxis in the future.
current_cluster = cluster_factory.Current
infeed_context_scope = cluster.InfeedContextScope

# Core data-structure for aggregating tensors.
NestedMap = nested_map.NestedMap

# Note: HParams-related classes. This may possibly be removed post-Fiddle
# migration.
InstantiableParams = hyperparams.InstantiableParams
HParams = hyperparams.Params
