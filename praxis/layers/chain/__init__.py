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

"""Expose public api of Chain and utilities."""

from praxis.layers.chain.chain import Chain
from praxis.layers.chain.chain_extensions import add_residual
from praxis.layers.chain.chain_extensions import apply_padding
from praxis.layers.chain.chain_extensions import chain
from praxis.layers.chain.chain_extensions import copy_n_times
from praxis.layers.chain.chain_extensions import dict_to_args
from praxis.layers.chain.chain_extensions import feed_forward
from praxis.layers.chain.chain_extensions import full_like
from praxis.layers.chain.chain_extensions import kwargs_with_name
from praxis.layers.chain.chain_extensions import log_args
from praxis.layers.chain.chain_extensions import repeat
