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

"""Tests for utilities."""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import fiddle as fdl
import jax
from jax import numpy as jnp
import numpy as np
from praxis import test_utils
from praxis.layers.quantization import utils


class UtilsTest(test_utils.TestCase):

  @parameterized.parameters(
      dict(eqn='ANH,NHD->AD', lhs_shape=(2, 3, 4), rhs_shape=(3, 4, 2)),
      dict(eqn='ANH,DNH->AD', lhs_shape=(2, 3, 4), rhs_shape=(2, 3, 4)),
      dict(eqn='AD,DNH->ANH', lhs_shape=(2, 3), rhs_shape=(3, 4, 2)),
      dict(eqn='AD,KDNH->KANH', lhs_shape=(2, 3), rhs_shape=(2, 3, 4, 2)),
      dict(
          eqn='BTNH,BSNH->BTNS',
          lhs_shape=(1, 2, 3, 4),
          rhs_shape=(1, 5, 3, 4),
      ),
      dict(
          eqn='BTNH,HNBS->STNB',
          lhs_shape=(1, 2, 3, 4),
          rhs_shape=(4, 3, 1, 5),
      ),
      dict(
          # Tests einsum with non-matching contraction dimension orders.
          eqn='ABCD,DC->AB',
          lhs_shape=(1, 2, 3, 4),
          rhs_shape=(4, 3),
      ),
  )
  def test_einsum_equation_conversion(self, eqn, lhs_shape, rhs_shape):
    """Validate that lax.dot_general produces the same output as jnp.einsum."""
    lhs = jnp.arange(np.prod(lhs_shape)).reshape(lhs_shape)
    rhs = jnp.arange(np.prod(rhs_shape)).reshape(rhs_shape)

    einsum_result = jnp.einsum(eqn, lhs, rhs)
    dimension_numbers, perm = utils.einsum_eqn_to_dimension_numbers(eqn)
    dot_general_result = jax.lax.dot_general(
        lhs, rhs, dimension_numbers=dimension_numbers
    )
    if perm is not None:
      dot_general_result = jax.lax.transpose(dot_general_result, perm)
    self.assertAllClose(einsum_result, dot_general_result)

  @parameterized.parameters(
      dict(eqn='...AB,BC->...AC', error=NotImplementedError, regex=r'\.\.\.'),
      dict(eqn='AB->BA', error=ValueError, regex=r'arguments'),
      dict(eqn='ABB,BC->AC', error=ValueError, regex=r'Repeated'),
      dict(eqn='ABC,AB->AB', error=ValueError, regex=r'Contraction.*C'),
  )
  def test_unsupported_einsum_equations(self, eqn, error, regex):
    with self.assertRaisesRegex(error, regex):
      utils.einsum_eqn_to_dimension_numbers(eqn)

  @parameterized.parameters(
      # Tests for packing/unpacking to/from int32
      dict(
          vals=[range(-8, 8)],
          shape=(8, 2),
          dtype=jnp.int8,
          pack_dim=0,
          packed_dtype=jnp.int32,
      ),
      dict(
          vals=[range(-8, 8)] * 2,
          shape=(8, 2, 2),
          dtype=jnp.int8,
          pack_dim=0,
          packed_dtype=jnp.int32,
      ),
      dict(
          vals=[range(-8, 8)] * 2,
          shape=(2, 8, 2),
          dtype=jnp.int8,
          pack_dim=1,
          packed_dtype=jnp.int32,
      ),
      dict(
          vals=[range(7, -9, -1)] * 2,
          shape=(2, 8, 2),
          dtype=jnp.int8,
          pack_dim=1,
          packed_dtype=jnp.int32,
      ),
      dict(
          vals=[range(0, 16)] * 2,
          shape=(8, 2, 2),
          dtype=jnp.uint8,
          pack_dim=0,
          packed_dtype=jnp.int32,
      ),
      dict(
          vals=[range(0, 16)] * 2,
          shape=(2, 8, 2),
          dtype=jnp.uint8,
          pack_dim=1,
          packed_dtype=jnp.int32,
      ),
      # Tests for packing/unpacking to/from int8
      dict(
          vals=[range(-8, 8)],
          shape=(8, 2),
          dtype=jnp.int8,
          pack_dim=0,
          packed_dtype=jnp.int8,
      ),
      dict(
          vals=[range(-8, 8)] * 2,
          shape=(8, 2, 2),
          dtype=jnp.int8,
          pack_dim=0,
          packed_dtype=jnp.int8,
      ),
      dict(
          vals=[range(-8, 8)] * 2,
          shape=(2, 8, 2),
          dtype=jnp.int8,
          pack_dim=1,
          packed_dtype=jnp.int8,
      ),
      dict(
          vals=[range(7, -9, -1)] * 2,
          shape=(2, 8, 2),
          dtype=jnp.int8,
          pack_dim=1,
          packed_dtype=jnp.int8,
      ),
  )
  def test_pack_4bit_unpack_4bit(
      self, vals, shape, dtype, pack_dim, packed_dtype
  ):
    x = jnp.array(vals).reshape(shape).astype(dtype)
    packed = utils.pack_4bit(x, pack_dim, packed_dtype)
    expected_packed_shape = list(x.shape)
    int4s_per_packed_type = 8 if packed_dtype == jnp.int32 else 2
    expected_packed_shape[pack_dim] //= int4s_per_packed_type
    self.assertSequenceEqual(packed.shape, expected_packed_shape)

    unpacked = utils.unpack_4bit(packed, pack_dim, x.dtype)
    self.assertArraysEqual(unpacked, x)

  def test_get_packed_shape(self):
    self.assertSequenceEqual(utils.get_packed_shape((4, 8, 3), 1, 8), (4, 1, 3))
    self.assertRaises(ValueError, utils.get_packed_shape, (4, 7, 3), 1, 8)

  @parameterized.named_parameters(
      ('single target', True),
      ('multiple targets', False),
  )
  def test_find_target_tpl(self, sequence_of_inputs):
    @dataclasses.dataclass(frozen=True)
    class Target:
      marker: str = 'default'

    @dataclasses.dataclass()
    class Inner:
      irrelevant: int
      target1: Target

    @dataclasses.dataclass()
    class Outer:
      irrelevant: int
      target2: Target
      inner_direct: Inner
      inner_list: list[Inner]
      inner_dict: dict[str, Inner]

    outer_p = (
        fdl.Config(
            Outer,
            irrelevant=-1,
            target2=fdl.Config(Target, marker='target2'),
            inner_direct=fdl.Config(
                Inner,
                irrelevant=-2,
                target1=fdl.Config(Target, marker='target1_1'),
            ),
            inner_list=[
                fdl.Config(
                    Inner,
                    irrelevant=-3,
                    target1=fdl.Config(Target, marker='target1_2'),
                ),
                fdl.Config(
                    Inner,
                    irrelevant=-4,
                    target1=fdl.Config(Target, marker='target1_3'),
                ),
            ],
            inner_dict={
                'a': fdl.Config(
                    Inner,
                    irrelevant=-5,
                    target1=fdl.Config(Target, marker='target1_4'),
                ),
                'b': fdl.Config(
                    Inner,
                    irrelevant=-5,
                    target1=fdl.Config(Target, marker='target1_5'),
                ),
            },
        ),
    )
    if sequence_of_inputs:
      targets = utils.find_target_tpl(outer_p, [Target, Target])
    else:
      targets = utils.find_target_tpl(outer_p, Target)
    # NOTE(yinzhong): fdl.Config is not hashable or sortable, so we have to
    # build before comparing.
    self.assertSameElements(
        fdl.build(targets),
        fdl.build([
            fdl.Config(Target, marker='target2'),
            fdl.Config(Target, marker='target1_1'),
            fdl.Config(Target, marker='target1_2'),
            fdl.Config(Target, marker='target1_3'),
            fdl.Config(Target, marker='target1_4'),
            fdl.Config(Target, marker='target1_5'),
        ]),
    )

  def test_lora_shape_and_eqn(self):
    # One reduction dimension.
    eqn = '...td,dD->...tD'
    shape = (3, 5)
    lora_size = 2
    (
        new_eqn_left,
        new_eqn_right,
        left_shape,
        right_shape,
        eqn_left_ind,
        eqn_right_ind,
    ) = utils.get_lora_shape_and_eqn(shape, lora_size, eqn)
    self.assertEqual(new_eqn_left, '...td,da->...ta')
    self.assertEqual(new_eqn_right, '...ta,aD->...tD')
    self.assertEqual(left_shape, [3, 2])
    self.assertEqual(right_shape, [2, 5])
    self.assertEqual(eqn_left_ind, [1])
    self.assertEqual(eqn_right_ind, [0])

    # Two reduction dimensions.
    eqn = '...tdh,dDh->...tD'
    shape = (4, 5, 8)  # Max reduction dim is 'h' = 8
    lora_size = 2
    (
        new_eqn_left,
        new_eqn_right,
        left_shape,
        right_shape,
        eqn_left_ind,
        eqn_right_ind,
    ) = utils.get_lora_shape_and_eqn(shape, lora_size, eqn)
    self.assertEqual(new_eqn_left, '...tdh,ha->...tda')
    self.assertEqual(new_eqn_right, '...tda,dDa->...tD')

    self.assertEqual(left_shape, [8, 2])
    self.assertEqual(right_shape, [4, 5, 2])
    self.assertEqual(eqn_left_ind, [1])
    self.assertEqual(eqn_right_ind, [2])

    # Two reduction dimensions.
    eqn = '...tdh,dDh->...tD'
    shape = (8, 5, 4)  # Max reduction dim is 'd' = 8
    lora_size = 2
    (
        new_eqn_left,
        new_eqn_right,
        left_shape,
        right_shape,
        eqn_left_ind,
        eqn_right_ind,
    ) = utils.get_lora_shape_and_eqn(shape, lora_size, eqn)
    self.assertEqual(new_eqn_left, '...tdh,da->...tah')
    self.assertEqual(new_eqn_right, '...tah,aDh->...tD')

    self.assertEqual(left_shape, [8, 2])
    self.assertEqual(right_shape, [2, 5, 4])
    self.assertEqual(eqn_left_ind, [1])
    self.assertEqual(eqn_right_ind, [0])

    # Two reduction dimensions.
    eqn = '...tdh,dDh->...tD'
    shape = (8, 5, 4)  # Max reduction dim is 'd' = 8
    lora_size = 2
    (
        new_eqn_left,
        new_eqn_right,
        left_shape,
        right_shape,
        eqn_left_ind,
        eqn_right_ind,
    ) = utils.get_lora_shape_and_eqn(shape, lora_size, eqn, max_reduction=False)

    self.assertEqual(new_eqn_left, '...tdh,dhab->...tab')
    self.assertEqual(new_eqn_right, '...tab,aDb->...tD')

    self.assertEqual(left_shape, [8, 4, 2, 2])
    self.assertEqual(right_shape, [2, 5, 2])
    self.assertEqual(eqn_left_ind, [2, 3])
    self.assertEqual(eqn_right_ind, [0, 2])


if __name__ == '__main__':
  absltest.main()
