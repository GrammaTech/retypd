# Retypd - machine code type inference
# Copyright (C) 2021 GrammaTech, Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# This project is sponsored by the Office of Naval Research, One Liberty
# Center, 875 N. Randolph Street, Arlington, VA 22203 under contract #
# N68335-17-C-0700.  The content of the information does not necessarily
# reflect the position or policy of the Government and no official
# endorsement should be inferred.

'''An abstract lattice type for atomic types (e.g., primitives). Also includes a small
implementation for reference.
'''

from typing import FrozenSet, Any
from .schema import DerivedTypeVariable, Lattice, LatticeCTypes
from .c_types import IntType, PointerType


class DummyLattice(Lattice[DerivedTypeVariable]):
    _int = DerivedTypeVariable('int')
    _success = DerivedTypeVariable('#SuccessZ')
    _fd = DerivedTypeVariable('#FileDescriptor')
    _str = DerivedTypeVariable('str')
    _top = DerivedTypeVariable('┬')
    _bottom = DerivedTypeVariable('┴')
    _internal = frozenset({_int, _fd, _success, _str})
    _endcaps = frozenset({_top, _bottom})

    def __init__(self) -> None:
        pass

    @property
    def atomic_types(self) -> FrozenSet[DerivedTypeVariable]:
        return DummyLattice._internal | DummyLattice._endcaps

    @property
    def internal_types(self) -> FrozenSet[DerivedTypeVariable]:
        return DummyLattice._internal

    @property
    def top(self) -> DerivedTypeVariable:
        return DummyLattice._top

    @property
    def bottom(self) -> DerivedTypeVariable:
        return DummyLattice._bottom

    def meet(self, t: DerivedTypeVariable, v: DerivedTypeVariable) -> DerivedTypeVariable:
        if t == v:
            return t
        # idempotence
        if t == DummyLattice._top:
            return v
        if v == DummyLattice._top:
            return t
        types = {t, v}
        # dominance
        if DummyLattice._bottom in types:
            return DummyLattice._bottom
        # the two types are not equal and neither is TOP or BOTTOM, so if either is STR then the two
        # are incomparable
        if DummyLattice._str in types:
            return DummyLattice._bottom
        # the remaining cases are integral types. If one is INT, they are comparable
        if DummyLattice._int in types:
            types -= {DummyLattice._int}
            return next(iter(types))
        # the only remaining case is SUCCESS and FILE_DESCRIPTOR, which are not comparable
        return DummyLattice._bottom

    def join(self, t: DerivedTypeVariable, v: DerivedTypeVariable) -> DerivedTypeVariable:
        if t == v:
            return t
        # idempotence
        if t == DummyLattice._bottom:
            return v
        if v == DummyLattice._bottom:
            return t
        types = {t, v}
        # dominance
        if DummyLattice._top in types:
            return DummyLattice._top
        # the two types are not equal and neither is TOP or BOTTOM, so if either is STR then the two
        # are incomparable
        if DummyLattice._str in types:
            return DummyLattice._top
        # the remaining cases are integral types. In all three combinations of two, the least upper
        # bound is INT.
        return DummyLattice._int

class DummyLatticeCTypes(LatticeCTypes):
    def atom_to_ctype(self, atom_lower: Any, atom_upper: Any, byte_size: int):
        return {
            DummyLattice._int: IntType(byte_size, True),
            DummyLattice._success: IntType(byte_size, True),
            DummyLattice._fd: IntType(byte_size, False),
            DummyLattice._str: PointerType(CharType(1), byte_size)
        }.get(atom, ArrayType(IntType(1, False), byte_size))
