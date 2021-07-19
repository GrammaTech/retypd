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

from abc import ABC
from typing import Sequence

class CType(ABC):
    @property
    def size(self) -> int:
        pass


class VoidType(CType):
    @property
    def size(self) -> int:
        return 0


class IntType(CType):
    def __init__(self, width: int, signed: bool) -> None:
        self.width = width
        self.signed = signed

    @property
    def size(self) -> int:
        return self.width


class FloatType(CType):
    def __init__(self, width: int) -> None:
        self.width = width

    @property
    def size(self) -> int:
        return self.width


class CharType(CType):
    # no chars aren't _always_ 8 bits
    def __init__(self, width: int) -> None:
        self.width = width

    @property
    def size(self) -> int:
        return self.width


class FieldType(CType):
    def __init__(self, ctype: CType, offset: int) -> None:
        self.ctype = ctype
        self.offset = offset

    @property
    def size(self) -> int:
        return self.ctype.size


class StructType(CType):
    def __init__(self, fields: Sequence[FieldType]) -> None:
        self.fields = fields

    @property
    def size(self) -> int:
        raise NotImplemented


class ArrayType(CType):
    def __init__(self, member_type: CType, length: int) -> None:
        self.member_type = member_type
        self.length = length

    @property
    def size(self) -> int:
        return self.member_type.size * self.length


class PointerType(CType):
    def __init__(self, target_type: CType) -> None:
        self.target_type = target_type

    @property
    def size(self) -> int:
        raise NotImplemented


class FunctionType(CType):
    def __init__(self, return_type: CType, params: Sequence[CType]) -> None:
        self.return_type = return_type
        self.params = params

    @property
    def size(self) -> int:
        raise NotImplemented


def UnionType(CType):
    def __init__(self, ctypes: Set[CType]) -> None:
        self.ctypes = ctypes

    @property
    def size(self) -> int:
        return max(map(lambda t: t.size, self.ctypes))
