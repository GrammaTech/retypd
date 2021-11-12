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
from typing import Iterable, Optional, Sequence
import os

class CType(ABC):
    @property
    def size(self) -> int:
        pass

    @property
    def comment(self) -> Optional[str]:
        return None

    def declare(self, name: str) -> str:
        return f'{self} {name}'


class VoidType(CType):
    @property
    def size(self) -> int:
        return 0

    def __str__(self) -> str:
        return 'void'


class IntType(CType):
    def __init__(self, width: int, signed: bool) -> None:
        self.width = width
        self.signed = signed

    @property
    def size(self) -> int:
        return self.width

    def __str__(self) -> str:
        signed_tag = ''
        if not self.signed:
            signed_tag = 'u'
        return f'{signed_tag}int{self.width*8}_t'


class FloatType(CType):
    def __init__(self, width: int) -> None:
        self.width = width

    @property
    def size(self) -> int:
        return self.width

    def __str__(self) -> str:
        return f'float{self.width}_t'


class CharType(CType):
    # no chars aren't _always_ 8 bits
    def __init__(self, width: int) -> None:
        self.width = width

    @property
    def size(self) -> int:
        return self.width

    def __str__(self) -> str:
        return f'char{self.width}_t'


class ArrayType(CType):
    next_id = 0
    def __init__(self, member_type: CType, length: int) -> None:
        self.member_type = member_type
        self.length = length
        self.id = ArrayType.next_id
        ArrayType.next_id += 1

    @property
    def size(self) -> int:
        return self.member_type.size * self.length

    def __str__(self) -> str:
        return f'{self.member_type}[{self.length}]'

    def declare(self, name: str) -> str:
        return f'{self.member_type} {name}[{self.length}]'


class PointerType(CType):
    def __init__(self, target_type: CType) -> None:
        self.target_type = target_type

    @property
    def size(self) -> int:
        raise NotImplementedError()

    def __str__(self) -> str:
        return f'{self.target_type}*'


class FunctionType(CType):
    next_id = 0
    def __init__(self,
                 return_type: CType,
                 params: Sequence[CType],
                 name: Optional[str]=None) -> None:
        self.return_type = return_type
        self.params = params
        if name:
            self.name = name
        else:
            self.name = f'function_{FunctionType.next_id}'
            FunctionType.next_id += 1

    @property
    def size(self) -> int:
        raise NotImplementedError()

    def __str__(self) -> str:
        return self.name

    def declare(self, _name: str) -> str:
        return f'{self.return_type} {self.name}({", ".join(map(str, self.params))});'


class Field:
    def __init__(self, ctype: CType, offset: Optional[int]=None) -> None:
        self.ctype = ctype
        self.offset = offset

    @property
    def size(self) -> int:
        return self.ctype.size


class CompoundType(CType):
    @property
    def compound_type(self) -> str:
        return 'compound'

    @property
    def fields(self) -> Iterable[Field]:
        return []

    @property
    def name(self) -> str:
        return 'UNKNOWN'

    def __str__(self) -> str:
        return f'{self.compound_type} {self.name}'

    def declare(self, name: str) -> str:
        nt = f'{os.linesep}\t'
        result = f'self.compound_type {name} {{'
        for index, field in enumerate(self.fields):
            name = f'field_{index}'
            result += f'{nt}{field.ctype.declare(name)};'
            if field.offset is not None:
                result += f' // offset {field.offset}'
        return f'{result}{os.linesep}}};'


class StructType(CompoundType):
    next_id = 0

    def __init__(self, name: Optional[str]=None) -> None:
        if name:
            self._name = name
        else:
            self._name = f'struct_{StructType.next_id}'
            StructType.next_id += 1

    def set_fields(self, fields: Iterable[Field]):
        """
        We need to be able to construct a Struct before populating it so that we can
        represent recursive types.
        """
        self._fields = sorted(fields, key=lambda f: f.offset)

    @property
    def size(self) -> int:
        raise NotImplementedError()

    @property
    def compound_type(self) -> str:
        return 'struct'

    @property
    def name(self) -> str:
        return self._name

    @property
    def fields(self) -> Iterable[Field]:
        return self._fields


class UnionType(CompoundType):
    next_id = 0
    def __init__(self, fields: Iterable[Field], name: Optional[str]=None) -> None:
        self._fields = fields
        if name:
            self._name = name
        else:
            self._name = f'union_{UnionType.next_id}'
            UnionType.next_id += 1

    @property
    def size(self) -> int:
        return max(map(lambda t: t.size, self._fields))

    @property
    def compound_type(self) -> str:
        return 'union'

    @property
    def name(self) -> str:
        return self._name

    @property
    def fields(self) -> Iterable[Field]:
        return self._fields
