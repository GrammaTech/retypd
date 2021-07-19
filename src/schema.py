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

'''Data types for an implementation of retypd analysis.
'''

from abc import ABC
from enum import Enum, unique
from functools import reduce
from typing import Any, Iterable, List, Optional, Sequence, Tuple
import logging
import os


logging.basicConfig()


@unique
class Variance(Enum):
    '''Represents a capability's variance (or that of some sequence of capabilities).
    '''
    CONTRAVARIANT = 0
    COVARIANT = 1

    @staticmethod
    def invert(variance: 'Variance') -> 'Variance':
        if variance == Variance.CONTRAVARIANT:
            return Variance.COVARIANT
        return Variance.CONTRAVARIANT

    @staticmethod
    def combine(lhs: 'Variance', rhs: 'Variance') -> 'Variance':
        if lhs == rhs:
            return Variance.COVARIANT
        return Variance.CONTRAVARIANT


class AccessPathLabel(ABC):
    '''Abstract class for capabilities that can be part of a path. See Table 1.

    All :py:class:`AccessPathLabel` objects are comparable to each other; objects are ordered by
    their classes (in an arbitrary order defined by the string representation of their type), then
    by values specific to their subclass. So objects of class A always precede objects of class B
    and objects of class A are ordered with respect to each other by :py:method:`_less_than`.
    '''
    def __lt__(self, other: 'AccessPathLabel') -> bool:
        s_type = str(type(self))
        o_type = str(type(other))
        if s_type == o_type:
            return self._less_than(other)
        return s_type < o_type

    def _less_than(self, _other) -> bool:
        '''Compare two objects of the same exact type. Return True if self is less than other; true
        otherwise. Several of the subclasses are singletons, so we return False unless there is a
        need for an overriding implementation.
        '''
        return False

    def variance(self) -> Variance:
        '''Determines if the access path label is covariant or contravariant, per Table 1.
        '''
        return Variance.COVARIANT


class LoadLabel(AccessPathLabel):
    '''A singleton representing the load (read) capability.
    '''
    _instance = None

    def __init__(self) -> None:
        raise ValueError("Can't instantiate; call instance() instead")

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
        return cls._instance

    def __eq__(self, other: Any) -> bool:
        return self is other

    def __hash__(self) -> int:
        return 0

    def __str__(self) -> str:
        return 'load'


class StoreLabel(AccessPathLabel):
    '''A singleton representing the store (write) capability.
    '''
    _instance = None

    def __init__(self) -> None:
        raise ValueError("Can't instantiate; call instance() instead")

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
        return cls._instance

    def __eq__(self, other: Any) -> bool:
        return self is other

    def __hash__(self) -> int:
        return 1

    def variance(self) -> Variance:
        return Variance.CONTRAVARIANT

    def __str__(self) -> str:
        return 'store'


class InLabel(AccessPathLabel):
    '''Represents a parameter to a function, specified by an index (e.g., the first argument might
    use index 0, the second might use index 1, and so on). N.B.: this is a capability and is not
    tied to any particular function.
    '''
    def __init__(self, index: int) -> None:
        self.index = index

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, InLabel) and self.index == other.index

    def _less_than(self, other: 'InLabel') -> bool:
        return self.index < other.index

    def __hash__(self) -> int:
        return hash(self.index)

    def variance(self) -> Variance:
        return Variance.CONTRAVARIANT

    def __str__(self) -> str:
        return f'in_{self.index}'


class OutLabel(AccessPathLabel):
    '''Represents a return from a function. This class is a singleton.
    '''
    _instance = None

    def __init__(self) -> None:
        raise ValueError("Can't instantiate; call instance() instead")

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
        return cls._instance

    def __eq__(self, other: Any) -> bool:
        return self is other

    def __hash__(self) -> int:
        return 2

    def __str__(self) -> str:
        return 'out'


class DerefLabel(AccessPathLabel):
    '''Represents a dereference in an access path. Specifies a size (the number of bytes read or
    written) and an offset (the number of bytes from the base).
    '''
    def __init__(self, size: int, offset: int) -> None:
        self.size = size
        self.offset = offset

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, DerefLabel) and
                self.size == other.size and
                self.offset == other.offset)

    def _less_than(self, other: 'DerefLabel') -> bool:
        if self.offset == other.offset:
            return self.size < other.size
        return self.offset < other.offset

    def __hash__(self) -> int:
        return hash(self.offset) ^ hash(self.size)

    def __str__(self) -> str:
        return f'σ{self.size}@{self.offset}'


class DerivedTypeVariable:
    '''A _derived_ type variable, per Definition 3.1. Immutable (by convention).
    '''
    def __init__(self, type_var: str, path: Optional[Sequence[AccessPathLabel]] = None) -> None:
        self.base = type_var
        if path is None:
            self.path: Sequence[AccessPathLabel] = ()
        else:
            self.path = tuple(path)
        if self.path:
            self._str: str = f'{self.base}.{".".join(map(str, self.path))}'
        else:
            self._str: str = self.base

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, DerivedTypeVariable) and
                self.base == other.base and
                self.path == other.path)

    def __lt__(self, other: 'DerivedTypeVariable') -> bool:
        if self.base == other.base:
            return list(self.path) < list(other.path)
        return self.base < other.base

    def __hash__(self) -> int:
        return hash(self.base) ^ hash(self.path)

    def largest_prefix(self) -> Optional['DerivedTypeVariable']:
        '''Return the prefix obtained by removing the last item from the type variable's path. If
        there is no path, return None.
        '''
        if self.path:
            return DerivedTypeVariable(self.base, self.path[:-1])
        return None

    def get_suffix(self, other: 'DerivedTypeVariable') -> Optional[Sequence[AccessPathLabel]]:
        '''If self is a prefix of other, return the suffix of other's path that is not part of self.
        Otherwise, return None.
        '''
        if self.base != other.base:
            return None
        if len(self.path) > len(other.path):
            return None
        for s_item, o_item in zip(self.path, other.path):
            if s_item != o_item:
                return None
        return other.path[len(self.path):]

    def tail(self) -> AccessPathLabel:
        '''Retrieve the last item in the access path, if any. Return None if
        the path is empty.
        '''
        if self.path:
            return self.path[-1]
        return None

    def add_suffix(self, suffix: AccessPathLabel) -> 'DerivedTypeVariable':
        '''Create a new :py:class:`DerivedTypeVariable` identical to :param:`self` (which is
        unchanged) but with suffix appended to its path.
        '''
        path: List[AccessPathLabel] = list(self.path)
        path.append(suffix)
        return DerivedTypeVariable(self.base, path)

    def get_single_suffix(self, prefix: 'DerivedTypeVariable') -> Optional[AccessPathLabel]:
        '''If :param:`prefix` is a prefix of :param:`self` with exactly one additional
        :py:class:`AccessPathLabel`, return the additional label. If not, return `None`.
        '''
        if (self.base != prefix.base or
                len(self.path) != (len(prefix.path) + 1) or
                self.path[:-1] != prefix.path):
            return None
        return self.tail()

    def path_variance(self) -> Variance:
        '''Determine the variance of the access path.
        '''
        variances = map(lambda label: label.variance(), self.path)
        return reduce(Variance.combine, variances, Variance.COVARIANT)

    def __str__(self) -> str:
        return self._str


class SubtypeConstraint:
    '''A type constraint of the form left ⊑ right (see Definition 3.3)
    '''
    def __init__(self, left: DerivedTypeVariable, right: DerivedTypeVariable) -> None:
        self.left = left
        self.right = right

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, SubtypeConstraint) and
                self.left == other.left and
                self.right == other.right)

    def __lt__(self, other: 'SubtypeConstraint') -> bool:
        if self.left == other.left:
            return self.right < other.right
        return self.left < other.left

    def __hash__(self) -> int:
        return hash(self.left) ^ hash(self.right)

    def __str__(self) -> str:
        return f'{self.left} ⊑ {self.right}'


class ConstraintSet:
    '''A (partitioned) set of type constraints
    '''
    def __init__(self, subtype: Optional[Iterable[SubtypeConstraint]] = None) -> None:
        if subtype:
            self.subtype = set(subtype)
        else:
            self.subtype = set()
        self.logger = logging.getLogger('ConstraintSet')

    def add_subtype(self, left: DerivedTypeVariable, right: DerivedTypeVariable) -> bool:
        '''Add a subtype constraint
        '''
        constraint = SubtypeConstraint(left, right)
        return self.add(constraint)

    def add(self, constraint: SubtypeConstraint) -> bool:
        if constraint in self.subtype:
            return False
        self.subtype.add(constraint)
        return True

    def __str__(self) -> str:
        nt = os.linesep + '\t'
        return f'ConstraintSet:{nt}{nt.join(map(str,self.subtype))}'



class EdgeLabel:
    '''A forget or recall label in the graph. Instances should never be mutated.
    '''
    @unique
    class Kind(Enum):
        FORGET = 1
        RECALL = 2

    def __init__(self, capability: AccessPathLabel, kind: Kind) -> None:
        self.capability = capability
        self.kind = kind
        if self.kind == EdgeLabel.Kind.FORGET:
            type_str = 'forget'
        else:
            type_str = 'recall'
        self._str = f'{type_str} {self.capability}'
        self._hash = hash(self.capability) ^ hash(self.kind)

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, EdgeLabel) and
                self.capability == other.capability and
                self.kind == other.kind)

    def __hash__(self) -> int:
        return self._hash

    def __str__(self) -> str:
        return self._str


class Node:
    '''A node in the graph of constraints. Node objects are immutable.

    Unforgettable is a flag used to differentiate between two subgraphs later in the algorithm. See
    :py:method:`Solver._unforgettable_subgraph_split` for details.
    '''

    @unique
    class Unforgettable(Enum):
        PRE_RECALL = 0
        POST_RECALL = 1

    def __init__(self,
                 base: DerivedTypeVariable,
                 suffix_variance: Variance,
                 unforgettable: Unforgettable = Unforgettable.PRE_RECALL) -> None:
        self.base = base
        self.suffix_variance = suffix_variance
        if suffix_variance == Variance.COVARIANT:
            variance = '.⊕'
            summary = 2
        else:
            variance = '.⊖'
            summary = 0
        self._unforgettable = unforgettable
        if unforgettable == Node.Unforgettable.POST_RECALL:
            self._str = 'R:' + str(self.base) + variance
            summary += 1
        else:
            self._str = str(self.base) + variance
        self._hash = hash(self.base) ^ hash(summary)

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, Node) and
                self.base == other.base and
                self.suffix_variance == other.suffix_variance and
                self._unforgettable == other._unforgettable)

    def __hash__(self) -> int:
        return self._hash

    def forget_once(self) -> Tuple[Optional[AccessPathLabel], Optional['Node']]:
        '''"Forget" the last element in the access path, creating a new Node. The new Node has
        variance that reflects this change.
        '''
        if self.base.path:
            prefix_path = list(self.base.path)
            last = prefix_path.pop()
            prefix = DerivedTypeVariable(self.base.base, prefix_path)
            return (last, Node(prefix, Variance.combine(last.variance(), self.suffix_variance)))
        return (None, None)

    def recall(self, label: AccessPathLabel) -> 'Node':
        '''"Recall" label, creating a new Node. The new Node has variance that reflects this
        change.
        '''
        path = list(self.base.path)
        path.append(label)
        variance = Variance.combine(self.suffix_variance, label.variance())
        return Node(DerivedTypeVariable(self.base.base, path), variance)

    def __str__(self) -> str:
        return self._str

    def split_unforgettable(self) -> 'Node':
        '''Get a duplicate of self for use in the post-recall subgraph.
        '''
        return Node(self.base, self.suffix_variance, Node.Unforgettable.POST_RECALL)

    def inverse(self) -> 'Node':
        '''Get a Node identical to this one but with inverted variance.
        '''
        return Node(self.base, Variance.invert(self.suffix_variance), self._unforgettable)
