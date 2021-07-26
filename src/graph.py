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

from enum import Enum, unique
from typing import Any, Optional, Tuple
from .schema import AccessPathLabel, DerivedTypeVariable, Variance


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
