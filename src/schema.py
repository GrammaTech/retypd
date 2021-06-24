'''Data types for an implementation of retypd analysis. See `Link the paper
<https://arxiv.org/pdf/1603.05495v1.pdf>`, `Link the slides
<https://raw.githubusercontent.com/emeryberger/PLDI-2016/master/presentations/pldi16-presentation241.pdf>`,
and `Link the notes
<https://git.grammatech.com/reverse-engineering/common/re_facts/-/blob/paldous/type-recovery/docs/how-to/type-recovery.rst>`
for details

author: Peter Aldous
'''

from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Optional, Sequence
import logging
import os


logging.basicConfig()


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

    def is_covariant(self) -> bool:
        '''Determines if the access path label is covariant (True) or contravariant (False), per
        Table 1.
        '''
        return True


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

    def is_covariant(self) -> bool:
        return False

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

    def is_covariant(self) -> bool:
        return False

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

    def get_suffix(self, other: 'DerivedTypeVariable') -> Optional[Iterable[AccessPathLabel]]:
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

    def path_is_covariant(self):
        '''Determine if the access path is covariant or contravariant. This is a special case of
        :py:classmethod:`suffix_is_covariant`.
        '''
        return DerivedTypeVariable.suffix_is_covariant(self.path)

    @classmethod
    def suffix_is_covariant(cls, suffix: Sequence[AccessPathLabel]) -> bool:
        '''Given a sequence of :py:class:`AccessPathLabel` objects, determine if the suffix is
        covariant or contravariant.
        '''
        is_covariant = True
        for label in suffix:
            if not label.is_covariant():
                is_covariant = not is_covariant
        return is_covariant

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
        return (f'ConstraintSet:{nt}{nt.join(map(str,self.subtype))}')


class EdgeLabel(ABC):
    @abstractmethod
    def is_forget(self) -> bool:
        pass


class ForgetLabel(EdgeLabel):
    '''A forget label in the graph.
    '''
    def __init__(self, capability: AccessPathLabel) -> None:
        self.capability = capability

    def is_forget(self) -> bool:
        return True

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, ForgetLabel) and self.capability == other.capability

    def __hash__(self) -> int:
        return hash(self.capability)

    def __str__(self) -> str:
        return f'forget {self.capability}'


class RecallLabel(EdgeLabel):
    '''A recall label in the graph.
    '''
    def __init__(self, capability: AccessPathLabel) -> None:
        self.capability = capability

    def is_forget(self) -> bool:
        return False

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, RecallLabel) and self.capability == other.capability

    def __hash__(self) -> int:
        return ~hash(self.capability)

    def __str__(self) -> str:
        return f'recall {self.capability}'


class Vertex:
    '''A vertex in the graph of constraints. Vertex objects are immutable (by convention).
    '''
    def __init__(self,
                 base: DerivedTypeVariable,
                 suffix_variance: bool,
                 unforgettable: bool = False) -> None:
        self.base = base
        self.suffix_variance = suffix_variance
        if suffix_variance:
            variance = '.⊕'
        else:
            variance = '.⊖'
        self._unforgettable = unforgettable
        if unforgettable:
            self._str = 'R:' + str(self.base) + variance
        else:
            self._str = str(self.base) + variance

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, Vertex) and
                self.base == other.base and
                self.suffix_variance == other.suffix_variance and
                self._unforgettable == other._unforgettable)

    def __hash__(self) -> int:
        # TODO rotate one of the bool digests
        return hash(self.base) ^ hash(self.suffix_variance) ^ hash(self._unforgettable)

    def forget_once(self) -> Optional['Vertex']:
        '''"Forget" the last element in the access path, creating a new Vertex. The new Vertex has
        variance that reflects this change.
        '''
        prefix = self.base.largest_prefix()
        if prefix:
            last = self.base.path[-1]
            return Vertex(prefix, last.is_covariant() == self.suffix_variance)
        return None

    def recall(self, label: AccessPathLabel) -> 'Vertex':
        '''"Recall" label, creating a new Vertex. The new Vertex has variance that reflects this
        change.
        '''
        path = list(self.base.path)
        path.append(label)
        variance = self.suffix_variance == label.is_covariant()
        return Vertex(DerivedTypeVariable(self.base.base, path), variance)

    def _replace_last(self, label: AccessPathLabel) -> 'Vertex':
        '''Create a new Vertex whose access path's last label has been replaced by :param:`label`.
        Does not change variance.
        '''
        path = list(self.base.path[:-1])
        path.append(label)
        return Vertex(DerivedTypeVariable(self.base.base, path), self.suffix_variance)

    def implicit_target(self) -> Optional['Vertex']:
        '''If there is a lazily instantiated store/load edge from this node, find its target.
        '''
        if self.base.path:
            last = self.base.path[-1]
            if last is StoreLabel.instance() and self.suffix_variance:
                return self._replace_last(LoadLabel.instance())
            if last is LoadLabel.instance() and not self.suffix_variance:
                return self._replace_last(StoreLabel.instance())
        return None

    def __str__(self) -> str:
        return self._str

    def split_unforgettable(self) -> 'Vertex':
        return Vertex(self.base, self.suffix_variance, not self._unforgettable)

    def inverse(self) -> 'Vertex':
        return Vertex(self.base, not self.suffix_variance, self._unforgettable)
