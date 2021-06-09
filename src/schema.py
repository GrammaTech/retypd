'''Data types for an implementation of retypd analysis.
See `Link the paper <https://arxiv.org/pdf/1603.05495v1.pdf>` and `Link the
slides
<https://raw.githubusercontent.com/emeryberger/PLDI-2016/master/presentations/pldi16-presentation241.pdf>`
for details

author: Peter Aldous
'''

from abc import ABC
from typing import Iterable, List, Optional, Sequence, Set
import os


class AccessPathLabel(ABC):
    '''Abstract class for capabilities that can be part of a path. See Table 1.

    All :py:class:`AccessPathLabel` objects are comparable to each other; objects are ordered by
    their classes (in the order they are created), then by internal values.
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

    def __eq__(self, other) -> bool:
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

    def __eq__(self, other) -> bool:
        return self is other

    def __hash__(self) -> int:
        return 1

    def is_covariant(self) -> bool:
        return False

    def __str__(self) -> str:
        return 'store'


class InLabel(AccessPathLabel):
    '''Represents a parameter to a function, specified by an index.
    '''
    def __init__(self, index: int) -> None:
        self.index = index

    def __eq__(self, other) -> bool:
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

    def __eq__(self, other) -> bool:
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

    def __eq__(self, other) -> bool:
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
    '''A _derived_ type variable, per Definition 3.1
    '''
    def __init__(self, type_var: str, path: Sequence[AccessPathLabel]) -> None:
        self.base = type_var
        if path is None:
            self.path: Sequence[AccessPathLabel] = ()
        else:
            self.path = tuple(path)

    def __eq__(self, other) -> bool:
        return (isinstance(other, DerivedTypeVariable) and
                self.base == other.base and
                self.path == other.path)

    def __lt__(self, other: 'DerivedTypeVariable') -> bool:
        if self.base == other.base:
            return list(self.path) < list(other.path)
        return self.base < other.base

    def __hash__(self) -> int:
        return hash(self.base) ^ hash(self.path)

    def prefixes(self) -> Set['ExistenceConstraint']:
        '''Retrieve all prefixes of the derived type variable as a set.
        '''
        path = tuple(self.path)
        result = set()
        while path:
            path = path[:-1]
            prefix = DerivedTypeVariable(self.base, path)
            result.add(ExistenceConstraint(prefix))
        return result

    def tail(self) -> AccessPathLabel:
        '''Retrieve the last item in the access path, if any. Return None if
        the path is empty.
        '''
        if self.path:
            return self.path[-1]
        return None

    def add_suffix(self, suffix: Sequence[AccessPathLabel]) -> 'DerivedTypeVariable':
        '''Create a new :py:class:`DerivedTypeVariable` identical to :param:`self` (which is
        unchanged) but with suffix appended to its path.
        '''
        path: List[AccessPathLabel] = list(self.path)
        path.extend(suffix)
        return DerivedTypeVariable(self.base, path)

    def get_suffix(self, prefix: 'DerivedTypeVariable') -> Optional[Sequence[AccessPathLabel]]:
        '''If :param:`prefix` is a prefix of :param:`self`, return the remainder of :param:`self`.
        If not, return `None`.
        '''
        prefix_length = len(prefix.path)
        if (self.base != prefix.base or
                len(self.path) <= prefix_length or
                self.path[:prefix_length] != prefix.path):
            return None
        return self.path[prefix_length:]

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
        if self.path:
            return f'{self.base}.{".".join(map(str, self.path))}'
        return self.base


class ExistenceConstraint:
    '''A type constraint of the form VAR a (see Definition 3.3)
    '''
    def __init__(self, var: DerivedTypeVariable) -> None:
        self.var = var

    def __eq__(self, other) -> bool:
        return (isinstance(other, ExistenceConstraint) and
                self.var == other.var)

    def __lt__(self, other: 'ExistenceConstraint') -> bool:
        return self.var < other.var

    def __hash__(self) -> int:
        return hash(self.var)

    def __str__(self) -> str:
        return f'VAR {self.var}'


class SubtypeConstraint:
    '''A type constraint of the form left ⊑ right (see Definition 3.3)
    '''
    def __init__(self, left: DerivedTypeVariable, right: DerivedTypeVariable) -> None:
        self.left = left
        self.right = right

    def __eq__(self, other) -> bool:
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
    def __init__(self,
                 existence: Optional[Iterable[ExistenceConstraint]] = None,
                 subtype: Optional[Iterable[SubtypeConstraint]] = None) -> None:
        if existence:
            self.existence = set(existence)
        else:
            self.existence = set()
        if subtype:
            self.subtype = set(subtype)
        else:
            self.subtype = set()

    def add_subtype(self, left: DerivedTypeVariable, right: DerivedTypeVariable) -> bool:
        '''Add a subtype constraint
        '''
        constraint = SubtypeConstraint(left, right)
        if constraint in self.subtype:
            return False
        self.subtype.add(constraint)
        return True

    def add_existence(self, var: DerivedTypeVariable) -> bool:
        '''Add an existence constraint
        '''
        ex = ExistenceConstraint(var)
        if ex in self.existence:
            return False
        self.existence.add(ex)
        return True

    def fix(self) -> 'ConstraintSet':
        '''Compute and return a fixed point on self's constraints. Does not
        mutate self; returns a new object. See Figure 3.
        '''
        new_existence: Optional[Set[ExistenceConstraint]] = None
        new_subtype: Optional[Set[SubtypeConstraint]] = None
        existence = set(self.existence)
        subtype = set(self.subtype)

        while (new_existence is None or
               new_subtype is None or
               not (new_existence <= existence and
                    new_subtype <= subtype)):
            if new_existence is not None:
                existence |= new_existence
            if new_subtype is not None:
                subtype |= new_subtype
            new_existence = set()
            new_subtype = set()

            for ex in existence:
                var = ex.var
                # S-Refl
                new_subtype.add(SubtypeConstraint(var, var))
                # T-Prefix
                new_existence |= var.prefixes()
                # S-Pointer
                if var.tail() == LoadLabel.instance():
                    s_path = list(var.path[:-1])
                    s_path.append(StoreLabel.instance())
                    s_var = DerivedTypeVariable(var.base, s_path)
                    store = ExistenceConstraint(s_var)
                    if store in existence:
                        new_subtype.add(SubtypeConstraint(store.var, var))
            for sub_constraint in subtype:
                left = sub_constraint.left
                right = sub_constraint.right
                # T-Left
                new_existence.add(ExistenceConstraint(left))
                # T-Right
                new_existence.add(ExistenceConstraint(right))
                for ex in existence:
                    var = ex.var
                    # T-InheritL
                    l_suffix = var.get_suffix(left)
                    if l_suffix:
                        new_existence.add(ExistenceConstraint(right.add_suffix(l_suffix)))
                    r_suffix = var.get_suffix(right)
                    if r_suffix:
                        l_with_suffix = left.add_suffix(r_suffix)
                        # T-InheritR
                        new_existence.add(ExistenceConstraint(l_with_suffix))
                        if DerivedTypeVariable.suffix_is_covariant(r_suffix):
                            # S-Field⊕
                            forwards = SubtypeConstraint(l_with_suffix, var)
                            new_subtype.add(forwards)
                        else:
                            # S-Field⊖
                            backwards = SubtypeConstraint(var, l_with_suffix)
                            new_subtype.add(backwards)
                for sub in subtype:
                    # S-Trans
                    if sub.left == right:
                        new_subtype.add(SubtypeConstraint(left, sub.right))
        return ConstraintSet(existence, subtype)

    def __str__(self) -> str:
        nt = os.linesep + '\t'
        return (f'ConstraintSet:{nt}{nt.join(map(str, self.existence))}'
                f'{os.linesep}{nt}{nt.join(map(str,self.subtype))}')


def run_basic_tests():
    '''Run a suite of simple tests.
    '''
    unfixed = ConstraintSet()
    unfixed.add_existence(DerivedTypeVariable('a', [LoadLabel.instance()]))
    unfixed.add_existence(DerivedTypeVariable('a', [StoreLabel.instance()]))
    print(unfixed)
    fixed = unfixed.fix()
    print(fixed)

if __name__ == '__main__':
    run_basic_tests()
