
from .schema import DerivedTypeVariable

from typing import Set, Union

class SketchNode:
    def __init__(self, dtv: DerivedTypeVariable,
                 lower_bound: DerivedTypeVariable,
                 upper_bound: DerivedTypeVariable) -> None:
        self._dtv = dtv
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        # Reference to SketchNodes (in other SCCs) that this node came from
        self.source: Set[SketchNode] = set()
        self._hash = hash(self._dtv)

    @property
    def dtv(self):
        return self._dtv

    @dtv.setter
    def dtv(self, value):
        raise NotImplementedError("Read-only property")

    # the atomic type of a DTV is an annotation, not part of its identity
    def __eq__(self, other) -> bool:
        if isinstance(other, SketchNode):
            return self.dtv == other.dtv
        return False

    def __hash__(self) -> int:
        return self._hash

    def __str__(self) -> str:
        return f'({self.lower_bound} <= {self.dtv} <= {self.upper_bound})'

    def __repr__(self) -> str:
        return f'SketchNode({self})'

    def get_usable_type(self) -> DerivedTypeVariable:
        # TODO this might not always be the right solution
        return self.lower_bound


class LabelNode:
    counter = 0
    def __init__(self, target: DerivedTypeVariable) -> None:
        self.target = target
        self.id = LabelNode.counter
        LabelNode.counter += 1

    def __eq__(self, other) -> bool:
        if isinstance(other, LabelNode):
            # return self.id == other.id and self.target == other.target
            return self.target == other.target
        return False

    def __hash__(self) -> int:
        # return hash(self.target) ^ hash(self.id)
        return hash(self.target)

    def __str__(self) -> str:
        return f'{self.target}.label_{self.id}'

    def __repr__(self) -> str:
        return str(self)


SkNode = Union[SketchNode, LabelNode]