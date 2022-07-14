from .schema import DerivedTypeVariable, Lattice, LatticeCTypes
from .c_types import (
    ArrayType,
    BoolType,
    CharType,
    FloatType,
    IntType,
    VoidType,
)
from typing import FrozenSet
import networkx


class CLattice(Lattice[DerivedTypeVariable]):
    INT_SIZES = [8, 16, 32, 64]

    # Unsized C integers
    _int = DerivedTypeVariable("int")
    _int_size = [DerivedTypeVariable(f"int{z}") for z in INT_SIZES]
    _uint = DerivedTypeVariable("uint")
    _uint_size = [DerivedTypeVariable(f"uint{z}") for z in INT_SIZES]

    # Floats
    _float = DerivedTypeVariable("float")
    _double = DerivedTypeVariable("double")

    # Special types
    _void = DerivedTypeVariable("void")
    _char = DerivedTypeVariable("char")
    _bool = DerivedTypeVariable("bool")

    _top = DerivedTypeVariable("┬")
    _bottom = DerivedTypeVariable("┴")

    _internal = (
        frozenset(
            {
                _int,
                _uint,
                _float,
                _double,
                _void,
                _char,
                _bool,
            }
        )
        | frozenset(_int_size)
        | frozenset(_uint_size)
    )
    _endcaps = frozenset({_top, _bottom})

    def __init__(self) -> None:
        self.graph = networkx.DiGraph()
        self.graph.add_edge(self._uint, self.top)

        for dtv in self._uint_size:
            self.graph.add_edge(dtv, self._uint)

        self.graph.add_edge(self._int, self._uint)

        for dtv in self._int_size:
            self.graph.add_edge(dtv, self._int)

        for int_dtv, uint_dtv in zip(self._int_size, self._uint_size):
            self.graph.add_edge(int_dtv, uint_dtv)

        self.graph.add_edge(self._float, self.top)
        self.graph.add_edge(self._double, self.top)

        # char is a int8_t with some semantic information. NOTE: This assumes
        # that INT_SIZES[0] == 8
        self.graph.add_edge(self._char, self._int_size[0])
        self.graph.add_edge(self._void, self.top)
        self.graph.add_edge(self._bool, self.top)

        for dtv in self._int_size:
            self.graph.add_edge(self._bottom, dtv)

        self.graph.add_edge(self._bottom, self._int)
        self.graph.add_edge(self._bottom, self._float)
        self.graph.add_edge(self._bottom, self._double)
        self.graph.add_edge(self._bottom, self._void)
        self.graph.add_edge(self._bottom, self._char)
        self.graph.add_edge(self._bottom, self._bool)

        assert all(
            len(self.graph.out_edges(dtv)) > 0
            and len(self.graph.in_edges(dtv)) > 0
            for dtv in self._internal
        )

        try:
            networkx.find_cycle(self.graph)
            assert False, "Lattice cannot be circular"
        except networkx.NetworkXNoCycle:
            pass

        self.revgraph = self.graph.reverse()

    @property
    def atomic_types(self) -> FrozenSet[DerivedTypeVariable]:
        return CLattice._internal | CLattice._endcaps

    @property
    def internal_types(self) -> FrozenSet[DerivedTypeVariable]:
        return CLattice._internal

    @property
    def top(self) -> DerivedTypeVariable:
        return CLattice._top

    @property
    def bottom(self) -> DerivedTypeVariable:
        return CLattice._bottom

    def meet(
        self, t: DerivedTypeVariable, v: DerivedTypeVariable
    ) -> DerivedTypeVariable:
        return networkx.lowest_common_ancestor(self.graph, t, v)

    def join(
        self, t: DerivedTypeVariable, v: DerivedTypeVariable
    ) -> DerivedTypeVariable:
        return networkx.lowest_common_ancestor(self.revgraph, t, v)


class CLatticeCTypes(LatticeCTypes):
    def atom_to_ctype(self, lower_bound, upper_bound, byte_size):
        if upper_bound == CLattice._top:
            atom = lower_bound
        elif lower_bound == CLattice._bottom:
            atom = upper_bound
        else:
            atom = lower_bound

        if atom in CLattice._int_size:
            return IntType(
                CLattice.INT_SIZES[CLattice._int_size.index(atom)] // 8, True
            )

        if atom in CLattice._uint_size:
            return IntType(
                CLattice.INT_SIZES[CLattice._uint_size.index(atom)] // 8, False
            )

        default = ArrayType(CharType(1), byte_size)

        return {
            CLattice._int: IntType(byte_size, True),
            CLattice._uint: IntType(byte_size, False),
            CLattice._void: VoidType(),
            CLattice._char: CharType(byte_size),
            CLattice._float: FloatType(4),
            CLattice._bool: BoolType(byte_size),
            CLattice._double: FloatType(8),
        }.get(atom, default)
