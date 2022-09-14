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

from .schema import (
    AccessPathLabel,
    InLabel,
    OutLabel,
    LoadLabel,
    StoreLabel,
    DerefLabel,
    DerivedTypeVariable,
    LatticeCTypes,
    Lattice,
)
from .sketches import (
    SketchNode,
    Sketches,
    LabelNode,
    SkNode,
)
from .c_types import (
    CType,
    PointerType,
    FunctionType,
    ArrayType,
    StructType,
    IntType,
    UnionType,
    Field,
    FloatType,
    CharType,
)
from .loggable import Loggable, LogLevel
from typing import Set, Dict, Optional, List, Tuple
from collections import defaultdict
import itertools


class CTypeGenerationError(Exception):
    """
    Exception raised when an unexpected situation occurs during CType generation.
    """

    pass


class CTypeGenerator(Loggable):
    """
    Generate C-like types from sketches.
    """

    def __init__(
        self,
        sketch_map: Dict[DerivedTypeVariable, Sketches],
        lattice: Lattice,
        lattice_ctypes: LatticeCTypes,
        default_int_size: int,
        default_ptr_size: int,
        verbose: LogLevel = LogLevel.QUIET,
    ):
        super(CTypeGenerator, self).__init__(verbose)
        self.default_int_size = default_int_size
        self.default_ptr_size = default_ptr_size
        self.sketch_map = sketch_map
        self.struct_types = {}
        self.dtv2type = defaultdict(dict)
        self.lattice = lattice
        self.lattice_ctypes = lattice_ctypes

    def union_types(
        self, a: Optional[CType], b: Optional[CType]
    ) -> Optional[CType]:
        """
        This function decides how to merge two CTypes for the same access path. This differs from
        the lattice, which only considers "atomic" or "terminal" types. This can take, for example,
        two pointers to structs. Or an integer and a pointer to a struct.
        """
        if a is None:
            return b
        if b is None:
            return a
        at = type(a)
        bt = type(b)
        if at == IntType and bt in (PointerType, StructType, ArrayType):
            return b
        if bt == IntType and at in (PointerType, StructType, ArrayType):
            return a
        if at == bt:
            if at == IntType:
                if a.width == b.width:
                    return IntType(a.width, a.signed or b.signed)
            elif at in (FloatType, CharType) and a.width == b.width:
                return a
            elif at == ArrayType:
                am = a.member_type
                bm = b.member_type
                if a.length == b.length and self.union_types(am, bm) in (
                    am,
                    bm,
                ):
                    return a
            elif at == PointerType:
                ap = a.target_type
                bp = b.target_type

                if self.union_types(ap, bp) in (ap, bp):
                    return a

        unioned_types = []
        if at == UnionType:
            unioned_types.extend(a.fields)
        else:
            unioned_types.append(Field(a))
        if bt == UnionType:
            unioned_types.extend(b.fields)
        else:
            unioned_types.append(Field(b))
        self.debug("Unioning: %s", unioned_types)
        return UnionType(unioned_types)

    def resolve_label(self, sketches: Sketches, node: SkNode) -> SketchNode:
        if isinstance(node, LabelNode):
            self.info("Resolved label: %s", node)
            return sketches.lookup(node.target)
        return node

    def _succ_no_loadstore(
        self,
        curr_access_path: List[AccessPathLabel],
        sketches: Sketches,
        node: SketchNode,
        seen: Set[SketchNode],
    ) -> List[Tuple[List[AccessPathLabel], SketchNode]]:
        """
        Collect a list of SketchNode successors of `node`
        in the sketch graph `sketches` by traversing load and store
        labels. Each `SketchNode` successor is accompanied
        by the access path that leads to it.
        The access path will coincide with the node's DTV unless
        LabelNodes have been traversed.
        """
        successors = []
        if node not in seen:
            seen.add(node)
            for n in sketches.sketches.successors(node):
                new_label = sketches.sketches[node][n]["label"]
                n = self.resolve_label(sketches, n)
                if n is None:
                    continue
                if n.dtv.tail in (StoreLabel.instance(), LoadLabel.instance()):
                    successors.extend(
                        self._succ_no_loadstore(
                            curr_access_path + [new_label], sketches, n, seen
                        )
                    )
                else:
                    successors.append((curr_access_path + [new_label], n))
        self.debug("Successors %s --> %s", node, successors)
        return successors

    def merge_counts(self, count_set: Set[int]) -> int:
        """
        Given a set of element counts from a node, merge them into a single count.
        """
        if not count_set:
            return 1
        elif len(count_set) == 1:
            return list(count_set)[0]
        elif DerefLabel.COUNT_NULLTERM in count_set:
            return DerefLabel.COUNT_NULLTERM
        return DerefLabel.COUNT_NOBOUND

    def c_type_from_nodeset(
        self,
        base_dtv: DerivedTypeVariable,
        sketches: Sketches,
        ns: Set[SketchNode],
    ) -> Optional[CType]:
        """
        Given a derived type var, sketches, and a set of nodes, produce a C-like type. The
        set of nodes must be all for the same access path (e.g., foo[0]), but can be different
        _ways_ of accessing it (e.g., foo.load.s8@0 and foo.store.s8@0).
        """
        ns = set([self.resolve_label(sketches, n) for n in ns])
        assert None not in ns

        # Check cache
        for n in ns:
            if n.dtv in self.dtv2type[base_dtv]:
                self.info("Already cached (recursive type): %s", n.dtv)
                return self.dtv2type[base_dtv][n.dtv]

        children = list(
            itertools.chain(
                *[self._succ_no_loadstore([], sketches, n, set()) for n in ns]
            )
        )
        available_tails = {access_paths[-1] for access_paths, _ in children}

        if len(children) == 0:
            # Compute the atomic type bounds and size bound
            lb = self.lattice.bottom
            ub = self.lattice.top
            sz = 0
            counts = set()
            for n in ns:
                tail = n.dtv.tail
                if tail is not None and isinstance(tail, DerefLabel):
                    byte_size = tail.size
                    counts.add(tail.count)
                else:
                    byte_size = self.default_int_size
                lb = self.lattice.join(lb, n.lower_bound)
                ub = self.lattice.meet(ub, n.upper_bound)
                sz = max(sz, byte_size)

            # Convert it to a CType
            rv = self.lattice_ctypes.atom_to_ctype(lb, ub, sz)
            count = self.merge_counts(counts)
            if count > 1:
                rv = ArrayType(rv, count)
            elif count == DerefLabel.COUNT_NULLTERM:
                # C type for null terminated string is [w]char[_t]*
                if rv.size in (1, 2, 4):  # Valid character sizes
                    rv = CharType(rv.size)
                else:
                    self.info(
                        "Unexpected character size for null-terminated string: %d",
                        rv.size,
                    )
            # In C, unbounded arrays are represented as pointers, which is what this deref
            # will be represented as default. XXX in future we could change ArrayType to allow
            # for representing unboundedness.

            for n in ns:
                self.dtv2type[base_dtv][n.dtv] = rv
            self.debug("Terminal type: %s -> %s", ns, rv)
        elif all(
            isinstance(tail, (InLabel, OutLabel)) for tail in available_tails
        ):
            outputs = set()
            inputs = defaultdict(set)

            ptr_func = PointerType(
                FunctionType(None, []), self.default_ptr_size
            )

            for n in ns:
                self.dtv2type[base_dtv][n.dtv] = ptr_func

            for access_path, child in children:
                tail = access_path[-1]

                if isinstance(tail, OutLabel):
                    outputs.add(child)
                elif isinstance(tail, InLabel):
                    inputs[tail.index].add(child)
                else:
                    assert False, "Unreachable type generation state"

            ptr_func.target_type.return_type = self.c_type_from_nodeset(
                base_dtv, sketches, outputs
            )
            ptr_func.target_type.params = [
                self.c_type_from_nodeset(base_dtv, sketches, inputs[index])
                if len(inputs[index]) > 0
                else self.lattice_ctypes.atom_to_ctype(
                    self.lattice.bottom,
                    self.lattice.top,
                    self.default_int_size,
                )
                for index in range(max(inputs.keys(), default=-1) + 1)
            ]

            return ptr_func
        else:
            # We could recurse on types below, so we populate the struct _first_
            s = StructType()
            self.struct_types[s.name] = s
            count = self.merge_counts(
                [
                    n.dtv.tail.count
                    for n in ns
                    if isinstance(n.dtv.tail, DerefLabel)
                ]
            )
            if count > 1:
                rv = ArrayType(s, count)
            else:
                rv = PointerType(s, self.default_ptr_size)
            for n in ns:
                self.dtv2type[base_dtv][n.dtv] = rv

            self.debug("%s has %d children", ns, len(children))
            children_by_offset = defaultdict(set)
            for access_path, c in children:
                tail = access_path[-1]
                if not isinstance(tail, DerefLabel):
                    self.info(f"WARNING: {c.dtv} does not end in DerefLabel")
                    continue
                children_by_offset[tail.offset].add(c)

            fields = []
            for offset, siblings in children_by_offset.items():
                child_type = self.c_type_from_nodeset(
                    base_dtv, sketches, siblings
                )
                fields.append(Field(child_type, offset=offset))
            s.set_fields(fields=fields)
        return rv

    def _simplify_pointers(self, typ: CType, seen: Set[CType]) -> CType:
        """
        Look for all Pointer(Struct(FieldType)) patterns where the struct has a single field at
        offset = 0 and convert it to Pointer(FieldType).
        """
        if typ in seen:
            return typ
        seen.add(typ)
        if isinstance(typ, Field):
            return Field(self._simplify_pointers(typ.ctype, seen), typ.offset)
        elif isinstance(typ, ArrayType):
            return ArrayType(
                self._simplify_pointers(typ.member_type, seen), typ.length
            )
        elif isinstance(typ, PointerType):
            if isinstance(typ.target_type, StructType):
                s = typ.target_type
                if len(s.fields) == 1 and s.fields[0].offset == 0:
                    rv = PointerType(
                        self._simplify_pointers(s.fields[0].ctype, seen),
                        self.default_ptr_size,
                    )
                    self.info("Simplified pointer: %s", rv)
                    return rv
            return PointerType(
                self._simplify_pointers(typ.target_type, seen),
                self.default_ptr_size,
            )
        elif isinstance(typ, FunctionType):
            params = [self._simplify_pointers(t, seen) for t in typ.params]
            rt = self._simplify_pointers(typ.return_type, seen)
            return FunctionType(rt, params, name=typ.name)
        elif isinstance(typ, StructType):
            s = StructType(name=typ.name)
            s.set_fields(
                [self._simplify_pointers(t, seen) for t in typ.fields]
            )
            return s
        return typ

    def __call__(
        self,
        simplify_pointers: bool = True,
        filter_to: Optional[Set[DerivedTypeVariable]] = None,
    ) -> Dict[DerivedTypeVariable, CType]:
        """
        Generate CTypes.

        :param simplify_pointers: By default pointers to single-field structs are simplified to
            just be pointers to the base type of the first field. Set this to False to keep types
            normalized to always use structs to contain pointed-to data.
        :param filter_to: If specified, only emit types for the given base DerivedTypeVariables
            (typically globals or functions). If None (default), emit all types.
        """
        dtv_to_type = {}
        for dtv, sketches in self.sketch_map.items():
            if filter_to is not None and dtv not in filter_to:
                continue

            node = sketches.lookup(dtv)

            if node is None:
                continue

            maybe_ptr_func = self.c_type_from_nodeset(dtv, sketches, {node})

            if isinstance(maybe_ptr_func, PointerType) and isinstance(
                maybe_ptr_func.target_type, FunctionType
            ):
                # Consumers expect the function itself, not a pointer to it
                dtv_to_type[dtv] = maybe_ptr_func.target_type
            else:
                dtv_to_type[dtv] = maybe_ptr_func

        if simplify_pointers:
            self.debug("Simplifying pointers")
            new_dtv_to_type = {}
            for dtv, typ in dtv_to_type.items():
                new_dtv_to_type[dtv] = self._simplify_pointers(typ, set())
            dtv_to_type = new_dtv_to_type

        return dtv_to_type
