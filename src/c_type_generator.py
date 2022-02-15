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
    InLabel,
    OutLabel,
    LoadLabel,
    StoreLabel,
    DerefLabel,
    DerivedTypeVariable,
    LatticeCTypes,
)
from .solver import SketchNode, Sketches, LabelNode
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
from typing import Set, Dict, Optional
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
    def __init__(self,
                 sketch_map: Dict[DerivedTypeVariable, Sketches],
                 lattice_ctypes: LatticeCTypes,
                 default_int_size: int,
                 default_ptr_size: int,
                 verbose: LogLevel = LogLevel.QUIET):
        super(CTypeGenerator, self).__init__(verbose)
        self.default_int_size = default_int_size
        self.default_ptr_size = default_ptr_size
        self.sketch_map = sketch_map
        self.struct_types = {}
        self.dtv2type = defaultdict(dict)
        self.lattice_ctypes = lattice_ctypes

    def union_types(self, a: CType, b: CType):
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
                if (
                    a.length == b.length
                    and self.union_types(am, bm) in (am, bm)
                ):
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

    def resolve_label(self, sketches, node):
        if isinstance(node, LabelNode):
            self.info("Resolved label: %s", node)
            return sketches.lookup.get(node.target)
        return node

    def _succ_no_loadstore(self, base_dtv, sketches, node, seen):
        successors = []
        if node not in seen:
            seen.add(node)
            for n in sketches.sketches.successors(node):
                n = self.resolve_label(sketches, n)
                if n is None:
                    continue
                if n.dtv.tail in (StoreLabel.instance(), LoadLabel.instance()):
                    successors.extend(self._succ_no_loadstore(base_dtv, sketches, n, seen))
                else:
                    successors.append(n)
        self.debug("Successors %s --> %s", node, successors)
        return successors

    def examine_sources(self, sources: Set[SketchNode], my_node: SketchNode, my_type: CType):
        """
        Callback that is invoked on every C type creation on a SketchNode that was generated from
        other nodes (e.g., callee information). This gives the client code an opportunity to see
        where the type came from and do unification if they want.

        :param sources: Set of SketchNodes (from other SCCs in the callgraph or global dependence
            graph) that contributed to the current CType.
        :param my_node: The current SketchNode that we are resolving.
        :param my_type: The current CType for the SketchNode (i.e., the type derived directly from
            the sketchnode).
        :returns: The CType to use.
        """
        return my_type

    def c_type_from_nodeset(self,
                            base_dtv: DerivedTypeVariable,
                            sketches: Sketches,
                            ns: Set[SketchNode]) -> Optional[CType]:
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

        children = list(itertools.chain(
            *[self._succ_no_loadstore(base_dtv, sketches, n, set()) for n in ns]
        ))
        if len(children) == 0:
            rv = None
            for n in ns:
                tail = n.dtv.tail
                if tail is not None and isinstance(tail, DerefLabel):
                    byte_size = tail.size
                else:
                    byte_size = self.default_int_size
                ntype = self.lattice_ctypes.atom_to_ctype(n.get_usable_type(), byte_size)
                rv = self.union_types(rv, ntype)
            for n in ns:
                self.dtv2type[base_dtv][n.dtv] = rv
            self.debug("Terminal type: %s -> %s", ns, rv)
        else:
            # We could recurse on types below, so we populate the struct _first_
            s = StructType()
            self.struct_types[s.name] = s
            rv = PointerType(s, self.default_ptr_size)
            for n in ns:
                self.dtv2type[base_dtv][n.dtv] = rv

            self.debug("%s has %d children", ns, len(children))
            # XXX this is aggressively merging different edges for the same deref offset, no
            # matter the atom (TOP, BOTTOM, lattice type, ...)
            children_by_offset = defaultdict(set)
            for c in children:
                tail = c.dtv.tail
                if not isinstance(tail, DerefLabel):
                    print(f"WARNING: {c.dtv} does not end in DerefLabel")
                    continue
                #assert isinstance(tail, DerefLabel)
                children_by_offset[tail.offset].add(c)

            fields = []
            for offset, siblings in children_by_offset.items():
                child_type = self.c_type_from_nodeset(base_dtv, sketches, siblings)
                # TODO
                #if c.source:
                #    child_type = self.examine_sources(c.source, c, child_type)
                fields.append(Field(child_type, offset=offset))
            s.set_fields(fields=fields)
        return rv

    def _simplify_pointers(self, typ: CType, seen_structs: Set[CType]):
        """
        Look for all Pointer(Struct(FieldType)) patterns where the struct has a single field at
        offset = 0 and convert it to Pointer(FieldType).
        """
        if isinstance(typ, Field):
            return Field(self._simplify_pointers(typ.ctype, seen_structs), typ.offset)
        elif isinstance(typ, ArrayType):
            return ArrayType(self._simplify_pointers(typ.member_type, seen_structs), typ.length)
        elif isinstance(typ, PointerType):
            if isinstance(typ.target_type, StructType):
                s = typ.target_type
                if len(s.fields) == 1 and s.fields[0].offset == 0:
                    rv = PointerType(
                        self._simplify_pointers(s.fields[0].ctype, seen_structs),
                        self.default_ptr_size)
                    self.info("Simplified pointer: %s", rv)
                    return rv
            return PointerType(
                self._simplify_pointers(typ.target_type, seen_structs),
                self.default_ptr_size)
        elif isinstance(typ, FunctionType):
            params = [self._simplify_pointers(t, seen_structs) for t in typ.params]
            rt = self._simplify_pointers(typ.return_type, seen_structs)
            return FunctionType(rt, params, name=typ.name)
        elif isinstance(typ, StructType):
            if typ in seen_structs:
                return typ
            seen_structs.add(typ)
            s = StructType(name=typ.name)
            s.set_fields([self._simplify_pointers(t, seen_structs) for t in typ.fields])
            return s
        return typ

    def __call__(self, simplify_pointers=True, filter_to: Optional[Set[DerivedTypeVariable]] = None):
        """
        Generate CTypes.

        :param simplify_pointers: By default pointers to single-field structs are simplified to
            just be pointers to the base type of the first field. Set this to False to keep types
            normalized to always use structs to contain pointed-to data.
        :param filter_to: If specified, only emit types for the given base DerivedTypeVariables
            (typically globals or functions). If None (default), emit all types.
        """
        dtv_to_type = {}
        for base_dtv, sketches in self.sketch_map.items():
            node = sketches.lookup.get(base_dtv)
            if node is None:
                continue
            if filter_to is not None and base_dtv not in filter_to:
                continue
            # First, see if it is a function
            params = []
            rtype = None
            is_func = False
            for succ in self._succ_no_loadstore(base_dtv, sketches, node, set()):
                assert isinstance(succ, SketchNode)
                if isinstance(succ.dtv.tail, InLabel):
                    self.debug("(1) Processing %s", succ.dtv)
                    p = self.c_type_from_nodeset(base_dtv, sketches, {succ})
                    params.append(p)
                    is_func = True
                elif isinstance(succ.dtv.tail, OutLabel):
                    self.debug("(2) Processing %s", succ.dtv)
                    assert (rtype is None)
                    rtype = self.c_type_from_nodeset(base_dtv, sketches, {succ})
                    is_func = True
            # Not a function
            if is_func:
                dtv_to_type[base_dtv] = FunctionType(rtype, params, name=str(base_dtv))
            else:
                self.debug("(3) Processing %s", base_dtv)
                dtv_to_type[base_dtv] = self.c_type_from_nodeset(base_dtv, sketches, {node})

        if simplify_pointers:
            self.debug("Simplifying pointers")
            new_dtv_to_type = {}
            for dtv,typ in dtv_to_type.items():
                new_dtv_to_type[dtv] = self._simplify_pointers(typ, set())
            dtv_to_type = new_dtv_to_type

        return dtv_to_type

