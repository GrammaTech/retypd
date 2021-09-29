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
    Field,
)
from .loggable import Loggable
from typing import Set, Dict
from collections import defaultdict


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
                 verbose: int = 0):
        super(CTypeGenerator, self).__init__(verbose)
        self.default_int_size = default_int_size
        self.sketch_map = sketch_map
        self.struct_types = {}
        self.dtv2type = defaultdict(dict)
        self.lattice_ctypes = lattice_ctypes

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
        return successors

    def pick_best(self, *candidates):
        "When a specific offset has multiple candidate atomic types, pick the best one"
        for t in candidates:
            if t is not None:
                return t
        return candidates[0]

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

    def c_type_from_node(self, base_dtv: DerivedTypeVariable, sketches, n, depth=2):
        n = self.resolve_label(sketches, n)
        assert n is not None

        # Check cache
        dtv = n.dtv
        if dtv in self.dtv2type[base_dtv]:
            return self.dtv2type[base_dtv][dtv]

        children = self._succ_no_loadstore(base_dtv, sketches, n, set())
        if len(children) == 0:
            tail = n.dtv.tail
            if tail is not None and isinstance(tail, DerefLabel):
                byte_size = tail.size
            else:
                byte_size = self.default_int_size
            rv = self.lattice_ctypes.atom_to_ctype(n.atom, byte_size)
            self.dtv2type[base_dtv][dtv] = rv
        else:
            # We could recurse on types below, so we populate the struct _first_
            s = StructType()
            self.struct_types[s.name] = s
            rv = PointerType(s)
            self.dtv2type[base_dtv][dtv] = rv

            fields = {}
            for c in children:
                tail = c.dtv.tail
                if tail is None:
                    continue
                if isinstance(tail, DerefLabel):
                    child_type = self.c_type_from_node(base_dtv, sketches, c, depth+2)
                    if c.source:
                        child_type = self.examine_sources(c.source, c, child_type)
                    prev_field = fields.get(tail.offset, Field(None)).ctype
                    fields[tail.offset] = Field(self.pick_best(child_type, prev_field),
                                                offset=tail.offset)
                else:
                    raise CTypeGenerationError(f"Unexpected child: {tail}")
            s.set_fields(fields=list(fields.values()))
        return rv

    def _simplify_pointers(self, typ: CType, seen_structs: Set[CType]):
        """
        Look for all Pointer(Struct(FieldType)) patterns where the struct has a single field at
        offset = 0 and convert it to Pointer(FieldType).
        """
        if isinstance(typ, Field):
            return Field(self._simplify_pointers(typ.ctype, seen_structs), typ.offset)
        elif isinstance(typ, ArrayType):
            return ArrayType(self._simplify_pointers(typ.member_type, seen_structs))
        elif isinstance(typ, PointerType):
            if isinstance(typ.target_type, StructType):
                s = typ.target_type
                if len(s.fields) == 1 and s.fields[0].offset == 0:
                    rv = PointerType(self._simplify_pointers(s.fields[0].ctype, seen_structs))
                    self.info("Simplified pointer: %s", rv)
                    return rv
            return PointerType(self._simplify_pointers(typ.target_type, seen_structs))
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

    def __call__(self, simplify_pointers=True):
        """
        Generate CTypes.

        :param simplify_pointers: By default pointers to single-field structs are simplified to
            just be pointers to the base type of the first field. Set this to False to keep types
            normalized to always use structs to contain pointed-to data.
        """
        dtv_to_type = {}
        for base_dtv, sketches in self.sketch_map.items():
            node = sketches.lookup.get(base_dtv)
            if node is None:
                continue
            # First, see if it is a function
            params = []
            rtype = None
            for succ in self._succ_no_loadstore(base_dtv, sketches, node, set()):
                assert isinstance(succ, SketchNode)
                if isinstance(succ.dtv.tail, InLabel):
                    p = self.c_type_from_node(base_dtv, sketches, succ)
                    params.append(p)
                elif isinstance(succ.dtv.tail, OutLabel):
                    assert (rtype is None)
                    rtype = self.c_type_from_node(base_dtv, sketches, succ)
            # Not a function
            if rtype is None and not params:
                dtv_to_type[base_dtv] = self.c_type_from_node(base_dtv, sketches, node)
            else:
                dtv_to_type[base_dtv] = FunctionType(rtype, params, name=str(base_dtv))

        if simplify_pointers:
            for typ in dtv_to_type.values():
                self._simplify_pointers(typ, set())

        return dtv_to_type

