from __future__ import annotations
from .schema import (
    DerivedTypeVariable,
    Lattice,
    FreshVarFactory,
    ConstraintSet,
    SubtypeConstraint,
    Variance,
    RetypdError,
)
from .loggable import Loggable, LogLevel
import os
import networkx
from typing import Set, Union, Optional, Tuple, Dict


class SketchNode:
    """
    A node in a sketch graph. The node is associated with a DTV and it
    captures its upper and lower bound in the type lattice.
    A sketch node might be referenced by `LabelNode` in recursive types.
    If that is the case, the sketch node can represent the primitive
    type infinite DTVs, e.g.:
    f.in_0, f.in_0.load.σ4@0, f.in_0.load.σ4@0.load.σ4@0, ...
    """

    def __init__(
        self,
        dtv: DerivedTypeVariable,
        lower_bound: DerivedTypeVariable,
        upper_bound: DerivedTypeVariable,
    ) -> None:
        self._dtv = dtv
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self._hash = hash(self._dtv)

    @property
    def dtv(self):
        """
        The main DTV represented by the sketch node.
        """
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
        return f"({self.lower_bound} <= {self.dtv} <= {self.upper_bound})"

    def __repr__(self) -> str:
        return f"SketchNode({self})"


class LabelNode:
    """
    LableNodes are used to capture cycles in sketches
    (recursive types). A LabelNode has a target that
    is a DTV, which uniquely identifies the SketchNode
    that it points to.
    There can be multiple LabelNodes pointing to the
    same sketch node, since a type can have multiple
    recursive references (e.g., 'previous' and
    'next' references in a doubly linked list).
    """

    counter = 0

    def __init__(self, target: DerivedTypeVariable) -> None:
        self.target = target
        self.id = LabelNode.counter
        LabelNode.counter += 1

    def __eq__(self, other) -> bool:
        if isinstance(other, LabelNode):
            return self.id == other.id and self.target == other.target
        return False

    def __hash__(self) -> int:
        return hash(self.target) ^ hash(self.id)

    def __str__(self) -> str:
        return f"{self.target}.label_{self.id}"

    def __repr__(self) -> str:
        return str(self)


SkNode = Union[SketchNode, LabelNode]


class Sketches(Loggable):
    """The set of sketches from a set of constraints. Intended to be updated incrementally, per the
    Solver's reverse topological ordering.
    """

    def __init__(
        self,
        types: Lattice[DerivedTypeVariable],
        verbose: LogLevel = LogLevel.QUIET,
    ) -> None:
        super(Sketches, self).__init__(verbose)
        # We maintain the invariant that if a node is in `lookup` then it should also be in
        # `sketches` as a node (even if there are no edges)
        self.sketches = networkx.DiGraph()
        self._lookup: Dict[DerivedTypeVariable, SketchNode] = {}
        self.types = types

    def lookup(self, dtv: DerivedTypeVariable) -> Optional[SketchNode]:
        """
        Return the sketch node corresponding to the path
        represented in the given dtv
        """
        if dtv in self._lookup:
            return self._lookup[dtv]
        # if it is not in the dictionary we traverse the graph
        beg = dtv.base_var
        curr_node = self._lookup.get(beg)
        if curr_node is None:
            return None
        for access_path in dtv.path:
            succs = [
                dest
                for (_, dest, label) in self.sketches.out_edges(
                    curr_node, data="label"
                )
                if label == access_path
            ]
            if len(succs) == 0:
                return None
            elif len(succs) > 1:
                raise ValueError(
                    f"{curr_node} has multiple successors in sketches"
                )
            curr_node = succs[0]
            if isinstance(curr_node, LabelNode):
                curr_node = self._lookup[curr_node.target]
        return curr_node

    def _add_node(self, node: SketchNode) -> None:
        """
        Add node to the sketch graph
        """
        self._lookup[node.dtv] = node
        self.sketches.add_node(node)

    def ref_node(self, node: SkNode) -> SkNode:
        """Add a reference to the given node (no copy)"""
        if isinstance(node, LabelNode):
            return node
        if node.dtv in self._lookup:
            return self._lookup[node.dtv]
        self._add_node(node)
        return node

    def make_node(
        self,
        variable: DerivedTypeVariable,
        *,
        lower_bound: Optional[DerivedTypeVariable] = None,
        upper_bound: Optional[DerivedTypeVariable] = None,
    ) -> SketchNode:
        """Make a node from a DTV. Compute its atom from its access path."""

        lower_bound = lower_bound or self.types.bottom
        upper_bound = upper_bound or self.types.top
        result = SketchNode(variable, lower_bound, upper_bound)
        self._add_node(result)
        return result

    def add_edge(self, head: SketchNode, tail: SkNode, label: str) -> None:
        """
        Add edge labeled with `label` in the sketch graph between `head`
        and `tail`.
        """
        # don't emit duplicate edges
        if (head, tail) not in self.sketches.edges:
            self.sketches.add_edge(head, tail, label=label)
        else:
            if label != self.sketches.edges[head][tail]["label"]:
                raise RetypdError(
                    f"Failed to add edge {label} between {head} and {tail}."
                    f" Label {self.sketches.edges[head][tail]['label']} exists"
                )

    def _copy_global_recursive(
        self, node: SketchNode, sketches: Sketches
    ) -> SketchNode:
        """
        Auxiliary method to recusively copy a sketch tree
        rooted in `node` from `sketches` to `self`.
        """
        # TODO: this probably needs to handle atoms properly, using the Lattice. Needs
        # some thought.
        our_node = self.ref_node(node)
        if node in sketches.sketches.nodes:
            for _, dst in sketches.sketches.out_edges(node):
                our_dst = dst
                if not isinstance(dst, LabelNode):
                    our_dst = self._copy_global_recursive(dst, sketches)
                self.add_edge(
                    our_node, our_dst, sketches.sketches[node][dst]["label"]
                )
        return our_node

    def copy_globals_from_sketch(
        self, global_vars: Set[DerivedTypeVariable], sketches: Sketches
    ):
        """
        Copy the sketch trees of global variables from `sketches` to `self`.
        """
        global_roots = set()
        for dtv, node in sketches._lookup.items():
            if dtv.base_var in global_vars:
                global_roots.add(dtv.base_var)
        for g in global_roots:
            node = sketches.lookup(g)
            if node is None:
                continue
            self._copy_global_recursive(node, sketches)

    def instantiate_sketch_capabilities(
        self,
        proc: DerivedTypeVariable,
        types: Lattice[DerivedTypeVariable],
        fresh_var_factory: FreshVarFactory,
    ) -> ConstraintSet:
        """
        Encode all the capability information present in the sketch
        using fake variables.
        """
        all_constraints = ConstraintSet()
        for node in self.sketches.nodes:
            if isinstance(node, SketchNode) and node.dtv.base_var == proc:
                constraints = []
                # if the node is a leaf, capture the capability using fake variables
                # this could be avoided if we support capability constraints  (Var x.l) in
                # addition to subtype constraints
                if next(self.sketches.successors(node), None) is None:
                    fresh_var = fresh_var_factory.fresh_var()
                    if node.dtv.path_variance == Variance.CONTRAVARIANT:
                        constraints.append(
                            SubtypeConstraint(node.dtv, fresh_var)
                        )
                    else:
                        constraints.append(
                            SubtypeConstraint(fresh_var, node.dtv)
                        )
                all_constraints |= ConstraintSet(constraints)
        return all_constraints

    def add_constraints(self, constraints: ConstraintSet) -> None:
        """Extend the set of sketches with the new set of constraints."""

        for constraint in constraints:
            left = constraint.left
            right = constraint.right
            if (
                left in self.types.internal_types
                and right not in self.types.internal_types
            ):
                right_node = self.lookup(right)
                if right_node is None:
                    raise RetypdError(
                        f"Sketch node corresponding to {right} does not exist"
                    )
                self.debug("JOIN: %s, %s", right_node, left)
                right_node.lower_bound = self.types.join(
                    right_node.lower_bound, left
                )
                self.debug("   --> %s", right_node)
            elif (
                right in self.types.internal_types
                and left not in self.types.internal_types
            ):
                left_node = self.lookup(left)
                if left_node is None:
                    raise RetypdError(
                        f"Sketch node corresponding to {left} does not exist"
                    )
                self.debug("MEET: %s, %s", left_node, left)
                left_node.upper_bound = self.types.meet(
                    left_node.upper_bound, right
                )
                self.debug("   --> %s", left_node)

    def to_dot(self, dtv: DerivedTypeVariable) -> str:
        nt = f"{os.linesep}\t"
        graph_str = f"digraph {dtv} {{"
        start = self._lookup[dtv]
        edges_str = ""
        # emit edges and identify nodes
        nodes = {start}
        seen: Set[Tuple[SketchNode, SkNode]] = {(start, start)}
        frontier = {
            (start, succ) for succ in self.sketches.successors(start)
        } - seen
        while frontier:
            new_frontier: Set[Tuple[SketchNode, SkNode]] = set()
            for pred, succ in frontier:
                edges_str += nt
                nodes.add(succ)
                new_frontier |= {
                    (succ, s_s) for s_s in self.sketches.successors(succ)
                }
                edges_str += f'"{pred}" -> "{succ}"'
                edges_str += (
                    f' [label="{self.sketches[pred][succ]["label"]}"];'
                )
            frontier = new_frontier - seen
        # emit nodes
        for node in nodes:
            if isinstance(node, SketchNode):
                if node.dtv == dtv:
                    graph_str += nt
                    graph_str += f'"{node}" [label="{node.dtv}"];'
                elif node.dtv.base_var == dtv:
                    graph_str += nt
                    graph_str += f'"{node}" [label="{node.lower_bound}..{node.upper_bound}"];'
            elif node.target.base_var == dtv:
                graph_str += nt
                graph_str += f'"{node}" [label="{node.target}", shape=house];'
        graph_str += edges_str
        graph_str += f"{os.linesep}}}"
        return graph_str

    def __str__(self) -> str:
        if self._lookup:
            nt = f"{os.linesep}\t"

            def format(k: DerivedTypeVariable) -> str:
                return str(self._lookup[k])

            return f"nodes:{nt}{nt.join(map(format, self._lookup.keys()))})"
        return "no sketches"
