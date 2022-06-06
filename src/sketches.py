from __future__ import annotations
from .schema import (
    DerivedTypeVariable,
    Lattice,
    FreshVarFactory,
    ConstraintSet,
    SubtypeConstraint,
    Variance,
)
from .loggable import Loggable, LogLevel
import os
import networkx
from typing import Set, Union, Optional, Tuple, Dict


class SketchNode:
    def __init__(
        self,
        dtv: DerivedTypeVariable,
        lower_bound: DerivedTypeVariable,
        upper_bound: DerivedTypeVariable,
    ) -> None:
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
        return f"({self.lower_bound} <= {self.dtv} <= {self.upper_bound})"

    def __repr__(self) -> str:
        return f"SketchNode({self})"


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
        self.lookup: Dict[DerivedTypeVariable, SketchNode] = {}
        self.types = types

    def ref_node(self, node: SketchNode) -> SketchNode:
        """Add a reference to the given node (no copy)"""
        if isinstance(node, LabelNode):
            return node
        if node.dtv in self.lookup:
            return self.lookup[node.dtv]
        self.lookup[node.dtv] = node
        self.sketches.add_node(node)
        return node

    def make_node(
        self,
        variable: DerivedTypeVariable,
        *,
        lower_bound: Optional[DerivedTypeVariable] = None,
        upper_bound: Optional[DerivedTypeVariable] = None,
    ) -> SketchNode:
        """Make a node from a DTV. Compute its atom from its access path."""
        if lower_bound is None:
            lower_bound = self.types.bottom
        if upper_bound is None:
            upper_bound = self.types.top
        result = SketchNode(variable, lower_bound, upper_bound)
        self.lookup[variable] = result
        self.sketches.add_node(result)
        return result

    def add_variable(self, variable: DerivedTypeVariable) -> SketchNode:
        """Add a variable and its prefixes to the set of sketches. Each node's atomic type is either
        TOP or BOTTOM, depending on the variance of the variable's access path. If the variable
        already exists, skip it.
        """
        if variable in self.lookup:
            return self.lookup[variable]
        node = self.make_node(variable)
        created = node
        self.sketches.add_node(node)
        prefix = variable.largest_prefix
        while prefix:
            prefix_node = self.make_node(prefix)
            self.sketches.add_edge(prefix_node, node, label=variable.tail)
            variable = prefix
            node = prefix_node
            prefix = variable.largest_prefix
        return created

    def _add_edge(self, head: SketchNode, tail: SkNode, label: str) -> None:
        # don't emit duplicate edges
        if (head, tail) not in self.sketches.edges:
            self.sketches.add_edge(head, tail, label=label)

    def _copy_global_recursive(
        self, node: SketchNode, sketches: Sketches
    ) -> SketchNode:
        # TODO: this probably needs to handle atoms properly, using the Lattice. Needs
        # some thought.
        our_node = self.ref_node(node)
        if node in sketches.sketches.nodes:
            for _, dst in sketches.sketches.out_edges(node):
                our_dst = dst
                if not isinstance(dst, LabelNode):
                    our_dst = self._copy_global_recursive(dst, sketches)
                self._add_edge(
                    our_node, our_dst, sketches.sketches[node][dst]["label"]
                )
        return our_node

    def copy_globals_from_sketch(
        self, global_vars: Set[DerivedTypeVariable], sketches: Sketches
    ):
        global_roots = set()
        for dtv, node in sketches.lookup.items():
            if dtv.base_var in global_vars:
                global_roots.add(dtv.base_var)
        for g in global_roots:
            node = sketches.lookup.get(g)
            if node is None:
                continue
            self._copy_global_recursive(node, sketches)

    def instantiate_sketch(
        self,
        proc: DerivedTypeVariable,
        types: Lattice[DerivedTypeVariable],
        fresh_var_factory: FreshVarFactory,
    ) -> ConstraintSet:
        """
        Encode all the information present in the sketch into constraints.
        - For each node that is not top or botton, generate a constraint with its type
        - Generate dummy constraints to encode the capabilities of the sketch
        - Generate constraints to encode the cycles (recursive types) in the sketch.
        """
        all_constraints = ConstraintSet()
        for node in self.sketches.nodes:
            if isinstance(node, SketchNode) and node.dtv.base_var == proc:
                constraints = []
                # if the node has some type, capture that in a constraint
                if node.lower_bound != types.bottom:
                    constraints.append(
                        SubtypeConstraint(node.lower_bound, node.dtv)
                    )
                if node.upper_bound != types.top:
                    constraints.append(
                        SubtypeConstraint(node.dtv, node.upper_bound)
                    )
                # if the node is a leaf, capture the capability using fake variables
                # this could be avoided if we support capability constraints  (Var x.l) in
                # addition to subtype constraints
                if (
                    len(constraints) == 0
                    and len(list(self.sketches.successors(node))) == 0
                ):
                    fresh_var = fresh_var_factory.fresh_var()
                    # FIXME check if this should be the other way around
                    if node.dtv.path_variance == Variance.COVARIANT:
                        constraints.append(
                            SubtypeConstraint(node.dtv, fresh_var)
                        )
                    else:
                        constraints.append(
                            SubtypeConstraint(fresh_var, node.dtv)
                        )
                # I am not sure about this, but I think label nodes should
                # not be completely ignored
                for succ in self.sketches.successors(node):
                    if isinstance(succ, LabelNode):
                        label = self.sketches[node][succ].get("label")
                        loop_back = node.dtv.add_suffix(label)
                        if loop_back.path_variance == Variance.COVARIANT:
                            constraints.append(
                                SubtypeConstraint(loop_back, succ.target)
                            )
                        else:
                            constraints.append(
                                SubtypeConstraint(succ.target, loop_back)
                            )
                all_constraints |= ConstraintSet(constraints)
        return all_constraints

    def add_constraints(self, constraints: ConstraintSet) -> None:
        """Extend the set of sketches with the new set of constraints."""

        # Step 1: Apply constraints that make use of lattice types, so that we have the seeds
        # for atomic types in our sketch graph
        for constraint in constraints:
            left = constraint.left
            right = constraint.right
            if (
                left in self.types.internal_types
                and right not in self.types.internal_types
            ):
                right_node = self.lookup[right]
                self.debug("JOIN: %s, %s", right_node, left)
                right_node.lower_bound = self.types.join(
                    right_node.lower_bound, left
                )
                self.debug("   --> %s", right_node)
            if (
                right in self.types.internal_types
                and left not in self.types.internal_types
            ):
                left_node = self.lookup[left]
                self.debug("MEET: %s, %s", left_node, left)
                left_node.upper_bound = self.types.meet(
                    left_node.upper_bound, right
                )
                self.debug("   --> %s", left_node)

    def to_dot(self, dtv: DerivedTypeVariable) -> str:
        nt = f"{os.linesep}\t"
        graph_str = f"digraph {dtv} {{"
        start = self.lookup[dtv]
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
        if self.lookup:
            nt = f"{os.linesep}\t"

            def format(k: DerivedTypeVariable) -> str:
                return str(self.lookup[k])

            return f"nodes:{nt}{nt.join(map(format, self.lookup.keys()))})"
        return "no sketches"
