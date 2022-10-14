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
        self._hash = hash((self.target, self.id))

    def __eq__(self, other) -> bool:
        if isinstance(other, LabelNode):
            return self.id == other.id and self.target == other.target
        return False

    def __hash__(self) -> int:
        return self._hash

    def __str__(self) -> str:
        return f"{self.target}.label_{self.id}"

    def __repr__(self) -> str:
        return str(self)


SkNode = Union[SketchNode, LabelNode]


class Sketch(Loggable):
    """
    The sketch of a type variable.
    """

    def __init__(
        self,
        root: DerivedTypeVariable,
        types: Lattice[DerivedTypeVariable],
        verbose: LogLevel = LogLevel.QUIET,
    ) -> None:
        super(Sketch, self).__init__(verbose)
        # We maintain the invariant that if a node is in `lookup` then it should also be in
        # `sketches` as a node (even if there are no edges)
        self.sketches = networkx.DiGraph()
        self._lookup: Dict[DerivedTypeVariable, SketchNode] = {}
        self.types = types
        self.root = self.make_node(root)

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

    def make_node(self, variable: DerivedTypeVariable) -> SketchNode:
        """Make a node from a DTV. Compute its atom from its access path."""
        result = SketchNode(variable, self.types.bottom, self.types.top)
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
            if label != self.sketches.edges[head, tail]["label"]:
                raise RetypdError(
                    f"Failed to add edge {label} between {head} and {tail}."
                    f" Label {self.sketches.edges[head, tail]['label']} exists"
                )

    def instantiate_sketch(
        self,
        proc: DerivedTypeVariable,
        fresh_var_factory: FreshVarFactory,
        only_capabilities: bool = False,
    ) -> ConstraintSet:
        """
        Encode all the capability and primitive type information present in the sketch.
        """
        all_constraints = ConstraintSet()
        for node in self.sketches.nodes:
            if isinstance(node, SketchNode) and node.dtv.base_var == proc:
                constraints = []
                if not only_capabilities:
                    if node.lower_bound != self.types.bottom:
                        constraints.append(
                            SubtypeConstraint(node.lower_bound, node.dtv)
                        )
                    if node.upper_bound != self.types.top:
                        constraints.append(
                            SubtypeConstraint(node.dtv, node.upper_bound)
                        )

                # if the node is a leaf, capture the capability using fake variables
                # this could be avoided if we support capability constraints  (Var x.l) in
                # addition to subtype constraints
                if (
                    len(constraints) == 0
                    and self.sketches.out_degree(node) == 0
                ):
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

    def remove_subtree(self, node: SkNode) -> None:
        """
        Remove the subtree with root node from the sketch.
        """
        worklist = [node]
        while len(worklist) > 0:
            node = worklist.pop()
            worklist.extend(self.sketches.successors(node))
            self.sketches.remove_node(node)
            if isinstance(node, SketchNode):
                if node.dtv in self._lookup:
                    del self._lookup[node.dtv]

    def meet(self, other: Sketch) -> None:
        """
        Compute in-place meet of self and another sketch
        """
        if self.root.dtv != other.root.dtv:
            raise RetypdError(
                "Cannot compute a meet of two sketches with different root"
            )

        worklist = [(self.root, other.root)]
        met_nodes = set()
        while len(worklist) > 0:
            curr_node, other_node = worklist.pop()
            # Avoid infinite loop in case of label nodes
            if (curr_node, other_node) in met_nodes:
                continue
            met_nodes.add((curr_node, other_node))

            # Deal with primitive type
            curr_node.lower_bound = self.types.join(
                curr_node.lower_bound, other_node.lower_bound
            )
            curr_node.upper_bound = self.types.meet(
                curr_node.upper_bound, other_node.upper_bound
            )
            # Meet of successors: language union
            curr_succs = {
                label: succ
                for _, succ, label in self.sketches.out_edges(
                    curr_node, data="label"
                )
            }
            for _, other_succ, label in other.sketches.out_edges(
                other_node, data="label"
            ):
                if label not in curr_succs:
                    # create new node
                    if isinstance(other_succ, SketchNode):
                        curr_succ = self.make_node(other_succ.dtv)
                        curr_succ.upper_bound = other_succ.upper_bound
                        curr_succ.lower_bound = other_succ.lower_bound
                    else:  # LabelNode
                        curr_succ = LabelNode(other_succ.target)
                    self.add_edge(curr_node, curr_succ, label)
                else:
                    curr_succ = curr_succs[label]
                # follow label nodes
                if isinstance(curr_succ, LabelNode):
                    curr_succ = self.lookup(curr_succ.target)
                if isinstance(other_succ, LabelNode):
                    other_succ = other.lookup(other_succ.target)
                worklist.append((curr_succ, other_succ))

    def join(self, other: Sketch) -> None:
        """
        Compute in-place join of self and another sketch
        """

        if self.root.dtv != other.root.dtv:
            raise RetypdError(
                "Cannot compute a join of two sketches with different root"
            )
        worklist = [(self.root, other.root)]
        while len(worklist) > 0:
            curr_node, other_node = worklist.pop()
            # Deal with primitive type
            curr_node.lower_bound = self.types.meet(
                curr_node.lower_bound, other_node.lower_bound
            )
            curr_node.upper_bound = self.types.join(
                curr_node.upper_bound, other_node.upper_bound
            )

            # Join successors: Language intersection
            other_succs = {
                label: succ
                for _, succ, label in other.sketches.out_edges(
                    other_node, data="label"
                )
            }
            for _, curr_succ, label in list(
                self.sketches.out_edges(curr_node, data="label")
            ):
                if label not in other_succs:
                    self.remove_subtree(curr_succ)
                else:
                    other_succ = other_succs[label]
                    if isinstance(curr_succ, SketchNode) and isinstance(
                        other_succ, SketchNode
                    ):
                        worklist.append((curr_succ, other_succ))
                    # TODO what to do with LabelNodes?

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
