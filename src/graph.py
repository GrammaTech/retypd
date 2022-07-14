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

from __future__ import annotations
from enum import Enum, unique
from typing import Any, Dict, Optional, Set, Tuple
from .schema import (
    AccessPathLabel,
    ConstraintSet,
    DerivedTypeVariable,
    LoadLabel,
    StoreLabel,
    Variance,
)
import networkx
import os


class EdgeLabel:
    """A forget or recall label in the graph. Instances should never be mutated."""

    @unique
    class Kind(Enum):
        FORGET = 1
        RECALL = 2

    def __init__(self, capability: AccessPathLabel, kind: Kind) -> None:
        self.capability = capability
        self.kind = kind
        if self.kind == EdgeLabel.Kind.FORGET:
            type_str = "forget"
        else:
            type_str = "recall"
        self._str = f"{type_str} {self.capability}"
        self._hash = hash(self.capability) ^ hash(self.kind)

    def __eq__(self, other: EdgeLabel) -> bool:
        return (
            isinstance(other, EdgeLabel)
            and self.capability == other.capability
            and self.kind == other.kind
        )

    def __lt__(self, other: EdgeLabel) -> bool:
        if not isinstance(other, EdgeLabel):
            raise ValueError(f"Cannot compare EdgeLabel to {type(other)}")
        return self._str < other._str

    def __hash__(self) -> int:
        return self._hash

    def __str__(self) -> str:
        return self._str

    def __repr__(self) -> str:
        return self._str


@unique
class SideMark(Enum):
    """
    Marking of interesting graph nodes to avoid non-elementary proofs.
    See Definition D.2 and Note 1 in section D.1 of the paper.
    """

    NO = 0
    LEFT = 1
    RIGHT = 2


class Node:
    """A node in the graph of constraints. Node objects are immutable.

    Forgotten is a flag used to differentiate between two subgraphs later in the algorithm. See
    :py:method:`Solver._recall_forget_split` for details.
    """

    @unique
    class Forgotten(Enum):
        PRE_FORGET = 0
        POST_FORGET = 1

    def __init__(
        self,
        base: DerivedTypeVariable,
        suffix_variance: Variance,
        side_mark: SideMark = SideMark.NO,
        forgotten: Forgotten = Forgotten.PRE_FORGET,
    ) -> None:
        self.base = base
        self.suffix_variance = suffix_variance
        self.side_mark = side_mark
        if side_mark == SideMark.LEFT:
            side_mark_str = "L:"
        elif side_mark == SideMark.RIGHT:
            side_mark_str = "R:"
        else:
            side_mark_str = ""
        if suffix_variance == Variance.COVARIANT:
            variance = ".⊕"
        else:
            variance = ".⊖"
        self._forgotten = forgotten
        if forgotten == Node.Forgotten.POST_FORGET:
            self._str = "F:" + side_mark_str + str(self.base) + variance
        else:
            self._str = side_mark_str + str(self.base) + variance
        self._hash = hash(
            (self.base, self.suffix_variance, self.side_mark, self._forgotten)
        )

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Node)
            and self.base == other.base
            and self.suffix_variance == other.suffix_variance
            and self._forgotten == other._forgotten
            and self.side_mark == other.side_mark
        )

    def __lt__(self, other: Node) -> bool:
        if not isinstance(other, Node):
            raise ValueError(
                f"Cannot compare objects of type Node and {type(other)} "
            )
        return self._str < other._str

    def __hash__(self) -> int:
        return self._hash

    def forget_once(
        self,
    ) -> Tuple[Optional[AccessPathLabel], Optional[Node]]:
        """ "Forget" the last element in the access path, creating a new Node. The new Node has
        variance that reflects this change.
        """
        if self.base.path:
            prefix_path = list(self.base.path)
            last = prefix_path.pop()
            prefix = DerivedTypeVariable(self.base.base, prefix_path)
            return (
                last,
                Node(
                    prefix,
                    Variance.combine(last.variance(), self.suffix_variance),
                    self.side_mark,
                ),
            )
        return (None, None)

    def recall(self, label: AccessPathLabel) -> Node:
        """ "Recall" label, creating a new Node. The new Node has variance that reflects this
        change.
        """
        path = list(self.base.path)
        path.append(label)
        variance = Variance.combine(self.suffix_variance, label.variance())
        return Node(
            DerivedTypeVariable(self.base.base, path),
            variance,
            self.side_mark,
        )

    def __str__(self) -> str:
        return self._str

    def __repr__(self) -> str:
        return self._str

    def split_recall_forget(self) -> Node:
        """Get a duplicate of self for use in the post-recall subgraph."""
        return Node(
            self.base,
            self.suffix_variance,
            self.side_mark,
            Node.Forgotten.POST_FORGET,
        )

    def inverse(self) -> Node:
        """Get a Node identical to this one but with inverted variance and mark."""
        new_side_mark = SideMark.NO
        if self.side_mark == SideMark.LEFT:
            new_side_mark = SideMark.RIGHT
        elif self.side_mark == SideMark.RIGHT:
            new_side_mark = SideMark.LEFT
        return Node(
            self.base,
            Variance.invert(self.suffix_variance),
            new_side_mark,
            self._forgotten,
        )


class ConstraintGraph:
    """Represents the constraint graph in the slides. Essentially the same as the transducer from
    Appendix D. Edge weights use the formulation from the paper.
    """

    def __init__(
        self,
        constraints: ConstraintSet,
        interesting_vars: Set[DerivedTypeVariable],
    ) -> None:
        self.graph = networkx.DiGraph()
        for constraint in constraints.subtype:
            self.add_edges(constraint.left, constraint.right, interesting_vars)
        self.saturate()
        self._remove_self_loops()

    def add_edge(self, head: Node, tail: Node, **atts) -> bool:
        """Add an edge to the graph. The optional atts dict should include, if anything, a mapping
        from the string 'label' to an EdgeLabel object.
        """
        if head not in self.graph or tail not in self.graph[head]:
            self.graph.add_edge(head, tail, **atts)
            return True
        return False

    def add_edges(
        self,
        sub: DerivedTypeVariable,
        sup: DerivedTypeVariable,
        interesting_vars: Set[DerivedTypeVariable],
        **atts,
    ) -> bool:
        """Add an edge to the underlying graph. Also add its reverse with reversed variance.
        Each constraint, becomes two pushdown rules in the paper.
        In each case, we add recall edges only to the left-hand term of the rule
        and forget edges to the right-hand side.
        """
        changed = False
        left = (
            SideMark.LEFT if sub.base_var in interesting_vars else SideMark.NO
        )
        right = (
            SideMark.RIGHT if sup.base_var in interesting_vars else SideMark.NO
        )
        forward_from = Node(sub, Variance.COVARIANT, left)
        forward_to = Node(sup, Variance.COVARIANT, right)
        changed = self.add_edge(forward_from, forward_to, **atts) or changed
        self.add_recalls(forward_from)
        self.add_forgets(forward_to)
        backward_from = forward_to.inverse()
        backward_to = forward_from.inverse()
        changed = self.add_edge(backward_from, backward_to, **atts) or changed
        self.add_recalls(backward_from)
        self.add_forgets(backward_to)
        return changed

    def add_recalls(self, node: Node) -> None:
        """
        Recall edges are added for the left-hand side of constraints
        """
        (capability, prefix) = node.forget_once()
        while prefix:
            self.add_edge(
                prefix,
                node,
                label=EdgeLabel(capability, EdgeLabel.Kind.RECALL),
            )
            node = prefix
            (capability, prefix) = node.forget_once()

    def add_forgets(self, node: Node) -> None:
        """
        Forget edges are added for the right-hand side of constraints
        """
        (capability, prefix) = node.forget_once()
        while prefix:
            self.add_edge(
                node,
                prefix,
                label=EdgeLabel(capability, EdgeLabel.Kind.FORGET),
            )
            node = prefix
            (capability, prefix) = node.forget_once()

    def saturate(self) -> None:
        """Add "shortcut" edges, per algorithm D.2 in the paper."""
        changed = False
        reaching_R: Dict[Node, Set[Tuple[AccessPathLabel, Node]]] = {}

        def add_forgets(
            dest: Node, forgets: Set[Tuple[AccessPathLabel, Node]]
        ):
            nonlocal changed
            if dest not in reaching_R or not (forgets <= reaching_R[dest]):
                changed = True
                reaching_R.setdefault(dest, set()).update(forgets)

        def add_edge(origin: Node, dest: Node):
            nonlocal changed
            changed = self.add_edge(origin, dest) or changed

        def is_contravariant(node: Node) -> bool:
            return node.suffix_variance == Variance.CONTRAVARIANT

        for head_x, tail_y in self.graph.edges:
            label = self.graph[head_x][tail_y].get("label")
            if label and label.kind == EdgeLabel.Kind.FORGET:
                add_forgets(tail_y, {(label.capability, head_x)})
        while changed:
            changed = False
            for head_x, tail_y in self.graph.edges:
                if not self.graph[head_x][tail_y].get("label"):
                    add_forgets(tail_y, reaching_R.get(head_x, set()))
            existing_edges = list(self.graph.edges)
            for head_x, tail_y in existing_edges:
                label = self.graph[head_x][tail_y].get("label")
                if label and label.kind == EdgeLabel.Kind.RECALL:
                    capability_l = label.capability
                    for (label, origin_z) in reaching_R.get(head_x, set()):
                        if label == capability_l:
                            add_edge(origin_z, tail_y)

            contravariant_vars = filter(is_contravariant, self.graph.nodes)
            for x in contravariant_vars:
                for (capability_l, origin_z) in reaching_R.get(x, set()):
                    label = None
                    if capability_l == StoreLabel.instance():
                        label = LoadLabel.instance()
                    if capability_l == LoadLabel.instance():
                        label = StoreLabel.instance()
                    if label:
                        add_forgets(x.inverse(), {(label, origin_z)})

    def _remove_self_loops(self) -> None:
        """Loops from a node directly to itself are not useful, so it's useful to remove them."""
        self.graph.remove_edges_from(
            {(node, node) for node in self.graph.nodes}
        )

    @staticmethod
    def edge_to_str(graph, edge: Tuple[Node, Node]) -> str:
        """A helper for __str__ that formats an edge"""
        width = 2 + max(map(lambda v: len(str(v)), graph.nodes))
        (sub, sup) = edge
        label = graph[sub][sup].get("label")
        edge_str = f"{str(sub):<{width}}→  {str(sup):<{width}}"
        if label:
            return edge_str + f" ({label})"
        else:
            return edge_str

    @staticmethod
    def graph_to_str(graph: networkx.DiGraph) -> str:
        nt = os.linesep + "\t"
        edge_to_str = lambda edge: ConstraintGraph.edge_to_str(graph, edge)
        return f"{nt.join(map(edge_to_str, graph.edges))}"

    def __str__(self) -> str:
        nt = os.linesep + "\t"
        return (
            f"ConstraintGraph:{nt}{ConstraintGraph.graph_to_str(self.graph)}"
        )


def remove_unreachable_states(
    graph: networkx.DiGraph, start_nodes: Set[Node], end_nodes: Set[Node]
) -> Tuple[networkx.DiGraph, Set[Node], Set[Node]]:
    """
    Remove states that not reachable from start_nodes or do not reach end_nodes.
    This can speed up path exploration since we do not have to search
    paths through nodes that do not reach interesting destinations.
    """
    if len(graph) == 0 or len(start_nodes) == 0 or len(end_nodes) == 0:
        return graph, set(), set()

    reachable_nodes = set(
        networkx.multi_source_dijkstra_path_length(graph, start_nodes).keys()
    )
    rev_reachable_nodes = set(
        networkx.multi_source_dijkstra_path_length(
            graph.reverse(), end_nodes
        ).keys()
    )
    keep = reachable_nodes & rev_reachable_nodes
    keep_start = start_nodes & keep
    keep_end = end_nodes & keep
    return graph.subgraph(keep), keep_start, keep_end
