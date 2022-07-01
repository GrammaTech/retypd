from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
from typing import Any, Dict, List, Tuple
import enum
import networkx


class RExp:
    """Regular expression class with some helper methods and simplification"""

    class Label(enum.Enum):
        NULL = 0
        EMPTY = 1
        NODE = 2
        DOT = 3
        OR = 4
        STAR = 5

    def __init__(self, label: Label, data=None, children=[]):
        self.label = label
        self.data: Any = data
        self.children: Tuple[RExp] = tuple(children)
        self.hash = hash((self.label, self.data, self.children))

    def __hash__(self) -> int:
        return self.hash

    def __and__(self, rhs: RExp) -> RExp:
        return RExp(self.Label.DOT, children=(self, rhs))

    def __or__(self, rhs: RExp) -> RExp:
        return RExp(self.Label.OR, children=(self, rhs))

    def star(self):
        return RExp(self.Label.STAR, children=(self,))

    def __eq__(self, other: RExp) -> bool:
        if self.label != other.label:
            return False
        if self.label in (self.Label.NULL, self.Label.EMPTY):
            return True
        elif self.label in (self.Label.DOT, self.Label.OR, self.Label.STAR):
            return all(a == b for a, b in zip(self.children, other.children))
        elif self.label == self.Label.NODE:
            return self.data == other.data
        else:
            raise NotImplementedError()

    def __lt__(self, other: RExp) -> bool:
        """
        Compare two regular expressions.
        """
        if not isinstance(other, RExp):
            raise ValueError(f"Cannot compare RExp to {type(other)}")
        if self.label != other.label:
            return self.label.value < other.label.value
        # Same label
        if self.label in (self.Label.NULL, self.Label.EMPTY):
            return False
        elif self.label in (self.Label.DOT, self.Label.OR, self.Label.STAR):
            if len(self.children) != len(other.children):
                return len(self.children) < len(other.children)
            else:
                for a, b in zip(self.children, other.children):
                    if a < b:
                        return True
                    elif a > b:
                        return False
                # all equal
                return False
        elif self.label == self.Label.NODE:
            return self.data < other.data

        raise NotImplementedError()

    @classmethod
    def null(cls) -> RExp:
        return RExp(cls.Label.NULL)

    @classmethod
    def empty(cls) -> RExp:
        return RExp(cls.Label.EMPTY)

    @classmethod
    def node(cls, data) -> RExp:
        return RExp(cls.Label.NODE, data=data)

    @classmethod
    def from_graph_edge(
        cls, graph: networkx.DiGraph, src: Any, dest: Any, data: str
    ) -> RExp:
        """Generate a regular expression from a graph node. For no label on
        data we assume an empty string, otherwise a node labeled by that label
        """
        attrs = graph[src][dest]
        if data not in attrs:
            return cls.empty()
        else:
            return cls.node(attrs[data])

    def simplify(self) -> RExp:
        """Regular expression simplification procedure, page 9

        This simplification procedure has been expanded to
        deal with many other sources of redundancy.
        The simplification is not recursive, it only simplifies
        the two top levels of the RExp, deeper levels have
        been already simplified.

        """
        children = self.children

        if self.label == self.Label.OR:
            # use a set to avoid duplicates
            new_children = set()
            for child in children:
                # flatten nested OR
                if child.label == RExp.Label.OR:
                    new_children |= set(child.children)
                else:
                    if not child.is_null:
                        new_children.add(child)
            if len(new_children) == 1:
                return new_children.pop()
            else:
                return RExp(RExp.Label.OR, children=sorted(new_children))
        elif self.label == self.Label.DOT:
            if any(child.is_null for child in children):
                return RExp.null()
            new_children = []
            for child in children:
                # flatten nested DOT
                if child.label == RExp.Label.DOT:
                    new_children.extend(child.children)
                else:
                    if not child.is_empty:
                        new_children.append(child)
            if len(new_children) == 1:
                return new_children.pop()
            return RExp(RExp.Label.DOT, children=new_children)
        elif self.label == self.Label.STAR:
            if children[0].is_null or children[0].is_empty:
                return RExp.empty()
            else:
                return children[0].star()
        return self

    @property
    def is_null(self) -> bool:
        return self.label == self.Label.NULL

    @property
    def is_empty(self) -> bool:
        return self.label == self.Label.EMPTY

    @property
    def is_node(self) -> bool:
        return self.label == self.Label.NODE

    @lru_cache
    def __repr__(self) -> str:
        if self.label == self.Label.OR:
            return (
                "("
                + " U ".join(child.__repr__() for child in self.children)
                + ")"
            )

        elif self.label == self.Label.DOT:
            return (
                "("
                + " . ".join(child.__repr__() for child in self.children)
                + ")"
            )
        elif self.label == self.Label.STAR:
            return f"{self.children[0]}*"
        elif self.label == self.Label.EMPTY:
            return "Λ"
        elif self.label == self.Label.NULL:
            return "∅"
        elif self.label == self.Label.NODE:
            return f"{self.data}"
        else:
            raise NotImplementedError()


def eliminate(
    graph: networkx.DiGraph, data: str, min_num: int, max_num: int
) -> Dict[Tuple[int, int], RExp]:
    """ELIMINATE procedure of Tarjan's path expression algorithm, page 13"""
    # Initialize
    P = defaultdict(lambda: RExp.null())

    # consider edges in the subgraph defined by the range
    for h, t in graph.edges(range(min_num, max_num)):
        if t < min_num or t >= max_num:
            continue
        edge = RExp.from_graph_edge(graph, h, t, data)
        P[h, t] = (P[h, t] | edge).simplify()

    # Loop
    for v in range(min_num, max_num):
        P[v, v] = P[v, v].star().simplify()

        for u in range(v + 1, max_num):
            if P[u, v].is_null:
                continue

            P[u, v] = (P[u, v] & P[v, v]).simplify()

            for w in range(v + 1, max_num):
                if P[v, w].is_null:
                    continue

                P[u, w] = (P[u, w] | (P[u, v] & P[v, w]).simplify()).simplify()

    return P


PathSeq = List[Tuple[Tuple[int, int], RExp]]


def compute_path_sequence(
    P: Dict[Tuple[int, int], RExp],
    min_num: int,
    max_num: int,
) -> PathSeq:
    """Compute path sequence from the ELIMINATE procedure, per Theorem 4 on
    page 14
    """

    # Compute ascending and descending path sequences that are in the queried
    # range for this path sequence
    valid_range = range(min_num, max_num)
    ascending = []
    descending = []

    for indices, expr in P.items():
        start, end = indices

        if start not in valid_range or end not in valid_range:
            continue

        if expr.is_null:
            continue
        # no need to include empty self-paths
        if expr.is_empty and start == end:
            continue

        if start <= end:
            ascending.append((indices, expr))
        else:
            descending.append((indices, expr))

    # Sort by the starting node
    output = sorted(ascending, key=lambda pair: pair[0][0]) + sorted(
        descending, key=lambda pair: pair[0][0], reverse=True
    )

    return output


def solve_paths_from(
    path_seq: PathSeq, source: int
) -> Dict[Tuple[int, int], RExp]:
    """Solve path expressions from a source given a path sequence for a
    numbered graph, per procedure SOLVE on page 9 of Tarjan
    """
    P = defaultdict(lambda: RExp.null())
    P[source, source] = RExp.empty()

    for (v_i, w_i), P_i in path_seq:
        if v_i == w_i:
            P[source, v_i] = (P[source, v_i] & P_i).simplify()
        else:
            P[source, w_i] = (
                P[source, w_i] | (P[source, v_i] & P_i).simplify()
            ).simplify()

    return P


def from_numeral_graph(numeral_graph: networkx.DiGraph, number: int) -> Any:
    """Translate from numeral graph to the original node in the graph"""
    return numeral_graph.nodes[number]["original"]


GraphNumbering = Dict[Any, int]


def topological_numbering(
    graph: networkx.DiGraph,
) -> Tuple[GraphNumbering, networkx.DiGraph]:
    """Generate a numeral graph from the topological sort of the DAG"""
    nodes = list(networkx.topological_sort(graph))
    numbering = {node: num for num, node in enumerate(nodes)}
    rev_numbering = dict(enumerate(nodes))
    numeral_graph = networkx.relabel_nodes(graph, numbering)
    networkx.set_node_attributes(numeral_graph, rev_numbering, name="original")
    return numbering, numeral_graph


def dag_path_seq(graph: networkx.DiGraph, data: str) -> PathSeq:
    """Per Theorem 5, Page 14 of the paper, generate path sequences for a
    directed acyclic graph in a more efficient manner.
    """
    # Sort edges by increasing source node
    edges = sorted(graph.edges(), key=lambda x: x[0])
    return [
        ((h, t), RExp.from_graph_edge(graph, h, t, data)) for h, t in edges
    ]


def scc_decompose_path_seq(
    graph: networkx.DiGraph, data: str
) -> Tuple[GraphNumbering, PathSeq]:
    """Per Theorem 6, Page 14 of the paper, generate path sequences for a graph
    that has been decomposed into strongly connected components.
    """
    # Generate the graph of SCCs
    component_graph = networkx.condensation(graph)

    scc_numberings = {}
    graph_numbering = {}
    curr_number = 0

    # Generate a numbering for each SCC in increasing topological order to
    # maintain that any edge G_i -> G_j whare are in SCCs S_i and S_j
    # respectively, if theres an edge in the condensation from S_i -> S_j then
    # G_i < G_j
    for component in networkx.topological_sort(component_graph):
        # Generate numbering for this SCC, we do so in a sorted order as
        # NetworkX returns a set whose non-deterministic ordering can return
        # inconsistent results
        scc = component_graph.nodes[component]["members"]
        start_number = curr_number

        for elem in sorted(scc):
            graph_numbering[elem] = curr_number
            curr_number += 1

        # Update whole-graph numbering, and keep note of SCC ranges
        scc_numberings[component] = (start_number, curr_number)

    # Do the actual relabeling of the graph to the numeral graph
    number_graph = networkx.relabel_nodes(graph, graph_numbering)
    rev_numbering = {v: k for k, v in graph_numbering.items()}
    networkx.set_node_attributes(number_graph, rev_numbering, name="original")

    scc_seqs: List[Tuple[int, PathSeq]] = []

    # Do ELIMINATE for every SCC, by using a slice of the graph from the
    # min/max of the given SCC's numbering
    for component in networkx.topological_sort(component_graph):
        min_num, max_num = scc_numberings[component]
        P = eliminate(number_graph, data, min_num, max_num)
        seqs = compute_path_sequence(P, min_num, max_num)
        scc_seqs.append((component, seqs))

    output: PathSeq = []

    for component, seqs in scc_seqs:
        # Add the intra-SCC path sequence nodes
        output += seqs

        # Add inter-SCC path sequence nodes
        scc = component_graph.nodes[component]["members"]

        for in_edge, out_edge in graph.out_edges(scc):
            if out_edge not in scc:
                in_num = graph_numbering[in_edge]
                out_num = graph_numbering[out_edge]
                output.append(
                    (
                        (in_num, out_num),
                        RExp.from_graph_edge(graph, in_edge, out_edge, data),
                    )
                )

    return graph_numbering, output


def path_expression_between(
    graph: networkx.DiGraph,
    data: str,
    source: Any,
    sink: Any,
    decompose=True,
):
    """Per Lemma 1 on page 9, handle output of SOLVE procedure"""
    # First, compute path sequences of the graph
    if not decompose:
        # Generate numberings for the nodes
        number_graph = networkx.convert_node_labels_to_integers(
            graph, label_attribute="original"
        )

        numbering = {
            original: number
            for number, original in number_graph.nodes(data="original")
        }

        N = len(number_graph.nodes)
        P = eliminate(number_graph, data, 0, N)
        seqs = compute_path_sequence(P, 0, N)
    elif networkx.is_directed_acyclic_graph(graph):
        # Fast path for DAGs
        numbering, number_graph = topological_numbering(graph)
        seqs = dag_path_seq(number_graph, data)
    else:
        numbering, seqs = scc_decompose_path_seq(graph, data)

    # Solve all paths for source, and output the one for (source, sink)
    paths = solve_paths_from(seqs, numbering[source])
    return paths[(numbering[source], numbering[sink])]
