from __future__ import annotations
from typing import Any, Dict, List, Tuple
import enum
import networkx


class RExp:
    """Regular expression class with some helper methods and simplification"""
    class Label(enum.Enum):
        NULL = "null"
        EMPTY = "empty"
        DOT = "dot"
        OR = "or"
        STAR = "star"
        NODE = "node"

    def __init__(self, label: Label, data=None, children=[]):
        self.label = label
        self.data: Any = data
        self.children: List[RExp] = children
    
    def __and__(self, rhs: RExp) -> RExp:
        return RExp(self.Label.DOT, children=[self, rhs])
    
    def __or__(self, rhs: RExp) -> RExp:
        return RExp(self.Label.OR, children=[self, rhs])

    def star(self):
        return RExp(self.Label.STAR, children=[self])

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

    @classmethod
    def null(cls) -> RExp:
        return RExp(cls.Label.NULL)

    @classmethod
    def empty(cls) -> RExp:
        return RExp(cls.Label.EMPTY)

    @classmethod
    def node(cls, data) -> RExp:
        return RExp(cls.Label.NODE, data=data)

    def simplify(self) -> RExp:
        """Regular expression simplification procedure, page 9"""
        children = [child.simplify() for child in self.children]

        if self.label == self.Label.OR:
            if children[0].is_null:
                return children[1]
            elif children[1].is_null:
                return children[0]
            else:
                return children[0] | children[1]
        elif self.label == self.Label.DOT:
            if children[0].is_null or children[1].is_null:
                return RExp.null()
            elif children[0].is_empty:
                return children[1]
            elif children[1].is_empty:
                return children[0]
            else:
                return children[0] & children[1]
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

    def __repr__(self) -> str:
        if self.label == self.Label.OR:
            return f'({self.children[0]} U {self.children[1]})'
        elif self.label == self.Label.DOT:
            return f'({self.children[0]} . {self.children[1]})'
        elif self.label == self.Label.STAR:
            return f'{self.children[0]}*'
        elif self.label == self.Label.EMPTY:
            return 'Λ'
        elif self.label == self.Label.NULL:
            return '∅'
        elif self.label == self.Label.NODE:
            return f'{self.data}'
        else:
            raise NotImplementedError()


def eliminate(
    graph: networkx.DiGraph, data: str, first_number = 0
) -> Tuple[networkx.DiGraph, Dict[Tuple[int, int], RExp]]:
    """ELIMINATE procedure of Tarjan's path expression algorithm, page 13"""
    numeral_graph = networkx.convert_node_labels_to_integers(
        graph, label_attribute="original", first_label=first_number
    )

    # Initialize
    N = len(numeral_graph)
    P = {}

    for v in range(N):
        for w in range(N):
            P[v, w] = RExp.null()

    for (h, t, data) in numeral_graph.edges(data=data):
        P[h, t] = (P[h, t] | RExp.node(data)).simplify()

    # Loop
    for v in range(N):
        P[v, v] = P[v, v].star().simplify()

        for u in range(v + 1, N):
            if P[u, v].is_null:
                continue
        
            P[u, v] = (P[u, v] & P[v, v]).simplify()

            for w in range(v + 1, N):
                if P[v, w].is_null:
                    continue

                P[u, w] = P[u, w] | (P[u, v] & P[v, w])
                P[u, w].simplify()

    return numeral_graph, P

def compute_path_sequence(
    numbered_graph: networkx.DiGraph, P: Dict[Tuple[int, int], RExp]
) -> List[Tuple[RExp, int, int]]:
    """Compute path sequence from the ELIMINATE procedure, per Theorem 4 on 
    page 14
    """
    N = len(numbered_graph)
    output = []

    for u in range(N):
        for v in range(u, N):
            expr = P[u, v]
            if not expr.is_empty and not expr.is_null:
                output.append((expr, u, v))
    
    for u in range(N - 1, -1, -1):
        for w in range(0, u):
            expr = P[u, w]
            if not expr.is_empty and not expr.is_null:
                output.append((expr, u, w))

    return output


def solve_paths_from(
    numeral_graph: networkx.DiGraph,
    path_seq: List[Tuple[RExp, int, int]],
    source: int
) -> Dict[Tuple[int, int], RExp]:
    """Solve path expressions from a source given a path sequence for a 
    numbered graph, per procedure SOLVE on page 9 of Tarjan
    """
    P = {}

    P[source, source] = RExp.empty()

    for v in numeral_graph.nodes - {source}:
        P[source, v] = RExp.null()

    for i in range(len(path_seq)):
        P_i, v_i, w_i = path_seq[i]

        if v_i == w_i:
            P[source, v_i] = (P[source, v_i] & P_i).simplify()
        else:
            P[source, w_i] = (P[source, w_i] | (P[source, v_i] & P_i)).simplify()

    return P


def from_numeral_graph(numeral_graph: networkx.DiGraph, number: int) -> Any:
    """Translate from numeral graph to the original node in the graph"""
    return numeral_graph.nodes[number]["original"]


def topological_numbering(graph: networkx.DiGraph) -> networkx.DiGraph:
    """ Generate a numeral graph from the topological sort of the DAG """
    nodes = list(networkx.topological_sort(graph))
    numbering = {node: num for num, node in enumerate(nodes)}
    rev_numbering = dict(enumerate(nodes))
    numeral_graph = networkx.relabel_nodes(graph, numbering, copy=True)
    networkx.set_node_attributes(numeral_graph, rev_numbering, name="original")
    return numeral_graph


def dag_path_seq(graph: networkx.DiGraph, data: str) -> Tuple[networkx.DiGraph, List[Tuple[RExp, int, int]]]:
    """Per Theorem 5, Page 14 of the paper, generate path sequences for a 
    directed acyclic graph in a more efficient manner.
    """
    numeral_graph = topological_numbering(graph)
    # Sort edges by increasing source node
    edges = list(numeral_graph.edges(data=data))
    edges.sort(key=lambda x: x[0])
    return numeral_graph, [(RExp.node(e), h, t) for h, t, e in edges]


def scc_decompose_path_seq(
    graph: networkx.DiGraph, data: str
) -> Tuple[networkx.DiGraph, List[Tuple[RExp, int, int]]]:
    """ Per Theorem 6, Page 14 of the paper, generate path sequences for a graph
    that has been decomposed into strongly connected components.
    """
    # Generate the graph of SCCs
    component_graph = networkx.condensation(graph)

    component_graph_nodes = networkx.topological_sort(component_graph)
    component_numbering = []

    # The goal here it to assign a number to each node in the components graph
    # such that for any edge G_i -> G_j, i < j. Typically this is just a 
    # topological sort, however, when we construct the final path sequence we
    # also want to be able to insert our intra-SCC nodes into the path sequence
    # and hold the property that any node in G_i has a number less than G_j if
    # there exists an edge from G_i -> G_j. To do so we go through the nodes of
    # the condensation of the graph and assign numbers to the nodes that 
    # represent the SCCs that increase by the size of the SCC (plus one for the
    # SCC node itself). 
    for node in component_graph_nodes:
        num_members = len(component_graph.nodes[node]["members"])
        preceding = (
            component_numbering[-1]
            if len(component_numbering) > 0 else 0
        ) + 1
        component_numbering.append(num_members + preceding)

    component_numbering = dict(zip(component_graph_nodes, component_numbering))
    networkx.relabel_nodes(component_graph, component_numbering)

    output = []
    merged_graph = networkx.DiGraph()
    scc_seqs = []

    # Do ELIMINATE for every SCC, and make a merged graph with all the 
    # numberings available.
    for scc, members in component_graph.nodes(data="members"):
        scc_graph = graph.subgraph(members)
        scc_number_graph, P = eliminate(scc_graph, data, first_number=scc)
        seqs = compute_path_sequence(scc_number_graph, P)
        scc_seqs.append((scc, seqs))

        merged_graph = networkx.compose(merged_graph, scc_number_graph)

    global_numbering = {
        original: number
        for number, original in merged_graph.nodes(data="original")
    }

    output = []

    for scc, seqs in scc_seqs:
        # Add the intra-SCC path sequence nodes
        output += seqs

        # Add inter-SCC path sequence nodes
        members = component_graph.nodes[scc]["members"]
        for in_edge, out_edge, data in graph.out_edges(members, data=data):
            if out_edge not in members:
                in_num = global_numbering[in_edge]
                out_num = global_numbering[out_edge]
                output.append((RExp.node(data), in_num, out_num))

    return merged_graph, output

def path_expression_between(
    graph: networkx.DiGraph,
    data: str,
    source: Any,
    sink: Any,
    no_decompose=True,
):
    """Per Lemma 1 on page 9, handle output of SOLVE procedure"""
    # First, compute path sequences of the graph
    if no_decompose:
        number_graph, P = eliminate(graph, data)
        seqs = compute_path_sequence(number_graph, P)
    elif networkx.is_directed_acyclic_graph(graph):
        number_graph, seqs = dag_path_seq(graph, data)
    else:
        number_graph, seqs = scc_decompose_path_seq(graph, data)

    # Compute map of number to original node for back-translation
    numbering = {
        original: number
        for number, original in number_graph.nodes(data="original")
    }

    # Solve all paths for source, and output the one for (source, sink)
    paths = solve_paths_from(number_graph, seqs, numbering[source])
    return paths[(numbering[source], numbering[sink])]
