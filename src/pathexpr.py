from __future__ import annotations
from collections import defaultdict
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
    graph: networkx.DiGraph, data: str, min_num: int, max_num: int
) -> Dict[Tuple[int, int], RExp]:
    """ELIMINATE procedure of Tarjan's path expression algorithm, page 13"""
    # Initialize
    P = defaultdict(lambda: RExp.null())

    for (h, t, data) in graph.edges(data=data):
        P[h, t] = (P[h, t] | RExp.node(data)).simplify()

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

        if expr.is_empty or expr.is_null:
            continue
        
        if start <= end:
            ascending.append((indices, expr))
        else:
            descending.append((indices, expr))

    # Sort by the starting node 
    output = (
        sorted(ascending, key=lambda pair: pair[0][0])
        + sorted(descending, key=lambda pair: pair[0][0], reverse=True)
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
    graph: networkx.DiGraph
) -> Tuple[GraphNumbering, networkx.DiGraph]:
    """ Generate a numeral graph from the topological sort of the DAG """
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
    edges = list(graph.edges(data=data))
    edges.sort(key=lambda x: x[0])
    return [((h, t), RExp.node(e)) for h, t, e in edges]


def scc_decompose_path_seq(
    graph: networkx.DiGraph, data: str
) -> Tuple[GraphNumbering, PathSeq]:
    """ Per Theorem 6, Page 14 of the paper, generate path sequences for a graph
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

    # Do ELIMINATE for every SCC, and make a merged graph with all the 
    # numberings available.
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

        for in_edge, out_edge, label in graph.out_edges(scc, data=data):
            if out_edge not in scc:
                in_num = graph_numbering[in_edge]
                out_num = graph_numbering[out_edge]
                output.append(((in_num, out_num), RExp.node(label)))

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


if __name__ == "__main__":
    test = networkx.DiGraph()

    test.add_edge('a', 'b', data='A')
    test.add_edge('b', 'c', data='B')
    test.add_edge('b', 'b', data='C')
    test.add_edge('c', 'd', data='D')
    test.add_edge('d', 'd', data='E')

    test.add_edge('a', 'e', data='F')
    test.add_edge('e', 'f', data='G')
    test.add_edge('f', 'e', data='H')
    test.add_edge('e', 'd', data='I')

    print(path_expression_between(test, 'data', 'a', 'd', True))
