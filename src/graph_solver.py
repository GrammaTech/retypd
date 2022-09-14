from __future__ import annotations
from typing import List, Optional, Set, Any
from .fast_enfa import FastENFA
from .pathexpr import RExp, scc_decompose_path_seq, solve_paths_from
from .graph import (
    EdgeLabel,
    Node,
    remove_unreachable_states,
)
from .schema import (
    ConstraintSet,
    Program,
    SubtypeConstraint,
    Variance,
)
import abc
import networkx
from dataclasses import dataclass
from pyformlang.finite_automaton import (
    EpsilonNFA,
    State,
    Symbol,
    Epsilon,
)


@dataclass
class GraphSolverConfig:
    # Maximum path length when converts the constraint graph into output constraints
    max_path_length: int = 2**64
    # Maximum number of paths to explore per type variable root when generating output constraints
    max_paths_per_root: int = 2**64
    # Maximum paths total to explore per SCC
    max_total_paths: int = 2**64
    # Restrict graph to reachable nodes from and to endpoints
    # before doing the path exploration.
    restrict_graph_to_reachable: bool = True


def _maybe_constraint(
    origin: Node, dest: Node, string: List[EdgeLabel]
) -> Optional[SubtypeConstraint]:
    """Generate constraints by adding the forgets in string to origin and the recalls in string
    to dest. If both of the generated vertices are covariant (the empty string's variance is
    covariant, so only covariant vertices can represent a type_scheme type variable without an
    elided portion of its path) and if the two variables are not equal, emit a constraint.
    """
    lhs = origin
    rhs = dest
    forgets = []
    recalls = []
    for label in string:
        if label.kind == EdgeLabel.Kind.FORGET:
            forgets.append(label.capability)
        else:
            recalls.append(label.capability)
    for recall in recalls:
        lhs = lhs.recall(recall)
    for forget in reversed(forgets):
        rhs = rhs.recall(forget)

    if (
        lhs.suffix_variance == Variance.COVARIANT
        and rhs.suffix_variance == Variance.COVARIANT
    ):
        lhs_var = lhs.base
        rhs_var = rhs.base
        if lhs_var != rhs_var:
            return SubtypeConstraint(lhs_var, rhs_var)
    return None


class GraphSolver(abc.ABC):
    """
    Given a graph of constraints, solve this to a new set of constraints, which
    should be smaller than the original set of constraints
    """

    def __init__(self, config: GraphSolverConfig, program: Program):
        self.config = config
        self.program = program

    def _generate_constraints_from_to_internal(
        self,
        graph: networkx.DiGraph,
        start_nodes: Set[Node],
        end_nodes: Set[Node],
    ) -> ConstraintSet:
        raise NotImplementedError()

    def generate_constraints_from_to(
        self,
        graph: networkx.DiGraph,
        start_nodes: Set[Node],
        end_nodes: Set[Node],
    ) -> ConstraintSet:
        """
        Generate a set of final constraints from a set of start_nodes to a set of
        end_nodes based on the given graph.
        Use path expressions or naive exploration depending on the
        Solver's configuration.
        """
        if self.config.restrict_graph_to_reachable:
            graph, start_nodes, end_nodes = remove_unreachable_states(
                graph, start_nodes, end_nodes
            )

        return self._generate_constraints_from_to_internal(
            graph, start_nodes, end_nodes
        )


class DFAGraphSolver(GraphSolver):
    START = State("$$START$$")
    FINAL = State("$$FINAL$$")

    def _graph_to_dfa(
        self,
        graph: networkx.DiGraph,
        start_nodes: Set[Node],
        end_nodes: Set[Node],
    ) -> EpsilonNFA:
        """
        Generate an ε-NFA from graph.
        """
        enfa = FastENFA()

        for (from_node, to_node, label) in graph.edges(data="label"):
            if label is None:
                sym = Epsilon()
            else:
                sym = Symbol(label)

            enfa.add_transition(State(from_node), sym, State(to_node))

        # In order to keep recursive constraints, we mark generated
        # type-variables as also final states
        for node in graph.nodes():
            if node.base.base.startswith("τ"):
                enfa.add_transition(State(node), Epsilon(), self.FINAL)

        enfa.add_start_state(self.START)
        enfa.add_final_state(self.FINAL)

        for start in start_nodes:
            enfa.add_transition(self.START, Symbol(start), State(start))

        for end in end_nodes:
            enfa.add_transition(State(end), Symbol(end), self.FINAL)

        return enfa

    def _generate_constraints_from_to_internal(
        self,
        graph: networkx.DiGraph,
        start_nodes: Set[Node],
        end_nodes: Set[Node],
    ) -> ConstraintSet:
        """
        Treat the graph as a ε-NFA, then convert to a DFA and subsequent minimal
        DFA. Compute path labels between start/ends over minimized DFA.
        """
        enfa = self._graph_to_dfa(graph, start_nodes, end_nodes)
        mdfa = enfa.minimize()
        dfa_g = mdfa.to_networkx()

        constraints = ConstraintSet()

        for final_state in mdfa.final_states:
            for path in networkx.all_simple_edge_paths(
                dfa_g, mdfa.start_state, final_state
            ):
                path_labels = [
                    dfa_g.get_edge_data(s, e)[index]["label"]
                    for s, e, index in path
                ]
                start_node = path_labels[0]
                end_node = path_labels[-1]

                # In minimized form these might be created, so we have to
                # explicitly check even though our original NFA would not have
                # these
                if not isinstance(start_node, Node) or not isinstance(
                    end_node, Node
                ):
                    continue

                constraint = _maybe_constraint(
                    start_node, end_node, path_labels[1:-1]
                )

                if constraint:
                    constraints.add(constraint)

        return constraints


class PathExprGraphSolver(GraphSolver):
    @staticmethod
    def cross_concatenation(
        prefix_list: List[List[Any]], postfix_list: List[List[Any]]
    ) -> List[List[Any]]:
        """
        Compute the cross product concatenation of two lists of lists.
        """
        combined = []
        for prefix in prefix_list:
            for postfix in postfix_list:
                combined.append(prefix + postfix)
        return combined

    @classmethod
    def enumerate_non_looping_paths(
        cls, path_expr: RExp
    ) -> List[List[EdgeLabel]]:
        """
        Given a path expression, return a list of all the paths
        that do not involve loops.
        """
        if path_expr.label == RExp.Label.NULL:
            return []
        elif path_expr.label == RExp.Label.EMPTY:
            return [[]]
        elif path_expr.label == RExp.Label.NODE:
            return [[path_expr.data]]
        # ignore looping paths
        elif path_expr.label == RExp.Label.STAR:
            return [[]]
        elif path_expr.label == RExp.Label.DOT:
            paths = [[]]
            for child in path_expr.children:
                paths = cls.cross_concatenation(
                    paths, cls.enumerate_non_looping_paths(child)
                )
            return paths
        elif path_expr.label == RExp.Label.OR:
            paths = []
            for child in path_expr.children:
                paths.extend(cls.enumerate_non_looping_paths(child))
            return paths
        else:
            assert False

    def _generate_constraints_from_to_internal(
        self,
        graph: networkx.DiGraph,
        start_nodes: Set[Node],
        end_nodes: Set[Node],
    ) -> ConstraintSet:
        """
        Generate constraints based on the computation of path expressions.
        Compute path expressions for each pair of start and end nodes.
        For each path expression, enumerate non-looping paths.
        """
        lattice_types = self.program.types.atomic_types
        numbering, path_seq = scc_decompose_path_seq(graph, "label")
        constraints = ConstraintSet()
        for start_node in start_nodes:
            path_exprs = solve_paths_from(path_seq, numbering[start_node])
            for end_node in end_nodes:
                if (
                    start_node.base in lattice_types
                    and end_node.base in lattice_types
                ):
                    continue
                indices = (numbering[start_node], numbering[end_node])
                path_expr = path_exprs[indices]
                for path in self.enumerate_non_looping_paths(path_expr):
                    constraint = _maybe_constraint(start_node, end_node, path)
                    if constraint:
                        constraints.add(constraint)
        return constraints


class NaiveGraphSolver(GraphSolver):
    def _generate_constraints_from_to_internal(
        self,
        graph: networkx.DiGraph,
        start_nodes: Set[Node],
        end_nodes: Set[Node],
    ) -> ConstraintSet:
        """
        Generate constraints based on the naive exploration of the graph.
        """
        lattice_types = self.program.types.atomic_types
        constraints = ConstraintSet()
        npaths = 0
        # On large procedures, the graph this is exploring can be quite large (hundreds of nodes,
        # thousands of edges). This can result in an insane number of paths - most of which do not
        # result in a constraint, and most of the ones that do result in constraints are redundant.
        def explore(
            current_node: Node,
            path: List[Node] = [],
            string: List[EdgeLabel] = [],
        ) -> None:
            """Find all non-empty paths that begin at start_nodes and end at end_nodes. Return
            the list of labels encountered along the way as well as the current_node and destination.
            """
            nonlocal max_paths_per_root
            nonlocal npaths
            if len(path) > self.config.max_path_length:
                return
            if npaths > max_paths_per_root:
                return
            if path and current_node in end_nodes:
                if (
                    current_node.base in lattice_types
                    and path[0] in lattice_types
                ):
                    return
                constraint = _maybe_constraint(path[0], current_node, string)

                if constraint:
                    constraints.add(constraint)
                npaths += 1
                return
            if current_node in path:
                npaths += 1
                return

            path = list(path)
            path.append(current_node)
            if current_node in graph:
                for succ in graph[current_node]:
                    label = graph[current_node][succ].get("label")
                    new_string = list(string)
                    if label:
                        new_string.append(label)
                    explore(succ, path, new_string)

        # We evenly distribute the maximum number of paths that we are willing to explore
        # across all origin nodes here.
        max_paths_per_root = int(
            min(
                self.config.max_paths_per_root,
                self.config.max_total_paths / float(len(start_nodes) + 1),
            )
        )
        for origin in start_nodes:
            npaths = 0
            explore(origin)
        return constraints
