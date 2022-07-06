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

"""The driver for the retypd analysis.
"""

from __future__ import annotations
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Any

from .pathexpr import RExp, scc_decompose_path_seq, solve_paths_from
from .graph import EdgeLabel, Node, ConstraintGraph, remove_unreachable_states
from .schema import (
    ConstraintSet,
    DerivedTypeVariable,
    FreshVarFactory,
    Lattice,
    Program,
    SubtypeConstraint,
    Variance,
    LoadLabel,
    StoreLabel,
)
from .global_handler import (
    GlobalHandler,
    PreciseGlobalHandler,
    UnionGlobalHandler,
)

from .sketches import LabelNode, SketchNode, Sketches
from .loggable import Loggable, LogLevel
import networkx
import tqdm
from dataclasses import dataclass
from graphviz import Digraph


def dump_labeled_graph(graph, label, filename):
    G = Digraph(label)
    G.attr(label=label, labeljust="l", labelloc="t")
    nodes = {}
    for i, n in enumerate(graph.nodes):
        nodes[n] = i
        G.node(f"n{i}", label=str(n))
    for head, tail in graph.edges:
        label = str(graph.get_edge_data(head, tail).get("label", "<NO LABEL>"))
        G.edge(f"n{nodes[head]}", f"n{nodes[tail]}", label=label)
    G.render(filename, format="svg", view=False)


@dataclass
class SolverConfig:
    """
    Parameters that change how the type solver behaves.
    """

    # Maximum path length when converts the constraint graph into output constraints
    max_path_length: int = 2**64
    # Maximum number of paths to explore per type variable root when generating output constraints
    max_paths_per_root: int = 2**64
    # Maximum paths total to explore per SCC
    max_total_paths: int = 2**64
    # Use path expressions or naive exploration of the graph.
    use_path_expressions: bool = False
    # Restrict graph to reachable nodes from and to endpoints
    # before doing the path exploration.
    restrict_graph_to_reachable: bool = True
    # More precise global handling
    # By default, we propagate globals up the callgraph, inlining them into the sketches as we
    # go, and the global sketches in the final (synthetic) root of the callgraph are the results
    # However, this can be slow and for large binaries the extra precision may not be worth the
    # time cost. If this is set to False we do a single unification of globals at the end, instead
    # of pulling them up the callgraph.
    precise_globals: bool = True


class EquivRelation:
    """
    This class represents an equivalence relation
    that can be computed incrementally

    TODO: Consider reimplementing using disjoint-set data structure
    for better performance.
    """

    def __init__(self, elems: Set[DerivedTypeVariable]) -> None:
        self._equiv_repr = {elem: frozenset((elem,)) for elem in elems}

    def make_equiv(
        self,
        x: FrozenSet[DerivedTypeVariable],
        y: FrozenSet[DerivedTypeVariable],
    ) -> None:
        """
        Merge the equivalence classes of x and y.
        """
        new_set = x | y
        for elem in new_set:
            self._equiv_repr[elem] = new_set

    def find_equiv_rep(
        self, x: DerivedTypeVariable
    ) -> Optional[Set[DerivedTypeVariable]]:
        """
        Return the equivalence class of x.
        """
        return self._equiv_repr.get(x)

    def get_equivalence_classes(self) -> Set[FrozenSet[DerivedTypeVariable]]:
        """
        Return a set with all the equivalence classes
        (represented as frozen set) in the equivalence relation.
        """
        return set(self._equiv_repr.values())


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


def enumerate_non_looping_paths(path_expr: RExp) -> List[List[EdgeLabel]]:
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
            paths = cross_concatenation(
                paths, enumerate_non_looping_paths(child)
            )
        return paths
    # path_expr.label == RExp.Label.OR
    else:
        paths = []
        for child in path_expr.children:
            paths.extend(enumerate_non_looping_paths(child))
        return paths


# There are two main aspects created by the solver: output constraints and sketches. The output
# constraints are _intra-procedural_: the constraints for a function f() will not contain
# constraints from its callees or callers.
# Sketches are also intra-procedural, but they do _copy_ information. So for example a sketch for
# function f() will include information from its callees. It does _not_ include information from
# its callers.
# We do the main solve in two passes:
# * First, over the callgraph in reverse topological order, using the sketches to pass information
#   between functions (or SCCs of functions)
# * Then, over the dependence graph of global variables in reverse topological order
#
# These two passes are mostly identical except for which graph they are iterating and which
# constraints they use.
class Solver(Loggable):
    """Takes a program and generates subtype constraints. The constructor does not perform the
    computation; rather, :py:class:`Solver` objects are callable as thunks.
    """

    def __init__(
        self,
        program: Program,
        config: SolverConfig = SolverConfig(),
        verbose: LogLevel = LogLevel.QUIET,
    ) -> None:
        super(Solver, self).__init__(verbose)
        self.program = program
        # TODO possibly make these values shared across a function
        self.config = config

    @staticmethod
    def instantiate_calls(
        cs: ConstraintSet,
        sketch_map: Dict[DerivedTypeVariable, Sketches],
        type_schemes: Dict[DerivedTypeVariable, ConstraintSet],
        types: Lattice[DerivedTypeVariable],
    ) -> ConstraintSet:
        """
        For every constraint involving a procedure that has already been
        analyzed, generate constraints based on the type scheme and
        capability constraints based on the sketch.
        """
        fresh_var_factory = FreshVarFactory()
        callees = set()
        for dtv in cs.all_dtvs():
            if dtv.base_var in sketch_map:
                callees.add(dtv.base_var)

        new_constraints = ConstraintSet()
        for callee in callees:
            new_constraints |= sketch_map[
                callee
            ].instantiate_sketch_capabilities(callee, types, fresh_var_factory)
            new_constraints |= type_schemes[callee]
        return new_constraints

    @staticmethod
    def compute_quotient_graph(
        constraints: ConstraintSet,
    ) -> Tuple[EquivRelation, networkx.DiGraph]:
        """
        Compute the quotient graph corresponding to a set of
        constraints.
        This graph allows us to infer the capabilities of all
        the DTVs that appear in the constraints.

        This corresponds to the first half of Algorithm E.1 InferShapes
        in the original Retypd paper.
        """
        # create initial graph
        g = networkx.DiGraph()
        for dtv in constraints.all_dtvs():
            g.add_node(dtv)
            while len(dtv.path) > 0:
                prefix = dtv.largest_prefix
                g.add_edge(prefix, dtv, label=dtv.tail)
                dtv = prefix

        # compute quotient graph
        equiv = EquivRelation(g.nodes)

        def unify(
            x_class: FrozenSet[DerivedTypeVariable],
            y_class: FrozenSet[DerivedTypeVariable],
        ) -> None:
            """
            Unify two equivalent classes and all the successors
            that can be reached:
             - Throught the same label
             - Through a 'load' label in one and a 'store' label in the other.
            See UNIFY in Algorithm E.1 and proof of Theorem E.1
            """
            if x_class != y_class:
                equiv.make_equiv(x_class, y_class)
                for (_, dest, label) in g.out_edges(x_class, data="label"):
                    if label is not None:
                        for (_, dest2, label2) in g.out_edges(
                            y_class, data="label"
                        ):
                            # The second condition does not appear in the
                            # Algorithm E.1 in the paper but it does appear
                            # in the proof of Theorem E.1.
                            if label2 == label or (
                                label == LoadLabel.instance()
                                and label2 == StoreLabel.instance()
                            ):
                                unify(
                                    equiv.find_equiv_rep(dest),
                                    equiv.find_equiv_rep(dest2),
                                )

        for constraint in constraints:
            unify(
                equiv.find_equiv_rep(constraint.left),
                equiv.find_equiv_rep(constraint.right),
            )

        return equiv, networkx.quotient_graph(
            g,
            equiv.get_equivalence_classes(),
            create_using=networkx.MultiDiGraph,
        )

    @staticmethod
    def infer_shapes(
        scc_and_globals: Set[DerivedTypeVariable],
        sketches: Sketches,
        constraints: ConstraintSet,
    ) -> None:
        """
        Infer shapes takes a set of constraints and populates shapes of the sketches
        for all DVS in scc.

        This corresponds to Algorithm E.1 'InferShapes' in the original Retypd paper.
        """
        if len(constraints) == 0:
            return
        equiv, g_quotient = Solver.compute_quotient_graph(constraints)

        # The paper says "By collapsing isomorphic subtrees, we can represent
        # sketches as deterministic finite state automata with each state labeled by
        # an element of /\ (the lattice)"
        # So we don't have to represent sketches as trees. We can have a graph
        # that summarizes all the trees. In that case, we could just take
        # the quotient graph and use it as a sketch automaton. The only change
        # needed is that we have to split some nodes in two so they can have Top or Bottom
        # depending on the paths that are taken to reach it (their variance).

        # In conclusion, we don't actually have enumerate all paths here!
        # just have an automaton that accepts the same paths as the quotient graph.

        # However, if we do this, we will have to split some of these nodes in
        # the future once they start having more detailed type information.
        # also this idea does not mesh up well with the current implementation of
        # sketches in which each node has 1 DTV associated. Technically
        # a SketchNode could have a set of DTVs associated (all the paths reaching a node,
        # one DTV per isomorphic subtree).

        # For now, create sketches that are trees pending a revision
        # of the implementation of sketches.
        def all_paths(
            curr_quotient_node: FrozenSet[DerivedTypeVariable],
            visited_nodes: Dict[FrozenSet[DerivedTypeVariable], SketchNode],
        ):
            """
            Explore all paths in the quotient graph starting from the curr_quotient_node.
            For each path, create the corresponding Sketch tree.

            Everytime we reach a new node, we create a new sketch node. If we reach
            a node that we have already visited, we create a label node instead.

            The `visited_nodes` dictionary captures the set of visited nodes so far
            and their correspondence to sketch nodes.
            """
            curr_node = visited_nodes[curr_quotient_node]
            for _, dest, label in set(
                g_quotient.out_edges(curr_quotient_node, data="label")
            ):
                if dest not in visited_nodes:
                    dest_dtv = curr_node.dtv.add_suffix(label)
                    dest_node = sketches.make_node(dest_dtv)
                    sketches.add_edge(curr_node, dest_node, label)
                    visited_nodes[dest] = dest_node
                    all_paths(dest, visited_nodes)
                    del visited_nodes[dest]
                else:
                    label_node = LabelNode(visited_nodes[dest].dtv)
                    sketches.add_edge(curr_node, label_node, label)

        for proc_or_global in scc_and_globals:
            proc_or_global_node = sketches.make_node(proc_or_global)
            quotient_node = equiv.find_equiv_rep(proc_or_global)
            if quotient_node is None:
                continue
            visited_nodes = {quotient_node: proc_or_global_node}
            all_paths(quotient_node, visited_nodes)

    # Regular language: RECALL*FORGET*  (i.e., FORGET cannot precede RECALL)
    @staticmethod
    def _recall_forget_split(graph: networkx.DiGraph) -> None:
        """The algorithm, after saturation, only admits paths such that recall edges all precede
        the first forget edge (if there is such an edge). To enforce this, we modify the graph by
        splitting each node and the unlabeled and forget edges (but not recall edges!). Forget edges
        in the original graph are changed to point to the 'forgotten' duplicate of their original
        target. As a result, no recall edges are reachable after traversing a single forget edge.
        """
        for head, tail in list(graph.edges):
            atts = graph[head][tail]
            label = atts.get("label")
            if label and label.kind == EdgeLabel.Kind.RECALL:
                continue
            forget_head = head.split_recall_forget()
            forget_tail = tail.split_recall_forget()
            if label and label.kind == EdgeLabel.Kind.FORGET:
                graph.remove_edge(head, tail)
                graph.add_edge(head, forget_tail, **atts)
            graph.add_edge(forget_head, forget_tail, **atts)

    @staticmethod
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

    def _pathexpr_generate_constraints_from_to(
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
        numbering, path_seq = scc_decompose_path_seq(graph, "label")
        constraints = ConstraintSet()
        for start_node in start_nodes:
            path_exprs = solve_paths_from(path_seq, numbering[start_node])
            for end_node in end_nodes:
                indices = (numbering[start_node], numbering[end_node])
                path_expr = path_exprs[indices]
                for path in enumerate_non_looping_paths(path_expr):
                    constraint = Solver._maybe_constraint(
                        start_node, end_node, path
                    )
                    if constraint:
                        constraints.add(constraint)
        return constraints

    def _naive_generate_constraints_from_to(
        self,
        graph: networkx.DiGraph,
        start_nodes: Set[Node],
        end_nodes: Set[Node],
    ) -> ConstraintSet:
        """
        Generate constraints based on the naive exploration of the graph.
        """
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
                constraint = self._maybe_constraint(
                    path[0], current_node, string
                )
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

    def _generate_constraints_from_to(
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

        if len(graph) == 0:
            return ConstraintSet()
        if self.config.use_path_expressions:
            return self._pathexpr_generate_constraints_from_to(
                graph, start_nodes, end_nodes
            )
        else:
            return self._naive_generate_constraints_from_to(
                graph, start_nodes, end_nodes
            )

    @staticmethod
    def _generate_type_vars(
        graph: networkx.DiGraph, interesting_nodes: Set[Node]
    ) -> Set[DerivedTypeVariable]:
        """Identify at least one node in each nontrivial SCC and generate a type variable for it.
        This ensures that we capture recursive types even if they are not part
        of a path between interesting variables.
        """

        def collect_recursive_sccs(graph: networkx.DiGraph):
            condensation = networkx.condensation(graph)
            recursive_sccs: List[networkx.DiGraph] = []
            # collect recursive SCCs
            for scc_node in condensation.nodes:
                scc_nodes = condensation.nodes[scc_node]["members"]
                if len(scc_nodes) == 1:
                    continue
                recursive_sccs.append(graph.subgraph(scc_nodes).copy())
            return recursive_sccs

        graph = graph.copy()
        # we are only interested in cycles remaining without crossing interesting variables.
        graph.remove_nodes_from(interesting_nodes)
        recursive_sccs = collect_recursive_sccs(graph)
        # greedily break SCCs
        type_vars = set()
        while len(recursive_sccs) > 0:
            recursive_scc_graph = recursive_sccs.pop()
            candidates = set()
            for (src, dest, label) in recursive_scc_graph.edges(data="label"):
                if label is not None:
                    if label.kind == EdgeLabel.Kind.FORGET:
                        candidates.add(dest)
                    else:
                        candidates.add(src)
            # ignore SCCs without labeled paths
            # e.g. A <= B, B <= C
            # we don't want type vars for those
            if len(candidates) == 0:
                continue
            # prefer DTVs with shortest paths
            best_candidate = min(candidates, key=lambda x: x.base)
            type_vars.add(best_candidate.base)
            recursive_scc_graph.remove_node(best_candidate)
            recursive_sccs = (
                collect_recursive_sccs(recursive_scc_graph) + recursive_sccs
            )

        return type_vars

    @staticmethod
    def get_start_nodes(
        graph: networkx.Digraph, dtvs: Set[DerivedTypeVariable]
    ) -> Set[Node]:
        """
        We allow start  nodes to be both PRE_FORGET and POST_FORGET
        because we could have paths of the form FORGET* (without any recalls).
        """
        return {node for node in graph.nodes if node.base in dtvs}

    @staticmethod
    def get_end_nodes(
        graph: networkx.Digraph, dtvs: Set[DerivedTypeVariable]
    ) -> Set[Node]:
        """
        We allow end_nodes to be both PRE_FORGET and POST_FORGET
        because we could have paths of the form RECALL* (without any forgets).
        """
        return {node for node in graph.nodes if node.base in dtvs}

    @staticmethod
    def substitute_type_vars(
        constraints: ConstraintSet, type_vars: Set[DerivedTypeVariable]
    ) -> ConstraintSet:
        modified_cs = ConstraintSet()
        type_var_map = {}
        for i, type_var in enumerate(type_vars):
            type_var_map[type_var] = DerivedTypeVariable(f"τ${i}")
        left_suffix = None
        right_suffix = None
        for cs in constraints:
            for type_var in type_vars:
                left_suffix = type_var.get_suffix(cs.left)
                if left_suffix is not None:
                    left_base = type_var_map[type_var]
                    break
            for type_var in type_vars:
                right_suffix = type_var.get_suffix(cs.right)
                if right_suffix is not None:
                    right_base = type_var_map[type_var]
                    break
            new_left = (
                left_base.extend(left_suffix)
                if left_suffix is not None
                else cs.left
            )
            new_right = (
                right_base.extend(right_suffix)
                if right_suffix is not None
                else cs.right
            )
            modified_cs.add(SubtypeConstraint(new_left, new_right))
        return modified_cs

    def _generate_type_scheme(
        self,
        graph: networkx.DiGraph,
        non_primitive_end_points: Set[DerivedTypeVariable],
        internal_types: Set[DerivedTypeVariable],
    ) -> ConstraintSet:
        """Generate reduced set of constraints
        that constitute a type scheme.
        """
        interesting_dtvs = non_primitive_end_points | internal_types
        all_interesting_nodes = {
            node for node in graph.nodes if node.base in interesting_dtvs
        }
        type_vars = self._generate_type_vars(graph, all_interesting_nodes)
        interesting_dtvs |= type_vars
        start_nodes = Solver.get_start_nodes(graph, interesting_dtvs)
        end_nodes = Solver.get_end_nodes(graph, interesting_dtvs)
        constraints = self._generate_constraints_from_to(
            graph, start_nodes, end_nodes
        )
        return Solver.substitute_type_vars(constraints, type_vars)

    def _generate_primitive_constraints(
        self,
        graph: networkx.DiGraph,
        non_primitive_end_points: Set[DerivedTypeVariable],
        internal_types: Set[DerivedTypeVariable],
    ) -> ConstraintSet:
        """Generate constraints to populate
        the sketch nodes with primitive types

        We explore paths:
         - From internal_types to non_primitive_end_points.
         - From non_primitive_end_points to internal_types.
        """
        constraints = ConstraintSet()

        # from proc and global vars to primitive types
        start_nodes = Solver.get_start_nodes(graph, non_primitive_end_points)
        end_nodes = Solver.get_end_nodes(graph, internal_types)
        constraints |= self._generate_constraints_from_to(
            graph, start_nodes, end_nodes
        )

        # from primitive types to proc and global vars
        start_nodes = Solver.get_start_nodes(graph, internal_types)
        end_nodes = Solver.get_end_nodes(graph, non_primitive_end_points)
        constraints |= self._generate_constraints_from_to(
            graph, start_nodes, end_nodes
        )
        return constraints

    def _solve_topo_graph(
        self,
        global_handler: GlobalHandler,
        scc_dag: networkx.DiGraph,
        constraint_map: Dict[Any, ConstraintSet],
        sketches_map: Dict[DerivedTypeVariable, Sketches],
        type_schemes: Dict[DerivedTypeVariable, ConstraintSet],
    ):
        """
        For each SCC we:
        - Get the constraints of each of the procedures in the SCC
        - Add constraints representing all the information of the callees (instantiate_calls)
        - Using those constraints, we `infer_shapes`, which populates sketches with all the capabilities
          that they have but no primitive types.
        - Build the constraint graph and use it to infer final constraints
        - Add the information of the final constraints to the pre-populated sketches.
        """

        def show_progress(iterable):
            if self.verbose:
                return tqdm.tqdm(iterable)
            return iterable

        for scc_node in show_progress(
            reversed(list(networkx.topological_sort(scc_dag)))
        ):
            global_handler.pre_scc(scc_node)
            scc = scc_dag.nodes[scc_node]["members"]
            scc_initial_constraints = ConstraintSet()
            for proc in scc:
                constraints = constraint_map.get(proc, ConstraintSet())
                constraints |= Solver.instantiate_calls(
                    constraints, sketches_map, type_schemes, self.program.types
                )
                scc_initial_constraints |= constraints

            self.debug("# Processing SCC: %s", "_".join([str(s) for s in scc]))

            scc_sketches = Sketches(self.program.types, self.verbose)
            Solver.infer_shapes(
                scc | self.program.global_vars,
                scc_sketches,
                scc_initial_constraints,
            )
            non_primitive_endpoints = frozenset(
                scc | set(self.program.global_vars)
            )

            graph = ConstraintGraph(scc_initial_constraints).graph
            # Uncomment to output graph for debugging
            # dump_labeled_graph(graph, "graph", f"/tmp/scc_graph")
            Solver._recall_forget_split(graph)

            type_scheme = self._generate_type_scheme(
                graph,
                non_primitive_endpoints,
                self.program.types.internal_types,
            )
            primitive_constraints = self._generate_primitive_constraints(
                graph,
                non_primitive_endpoints,
                self.program.types.internal_types,
            )
            scc_sketches.add_constraints(primitive_constraints)

            # Copy globals from our callees, if we are analyzing globals precisely.
            global_handler.copy_globals(scc_sketches, scc, sketches_map)

            for proc in scc:
                sketches_map[proc] = scc_sketches
                type_schemes[proc] = type_scheme

            global_handler.post_scc(scc_node, sketches_map)

    def __call__(
        self,
    ) -> Tuple[
        Dict[DerivedTypeVariable, ConstraintSet],
        Dict[DerivedTypeVariable, Sketches],
    ]:
        """Perform the retypd calculation."""

        type_schemes: Dict[DerivedTypeVariable, ConstraintSet] = {}
        sketches_map: Dict[DerivedTypeVariable, Sketches] = {}

        def find_roots(digraph: networkx.DiGraph):
            roots = []
            for n in digraph.nodes:
                if digraph.in_degree(n) == 0:
                    roots.append(n)
            return roots

        # The idea here is that functions need to satisfy their clients' needs for inputs and need
        # to use their clients' return values without overextending them. So we do a reverse
        # topological order on functions, lumping SCCs together, and slowly extend the graph.
        # It would be interesting to explore how this algorithm compares with a more restricted one
        # as far as asymptotic complexity, but this is simple to understand and so the right choice
        # for now. I suspect that most of the graphs are fairly sparse and so the practical reality
        # is that there aren't many paths.
        self.info("Solving functions")
        callgraph = networkx.DiGraph(self.program.callgraph)
        fake_root = DerivedTypeVariable("$$FAKEROOT$$")
        for r in find_roots(callgraph):
            callgraph.add_edge(fake_root, r)
        scc_dag = networkx.condensation(callgraph)

        if self.config.precise_globals:
            global_handler = PreciseGlobalHandler(
                self.program.global_vars, callgraph, scc_dag, fake_root
            )
        else:
            global_handler = UnionGlobalHandler(
                self.program.global_vars, callgraph, scc_dag, fake_root
            )

        self._solve_topo_graph(
            global_handler,
            scc_dag,
            self.program.proc_constraints,
            sketches_map,
            type_schemes,
        )
        global_handler.finalize(self, sketches_map)

        # Note: all globals point to the same "Sketches" graph, which has all the globals
        # in it. It would be nice to separate them out, but not a priority right now (clients
        # can do it easily).
        for g in self.program.global_vars:
            type_schemes[g] = type_schemes[fake_root]
            sketches_map[g] = sketches_map[fake_root]
        if fake_root in type_schemes:
            del type_schemes[fake_root]
        if fake_root in sketches_map:
            del sketches_map[fake_root]

        return (type_schemes, sketches_map)
