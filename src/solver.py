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
from collections import defaultdict
from typing import AbstractSet, Dict, FrozenSet, List, Optional, Set, Tuple
from .graph import (
    EdgeLabel,
    SideMark,
    Node,
    ConstraintGraph,
)
from .graph_solver import (
    GraphSolverConfig,
    DFAGraphSolver,
    PathExprGraphSolver,
    NaiveGraphSolver,
)
from .schema import (
    AccessPathLabel,
    ConstraintSet,
    DerivedTypeVariable,
    FreshVarFactory,
    Program,
    LoadLabel,
    StoreLabel,
    SubtypeConstraint,
)
from .global_handler import (
    GlobalHandler,
    PreciseGlobalHandler,
    UnionGlobalHandler,
)

from .sketches import LabelNode, SketchNode, Sketches, SkNode
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

    if isinstance(graph, networkx.MultiGraph):
        for head, tail, label in graph.edges(data="label"):
            G.edge(f"n{nodes[head]}", f"n{nodes[tail]}", label=str(label))
    else:
        for head, tail in graph.edges:
            label = str(
                graph.get_edge_data(head, tail).get("label", "<NO LABEL>")
            )
            G.edge(f"n{nodes[head]}", f"n{nodes[tail]}", label=label)
    G.render(filename, format="svg", view=False)


@dataclass
class SolverConfig:
    """
    Parameters that change how the type solver behaves.
    """

    # Use `naive`, `pathexpr`, or `dfa` f
    graph_solver: str = "dfa"
    # Graph solver configuration
    graph_solver_config: GraphSolverConfig = GraphSolverConfig()
    # More precise global handling
    # By default, we propagate globals up the callgraph, inlining them into the sketches as we
    # go, and the global sketches in the final (synthetic) root of the callgraph are the results
    # However, this can be slow and for large binaries the extra precision may not be worth the
    # time cost. If this is set to False we do a single unification of globals at the end, instead
    # of pulling them up the callgraph.
    precise_globals: bool = True
    # After the initial bottom-up face, an optional pass to take the most general types of those
    # functions and re-propagating to make most specific types at their callsites.
    top_down_propagation: bool = False


class EquivRelation:
    """
    This class represents an equivalence relation
    that can be computed incrementally

    TODO: Consider reimplementing using disjoint-set data structure
    for better performance.
    """

    def __init__(self, elems: AbstractSet[DerivedTypeVariable]) -> None:
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
    ) -> Optional[FrozenSet[DerivedTypeVariable]]:
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
        if config.graph_solver == "dfa":
            self.graph_solver = DFAGraphSolver(config.graph_solver_config)
        elif config.graph_solver == "pathexpr":
            self.graph_solver = PathExprGraphSolver(config.graph_solver_config)
        elif config.graph_solver == "naive":
            self.graph_solver = NaiveGraphSolver(config.graph_solver_config)
        else:
            raise ValueError(f"Unknown graph solver {config.graph_solver}")

    @staticmethod
    def specialize_temporaries(
        base: DerivedTypeVariable,
        constraints: ConstraintSet,
        interesting_vars: Set[DerivedTypeVariable],
    ) -> ConstraintSet:
        """
        Specialize temporary variables to a specific function
        """

        def fix_dtv(dtv: DerivedTypeVariable) -> DerivedTypeVariable:
            if DerivedTypeVariable(dtv.base) not in interesting_vars:
                return DerivedTypeVariable(f"{base}${dtv.base}", dtv.path)
            else:
                return dtv

        output_cs = ConstraintSet()

        for constraint in constraints:
            output_cs.add(
                SubtypeConstraint(
                    fix_dtv(constraint.left), fix_dtv(constraint.right)
                )
            )

        return output_cs

    @staticmethod
    def instantiate_type_scheme(
        fresh_var_factory: FreshVarFactory, type_scheme: ConstraintSet
    ) -> ConstraintSet:
        """
        Instantiate a type scheme by renaming the anonymous variables
        in the type scheme with fresh names in the current SCC context.

        This guarantees that there will be no naming conflics from anonymous variables
        coming from different calls.
        """
        anonymous_vars = {
            dtv.base_var
            for dtv in type_scheme.all_dtvs()
            if fresh_var_factory.is_anonymous_variable(dtv)
        }
        # sort to avoid non-determinism
        instantiation_map = {
            original: fresh_var_factory.fresh_var()
            for original in sorted(anonymous_vars)
        }
        return type_scheme.apply_mapping(instantiation_map)

    @staticmethod
    def instantiate_calls(
        cs: ConstraintSet,
        sketch_map: Dict[DerivedTypeVariable, Sketches],
        type_schemes: Dict[DerivedTypeVariable, ConstraintSet],
    ) -> ConstraintSet:
        """
        For every constraint involving a procedure that has already been
        analyzed, generate constraints based on the type scheme and
        capability constraints based on the sketch.
        """
        fresh_var_factory = FreshVarFactory()

        # TODO in order to support different instantiations for different calls
        # to the same function, we need to encode function actuals
        # differently than function formals
        callees = {tv for tv in cs.all_tvs() if tv in sketch_map}

        new_constraints = ConstraintSet()
        # sort to avoid non-determinism
        for callee in sorted(callees):
            new_constraints |= sketch_map[
                callee
            ].instantiate_sketch_capabilities(callee, fresh_var_factory)
            new_constraints |= Solver.instantiate_type_scheme(
                fresh_var_factory, type_schemes[callee]
            )
        return new_constraints

    @staticmethod
    def compute_quotient_graph(
        constraints: ConstraintSet,
        lattice_types: Set[DerivedTypeVariable],
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
                assert prefix, "Failed to calculate largest prefix"
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
             - Through the same label
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

        # We always have 'a <= a' (S-Refl) so if a.load and a.store exist,
        # we have an implicit rule 'a.load <= a.store' (S-Pointer) and we need to unify a.load and a.store.
        for node in g.nodes:
            out_edges = {
                label: dest
                for (_, dest, label) in g.out_edges(node, data="label")
            }
            if (
                LoadLabel.instance() in out_edges
                and StoreLabel.instance() in out_edges
            ):
                unify(
                    equiv.find_equiv_rep(out_edges[LoadLabel.instance()]),
                    equiv.find_equiv_rep(out_edges[StoreLabel.instance()]),
                )

        for constraint in constraints:
            # Don't unify across lattice types
            if (
                constraint.left in lattice_types
                or constraint.right in lattice_types
            ):
                continue
            unify(
                equiv.find_equiv_rep(constraint.left),
                equiv.find_equiv_rep(constraint.right),
            )

        # compute quotient graph
        # networkx.quotient_graph does not support self-edges
        # https://github.com/networkx/networkx/issues/5853
        # and is very inefficient
        # https://github.com/networkx/networkx/issues/4935
        quotient_g = networkx.MultiDiGraph()
        quotient_g.add_nodes_from(equiv.get_equivalence_classes())
        for (src, dest, label) in g.edges(data="label"):
            src_class = equiv.find_equiv_rep(src)
            dest_class = equiv.find_equiv_rep(dest)
            edge_data = quotient_g.get_edge_data(src_class, dest_class)
            if edge_data is None or label not in {
                data["label"] for data in edge_data.values()
            }:
                quotient_g.add_edge(src_class, dest_class, label=label)
        return equiv, quotient_g

    @staticmethod
    def infer_shapes(
        proc_or_globals: Set[DerivedTypeVariable],
        sketches: Sketches,
        constraints: ConstraintSet,
        lattice_types: FrozenSet[DerivedTypeVariable],
    ) -> None:
        """
        Infer shapes takes a set of constraints and populates shapes of the sketches
        for all DTVs in the program.

        This corresponds to Algorithm E.1 'InferShapes' in the original Retypd paper.
        """
        if len(constraints) == 0:
            return
        equiv, g_quotient = Solver.compute_quotient_graph(
            constraints, lattice_types
        )

        # Uncomment to generate graph for debugging
        # dump_labeled_graph(g_quotient,"quotient","/tmp/quotient")

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

        for proc_or_global in proc_or_globals:
            proc_or_global_node = sketches.make_node(proc_or_global)
            quotient_node = equiv.find_equiv_rep(proc_or_global)
            if quotient_node is None:
                continue
            visited_nodes = {quotient_node: proc_or_global_node}
            all_paths(quotient_node, visited_nodes)

    @staticmethod
    def _generate_type_vars(
        graph: networkx.DiGraph, interesting_nodes: Set[Node]
    ) -> Set[DerivedTypeVariable]:
        """
        Select a set of DTVs that should become type variables to break
        any cycles in the graph with non-empty paths.
        This allows us to capture all the recursive type information.

        The set of type variables is not guaranteed to be minimal since computing
        that "Minimum feedback vertex set" is NP-complete.
        The approach taken here is greedy, we increasingly add type vars
        which breaks SCCs into smaller SCCs until all SCCs are broken.
        The selection of the next type var is heuristic but guarantees
        that at each step we break at least one non-empty cycle.
        """

        def collect_recursive_sccs(
            graph: networkx.DiGraph,
        ) -> List[networkx.DiGraph]:
            """
            Given a graph, collect the set of recursive SCCs
            of size greater than 1.
            """
            condensation = networkx.condensation(graph)
            recursive_sccs: List[networkx.DiGraph] = []
            # collect recursive SCCs
            for scc_node in condensation.nodes:
                scc_nodes = condensation.nodes[scc_node]["members"]
                if len(scc_nodes) == 1:
                    continue
                recursive_sccs.append(graph.subgraph(scc_nodes).copy())
            return recursive_sccs

        # we are only interested in cycles remaining without crossing interesting variables.
        graph = graph.copy()
        graph.remove_nodes_from(interesting_nodes)
        recursive_sccs = collect_recursive_sccs(graph)
        # greedily break SCCs
        type_vars = set()
        while len(recursive_sccs) > 0:
            recursive_scc_graph = recursive_sccs.pop()
            # collect at least one node involved in each non-empty path
            candidates = set()
            for (src, dest, label) in recursive_scc_graph.edges(data="label"):
                if label is not None:
                    # choose the one with shorter prefix
                    if label.kind == EdgeLabel.Kind.FORGET:
                        candidates.add(dest)
                    else:
                        candidates.add(src)
            # ignore SCCs without labeled paths
            # e.g. A <= B, B <= C
            # we do not need type vars for those
            if len(candidates) == 0:
                continue
            # prefer DTVs with shortest capabilities
            best_candidate = min(candidates, key=lambda x: x.base)
            type_vars.add(best_candidate.base)
            recursive_scc_graph.remove_node(best_candidate)
            recursive_sccs = (
                collect_recursive_sccs(recursive_scc_graph) + recursive_sccs
            )

        return type_vars

    @staticmethod
    def get_start_end_nodes(
        graph: networkx.DiGraph,
        start_dtvs: AbstractSet[DerivedTypeVariable],
        end_dtvs: AbstractSet[DerivedTypeVariable],
    ) -> Tuple[Set[Node], Set[Node]]:
        """
        Obtain the start and end graph nodes corresponding to the given
        sets of DTVs.

        We allow start  nodes to be both PRE_FORGET and POST_FORGET
        because we could have paths of the form FORGET* (without any recalls).
        We allow end_nodes to be both PRE_FORGET and POST_FORGET
        because we could have paths of the form RECALL* (without any forgets).
        """
        start_nodes = set()
        end_nodes = set()
        for node in graph.nodes:
            if node.base in start_dtvs and node.side_mark == SideMark.LEFT:
                start_nodes.add(node)
            if node.base in end_dtvs and node.side_mark == SideMark.RIGHT:
                end_nodes.add(node)
        return start_nodes, end_nodes

    @staticmethod
    def substitute_type_vars(
        constraints: ConstraintSet, type_vars: Set[DerivedTypeVariable]
    ) -> ConstraintSet:
        """
        The type variables in `type_vars` become existentially quantified
        anonymous variables in the returned constraint set.
        """
        fresh_var_factory = FreshVarFactory()
        # assign names to type vars
        # sort to avoid non-determinism
        type_var_map = {
            type_var: fresh_var_factory.fresh_var()
            for type_var in sorted(type_vars)
        }
        return constraints.apply_mapping(type_var_map)

    def _filter_derived_lattices(self, cs: ConstraintSet) -> ConstraintSet:
        """
        Remove constraints which involve derive type variables from lattice elements
        For example, int64.in_0 is something that would be removed. These are th byproduct of
        invalid constraints being fed to Retypd.
        """

        def is_derived(dtv: DerivedTypeVariable) -> bool:
            return (
                DerivedTypeVariable(dtv.base)
                in self.program.types.internal_types
                and len(dtv.path) > 0
            )

        output = ConstraintSet()

        for constraint in cs:
            if is_derived(constraint.left) or is_derived(constraint.right):
                continue
            output.add(constraint)

        return output

    def _solve_constraints_between(
        self,
        graph: networkx.DiGraph,
        start_dtvs: AbstractSet[DerivedTypeVariable],
        end_dtvs: AbstractSet[DerivedTypeVariable],
    ) -> ConstraintSet:
        """
        Get graph nodes from start/end DTVs, and solve for constraints that
        exist between those nodes in the graph
        """
        start_nodes, end_nodes = Solver.get_start_end_nodes(
            graph, start_dtvs, end_dtvs
        )
        unfiltered_constraints = (
            self.graph_solver.generate_constraints_from_to(
                graph, start_nodes, end_nodes
            )
        )
        return self._filter_derived_lattices(unfiltered_constraints)

    def _generate_type_scheme(
        self,
        initial_constraints: ConstraintSet,
        non_primitive_end_points: AbstractSet[DerivedTypeVariable],
        primitive_types: AbstractSet[DerivedTypeVariable],
    ) -> ConstraintSet:
        """Generate a reduced set of constraints
        that constitute a type scheme.

        These are constraints that relate:
            - The procedure and procedure arguments
            - Primitive types
            - Type variables capturing recursive types
        """
        interesting_dtvs = non_primitive_end_points | primitive_types
        graph = ConstraintGraph.from_constraints(
            initial_constraints,
            interesting_dtvs,
        )
        all_interesting_nodes = {
            node for node in graph.nodes if node.base in interesting_dtvs
        }
        type_vars = Solver._generate_type_vars(graph, all_interesting_nodes)

        if len(type_vars) > 0:
            interesting_dtvs |= type_vars

            # If we have type vars, we recompute the graph
            # considering type vars as interesting
            graph = ConstraintGraph.from_constraints(
                initial_constraints,
                interesting_dtvs,
            )
        # Uncomment to output graph for debugging
        # dump_labeled_graph(graph, "graph", f"/tmp/scc_graph")

        constraints = self._solve_constraints_between(
            graph, interesting_dtvs, interesting_dtvs
        )
        return Solver.substitute_type_vars(constraints, type_vars)

    def _generate_primitive_constraints(
        self,
        initial_constraints: ConstraintSet,
        non_primitive_end_points: AbstractSet[DerivedTypeVariable],
        primitive_types: AbstractSet[DerivedTypeVariable],
    ) -> ConstraintSet:
        """Generate constraints to populate
        the sketch nodes with primitive types

        We explore paths:
         - From primitive_types to non_primitive_end_points.
         - From non_primitive_end_points to primitive_types.
        """

        graph = ConstraintGraph.from_constraints(
            initial_constraints,
            non_primitive_end_points | primitive_types,
        )
        constraints = ConstraintSet()

        # from proc and global vars to primitive types
        constraints |= self._solve_constraints_between(
            graph, non_primitive_end_points, primitive_types
        )

        # from primitive types to proc and global vars
        constraints |= self._solve_constraints_between(
            graph, primitive_types, non_primitive_end_points
        )

        return constraints

    def _actual_in_outs(
        self,
        proc: DerivedTypeVariable,
        callgraph: networkx.DiGraph,
        sketch_map: Dict[DerivedTypeVariable, Sketches],
    ) -> Tuple[Set[SkNode], Set[SkNode]]:
        """
        Gather the set of sketch nodes for a procedure's actual ins and outs
        """
        incoming_procs = {in_edge for in_edge, _ in callgraph.in_edges(proc)}
        all_in_sketches = set()
        all_out_sketches = set()

        for incoming_proc in incoming_procs:
            if incoming_proc not in self.program.callgraph.nodes:
                continue

            in_sketches, out_sketches = sketch_map[
                incoming_proc
            ].in_out_sketches(proc)
            all_out_sketches |= out_sketches
            all_in_sketches |= in_sketches

        return all_in_sketches, all_out_sketches

    def _procedure_specialization(
        self,
        proc: DerivedTypeVariable,
        callgraph: networkx.DiGraph,
        sketch_map: Dict[DerivedTypeVariable, Sketches],
    ):
        """
        Do procedural specialization over the sketches, as defined in Algorithm F.3
        """
        dtv2node: Dict[DerivedTypeVariable, SketchNode] = {}

        formal_ins, formal_outs = sketch_map[proc].in_out_sketches(proc)
        actual_ins, actual_outs = self._actual_in_outs(
            proc, callgraph, sketch_map
        )

        def accumulate(nodes: Set[SkNode], accumulator):
            for node in nodes:
                if not isinstance(node, SketchNode):
                    continue

                if node.dtv not in dtv2node:
                    dtv2node[node.dtv] = node
                else:
                    dtv2node[node.dtv] = accumulator(dtv2node[node.dtv], node)

        # Join all actual ins
        accumulate(actual_ins, lambda x, y: x.join(y, self.program.types))

        # Meet all actual outs
        accumulate(actual_outs, lambda x, y: x.meet(y, self.program.types))

        # Merge into the formal ins
        accumulate(formal_ins, lambda x, y: y.meet(x, self.program.types))

        # Join into the formal outs
        accumulate(formal_outs, lambda x, y: y.join(x, self.program.types))

        # Update our sketch with the new information
        for dtv, node in dtv2node.items():
            ref_node = sketch_map[proc].lookup(dtv)

            if ref_node:
                # Update an existing sketch node to new bounds
                assert isinstance(ref_node, SketchNode)
                ref_node.lower_bound = node.lower_bound
                ref_node.upper_bound = node.upper_bound
            else:
                # Insert a new sketch node and its dependent nodes
                largest = dtv
                tails: List[AccessPathLabel] = []

                # Calculate the largest dtv that is in the sketch
                while largest and sketch_map[proc].lookup(largest) is None:
                    tails.append(largest.tail)
                    largest = largest.largest_prefix

                current = largest

                if not current:
                    current = DerivedTypeVariable(dtv.base)
                    tails = tails[:-1]

                tails = tails[::-1]

                # Insert sketches from the existing largest to the new node
                for tail in tails:
                    next = current.add_suffix(tail)
                    current_node = sketch_map[proc].lookup(current)

                    if not current_node:
                        current_node = sketch_map[proc].make_node(current)

                    next_node = sketch_map[proc].make_node(next)
                    sketch_map[proc].add_edge(current_node, next_node, tail)
                    current = next

                ref_node = sketch_map[proc].lookup(dtv)
                assert ref_node

            ref_node.lower_bound = node.lower_bound
            ref_node.upper_bound = node.upper_bound

        self.debug(
            "Finished refine parameters for %s with %s", proc, sketch_map[proc]
        )

    def _solve_topo_graph(
        self,
        global_handler: GlobalHandler,
        callgraph: networkx.DiGraph,
        scc_dag: networkx.DiGraph,
        sketches_map: Dict[DerivedTypeVariable, Sketches],
        type_schemes: Dict[DerivedTypeVariable, ConstraintSet],
    ):
        """
        For each SCC we:
        - Get the constraints of each of the procedures in the SCC
        - Add constraints representing all the information of the callees (instantiate_calls)
        - Using those constraints, we `infer_shapes`, which populates sketches with all the capabilities
          that they have but no primitive types.
        - Build the constraint graph and use it to:
           - Compute type schemes (to be used by later instantiate_calls)
           - Compute constraints to populate primitive type information in sketches.
        - Add primitive type information to the pre-populated sketches.
        """

        def show_progress(iterable):
            if self.verbose:
                return tqdm.tqdm(iterable)
            return iterable

        all_interesting = frozenset(callgraph.nodes | self.program.global_vars)
        scc_dag_topo = list(networkx.topological_sort(scc_dag))
        universal_schemas = {}

        for scc_node in show_progress(reversed(scc_dag_topo)):
            global_handler.pre_scc(scc_node)
            scc = scc_dag.nodes[scc_node]["members"]
            self.debug("# Processing SCC: %s", "_".join([str(s) for s in scc]))
            scc_initial_constraints = ConstraintSet()

            for proc in scc:
                self.debug("# Initializing call-sites for %s", proc)
                constraints = self.program.proc_constraints.get(
                    proc, ConstraintSet()
                )
                constraints = Solver.specialize_temporaries(
                    proc,
                    constraints,
                    all_interesting | self.program.types.atomic_types,
                )
                constraints |= Solver.instantiate_calls(
                    constraints, sketches_map, type_schemes
                )
                scc_initial_constraints |= constraints

            self.debug("# Inferring shapes")
            scc_sketches = Sketches(self.program.types, self.verbose)

            Solver.infer_shapes(
                all_interesting,
                scc_sketches,
                scc_initial_constraints,
                self.program.types.atomic_types,
            )

            for proc in scc:
                self.debug("# Inferring type scheme of proc: %s", proc)
                sketches_map[proc] = scc_sketches

                # Generate type scheme for all interesting nodes, since the sketches can have
                # other procedures' sketch nodes
                universal_type_scheme = self._generate_type_scheme(
                    scc_initial_constraints,
                    all_interesting,
                    self.program.types.internal_types,
                )
                universal_schemas[proc] = universal_type_scheme

                self.debug(
                    "# Inferring universal constraints for %s %s",
                    proc,
                    universal_type_scheme,
                )

                # Specialize type-scheme further for just this procedure
                type_schemes[proc] = self._generate_type_scheme(
                    universal_type_scheme,
                    frozenset({proc}),
                    self.program.types.internal_types,
                )

                self.debug(
                    "# Inferring primitive constraints of proc: %s", proc
                )

                # Generate primitive between all procedures and variables using the type scheme
                # defined for all procedures and variables.
                primitive_constraints = self._generate_primitive_constraints(
                    type_schemes[proc],
                    all_interesting,
                    self.program.types.internal_types,
                )

                self.debug("# Created for %s %s", proc, primitive_constraints)
                scc_sketches.add_constraints(primitive_constraints)

            involved_globals = (
                self.program.global_vars & scc_initial_constraints.all_tvs()
            )
            if len(involved_globals) > 0:
                self.debug(
                    "# Inferring primitive constraints of globals: %s", scc
                )
                # Compute primitive constraints for globals
                primitive_global_constraints = (
                    self._generate_primitive_constraints(
                        scc_initial_constraints,
                        self.program.global_vars,
                        self.program.types.internal_types,
                    )
                )
                scc_sketches.add_constraints(primitive_global_constraints)
            # Copy globals from our callees, if we are analyzing globals precisely.
            global_handler.copy_globals(scc_sketches, scc, sketches_map)
            global_handler.post_scc(scc_node, sketches_map)

        if not self.config.top_down_propagation:
            # We're dont after our initial bottom-up phase
            return

        self.debug("# Propgating top-down")
        prim_constraints = defaultdict(lambda: ConstraintSet())

        for scc_node in show_progress(scc_dag_topo):
            scc = scc_dag.nodes[scc_node]["members"]
            self.debug(
                "# Processing top-to-down SCC: %s",
                "_".join([str(s) for s in scc]),
            )

            for proc in scc:
                constraints = universal_schemas[proc]

                self.debug("# Starting proc %s with %s", proc, constraints)

                if proc in self.program.callgraph.nodes:
                    self._procedure_specialization(
                        proc, callgraph, sketches_map
                    )

                proc_vars = FreshVarFactory()
                constraints = Solver.instantiate_type_scheme(
                    proc_vars, constraints
                )
                callers = {in_edge for in_edge, _ in callgraph.in_edges(proc)}

                # sort to avoid non-determinism
                for caller in sorted(callers):
                    caller_type_scheme = Solver.instantiate_type_scheme(
                        proc_vars, universal_schemas[caller]
                    )

                    caller_prims = prim_constraints[caller]

                    self.debug(
                        "Integrating caller %s into %s with", caller, proc
                    )
                    self.debug("   - TYPE SCHEME: %s", caller_type_scheme)
                    self.debug("   - PRIMITIVES: %s", caller_prims)

                    constraints |= caller_type_scheme
                    constraints |= caller_prims

                self.debug(
                    "Final top-down constraints for %s is %s",
                    proc,
                    constraints,
                )

                sketch = Sketches(self.program.types, self.verbose)
                Solver.infer_shapes(
                    all_interesting,
                    sketch,
                    constraints,
                    self.program.types.atomic_types,
                )

                # Generate primitive between all procedures and variables using the type scheme
                # defined for all procedures and variables.
                type_scheme = self._generate_type_scheme(
                    constraints,
                    {proc},
                    self.program.types.internal_types,
                )

                # Generate primitive between all procedures and variables using the type scheme
                # defined for all procedures and variables.
                primitive_constraints = self._generate_primitive_constraints(
                    constraints,
                    {proc},
                    self.program.types.internal_types,
                )
                self.debug(
                    "# Created primitive constraints for %s %s",
                    proc,
                    primitive_constraints,
                )

                prim_constraints[proc] = primitive_constraints
                type_schemes[proc] = type_scheme
                sketches_map[proc] = sketch
                sketch.add_constraints(primitive_constraints)

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
            callgraph,
            scc_dag,
            sketches_map,
            type_schemes,
        )
        global_handler.finalize(self.program.types, sketches_map)

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
