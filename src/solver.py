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
    ConstraintSet,
    DerivedTypeVariable,
    FreshVarFactory,
    Program,
    LoadLabel,
    StoreLabel,
    Lattice,
    MaybeVar,
    maybe_to_var,
)
from .sketches import LabelNode, SketchNode, Sketch
from .loggable import Loggable, LogLevel, show_progress
import networkx
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
        callees: Set[DerivedTypeVariable],
        cs: ConstraintSet,
        sketch_map: Dict[DerivedTypeVariable, Sketch],
        type_schemes: Dict[DerivedTypeVariable, ConstraintSet],
        fresh_var_factory: FreshVarFactory,
    ) -> ConstraintSet:
        """
        For every constraint involving a procedure that has already been
        analyzed, generate constraints based on the type scheme and
        capability constraints based on the sketch.
        """
        # TODO in order to support different instantiations for different calls
        # to the same function, we need to encode function actuals
        # differently than function formals

        connected_callees = {tv for tv in cs.all_tvs() & callees}

        new_constraints = ConstraintSet()
        # sort to avoid non-determinism
        for callee in sorted(connected_callees):
            new_constraints |= sketch_map[callee].instantiate_sketch(
                callee, fresh_var_factory, only_capabilities=True
            )
            new_constraints |= Solver.instantiate_type_scheme(
                fresh_var_factory, type_schemes[callee]
            )
        return new_constraints

    @staticmethod
    def instantiate_globals(
        globals: Set[DerivedTypeVariable],
        sketch_map: Dict[DerivedTypeVariable, Sketch],
        fresh_var_factory: FreshVarFactory,
    ) -> ConstraintSet:
        """
        Instantiate any existing sketches of the provided
        global variables.
        """
        new_constraints = ConstraintSet()
        # sort to avoid non-determinism
        for global_var in sorted(globals):
            if global_var in sketch_map:
                new_constraints |= sketch_map[global_var].instantiate_sketch(
                    global_var, fresh_var_factory, only_capabilities=False
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

    def infer_shapes(
        self,
        proc_or_globals: Set[DerivedTypeVariable],
        types: Lattice[DerivedTypeVariable],
        constraints: ConstraintSet,
    ) -> Dict[DerivedTypeVariable, Sketch]:
        """
        Infer shapes takes a set of constraints and populates shapes of the sketches
        for all DTVs in the program.

        This corresponds to Algorithm E.1 'InferShapes' in the original Retypd paper.
        """
        sketches = {
            proc_or_global: Sketch(proc_or_global, types, verbose=self.verbose)
            for proc_or_global in proc_or_globals
        }

        if len(constraints) == 0:
            return sketches
        equiv, g_quotient = Solver.compute_quotient_graph(
            constraints, types.atomic_types
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
            sketch: Sketch,
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
                    dest_node = sketch.make_node(dest_dtv)
                    sketch.add_edge(curr_node, dest_node, label)
                    visited_nodes[dest] = dest_node
                    all_paths(sketch, dest, visited_nodes)
                    del visited_nodes[dest]
                else:
                    label_node = LabelNode(visited_nodes[dest].dtv)
                    sketch.add_edge(curr_node, label_node, label)

        for proc_or_global, sketch in sketches.items():
            proc_or_global_node = sketch.lookup(proc_or_global)
            quotient_node = equiv.find_equiv_rep(proc_or_global)
            if quotient_node is None:
                continue
            visited_nodes = {quotient_node: proc_or_global_node}
            all_paths(sketch, quotient_node, visited_nodes)
        return sketches

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
        For example, int64.in_0 is something that would be removed. These are the byproduct of
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

    def _solve_bottom_up(
        self,
        callgraph: networkx.DiGraph,
        scc_dag_topo: List[FrozenSet[DerivedTypeVariable]],
        sketches_map: Dict[DerivedTypeVariable, Sketch],
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
        universal_schemas = {}

        self.debug("# Propagating bottom-up")

        for scc in show_progress(self.verbose, reversed(scc_dag_topo)):
            self.debug("# Processing SCC: %s", "_".join([str(s) for s in scc]))
            scc_initial_constraints = ConstraintSet()
            fresh_var_factory = FreshVarFactory()

            non_scc_callees = set()
            for proc in scc:
                self.debug("# Collecting constraints for %s", proc)
                constraints = self.program.proc_constraints.get(
                    proc, ConstraintSet()
                )
                non_scc_proc_callees = {
                    succ
                    for succ in callgraph.successors(proc)
                    if succ not in scc
                }
                constraints |= Solver.instantiate_calls(
                    non_scc_proc_callees,
                    constraints,
                    sketches_map,
                    type_schemes,
                    fresh_var_factory,
                )
                non_scc_callees |= non_scc_proc_callees
                scc_initial_constraints |= constraints
            # Instantiate global sketches (if they exist)
            involved_globals = (
                scc_initial_constraints.all_tvs() & self.program.global_vars
            )
            scc_initial_constraints |= Solver.instantiate_globals(
                involved_globals, sketches_map, fresh_var_factory
            )

            self.debug("# Inferring shapes")
            scc_sketches_map = self.infer_shapes(
                scc | involved_globals,
                self.program.types,
                scc_initial_constraints,
            )
            # Update sketches
            for proc_or_global, sketch in scc_sketches_map.items():
                if proc_or_global in sketches_map:
                    # This should only apply to globals
                    sketches_map[proc_or_global].meet(sketch)
                else:
                    sketches_map[proc_or_global] = sketch

            # Generate type scheme for all interesting nodes, since the sketches can have
            # other procedures' sketch nodes
            self.debug("# Inferring universal type scheme of scc: %s", scc)
            global_procs_and_vars = frozenset(
                scc | non_scc_proc_callees | involved_globals
            )
            universal_type_scheme = self._generate_type_scheme(
                scc_initial_constraints,
                global_procs_and_vars,
                self.program.types.internal_types,
            )
            universal_schemas[scc] = universal_type_scheme

            self.debug(
                "# Inferred universal constraints for %s: %s",
                scc,
                universal_type_scheme,
            )
            for proc in scc:
                self.debug("# Inferring type scheme of proc: %s", proc)
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
                    frozenset({proc}),
                    self.program.types.internal_types,
                )

                self.debug("# Created for %s %s", proc, primitive_constraints)
                sketches_map[proc].add_constraints(primitive_constraints)

            for global_var in involved_globals:
                self.debug(
                    "# Inferring primitive constraints of global: %s",
                    global_var,
                )
                primitive_constraints = self._generate_primitive_constraints(
                    universal_type_scheme,
                    frozenset({global_var}),
                    self.program.types.internal_types,
                )

                self.debug(
                    "# Created for %s %s", global_var, primitive_constraints
                )
                sketches_map[global_var].add_constraints(primitive_constraints)

    def _solve_top_down(
        self,
        callgraph: networkx.DiGraph,
        scc_dag_topo: List[FrozenSet[DerivedTypeVariable]],
        sketches_map: Dict[DerivedTypeVariable, Sketch],
        type_schemes: Dict[DerivedTypeVariable, ConstraintSet],
    ):
        self.debug("# Propagating top-down")
        actuals_sketch_map: Dict[DerivedTypeVariable, Sketch] = {}

        for scc in show_progress(self.verbose, scc_dag_topo):
            scc_initial_constraints = ConstraintSet()
            fresh_var_factory = FreshVarFactory()
            self.debug(
                "# Processing top-down SCC: %s",
                "_".join([str(s) for s in scc]),
            )

            # refine parameters
            # only for non-recursive sccs
            if len(scc) == 1:
                proc = list(scc)[0]
                if (
                    proc not in callgraph.successors(proc)
                    and proc in actuals_sketch_map
                ):
                    sketches_map[proc].meet(actuals_sketch_map[proc])

            # We should keep the quotient graph around to save time here.
            non_scc_callees = set()
            for proc in scc:
                self.debug("# Collecting constraints for %s", proc)
                constraints = self.program.proc_constraints.get(
                    proc, ConstraintSet()
                )
                non_scc_proc_callees = {
                    succ
                    for succ in callgraph.successors(proc)
                    if succ not in scc
                }
                constraints |= Solver.instantiate_calls(
                    non_scc_proc_callees,
                    constraints,
                    sketches_map,
                    type_schemes,
                    fresh_var_factory,
                )
                constraints |= sketches_map[proc].instantiate_sketch(
                    proc, fresh_var_factory
                )
                non_scc_callees |= non_scc_proc_callees
                scc_initial_constraints |= constraints

            callees_sketches_map = self.infer_shapes(
                non_scc_callees,
                self.program.types,
                scc_initial_constraints,
            )
            for callee in non_scc_callees:
                primitive_constraints = self._generate_primitive_constraints(
                    scc_initial_constraints,
                    frozenset({callee}),
                    self.program.types.internal_types,
                )

                self.debug(
                    "# Created for %s %s", callee, primitive_constraints
                )
                callees_sketches_map[callee].add_constraints(
                    primitive_constraints
                )

            # join the sketches for the actuals
            for proc, sketch in callees_sketches_map.items():
                if proc in actuals_sketch_map:
                    actuals_sketch_map[proc].join(sketch)
                else:
                    actuals_sketch_map[proc] = sketch

    def get_type_of_variables(
        self,
        sketches_map: Dict[DerivedTypeVariable, Sketch],
        type_schemes: Dict[DerivedTypeVariable, ConstraintSet],
        proc: MaybeVar,
        vars: Set[DerivedTypeVariable],
    ):
        """
        Solving sketches for an additional set of derived type variables.
        """
        proc = maybe_to_var(proc)
        constraints = self.program.proc_constraints.get(
                        proc, ConstraintSet()
                    )
        callees = set(networkx.DiGraph(self.program.callgraph).successors(proc))
        fresh_var_factory = FreshVarFactory()
        constraints |= Solver.instantiate_calls(
            callees,
            constraints,
            sketches_map,
            type_schemes,
            fresh_var_factory,
        )
        constraints |= sketches_map[proc].instantiate_sketch(
            proc, fresh_var_factory
        )
    
        var_sketches = self.infer_shapes(
            vars,
            self.program.types,
            constraints,
        )
        for var in vars:
            primitive_constraints = self._generate_primitive_constraints(
                constraints,
                frozenset({var}),
                self.program.types.internal_types)
    
            var_sketches[var].add_constraints(
                primitive_constraints
            )
        return var_sketches

    def __call__(
        self,
    ) -> Tuple[
        Dict[DerivedTypeVariable, ConstraintSet],
        Dict[DerivedTypeVariable, Sketch],
    ]:
        """Perform the retypd calculation."""

        type_schemes: Dict[DerivedTypeVariable, ConstraintSet] = {}
        sketches_map: Dict[DerivedTypeVariable, Sketch] = {}

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
        fake_root = "$$FAKEROOT$$"
        scc_dag = networkx.condensation(callgraph)
        roots = list(find_roots(scc_dag))
        scc_dag.add_node(fake_root, members=[fake_root])
        for r in roots:
            scc_dag.add_edge(fake_root, r)
        scc_dag_topo = [
            frozenset(scc_dag.nodes[scc_node]["members"])
            for scc_node in networkx.topological_sort(scc_dag)
        ][1:]

        self._solve_bottom_up(
            callgraph,
            scc_dag_topo,
            sketches_map,
            type_schemes,
        )
        if self.config.top_down_propagation:
            self._solve_top_down(
                callgraph,
                scc_dag_topo,
                sketches_map,
                type_schemes,
            )
        return (type_schemes, sketches_map)
