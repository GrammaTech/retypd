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

'''The driver for the retypd analysis.
'''

from typing import Dict, List, Optional, Set, Tuple, Union, Any, Callable
from .graph import EdgeLabel, Node, ConstraintGraph
from .schema import (
    ConstraintSet,
    DerivedTypeVariable,
    Program,
    SubtypeConstraint,
    Variance,
)
from .parser import SchemaParser
import os
import itertools
import networkx
import tqdm
from collections import defaultdict
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
    G.render(filename, format='svg', view=False)


# Unfortunable, the python logging class is a bit flawed and overly complex for what we need
# When you use info/debug you can use %s/%d/etc formatting ala logging to lazy evaluate
class Loggable:
    def __init__(self, verbose: int = 0):
        self.verbose = verbose

    def info(self, *args):
        if self.verbose > 0:
            print(str(args[0]) % tuple(args[1:]))

    def debug(self, *args):
        if self.verbose > 1:
            print(str(args[0]) % tuple(args[1:]))

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
    # Keep global constraints in the function pass.
    globals_in_func_pass: bool = True

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
    '''Takes a program and generates subtype constraints. The constructor does not perform the
    computation; rather, :py:class:`Solver` objects are callable as thunks.
    '''
    def __init__(self,
                 program: Program,
                 config: SolverConfig = SolverConfig(),
                 verbose: int = 0) -> None:
        super(Solver, self).__init__(verbose)
        self.program = program
        # TODO possibly make these values shared across a function
        self.next = 0
        self._type_vars: Dict[DerivedTypeVariable, DerivedTypeVariable] = {}
        self.config = config

    def _get_type_var(self, var: DerivedTypeVariable) -> DerivedTypeVariable:
        '''Look up a type variable by name. If it (or a prefix of it) exists in _type_vars, form the
        appropriate variable by adding the leftover part of the suffix. If not, return the variable
        as passed in.
        '''
        # Starting with the longest prefix (just var), check all prefixes against
        # the existing typevars
        for prefix in itertools.chain([var], var.all_prefixes()):
            if prefix in self._type_vars:
                suffix = prefix.get_suffix(var)
                if suffix is not None:
                    type_var = self._type_vars[prefix]
                    return DerivedTypeVariable(type_var.base, suffix)
        return var

    def reverse_type_var(self, var: DerivedTypeVariable) -> DerivedTypeVariable:
        '''Look up the canonical version of a variable; if it begins with a type variable
        '''
        base = var.base_var
        if base in self._rev_type_vars:
            return self._rev_type_vars[base].extend(var.path)
        return var

    def lookup_type_var(self, var: Union[str, DerivedTypeVariable]) -> DerivedTypeVariable:
        '''Take a string, convert it to a DerivedTypeVariable, and if there is a type variable that
        stands in for a prefix of it, change the prefix to the type variable.
        '''
        if isinstance(var, str):
            var = SchemaParser.parse_variable(var)
        return self._get_type_var(var)

    def _make_type_var(self, base: DerivedTypeVariable) -> None:
        '''Generate a type variable.
        '''
        if base in self._type_vars:
            return
        var = DerivedTypeVariable(f'τ${self.next}')
        self._type_vars[base] = var
        self.next += 1
        self.debug("Generate type_var: %s", var)

    def _generate_type_vars(self, graph: networkx.DiGraph) -> None:
        '''Identify at least one node in each nontrivial SCC and generate a type variable for it.
        This ensures that the path exploration algorithm will never loop; instead, it will generate
        constraints that are recursive on the type variable.

        To do so, find strongly connected components (such that loops may contain forget or recall
        edges but never both). In each nontrivial SCC (in reverse topological order), identify all
        vertices with predecessors in a strongly connected component that has not yet been visited.
        If any of the identified variables is a prefix of another (e.g., φ and φ.load.σ8@0),
        eliminate the longer one. Each of these variables is added to the set of variables at which
        graph exploration stops. Excepting those with a prefix already in the set, these variables
        are made into named type variables.

        Once all SCCs have been processed, minimize the set of candidates as before (remove
        variables that have a prefix in the set) and emit type variables for each remaining
        candidate.

        Side-effects: update the set of interesting endpoints, update the map of typevars.
        '''
        forget_graph = networkx.DiGraph(graph)
        recall_graph = networkx.DiGraph(graph)
        for head, tail in graph.edges:
            label = graph[head][tail].get('label')
            if label:
                if label.kind == EdgeLabel.Kind.FORGET:
                    recall_graph.remove_edge(head, tail)
                else:
                    forget_graph.remove_edge(head, tail)
        endpoints: Set[DerivedTypeVariable] = set()
        for fr_graph in [forget_graph, recall_graph]:
            condensation = networkx.condensation(fr_graph)
            visited = set()
            for scc_node in reversed(list(networkx.topological_sort(condensation))):
                candidates: Set[DerivedTypeVariable] = set()
                scc = condensation.nodes[scc_node]['members']
                visited.add(scc_node)
                if len(scc) == 1:
                    continue
                for node in scc:
                    # Look in graph for predecessors, not fr_graph; the purpose of fr_graph is
                    # merely to ensure that the SCCs we examine don't contain both forget and recall
                    # edges.
                    for predecessor in graph.predecessors(node):
                        scc_index = condensation.graph['mapping'][predecessor]
                        if scc_index not in visited:
                            candidates.add(node.base)
                candidates = Solver._filter_no_prefix(candidates)
                endpoints |= candidates
        # Types from the lattice should never end up pointing at typevars
        endpoints -= set(self.program.types.internal_types)
        self.all_endpoints = frozenset(endpoints |
                                       set(self.program.callgraph) |
                                       set(self.program.global_vars) |
                                       self.program.types.internal_types)
        for var in Solver._filter_no_prefix(endpoints):
            self._make_type_var(var)
        self._rev_type_vars = {tv: var for (var, tv) in self._type_vars.items()}

    def _maybe_constraint(self,
                          origin: Node,
                          dest: Node,
                          string: List[EdgeLabel]) -> Optional[SubtypeConstraint]:
        '''Generate constraints by adding the forgets in string to origin and the recalls in string
        to dest. If both of the generated vertices are covariant (the empty string's variance is
        covariant, so only covariant vertices can represent a derived type variable without an
        elided portion of its path) and if the two variables are not equal, emit a constraint.
        '''
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

        if (lhs.suffix_variance == Variance.COVARIANT and
                rhs.suffix_variance == Variance.COVARIANT):
            lhs_var = self._get_type_var(lhs.base)
            rhs_var = self._get_type_var(rhs.base)
            if lhs_var != rhs_var:
                return SubtypeConstraint(lhs_var, rhs_var)
        return None

    def _generate_constraints(self, graph: networkx.DiGraph) -> ConstraintSet:
        '''Now that type variables have been computed, no cycles can be produced. Find paths from
        endpoints to other endpoints and generate constraints.
        '''
        npaths = 0
        constraints = ConstraintSet()
        # On large procedures, the graph this is exploring can be quite large (hundreds of nodes,
        # thousands of edges). This can result in an insane number of paths - most of which do not
        # result in a constraint, and most of the ones that do result in constraints are redundant.
        def explore(current_node: Node,
                    path: List[Node] = [],
                    string: List[EdgeLabel] = []) -> None:
            '''Find all non-empty paths that begin and end on members of self.all_endpoints. Return
            the list of labels encountered along the way as well as the current_node and destination.
            '''
            nonlocal max_paths_per_root
            nonlocal npaths
            if len(path) > self.config.max_path_length:
                return
            if npaths > max_paths_per_root:
                return
            if current_node in path:
                npaths += 1
                return
            if path and current_node.base in self.all_endpoints:
                constraint = self._maybe_constraint(path[0], current_node, string)
                if constraint:
                    constraints.add(constraint)
                npaths += 1
                return
            path = list(path)
            path.append(current_node)
            if current_node in graph:
                for succ in graph[current_node]:
                    label = graph[current_node][succ].get('label')
                    new_string = list(string)
                    if label:
                        new_string.append(label)
                    explore(succ, path, new_string)
        start_nodes = {node for node in graph.nodes if node.base in self.all_endpoints and
                                                       node._unforgettable ==
                                                       Node.Unforgettable.PRE_RECALL}
        # We evenly distribute the maximum number of paths that we are willing to explore
        # across all origin nodes here.
        max_paths_per_root = int(min(self.config.max_paths_per_root,
                                     self.config.max_total_paths / float(len(start_nodes)+1)))
        for origin in start_nodes:
            npaths = 0
            explore(origin)
        return constraints

    def _solve_topo_graph(self,
                          scc_dag: networkx.DiGraph,
                          constraint_map: Dict[Any, ConstraintSet],
                          sketches_map: Dict[DerivedTypeVariable, "Sketches"],
                          derived: Dict[DerivedTypeVariable, ConstraintSet],
                          constraint_visitor: Optional[Callable[[SubtypeConstraint], None]] = None):
        def show_progress(iterable):
            if self.verbose:
                return tqdm.tqdm(iterable)
            return iterable

        for scc_node in show_progress(reversed(list(networkx.topological_sort(scc_dag)))):
            scc = scc_dag.nodes[scc_node]['members']
            scc_graph = networkx.DiGraph()
            for proc_or_global in scc:
                constraints = constraint_map.get(proc_or_global, ConstraintSet())
                if constraint_visitor is not None:
                    new_constraints = ConstraintSet()
                    for c in constraints:
                        if constraint_visitor(c):
                            new_constraints.add(c)
                    constraints = new_constraints
                graph = ConstraintGraph(constraints)
                for head, tail in graph.graph.edges:
                    label = graph.graph[head][tail].get('label')
                    if label:
                        scc_graph.add_edge(head, tail, label=label)
                    else:
                        scc_graph.add_edge(head, tail)

            # Uncomment this out to dump the constraint graph
            #name = "_".join([str(s) for s in scc])
            #print(f"SCC: {name}")
            #dump_labeled_graph(scc_graph, name, f"/tmp/scc_{name}")

            # make a copy; some of this analysis mutates the graph
            self._generate_type_vars(scc_graph)
            Solver._unforgettable_subgraph_split(scc_graph)
            generated = self._generate_constraints(scc_graph)

            # The sketches for this SCC; note it may pull in data from other sketches
            scc_sketches = Sketches(self, self.verbose)
            scc_sketches.add_constraints(generated, scc, sketches_map)

            for proc_or_global in scc:
                sketches_map[proc_or_global] = scc_sketches
                derived[proc_or_global] = generated


    def __call__(self) -> Tuple[Dict[DerivedTypeVariable, ConstraintSet],
                                Dict[DerivedTypeVariable, "Sketches"]]:
        '''Perform the retypd calculation.
        '''

        derived: Dict[DerivedTypeVariable, ConstraintSet] = {}
        sketches: Dict[DerivedTypeVariable, Sketches] = {}

        global_constraints = defaultdict(ConstraintSet)
        global_graph = networkx.DiGraph()

        # Callback and set for collecting any constraints related to globals
        global_bases = set([str(v) for v in self.program.global_vars])
        def _collect_globals(c: SubtypeConstraint):
            global_count = 0
            if c.left.base in global_bases:
                global_constraints[c.left.base].add(c)
                global_count += 1
            if c.right.base in global_bases:
                global_constraints[c.right.base].add(c)
                global_count += 1
            if global_count == 2:
                # The sub-typing relationship is reversed from the "depends on results
                # from" relationship
                global_graph.add_edge(c.right.base, c.left.base)
            # Return "True" to keep this constraint.
            return self.config.globals_in_func_pass or (global_count == 0)

        # The idea here is that functions need to satisfy their clients' needs for inputs and need
        # to use their clients' return values without overextending them. So we do a reverse
        # topological order on functions, lumping SCCs together, and slowly extend the graph.
        # It would be interesting to explore how this algorithm compares with a more restricted one
        # as far as asymptotic complexity, but this is simple to understand and so the right choice
        # for now. I suspect that most of the graphs are fairly sparse and so the practical reality
        # is that there aren't many paths.
        self.info("Solving functions")
        scc_dag = networkx.condensation(self.program.callgraph)
        self._solve_topo_graph(scc_dag, self.program.proc_constraints, sketches,
                               derived, _collect_globals)

        self.info("Solving globals")
        global_scc_dag = networkx.condensation(global_graph)
        self._solve_topo_graph(global_scc_dag, global_constraints, sketches, derived)

        return (derived, sketches)

    @staticmethod
    def _unforgettable_subgraph_split(graph: networkx.DiGraph) -> None:
        '''The algorithm, after saturation, only admits paths such that forget edges all precede
        the first recall edge (if there is such an edge). To enforce this, we modify the graph by
        splitting each node and the unlabeled and recall edges (but not forget edges!). Recall edges
        in the original graph are changed to point to the 'unforgettable' duplicate of their
        original target. As a result, no forget edges are reachable after traversing a single recall
        edge.
        '''
        edges = set(graph.edges)
        for head, tail in edges:
            label = graph[head][tail].get('label')
            if label and label.kind == EdgeLabel.Kind.FORGET:
                continue
            recall_head = head.split_unforgettable()
            recall_tail = tail.split_unforgettable()
            atts = graph[head][tail]
            if label and label.kind == EdgeLabel.Kind.RECALL:
                graph.remove_edge(head, tail)
                graph.add_edge(head, recall_tail, **atts)
            graph.add_edge(recall_head, recall_tail, **atts)

    @staticmethod
    def _filter_no_prefix(variables: Set[DerivedTypeVariable]) -> Set[DerivedTypeVariable]:
        '''Return a set of elements from variables for which no prefix exists in variables. For
        example, if variables were the set {A, A.load}, return the set {A}.
        '''
        selected = set()
        candidates = sorted(variables, reverse=True)
        for index, candidate in enumerate(candidates):
            emit = True
            for other in candidates[index+1:]:
                # other and candidate cannot be equal because they're distinct elements of a set
                if other.get_suffix(candidate) is not None:
                    emit = False
                    break
            if emit:
                selected.add(candidate)
        return selected


class SketchNode:
    def __init__(self, dtv: DerivedTypeVariable, atom: DerivedTypeVariable) -> None:
        self.dtv = dtv
        self.atom = atom

    # the atomic type of a DTV is an annotation, not part of its identity
    def __eq__(self, other) -> bool:
        if isinstance(other, SketchNode):
            return self.dtv == other.dtv
        return False

    def __hash__(self) -> int:
        return hash(self.dtv)

    def __str__(self) -> str:
        return f'{self.dtv}({self.atom})'

    def __repr__(self) -> str:
        return f'SketchNode({self})'


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
        return f'{self.target}.label_{self.id}'

    def __repr__(self) -> str:
        return str(self)


SkNode = Union[SketchNode, LabelNode]


class Sketches(Loggable):
    '''The set of sketches from a set of constraints. Intended to be updated incrementally, per the
    Solver's reverse topological ordering.
    '''
    def __init__(self, solver: Solver, verbose: int = 0) -> None:
        super(Sketches, self).__init__(verbose)
        self.sketches = networkx.DiGraph()
        self.lookup: Dict[DerivedTypeVariable, SketchNode] = {}
        self.solver = solver
        self.dependencies: Dict[DerivedTypeVariable, Set[DerivedTypeVariable]] = {}

    def make_node(self, variable: DerivedTypeVariable) -> SketchNode:
        '''Make a node from a DTV. Compute its atom from its access path.
        '''
        if variable in self.lookup:
            return self.lookup[variable]
        variance = variable.path_variance
        if variance == Variance.COVARIANT:
            result = SketchNode(variable, self.solver.program.types.top)
        else:
            result = SketchNode(variable, self.solver.program.types.bottom)
        self.lookup[variable] = result
        return result

    def add_variable(self, variable: DerivedTypeVariable) -> SketchNode:
        '''Add a variable and its prefixes to the set of sketches. Each node's atomic type is either
        TOP or BOTTOM, depending on the variance of the variable's access path. If the variable
        already exists, skip it.
        '''
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

    def replace_edge(self, head: SkNode, tail: SkNode, new_head: SkNode, new_tail: SkNode) -> None:
        '''Replace an edge, keeping its attributes intact.
        '''
        atts = self.sketches[head][tail]
        self.sketches.remove_edge(head, tail)
        self.sketches.add_edge(new_head, new_tail, **atts)

    def replace_node(self, old: SketchNode, new: SkNode) -> None:
        '''Replace a node, updating incoming and outgoing edges but preserving their attributes.
        '''
        for pred in self.sketches.predecessors(old):
            self.replace_edge(pred, old, pred, new)
        for succ in self.sketches.successors(old):
            self.replace_edge(old, succ, new, succ)

    def remove_cycles(self) -> None:
        '''Identify cycles in the graph and remove them by replacing instances of every node
        downstream from itself with a label (the labeled polymorphism mentioned in Definition 3.5).
        '''
        for cycle in networkx.simple_cycles(self.sketches):
            node = cycle[1]
            last = cycle[-1]
            self.replace_edge(last, node, last, node.dtv)

    def _add_edge(self, head: SketchNode, tail: SkNode, label: str) -> None:
        # don't emit duplicate edges
        if (head, tail) not in self.sketches.edges:
            self.sketches.add_edge(head, tail, label=label)

    def _copy_inner(self,
                    onto: SketchNode,
                    origin: SketchNode,
                    other_sketches: networkx.DiGraph,
                    dependencies: Dict[SketchNode, Set[SketchNode]],
                    seen: Dict[SketchNode, SketchNode]) -> None:
        seen[origin] = onto
        # If the dependencies graph says that onto is a subtype of origin, copy all of origin's
        # outgoing edges and their targets. If this introduces a loop, create a label instead of
        # a loop.
        for dependency in dependencies.get(origin, set()):
            if dependency not in other_sketches.nodes:
                continue
            if dependency in seen:
                original = seen[dependency]
                for succ in other_sketches.successors(dependency):
                    label = other_sketches[dependency][succ]['label']
                    new_succ = LabelNode(original.dtv.add_suffix(label))
                    self._add_edge(onto, new_succ, label=label)
            else:
                self._copy_inner(onto, dependency, other_sketches, dependencies, seen)

        # Just because we have a constraint about a downstream base DTV (in the topological
        # order) doesn't mean that downstream node has any information about the specific
        # node we are looking at.
        if origin not in other_sketches.nodes:
            return

        # Copy all of origin's outgoing edges onto onto.
        for succ in other_sketches.successors(origin):
            label = other_sketches[origin][succ]['label']

            # If succ has been seen, look up the last time it was seen and instead create a
            # label that links to the previous location where it was seen.
            if succ in seen:
                self._add_edge(onto, LabelNode(seen[succ].dtv), label=label)
            # If the successor is a label, translate it into this sketch.
            elif isinstance(succ, LabelNode):
                suffix = succ.target.get_suffix(origin.dtv)
                if suffix is None:
                    raise ValueError(f'Suffix is None {origin.dtv} --> {succ.target}')
                succ_dtv = onto.dtv.remove_suffix(suffix)
                assert succ_dtv
                self._add_edge(onto, LabelNode(succ_dtv), label=label)
            # Otherwise, it's a regular (non-label) node that hasn't been seen, so emit it and
            # explore its successors.
            else:
                succ_node = self.make_node(onto.dtv.add_suffix(label))
                self._add_edge(onto, succ_node, label=label)
                seen[succ_node] = onto
                self._copy_inner(succ_node, succ, other_sketches, dependencies, seen)

    def instantiate_intra(self, dependencies: Dict[SketchNode, Set[SketchNode]]) -> None:
        self.debug("instantiate_intra: %s", dependencies)
        old_sketches = networkx.DiGraph(self.sketches)
        for dest in dependencies.keys():
            for origin in dependencies.get(dest, set()):
                seen = {self.lookup[n]: self.lookup[n] for n in dest.dtv.all_prefixes()}
                # Each iteration we take a copy of the current sketches because they will be
                # updated inside of _copy_inner.
                self._copy_inner(dest, origin, old_sketches, dependencies, seen)

    def copy_inter(self,
                   dependencies: Dict[SketchNode, Set[SketchNode]],
                   sketches_map: Dict[DerivedTypeVariable, "Sketches"]) -> None:
        '''Copy the structures from self.sketches corresponding to the keys in dependencies to each
        other simultaneously. Each copy is a DFS that identifies duplicates and replaces them with
        labels, so the resulting graphs are trees. All reads are from self.sketches and all writes
        go to a temporary variable, so no fixed point calculation is needed.
        '''
        self.debug("copy_inter: %s", dependencies)
        for dest in dependencies.keys():
            for origin in dependencies.get(dest, set()):
                seen = {self.lookup[n]: self.lookup[n] for n in dest.dtv.all_prefixes()}
                origin_base = origin.dtv.base_var
                # If we have funcvar <= globalvar, the global variable won't yet be in
                # the sketches_map.
                if origin_base in sketches_map:
                    self._copy_inner(dest, origin, sketches_map[origin_base].sketches,
                                     dependencies, seen)

    counter = 0
    def add_constraints(self,
                        constraints: ConstraintSet,
                        nodes: Set[DerivedTypeVariable],
                        sketches_map: Dict[DerivedTypeVariable, "Sketches"]) -> None:
        '''Extend the set of sketches with the new set of constraints.
        '''
        inter_dependencies: Dict[SketchNode, Set[SketchNode]] = {}
        intra_dependencies: Dict[SketchNode, Set[SketchNode]] = {}
        for constraint in constraints:
            left = self.solver.reverse_type_var(constraint.left)
            right = self.solver.reverse_type_var(constraint.right)
            left_node = self.add_variable(left)
            right_node = self.add_variable(right)
            # Base variables are the type variable in question without the access path; e.g.,
            # F.in_0.load.σ4@0's base variable is F.
            left_base_var = left.base_var
            right_base_var = right.base_var
            # Skip constraints X <= LatticeType, they get handled at the bottom of this function
            if right_base_var in self.solver.program.types.internal_types:
                continue
            # Type variables are generated only to break cycles in the constraint graph. If the rhs
            # is a type variable, that means that the relationship between the sketches is
            # intraprocedural and not interprocedural.
            elif right_base_var in self.solver._type_vars:
                self.debug("Found in type_vars: %s" % right_base_var)
                intra_dependencies.setdefault(left_node, set()).add(right_node)
            # If the right-hand side is not a type variable, it must be an interesting node (a
            # function or global). The left and right base nodes may be different or the same;
            # either way, treat it as an interprocedural dependency
            elif left_base_var in nodes:
                self.debug("Have %d nodes" % len(nodes))
                inter_dependencies.setdefault(left_node, set()).add(right_node)
        # Intra-SCC dependencies just get instantiated using "this" set of sketches
        self.instantiate_intra(intra_dependencies)
        # Inter-SCC dependencies make use of the sketches_map, which maps each base DTV to its
        # sketch graph. This way we don't need to copy and iterate over a huge DiGraph for the
        # entire program when we only need the direct callees (or referent's, for globals)
        self.copy_inter(inter_dependencies, sketches_map)

        for constraint in constraints:
            left = self.solver.reverse_type_var(constraint.left)
            right = self.solver.reverse_type_var(constraint.right)
            if left in self.solver.program.types.internal_types:
                node = self.lookup[right]
                node.atom = self.solver.program.types.join(node.atom, left)
            if right in self.solver.program.types.internal_types:
                node = self.lookup[left]
                node.atom = self.solver.program.types.meet(node.atom, right)

    def to_dot(self, dtv: DerivedTypeVariable) -> str:
        nt = f'{os.linesep}\t'
        graph_str = f'digraph {dtv} {{'
        start = self.lookup[dtv]
        edges_str = ''
        # emit edges and identify nodes
        nodes = {start}
        seen: Set[Tuple[SketchNode, SkNode]] = {(start, start)}
        frontier = {(start, succ) for succ in self.sketches.successors(start)} - seen
        while frontier:
            new_frontier: Set[Tuple[SketchNode, SkNode]] = set()
            for pred, succ in frontier:
                edges_str += nt
                nodes.add(succ)
                new_frontier |= {(succ, s_s) for s_s in self.sketches.successors(succ)}
                edges_str += f'"{pred}" -> "{succ}"'
                edges_str += f' [label="{self.sketches[pred][succ]["label"]}"];'
            frontier = new_frontier - seen
        # emit nodes
        for node in nodes:
            if isinstance(node, SketchNode):
                if node.dtv == dtv:
                    graph_str += nt
                    graph_str += f'"{node}" [label="{node.dtv}"];'
                elif node.dtv.base_var == dtv:
                    graph_str += nt
                    graph_str += f'"{node}" [label="{node.atom}"];'
            elif node.target.base_var == dtv:
                graph_str += nt
                graph_str += f'"{node}" [label="{node.target}", shape=house];'
        graph_str += edges_str
        graph_str += f'{os.linesep}}}'
        return graph_str

    def __str__(self) -> str:
        if self.lookup:
            nt = f'{os.linesep}\t'
            return f'nodes:{nt}{nt.join(map(lambda k: f"{k} ({self.lookup[k].atom})", self.lookup))})'
        return 'no sketches'
