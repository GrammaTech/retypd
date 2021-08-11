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

from typing import Dict, List, Optional, Set, Tuple, Union
from .graph import EdgeLabel, Node, ConstraintGraph
from .schema import ConstraintSet, DerivedTypeVariable, Program, SubtypeConstraint, Variance
from .parser import SchemaParser
import os
import networkx


class Solver:
    '''Takes a program and generates subtype constraints. The constructor does not perform the
    computation; rather, :py:class:`Solver` objects are callable as thunks.
    '''
    def __init__(self, program: Program) -> None:
        self.program = program
        # TODO possibly make these values shared across a function
        self.next = 0
        self._type_vars: Dict[DerivedTypeVariable, DerivedTypeVariable] = {}

    def _get_type_var(self, var: DerivedTypeVariable) -> DerivedTypeVariable:
        '''Look up a type variable by name. If it (or a prefix of it) exists in _type_vars, form the
        appropriate variable by adding the leftover part of the suffix. If not, return the variable
        as passed in.
        '''
        for expanded in sorted(self._type_vars, reverse=True):
            suffix = expanded.get_suffix(var)
            if suffix is not None:
                type_var = self._type_vars[expanded]
                return DerivedTypeVariable(type_var.base, suffix)
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
                    for predecessor in graph.predecessors(node):
                        scc_index = condensation.graph['mapping'][predecessor]
                        if scc_index not in visited:
                            candidates.add(node.base)
                candidates = Solver._filter_no_prefix(candidates)
                endpoints |= candidates
        self.all_endpoints = frozenset(endpoints |
                                       set(self.program.callgraph) |
                                       set(self.program.global_vars) |
                                       self.program.types.internal_types)
        for var in Solver._filter_no_prefix(endpoints):
            self._make_type_var(var)

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
        for label in string:
            if label.kind == EdgeLabel.Kind.FORGET:
                rhs = rhs.recall(label.capability)
            else:
                lhs = lhs.recall(label.capability)
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
        constraints = ConstraintSet()
        def explore(origin: Node,
                    path: List[Node] = [],
                    string: List[EdgeLabel] = []) -> None:
            '''Find all non-empty paths that begin and end on members of self.all_endpoints. Return
            the list of labels encountered along the way as well as the origin and destination.
            '''
            if path and origin.base in self.all_endpoints:
                constraint = self._maybe_constraint(path[0], origin, string)
                if constraint:
                    constraints.add(constraint)
                return
            if origin in path:
                return
            path = list(path)
            path.append(origin)
            if origin in graph:
                for succ in graph[origin]:
                    label = graph[origin][succ].get('label')
                    new_string = list(string)
                    if label:
                        new_string.append(label)
                    explore(succ, path, new_string)
        for origin in {node for node in graph.nodes if node.base in self.all_endpoints and
                                                       node._unforgettable ==
                                                       Node.Unforgettable.PRE_RECALL}:
            explore(origin)
        return constraints

    def __call__(self) -> Tuple[Dict[DerivedTypeVariable, ConstraintSet], 'Sketches']:
        '''Perform the retypd calculation.
        '''
        accumulated = networkx.DiGraph()
        derived: Dict[DerivedTypeVariable, ConstraintSet] = {}
        scc_dag = networkx.condensation(self.program.callgraph)
        sketches = Sketches(self)
        # The idea here is that functions need to satisfy their clients' needs for inputs and need
        # to use their clients' return values without overextending them. So we do a reverse
        # topological order on functions, lumping SCCs together, and slowly extend the graph.
        # It would be interesting to explore how this algorithm compares with a more restricted one
        # as far as asymptotic complexity, but this is simple to understand and so the right choice
        # for now. I suspect that most of the graphs are fairly sparse and so the practical reality
        # is that there aren't many paths.
        for scc_node in reversed(list(networkx.topological_sort(scc_dag))):
            scc = scc_dag.nodes[scc_node]['members']
            for proc in scc:
                graph = ConstraintGraph(self.program.proc_constraints[proc])
                accumulated.add_edges_from(graph.graph.edges)
                for head, tail in graph.graph.edges:
                    label = graph.graph[head][tail].get('label')
                    if label:
                        accumulated[head][tail]['label'] = label
            # make a copy; some of this analysis mutates the graph
            scc_graph = networkx.DiGraph(accumulated)
            # TODO could this be done on the proc graph? Does mutual recursion potentially introduce
            # loops?
            self._generate_type_vars(scc_graph)
            Solver._unforgettable_subgraph_split(scc_graph)
            generated = self._generate_constraints(scc_graph)
            sketches.add_constraints(generated, scc)
            derived.update({proc: generated for proc in scc})
        # generate constraints for globals once all information is available
        generated = self._generate_constraints(scc_graph)
        sketches.add_constraints(generated, self.program.global_vars)
        derived.update({glob: generated for glob in self.program.global_vars})
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
        return f'{self.dtv} ({self.atom})'

    def __repr__(self) -> str:
        return f'SketchNode({self})'


SkNode = Union[SketchNode, DerivedTypeVariable]


class Sketches:
    '''The set of sketches from a set of constraints. Intended to be updated incrementally, per the
    Solver's reverse topological ordering.
    '''
    def __init__(self, solver: Solver) -> None:
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

    def convert_to_label(self, old: SketchNode) -> None:
        self.replace_node(old, old.dtv)

    def remove_cycles(self) -> None:
        '''Identify cycles in the graph and remove them by replacing instances of every node
        downstream from itself with a label (the labeled polymorphism mentioned in Definition 3.5).
        '''
        for cycle in networkx.simple_cycles(self.sketches):
            node = cycle[1]
            last = cycle[-1]
            self.replace_edge(last, node, last, node.dtv)

    def copy_dependencies(self, dependencies: Dict[SketchNode, Set[SketchNode]]) -> None:
        '''Copy the structures from self.sketches corresponding to the keys in dependencies to each
        other simultaneously. Each copy is a DFS that identifies duplicates and replaces them with
        labels, so the resulting graphs are trees. All reads are from self.sketches and all writes
        go to a temporary variable, so no fixed point calculation is needed.
        '''
        new_sketches = networkx.DiGraph(self.sketches)
        def copy_inner(onto: SketchNode, origin: SketchNode, seen: Set[SketchNode]) -> None:
            if origin not in seen:
                seen = seen | {origin}
                for dependency in dependencies.get(origin, set()):
                    copy_inner(onto, dependency, seen)
                for succ in set(self.sketches.successors(origin)):
                    label = self.sketches[origin][succ]['label']
                    if succ in seen:
                        new_succ = succ.dtv
                    else:
                        new_succ = self.make_node(onto.dtv.add_suffix(label))
                        copy_inner(new_succ, succ, seen | {origin, new_succ})
                    new_sketches.add_edge(onto, new_succ, label=label)
        for dest in dependencies:
            for origin in dependencies.get(dest, set()):
                seen = set(map(lambda n: self.lookup[n], dest.dtv.all_prefixes()))
                copy_inner(dest, origin, seen)
        self.sketches = new_sketches

    def add_constraints(self, constraints: ConstraintSet, nodes: Set[DerivedTypeVariable]) -> None:
        '''Extend the set of sketches with the new set of constraints.
        '''
        inter_dependencies: Dict[SketchNode, Set[SketchNode]] = {}
        intra_dependencies: Dict[SketchNode, Set[SketchNode]] = {}
        for constraint in constraints:
            left_node = self.add_variable(constraint.left)
            right_node = self.add_variable(constraint.right)
            left_base_var = constraint.left.base_var
            right_base_var = constraint.right.base_var
            # TODO need to resolve intraprocedural dependencies on endpoints first
            if right_base_var in self.solver._type_vars:
                intra_dependencies.setdefault(left_node, set()).add(right_node)
            if left_base_var in nodes:
                inter_dependencies.setdefault(left_node, set()).add(right_node)
        self.copy_dependencies(intra_dependencies)
        self.copy_dependencies(inter_dependencies)
        for constraint in constraints:
            left = constraint.left
            right = constraint.right
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
        # emit nodes
        for node in self.sketches.nodes:
            if isinstance(node, SketchNode) and node.dtv.base_var == dtv:
                graph_str += nt
                graph_str += f'"{node.dtv}" [label="{node.atom}"];'
            else:
                # TODO I should probably create new, distinct objects instead of reusing the same
                # DTV for every label
                pass
        # emit edges
        seen = {(start, start)}
        frontier = {(start, succ) for succ in self.sketches.successors(start)} - seen
        while frontier:
            new_frontier = set()
            for node, succ in frontier:
                graph_str += nt
                f_label = node
                if isinstance(node, SketchNode):
                    f_label = node.dtv
                t_label = succ
                if isinstance(succ, SketchNode):
                    t_label = succ.dtv
                    new_frontier |= {(succ, s_s) for s_s in self.sketches.successors(succ)}
                graph_str += f'"{f_label}" -> "{t_label}"'
                graph_str += f' [label="{self.sketches[node][succ]["label"]}"];'
            frontier = new_frontier - seen
        graph_str += f'{os.linesep}}}'
        return graph_str

    def __str__(self) -> str:
        if self.lookup:
            nt = f'{os.linesep}\t'
            return f'nodes:{nt}{nt.join(map(lambda k: f"{k} ({self.lookup[k].atom}", self.lookup))})'
        return 'no sketches'
