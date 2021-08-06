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

from typing import Dict, FrozenSet, List, Iterable, Optional, Set, Tuple, Union
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
        self.endpoints: Set[DerivedTypeVariable] = \
                set(program.callgraph) | set(program.globs) | program.types.internal_types
        self.interesting: FrozenSet[DerivedTypeVariable] = frozenset(self.endpoints)
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
                self.endpoints |= candidates
                endpoints |= candidates
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
        interesting nodes to other interesting nodes and generate constraints.
        '''
        constraints = ConstraintSet()
        def explore(origin: Node,
                    path: List[Node] = [],
                    string: List[EdgeLabel] = []) -> None:
            '''Find all non-empty paths that begin and end members of self.interesting. Return the
            list of labels encountered along the way as well as the origin and destination.
            '''
            if path and origin.base in self.endpoints:
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
        for origin in {node for node in graph.nodes if node.base in self.endpoints and
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
                # TODO I think these three calls should move into the constructor
                graph.add_forget_recall()
                graph.saturate()
                Solver._remove_self_loops(graph)
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
            sketches.add_sketches(generated)
            derived.update({proc: generated for proc in scc})
        # generate constraints for globals once all information is available
        generated = self._generate_constraints(scc_graph)
        sketches.add_sketches(generated)
        derived.update({glob: generated for glob in self.program.globs})
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
    def _filter_no_prefix(variables: Iterable[DerivedTypeVariable]) -> Set[DerivedTypeVariable]:
        '''Return a set of elements from variables for which no prefix exists in variables. For
        example, if variables were the set {A, A.load}, return the set {A}.
        '''
        selected = set()
        candidates = sorted(variables, reverse=True)
        for index, candidate in enumerate(candidates):
            emit = True
            for other in candidates[index+1:]:
                if other.get_suffix(candidate):
                    emit = False
                    break
            if emit:
                selected.add(candidate)
        return selected

    @staticmethod
    def _remove_self_loops(graph: ConstraintGraph) -> None:
        '''Loops from a node directly to itself are not useful, so it's useful to remove them.
        '''
        graph.graph.remove_edges_from({(node, node) for node in graph.graph.nodes})


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
    def __init__(self, solver: Solver) -> None:
        self.sketches = networkx.DiGraph()
        self.lookup: Dict[DerivedTypeVariable, SketchNode] = {}
        self.solver = solver

    def make_node(self, variable: DerivedTypeVariable) -> SketchNode:
        if variable in self.lookup:
            return self.lookup[variable]
        variance = variable.path_variance
        if variance == Variance.COVARIANT:
            result = SketchNode(variable, self.solver.program.types.top)
        else:
            result = SketchNode(variable, self.solver.program.types.bottom)
        self.lookup[variable] = result
        return result

    def add_variable(self, variable: DerivedTypeVariable) -> None:
        if variable in self.lookup:
            return
        node = self.make_node(variable)
        self.sketches.add_node(node)
        prefix = variable.largest_prefix
        while prefix:
            prefix_node = self.make_node(prefix)
            self.sketches.add_edge(prefix_node, node, label=variable.tail)
            variable = prefix
            node = prefix_node
            prefix = variable.largest_prefix

    def replace_edge(self, head: SkNode, tail: SkNode, new_head: SkNode, new_tail: SkNode) -> None:
        atts = self.sketches[head][tail]
        self.sketches.remove_edge(head, tail)
        self.sketches.add_edge(new_head, new_tail, **atts)

    def replace_node(self, old: SketchNode, new: SkNode) -> None:
        for pred in self.sketches.predecessors(old):
            self.replace_edge(pred, old, pred, new)
        for succ in self.sketches.successors(old):
            self.replace_edge(old, succ, new, succ)

    # TODO dead code - remove when sure it isn't needed
    def normalize(self) -> None:
        # Sort to make this operation deterministic; it's possible that multiple interesting
        # variables could reach the same node. NB this doesn't affect correctness, but it does make
        # debugging easier.
        for variable in sorted(self.solver.interesting):
            node = self.make_node(variable)
            #

        # TODO old dead code
        normal = self.solver.lookup_type_var(variable.base_var)
        if normal:
            result = normal.extend(variable.path)
        result = variable
        print(f'normalized {variable} to {result}')

    def add_sketches(self, constraints: ConstraintSet) -> None:
        for constraint in constraints:
            self.add_variable(constraint.left)
            self.add_variable(constraint.right)
        # remove cycles by replacing nodes with variables when downstream from themselves
        for cycle in networkx.simple_cycles(self.sketches):
            node = cycle[1]
            last = cycle[-1]
            self.replace_edge(last, node, last, node.dtv)
        # refine the types of the nodes. NB this will ignore labels, as self.lookup only tracks
        # nodes
        for constraint in constraints:
            left = constraint.left
            right = constraint.right
            if left in self.solver.program.types.internal_types:
                node = self.lookup[right]
                node.atom = self.solver.program.types.join(node.atom, left)
            if right in self.solver.program.types.internal_types:
                node = self.lookup[left]
                node.atom = self.solver.program.types.meet(node.atom, right)

    def __str__(self) -> str:
        if self.lookup:
            nt = f'{os.linesep}\t'
            return f'sketches:{nt}{nt.join(map(lambda k: f"{k} → {self.lookup[k].atom}", self.lookup))}'
        return 'no sketches'
