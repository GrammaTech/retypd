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

from typing import Dict, List, Iterable, Set, Tuple, Union
from .graph import EdgeLabel, Node, ConstraintGraph
from .schema import ConstraintSet, DerivedTypeVariable, SubtypeConstraint, Variance
from .parser import SchemaParser
import networkx


class Solver:
    '''Takes a saturated constraint graph and a set of interesting variables and generates subtype
    constraints. The constructor does not perform the computation; rather, :py:class:`Solver`
    objects are callable as thunks.
    '''
    def __init__(self,
                 constraints: ConstraintSet,
                 interesting: Iterable[Union[DerivedTypeVariable, str]]) -> None:
        self.constraint_graph = ConstraintGraph(constraints)
        self.interesting: Set[DerivedTypeVariable] = set()
        for var in interesting:
            if isinstance(var, str):
                self.interesting.add(DerivedTypeVariable(var))
            else:
                self.interesting.add(var)
        self.next = 0
        self.constraints: Set[SubtypeConstraint] = set()
        self._type_vars: Dict[DerivedTypeVariable, DerivedTypeVariable] = {}

    def _add_forget_recall_edges(self) -> None:
        '''Passes through to ConstraintGraph.add_forget_recall()
        '''
        self.constraint_graph.add_forget_recall()

    def _saturate(self) -> None:
        '''Passes through to ConstraintGraph.saturate()
        '''
        self.constraint_graph.saturate()

    def _unforgettable_subgraph_split(self) -> None:
        '''The algorithm, after saturation, only admits paths such that forget edges all precede
        the first recall edge (if there is such an edge). To enforce this, we modify the graph by
        splitting each node and the unlabeled and recall edges (but not forget edges!). Recall edges
        in the original graph are changed to point to the 'unforgettable' duplicate of their
        original target. As a result, no forget edges are reachable after traversing a single recall
        edge.
        '''
        edges = set(self.graph.edges)
        for head, tail in edges:
            label = self.graph[head][tail].get('label')
            if label and label.kind == EdgeLabel.Kind.FORGET:
                continue
            recall_head = head.split_unforgettable()
            recall_tail = tail.split_unforgettable()
            atts = self.graph[head][tail]
            if label and label.kind == EdgeLabel.Kind.RECALL:
                self.graph.remove_edge(head, tail)
                self.graph.add_edge(head, recall_tail, **atts)
            self.graph.add_edge(recall_head, recall_tail, **atts)

    def lookup_type_var(self, var_str: str) -> DerivedTypeVariable:
        '''Take a string, convert it to a DerivedTypeVariable, and if there is a type variable that
        stands in for a prefix of it, change the prefix to the type variable.
        '''
        var = SchemaParser.parse_variable(var_str)
        return self._get_type_var(var)

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

    def _make_type_var(self, base: DerivedTypeVariable) -> None:
        '''Retrieve or generate a type variable. Automatically adds this variable to the set of
        interesting variables.
        '''
        if base in self._type_vars:
            return
        var = DerivedTypeVariable(f'τ_{self.next}')
        self._type_vars[base] = var
        self.interesting.add(base)
        self.next += 1

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

    def _generate_type_vars(self) -> None:
        '''Identify at least one node in each nontrivial SCC and generate a type variable for it.
        This ensures that the path exploration algorithm will never loop; instead, it will generate
        constraints that are recursive on the type variable.

        To do so, find strongly connected components (such that loops may contain forget or recall
        edges but never both). In each nontrivial SCC (in reverse topological order), identify all
        vertices with predecessors in a strongly connected component that has not yet been visited.
        If any of the identified variables is a prefix of another (e.g., φ and φ.load.σ8@0),
        eliminate the longer one. Add these variables to a set of candidates and to the set of
        interesting variables, where path execution will stop.

        Once all SCCs have been processed, minimize the set of candidates as before (remove
        variables that have a prefix in the set) and emit type variables for each remaining
        candidate.
        '''
        forget_graph = networkx.DiGraph(self.graph)
        recall_graph = networkx.DiGraph(self.graph)
        for head, tail in self.graph.edges:
            label = self.graph[head][tail].get('label')
            if label:
                if label.kind == EdgeLabel.Kind.FORGET:
                    recall_graph.remove_edge(head, tail)
                else:
                    forget_graph.remove_edge(head, tail)
        loop_breakers: Set[DerivedTypeVariable] = set()
        for graph in [forget_graph, recall_graph]:
            condensation = networkx.condensation(graph)
            visited = set()
            for scc_node in reversed(list(networkx.topological_sort(condensation))):
                candidates: Set[DerivedTypeVariable] = set()
                scc = condensation.nodes[scc_node]['members']
                visited.add(scc_node)
                if len(scc) == 1:
                    continue
                for node in scc:
                    for predecessor in self.graph.predecessors(node):
                        scc_index = condensation.graph['mapping'][predecessor]
                        if scc_index not in visited:
                            candidates.add(node.base)
                candidates = Solver._filter_no_prefix(candidates)
                loop_breakers |= candidates
                self.interesting |= candidates
        loop_breakers = Solver._filter_no_prefix(loop_breakers)
        for var in loop_breakers:
            self._make_type_var(var)

    def _find_paths(self,
                    origin: Node,
                    path: List[Node] = [],
                    string: List[EdgeLabel] = []) -> \
                        List[Tuple[List[EdgeLabel], Node]]:
        '''Find all non-empty paths from origin to nodes that represent interesting type variables.
        Return the list of labels encountered along the way as well as the destination reached.
        '''
        if path and origin.base in self.interesting:
            return [(string, origin)]
        if origin in path:
            return []
        path = list(path)
        path.append(origin)
        all_paths: List[Tuple[List[EdgeLabel], Node]] = []
        if origin in self.graph:
            for succ in self.graph[origin]:
                label = self.graph[origin][succ].get('label')
                new_string = list(string)
                if label:
                    new_string.append(label)
                all_paths += self._find_paths(succ, path, new_string)
        return all_paths

    def _maybe_add_constraint(self,
                        origin: Node,
                        dest: Node,
                        string: List[EdgeLabel]) -> None:
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
                constraint = SubtypeConstraint(lhs_var, rhs_var)
                self.constraints.add(constraint)

    def _generate_constraints(self) -> None:
        '''Now that type variables have been computed, no cycles can be produced. Find paths from
        interesting nodes to other interesting nodes and generate constraints.
        '''
        for node in self.graph.nodes:
            if node.base in self.interesting:
                for string, dest in self._find_paths(node):
                    self._maybe_add_constraint(node, dest, string)

    def _remove_self_loops(self) -> None:
        '''Loops from a node directly to itself are not useful, so it's useful to remove them.
        '''
        self.graph.remove_edges_from({(node, node) for node in self.graph.nodes})

    def __call__(self) -> Set[SubtypeConstraint]:
        '''Perform the retypd calculation.
        '''
        self._add_forget_recall_edges()
        self._saturate()
        self.graph = networkx.DiGraph(self.constraint_graph.graph)
        self._remove_self_loops()
        self._generate_type_vars()
        self._unforgettable_subgraph_split()
        self._generate_constraints()
        return self.constraints


