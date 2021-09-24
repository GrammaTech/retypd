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
import tqdm

class Solver:
    '''Takes a program and generates subtype constraints. The constructor does not perform the
    computation; rather, :py:class:`Solver` objects are callable as thunks.
    '''
    def __init__(self, program: Program, verbose: bool = False) -> None:
        self.program = program
        self.verbose = verbose
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
                    # Look in graph for predecessors, not fr_graph; the purpose of fr_graph is
                    # merely to ensure that the SCCs we examine don't contain both forget and recall
                    # edges.
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
        forgets.reverse()
        for forget in forgets:
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
        def show_progress(iterable):
            if self.verbose:
                return tqdm.tqdm(iterable)
            return iterable

        derived: Dict[DerivedTypeVariable, ConstraintSet] = {}
        scc_dag = networkx.condensation(self.program.callgraph)
        sketches = Sketches(self)

        accumulated = networkx.DiGraph()

        # The idea here is that functions need to satisfy their clients' needs for inputs and need
        # to use their clients' return values without overextending them. So we do a reverse
        # topological order on functions, lumping SCCs together, and slowly extend the graph.
        # It would be interesting to explore how this algorithm compares with a more restricted one
        # as far as asymptotic complexity, but this is simple to understand and so the right choice
        # for now. I suspect that most of the graphs are fairly sparse and so the practical reality
        # is that there aren't many paths.
        for scc_node in show_progress(reversed(list(networkx.topological_sort(scc_dag)))):
            scc = scc_dag.nodes[scc_node]['members']
            scc_graph = networkx.DiGraph()
            for proc in scc:
                constraints = self.program.proc_constraints.get(proc, ConstraintSet())
                graph = ConstraintGraph(constraints)
                scc_graph.add_edges_from(graph.graph.edges)
                accumulated.add_edges_from(graph.graph.edges)
                for head, tail in graph.graph.edges:
                    label = graph.graph[head][tail].get('label')
                    if label:
                        scc_graph[head][tail]['label'] = label
                        accumulated[head][tail]['label'] = label
            # make a copy; some of this analysis mutates the graph
            self._generate_type_vars(scc_graph)
            Solver._unforgettable_subgraph_split(scc_graph)
            generated = self._generate_constraints(scc_graph)
            sketches.add_constraints(generated, scc)
            derived.update({proc: generated for proc in scc})
        # TODO while examining constraints for each function, accumulate constraints into a dict
        # whenever a constraint includes a global (keys should be globals; values should be
        # ConstraintSets). Create a graph with relationships between globals, identify SCCs, and
        # iterate in reverse topological order, generating and copying sketches as with functions.
        # This should remove the need for accumulated.

        # generate constraints for globals once all information is available
        # TODO: restrict the "interesting endpoints" such that at least one of them
        # must be a global. When updating `derived` below, filter out the constraints
        # that refer to each global.
        generated = self._generate_constraints(accumulated)
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
        return f'{self.dtv}'

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
        return f'{self.target}.{self.id}'

    def __repr__(self) -> str:
        return str(self)


SkNode = Union[SketchNode, LabelNode]


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
        def add_edge(head: SketchNode, tail: SkNode, label: str) -> None:
            # don't emit duplicate edges
            if (head, tail) not in new_sketches.edges:
                new_sketches.add_edge(head, tail, label=label)
        def copy_inner(onto: SketchNode, origin: SketchNode, seen: Dict[SketchNode, SketchNode]) -> None:
            seen[origin] = onto
            # If the dependencies graph says that onto is a subtype of origin, copy all of origin's
            # outgoing edges and their targets. If this introduces a loop, create a label instead of
            # a loop.
            for dependency in dependencies.get(origin, set()):
                if dependency in seen:
                    original = seen[dependency]
                    for succ in self.sketches.successors(dependency):
                        label = self.sketches[dependency][succ]['label']
                        new_succ = LabelNode(original.dtv.add_suffix(label))
                        add_edge(onto, new_succ, label=label)
                else:
                    copy_inner(onto, dependency, seen)
            # Copy all of origin's outgoing edges onto onto.
            for succ in self.sketches.successors(origin):
                label = self.sketches[origin][succ]['label']
                # If succ has been seen, look up the last time it was seen and instead create a
                # label that links to the previous location where it was seen.
                if succ in seen:
                    add_edge(onto, LabelNode(seen[succ].dtv), label=label)
                # If the successor is a label, translate it into this sketch.
                elif isinstance(succ, LabelNode):
                    suffix = succ.target.get_suffix(origin.dtv)
                    if suffix is None:
                        raise ValueError('Sanity check: suffix is non-None')
                    succ_dtv = onto.dtv.remove_suffix(suffix)
                    if not succ_dtv:
                        raise ValueError('Sanity check: remove_suffix was successful')
                    add_edge(onto, LabelNode(succ_dtv), label=label)
                # Otherwise, it's a regular (non-label) node that hasn't been seen, so emit it and
                # explore its successors.
                else:
                    succ_node = self.make_node(onto.dtv.add_suffix(label))
                    add_edge(onto, succ_node, label=label)
                    seen[succ_node] = onto
                    copy_inner(succ_node, succ, seen)
        for dest in dependencies:
            for origin in dependencies.get(dest, set()):
                seen = {self.lookup[n]: self.lookup[n] for n in dest.dtv.all_prefixes()}
                copy_inner(dest, origin, seen)
        self.sketches = new_sketches

    counter = 0
    def add_constraints(self, constraints: ConstraintSet, nodes: Set[DerivedTypeVariable]) -> None:
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
            # Type variables are generated only to break cycles in the constraint graph. If the rhs
            # is a type variable, that means that the relationship between the sketches is
            # intraprocedural and not interprocedural.
            if right_base_var in self.solver._type_vars:
                intra_dependencies.setdefault(left_node, set()).add(right_node)
            # If the right-hand side is not a type variable, it must be an interesting node (a
            # function or global). The left and right base nodes may be different or the same;
            # either way, treat it as an interprocedural dependency (TODO might there be constraints
            # with matching left and right base variables that don't need this treatment).
            elif left_base_var in nodes:
                inter_dependencies.setdefault(left_node, set()).add(right_node)
        self.copy_dependencies(intra_dependencies)
        self.copy_dependencies(inter_dependencies)
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
            return f'nodes:{nt}{nt.join(map(lambda k: f"{k} ({self.lookup[k].atom}", self.lookup))})'
        return 'no sketches'
