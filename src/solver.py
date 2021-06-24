from abc import ABC
from typing import Any, Dict, List, Iterable, Optional, Set, Tuple, Union
from .schema import AccessPathLabel, ConstraintSet, DerivedTypeVariable, LoadLabel, StoreLabel, \
        SubtypeConstraint, Vertex, EdgeLabel, ForgetLabel, RecallLabel
from .parser import SchemaParser
import os
import networkx


class ConstraintGraph:
    '''Represents the constraint graph in the slides. Essentially the same as the transducer from
    Appendix D. Edge weights use the formulation from the paper.
    '''
    def __init__(self, constraints: ConstraintSet) -> None:
        self.graph = networkx.DiGraph()
        for constraint in constraints.subtype:
            self.add_edges(constraint.left, constraint.right)

    def add_node(self, node: DerivedTypeVariable) -> None:
        '''Add a node with covariant and contravariant suffixes to the graph.
        '''
        self.graph.add_node(Vertex(node, True))
        self.graph.add_node(Vertex(node, False))

    def add_edge(self, head: Vertex, tail: Vertex, **atts) -> bool:
        if head not in self.graph or tail not in self.graph[head]:
            self.graph.add_edge(head, tail, **atts)
            return True
        return False

    def add_edges(self, sub: DerivedTypeVariable, sup: DerivedTypeVariable, **atts) -> bool:
        '''Add an edge to the underlying graph. Also add its reverse with reversed variance.
        '''
        changed = False
        forward_from = Vertex(sub, True)
        forward_to = Vertex(sup, True)
        changed = self.add_edge(forward_from, forward_to, **atts) or changed
        backward_from = forward_to.inverse()
        backward_to = forward_from.inverse()
        changed = self.add_edge(backward_from, backward_to, **atts) or changed
        return changed

    def add_forget_recall(self) -> None:
        '''Add forget and recall nodes to the graph. Step 4 in the notes.
        '''
        existing_nodes = set(self.graph.nodes)
        for node in existing_nodes:
            prefix = node.forget_once()
            while prefix:
                forgotten = node.base.path[-1]
                self.graph.add_edge(node, prefix, label=ForgetLabel(forgotten))
                self.graph.add_edge(prefix, node, label=RecallLabel(forgotten))
                node = prefix
                prefix = node.forget_once()

    def saturate(self) -> None:
        '''Add "shortcut" edges, per algorithm D.2 in the paper.
        '''
        changed = False
        reaching_R: Dict[Vertex, Set[Tuple[AccessPathLabel, Vertex]]] = {}

        def add_forgets(dest: Vertex, forgets: Set[Tuple[AccessPathLabel, Vertex]]):
            nonlocal changed
            if dest not in reaching_R or not (forgets <= reaching_R[dest]):
                changed = True
                reaching_R.setdefault(dest, set()).update(forgets)

        def add_edge(origin: Vertex, dest: Vertex):
            nonlocal changed
            changed = self.add_edge(origin, dest) or changed

        for head_x, tail_y in self.graph.edges:
            label = self.graph[head_x][tail_y].get('label')
            if label and label.is_forget():
                add_forgets(tail_y, {(label.capability, head_x)})
        while changed:
            changed = False
            for head_x, tail_y in self.graph.edges:
                if not self.graph[head_x][tail_y].get('label'):
                    add_forgets(tail_y, reaching_R.get(head_x, set()))
            existing_edges = list(self.graph.edges)
            for head_x, tail_y in existing_edges:
                label = self.graph[head_x][tail_y].get('label')
                if label and not label.is_forget():
                    capability_l = label.capability
                    for (label, origin_z) in reaching_R.get(head_x, set()):
                        if label == capability_l:
                            add_edge(origin_z, tail_y)
            contravariant_vars = list(filter(lambda v: not v.suffix_variance, self.graph.nodes))
            for x in contravariant_vars:
                for (capability_l, origin_z) in reaching_R.get(x, set()):
                    label = None
                    if capability_l == StoreLabel.instance():
                        label = LoadLabel.instance()
                    if capability_l == LoadLabel.instance():
                        label = StoreLabel.instance()
                    if label:
                        add_forgets(x.inverse(), {(label, origin_z)})

    @staticmethod
    def edge_to_str(graph, edge: Tuple[Vertex, Vertex]) -> str:
        '''A helper for __str__ that formats an edge
        '''
        width = 2 + max(map(lambda v: len(str(v)), graph.nodes))
        (sub, sup) = edge
        label = graph[sub][sup].get('label')
        edge_str = f'{str(sub):<{width}}→  {str(sup):<{width}}'
        if label:
            return edge_str + f' ({label})'
        else:
            return edge_str

    @staticmethod
    def graph_to_dot(name: str, graph: networkx.DiGraph) -> str:
        nt = os.linesep + '\t'
        def edge_to_str(edge: Tuple[Vertex, Vertex]) -> str:
            (sub, sup) = edge
            label = graph[sub][sup].get('label')
            label_str = ''
            if label:
                label_str = f' [label="{label}"]'
            return f'"{sub}" -> "{sup}"{label_str};'
        def node_to_str(node: Vertex) -> str:
            return f'"{node}";'
        return (f'digraph {name} {{{nt}{nt.join(map(node_to_str, graph.nodes))}{nt}'
                f'{nt.join(map(edge_to_str, graph.edges))}{os.linesep}}}')

    @staticmethod
    def write_to_dot(name: str, graph: networkx.DiGraph) -> None:
        with open(f'{name}.dot', 'w') as dotfile:
            print(ConstraintGraph.graph_to_dot(name, graph), file=dotfile)

    @staticmethod
    def graph_to_str(graph: networkx.DiGraph) -> str:
        nt = os.linesep + '\t'
        edge_to_str = lambda edge: ConstraintGraph.edge_to_str(graph, edge)
        return (f'{nt.join(map(edge_to_str, graph.edges))}')

    def __str__(self) -> str:
        nt = os.linesep + '\t'
        return f'ConstraintGraph:{nt}{ConstraintGraph.graph_to_str(self.graph)}'


class Solver:
    '''Takes a saturated constraint graph and a set of interesting variables and generates subtype
    constraints. The constructor does not perform the computation; rather, :py:class:`Solver`
    objects are callable as thunks.
    '''
    def __init__(self,
                 constraints: ConstraintSet,
                 interesting: Set[Union[DerivedTypeVariable, str]]) -> None:
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
        self.constraint_graph.add_forget_recall()

    def _saturate(self) -> None:
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
            if label and label.is_forget():
                continue
            recall_head = head.split_unforgettable()
            recall_tail = tail.split_unforgettable()
            atts = self.graph[head][tail]
            if label and not label.is_forget():
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
    def _filter_no_prefix(dtvs: Iterable[DerivedTypeVariable]) -> Set[DerivedTypeVariable]:
        selected = set()
        candidates = sorted(dtvs, reverse=True)
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
                if label.is_forget():
                    recall_graph.remove_edge(head, tail)
                else:
                    forget_graph.remove_edge(head, tail)
        loop_breakers = set()
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
                    origin: Vertex,
                    path: List[Vertex] = [],
                    string: List[EdgeLabel] = []) -> \
                        List[Tuple[List[EdgeLabel], Vertex]]:
        '''Find all non-empty paths from origin to nodes that represent interesting type variables.
        '''
        if path and origin.base in self.interesting:
            return [(string, origin)]
        if origin in path:
            return []
        path = list(path)
        path.append(origin)
        all_paths: List[Tuple[List[EdgeLabel], Vertex]] = []
        if origin in self.graph:
            for succ in self.graph[origin]:
                label = self.graph[origin][succ].get('label')
                new_string = list(string)
                if label:
                    new_string.append(label)
                all_paths += self._find_paths(succ, path, new_string)
        return all_paths

    def _maybe_add_constraint(self,
                        origin: Vertex,
                        dest: Vertex,
                        string: List[EdgeLabel]) -> None:
        '''Generate constraints by adding the forgets in string to origin and the recalls in string
        to dest. If both of the generated vertices are covariant (the empty string's variance is
        covariant, so only covariant vertices can represent a derived type variable without an
        elided portion of its path) and if the two variables are not equal, emit a constraint.
        '''
        lhs = origin
        rhs = dest
        for label in string:
            if label.is_forget():
                rhs = rhs.recall(label.capability)
            else:
                lhs = lhs.recall(label.capability)
        if lhs.suffix_variance and rhs.suffix_variance:
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


