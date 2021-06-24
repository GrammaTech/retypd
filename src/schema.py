'''Data types for an implementation of retypd analysis. See `Link the paper
<https://arxiv.org/pdf/1603.05495v1.pdf>`, `Link the slides
<https://raw.githubusercontent.com/emeryberger/PLDI-2016/master/presentations/pldi16-presentation241.pdf>`,
and `Link the notes
<https://git.grammatech.com/reverse-engineering/common/re_facts/-/blob/paldous/type-recovery/docs/how-to/type-recovery.rst>`
for details

author: Peter Aldous
'''

from abc import ABC
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
import logging
import os
import networkx


logging.basicConfig()


class AccessPathLabel(ABC):
    '''Abstract class for capabilities that can be part of a path. See Table 1.

    All :py:class:`AccessPathLabel` objects are comparable to each other; objects are ordered by
    their classes (in the order they are created), then by internal values.
    '''
    def __lt__(self, other: 'AccessPathLabel') -> bool:
        s_type = str(type(self))
        o_type = str(type(other))
        if s_type == o_type:
            return self._less_than(other)
        return s_type < o_type

    def _less_than(self, _other) -> bool:
        '''Compare two objects of the same exact type. Return True if self is less than other; true
        otherwise. Several of the subclasses are singletons, so we return False unless there is a
        need for an overriding implementation.
        '''
        return False

    def is_covariant(self) -> bool:
        '''Determines if the access path label is covariant (True) or contravariant (False), per
        Table 1.
        '''
        return True


class LoadLabel(AccessPathLabel):
    '''A singleton representing the load (read) capability.
    '''
    _instance = None

    def __init__(self) -> None:
        raise ValueError("Can't instantiate; call instance() instead")

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
        return cls._instance

    def __eq__(self, other: Any) -> bool:
        return self is other

    def __hash__(self) -> int:
        return 0

    def __str__(self) -> str:
        return 'load'


class StoreLabel(AccessPathLabel):
    '''A singleton representing the store (write) capability.
    '''
    _instance = None

    def __init__(self) -> None:
        raise ValueError("Can't instantiate; call instance() instead")

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
        return cls._instance

    def __eq__(self, other: Any) -> bool:
        return self is other

    def __hash__(self) -> int:
        return 1

    def is_covariant(self) -> bool:
        return False

    def __str__(self) -> str:
        return 'store'


class InLabel(AccessPathLabel):
    '''Represents a parameter to a function, specified by an index.
    '''
    def __init__(self, index: int) -> None:
        self.index = index

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, InLabel) and self.index == other.index

    def _less_than(self, other: 'InLabel') -> bool:
        return self.index < other.index

    def __hash__(self) -> int:
        return hash(self.index)

    def is_covariant(self) -> bool:
        return False

    def __str__(self) -> str:
        return f'in_{self.index}'


class OutLabel(AccessPathLabel):
    '''Represents a return from a function. This class is a singleton.
    '''
    _instance = None

    def __init__(self) -> None:
        raise ValueError("Can't instantiate; call instance() instead")

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
        return cls._instance

    def __eq__(self, other: Any) -> bool:
        return self is other

    def __hash__(self) -> int:
        return 2

    def __str__(self) -> str:
        return 'out'


class DerefLabel(AccessPathLabel):
    '''Represents a dereference in an access path. Specifies a size (the number of bytes read or
    written) and an offset (the number of bytes from the base).
    '''
    def __init__(self, size: int, offset: int) -> None:
        self.size = size
        self.offset = offset

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, DerefLabel) and
                self.size == other.size and
                self.offset == other.offset)

    def _less_than(self, other: 'DerefLabel') -> bool:
        if self.offset == other.offset:
            return self.size < other.size
        return self.offset < other.offset

    def __hash__(self) -> int:
        return hash(self.offset) ^ hash(self.size)

    def __str__(self) -> str:
        return f'σ{self.size}@{self.offset}'


class DerivedTypeVariable:
    '''A _derived_ type variable, per Definition 3.1. Immutable (by convention).
    '''
    def __init__(self, type_var: str, path: Optional[Sequence[AccessPathLabel]] = None) -> None:
        self.base = type_var
        if path is None:
            self.path: Sequence[AccessPathLabel] = ()
        else:
            self.path = tuple(path)
        if self.path:
            self._str: str = f'{self.base}.{".".join(map(str, self.path))}'
        else:
            self._str: str = self.base

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, DerivedTypeVariable) and
                self.base == other.base and
                self.path == other.path)

    def __lt__(self, other: 'DerivedTypeVariable') -> bool:
        if self.base == other.base:
            return list(self.path) < list(other.path)
        return self.base < other.base

    def __hash__(self) -> int:
        return hash(self.base) ^ hash(self.path)

    def largest_prefix(self) -> Optional['DerivedTypeVariable']:
        '''Return the prefix obtained by removing the last item from the type variable's path. If
        there is no path, return None.
        '''
        if self.path:
            return DerivedTypeVariable(self.base, self.path[:-1])
        return None

    def prefixes(self) -> Set['ExistenceConstraint']:
        '''Retrieve all prefixes of the derived type variable as a set.
        '''
        path = tuple(self.path)
        result = set()
        while path:
            path = path[:-1]
            prefix = DerivedTypeVariable(self.base, path)
            result.add(ExistenceConstraint(prefix))
        return result

    def tail(self) -> AccessPathLabel:
        '''Retrieve the last item in the access path, if any. Return None if
        the path is empty.
        '''
        if self.path:
            return self.path[-1]
        return None

    def add_suffix(self, suffix: AccessPathLabel) -> 'DerivedTypeVariable':
        '''Create a new :py:class:`DerivedTypeVariable` identical to :param:`self` (which is
        unchanged) but with suffix appended to its path.
        '''
        path: List[AccessPathLabel] = list(self.path)
        path.append(suffix)
        return DerivedTypeVariable(self.base, path)

    def get_single_suffix(self, prefix: 'DerivedTypeVariable') -> Optional[AccessPathLabel]:
        '''If :param:`prefix` is a prefix of :param:`self` with exactly one additional
        :py:class:`AccessPathLabel`, return the additional label. If not, return `None`.
        '''
        if (self.base != prefix.base or
                len(self.path) != (len(prefix.path) + 1) or
                self.path[:-1] != prefix.path):
            return None
        return self.tail()

    def path_is_covariant(self):
        '''Determine if the access path is covariant or contravariant. This is a special case of
        :py:classmethod:`suffix_is_covariant`.
        '''
        return DerivedTypeVariable.suffix_is_covariant(self.path)

    @classmethod
    def suffix_is_covariant(cls, suffix: Sequence[AccessPathLabel]) -> bool:
        '''Given a sequence of :py:class:`AccessPathLabel` objects, determine if the suffix is
        covariant or contravariant.
        '''
        is_covariant = True
        for label in suffix:
            if not label.is_covariant():
                is_covariant = not is_covariant
        return is_covariant

    def __str__(self) -> str:
        return self._str


class ExistenceConstraint:
    '''A type constraint of the form VAR a (see Definition 3.3)
    '''
    def __init__(self, var: DerivedTypeVariable) -> None:
        self.var = var

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, ExistenceConstraint) and
                self.var == other.var)

    def __lt__(self, other: 'ExistenceConstraint') -> bool:
        return self.var < other.var

    def __hash__(self) -> int:
        return hash(self.var)

    def __str__(self) -> str:
        return f'VAR {self.var}'


class SubtypeConstraint:
    '''A type constraint of the form left ⊑ right (see Definition 3.3)
    '''
    def __init__(self, left: DerivedTypeVariable, right: DerivedTypeVariable) -> None:
        self.left = left
        self.right = right

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, SubtypeConstraint) and
                self.left == other.left and
                self.right == other.right)

    def __lt__(self, other: 'SubtypeConstraint') -> bool:
        if self.left == other.left:
            return self.right < other.right
        return self.left < other.left

    def __hash__(self) -> int:
        return hash(self.left) ^ hash(self.right)

    def __str__(self) -> str:
        return f'{self.left} ⊑ {self.right}'


class ConstraintSet:
    '''A (partitioned) set of type constraints
    '''
    def __init__(self,
                 existence: Optional[Iterable[ExistenceConstraint]] = None,
                 subtype: Optional[Iterable[SubtypeConstraint]] = None) -> None:
        if existence:
            self.existence = set(existence)
        else:
            self.existence = set()
        if subtype:
            self.subtype = set(subtype)
        else:
            self.subtype = set()
        self.logger = logging.getLogger('ConstraintSet')

    def add_subtype(self, left: DerivedTypeVariable, right: DerivedTypeVariable) -> bool:
        '''Add a subtype constraint
        '''
        constraint = SubtypeConstraint(left, right)
        if constraint in self.subtype:
            return False
        self.subtype.add(constraint)
        return True

    def add_existence(self, var: DerivedTypeVariable) -> bool:
        '''Add an existence constraint
        '''
        ex = ExistenceConstraint(var)
        if ex in self.existence:
            return False
        self.existence.add(ex)
        return True

    def __str__(self) -> str:
        nt = os.linesep + '\t'
        return (f'ConstraintSet:{nt}{nt.join(map(str, self.existence))}'
                f'{os.linesep}{nt}{nt.join(map(str,self.subtype))}')

    def generate_graph(self) -> 'ConstraintGraph':
        '''Produce a graph from the set of constraints. This corresponds to step 3 in the notes.
        '''
        graph = ConstraintGraph()
        for ex in self.existence:
            var = ex.var
            graph.add_node(var)
        for sub_constraint in self.subtype:
            graph.add_edges(sub_constraint.left, sub_constraint.right)
        return graph


class Vertex:
    '''A vertex in the graph of constraints. Vertex objects are immutable (by convention).
    '''
    def __init__(self,
                 base: DerivedTypeVariable,
                 suffix_variance: bool,
                 recall: bool = False) -> None:
        self.base = base
        self.suffix_variance = suffix_variance
        if suffix_variance:
            variance = '.⊕'
        else:
            variance = '.⊖'
        self._recall = recall
        if recall:
            self._str = 'R:' + str(self.base) + variance
        else:
            self._str = str(self.base) + variance

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, Vertex) and
                self.base == other.base and
                self.suffix_variance == other.suffix_variance and
                self._recall == other._recall)

    def __hash__(self) -> int:
        # TODO rotate one of the bool digests
        return hash(self.base) ^ hash(self.suffix_variance) ^ hash(self._recall)

    def forget_once(self) -> Optional['Vertex']:
        '''"Forget" the last element in the access path, creating a new Vertex. The new Vertex has
        variance that reflects this change.
        '''
        prefix = self.base.largest_prefix()
        if prefix:
            last = self.base.path[-1]
            return Vertex(prefix, last.is_covariant() == self.suffix_variance)
        return None

    def recall(self, label: AccessPathLabel) -> 'Vertex':
        '''"Recall" label, creating a new Vertex. The new Vertex has variance that reflects this
        change.
        '''
        path = list(self.base.path)
        path.append(label)
        variance = self.suffix_variance == label.is_covariant()
        return Vertex(DerivedTypeVariable(self.base.base, path), variance)

    def _replace_last(self, label: AccessPathLabel) -> 'Vertex':
        '''Create a new Vertex whose access path's last label has been replaced by :param:`label`.
        Does not change variance.
        '''
        path = list(self.base.path[:-1])
        path.append(label)
        return Vertex(DerivedTypeVariable(self.base.base, path), self.suffix_variance)

    def implicit_target(self) -> Optional['Vertex']:
        '''If there is a lazily instantiated store/load edge from this node, find its target.
        '''
        if self.base.path:
            last = self.base.path[-1]
            if last is StoreLabel.instance() and self.suffix_variance:
                return self._replace_last(LoadLabel.instance())
            if last is LoadLabel.instance() and not self.suffix_variance:
                return self._replace_last(StoreLabel.instance())
        return None

    def __str__(self) -> str:
        return self._str

    def dual(self) -> 'Vertex':
        return Vertex(self.base, self.suffix_variance, not self._recall)

    def inverse(self) -> 'Vertex':
        return Vertex(self.base, not self.suffix_variance, self._recall)


class ForgetLabel:
    '''A forget label in the graph.
    '''
    def __init__(self, capability: AccessPathLabel) -> None:
        self.capability = capability

    def is_forget(self) -> bool:
        return True

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, ForgetLabel) and self.capability == other.capability

    def __hash__(self) -> int:
        return hash(self.capability)

    def __str__(self) -> str:
        return f'forget {self.capability}'


class RecallLabel:
    '''A recall label in the graph.
    '''
    def __init__(self, capability: AccessPathLabel) -> None:
        self.capability = capability

    def is_forget(self) -> bool:
        return False

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, RecallLabel) and self.capability == other.capability

    def __hash__(self) -> int:
        return ~hash(self.capability)

    def __str__(self) -> str:
        return f'recall {self.capability}'


EdgeLabel = Union[ForgetLabel, RecallLabel]


class ConstraintGraph:
    '''Represents the constraint graph in the slides. Essentially the same as the transducer from
    Appendix D. Edge weights use the formulation from the paper.
    '''
    def __init__(self) -> None:
        self.graph = networkx.DiGraph()

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
        return (f'{nt.join(map(str, graph.nodes))}{os.linesep}'
                f'{nt}{nt.join(map(edge_to_str, graph.edges))}')

    def __str__(self) -> str:
        nt = os.linesep + '\t'
        return f'ConstraintGraph:{nt}{ConstraintGraph.graph_to_str(self.graph)}'


class Solver:
    '''Takes a saturated constraint graph and a set of interesting variables and generates subtype
    constraints.
    '''
    def __init__(self, graph: ConstraintGraph, interesting: Set[DerivedTypeVariable]) -> None:
        self.graph = networkx.DiGraph(graph.graph)
        self.interesting = interesting
        self.next = 0
        self.constraints: Set[SubtypeConstraint] = set()
        self._type_vars: Dict[DerivedTypeVariable, DerivedTypeVariable] = {}

    def _forget_recall_transform(self):
        '''Transform the graph so that no paths can be found such that a forget edge succeeds a
        recall edge.
        '''
        edges = set(self.graph.edges)
        for head, tail in edges:
            label = self.graph[head][tail].get('label')
            if label and label.is_forget():
                continue
            recall_head = head.dual()
            recall_tail = tail.dual()
            atts = self.graph[head][tail]
            if label and not label.is_forget():
                self.graph.remove_edge(head, tail)
                self.graph.add_edge(head, recall_tail, **atts)
            self.graph.add_edge(recall_head, recall_tail, **atts)

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

    def _get_type_var(self, v: Vertex) -> DerivedTypeVariable:
        var = v.base
        if var in self._type_vars:
            return self._type_vars[var]
        return var

    def _find_paths(self,
                    origin: Vertex,
                    path: List[Vertex] = [],
                    string: List[EdgeLabel] = []) -> \
                        List[Tuple[List[EdgeLabel], Vertex]]:
        '''Find all non-empty paths from origin to nodes that represent interesting type variables.
        '''
        # print(f'checking if {origin.base} is interesting')
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

    def _add_constraint(self,
                        origin: Vertex,
                        dest: Vertex,
                        string: List[EdgeLabel]) -> None:
        lhs = origin
        rhs = dest
        for label in string:
            if label.is_forget():
                rhs = rhs.recall(label.capability)
            else:
                lhs = lhs.recall(label.capability)
        lhs_var = self._get_type_var(lhs)
        rhs_var = self._get_type_var(rhs)
        constraint = SubtypeConstraint(lhs_var, rhs_var)
        # print(f'Maybe adding constraint: {constraint}')
        if lhs_var != rhs_var and lhs.suffix_variance and rhs.suffix_variance:
            # print('\tdoing it')
            self.constraints.add(constraint)

    def _remove_SCC_internal_edges(self, scc: Set[Vertex]) -> None:
        self.graph.remove_edges_from({(head, tail) for head in scc for tail in scc})

    def _generate_type_vars(self) -> None:
        forget_graph = networkx.DiGraph(self.graph)
        recall_graph = networkx.DiGraph(self.graph)
        for head, tail in self.graph.edges:
            label = self.graph[head][tail].get('label')
            if label:
                if label.is_forget():
                    recall_graph.remove_edge(head, tail)
                else:
                    forget_graph.remove_edge(head, tail)
        for graph in [forget_graph, recall_graph]:
            condensation = networkx.condensation(graph)
            visited = set()
            for scc_node in reversed(list(networkx.topological_sort(condensation))):
                scc = condensation.nodes[scc_node]['members']
                visited.add(scc_node)
                if len(scc) == 1:
                    continue
                for node in scc:
                    for predecessor in self.graph.predecessors(node):
                        scc_index = condensation.graph['mapping'][predecessor]
                        if scc_index not in visited:
                            self._make_type_var(node.base)

    def _generate_cyclical_constraints(self) -> None:
        condensation = networkx.condensation(self.graph)
        for scc_node in reversed(list(networkx.topological_sort(condensation))):
            scc = condensation.nodes[scc_node]['members']
            # generate type constraints from each header node to any interesting nodes it can reach
            # (including itself)
            for node in scc:
                if node.base in self.interesting:
                    print(f'found {node} in an SCC')
                    for string, dest in self._find_paths(node):
                        self._add_constraint(node, dest, string)
            # break the cycle by dropping all edges between the vertices
            self._remove_SCC_internal_edges(scc)

    def _remove_self_loops(self) -> None:
        self.graph.remove_edges_from({(node, node) for node in self.graph.nodes})

    def __call__(self) -> Set[SubtypeConstraint]:
        # TODO forget/recall transform introduces artifacts into the header calculation - but the
        # SCC algorithm depends on not having bogus cycles
        #
        self._remove_self_loops()
        self._generate_type_vars()
        self._forget_recall_transform()
        self._generate_cyclical_constraints()
        return self.constraints

