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

from __future__ import annotations
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple, Union, Any
from .graph import EdgeLabel, Node, ConstraintGraph
from .schema import (
    ConstraintSet,
    DerivedTypeVariable,
    Lattice,
    Program,
    SubtypeConstraint,
    Variance,
    LoadLabel, 
    StoreLabel
)

from .sketches import LabelNode,SkNode, SketchNode
from .loggable import Loggable, LogLevel
from .parser import SchemaParser
import os
import itertools
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
    G.render(filename, format='svg', view=False)


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
    # Keep output constraints after use; set to False to save memory
    keep_output_constraints: bool = True
    # More precise global handling
    # By default, we propagate globals up the callgraph, inlining them into the sketches as we
    # go, and the global sketches in the final (synthetic) root of the callgraph are the results
    # However, this can be slow and for large binaries the extra precision may not be worth the
    # time cost. If this is set to False we do a single unification of globals at the end, instead
    # of pulling them up the callgraph.
    precise_globals: bool = True


class GlobalHandler:
    def __init__(self,
                 global_vars: Set[DerivedTypeVariable],
                 callgraph: networkx.DiGraph,
                 scc_dag: networkx.DiGraph,
                 fake_root: DerivedTypeVariable) -> None:
        self.scc_dag = scc_dag
        self.callgraph = callgraph
        self.global_vars = global_vars
        self.fake_root = fake_root

    def pre_scc(self, scc_node: Any):
        raise NotImplementedError("Child class must implement")

    def post_scc(self,
                 scc_node: Any,
                 sketches_map: Dict[DerivedTypeVariable, Sketches]) -> None:
        raise NotImplementedError("Child class must implement")

    def copy_globals(self,
                     current_sketch: Sketches,
                     nodes: Set[DerivedTypeVariable],
                     sketches_map: Dict[DerivedTypeVariable, Sketches]) -> None:
        raise NotImplementedError("Child class must implement")

    def finalize(self,
                 solver: "Solver",
                 sketches_map: Dict[DerivedTypeVariable, Sketches]) -> None:
        raise NotImplementedError("Child class must implement")


class PreciseGlobalHandler(GlobalHandler):
    """
    Handle globals precisely, by bubbling sketches up the callgraph, accumulating more and more
    globals (and sketch nodes) as it goes. Once it reaches the roots of the callgraph, we have
    all the globals and these are the final sketches.
    """
    def __init__(self,
                 global_vars: Set[DerivedTypeVariable],
                 callgraph: networkx.DiGraph,
                 scc_dag: networkx.DiGraph,
                 fake_root: DerivedTypeVariable) -> None:
        super(PreciseGlobalHandler, self).__init__(global_vars, callgraph, scc_dag, fake_root)
        # SCCs that have not yet had their sketches cleansed of globals
        # TODO get proper type annotations
        self.not_cleaned_up: List[Tuple[Any, Set[Any]]] = []


    def cleanse_globals(self, sketches: Sketches):
        """
        Delete all the global nodes from a particular sketch graph.
        """
        for dtv, node in list(sketches.lookup.items()):
            if dtv.base_var in self.global_vars:
                if node in sketches.sketches.nodes:
                    sketches.sketches.remove_node(node)
                del sketches.lookup[dtv]


    def pre_scc(self, scc_node: Any) -> None:
        """
        Sets up a list of all SCCs that still have callers to be processing. See post_scc for
        what happens when all callers have been processed.
        """
        # TODO proper type annotation
        caller_scc_set: Set[Any] = set()
        for src_id, _ in self.scc_dag.in_edges(scc_node):
            caller_scc = self.scc_dag.nodes[src_id]["members"]
            caller_scc_set |= caller_scc
        scc = self.scc_dag.nodes[scc_node]["members"]
        if scc != {self.fake_root}:
            self.not_cleaned_up.append( (scc, caller_scc_set) )


    def post_scc(self,
                 scc_node: Any,
                 sketches_map: Dict[DerivedTypeVariable, Sketches]) -> None:
        """
        Checks each SCC to see if all callers of that SCC have been processed. If so, all the
        globals for that SCC can be cleaned up from the sketches (they will have been copied
        into the callers, and live on).

        Cleaning up like this gives massive reduction in RAM usage for large programs.
        """
        # Cleanup anything we can
        scc = self.scc_dag.nodes[scc_node]["members"]
        not_cleaned_up_new = []
        for other_scc, other_scc_callers in self.not_cleaned_up:
            other_scc_callers -= scc
            if len(other_scc_callers) == 0:
                for item in other_scc:
                    self.cleanse_globals(sketches_map[item])
            else:
                not_cleaned_up_new.append( (other_scc, other_scc_callers) )
        self.not_cleaned_up = not_cleaned_up_new


    def copy_globals(self,
                     current_sketch: Sketches,
                     nodes: Set[DerivedTypeVariable],
                     sketches_map: Dict[DerivedTypeVariable, Sketches]) -> None:
        """
        Copy all the global information from downstream functions (callees) to the current
        sketch context.
        """
        # This purposefully re-uses nodes to save memory. The only downside of this is the atoms
        # of nodes in callees can get mutated by callers; since they are globals, this seems like
        # a reasonable trade-off. The memory usage can get quite massive otherwise.
        callees = set()
        # Get all direct callees
        for n in nodes:
            for _, callee in self.callgraph.out_edges(n):
                callees.add(callee)

        for callee in callees:
            # This happens in recursive relationships, and is ok
            if callee not in sketches_map:
                continue
            sketches = sketches_map[callee]
            current_sketch.copy_globals_from_sketch(self.global_vars, sketches)


    def finalize(self,
                 solver: "Solver",
                 sketches_map: Dict[DerivedTypeVariable, Sketches]) -> None:
        pass


class UnionGlobalHandler(GlobalHandler):
    """
    Ignore globals (but leave them in sketches) during the call-graph pass, and at the end
    collect all the global sketches from all the functions.
    """
    def pre_scc(self, scc_node: Any) -> None:
        pass
    def post_scc(self,
                 scc_node: Any,
                 sketches_map: Dict[DerivedTypeVariable, Sketches]) -> None:
        pass
    def copy_globals(self,
                     current_sketch: Sketches,
                     nodes: Set[DerivedTypeVariable],
                     sketches_map: Dict[DerivedTypeVariable, Sketches]) -> None:
        pass

    def finalize(self,
                 solver: "Solver",
                 sketches_map: Dict[DerivedTypeVariable, Sketches]) -> None:
        global_sketches = Sketches(solver)
        for func_sketches in sketches_map.values():
            global_sketches.copy_globals_from_sketch(
                self.global_vars, func_sketches)
        sketches_map[self.fake_root] = global_sketches


class equiv_relation:
    """
    This class represents an equivalence relation
    that can be computed incrementally
    """
    def __init__(self, elems:Set[DerivedTypeVariable]) -> None:
        self._equiv_repr  = {elem: frozenset((elem,)) for elem in elems}

    def make_equiv(self,x:FrozenSet[DerivedTypeVariable],y:FrozenSet[DerivedTypeVariable]) -> None:
        new_set = x | y
        for elem in new_set:
            self._equiv_repr[elem] = new_set
    
    def find_equiv_rep(self,x:DerivedTypeVariable) -> Set[DerivedTypeVariable]:
        return self._equiv_repr[x]

    def get_equivalence_classes(self)-> Set[FrozenSet[DerivedTypeVariable]]:
        return set(self._equiv_repr.values())


def compute_quotient_graph(constraints:ConstraintSet) -> Tuple[equiv_relation,networkx.DiGraph]:
    """
    Compute the quotient graph corresponding to a set of 
    constraints.
    This graph allows us to infer the capabilities of all
    the DVTs that appear in the constraints.
    """
    # create initial graph
    g = networkx.DiGraph()
    for dvt in constraints.all_dvts():
        g.add_node(dvt)
        while dvt.largest_prefix:
            prefix = dvt.largest_prefix
            g.add_edge(prefix,dvt,label=dvt.tail)
            dvt = prefix
    
    # compute quotient graph
    equiv = equiv_relation(g.nodes)
    def unify(x_class:FrozenSet(DerivedTypeVariable),y_class:FrozenSet(DerivedTypeVariable)):
        if x_class != y_class:
            equiv.make_equiv(x_class, y_class)
            for (src,dest,label) in g.out_edges(x_class,data="label"):
                if label is not None:
                    for (src2,dest2,label2) in g.out_edges(y_class,data="label"):
                        if label2 == label or  (label==LoadLabel.instance()  and label2 == StoreLabel.instance()):
                            unify(equiv.find_equiv_rep(dest),equiv.find_equiv_rep(dest2))
                
    for constraint in constraints:
        unify(equiv.find_equiv_rep(constraint.left),equiv.find_equiv_rep(constraint.right))
    
    return equiv, networkx.quotient_graph(g,equiv.get_equivalence_classes(),create_using=networkx.MultiDiGraph)

def infer_shapes(scc:Set[DerivedTypeVariable],sketches:Sketches, constraints:ConstraintSet) -> Sketches:
    """
    Infer shapes takes a set of constraints and populates shapes of the sketches
    for all DVS in scc.
    """
    equiv,g_quotient = compute_quotient_graph(constraints)
    
    # I think the sketch graph is basically a projection
    # of the quotient graph as seen from the proc_var starting
    # point
    # We don't actually need to explore all paths, just make
    # sure they are all reflected in the sketch graph
    for proc_var in scc:
        # This is basically a graph per procedure variable
        correspondence = {equiv.find_equiv_rep(proc_var):proc_var}
        
        for (src, dest, edge_num) in networkx.edge_dfs(g_quotient,source=equiv.find_equiv_rep(proc_var)):      
            if dest not in correspondence:
                # TODO: Investigate cases with multiple edges with different labels
                label = g_quotient[src][dest][edge_num].get("label")
                dest_dvt = correspondence[src].add_suffix(label)
                correspondence[dest]=dest_dvt
                sketches.make_node(dest_dvt)
                sketches.sketches.add_edge(correspondence[src], dest_dvt, label=dest_dvt.tail)
            else:
                label_node = LabelNode(correspondence[dest])
                sketches.sketches.add_edge(correspondence[src], label_node, label=correspondence[dest].tail)


def instantiate_calls(cs:ConstraintSet,sketch_map:Dict[DerivedTypeVariable, Sketches], types:Lattice[DerivedTypeVariable]) -> ConstraintSet:
    """
    For every constraint involving a procedure that appears in the sketch map
    Generate constraints that summarize the contents of the sketch.
    """
    fresh_var_counter = 0

    def instantiate_sketch(proc:DerivedTypeVariable, sketch: Sketches,types:Lattice[DerivedTypeVariable]) -> ConstraintSet:
        nonlocal fresh_var_counter
        all_constraints = ConstraintSet()
        for node in sketch.sketches.nodes:
            if isinstance(node,SketchNode) and node.dtv.base_var == proc:
                constraints = []
                # if the node has some type, capture that in a constraint
                if node.lower_bound != types.bottom:
                    constraints.append(SubtypeConstraint(node.lower_bound,node.dtv))
                if node.upper_bound != types.top:
                    constraints.append(SubtypeConstraint(node.dtv,node.upper_bound))
                # if the node is a leaf, capture the capability using fake variables
                # this could be avoided if we support capability constraints  (Var x.l) in
                # addition to subtype constraints 
                if len(constraints) == 0 and len(sketch.sketches.successors(node)) == 0:
                    fresh_var = DerivedTypeVariable(f"$$fresh-var-{fresh_var_counter}$$")
                    fresh_var_counter +=1
                    # FIXME check if this should be the other way around
                    if node.dtv.path_variance == Variance.COVARIANT:
                        constraints.append(SubtypeConstraint(node.dtv,fresh_var))
                    else:
                        constraints.append(SubtypeConstraint(fresh_var,node.dtv))
                # I am not sure about this, but I think label nodes should
                # not be completely ignored
                for succ in sketch.sketches.successors(node):
                    if isinstance(succ,LabelNode):
                        label = sketch.sketches[node][succ].get("label")
                        loop_back = node.dtv.add_suffix(label)
                        if loop_back.path_variance == Variance.COVARIANT:
                            constraints.append(SubtypeConstraint(loop_back,succ.target))
                        else:
                            constraints.append(SubtypeConstraint(succ.target,loop_back))
                all_constraints |= ConstraintSet(constraints)
        return all_constraints

    callees = set()
    for constraint in cs:
        for side in [constraint.left,constraint.right]:
            if side.base_var in sketch_map:
                callees.add(side.base_var)
    
    new_constraints = ConstraintSet()
    for callee in callees:
        new_constraints |= instantiate_sketch(callee,sketch_map[callee],types)
    return new_constraints

        
        

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
                 verbose: LogLevel = LogLevel.QUIET) -> None:
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
        # As per the algorithm we only allow paths that follow the regular language
        # (recall _)* (forget _)*
        # We enforce this by looking for inversions from forget->recall.
        def matches_language(string):
            if len(string) >= 2 \
               and string[-2].kind == EdgeLabel.Kind.FORGET \
               and string[-1].kind == EdgeLabel.Kind.RECALL:
                return False
            return True
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
            if not matches_language(string):
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
                                                       node._forgotten ==
                                                       Node.Forgotten.PRE_FORGET}
        # We evenly distribute the maximum number of paths that we are willing to explore
        # across all origin nodes here.
        max_paths_per_root = int(min(self.config.max_paths_per_root,
                                     self.config.max_total_paths / float(len(start_nodes)+1)))
        for origin in start_nodes:
            npaths = 0
            explore(origin)
        return constraints


    def _solve_topo_graph(self,
                          global_handler: GlobalHandler,
                          scc_dag: networkx.DiGraph,
                          constraint_map: Dict[Any, ConstraintSet],
                          sketches_map: Dict[DerivedTypeVariable, Sketches],
                          derived: Dict[DerivedTypeVariable, ConstraintSet]):
        def show_progress(iterable):
            if self.verbose:
                return tqdm.tqdm(iterable)
            return iterable

        for scc_node in show_progress(reversed(list(networkx.topological_sort(scc_dag)))):
            global_handler.pre_scc(scc_node)
            scc = scc_dag.nodes[scc_node]['members']
            scc_graph = networkx.DiGraph()
            for proc_or_global in scc:
                constraints = constraint_map.get(proc_or_global, ConstraintSet())
                graph = ConstraintGraph(constraints)
                for head, tail in graph.graph.edges:
                    label = graph.graph[head][tail].get('label')
                    if label:
                        scc_graph.add_edge(head, tail, label=label)
                    else:
                        scc_graph.add_edge(head, tail)

            # Uncomment this out to dump the constraint graph
            self.debug("# Processing SCC: %s", "_".join([str(s) for s in scc]))
            #dump_labeled_graph(scc_graph, name, f"/tmp/scc_{name}")

            # make a copy; some of this analysis mutates the graph
            self._generate_type_vars(scc_graph)
            Solver._recall_forget_split(scc_graph)
            generated = self._generate_constraints(scc_graph)

            # The sketches for this SCC; note it may pull in data from other sketches
            scc_initial_constraints = ConstraintSet()
            for proc_or_global in scc:
                scc_initial_constraints |= constraint_map.get(proc_or_global, ConstraintSet())
            scc_complete_constraints = scc_initial_constraints | instantiate_calls(scc_initial_constraints, sketches_map, self.program.types)
            scc_sketches = Sketches(self.program.types, self.verbose)
            infer_shapes(scc, scc_sketches, scc_complete_constraints)

            scc_sketches.add_constraints(global_handler, self.program.global_vars,
                                         generated, scc, sketches_map, self)

            for proc_or_global in scc:
                sketches_map[proc_or_global] = scc_sketches
                if self.config.keep_output_constraints:
                    derived[proc_or_global] = generated

            global_handler.post_scc(scc_node, sketches_map)


    def __call__(self) -> Tuple[Dict[DerivedTypeVariable, ConstraintSet],
                                Dict[DerivedTypeVariable, Sketches]]:
        '''Perform the retypd calculation.
        '''

        derived: Dict[DerivedTypeVariable, ConstraintSet] = {}
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
            global_handler = PreciseGlobalHandler(self.program.global_vars, callgraph,
                                                  scc_dag, fake_root)
        else:
            global_handler = UnionGlobalHandler(self.program.global_vars, callgraph,
                                                scc_dag, fake_root)

        self._solve_topo_graph(global_handler, scc_dag, self.program.proc_constraints,
                               sketches_map, derived)
        global_handler.finalize(self, sketches_map)

        # Note: all globals point to the same "Sketches" graph, which has all the globals
        # in it. It would be nice to separate them out, but not a priority right now (clients
        # can do it easily).
        for g in self.program.global_vars:
            if self.config.keep_output_constraints:
                derived[g] = derived[fake_root]
            sketches_map[g] = sketches_map[fake_root]
        if self.config.keep_output_constraints and fake_root in derived:
            del derived[fake_root]
        if fake_root in sketches_map:
            del sketches_map[fake_root]

        return (derived, sketches_map)

    @staticmethod
    def _recall_forget_split(graph: networkx.DiGraph) -> None:
        '''The algorithm, after saturation, only admits paths such that recall edges all precede
        the first forget edge (if there is such an edge). To enforce this, we modify the graph by
        splitting each node and the unlabeled and forget edges (but not recall edges!). Forget edges
        in the original graph are changed to point to the 'forgotten' duplicate of their original
        target. As a result, no recall edges are reachable after traversing a single forget edge.
        '''
        edges = set(graph.edges)
        for head, tail in edges:
            label = graph[head][tail].get('label')
            if label and label.kind == EdgeLabel.Kind.RECALL:
                continue
            forget_head = head.split_recall_forget()
            forget_tail = tail.split_recall_forget()
            atts = graph[head][tail]
            if label and label.kind == EdgeLabel.Kind.FORGET:
                graph.remove_edge(head, tail)
                graph.add_edge(head, forget_tail, **atts)
            graph.add_edge(forget_head, forget_tail, **atts)

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


class Sketches(Loggable):
    '''The set of sketches from a set of constraints. Intended to be updated incrementally, per the
    Solver's reverse topological ordering.
    '''
    def __init__(self, types: Lattice[DerivedTypeVariable], verbose: LogLevel = LogLevel.QUIET) -> None:
        super(Sketches, self).__init__(verbose)
        # We maintain the invariant that if a node is in `lookup` then it should also be in
        # `sketches` as a node (even if there are no edges)
        self.sketches = networkx.DiGraph()
        self.lookup: Dict[DerivedTypeVariable, SketchNode] = {}
        self.types = types

    def ref_node(self, node: SketchNode) -> SketchNode:
        '''Add a reference to the given node (no copy)
        '''
        if isinstance(node, LabelNode):
            return node
        if node.dtv in self.lookup:
            return self.lookup[node.dtv]
        self.lookup[node.dtv] = node
        self.sketches.add_node(node)
        return node

    def make_node(self,
                  variable: DerivedTypeVariable,
                  *,
                  lower_bound: Optional[DerivedTypeVariable] = None,
                  upper_bound: Optional[DerivedTypeVariable] = None) -> SketchNode:
        '''Make a node from a DTV. Compute its atom from its access path.
        '''
        if lower_bound is None:
            lower_bound = self.types.bottom
        if upper_bound is None:
            upper_bound = self.types.top
        result = SketchNode(variable, lower_bound, upper_bound)
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


    def _add_edge(self, head: SketchNode, tail: SkNode, label: str) -> None:
        # don't emit duplicate edges
        if (head, tail) not in self.sketches.edges:
            self.sketches.add_edge(head, tail, label=label)

    def _copy_inner(self,
                    onto: SketchNode,
                    origin: SketchNode,
                    other_sketches: networkx.DiGraph,
                    dependencies: Dict[SketchNode, Set[SketchNode]],
                    seen: Dict[SketchNode, SketchNode],
                    populate_source: bool = False) -> None:
        """
        This function takes a target node (onto) that we are copying data onto. The data we are
        copying comes from a type constraint which tells us that a source node (origin) should have
        its type properties applied to onto.

        There are two kinds of information that we are collecting (recursively) from origin:
        1. Structural information. This comes in the form of SketchNode successors (children in the
           (DTV form). A typical example of this would be that a callee accesses fields in a struct
           that a caller does not, and so we copying those "accesses" as new sketch nodes.
        2. Atomic type information. When type variables are involved, we may not have a straight-
           forward path from onto -> (atomic type) in the output constraint graph. These typevars
           are specifically there to prevent looping in that graph. However, there may still be
           useful information in those typevars (like one of them indicates it is a "float", e.g.)
           We use the lattice to merge this information in the target node (onto).
        """
        # Compute a least upper bound (join) on the lower bounds for the DTV.
        # Analogously, compute a greatest upper bound (meet) on the upper bounds for
        # the DTV.
        onto.lower_bound = self.types.join(origin.lower_bound, onto.lower_bound)
        onto.upper_bound = self.types.meet(origin.upper_bound, onto.upper_bound)
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
        # Tells us where the type information came from; just auxiliary information for sketches
        if populate_source:
            onto.source.add(origin)

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
                    self.info("ERROR: Suffix is None %s --> %s", origin.dtv, succ.target)
                    continue # FIXME: this needs investigation
                succ_dtv = onto.dtv.remove_suffix(suffix)
                assert succ_dtv
                self._add_edge(onto, LabelNode(succ_dtv), label=label)
            # Otherwise, it's a regular (non-label) node that hasn't been seen, so emit it and
            # explore its successors.
            else:
                succ_node = self.make_node(onto.dtv.add_suffix(label),
                                           lower_bound=succ.lower_bound,
                                           upper_bound=succ.upper_bound)
                self._add_edge(onto, succ_node, label=label)
                seen[succ_node] = onto
                self._copy_inner(succ_node, succ, other_sketches, dependencies, seen)

    def instantiate_intra(self, dependencies: Dict[SketchNode, Set[SketchNode]]) -> None:
        self.debug("instantiate_intra: %s", dependencies)
        old_sketches = networkx.DiGraph(self.sketches)
        for dest in dependencies.keys():
            seen = {self.lookup[n]: self.lookup[n] for n in dest.dtv.all_prefixes()}
            for origin in dependencies.get(dest, set()):
                # Each iteration we take a copy of the current sketches because they will be
                # updated inside of _copy_inner.
                self._copy_inner(dest, origin, old_sketches, dependencies, seen)

    def copy_inter(self,
                   dependencies: Dict[SketchNode, Set[SketchNode]],
                   sketches_map: Dict[DerivedTypeVariable, Sketches]) -> None:
        '''Copy the structures from self.sketches corresponding to the keys in dependencies to each
        other simultaneously. Each copy is a DFS that identifies duplicates and replaces them with
        labels, so the resulting graphs are trees. All reads are from self.sketches and all writes
        go to a temporary variable, so no fixed point calculation is needed.
        '''
        self.debug("copy_inter: %s", dependencies)
        for dest in dependencies.keys():
            seen = {self.lookup[n]: self.lookup[n] for n in dest.dtv.all_prefixes()}
            for origin in dependencies.get(dest, set()):
                origin_base = origin.dtv.base_var
                # If we have funcvar <= globalvar, the global variable won't yet be in
                # the sketches_map.
                if origin_base in sketches_map:
                    self.debug("Copying %s<--%s from %s sketch", dest, origin, origin_base)
                    # If the node exists in the interprocedural sketch, use that one, otherwise
                    # just use the one we started with. It is possible to have interesting
                    # constraints on interprocedural deps without having a sketch for the dep
                    # (consider models for external functions)
                    origin_node = sketches_map[origin_base].lookup.get(origin.dtv)
                    if origin_node is not None:
                        origin = origin_node
                    self._copy_inner(dest, origin, sketches_map[origin_base].sketches,
                                     dependencies, seen, populate_source=True)

    def _copy_global_recursive(self, node: SketchNode, sketches: Sketches) -> SketchNode:
        # TODO: this probably needs to handle atoms properly, using the Lattice. Needs
        # some thought.
        our_node = self.ref_node(node)
        if node in sketches.sketches.nodes:
            for _, dst in sketches.sketches.out_edges(node):
                our_dst = dst
                if not isinstance(dst, LabelNode):
                    our_dst = self._copy_global_recursive(dst, sketches)
                self._add_edge(our_node, our_dst, sketches.sketches[node][dst]["label"])
        return our_node


    def copy_globals_from_sketch(self,
                                 global_vars: Set[DerivedTypeVariable],
                                 sketches: Sketches):
        global_roots = set()
        for dtv, node in sketches.lookup.items():
            if dtv.base_var in global_vars:
                global_roots.add(dtv.base_var)
        for g in global_roots:
            node = sketches.lookup.get(g)
            if node is None:
                continue
            self._copy_global_recursive(node, sketches)


    def add_constraints(self,
                        global_handler: GlobalHandler,
                        global_vars: Set[DerivedTypeVariable],
                        constraints: ConstraintSet,
                        nodes: Set[DerivedTypeVariable],
                        sketches_map: Dict[DerivedTypeVariable, Sketches],
                        solver: Solver
                        ) -> None:
        '''Extend the set of sketches with the new set of constraints.
        '''

        # Step 1: Apply constraints that make use of lattice types, so that we have the seeds
        # for atomic types in our sketch graph
        for constraint in constraints:
            left = solver.reverse_type_var(constraint.left)
            right = solver.reverse_type_var(constraint.right)
            # We don't want to add nodes for lattice types. This makes lots of things unpleasant.
            if left not in self.types.internal_types:
                if left.base_var in self.types.internal_types:
                    continue
            left_node = self.add_variable(left)
            if right not in self.types.internal_types:
                if right.base_var in self.types.internal_types:
                    continue
            right_node = self.add_variable(right)

            if left in self.types.internal_types:
                self.debug("JOIN: %s, %s", right_node, left)
                right_node.lower_bound = self.types.join(right_node.lower_bound, left)
                self.debug("   --> %s", right_node)
            if right in self.types.internal_types:
                self.debug("MEET: %s, %s", left_node, left)
                left_node.upper_bound = self.types.meet(left_node.upper_bound, right)
                self.debug("   --> %s", left_node)

        # Step 2: Now we've seeded our sketch graph with the interesting atomic types from this
        # SCC, we need to propagate the information. We do this two ways:
        #  1. Intraprocedurally, through typevars. Here we follow all typevar constraints as edges
        #     and copy any successors as new nodes in our sketch graph, and use the lattice to
        #     collect information from reachable atomic types as well.
        #  2. Interprocedurally, we do the same thing except we copy the information from callees
        #     and their sketches.
        inter_dependencies: Dict[SketchNode, Set[SketchNode]] = {}
        intra_dependencies: Dict[SketchNode, Set[SketchNode]] = {}
        for constraint in constraints:
            left = solver.reverse_type_var(constraint.left)
            right = solver.reverse_type_var(constraint.right)
            left_typevar = (left != constraint.left)
            right_typevar = (right != constraint.right)
            # Constraints that contain a lattice type will not have nodes, they are handled
            # in Step 1 above.
            left_node = self.lookup.get(left)
            right_node = self.lookup.get(right)
            if left_node is None or right_node is None:
                continue
            # Base variables are the type variable in question without the access path; e.g.,
            # F.in_0.load.σ4@0's base variable is F.
            left_base_var = left.base_var
            right_base_var = right.base_var
            if {left_base_var, right_base_var} & global_vars:
                self.debug("Skipping %s (global)", constraint)
                continue
            # If the right-hand side is not a type variable, it must be an interesting node (a
            # function or global). The left and right base nodes may be different or the same;
            # either way, treat it as an interprocedural dependency
            elif right_base_var not in nodes and not right_typevar:
                self.debug("Copying right %s (inter-procedural)", constraint)
                inter_dependencies.setdefault(left_node, set()).add(right_node)
            elif left_base_var not in nodes and not left_typevar:
                self.debug("Copying left %s (inter-procedural)", constraint)
                inter_dependencies.setdefault(right_node, set()).add(left_node)
            # Type variables are generated only to break cycles in the constraint graph. If the rhs
            # is a type variable, that means that the relationship between the sketches is
            # intraprocedural and not interprocedural.
            elif left_typevar or right_typevar:
                self.debug("Copying %s (intra-procedural)", constraint)
                intra_dependencies.setdefault(left_node, set()).add(right_node)
        # Intra-SCC dependencies just get instantiated using "this" set of sketches
        if intra_dependencies:
            self.instantiate_intra(intra_dependencies)
        # Inter-SCC dependencies make use of the sketches_map, which maps each base DTV to its
        # sketch graph. This way we don't need to copy and iterate over a huge DiGraph for the
        # entire program when we only need the direct callees (or referents, for globals)
        if inter_dependencies:
            self.copy_inter(inter_dependencies, sketches_map)

        # Copy globals from our callees, if we are analyzing globals precisely.
        global_handler.copy_globals(self, nodes, sketches_map)

        
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
                    graph_str += f'"{node}" [label="{node.lower_bound}..{node.upper_bound}"];'
            elif node.target.base_var == dtv:
                graph_str += nt
                graph_str += f'"{node}" [label="{node.target}", shape=house];'
        graph_str += edges_str
        graph_str += f'{os.linesep}}}'
        return graph_str

    def __str__(self) -> str:
        if self.lookup:
            nt = f'{os.linesep}\t'
            def format(k: DerivedTypeVariable) -> str:
                return str(self.lookup[k])
            return f'nodes:{nt}{nt.join(map(format, self.lookup.keys()))})'
        return 'no sketches'
