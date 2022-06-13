from __future__ import annotations
from typing import Dict, List, Set, Tuple, Any
from .schema import DerivedTypeVariable, Lattice
from abc import ABC, abstractmethod
from .sketches import Sketches

import networkx


class GlobalHandler(ABC):
    def __init__(
        self,
        global_vars: Set[DerivedTypeVariable],
        callgraph: networkx.DiGraph,
        scc_dag: networkx.DiGraph,
        fake_root: DerivedTypeVariable,
    ) -> None:
        self.scc_dag = scc_dag
        self.callgraph = callgraph
        self.global_vars = global_vars
        self.fake_root = fake_root

    @abstractmethod
    def pre_scc(self, scc_node: Any):
        raise NotImplementedError("Child class must implement")

    @abstractmethod
    def post_scc(
        self, scc_node: Any, sketches_map: Dict[DerivedTypeVariable, Sketches]
    ) -> None:
        raise NotImplementedError("Child class must implement")

    @abstractmethod
    def copy_globals(
        self,
        current_sketch: Sketches,
        nodes: Set[DerivedTypeVariable],
        sketches_map: Dict[DerivedTypeVariable, Sketches],
    ) -> None:
        raise NotImplementedError("Child class must implement")

    @abstractmethod
    def finalize(
        self, types: Lattice, sketches_map: Dict[DerivedTypeVariable, Sketches]
    ) -> None:
        raise NotImplementedError("Child class must implement")


class PreciseGlobalHandler(GlobalHandler):
    """
    Handle globals precisely, by bubbling sketches up the callgraph, accumulating more and more
    globals (and sketch nodes) as it goes. Once it reaches the roots of the callgraph, we have
    all the globals and these are the final sketches.
    """

    def __init__(
        self,
        global_vars: Set[DerivedTypeVariable],
        callgraph: networkx.DiGraph,
        scc_dag: networkx.DiGraph,
        fake_root: DerivedTypeVariable,
    ) -> None:
        super(PreciseGlobalHandler, self).__init__(
            global_vars, callgraph, scc_dag, fake_root
        )
        # SCCs that have not yet had their sketches cleansed of globals
        # TODO get proper type annotations
        self.not_cleaned_up: List[Tuple[Any, Set[Any]]] = []

    def cleanse_globals(self, sketches: Sketches):
        """
        Delete all the global nodes from a particular sketch graph.
        """
        for dtv, node in list(sketches._lookup.items()):
            if dtv.base_var in self.global_vars:
                if node in sketches.sketches.nodes:
                    sketches.sketches.remove_node(node)
                del sketches._lookup[dtv]

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
            self.not_cleaned_up.append((scc, caller_scc_set))

    def post_scc(
        self, scc_node: Any, sketches_map: Dict[DerivedTypeVariable, Sketches]
    ) -> None:
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
                not_cleaned_up_new.append((other_scc, other_scc_callers))
        self.not_cleaned_up = not_cleaned_up_new

    def copy_globals(
        self,
        current_sketch: Sketches,
        nodes: Set[DerivedTypeVariable],
        sketches_map: Dict[DerivedTypeVariable, Sketches],
    ) -> None:
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

    def finalize(
        self, types: Lattice, sketches_map: Dict[DerivedTypeVariable, Sketches]
    ) -> None:
        pass


class UnionGlobalHandler(GlobalHandler):
    """
    Ignore globals (but leave them in sketches) during the call-graph pass, and at the end
    collect all the global sketches from all the functions.
    """

    def pre_scc(self, scc_node: Any) -> None:
        pass

    def post_scc(
        self, scc_node: Any, sketches_map: Dict[DerivedTypeVariable, Sketches]
    ) -> None:
        pass

    def copy_globals(
        self,
        current_sketch: Sketches,
        nodes: Set[DerivedTypeVariable],
        sketches_map: Dict[DerivedTypeVariable, Sketches],
    ) -> None:
        pass

    def finalize(
        self, types: Lattice, sketches_map: Dict[DerivedTypeVariable, Sketches]
    ) -> None:
        global_sketches = Sketches(types)
        for func_sketches in sketches_map.values():
            global_sketches.copy_globals_from_sketch(
                self.global_vars, func_sketches
            )
        sketches_map[self.fake_root] = global_sketches
