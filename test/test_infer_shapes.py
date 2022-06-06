'''Simple unit tests from the paper and slides that only look at the final result (sketches)
'''

from abc import ABC

import networkx
import unittest


from retypd import (
    ConstraintSet,
    DummyLattice,
    Program,
    SchemaParser,
    Solver,
    DerivedTypeVariable
)
from typing import Dict,List


VERBOSE_TESTS = False


def compute_sketches(cs:Dict[str,List[str]], callgraph:Dict[str,List[str]]):

    parsed_cs = {}
    for proc,proc_cs in cs.items():

        proc_parsed_cs = ConstraintSet()
        for c in proc_cs:
            proc_parsed_cs.add(SchemaParser.parse_constraint(c))
        parsed_cs[DerivedTypeVariable(proc)] = proc_parsed_cs

    parsed_callgraph = {DerivedTypeVariable(proc): [DerivedTypeVariable(callee) for callee in callees] for proc,callees in callgraph.items()}
    lattice = DummyLattice()
    program = Program(lattice, {}, parsed_cs, parsed_callgraph)
    solver = Solver(program, verbose=VERBOSE_TESTS)
    return solver()


class InferTypesTest(unittest.TestCase):

    def test_input_arg_capability(self):
        """
        f.in_0 should get the load.σ1@0 capability even if
        we don't know anything about its type.
        The same with g.out
        """
        constraints = {
            # T-inheritR
            "f":["f.in_0 <= A","A.load.σ1@0 <= B","B.load.σ4@4 <= C"],
            # T-inheritL
            "g":["A <= g.out","A.load.σ1@0 <= B","B.load.σ4@4 <= C"]
        }
        callgraph = {"f":[],"g":[]}
        (gen_cs, sketches) = compute_sketches(constraints,callgraph)

        f_sketch = sketches[DerivedTypeVariable("f")]
        assert SchemaParser.parse_variable("f.in_0") in f_sketch.lookup
        assert SchemaParser.parse_variable("f.in_0.load.σ1@0.load.σ4@4") in f_sketch.lookup

        f_sketch = sketches[DerivedTypeVariable("g")]
        assert SchemaParser.parse_variable("g.out") in f_sketch.lookup
        assert SchemaParser.parse_variable("g.out.load.σ1@0") in f_sketch.lookup
        assert SchemaParser.parse_variable("g.out.load.σ1@0.load.σ4@4") in f_sketch.lookup

    def test_input_arg_capability_transitive(self):
        """
        g.in_0 should get the load.σ1@0 capability from f.
        """
        constraints = {
            "f":["f.in_0 <= A","A.load.σ1@0 <= B","B.load.σ4@4 <= C"],
            "g":["g.in_0 <= C", "C <= f.in_0", "C <= g.out" ]
        }
        callgraph = {"g":["f"]}
        (gen_cs, sketches) = compute_sketches(constraints,callgraph)

        f_sketch = sketches[DerivedTypeVariable("g")]
        assert SchemaParser.parse_variable("g.in_0") in f_sketch.lookup
        assert SchemaParser.parse_variable("g.in_0.load.σ1@0.load.σ4@4") in f_sketch.lookup
        assert SchemaParser.parse_variable("g.out.load.σ1@0.load.σ4@4") in f_sketch.lookup
    

           