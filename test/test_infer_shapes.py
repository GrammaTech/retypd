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
        constraints = {"f":["f.in_0 ⊑ A","A.load.σ1@0 ⊑ B"]}
        callgraph = {"f":[]}
        (gen_cs, sketches) = compute_sketches(constraints,callgraph)

        f_sketch = sketches[DerivedTypeVariable("f")]
        assert SchemaParser.parse_variable("f.in_0") in f_sketch.lookup


           