'''Simple unit tests from the paper and slides that rely on looking under-the-hood
at generated constraints.
'''

from abc import ABC
import networkx
import unittest

from retypd import ConstraintSet, DummyLattice, Program, SchemaParser, Solver


class BasicSchemaTest(unittest.TestCase):

    def test_simple_constraints(self):
        '''A simple test from the paper (the right side of Figure 4 on p. 6). This one has no
        recursive data structures; as such, the fixed point would suffice. However, we compute type
        constraints in the same way as in the presence of recursion.
        '''

        constraints = ConstraintSet()
        constraints.add(SchemaParser.parse_constraint('p ⊑ q'))
        constraints.add(SchemaParser.parse_constraint('x ⊑ q.store.σ4@0'))
        constraints.add(SchemaParser.parse_constraint('p.load.σ4@0 ⊑ y'))
        f = SchemaParser.parse_variable('f')
        x = SchemaParser.parse_variable('x')
        y = SchemaParser.parse_variable('y')

        program = Program(DummyLattice(), {x, y}, {f: constraints}, {f: {}})
        solver = Solver(program)
        (gen_const, sketches) = solver()

        self.assertTrue(SchemaParser.parse_constraint('x ⊑ y') in gen_const[f])

    def test_other_simple_constraints(self):
        '''Another simple test from the paper (the program modeled in Figure 14 on p. 26).
        '''

        constraints = ConstraintSet()
        constraints.add(SchemaParser.parse_constraint('y <= p'))
        constraints.add(SchemaParser.parse_constraint('p <= x'))
        constraints.add(SchemaParser.parse_constraint('A <= x.store'))
        constraints.add(SchemaParser.parse_constraint('y.load <= B'))
        f = SchemaParser.parse_variable('f')
        A = SchemaParser.parse_variable('A')
        B = SchemaParser.parse_variable('B')

        program = Program(DummyLattice(), {A, B}, {f: constraints}, {f: {}})
        solver = Solver(program)
        (gen_const, sketches) = solver()

        self.assertTrue(SchemaParser.parse_constraint('A ⊑ B') in gen_const[f])


class ForgetsTest(unittest.TestCase):
    def test_forgets(self):
        '''A simple test to check that paths that include "forgotten" labels reconstruct access
        paths in the correct order.
        '''
        F = SchemaParser.parse_variable('F')
        l = SchemaParser.parse_variable('l')

        constraints = {F: ConstraintSet()}
        constraint = SchemaParser.parse_constraint("l ⊑ F.in_1.load.σ8@0")
        constraints[F].add(constraint)

        program = Program(DummyLattice(), {l}, constraints, {F: {}})
        solver = Solver(program)
        (gen_const, sketches) = solver()

        self.assertTrue(constraint in gen_const[F])

if __name__ == '__main__':
    unittest.main()