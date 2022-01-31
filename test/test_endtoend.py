'''Simple unit tests from the paper and slides that only look at the final result (sketches)
'''

from abc import ABC
import networkx
import unittest

from retypd import ConstraintSet, DummyLattice, Program, SchemaParser, Solver


VERBOSE_TESTS = False


class RecursiveSchemaTest(unittest.TestCase):
    def test_recursive(self):
        '''A test based on the running example from the paper (Figure 2 on p. 3) and the slides
        (slides 67-83, labeled as slides 13-15).
        '''
        F = SchemaParser.parse_variable('F')
        close = SchemaParser.parse_variable('close')

        constraints = {F: ConstraintSet(), close: ConstraintSet()}
        constraints[F].add(SchemaParser.parse_constraint("F.in_0 ⊑ δ"))
        constraints[F].add(SchemaParser.parse_constraint("α ⊑ φ"))
        constraints[F].add(SchemaParser.parse_constraint("δ ⊑ φ"))
        constraints[F].add(SchemaParser.parse_constraint("φ.load.σ4@0 ⊑ α"))
        constraints[F].add(SchemaParser.parse_constraint("φ.load.σ4@4 ⊑ α'"))
        constraints[F].add(SchemaParser.parse_constraint("α' ⊑ close.in_0"))
        constraints[F].add(SchemaParser.parse_constraint("close.out ⊑ F.out"))

        constraints[close].add(SchemaParser.parse_constraint("close.in_0 ⊑ #FileDescriptor"))
        constraints[close].add(SchemaParser.parse_constraint("#SuccessZ ⊑ close.out"))

        program = Program(DummyLattice(), {}, constraints, {F: [close]})
        solver = Solver(program, verbose=VERBOSE_TESTS)
        (gen_const, sketches) = solver()

        # Inter-procedural results (sketches)
        fd = SchemaParser.parse_variable("#FileDescriptor")
        F_sketches = sketches[F]
        # Equivalent to "#SuccessZ ⊑ F.out"
        self.assertEqual(F_sketches.lookup[SchemaParser.parse_variable("F.out")].lower_bound,
                         SchemaParser.parse_variable("#SuccessZ"))
        self.assertEqual(F_sketches.lookup[SchemaParser.parse_variable(f"φ.load.σ4@4")].upper_bound,
                         SchemaParser.parse_variable("#FileDescriptor"))



if __name__ == '__main__':
    unittest.main()
