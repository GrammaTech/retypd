'''Simple unit tests from the paper and slides that only look at the final result (sketches)
'''

from abc import ABC
import networkx
import unittest

from retypd import ConstraintSet, DummyLattice, Program, SchemaParser, Solver


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
        solver = Solver(program)
        (gen_const, sketches) = solver()

        tv = solver.lookup_type_var('φ')
        self.assertTrue(SchemaParser.parse_constraint('#SuccessZ ⊑ F.out') in gen_const[F])
        self.assertTrue(SchemaParser.parse_constraint(f'F.in_0 ⊑ {tv}') in gen_const[F])
        self.assertTrue(SchemaParser.parse_constraint(f'{tv}.load.σ4@0 ⊑ {tv}') in gen_const[F])
        self.assertTrue(SchemaParser.parse_constraint(f'{tv}.load.σ4@4 ⊑ #FileDescriptor') in
                gen_const[F])


if __name__ == '__main__':
    unittest.main()
