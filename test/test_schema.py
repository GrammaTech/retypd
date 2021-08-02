'''Simple unit tests from the paper and slides.
'''

from abc import ABC
import networkx
import unittest

from retypd import ConstraintSet, DummyLattice, Program, SchemaParser, Solver

class SchemaTest(ABC):
    def graphs_are_equal(self, graph, edge_set) -> None:
        edges = graph.edges()
        self.assertEqual(len(edges), len(edge_set))
        for edge in edges:
            (head, tail) = edge
            self.assertTrue(edge in edge_set)
            self.assertEqual(graph[head][tail], edge_set[edge])

    @staticmethod
    def edges_to_dict(edges):
        '''Convert a collection of edge strings into a dict. Used for comparing a graph against an
        expected value.
        '''
        graph = {}
        for edge in edges:
            (head, tail, atts) = SchemaParser.parse_edge(edge)
            graph[(head, tail)] = atts
        return graph


class BasicSchemaTest(SchemaTest, unittest.TestCase):

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
        generated = solver()

        self.assertTrue(SchemaParser.parse_constraint('x ⊑ y') in generated[f])

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
        generated = solver()

        self.assertTrue(SchemaParser.parse_constraint('A ⊑ B') in generated[f])


class RecursiveSchemaTest(SchemaTest, unittest.TestCase):
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
        # How to make lattice elements into DTVs?
        constraints[close].add(SchemaParser.parse_constraint("close.in_0 ⊑ #FileDescriptor"))
        constraints[close].add(SchemaParser.parse_constraint("#SuccessZ ⊑ close.out"))

        program = Program(DummyLattice(), {F}, constraints, {F: {close}})
        solver = Solver(program)
        generated = solver()

        tv = solver.lookup_type_var('φ')
        self.assertTrue(SchemaParser.parse_constraint('#SuccessZ ⊑ F.out') in generated[F])
        self.assertTrue(SchemaParser.parse_constraint(f'F.in_0 ⊑ {tv}') in generated[F])
        self.assertTrue(SchemaParser.parse_constraint(f'{tv}.load.σ4@0 ⊑ {tv}') in generated[F])
        self.assertTrue(SchemaParser.parse_constraint(f'{tv}.load.σ4@4 ⊑ #FileDescriptor') in
                generated[F])

if __name__ == '__main__':
    unittest.main()
