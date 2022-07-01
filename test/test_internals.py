"""Simple unit tests from the paper and slides that rely on looking under-the-hood
at generated constraints.
"""

import unittest

from retypd import (
    CLattice,
    ConstraintSet,
    DerivedTypeVariable,
    DummyLattice,
    Program,
    SchemaParser,
    Solver,
    DerefLabel,
    CType,
    CLatticeCTypes,
    BoolType,
    CharType,
    FloatType,
    IntType,
    VoidType,
)


class BasicSchemaTest(unittest.TestCase):
    def test_parse_label(self):
        l = SchemaParser.parse_label("σ8@0")
        self.assertEqual((l.size, l.offset, l.count), (8, 0, 1))
        l = SchemaParser.parse_label("σ0@10000")
        self.assertEqual((l.size, l.offset, l.count), (0, 10000, 1))
        l = SchemaParser.parse_label("σ4@-32")
        self.assertEqual((l.size, l.offset, l.count), (4, -32, 1))
        l = SchemaParser.parse_label("σ2@32*[1000]")
        self.assertEqual((l.size, l.offset, l.count), (2, 32, 1000))
        l = SchemaParser.parse_label("σ2@32*[nobound]")
        self.assertEqual(
            (l.size, l.offset, l.count), (2, 32, DerefLabel.COUNT_NOBOUND)
        )
        l = SchemaParser.parse_label("σ2@32*[nullterm]")
        self.assertEqual(
            (l.size, l.offset, l.count), (2, 32, DerefLabel.COUNT_NULLTERM)
        )

        with self.assertRaises(ValueError):
            l = SchemaParser.parse_label("σ-9@100")

    def test_simple_constraints(self):
        """A simple test from the paper (the right side of Figure 4 on p. 6). This one has no
        recursive data structures; as such, the fixed point would suffice. However, we compute type
        constraints in the same way as in the presence of recursion.
        """

        constraints = ConstraintSet()
        constraints.add(SchemaParser.parse_constraint("p ⊑ q"))
        constraints.add(SchemaParser.parse_constraint("x ⊑ q.store.σ4@0"))
        constraints.add(SchemaParser.parse_constraint("p.load.σ4@0 ⊑ y"))
        f = SchemaParser.parse_variable("f")
        x = SchemaParser.parse_variable("x")
        y = SchemaParser.parse_variable("y")

        program = Program(DummyLattice(), {x, y}, {f: constraints}, {f: {}})
        solver = Solver(program)
        (gen_const, sketches) = solver()

        self.assertTrue(SchemaParser.parse_constraint("x ⊑ y") in gen_const[f])

    def test_other_simple_constraints(self):
        """Another simple test from the paper (the program modeled in Figure 14 on p. 26)."""

        constraints = ConstraintSet()
        constraints.add(SchemaParser.parse_constraint("y <= p"))
        constraints.add(SchemaParser.parse_constraint("p <= x"))
        constraints.add(SchemaParser.parse_constraint("A <= x.store"))
        constraints.add(SchemaParser.parse_constraint("y.load <= B"))
        f = SchemaParser.parse_variable("f")
        A = SchemaParser.parse_variable("A")
        B = SchemaParser.parse_variable("B")

        program = Program(DummyLattice(), {A, B}, {f: constraints}, {f: {}})
        solver = Solver(program)
        (gen_const, sketches) = solver()

        self.assertTrue(SchemaParser.parse_constraint("A ⊑ B") in gen_const[f])


class ForgetsTest(unittest.TestCase):
    def test_forgets(self):
        """A simple test to check that paths that include "forgotten" labels reconstruct access
        paths in the correct order.
        """
        F = SchemaParser.parse_variable("F")
        l = SchemaParser.parse_variable("l")

        constraints = {F: ConstraintSet()}
        constraint = SchemaParser.parse_constraint("l ⊑ F.in_1.load.σ8@0")
        constraints[F].add(constraint)

        program = Program(DummyLattice(), {l}, constraints, {F: {}})
        solver = Solver(program)
        (gen_const, sketches) = solver()

        self.assertTrue(constraint in gen_const[F])


class CLatticesTest(unittest.TestCase):
    def assertJoin(self, lhs: str, rhs: str, equals: str):
        """Validate that when two DTVs are joined, they equal a known value"""
        lhs_dtv = DerivedTypeVariable(lhs)
        rhs_dtv = DerivedTypeVariable(rhs)
        equal_dtv = DerivedTypeVariable(equals)
        self.assertEqual(CLattice().join(lhs_dtv, rhs_dtv), equal_dtv)

    def test_join(self):
        """Test C-lattice join operations against known values"""
        self.assertJoin("float", "uint", "┬")
        self.assertJoin("uint", "int", "uint")
        self.assertJoin("char", "int", "uint")
        self.assertJoin("int64", "int", "int")
        self.assertJoin("uint32", "uint64", "uint")

    def assertCType(self, name: str, ctype: CType, size: int = None):
        """Validate that an atomic DTV is translated to a known high-level
        CType
        """
        atom = DerivedTypeVariable(name)
        lattice = CLatticeCTypes()
        ctype_lhs = lattice.atom_to_ctype(atom, CLattice._top, size)
        ctype_rhs = lattice.atom_to_ctype(CLattice._bottom, atom, size)

        self.assertEqual(str(ctype), str(ctype_lhs))
        self.assertEqual(str(ctype), str(ctype_rhs))

    def test_atom_to_ctype(self):
        """Test C-lattice are converted to C-types correctly"""
        self.assertCType("int", IntType(4, True), 4)
        self.assertCType("int8", IntType(1, True))
        self.assertCType("int16", IntType(2, True))
        self.assertCType("int32", IntType(4, True))
        self.assertCType("int64", IntType(8, True))

        self.assertCType("uint", IntType(4, False), 4)
        self.assertCType("uint8", IntType(1, False))
        self.assertCType("uint16", IntType(2, False))
        self.assertCType("uint32", IntType(4, False))
        self.assertCType("uint64", IntType(8, False))

        self.assertCType("void", VoidType())
        self.assertCType("char", CharType(1), 1)
        self.assertCType("bool", BoolType(1), 1)
        self.assertCType("float", FloatType(4))
        self.assertCType("double", FloatType(8))


if __name__ == "__main__":
    unittest.main()
