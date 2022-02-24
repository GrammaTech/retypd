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
    CTypeGenerator,
    DummyLatticeCTypes,
)
from retypd.c_types import (
    PointerType,
    StructType,
    IntType,
    CharType,
    ArrayType,
)


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


class CTypeTest(unittest.TestCase):
    def test_simple_struct(self):
        """
        Verify that we will combine fields inferred from different callees.

            |-> F2  (part of struct fields)
         F1 -
            |-> F3  (other part of struct fields)
        """
        F1 = SchemaParser.parse_variable('F1')
        F2 = SchemaParser.parse_variable('F2')
        F3 = SchemaParser.parse_variable('F3')

        constraints = {F1: ConstraintSet(), F2: ConstraintSet(), F3: ConstraintSet()}
        constraints[F1].add(SchemaParser.parse_constraint("F1.in_0 ⊑ A"))
        constraints[F1].add(SchemaParser.parse_constraint("A ⊑ F2.in_1"))
        constraints[F1].add(SchemaParser.parse_constraint("A ⊑ F3.in_2"))
        # F2 accesses fields at offsets 0, 12
        constraints[F2].add(SchemaParser.parse_constraint("F2.in_1 ⊑ B"))
        constraints[F2].add(SchemaParser.parse_constraint("B.load.σ8@0 ⊑ int"))
        constraints[F2].add(SchemaParser.parse_constraint("B.load.σ4@12 ⊑ int"))
        # F3 accesses fields at offsets 8, 20
        constraints[F3].add(SchemaParser.parse_constraint("F3.in_2 ⊑ C"))
        constraints[F3].add(SchemaParser.parse_constraint("C.load.σ2@8 ⊑ int"))
        constraints[F3].add(SchemaParser.parse_constraint("C.load.σ8@20 ⊑ int"))

        lattice = DummyLattice()
        lattice_ctypes = DummyLatticeCTypes()
        program = Program(lattice, {}, constraints, {F1: [F2, F3]})
        solver = Solver(program, verbose=VERBOSE_TESTS)
        (gen_const, sketches) = solver()
        #print(sketches[F1])

        gen = CTypeGenerator(sketches, lattice, lattice_ctypes, 8, 8)
        dtv2type = gen()
        pstruct = dtv2type[F1].params[0]
        self.assertTrue(isinstance(pstruct, PointerType))
        struct = pstruct.target_type
        self.assertTrue(isinstance(struct, StructType))
        for f in struct.fields:
            self.assertEqual(type(f.ctype), IntType)
            expected_size = {0: 8, 12: 4, 8: 2, 20: 8}.get(f.offset, None)
            self.assertFalse(expected_size is None)
            self.assertEqual(f.size, expected_size)
        #print(list(map(lambda x: type(x.ctype), struct.fields)))

    def test_string_in_struct(self):
        """
        Model that strcpy() is called with a field from a struct as the destination, which
        _should_ tell us that the given field is a string.
        """
        F1 = SchemaParser.parse_variable('F1')
        strcpy = SchemaParser.parse_variable('strcpy')

        constraints = {F1: ConstraintSet(), strcpy: ConstraintSet()}
        constraints[F1].add(SchemaParser.parse_constraint("F1.in_0 ⊑ A"))
        constraints[F1].add(SchemaParser.parse_constraint("A.load.σ8@8 ⊑ strcpy.in_1"))
        constraints[strcpy].add(SchemaParser.parse_constraint("strcpy.in_1.load.σ1@0*[nullterm] ⊑ int"))

        lattice = DummyLattice()
        lattice_ctypes = DummyLatticeCTypes()
        program = Program(lattice, {}, constraints, {F1: [strcpy]})
        solver = Solver(program, verbose=VERBOSE_TESTS)
        (gen_const, sketches) = solver()
        #print(sketches[F1])

        gen = CTypeGenerator(sketches, lattice, lattice_ctypes, 8, 8)
        dtv2type = gen()
        pstruct = dtv2type[F1].params[0]
        self.assertTrue(isinstance(pstruct, PointerType))
        struct = pstruct.target_type
        self.assertTrue(isinstance(struct, StructType))
        for f in struct.fields:
            self.assertEqual(type(f.ctype), PointerType)
            self.assertEqual(type(f.ctype.target_type), CharType)
            self.assertEqual(f.offset, 8)

    def test_global_array(self):
        """
        Illustration of how a model of memcpy might work.
        """
        F1 = SchemaParser.parse_variable('F1')
        memcpy = SchemaParser.parse_variable('memcpy')
        some_global = SchemaParser.parse_variable('some_global')

        constraints = {F1: ConstraintSet()}
        constraints[F1].add(SchemaParser.parse_constraint("some_global ⊑ memcpy.in_1"))
        constraints[F1].add(SchemaParser.parse_constraint("memcpy.in_1.load.σ4@0*[10] ⊑ int"))

        lattice = DummyLattice()
        lattice_ctypes = DummyLatticeCTypes()
        program = Program(lattice, {some_global}, constraints, {F1: [memcpy]})
        solver = Solver(program, verbose=VERBOSE_TESTS)
        (gen_const, sketches) = solver()
        #print(sketches)

        gen = CTypeGenerator(sketches, lattice, lattice_ctypes, 8, 8)
        dtv2type = gen()
        t = dtv2type[some_global]
        self.assertEqual(type(t), PointerType)
        self.assertEqual(type(t.target_type), ArrayType)
        self.assertEqual(t.target_type.length, 10)
        self.assertEqual(t.target_type.member_type.size, 4)

    def test_load_v_store(self):
        """
        Half of struct fields are only loaded and half are only stored, should still result
        in all fields being properly inferred.
        """
        F1 = SchemaParser.parse_variable('F1')
        some_global = SchemaParser.parse_variable('some_global')

        constraints = {F1: ConstraintSet()}
        constraints[F1].add(SchemaParser.parse_constraint("some_global ⊑ A"))
        constraints[F1].add(SchemaParser.parse_constraint("A.load.σ4@0 ⊑ int"))
        constraints[F1].add(SchemaParser.parse_constraint("int ⊑ A.store.σ4@4"))
        constraints[F1].add(SchemaParser.parse_constraint("A.load.σ4@8 ⊑ int"))
        constraints[F1].add(SchemaParser.parse_constraint("int ⊑ A.store.σ4@12"))
        constraints[F1].add(SchemaParser.parse_constraint("A.load.σ4@16 ⊑ int"))
        constraints[F1].add(SchemaParser.parse_constraint("int ⊑ A.store.σ4@20"))

        lattice = DummyLattice()
        lattice_ctypes = DummyLatticeCTypes()
        program = Program(lattice, {some_global}, constraints, {F1: []})
        solver = Solver(program, verbose=VERBOSE_TESTS)
        (gen_const, sketches) = solver()
        #print(sketches[some_global])

        gen = CTypeGenerator(sketches, lattice, lattice_ctypes, 8, 8)
        dtv2type = gen()
        t = dtv2type[some_global]
        self.assertEqual(type(t), PointerType)
        self.assertEqual(type(t.target_type), StructType)
        self.assertEqual(len(t.target_type.fields), 6)
        self.assertEqual(max([f.offset for f in t.target_type.fields]), 20)
        self.assertEqual(min([f.offset for f in t.target_type.fields]), 0)



if __name__ == '__main__':
    unittest.main()
