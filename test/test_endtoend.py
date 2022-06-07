"""Simple unit tests from the paper and slides that only look at the final result (sketches)
"""

import unittest
from typing import Dict, List

from retypd import (
    ConstraintSet,
    CLattice,
    CLatticeCTypes,
    DummyLattice,
    Program,
    SchemaParser,
    Solver,
    CTypeGenerator,
    DummyLatticeCTypes,
    LogLevel,
    DerivedTypeVariable,
    Lattice,
)
from retypd.c_types import (
    FloatType,
    PointerType,
    StructType,
    IntType,
    CharType,
    ArrayType,
    FunctionType,
)


VERBOSE_TESTS = False


def compute_sketches(
    cs: Dict[str, List[str]],
    callgraph: Dict[str, List[str]],
    lattice: Lattice = DummyLattice(),
):
    """
    Auxiliary function that parses constraints, callgraph
    and solves the sketches using a default lattice.
    """
    parsed_cs = {}
    for proc, proc_cs in cs.items():

        proc_parsed_cs = ConstraintSet()
        for c in proc_cs:
            proc_parsed_cs.add(SchemaParser.parse_constraint(c))
        parsed_cs[DerivedTypeVariable(proc)] = proc_parsed_cs

    parsed_callgraph = {
        DerivedTypeVariable(proc): [
            DerivedTypeVariable(callee) for callee in callees
        ]
        for proc, callees in callgraph.items()
    }
    program = Program(lattice, {}, parsed_cs, parsed_callgraph)
    solver = Solver(program, verbose=VERBOSE_TESTS)
    return solver()


class RecursiveSchemaTest(unittest.TestCase):
    def test_recursive(self):
        """A test based on the running example from the paper (Figure 2 on p. 3) and the slides
        (slides 67-83, labeled as slides 13-15).
        """
        F = SchemaParser.parse_variable("F")
        close = SchemaParser.parse_variable("close")

        constraints = {F: ConstraintSet(), close: ConstraintSet()}
        constraints[F].add(SchemaParser.parse_constraint("F.in_0 ⊑ δ"))
        constraints[F].add(SchemaParser.parse_constraint("α ⊑ φ"))
        constraints[F].add(SchemaParser.parse_constraint("δ ⊑ φ"))
        constraints[F].add(SchemaParser.parse_constraint("φ.load.σ4@0 ⊑ α"))
        constraints[F].add(SchemaParser.parse_constraint("φ.load.σ4@4 ⊑ α'"))
        constraints[F].add(SchemaParser.parse_constraint("α' ⊑ close.in_0"))
        constraints[F].add(SchemaParser.parse_constraint("close.out ⊑ F.out"))

        constraints[close].add(
            SchemaParser.parse_constraint("close.in_0 ⊑ #FileDescriptor")
        )
        constraints[close].add(
            SchemaParser.parse_constraint("#SuccessZ ⊑ close.out")
        )

        program = Program(DummyLattice(), {}, constraints, {F: [close]})
        solver = Solver(program, verbose=VERBOSE_TESTS)
        (gen_const, sketches) = solver()

        # Inter-procedural results (sketches)
        F_sketches = sketches[F]
        # Equivalent to "#SuccessZ ⊑ F.out"
        self.assertEqual(
            F_sketches.lookup[
                SchemaParser.parse_variable("F.out")
            ].lower_bound,
            SchemaParser.parse_variable("#SuccessZ"),
        )
        self.assertEqual(
            F_sketches.lookup[
                SchemaParser.parse_variable("F.in_0.load.σ4@4")
            ].upper_bound,
            SchemaParser.parse_variable("#FileDescriptor"),
        )

    def test_recursive_through_procedures(self):
        """The type of f.in_0 is recursive.

        struct list{
            list* next;
            int elem;
        }

        The type of g.in_0 is the same and also recursive.
        This tests that the instantiation of recursive sketches works as expected.
        """
        constraints = {
            "f": [
                "f.in_0 <= list",
                "list.load.σ4@0 <= next",
                "next <= list",
                "list.load.σ4@4 <= elem",
                "elem <= int",
            ],
            "g": ["g.in_0 <= C", "C <= f.in_0"],
        }
        callgraph = {"g": ["f"]}
        lattice = CLattice()
        (gen_cs, sketches) = compute_sketches(
            constraints, callgraph, lattice=lattice
        )

        g_sketch = sketches[DerivedTypeVariable("g")]
        assert (
            SchemaParser.parse_variable("g.in_0.load.σ4@4") in g_sketch.lookup
        )
        assert (
            g_sketch.lookup[
                SchemaParser.parse_variable("g.in_0.load.σ4@4")
            ].upper_bound
            == DummyLattice._int
        )
        gen = CTypeGenerator(sketches, lattice, CLatticeCTypes(), 4, 4)
        dtv2type = gen()
        rec_struct_ptr = dtv2type[DerivedTypeVariable("g")].params[0]
        rec_struct = rec_struct_ptr.target_type
        self.assertEqual(len(rec_struct.fields), 2)
        # the field 0 is a pointer to the same struct
        self.assertEqual(
            rec_struct.fields[0].ctype.target_type.name, rec_struct.name
        )
        self.assertIsInstance(rec_struct.fields[1].ctype, IntType)

    def test_interleaving_elements(self):
        """
        There are two mutually recursive types

        struct A{
            B* nextB;
            int elemA;
        }
        struct B{
            A* nextA;
            float elemB;
        }

        The type of f.in_0 and  g.in_0 are of type A
        Tye type of f.in_1 is B
        """
        constraints = {
            "f": [
                "f.in_0 <= A",
                "A.load.σ4@0 <= nextB",
                "nextB <= B",
                "A.load.σ4@4 <= elemA",
                "elemA <= int",
                "B.load.σ4@0 <= nextA",
                "nextA <= A",
                "B.load.σ4@4 <= elemB",
                "elemB <= float",
                "f.in_1 <= B",
            ],
            "g": ["g.in_0 <= C", "C <= f.in_0", "g.in_1 <= D", "D <= f.in_1"],
        }
        callgraph = {"g": ["f"]}
        lattice = CLattice()
        (gen_cs, sketches) = compute_sketches(
            constraints, callgraph, lattice=lattice
        )

        gen = CTypeGenerator(sketches, lattice, CLatticeCTypes(), 4, 4)
        dtv2type = gen()
        A_ptr = dtv2type[DerivedTypeVariable("g")].params[0]
        A_struct = A_ptr.target_type
        self.assertEqual(len(A_struct.fields), 2)
        # A contains a pointer to B
        B_struct = A_struct.fields[0].ctype.target_type
        self.assertNotEqual(A_struct.name, B_struct.name)
        # B contains a pointer to A
        self.assertEqual(
            B_struct.fields[0].ctype.target_type.name, A_struct.name
        )
        # The element type of A is Int
        self.assertIsInstance(A_struct.fields[1].ctype, IntType)
        # The element type of B is float
        self.assertIsInstance(B_struct.fields[1].ctype, FloatType)

        # The type of the second argument is A too
        # Right now we get a different struct with the same structure
        # but it wouldn't have to be this way if sketches are not trees
        # but factor out common subtrees.
        A_struct2 = dtv2type[DerivedTypeVariable("g")].params[1].target_type
        self.assertEqual(len(A_struct2.fields), 2)
        self.assertIsInstance(A_struct.fields[1].ctype, IntType)

    def test_regression1(self):
        """
        When more than one typevar gets instantiated in a chain of constraints,
        we weren't following the entire chain (transitively) to get the atomic
        type information (for the lattice).
        """
        nf_apply = SchemaParser.parse_variable("nf")

        constraints = {nf_apply: ConstraintSet()}
        constraint_str = """
        v_41 ⊑ int
        v_162.load.σ4@0 ⊑ v_52
        v_55 ⊑ v_114
        v_73 ⊑ v_162.store.σ4@0
        v_162.load.σ4@4 ⊑ v_73
        v_41 ⊑ v_55
        nf.in_0 ⊑ v_162
        v_52 ⊑ v_55
        v_55 ⊑ v_162.store.σ4@4
        v_162.load.σ4@4 ⊑ v_41
        v_52 ⊑ int
        v_114 ⊑ nf.out
        """
        for line in constraint_str.split("\n"):
            line = line.strip()
            if line.strip():
                constraints[nf_apply].add(SchemaParser.parse_constraint(line))

        program = Program(DummyLattice(), {}, constraints, {nf_apply: []})
        solver = Solver(program, verbose=True)
        (gen_const, sketches) = solver()

        nf_sketches = sketches[nf_apply]
        self.assertEqual(
            nf_sketches.lookup[
                SchemaParser.parse_variable("nf.in_0.load.σ4@4")
            ].upper_bound,
            SchemaParser.parse_variable("int"),
        )
        self.assertEqual(
            nf_sketches.lookup[
                SchemaParser.parse_variable("nf.in_0.load.σ4@0")
            ].upper_bound,
            SchemaParser.parse_variable("int"),
        )


class InferTypesTest(unittest.TestCase):
    def test_input_arg_capability(self):
        """
        f.in_0 should get the load.σ1@0 capability even if
        we don't know anything about its type.
        The same with g.out
        """
        constraints = {
            # T-inheritR
            "f": ["f.in_0 <= A", "A.load.σ1@0 <= B", "B.load.σ4@4 <= C"],
            # T-inheritL
            "g": ["A <= g.out", "A.load.σ1@0 <= B", "B.load.σ4@4 <= C"],
        }
        callgraph = {"f": [], "g": []}
        (gen_cs, sketches) = compute_sketches(constraints, callgraph)

        f_sketch = sketches[DerivedTypeVariable("f")]
        assert SchemaParser.parse_variable("f.in_0") in f_sketch.lookup
        assert (
            SchemaParser.parse_variable("f.in_0.load.σ1@0.load.σ4@4")
            in f_sketch.lookup
        )

        f_sketch = sketches[DerivedTypeVariable("g")]
        assert SchemaParser.parse_variable("g.out") in f_sketch.lookup
        assert (
            SchemaParser.parse_variable("g.out.load.σ1@0") in f_sketch.lookup
        )
        assert (
            SchemaParser.parse_variable("g.out.load.σ1@0.load.σ4@4")
            in f_sketch.lookup
        )

    def test_input_arg_capability_transitive(self):
        """
        g.in_0 should get the load.σ1@0 capability from f.
        """
        constraints = {
            "f": ["f.in_0 <= A", "A.load.σ1@0 <= B", "B.load.σ4@4 <= C"],
            "g": ["g.in_0 <= C", "C <= f.in_0", "C <= g.out"],
        }
        callgraph = {"g": ["f"]}
        (gen_cs, sketches) = compute_sketches(constraints, callgraph)

        f_sketch = sketches[DerivedTypeVariable("g")]
        assert SchemaParser.parse_variable("g.in_0") in f_sketch.lookup
        assert (
            SchemaParser.parse_variable("g.in_0.load.σ1@0.load.σ4@4")
            in f_sketch.lookup
        )
        assert (
            SchemaParser.parse_variable("g.out.load.σ1@0.load.σ4@4")
            in f_sketch.lookup
        )


class CTypeTest(unittest.TestCase):
    def test_simple_struct(self):
        """
        Verify that we will combine fields inferred from different callees.

            |-> F2  (part of struct fields)
         F1 -
            |-> F3  (other part of struct fields)
        """
        F1 = SchemaParser.parse_variable("F1")
        F2 = SchemaParser.parse_variable("F2")
        F3 = SchemaParser.parse_variable("F3")

        constraints = {
            F1: ConstraintSet(),
            F2: ConstraintSet(),
            F3: ConstraintSet(),
        }
        constraints[F1].add(SchemaParser.parse_constraint("F1.in_0 ⊑ A"))
        constraints[F1].add(SchemaParser.parse_constraint("A ⊑ F2.in_1"))
        constraints[F1].add(SchemaParser.parse_constraint("A ⊑ F3.in_2"))
        # F2 accesses fields at offsets 0, 12
        constraints[F2].add(SchemaParser.parse_constraint("F2.in_1 ⊑ B"))
        constraints[F2].add(SchemaParser.parse_constraint("B.load.σ8@0 ⊑ int"))
        constraints[F2].add(
            SchemaParser.parse_constraint("B.load.σ4@12 ⊑ int")
        )
        # F3 accesses fields at offsets 8, 20
        constraints[F3].add(SchemaParser.parse_constraint("F3.in_2 ⊑ C"))
        constraints[F3].add(SchemaParser.parse_constraint("C.load.σ2@8 ⊑ int"))
        constraints[F3].add(
            SchemaParser.parse_constraint("C.load.σ8@20 ⊑ int")
        )

        lattice = DummyLattice()
        lattice_ctypes = DummyLatticeCTypes()
        program = Program(lattice, {}, constraints, {F1: [F2, F3]})
        solver = Solver(program, verbose=VERBOSE_TESTS)
        (gen_const, sketches) = solver()
        # print(sketches[F1])

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
        # print(list(map(lambda x: type(x.ctype), struct.fields)))

    def test_string_in_struct(self):
        """
        Model that strcpy() is called with a field from a struct as the destination, which
        _should_ tell us that the given field is a string.
        """
        F1 = SchemaParser.parse_variable("F1")
        strcpy = SchemaParser.parse_variable("strcpy")

        constraints = {F1: ConstraintSet(), strcpy: ConstraintSet()}
        constraints[F1].add(SchemaParser.parse_constraint("F1.in_0 ⊑ A"))
        constraints[F1].add(
            SchemaParser.parse_constraint("A.load.σ8@8 ⊑ strcpy.in_1")
        )
        constraints[strcpy].add(
            SchemaParser.parse_constraint(
                "strcpy.in_1.load.σ1@0*[nullterm] ⊑ int"
            )
        )

        lattice = DummyLattice()
        lattice_ctypes = DummyLatticeCTypes()
        program = Program(lattice, {}, constraints, {F1: [strcpy]})
        solver = Solver(program, verbose=VERBOSE_TESTS)
        (gen_const, sketches) = solver()
        # print(sketches[F1])

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
        F1 = SchemaParser.parse_variable("F1")
        memcpy = SchemaParser.parse_variable("memcpy")
        some_global = SchemaParser.parse_variable("some_global")

        constraints = {F1: ConstraintSet()}
        constraints[F1].add(
            SchemaParser.parse_constraint("some_global ⊑ memcpy.in_1")
        )
        constraints[F1].add(
            SchemaParser.parse_constraint("memcpy.in_1.load.σ4@0*[10] ⊑ int")
        )

        lattice = DummyLattice()
        lattice_ctypes = DummyLatticeCTypes()
        program = Program(lattice, {some_global}, constraints, {F1: [memcpy]})
        solver = Solver(program, verbose=VERBOSE_TESTS)
        (gen_const, sketches) = solver()
        # print(sketches)

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
        F1 = SchemaParser.parse_variable("F1")
        some_global = SchemaParser.parse_variable("some_global")

        constraints = {F1: ConstraintSet()}
        constraints[F1].add(SchemaParser.parse_constraint("some_global ⊑ A"))
        constraints[F1].add(SchemaParser.parse_constraint("A.load.σ4@0 ⊑ int"))
        constraints[F1].add(
            SchemaParser.parse_constraint("int ⊑ A.store.σ4@4")
        )
        constraints[F1].add(SchemaParser.parse_constraint("A.load.σ4@8 ⊑ int"))
        constraints[F1].add(
            SchemaParser.parse_constraint("int ⊑ A.store.σ4@12")
        )
        constraints[F1].add(
            SchemaParser.parse_constraint("A.load.σ4@16 ⊑ int")
        )
        constraints[F1].add(
            SchemaParser.parse_constraint("int ⊑ A.store.σ4@20")
        )

        lattice = DummyLattice()
        lattice_ctypes = DummyLatticeCTypes()
        program = Program(lattice, {some_global}, constraints, {F1: []})
        solver = Solver(program, verbose=VERBOSE_TESTS)
        (gen_const, sketches) = solver()
        # print(sketches[some_global])

        gen = CTypeGenerator(sketches, lattice, lattice_ctypes, 8, 8)
        dtv2type = gen()
        t = dtv2type[some_global]
        self.assertEqual(type(t), PointerType)
        self.assertEqual(type(t.target_type), StructType)
        self.assertEqual(len(t.target_type.fields), 6)
        self.assertEqual(max([f.offset for f in t.target_type.fields]), 20)
        self.assertEqual(min([f.offset for f in t.target_type.fields]), 0)

    def test_tight_bounds_in(self):
        constraints = ConstraintSet()
        constraints.add(SchemaParser.parse_constraint("f.in_0 ⊑ A"))
        constraints.add(SchemaParser.parse_constraint("int ⊑ A"))
        constraints.add(SchemaParser.parse_constraint("A ⊑ int"))
        f = SchemaParser.parse_variable("f")
        program = Program(DummyLattice(), set(), {f: constraints}, {f: {}})
        solver = Solver(program, verbose=LogLevel.DEBUG)
        (gen_const, sketches) = solver()
        f_in1 = SchemaParser.parse_variable("f.in_0")
        self.assertEqual(sketches[f].lookup[f_in1].upper_bound, CLattice._int)
        gen = CTypeGenerator(
            sketches,
            CLattice(),
            CLatticeCTypes(),
            4,
            4,
            verbose=LogLevel.DEBUG,
        )
        output = gen()
        f_ft = output[f]
        self.assertTrue(isinstance(f_ft, FunctionType))
        self.assertTrue(isinstance(f_ft.params[0], IntType))

    def test_tight_bounds_out(self):
        constraints = ConstraintSet()
        constraints.add(SchemaParser.parse_constraint("A ⊑ f.out"))
        constraints.add(SchemaParser.parse_constraint("int ⊑ A"))
        constraints.add(SchemaParser.parse_constraint("A ⊑ int"))
        f = SchemaParser.parse_variable("f")
        program = Program(DummyLattice(), set(), {f: constraints}, {f: {}})
        solver = Solver(program, verbose=LogLevel.DEBUG)
        (gen_const, sketches) = solver()
        print(gen_const[f])
        print(sketches[f])
        f_out = SchemaParser.parse_variable("f.out")
        self.assertEqual(sketches[f].lookup[f_out].lower_bound, CLattice._int)
        gen = CTypeGenerator(
            sketches,
            CLattice(),
            CLatticeCTypes(),
            4,
            4,
            verbose=LogLevel.DEBUG,
        )
        output = gen()
        f_ft = output[f]
        self.assertTrue(isinstance(f_ft, FunctionType))
        self.assertTrue(isinstance(f_ft.return_type, IntType))


if __name__ == "__main__":
    unittest.main()
