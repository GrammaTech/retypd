"""Simple unit tests from the paper and slides that only look at the final result (sketches)
"""

import unittest
from typing import Dict, List, Iterable, Set

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
    CType,
    CompoundType,
)
from retypd.solver import SolverConfig


def is_non_primitive(ctype: CType) -> bool:
    return isinstance(ctype, (FunctionType, CompoundType, PointerType))


def collect_all_non_primitive_types(
    function_types: Iterable[CType],
) -> Set[CType]:
    def get_subtypes(ctype: CType) -> List[CType]:
        if isinstance(ctype, FunctionType):
            return [ctype.return_type] + list(ctype.params)
        elif isinstance(ctype, PointerType):
            return [ctype.target_type]
        elif isinstance(ctype, CompoundType):
            return [field.ctype for field in ctype.fields]
        else:
            raise Exception(f"Cannot get subtypes of {type(ctype)}")

    worklist = list(function_types)
    collected_types = set()
    while len(worklist) > 0:
        ctype = worklist.pop()
        if ctype not in collected_types and is_non_primitive(ctype):
            if not isinstance(ctype, PointerType):
                collected_types.add(ctype)
            worklist.extend(get_subtypes(ctype))
    return collected_types


VERBOSE_TESTS = False


def compute_sketches(
    cs: Dict[str, List[str]],
    callgraph: Dict[str, List[str]],
    lattice: Lattice = DummyLattice(),
    config: SolverConfig = SolverConfig(),
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
    solver = Solver(program, verbose=VERBOSE_TESTS, config=config)
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
            F_sketches.lookup(
                SchemaParser.parse_variable("F.out")
            ).lower_bound,
            SchemaParser.parse_variable("#SuccessZ"),
        )
        self.assertEqual(
            F_sketches.lookup(
                SchemaParser.parse_variable("F.in_0.load.σ4@4")
            ).upper_bound,
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
        self.assertIsNotNone(
            g_sketch.lookup(SchemaParser.parse_variable("g.in_0.load.σ4@4"))
        )
        self.assertEqual(
            g_sketch.lookup(
                SchemaParser.parse_variable("g.in_0.load.σ4@4")
            ).upper_bound,
            DummyLattice._int,
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

    def test_multiple_label_nodes(self):
        """The type of f.in_0 is:

        struct list{
            list* next;
            list* prev;
            int elem;
            ?   elem2;
        }

        The sketch will have two label nodes for 'next'
        and 'prev' that point to the same type, so we need
        the hash and equality for label nodes to take the id
        into account.
        """
        constraints = {
            "f": [
                "f.in_0 <= list",
                "list.load.σ4@0 <= next",
                "list.load.σ4@4 <= prev",
                "next <= list",
                "prev <= list",
                "list.load.σ4@8 <= elem",
                "elem <= int",
                "list.load.σ4@12 <= elem2",
            ],
            "g": ["g.in_0 <= C", "C <= f.in_0"],
        }
        callgraph = {"g": ["f"]}
        lattice = CLattice()
        (gen_cs, sketches) = compute_sketches(
            constraints, callgraph, lattice=lattice
        )

        f_sketch = sketches[DerivedTypeVariable("f")]
        self.assertIsNotNone(
            f_sketch.lookup(SchemaParser.parse_variable("f.in_0.load.σ4@0"))
        )
        # we should have this capability even if we don't know the type
        # of the field
        self.assertIsNotNone(
            f_sketch.lookup(SchemaParser.parse_variable("f.in_0.load.σ4@12"))
        )
        self.assertEqual(
            f_sketch.lookup(SchemaParser.parse_variable("f.in_0.load.σ4@0")),
            f_sketch.lookup(SchemaParser.parse_variable("f.in_0")),
        )
        self.assertEqual(
            f_sketch.lookup(SchemaParser.parse_variable("f.in_0.load.σ4@4")),
            f_sketch.lookup(SchemaParser.parse_variable("f.in_0")),
        )

        # check that information is transferred correctly to "g"
        g_sketch = sketches[DerivedTypeVariable("g")]
        self.assertIsNotNone(
            g_sketch.lookup(SchemaParser.parse_variable("g.in_0.load.σ4@0"))
        )
        self.assertIsNotNone(
            g_sketch.lookup(SchemaParser.parse_variable("g.in_0.load.σ4@12"))
        )
        self.assertEqual(
            g_sketch.lookup(SchemaParser.parse_variable("g.in_0.load.σ4@0")),
            g_sketch.lookup(SchemaParser.parse_variable("g.in_0")),
        )
        self.assertEqual(
            g_sketch.lookup(SchemaParser.parse_variable("g.in_0.load.σ4@4")),
            g_sketch.lookup(SchemaParser.parse_variable("g.in_0")),
        )
        # check C types
        gen = CTypeGenerator(sketches, lattice, CLatticeCTypes(), 4, 4)
        dtv2type = gen()
        rec_struct_ptr = dtv2type[DerivedTypeVariable("f")].params[0]
        rec_struct = rec_struct_ptr.target_type
        self.assertEqual(len(rec_struct.fields), 4)
        # the field 0 and 1 are pointers to the same struct
        self.assertEqual(
            rec_struct.fields[0].ctype.target_type.name, rec_struct.name
        )
        self.assertEqual(
            rec_struct.fields[1].ctype.target_type.name, rec_struct.name
        )
        self.assertIsInstance(rec_struct.fields[2].ctype, IntType)

    def test_multiple_label_nodes_store(self):
        """The type of f.out and g.out is
        as subtype of:

        struct list{
            list* next;
            list* prev;
            int elem;
        }

        Example with multiple label nodes that are
        stores
        """
        constraints = {
            "f": [
                "f.out <= list",
                "next <= list.store.σ4@0",
                "prev <= list.store.σ4@4",
                "list <= next",
                "list <= prev",
                "elem <= list.store.σ4@8",
                "int <= elem",
            ],
            "g": ["C <= g.out", "f.out <= C"],
        }
        callgraph = {"g": ["f"]}
        lattice = CLattice()
        (gen_cs, sketches) = compute_sketches(
            constraints, callgraph, lattice=lattice
        )
        # check that information is transferred correctly to "g"
        g_sketch = sketches[DerivedTypeVariable("g")]
        self.assertIsNotNone(
            g_sketch.lookup(SchemaParser.parse_variable("g.out.store.σ4@0"))
        )

        self.assertEqual(
            g_sketch.lookup(SchemaParser.parse_variable("g.out.store.σ4@0")),
            g_sketch.lookup(SchemaParser.parse_variable("g.out")),
        )
        self.assertEqual(
            g_sketch.lookup(SchemaParser.parse_variable("g.out.store.σ4@4")),
            g_sketch.lookup(SchemaParser.parse_variable("g.out")),
        )
        self.assertEqual(
            g_sketch.lookup(
                SchemaParser.parse_variable("g.out.store.σ4@8")
            ).lower_bound,
            lattice._int,
        )

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
            nf_sketches.lookup(
                SchemaParser.parse_variable("nf.in_0.load.σ4@4")
            ).upper_bound,
            SchemaParser.parse_variable("int"),
        )
        self.assertEqual(
            nf_sketches.lookup(
                SchemaParser.parse_variable("nf.in_0.load.σ4@0")
            ).upper_bound,
            SchemaParser.parse_variable("int"),
        )

    def test_vListInsert_issue9(self):
        """
        This is a method from crazyflie
        vlistInsert(List_t * const pxList, ListItem_t * const pxNewListItem)

        typedef struct xLIST
        {
            volatile UBaseType_t uxNumberOfItems;
            ListItem_t *  pxIndex;
            MiniListItem_t xListEnd;
        } List_t;

        struct xLIST_ITEM
        {

            TickType_t xItemValue;
            struct xLIST_ITEM *  pxNext;
            struct xLIST_ITEM *  pxPrevious;
            void * pvOwner;
            struct xLIST *  pxContainer;
        };
        typedef struct xLIST_ITEM ListItem_t;

        The constraints are incomplete
        - Only the first element of List_t is accessed
        - for ListItem_t, we access xItemValue, pxNext, pxPrevious
          and pxContainer
        """
        constraints = {
            "vListInsert": [
                "v_334 ⊑ int32",
                "int32 ⊑ v_110",
                "v_3 ⊑ int32",
                "v_177 ⊑ v_387",
                "v_3 ⊑ uint32",
                "v_59 ⊑ v_258",
                "v_233 ⊑ v_258.store.σ4@4",
                "v_386 ⊑ v_255",
                "vListInsert.in_1 ⊑ v_233",
                "bool ⊑ v_198",
                "v_110 ⊑ v_232.store.σ4@0",
                "v_384 ⊑ v_386",
                "v_358.load.σ4@8 ⊑ v_59",
                "v_65 ⊑ v_233.store.σ4@4",
                "v_255.load.σ4@4 ⊑ v_189",
                "v_258.load.σ4@4 ⊑ v_65",
                "v_233.load.σ4@0 ⊑ v_3",
                "v_255 ⊑ v_258",
                "v_258 ⊑ v_233.store.σ4@8",
                "v_385 ⊑ v_386",
                "v_112 ⊑ int32",
                "vListInsert.in_0 ⊑ v_232",
                "v_101 ⊑ int32",
                "bool ⊑ v_18",
                "v_195 ⊑ uint32",
                "v_387 ⊑ v_384",
                "v_189 ⊑ v_385",
                "v_232.load.σ4@0 ⊑ v_101",
                "v_232 ⊑ v_233.store.σ4@16",
                "v_189.load.σ4@0 ⊑ v_195",
                "v_233 ⊑ v_65.store.σ4@8",
            ]
        }
        callgraph = {"vListInsert": []}
        lattice = CLattice()
        (gen_cs, sketches) = compute_sketches(
            constraints,
            callgraph,
            lattice=lattice,
            config=SolverConfig(max_paths_per_root=1000),
        )

        gen = CTypeGenerator(sketches, lattice, CLatticeCTypes(), 4, 4)
        dtv2type = gen()

        ListItem_t_ptr = dtv2type[DerivedTypeVariable("vListInsert")].params[1]
        ListItem_t = ListItem_t_ptr.target_type
        self.assertGreaterEqual(len(ListItem_t.fields), 4)
        # second and third are pointers to next and previous
        self.assertEqual(
            ListItem_t.fields[1].ctype.target_type.name, ListItem_t.name
        )
        self.assertEqual(
            ListItem_t.fields[2].ctype.target_type.name, ListItem_t.name
        )
        # last field is a pointer too
        self.assertIsInstance(ListItem_t.fields[3].ctype, PointerType)


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
        self.assertIsNotNone(
            f_sketch.lookup(SchemaParser.parse_variable("f.in_0"))
        )
        self.assertIsNotNone(
            f_sketch.lookup(
                SchemaParser.parse_variable("f.in_0.load.σ1@0.load.σ4@4")
            )
        )

        g_sketch = sketches[DerivedTypeVariable("g")]
        self.assertIsNotNone(
            g_sketch.lookup(SchemaParser.parse_variable("g.out"))
        )
        self.assertIsNotNone(
            g_sketch.lookup(SchemaParser.parse_variable("g.out.load.σ1@0"))
        )
        self.assertIsNotNone(
            g_sketch.lookup(
                SchemaParser.parse_variable("g.out.load.σ1@0.load.σ4@4")
            )
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

        g_sketch = sketches[DerivedTypeVariable("g")]
        self.assertIsNotNone(
            g_sketch.lookup(SchemaParser.parse_variable("g.in_0"))
        )
        self.assertIsNotNone(
            g_sketch.lookup(
                SchemaParser.parse_variable("g.in_0.load.σ1@0.load.σ4@4")
            )
        )
        self.assertIsNotNone(
            g_sketch.lookup(
                SchemaParser.parse_variable("g.out.load.σ1@0.load.σ4@4")
            )
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
        self.assertEqual(sketches[f].lookup(f_in1).upper_bound, CLattice._int)
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
        self.assertEqual(sketches[f].lookup(f_out).lower_bound, CLattice._int)
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
