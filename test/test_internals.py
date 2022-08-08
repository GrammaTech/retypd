"""Simple unit tests from the paper and slides that rely on looking under-the-hood
at generated constraints.
"""


import pytest
from retypd import (
    CLattice,
    ConstraintSet,
    CTypeGenerator,
    DerivedTypeVariable,
    DummyLattice,
    SchemaParser,
    Solver,
    SolverConfig,
    DerefLabel,
    Program,
    CType,
    CLatticeCTypes,
    BoolType,
    CharType,
    FunctionType,
    FloatType,
    IntType,
    PointerType,
    VoidType,
)
from test_endtoend import parse_cs_set, parse_cs, parse_var


@pytest.mark.commit
def test_parse_label():
    l = SchemaParser.parse_label("σ8@0")
    assert (l.size, l.offset, l.count) == (8, 0, 1)
    l = SchemaParser.parse_label("σ0@10000")
    assert (l.size, l.offset, l.count) == (0, 10000, 1)
    l = SchemaParser.parse_label("σ4@-32")
    assert (l.size, l.offset, l.count) == (4, -32, 1)
    l = SchemaParser.parse_label("σ2@32*[1000]")
    assert (l.size, l.offset, l.count) == (2, 32, 1000)
    l = SchemaParser.parse_label("σ2@32*[nobound]")
    assert (l.size, l.offset, l.count) == (2, 32, DerefLabel.COUNT_NOBOUND)

    l = SchemaParser.parse_label("σ2@32*[nullterm]")
    assert (l.size, l.offset, l.count) == (2, 32, DerefLabel.COUNT_NULLTERM)
    with pytest.raises(ValueError):
        l = SchemaParser.parse_label("σ-9@100")


@pytest.mark.commit
def test_simple_constraints():
    """A simple test from the paper (the right side of Figure 4 on p. 6). This one has no
    recursive data structures; as such, the fixed point would suffice. However, we compute type
    constraints in the same way as in the presence of recursion.
    """
    constraints = parse_cs_set(
        ["p ⊑ q", "x ⊑ q.store.σ4@0", "p.load.σ4@0 ⊑ y"]
    )
    f, x, y = SchemaParser.parse_variables(["f", "x", "y"])
    lattice = DummyLattice()
    solver = Solver(Program(CLattice(), {}, {}, {}))
    generated = solver._generate_type_scheme(
        constraints, {x, y}, lattice.internal_types
    )
    assert parse_cs("x ⊑ y") in generated


@pytest.mark.commit
def test_other_simple_constraints():
    """Another simple test from the paper (the program modeled in Figure 14 on p. 26)."""
    constraints = parse_cs_set(
        ["y <= p", "p <= x", "A <= x.store", "y.load <= B"]
    )
    f, A, B = SchemaParser.parse_variables(["f", "A", "B"])
    lattice = DummyLattice()
    solver = Solver(Program(CLattice(), {}, {}, {}))
    generated = solver._generate_type_scheme(
        constraints, {A, B}, lattice.internal_types
    )
    assert parse_cs("A ⊑ B") in generated


@pytest.mark.commit
def test_forgets():
    """A simple test to check that paths that include "forgotten" labels reconstruct access
    paths in the correct order.
    """
    l, F = SchemaParser.parse_variables(["l", "F"])
    constraints = ConstraintSet()
    constraint = parse_cs("l ⊑ F.in_1.load.σ8@0")
    constraints.add(constraint)
    lattice = DummyLattice()
    solver = Solver(Program(CLattice(), {}, {}, {}))
    generated = solver._generate_type_scheme(
        constraints, {l, F}, lattice.internal_types
    )
    assert constraint in generated


@pytest.mark.parametrize(
    ("lhs", "rhs", "expected"),
    [
        ("float", "uint", "┬"),
        ("uint", "int", "uint"),
        ("char", "int", "int"),
        ("int64", "int", "int"),
        ("uint32", "uint64", "uint"),
    ],
)
@pytest.mark.commit
def test_join(lhs: str, rhs: str, expected: str):
    """Test C-lattice join operations against known values"""
    lhs_dtv = DerivedTypeVariable(lhs)
    rhs_dtv = DerivedTypeVariable(rhs)
    equal_dtv = DerivedTypeVariable(expected)
    assert CLattice().join(lhs_dtv, rhs_dtv) == equal_dtv


@pytest.mark.parametrize(
    ("name", "ctype", "size"),
    [
        ("int", IntType(4, True), 4),
        ("int8", IntType(1, True), None),
        ("int16", IntType(2, True), None),
        ("int32", IntType(4, True), None),
        ("int64", IntType(8, True), None),
        ("uint", IntType(4, False), 4),
        ("uint8", IntType(1, False), None),
        ("uint16", IntType(2, False), None),
        ("uint32", IntType(4, False), None),
        ("uint64", IntType(8, False), None),
        ("void", VoidType(), None),
        ("char", CharType(1), 1),
        ("bool", BoolType(1), 1),
        ("float", FloatType(4), None),
        ("double", FloatType(8), None),
    ],
)
@pytest.mark.commit
def test_atom_to_ctype(name: str, ctype: CType, size: int):
    """Test C-lattice are converted to C-types correctly"""
    atom = DerivedTypeVariable(name)
    lattice = CLatticeCTypes()
    ctype_lhs = lattice.atom_to_ctype(atom, CLattice._top, size)
    ctype_rhs = lattice.atom_to_ctype(CLattice._bottom, atom, size)
    assert str(ctype) == str(ctype_lhs)
    assert str(ctype) == str(ctype_rhs)


@pytest.mark.commit
def test_infers_all_inputs():
    """Test that we infer all the inputs for a function"""
    F = parse_var("F")
    constraints = ConstraintSet()
    constraints.add(parse_cs("F.in_2.in_1 ⊑ int"))
    lattice = DummyLattice()
    solver = Solver(Program(CLattice(), {F}, {F: constraints}, {F: {}}))
    _, sketches = solver()

    gen = CTypeGenerator(sketches, lattice, CLatticeCTypes(), 4, 4, verbose=2)
    dtv2type = gen()
    assert isinstance(dtv2type[F], FunctionType)
    assert dtv2type[F].params[0] is not None
    assert dtv2type[F].params[1] is not None
    assert isinstance(dtv2type[F].params[2], PointerType)
    assert isinstance(dtv2type[F].params[2].target_type, FunctionType)
    assert dtv2type[F].params[2].target_type.params[0] is not None
    assert isinstance(dtv2type[F].params[2].target_type.params[1], IntType)
    assert dtv2type[F].params[2].target_type.params[1] is not None

@pytest.mark.commit
def test_top_down():
    """
    Test that top-down propagation can propagate information correctly
    """
    config = SolverConfig(top_down_propagation=True)
    F, G = SchemaParser.parse_variables(["F", "G"])
    constraints = {F: ConstraintSet(), G: ConstraintSet()}
    constraints[F].add(parse_cs("int ⊑ F.in_1.load.σ8@0"))
    constraints[G].add(parse_cs("int ⊑ x.load.σ8@8"))
    constraints[G].add(parse_cs("x ⊑ F.in_1"))
    constraints[G].add(parse_cs("F.in_1 ⊑ G.in_1"))

    solver = Solver(
        Program(CLattice(), {}, constraints, {G: {F}}), config=config
    )
    gen_cst, sketches = solver()

    assert parse_cs("int ⊑ F.in_1.load.σ8@8") in gen_cst[F]
    assert sketches[F].lookup(
        parse_var("F.in_1.load.σ8@8")
    ).lower_bound == DerivedTypeVariable("int")


@pytest.mark.commit
def test_top_down_two_levels():
    config = SolverConfig(top_down_propagation=True)
    F, G, H = SchemaParser.parse_variables(["F", "G", "H"])
    constraints = {F: ConstraintSet(), G: ConstraintSet(), H: ConstraintSet()}
    constraints[F].add(parse_cs("F.out ⊑ int"))
    constraints[G].add(parse_cs("G.in_1 ⊑ F.in_1"))

    constraints[H].add(parse_cs("int ⊑ G.in_1"))

    solver = Solver(
        Program(CLattice(), {}, constraints, {G: {F}, H: {G}}), config=config
    )
    gen_cst, sketches = solver()
    assert parse_cs("int ⊑ F.in_1") in gen_cst[F]
    assert sketches[F].lookup(
        parse_var("F.in_1")
    ).lower_bound == DerivedTypeVariable("int")


@pytest.mark.commit
def test_top_down_three_levels():
    config = SolverConfig(top_down_propagation=True)
    F, G, H, I = SchemaParser.parse_variables(["F", "G", "H", "I"])
    constraints = {
        F: ConstraintSet(),
        G: ConstraintSet(),
        H: ConstraintSet(),
        I: ConstraintSet(),
    }
    constraints[F].add(parse_cs("F.out ⊑ int"))
    constraints[G].add(parse_cs("G.in_1 ⊑ F.in_1"))
    constraints[G].add(parse_cs("G.in_2 ⊑ F.in_2"))
    constraints[H].add(parse_cs("H.in_1 ⊑ G.in_1"))
    constraints[I].add(parse_cs("int ⊑ H.in_1"))
    constraints[I].add(parse_cs("int ⊑ G.in_2"))

    solver = Solver(
        Program(CLattice(), {}, constraints, {G: {F}, H: {G}, I: {H, G}}),
        config=config,
    )
    gen_cst, sketches = solver()
    assert parse_cs("int ⊑ F.in_1") in gen_cst[F]
    assert parse_cs("int ⊑ F.in_2") in gen_cst[F]
    assert parse_cs("int ⊑ G.in_1") in gen_cst[G]
    assert parse_cs("int ⊑ G.in_2") in gen_cst[G]
    assert parse_cs("int ⊑ H.in_1") in gen_cst[H]
    assert sketches[F].lookup(
        parse_var("F.in_1")
    ).lower_bound == DerivedTypeVariable("int")
    assert sketches[F].lookup(
        parse_var("F.in_2")
    ).lower_bound == DerivedTypeVariable("int")
    assert sketches[G].lookup(
        parse_var("G.in_1")
    ).lower_bound == DerivedTypeVariable("int")
    assert sketches[G].lookup(
        parse_var("G.in_2")
    ).lower_bound == DerivedTypeVariable("int")
    assert sketches[H].lookup(
        parse_var("H.in_1")
    ).lower_bound == DerivedTypeVariable("int")
