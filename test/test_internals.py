"""Simple unit tests from the paper and slides that rely on looking under-the-hood
at generated constraints.
"""


import pytest
from retypd import (
    CLattice,
    ConstraintSet,
    DerivedTypeVariable,
    DummyLattice,
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
from test_endtoend import parse_cs_set, parse_cs


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
    solver = Solver(None)
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
    solver = Solver(None)
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
    solver = Solver(None)
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
