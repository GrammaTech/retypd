"""Simple unit tests from the paper and slides that only look at the final result (sketches)
"""

import pytest
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
from retypd.graph_solver import GraphSolverConfig
from retypd.solver import SolverConfig


VERBOSE_TESTS = 2
parse_var = SchemaParser.parse_variable
parse_cs = SchemaParser.parse_constraint
parse_cs_set = SchemaParser.parse_constraint_set


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
        parsed_cs[DerivedTypeVariable(proc)] = parse_cs_set(proc_cs)

    parsed_callgraph = {
        DerivedTypeVariable(proc): [
            DerivedTypeVariable(callee) for callee in callees
        ]
        for proc, callees in callgraph.items()
    }
    program = Program(lattice, {}, parsed_cs, parsed_callgraph)
    solver = Solver(program, verbose=VERBOSE_TESTS, config=config)
    return solver()


all_solver_configs = pytest.mark.parametrize(
    "config",
    [
        SolverConfig(
            graph_solver="naive",
            graph_solver_config=GraphSolverConfig(
                restrict_graph_to_reachable=True
            ),
        ),
        SolverConfig(
            graph_solver="pathexpr",
            graph_solver_config=GraphSolverConfig(
                restrict_graph_to_reachable=True
            ),
        ),
        SolverConfig(
            graph_solver="naive",
            graph_solver_config=GraphSolverConfig(
                restrict_graph_to_reachable=False
            ),
        ),
        SolverConfig(
            graph_solver="pathexpr",
            graph_solver_config=GraphSolverConfig(
                restrict_graph_to_reachable=False
            ),
        ),
        SolverConfig(graph_solver="dfa"),
        SolverConfig(graph_solver="dfa", top_down_propagation=True),
    ],
    ids=[
        "naive-reachable",
        "pathexpr-reachable",
        "naive-all",
        "pathexpr-all",
        "dfa-all",
        "dfa-all-topdown",
    ],
)


@all_solver_configs
@pytest.mark.commit
def test_recursive(config):
    """A test based on the running example from the paper (Figure 2 on p. 3) and the slides
    (slides 67-83, labeled as slides 13-15).
    """
    F = parse_var("F")
    constraints = {
        "F": [
            "F.in_0 ⊑ δ",
            "α ⊑ φ",
            "δ ⊑ φ",
            "φ.load.σ4@0 ⊑ α",
            "φ.load.σ4@4 ⊑ α'",
            "α' ⊑ close.in_0",
            "close.out ⊑ F.out",
        ],
        "close": ["close.in_0 ⊑ #FileDescriptor", "#SuccessZ ⊑ close.out"],
    }
    callgraph = {"F": ["close"]}
    (gen_cs, sketches) = compute_sketches(
        constraints, callgraph, lattice=DummyLattice(), config=config
    )
    # Check constraints
    assert gen_cs[F] == parse_cs_set(
        [
            "F.in_0 ⊑ τ$0",
            "τ$0.load.σ4@0 ⊑ τ$0",
            "τ$0.load.σ4@4 ⊑ #FileDescriptor",
            "#SuccessZ ⊑ F.out",
        ]
    )
    # Check sketches
    F_sketches = sketches[F]
    assert F_sketches.lookup(parse_var("F.out")).lower_bound == parse_var(
        "#SuccessZ"
    )
    assert F_sketches.lookup(
        parse_var("F.in_0.load.σ4@4")
    ).upper_bound == parse_var("#FileDescriptor")


@all_solver_configs
@pytest.mark.commit
def test_recursive_no_primitive(config):
    """The type of f.in_0 is recursive.
    struct list{
        list* next;
        int elem;
    }
    We don't know anything about elem, so we need to create a type
    variable for the recursive variable to capture that behavior.
    """
    constraints = {
        "f": [
            "f.in_0 <= list",
            "list.load.σ4@0 <= next",
            "next <= list",
        ],
        "g": ["g.in_0 <= f.in_0"],
    }
    callgraph = {"g": ["f"]}
    lattice = CLattice()
    (gen_cs, sketches) = compute_sketches(
        constraints, callgraph, lattice=lattice, config=config
    )

    assert gen_cs[parse_var("f")] == parse_cs_set(
        ["f.in_0 ⊑ τ$0", "τ$0.load.σ4@0 ⊑ τ$0"]
    )
    assert gen_cs[parse_var("g")] == parse_cs_set(
        ["g.in_0 ⊑ τ$0", "τ$0.load.σ4@0 ⊑ τ$0"]
    )

    g_sketch = sketches[DerivedTypeVariable("g")]
    assert g_sketch.lookup(parse_var("g.in_0.load.σ4@0")) == g_sketch.lookup(
        parse_var("g.in_0")
    )


@pytest.mark.commit
def test_recursive_through_procedures():
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
    assert gen_cs[parse_var("f")] == parse_cs_set(
        ["f.in_0 ⊑ τ$0", "τ$0.load.σ4@0 ⊑ τ$0", "τ$0.load.σ4@4 ⊑ int"]
    )
    assert gen_cs[parse_var("g")] == parse_cs_set(
        ["g.in_0 ⊑ τ$0", "τ$0.load.σ4@0 ⊑ τ$0", "τ$0.load.σ4@4 ⊑ int"]
    )

    g_sketch = sketches[DerivedTypeVariable("g")]
    assert g_sketch.lookup(parse_var("g.in_0.load.σ4@4")) is not None
    assert (
        g_sketch.lookup(parse_var("g.in_0.load.σ4@4")).upper_bound
        == DummyLattice._int
    )

    gen = CTypeGenerator(sketches, lattice, CLatticeCTypes(), 4, 4)
    dtv2type = gen()
    rec_struct_ptr = dtv2type[DerivedTypeVariable("g")].params[0]
    rec_struct = rec_struct_ptr.target_type
    assert len(rec_struct.fields) == 2
    # the field 0 is a pointer to the same struct
    assert rec_struct.fields[0].ctype.target_type.name == rec_struct.name
    assert isinstance(rec_struct.fields[1].ctype, IntType)


@all_solver_configs
@pytest.mark.commit
def test_multiple_label_nodes(config):
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
        constraints, callgraph, lattice=lattice, config=config
    )
    assert gen_cs[parse_var("f")] == parse_cs_set(
        [
            "f.in_0 <= τ$0",
            "τ$0.load.σ4@0 <= τ$0",
            "τ$0.load.σ4@4 <= τ$0",
            "τ$0.load.σ4@8 <= int",
        ]
    )
    assert gen_cs[parse_var("g")] == parse_cs_set(
        [
            "g.in_0 <= τ$0",
            "τ$0.load.σ4@0 <= τ$0",
            "τ$0.load.σ4@4 <= τ$0",
            "τ$0.load.σ4@8 <= int",
        ]
    )
    f_sketch = sketches[DerivedTypeVariable("f")]
    assert f_sketch.lookup(parse_var("f.in_0.load.σ4@0")) is not None
    # we should have this capability even if we don't know the type
    # of the field
    assert f_sketch.lookup(parse_var("f.in_0.load.σ4@12")) is not None
    assert f_sketch.lookup(parse_var("f.in_0.load.σ4@0")) == f_sketch.lookup(
        parse_var("f.in_0")
    )

    assert f_sketch.lookup(parse_var("f.in_0.load.σ4@4")) == f_sketch.lookup(
        parse_var("f.in_0")
    )

    # check that information is transferred correctly to "g"
    g_sketch = sketches[DerivedTypeVariable("g")]
    assert g_sketch.lookup(parse_var("g.in_0.load.σ4@0")) is not None
    assert g_sketch.lookup(parse_var("g.in_0.load.σ4@12")) is not None
    assert g_sketch.lookup(parse_var("g.in_0.load.σ4@0")) == g_sketch.lookup(
        parse_var("g.in_0")
    )
    assert g_sketch.lookup(parse_var("g.in_0.load.σ4@4")) == g_sketch.lookup(
        parse_var("g.in_0")
    )
    # check C types
    gen = CTypeGenerator(sketches, lattice, CLatticeCTypes(), 4, 4)
    dtv2type = gen()
    rec_struct_ptr = dtv2type[DerivedTypeVariable("f")].params[0]
    rec_struct = rec_struct_ptr.target_type
    assert len(rec_struct.fields) == 4
    # the field 0 and 1 are pointers to the same struct
    assert rec_struct.fields[0].ctype.target_type.name == rec_struct.name
    assert rec_struct.fields[1].ctype.target_type.name == rec_struct.name
    assert isinstance(rec_struct.fields[2].ctype, IntType)


@all_solver_configs
@pytest.mark.commit
def test_multiple_label_nodes_store(config):
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
        "g": ["g.out <= C ", "C <= f.out"],
    }
    callgraph = {"g": ["f"]}
    lattice = CLattice()
    (gen_cs, sketches) = compute_sketches(
        constraints, callgraph, lattice=lattice, config=config
    )
    # check that information is transferred correctly to "g"
    g_sketch = sketches[DerivedTypeVariable("g")]
    assert g_sketch.lookup(parse_var("g.out.store.σ4@0")) is not None
    assert g_sketch.lookup(parse_var("g.out.store.σ4@0")) == g_sketch.lookup(
        parse_var("g.out")
    )
    assert g_sketch.lookup(parse_var("g.out.store.σ4@4")) == g_sketch.lookup(
        parse_var("g.out")
    )
    assert (
        g_sketch.lookup(parse_var("g.out.store.σ4@8")).lower_bound
        == lattice._int
    )


@all_solver_configs
@pytest.mark.commit
def test_interleaving_elements(config):
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
        constraints, callgraph, lattice=lattice, config=config
    )
    assert gen_cs[parse_var("f")] == parse_cs_set(
        [
            "f.in_0 <= τ$0",
            "f.in_1.load.σ4@4 <= float",
            "f.in_1.load.σ4@0 <= τ$0",
            "τ$0.load.σ4@4 <= int",
            "τ$0.load.σ4@0.load.σ4@0 <= τ$0",
            "τ$0.load.σ4@0.load.σ4@4 <= float",
        ]
    )
    assert gen_cs[parse_var("g")] == parse_cs_set(
        [
            "g.in_0 <= τ$0",
            "g.in_1.load.σ4@4 <= float",
            "g.in_1.load.σ4@0 <= τ$0",
            "τ$0.load.σ4@4 <= int",
            "τ$0.load.σ4@0.load.σ4@0 <= τ$0",
            "τ$0.load.σ4@0.load.σ4@4 <= float",
        ]
    )
    gen = CTypeGenerator(sketches, lattice, CLatticeCTypes(), 4, 4)
    dtv2type = gen()
    A_ptr = dtv2type[DerivedTypeVariable("g")].params[0]
    A_struct = A_ptr.target_type
    assert len(A_struct.fields) == 2
    # A contains a pointer to B
    B_struct = A_struct.fields[0].ctype.target_type
    assert A_struct.name != B_struct.name
    # B contains a pointer to A
    assert B_struct.fields[0].ctype.target_type.name == A_struct.name
    # The element type of A is Int
    assert isinstance(A_struct.fields[1].ctype, IntType)
    # The element type of B is float
    assert isinstance(B_struct.fields[1].ctype, FloatType)
    # The type of the second argument is A too
    # Right now we get a different struct with the same structure
    # but it wouldn't have to be this way if sketches are not trees
    # but factor out common subtrees.
    A_struct2 = dtv2type[DerivedTypeVariable("g")].params[1].target_type
    assert len(A_struct2.fields) == 2
    assert isinstance(A_struct.fields[1].ctype, IntType)


@all_solver_configs
@pytest.mark.commit
def test_two_recursive_instantiations(config):
    """The types of f.in_0 and g.in_0 are recursive.
    These correspond to h.in_0 and h.in_1.
    """
    constraints = {
        "f": [
            "f.in_0 <= list",
            "list.load.σ4@0 <= next",
            "next <= list",
            "list.load.σ4@4 <= int",
        ],
        "g": [
            "g.in_0 <= list",
            "list.load.σ4@4 <= next",
            "next <= list",
            "list.load.σ4@0 <= float",
        ],
        "h": ["h.in_0 <= f.in_0", "h.in_1 <= g.in_0"],
    }
    callgraph = {"h": ["f", "g"]}
    lattice = CLattice()
    (gen_cs, sketches) = compute_sketches(
        constraints, callgraph, lattice=lattice, config=config
    )

    assert gen_cs[parse_var("f")] == parse_cs_set(
        ["f.in_0 ⊑ τ$0", "τ$0.load.σ4@0 ⊑ τ$0", "τ$0.load.σ4@4 ⊑ int"]
    )
    assert gen_cs[parse_var("g")] == parse_cs_set(
        ["g.in_0 ⊑ τ$0", "τ$0.load.σ4@4 ⊑ τ$0", "τ$0.load.σ4@0 ⊑ float"]
    )
    assert gen_cs[parse_var("h")] == parse_cs_set(
        [
            "h.in_0 ⊑ τ$0",
            "τ$0.load.σ4@0 ⊑ τ$0",
            "τ$0.load.σ4@4 ⊑ int",
            "h.in_1 ⊑ τ$1",
            "τ$1.load.σ4@4 ⊑ τ$1",
            "τ$1.load.σ4@0 ⊑ float",
        ]
    )


@all_solver_configs
@pytest.mark.commit
def test_in_out_constraints_propagation(config):
    """
    The instantiation of f should allows us to conclude that
    g.in_0 is int32.
    """
    constraints = {
        "f": [
            "f.in_0 <= f.out",
        ],
        "g": ["g.in_0 <= C", "C <= f.in_0", "f.out <= int32"],
    }
    callgraph = {"g": ["f"]}
    lattice = CLattice()
    (gen_cs, sketches) = compute_sketches(
        constraints, callgraph, lattice=lattice, config=config
    )
    assert parse_cs("g.in_0 <= int32") in gen_cs[parse_var("g")]
    gen = CTypeGenerator(sketches, lattice, CLatticeCTypes(), 4, 4)
    dtv2type = gen()
    assert isinstance(dtv2type[DerivedTypeVariable("g")].params[0], IntType)


@pytest.mark.commit
def test_argument_constraints_propagation():
    """
    The instantiation of f should allows us to conclude that
    g.in_0 is int32.
    """
    constraints = {
        "f": [
            "f.in_0 <= f.in_1",
        ],
        "g": ["g.in_0 <= C", "C <= f.in_0", "f.in_1 <= int32"],
    }
    callgraph = {"g": ["f"]}
    lattice = CLattice()
    (gen_cs, sketches) = compute_sketches(
        constraints, callgraph, lattice=lattice
    )
    assert parse_cs("g.in_0 <= int32") in gen_cs[parse_var("g")]
    gen = CTypeGenerator(sketches, lattice, CLatticeCTypes(), 4, 4)
    dtv2type = gen()
    assert isinstance(dtv2type[DerivedTypeVariable("g")].params[0], IntType)


@all_solver_configs
@pytest.mark.commit
def test_regression1(config):
    """
    When more than one typevar gets instantiated in a chain of constraints,
    we weren't following the entire chain (transitively) to get the atomic
    type information (for the lattice).
    """
    nf_apply = parse_var("nf")
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
            constraints[nf_apply].add(parse_cs(line))
    program = Program(DummyLattice(), {}, constraints, {nf_apply: []})
    solver = Solver(program, config=config, verbose=True)
    (gen_const, sketches) = solver()
    nf_sketches = sketches[nf_apply]
    assert nf_sketches.lookup(
        parse_var("nf.in_0.load.σ4@4")
    ).upper_bound == parse_var("int")
    assert nf_sketches.lookup(
        parse_var("nf.in_0.load.σ4@0")
    ).upper_bound == parse_var("int")


@all_solver_configs
@pytest.mark.commit
def test_vListInsert_issue9(config):
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
        config=config,
    )
    gen = CTypeGenerator(sketches, lattice, CLatticeCTypes(), 4, 4)
    dtv2type = gen()
    ListItem_t_ptr = dtv2type[DerivedTypeVariable("vListInsert")].params[1]
    ListItem_t = ListItem_t_ptr.target_type
    assert len(ListItem_t.fields) >= 4
    # field 0 is an integer
    assert isinstance(ListItem_t.fields[0].ctype, IntType)
    # second and third are pointers to next and previous
    assert ListItem_t.fields[1].ctype.target_type.name == ListItem_t.name
    assert ListItem_t.fields[2].ctype.target_type.name == ListItem_t.name
    # last field is a pointer too
    assert isinstance(ListItem_t.fields[3].ctype, PointerType)


@pytest.mark.commit
def test_input_arg_capability():
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
    assert f_sketch.lookup(parse_var("f.in_0")) is not None
    assert f_sketch.lookup(parse_var("f.in_0.load.σ1@0.load.σ4@4")) is not None

    g_sketch = sketches[DerivedTypeVariable("g")]
    assert g_sketch.lookup(parse_var("g.out")) is not None
    assert g_sketch.lookup(parse_var("g.out.load.σ1@0")) is not None
    assert g_sketch.lookup(parse_var("g.out.load.σ1@0.load.σ4@4")) is not None


@pytest.mark.commit
def test_input_arg_capability_transitive():
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
    assert g_sketch.lookup(parse_var("g.in_0")) is not None
    assert g_sketch.lookup(parse_var("g.in_0.load.σ1@0.load.σ4@4")) is not None
    assert g_sketch.lookup(parse_var("g.out.load.σ1@0.load.σ4@4")) is not None


@pytest.mark.commit
def test_simple_struct():
    """
    Verify that we will combine fields inferred from different callees.
        |-> F2  (part of struct fields)
     F1 -
        |-> F3  (other part of struct fields)
    """
    F1 = parse_var("F1")
    F2 = parse_var("F2")
    F3 = parse_var("F3")
    constraints = {
        F1: ConstraintSet(),
        F2: ConstraintSet(),
        F3: ConstraintSet(),
    }
    constraints[F1].add(parse_cs("F1.in_0 ⊑ A"))
    constraints[F1].add(parse_cs("A ⊑ F2.in_1"))
    constraints[F1].add(parse_cs("A ⊑ F3.in_2"))
    # F2 accesses fields at offsets 0, 12
    constraints[F2].add(parse_cs("F2.in_1 ⊑ B"))
    constraints[F2].add(parse_cs("B.load.σ8@0 ⊑ int"))
    constraints[F2].add(parse_cs("B.load.σ4@12 ⊑ int"))
    # F3 accesses fields at offsets 8, 20
    constraints[F3].add(parse_cs("F3.in_2 ⊑ C"))
    constraints[F3].add(parse_cs("C.load.σ2@8 ⊑ int"))
    constraints[F3].add(parse_cs("C.load.σ8@20 ⊑ int"))
    lattice = DummyLattice()
    lattice_ctypes = DummyLatticeCTypes()
    program = Program(lattice, {}, constraints, {F1: [F2, F3]})
    solver = Solver(program, verbose=VERBOSE_TESTS)
    (gen_const, sketches) = solver()
    # print(sketches[F1])
    gen = CTypeGenerator(sketches, lattice, lattice_ctypes, 8, 8)
    dtv2type = gen()
    pstruct = dtv2type[F1].params[0]
    assert isinstance(pstruct, PointerType)
    struct = pstruct.target_type
    assert isinstance(struct, StructType)
    for f in struct.fields:
        assert isinstance(f.ctype, IntType)
        expected_size = {0: 8, 12: 4, 8: 2, 20: 8}.get(f.offset, None)
        assert expected_size is not None
        assert f.size == expected_size
    # print(list(map(lambda x: type(x.ctype), struct.fields)))


@pytest.mark.commit
def test_string_in_struct():
    """
    Model that strcpy() is called with a field from a struct as the destination, which
    _should_ tell us that the given field is a string.
    """
    F1 = parse_var("F1")
    strcpy = parse_var("strcpy")
    constraints = {F1: ConstraintSet(), strcpy: ConstraintSet()}
    constraints[F1].add(parse_cs("F1.in_0 ⊑ A"))
    constraints[F1].add(parse_cs("A.load.σ8@8 ⊑ strcpy.in_1"))
    constraints[strcpy].add(parse_cs("strcpy.in_1.load.σ1@0*[nullterm] ⊑ int"))
    lattice = DummyLattice()
    lattice_ctypes = DummyLatticeCTypes()
    program = Program(lattice, {}, constraints, {F1: [strcpy]})
    solver = Solver(program, verbose=VERBOSE_TESTS)
    (gen_const, sketches) = solver()
    # print(sketches[F1])
    gen = CTypeGenerator(sketches, lattice, lattice_ctypes, 8, 8)
    dtv2type = gen()
    pstruct = dtv2type[F1].params[0]
    assert isinstance(pstruct, PointerType)
    struct = pstruct.target_type
    assert isinstance(struct, StructType)
    for f in struct.fields:
        assert type(f.ctype) == PointerType
        assert type(f.ctype.target_type) == CharType
        assert f.offset == 8


@pytest.mark.commit
def test_global_array():
    """
    Illustration of how a model of memcpy might work.
    """
    F1 = parse_var("F1")
    memcpy = parse_var("memcpy")
    some_global = parse_var("some_global")
    constraints = {F1: ConstraintSet()}
    constraints[F1].add(parse_cs("some_global ⊑ memcpy.in_1"))
    constraints[F1].add(parse_cs("memcpy.in_1.load.σ4@0*[10] ⊑ int"))
    lattice = DummyLattice()
    lattice_ctypes = DummyLatticeCTypes()
    program = Program(lattice, {some_global}, constraints, {F1: [memcpy]})
    solver = Solver(program, verbose=VERBOSE_TESTS)
    (gen_const, sketches) = solver()
    # print(sketches)
    gen = CTypeGenerator(sketches, lattice, lattice_ctypes, 8, 8)
    dtv2type = gen()
    t = dtv2type[some_global]
    assert type(t) == PointerType
    assert type(t.target_type) == ArrayType
    assert t.target_type.length == 10
    assert t.target_type.member_type.size == 4


@pytest.mark.commit
def test_load_v_store():
    """
    Half of struct fields are only loaded and half are only stored, should still result
    in all fields being properly inferred.
    """
    F1 = parse_var("F1")
    some_global = parse_var("some_global")
    constraints = {F1: ConstraintSet()}
    constraints[F1].add(parse_cs("some_global ⊑ A"))
    constraints[F1].add(parse_cs("A.load.σ4@0 ⊑ int"))
    constraints[F1].add(parse_cs("int ⊑ A.store.σ4@4"))
    constraints[F1].add(parse_cs("A.load.σ4@8 ⊑ int"))
    constraints[F1].add(parse_cs("int ⊑ A.store.σ4@12"))
    constraints[F1].add(parse_cs("A.load.σ4@16 ⊑ int"))
    constraints[F1].add(parse_cs("int ⊑ A.store.σ4@20"))
    lattice = DummyLattice()
    lattice_ctypes = DummyLatticeCTypes()
    program = Program(lattice, {some_global}, constraints, {F1: []})
    solver = Solver(program, verbose=VERBOSE_TESTS)
    (gen_const, sketches) = solver()
    # print(sketches[some_global])
    gen = CTypeGenerator(sketches, lattice, lattice_ctypes, 8, 8)
    dtv2type = gen()
    t = dtv2type[some_global]
    assert type(t) == PointerType
    assert type(t.target_type) == StructType
    assert len(t.target_type.fields) == 6
    assert max([f.offset for f in t.target_type.fields]) == 20
    assert min([f.offset for f in t.target_type.fields]) == 0


@pytest.mark.commit
def test_tight_bounds_in():
    constraints = ConstraintSet()
    constraints.add(parse_cs("f.in_0 ⊑ A"))
    constraints.add(parse_cs("int ⊑ A"))
    constraints.add(parse_cs("A ⊑ int"))
    f = parse_var("f")
    program = Program(DummyLattice(), set(), {f: constraints}, {f: {}})
    solver = Solver(program, verbose=LogLevel.DEBUG)
    (gen_const, sketches) = solver()
    f_in1 = parse_var("f.in_0")
    assert sketches[f].lookup(f_in1).upper_bound == CLattice._int
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
    assert isinstance(f_ft, FunctionType)
    assert isinstance(f_ft.params[0], IntType)


@pytest.mark.commit
def test_tight_bounds_out():
    constraints = ConstraintSet()
    constraints.add(parse_cs("A ⊑ f.out"))
    constraints.add(parse_cs("int ⊑ A"))
    constraints.add(parse_cs("A ⊑ int"))
    f = parse_var("f")
    program = Program(DummyLattice(), set(), {f: constraints}, {f: {}})
    solver = Solver(program, verbose=LogLevel.DEBUG)
    (gen_const, sketches) = solver()
    print(gen_const[f])
    print(sketches[f])
    f_out = parse_var("f.out")
    assert sketches[f].lookup(f_out).lower_bound == CLattice._int
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
    assert isinstance(f_ft, FunctionType)
    assert isinstance(f_ft.return_type, IntType)


@pytest.mark.commit
def test_unbound_and_discrete():
    """
    Validate that unbound dereferences colocated with discrete dereferences is typed correctly
    """
    constraints = ConstraintSet()
    constraints.add(parse_cs("F.in_0.load.σ1@0*[nobound] ⊑ int"))
    constraints.add(parse_cs("F.in_0.load.σ1@0 ⊑ char"))
    constraints.add(parse_cs("F.in_0.load.σ1@1 ⊑ char"))

    F = parse_var("F")
    program = Program(DummyLattice(), set(), {F: constraints}, {F: {}})
    solver = Solver(program)
    (gen_const, sketches) = solver()

    gen = CTypeGenerator(
        sketches,
        CLattice(),
        CLatticeCTypes(),
        4,
        4,
    )
    output = gen()
    f_ft = output[F]

    assert isinstance(f_ft, FunctionType)
    assert isinstance(f_ft.params[0], PointerType)
    assert isinstance(f_ft.params[0].target_type, IntType)
    assert f_ft.params[0].target_type.size == 1


@pytest.mark.commit
def test_regression2():
    """
    Extracted from:


    struct linkedlist {
    int value;
    struct linkedlist* next;
    };

    void test_ll(struct linkedlist* ll) {
        if (  ll->value > 2 ) {
                ll->value = 1;
                ll->next = ll->next->next;
        }
        else {
                ll->value = 0;
                ll->next = ll->next->next->next;
        }
    }
    """
    constraints = {
        "test_ll": [
            "RBP_1998 ⊑ RBP_2001",
            "RDX_2034 ⊑ RDX_2042",
            "RAX_2005 ⊑ RAX_2009",
            "RAX_2062 ⊑ RAX_2066",
            "RBP_1998 ⊑ RBP_2048",
            "RAX_2038 ⊑ RAX_2042",
            "RSP_2083 ⊑ RSP_2084",
            "RSP_1997 ⊑ RSP_2083",
            "RAX_2062.load.σ8@8 ⊑ RAX_2062",
            "stack_2001 ⊑ RAX_2005",
            "RBP_1998 ⊑ RBP_2058",
            "RDX_2042 ⊑ RAX_2042.store.σ8@8",
            "stack_2001 ⊑ RAX_2058",
            "test_ll.in_0 ⊑ RDI_2001",
            "stack_2001 ⊑ RAX_2038",
            "RAX_2070.load.σ8@8 ⊑ RDX_2070",
            "RAX_2074 ⊑ test_ll.out",
            "RAX_2066 ⊑ RAX_2070",
            "stack_2001 ⊑ RAX_2016",
            "RAX_2030.load.σ8@8 ⊑ RAX_2030",
            "RAX_2074 ⊑ RAX_2078",
            "RBP_1998 ⊑ RBP_2005",
            "RBP_1998 ⊑ RBP_2074",
            "RAX_2034.load.σ8@8 ⊑ RDX_2034",
            "RAX_2066.load.σ8@8 ⊑ RAX_2066",
            "RAX_2038 ⊑ test_ll.out",
            "RAX_2026 ⊑ RAX_2030",
            "RSP_1997 ⊑ RSP_1998",
            "RAX_2016 ⊑ RAX_2020",
            "RAX_2058 ⊑ RAX_2062",
            "int ⊑ RAX_2052.store.σ4@0",
            "RBP_1998 ⊑ RBP_2038",
            "RDX_2070 ⊑ RDX_2078",
            "RDX_2078 ⊑ RAX_2078.store.σ8@8",
            "RBP_1998 ⊑ RBP_2016",
            "RAX_2048 ⊑ RAX_2052",
            "RDI_2001 ⊑ stack_2001",
            "RAX_2011 ⊑ int",
            "int ⊑ RAX_2020.store.σ4@0",
            "RAX_2009.load.σ4@0 ⊑ RAX_2011",
            "RBP_1998 ⊑ RBP_2026",
            "stack_2001 ⊑ RAX_2048",
            "RFLAGS_2011 ⊑ RFLAGS_2014",
            "stack_2001 ⊑ RAX_2074",
            "RAX_2030 ⊑ RAX_2034",
            "stack_2001 ⊑ RAX_2026",
            "RSP_1998 ⊑ RBP_1998",
        ],
        "caller": ["caller.in_0 <= test_ll.in_0"],
    }

    callgraph = {"caller": ["test_ll"]}
    lattice = CLattice()
    (gen_cs, sketches) = compute_sketches(
        constraints,
        callgraph,
        lattice=lattice,
        config=SolverConfig(),
    )
    # TODO These final constraints contain some redundancy.
    # The three taus are basically equivalent.
    # We should revisit the process of tau variable creation
    # once more to try to avoid these duplicate but equivalent
    # taus.
    assert gen_cs[parse_var("test_ll")] == parse_cs_set(
        [
            "test_ll.in_0 ⊑ test_ll.out",
            "test_ll.in_0.load.σ4@0 ⊑ int",
            "test_ll.in_0 ⊑ τ$0",
            "test_ll.in_0 ⊑ τ$1",
            "τ$0.load.σ8@8 ⊑ τ$0",
            "τ$0.load.σ8@8 ⊑ test_ll.in_0.store.σ8@8",
            "τ$1 ⊑ τ$2",
            "τ$1.load.σ8@8 ⊑ τ$1",
            "τ$2.load.σ8@8 ⊑ τ$2",
            "τ$2.load.σ8@8 ⊑ test_ll.in_0.store.σ8@8",
            "int ⊑ test_ll.in_0.store.σ4@0",
        ]
    )
    gen = CTypeGenerator(sketches, lattice, CLatticeCTypes(), 4, 4)
    dtv2type = gen()
    ll_ptr = dtv2type[DerivedTypeVariable("test_ll")].params[0]
    ll_struct = ll_ptr.target_type
    # the struct has two fields
    assert len(ll_struct.fields) == 2
    # field 0 is an integer
    assert isinstance(ll_struct.fields[0].ctype, IntType)
    # field 1 is a recursive pointer
    assert ll_struct.fields[1].ctype.target_type.name == ll_struct.name

    caller_ptr = dtv2type[DerivedTypeVariable("caller")].params[0]
    caller_struct = caller_ptr.target_type
    # the struct has two fields
    assert len(caller_struct.fields) == 2
    # field 0 is an integer
    assert isinstance(caller_struct.fields[0].ctype, IntType)
    # field 1 is a recursive pointer
    assert caller_struct.fields[1].ctype.target_type.name == caller_struct.name


@pytest.mark.commit
def test_regression3():
    """Extracted from:

    static int do_callback(json_parser *parser, int type)
    {
        if (!parser->callback)
            return 0;
        return (*parser->callback)(parser->userdata, type, NULL, 0);
    }
    """
    constraints = {
        "do_callback": [
            "int64 ⊑ v_205",
            "do_callback.in_0 ⊑ v_128",
            "v_113 ⊑ v_112",
            "v_3.out ⊑ v_136",
            "v_129.load.σ8@40 ⊑ v_3",
            "v_2 ⊑ int64",
            "v_20 ⊑ int64",
            "do_callback.in_1 ⊑ v_129",
            "v_129.load.σ8@48 ⊑ v_63",
            "v_136 ⊑ do_callback.out",
            "v_205 ⊑ v_0",
            "v_63 ⊑ v_3.in_0",
            "v_62 ⊑ int64",
            "v_129 ⊑ int64",
            "v_112 ⊑ do_callback.out",
            "bool ⊑ v_18",
            "int64 ⊑ v_206",
            "v_206 ⊑ v_60",
            "v_3 ⊑ int64",
        ],
    }

    (gen_const, sketches) = compute_sketches(
        constraints, {"do_callback": {}}, CLattice()
    )

    assert gen_const[parse_var("do_callback")] == parse_cs_set(
        [
            "do_callback.in_1.load.σ8@40.out ⊑ do_callback.out",
            "do_callback.in_1.load.σ8@48 ⊑ do_callback.in_1.load.σ8@40.in_0",
            "do_callback.in_1 ⊑ int64",
            "do_callback.in_1.load.σ8@40 ⊑ int64",
        ]
    )

    gen = CTypeGenerator(sketches, CLattice(), CLatticeCTypes(), 4, 4)
    dtv2type = gen()

    types = dtv2type[parse_var("do_callback")]
    assert isinstance(types, FunctionType)
    assert isinstance(types.params[1], PointerType)
    assert isinstance(types.params[1].target_type, StructType)
    fields = list(types.params[1].target_type.fields)
    func_field = fields[0]
    assert func_field.offset == 40
    ctype = func_field.ctype
    assert isinstance(ctype, PointerType)
    assert isinstance(ctype.target_type, FunctionType)
