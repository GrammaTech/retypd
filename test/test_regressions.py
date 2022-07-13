from test_endtoend import (
    compute_sketches,
    all_solver_configs,
    parse_var,
)
import pytest
from retypd import CLattice


@all_solver_configs
@pytest.mark.commit
def test_regression3(config):
    """
    This test checks that infer_shapes
    correctly unifies 'a.load' and 'a.store'
    for every DTV 'a'. This is a special case
    of combining the S-Refl nand S-Pointer
    type rules.
    """
    constraints = {
        "prvListTasksWithinSingleList": [
            "v_566 ⊑ v_994",
            "v_181 ⊑ int32",
            "v_451.load.σ4@4 ⊑ v_86",
            "v_287 ⊑ int32",
            "v_281.load.σ4@12 ⊑ v_995",
            "v_572 ⊑ v_368",
            "v_211.load.σ4@4 ⊑ v_217",
            "v_807 ⊑ int32",
            "v_451.load.σ4@4 ⊑ v_211",
            "v_451.load.σ4@0 ⊑ v_69",
            "int32 ⊑ v_354",
            "v_991 ⊑ v_451.store.σ4@4",
            "bool ⊑ v_132",
            "bool ⊑ v_333",
            "v_175.load.σ4@12 ⊑ v_992",
            "v_971.load.σ4@4 ⊑ v_262",
            "v_217 ⊑ v_993",
            "bool ⊑ v_238",
            "v_92 ⊑ int32",
            "v_992 ⊑ v_181",
            "v_569 ⊑ v_991",
            "v_185 ⊑ v_184",
            "v_451.load.σ4@4 ⊑ v_281",
            "v_953.load.σ4@4 ⊑ v_156",
            "v_287 ⊑ vTaskGetInfo.in_0",
            "v_111 ⊑ int32",
            "v_293 ⊑ vTaskGetInfo.in_1",
            "v_80 ⊑ int32",
            "v_994 ⊑ v_451.store.σ4@4",
            "bool ⊑ v_78",
            "prvListTasksWithinSingleList.in_0 ⊑ v_450",
            "v_993 ⊑ v_451.store.σ4@4",
            "v_92 ⊑ v_989",
            "v_217 ⊑ int32",
            "v_989 ⊑ v_451.store.σ4@4",
            "v_69 ⊑ int32",
            "v_995 ⊑ v_287",
            "v_575 ⊑ int32",
            "prvListTasksWithinSingleList.in_1 ⊑ v_451",
            "v_451.load.σ4@4 ⊑ v_175",
            "prvListTasksWithinSingleList.in_2 ⊑ v_452",
            "v_86.load.σ4@4 ⊑ v_92",
            "v_990 ⊑ v_111",
            "v_368 ⊑ prvListTasksWithinSingleList.out",
        ]
    }
    callgraph = {"prvListTasksWithinSingleList": []}
    lattice = CLattice()
    (gen_cs, sketches) = compute_sketches(
        constraints,
        callgraph,
        lattice=lattice,
        config=config,
    )
    sk = sketches[parse_var("prvListTasksWithinSingleList")]
    # The sketch has a cycle thanks to the equivalence between
    # prvListTasksWithinSingleList.in_1.store.σ4@4 and
    # prvListTasksWithinSingleList.in_1.load.σ4@4
    assert sk.lookup(
        parse_var("prvListTasksWithinSingleList.in_1.load.σ4@4.load.σ4@4")
    ) is sk.lookup(parse_var("prvListTasksWithinSingleList.in_1.load.σ4@4"))
    assert sk.lookup(
        parse_var("prvListTasksWithinSingleList.in_1.store.σ4@4.load.σ4@4")
    ) is sk.lookup(parse_var("prvListTasksWithinSingleList.in_1.store.σ4@4"))


@pytest.mark.commit
@all_solver_configs
def test_regression4(config):
    constraints = {
        "b": [
            "RSP_1735 ⊑ int",
            "RBX_1732 ⊑ int",
            "RAX_1719.load.σ4@0 ⊑ RDX_1723",
            "int ⊑ RSP_1711",
            "RBP_1707.load.σ8@-24 ⊑ RAX_1725",
            "RAX_1729 ⊑ int",
            "RSP_1710 ⊑ int",
            "b.in_0 ⊑ RDI_1715",
            "int ⊑ RSP_1742",
            "RBP_1707.load.σ8@-24 ⊑ RAX_1719",
            "RDI_1715 ⊑ RBP_1707.store.σ8@-24",
            "RAX_1740 ⊑ b.out",
            "RAX_1725.load.σ4@4 ⊑ RAX_1729",
            "int ⊑ RAX_1740",
        ],
        "a": [
            "RAX_1771 ⊑ a.out",
            "stack_1757 ⊑ RAX_1761",
            "RSP_1749 ⊑ int",
            "RDI_1775 ⊑ b.in_0",
            "RDI_1757 ⊑ stack_1757",
            "int ⊑ RSP_1753",
            "a.in_0 ⊑ b.in_0",
            "int ⊑ RAX_1761.store.σ4@0",
            "a.in_0 ⊑ RDI_1757",
            "stack_1757 ⊑ RAX_1771",
        ],
        "main": [
            "RAX_1808 ⊑ main.out",
            "RSP_1785 ⊑ int",
            "int ⊑ RSP_1789",
            "RDX_1820 ⊑ uint",
            "RAX_1802 ⊑ stack_1802",
            "stack_1802 ⊑ RDX_1820",
            "uint ⊑ RDX_1824",
            "RDI_1812 ⊑ a.in_0",
        ],
    }
    callgraph = {"a": {"b"}, "main": {"a", "FUN_570"}, "b": {"FUN_580"}}
    (gen_const, sketches) = compute_sketches(
        constraints, callgraph, CLattice(), config
    )
