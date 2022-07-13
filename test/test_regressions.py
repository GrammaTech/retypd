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
