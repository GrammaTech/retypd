import pytest
from retypd import (
    Sketch,
    DerivedTypeVariable,
    CLattice,
)
from retypd.sketches import LabelNode
from retypd.parser import SchemaParser
from typing import Dict


def sketch_from_dict(root: str, tree: Dict[str, Dict[str, str]]):
    """
    Utility to build sketches from string dictionaries.
    """
    sk = Sketch(DerivedTypeVariable(root), CLattice())
    # transform to DTV
    dtv_tree = {}
    for k, succs in tree.items():
        dtv_succs = {}
        for label, succ in succs.items():
            dtv_succs[
                SchemaParser.parse_label(label)
            ] = SchemaParser.parse_variable(succ)
        dtv_tree[SchemaParser.parse_variable(k)] = dtv_succs

    for node in dtv_tree:
        sk.make_node(node)
    for dtv, succs in dtv_tree.items():
        for label, succ in succs.items():
            tail = dtv.get_suffix(succ)
            node = sk.lookup(dtv)
            if tail is not None:
                succ_node = sk.lookup(succ)
                if succ_node is None:
                    succ_node = sk.make_node(succ)
                sk.add_edge(node, succ_node, label)
            else:
                label_node = LabelNode(succ)
                sk.add_edge(node, label_node, label)
    return sk


@pytest.mark.commit
def test_join_sketch():
    """
    Test that joining recursive and non-recursive
    results in non-recursive sketch.
    """
    sk1 = sketch_from_dict(
        "f",
        {
            "f": {"in_0": "f.in_0"},
            "f.in_0": {"load": "f.in_0.load"},
            "f.in_0.load": {
                "σ8@0": "f.in_0",
                "σ8@4": "f.in_0.load.σ8@4",
            },
        },
    )
    sk2 = sketch_from_dict(
        "f",
        {
            "f": {"in_0": "f.in_0"},
            "f.in_0": {"load": "f.in_0.load"},
            "f.in_0.load": {
                "σ8@4": "f.in_0.load.σ8@4",
            },
        },
    )
    sk1.join(sk2)
    assert sk1.lookup(SchemaParser.parse_variable("f.in_0.load.σ8@0")) is None
