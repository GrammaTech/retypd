"""Simple unit tests from the paper and slides that only look at the final result (sketches)
"""

import pytest
from retypd import (
    ConstraintSet,
    SchemaParser,
)

from retypd.schema import DerefLabel, InLabel, LoadLabel, OutLabel, Variance
from retypd.graph import ConstraintGraph, EdgeLabel, SideMark, Node

VERBOSE_TESTS = False


@pytest.mark.commit
def test_simple():
    """
    Check that the constraint graph from one constraint has the expected elements.
    A constraint graph from one constraint has two paths that allow us to reconstruct
    the constraint, one in covariant version and one in contravariant.
    """
    cs = ConstraintSet(
        [SchemaParser.parse_constraint("f.in_0 <= A.load.σ4@0")]
    )
    graph = ConstraintGraph(
        cs, {SchemaParser.parse_variable("f")}, keep_graph_before_split=True
    ).graph_before_split
    f_co = Node(
        SchemaParser.parse_variable("f"), Variance.COVARIANT, SideMark.RIGHT
    )
    fin0_co = Node(
        SchemaParser.parse_variable("f.in_0"),
        Variance.COVARIANT,
        SideMark.LEFT,
    )
    a_load_0_co = Node(
        SchemaParser.parse_variable("A.load.σ4@0"), Variance.COVARIANT
    )
    a_load_co = Node(SchemaParser.parse_variable("A.load"), Variance.COVARIANT)
    a_co = Node(SchemaParser.parse_variable("A"), Variance.COVARIANT)

    f_cn = Node(
        SchemaParser.parse_variable("f"),
        Variance.CONTRAVARIANT,
        SideMark.LEFT,
    )
    fin0_cn = Node(
        SchemaParser.parse_variable("f.in_0"),
        Variance.CONTRAVARIANT,
        SideMark.RIGHT,
    )
    a_load_0_cn = Node(
        SchemaParser.parse_variable("A.load.σ4@0"), Variance.CONTRAVARIANT
    )
    a_load_cn = Node(
        SchemaParser.parse_variable("A.load"), Variance.CONTRAVARIANT
    )
    a_cn = Node(SchemaParser.parse_variable("A"), Variance.CONTRAVARIANT)
    assert {
        f_co,
        fin0_co,
        a_load_0_co,
        a_load_co,
        a_co,
        f_cn,
        fin0_cn,
        a_load_0_cn,
        a_load_cn,
        a_cn,
    } == set(graph.nodes)
    forget = EdgeLabel.Kind.FORGET
    recall = EdgeLabel.Kind.RECALL
    edges = {
        # one path from "f" to "A"
        (f_cn, fin0_co, EdgeLabel(InLabel(0), recall)),
        (fin0_co, a_load_0_co, None),
        (a_load_0_co, a_load_co, EdgeLabel(DerefLabel(4, 0), forget)),
        (a_load_co, a_co, EdgeLabel(LoadLabel.instance(), forget)),
        # the second path from "A" to "f"
        (a_cn, a_load_cn, EdgeLabel(LoadLabel.instance(), recall)),
        (a_load_cn, a_load_0_cn, EdgeLabel(DerefLabel(4, 0), recall)),
        (a_load_0_cn, fin0_cn, None),
        (fin0_cn, f_co, EdgeLabel(InLabel(0), forget)),
    }
    assert edges == set(graph.edges(data="label"))


@pytest.mark.commit
def test_two_constraints():
    """
    A constraint graph from two related constraints has two paths
    (a covariant and contravariant version) that allow us to conclude
    that A.out <= C.
    """
    constraints = ["A <= B", "B.out <= C"]
    cs = ConstraintSet(map(SchemaParser.parse_constraint, constraints))
    graph = ConstraintGraph(
        cs,
        {SchemaParser.parse_variable("A"), SchemaParser.parse_variable("C")},
        keep_graph_before_split=True,
    ).graph_before_split
    b_co = Node(SchemaParser.parse_variable("B"), Variance.COVARIANT)
    b_out_co = Node(SchemaParser.parse_variable("B.out"), Variance.COVARIANT)
    a_co = Node(
        SchemaParser.parse_variable("A"), Variance.COVARIANT, SideMark.LEFT
    )
    c_co = Node(
        SchemaParser.parse_variable("C"), Variance.COVARIANT, SideMark.RIGHT
    )
    b_cn = Node(SchemaParser.parse_variable("B"), Variance.CONTRAVARIANT)
    b_out_cn = Node(
        SchemaParser.parse_variable("B.out"), Variance.CONTRAVARIANT
    )
    a_cn = Node(
        SchemaParser.parse_variable("A"),
        Variance.CONTRAVARIANT,
        SideMark.RIGHT,
    )
    c_cn = Node(
        SchemaParser.parse_variable("C"),
        Variance.CONTRAVARIANT,
        SideMark.LEFT,
    )
    forget = EdgeLabel.Kind.FORGET
    recall = EdgeLabel.Kind.RECALL
    edges = {
        # path from A to C
        (a_co, b_co, None),
        (b_co, b_out_co, EdgeLabel(OutLabel.instance(), recall)),
        (b_out_co, c_co, None),
        # path from C to A
        (c_cn, b_out_cn, None),
        (b_out_cn, b_cn, EdgeLabel(OutLabel.instance(), forget)),
        (b_cn, a_cn, None),
    }
    assert edges == set(graph.edges(data="label"))
