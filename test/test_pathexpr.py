import pytest
from retypd.pathexpr import path_expression_between
import networkx


@pytest.mark.parametrize(
    ("edges", "path_exprs"),
    [
        (
            [
                ("a", "b", "A"),
                ("b", "c", "B"),
                ("c", "d", "C"),
            ],
            [
                ("a", "b", "A"),
                ("a", "c", "(A . B)"),
                ("a", "d", "((A . B) . C)"),
                ("b", "c", "B"),
                ("b", "d", "(B . C)"),
            ],
        ),
        (
            [
                ("a", "b", "A"),
                ("b", "b", "B"),
                ("b", "c", "C"),
            ],
            [
                ("a", "b", "(A . B*)"),
                ("a", "c", "((A . B*) . C)"),
            ],
        ),
        (
            [
                ("a", "b", "A"),
                ("b", "b", "B"),
                ("b", "c", "C"),
                ("a", "c", "D"),
            ],
            [
                ("a", "b", "(A . B*)"),
                ("a", "c", "(D U ((A . B*) . C))"),
            ],
        ),
        (
            [
                ("a", "b", "A"),
                ("b", "b", "B"),
                ("b", "c", "C"),
                ("a", "d", "D"),
                ("d", "d", "E"),
                ("d", "c", "F"),
            ],
            [
                ("a", "b", "(A . B*)"),
                ("a", "d", "(D . E*)"),
                ("a", "c", "(((A . B*) . C) U ((D . E*) . F))"),
            ],
        ),
        (
            [
                ("a", "b", "A"),
                ("b", "c", "B"),
                ("c", "a", "C"),
            ],
            [
                ("a", "b", "(A U (((A . B) . ((C . A) . B)*) . (C . A)))"),
                ("b", "c", "(B . ((C . A) . B)*)"),
                ("a", "c", "((A . B) . ((C . A) . B)*)"),
            ],
        ),
        (
            [("a", "b", "A"), ("b", "c", None)],
            [("a", "b", "A"), ("a", "c", "A")],
        ),
        (
            [
                ("a", "b", "A"),
                ("b", "c", "B"),
                ("c", "a", None),
            ],
            [
                ("a", "b", "(A U (((A . B) . (A . B)*) . A))"),
                ("a", "c", "((A . B) . (A . B)*)"),
            ],
        ),
        (
            [
                ("a", "b", "A"),
                ("b", "b", None),
                ("b", "c", "B"),
            ],
            [("a", "b", "A"), ("a", "c", "(A . B)")],
        ),
    ],
    ids=[
        "simple",
        "self_cycle",
        "self_cycle_with_shortcut",
        "multiple_cycles",
        "whole_graph_cycle",
        "empty_label",
        "empty_loop",
        "empty_self_loop",
    ],
)
@pytest.mark.parametrize("decompose", [True, False])
@pytest.mark.commit
def test_path_expr(edges, path_exprs, decompose):
    """Generate a unit test for a given set of edges and expected path
    expressions
    """
    graph = networkx.DiGraph()

    for (src, dest, label) in edges:
        if label is not None:
            graph.add_edge(src, dest, label=label)
        else:
            graph.add_edge(src, dest)

    for (source, dest, expr) in path_exprs:
        generated = path_expression_between(
            graph, "label", source, dest, decompose
        )
        assert str(generated) == expr
