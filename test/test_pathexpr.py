from retypd.pathexpr import path_expression_between
import networkx
import unittest


def generate_test(edges, path_exprs):
    """ Generate a unit test for a given set of edges and expected path 
    expressions
    """
    graph = networkx.DiGraph()

    for (src, dest, label) in edges:
        graph.add_edge(src, dest, label=label)

    def generated(self):
        for (source, dest, expr) in path_exprs:
            for decompose in (True, False):
                generated = path_expression_between(
                    graph, 'label', source, dest, decompose
                )

                self.assertEqual(
                    str(generated),
                    expr,
                    msg=(
                        f'Expected edge from {source} to {dest} to be {expr}, '
                        f'but found {generated} '
                        f'({"with" if decompose else "without"} SCC ' 
                        f'decomposition)'
                    )
                )

    return generated


class PathExprTest(unittest.TestCase):    
    test_simple = generate_test(
        [
            ('a', 'b', 'A'),
            ('b', 'c', 'B'),
            ('c', 'd', 'C'),
        ],
        [
            ('a', 'b', 'A'),
            ('a', 'c', '(A . B)'),
            ('a', 'd', '((A . B) . C)'),
            ('b', 'c', 'B'),
            ('b', 'd', '(B . C)'),
        ]
    )

    test_self_cycle = generate_test(
        [
            ('a', 'b', 'A'),
            ('b', 'b', 'B'),
            ('b', 'c', 'C'),
        ],
        [
            ('a', 'b', '(A . B*)'),
            ('a', 'c', '((A . B*) . C)'),
        ]
    )

    test_self_cycle_with_shortcut = generate_test(
        [
            ('a', 'b', 'A'),
            ('b', 'b', 'B'),
            ('b', 'c', 'C'),
            ('a', 'c', 'D')
        ],
        [
            ('a', 'b', '(A . B*)'),
            ('a', 'c', '(D U ((A . B*) . C))'),
        ]
    )

    test_multiple_cycles = generate_test(
        [
            ('a', 'b', 'A'),
            ('b', 'b', 'B'),
            ('b', 'c', 'C'),
            ('a', 'd', 'D'),
            ('d', 'd', 'E'),
            ('d', 'c', 'F'),
        ],
        [
            ('a', 'b', '(A . B*)'),
            ('a', 'd', '(D . E*)'),
            ('a', 'c', '(((A . B*) . C) U ((D . E*) . F))'),
        ]
    )

    test_whole_graph_cycle = generate_test(
        [
            ('a', 'b', 'A'),
            ('b', 'c', 'B'),
            ('c', 'a', 'C'),
        ],
        [
            ('a', 'b', '(A U (((A . B) . ((C . A) . B)*) . (C . A)))'),
            ('b', 'c', '(B . ((C . A) . B)*)'),
            ('a', 'c', '((A . B) . ((C . A) . B)*)'),
        ]
    )
