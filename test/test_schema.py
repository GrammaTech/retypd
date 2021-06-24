from abc import ABC
import re
import unittest

from type_inference import ConstraintSet, DerefLabel, DerivedTypeVariable, \
        ForgetLabel, InLabel, LoadLabel, OutLabel, RecallLabel, Solver, StoreLabel, \
        SubtypeConstraint, Vertex, ConstraintGraph

class SchemaTestHelper:
    '''Static helper functions for Schema tests. Since this parsing code is unlikely to be useful in
    the code itself, it is included here.
    '''

    subtype_pattern = re.compile('([^ ]*) ⊑ ([^ ]*)')
    in_pattern = re.compile('in_([0-9]+)')
    deref_pattern = re.compile('σ([0-9]+)@([0-9]+)')
    node_pattern = re.compile(r'([^ ]+)\.([⊕⊖])')
    edge_pattern = re.compile(r'(\S+)\s+→\s+(\S+)(\s+\((forget|recall) ([^ ]*)\))?')

    @staticmethod
    def parse_label(label):
        if label == 'load':
            return LoadLabel.instance()
        if label == 'store':
            return StoreLabel.instance()
        if label == 'out':
            return OutLabel.instance()
        in_match = SchemaTestHelper.in_pattern.match(label)
        if in_match:
            return InLabel(int(in_match.group(1)))
        deref_match = SchemaTestHelper.deref_pattern.match(label)
        if deref_match:
            return DerefLabel(int(deref_match.group(1)), int(deref_match.group(2)))
        raise ValueError

    @staticmethod
    def parse_variable(var):
        components = var.split('.')
        return DerivedTypeVariable(components[0], map(SchemaTestHelper.parse_label, components[1:]))

    @staticmethod
    def parse_constraint(constraint):
        subtype_match = SchemaTestHelper.subtype_pattern.match(constraint)
        if subtype_match:
            return SubtypeConstraint(SchemaTestHelper.parse_variable(subtype_match.group(1)),
                                     SchemaTestHelper.parse_variable(subtype_match.group(2)))
        raise ValueError

    @staticmethod
    def parse_node(node):
        node_match = SchemaTestHelper.node_pattern.match(node)
        if node_match:
            var = SchemaTestHelper.parse_variable(node_match.group(1))
            return Vertex(var, node_match.group(2) == '⊕')
        raise ValueError

    @staticmethod
    def parse_edge(edge):
        edge_match = SchemaTestHelper.edge_pattern.match(edge)
        if edge_match:
            sub = SchemaTestHelper.parse_node(edge_match.group(1))
            sup = SchemaTestHelper.parse_node(edge_match.group(2))
            atts = {}
            if edge_match.group(3):
                capability = SchemaTestHelper.parse_label(edge_match.group(5))
                if edge_match.group(4) == 'forget':
                    label = ForgetLabel(capability)
                elif edge_match.group(4) == 'recall':
                    label = RecallLabel(capability)
                else:
                    raise ValueError
                atts['label'] = label
            return (sub, sup, atts)
        raise ValueError

    @staticmethod
    def edges_to_dict(edges):
        graph = {}
        for edge in edges:
            (f, t, atts) = SchemaTestHelper.parse_edge(edge)
            graph[(f, t)] = atts
        return graph


class SchemaTest(ABC):
    def graphs_are_equal(self, graph, edge_set) -> bool:
        edges = graph.edges()
        self.assertEqual(len(edges), len(edge_set))
        for edge in edges:
            (f, t) = edge
            self.assertTrue(edge in edge_set)
            self.assertEqual(graph[f][t], edge_set[edge])


class BasicSchemaTest(SchemaTest, unittest.TestCase):

    def test_simple_constraints(self):
        '''A simple test from the paper (the right side of Figure 4 on p. 6). This one has no
        recursive data structures; as such, the fixed point would suffice. However, we compute type
        constraints in the same way as in the presence of recursion.
        '''

        constraints = ConstraintSet()
        p = DerivedTypeVariable('p')
        q = DerivedTypeVariable('q')
        x = DerivedTypeVariable('x')
        y = DerivedTypeVariable('y')
        q_store = DerivedTypeVariable('q', [StoreLabel.instance(), DerefLabel(4, 0)])
        p_load = DerivedTypeVariable('p', [LoadLabel.instance(), DerefLabel(4, 0)])
        constraints.add_subtype(p, q)
        constraints.add_subtype(x, q_store)
        constraints.add_subtype(p_load, y)

        graph = constraints.generate_graph()

        graph.add_forget_recall()

        forget_recall = ['p.load.⊕        →  p.⊕              (forget load)',
                         'p.load.⊖        →  p.⊖              (forget load)',
                         'p.load.⊕        →  p.load.σ4@0.⊕    (recall σ4@0)',
                         'p.load.⊖        →  p.load.σ4@0.⊖    (recall σ4@0)',
                         'p.load.σ4@0.⊕   →  p.load.⊕         (forget σ4@0)',
                         'p.load.σ4@0.⊖   →  p.load.⊖         (forget σ4@0)',
                         'p.load.σ4@0.⊕   →  y.⊕',
                         'p.⊕             →  p.load.⊕         (recall load)',
                         'p.⊖             →  p.load.⊖         (recall load)',
                         'p.⊕             →  q.⊕',
                         'q.⊖             →  p.⊖',
                         'q.⊕             →  q.store.⊖        (recall store)',
                         'q.⊖             →  q.store.⊕        (recall store)',
                         'q.store.⊕       →  q.⊖              (forget store)',
                         'q.store.⊖       →  q.⊕              (forget store)',
                         'q.store.⊕       →  q.store.σ4@0.⊕   (recall σ4@0)',
                         'q.store.⊖       →  q.store.σ4@0.⊖   (recall σ4@0)',
                         'q.store.σ4@0.⊕  →  q.store.⊕        (forget σ4@0)',
                         'q.store.σ4@0.⊖  →  q.store.⊖        (forget σ4@0)',
                         'q.store.σ4@0.⊖  →  x.⊖',
                         'x.⊕             →  q.store.σ4@0.⊕',
                         'y.⊖             →  p.load.σ4@0.⊖']

        forget_recall_graph = SchemaTestHelper.edges_to_dict(forget_recall)
        self.graphs_are_equal(graph.graph, forget_recall_graph)

        graph.saturate()

        saturated = ['p.load.⊕        →  p.⊕              (forget load)',
                     'p.load.⊖        →  p.⊖              (forget load)',
                     'p.load.⊕        →  p.load.⊕',
                     'p.load.⊖        →  p.load.⊖',
                     'p.load.⊕        →  p.load.σ4@0.⊕    (recall σ4@0)',
                     'p.load.⊖        →  p.load.σ4@0.⊖    (recall σ4@0)',
                     'p.load.⊖        →  q.store.⊖',
                     'p.load.σ4@0.⊕   →  p.load.⊕         (forget σ4@0)',
                     'p.load.σ4@0.⊖   →  p.load.⊖         (forget σ4@0)',
                     'p.load.σ4@0.⊕   →  p.load.σ4@0.⊕',
                     'p.load.σ4@0.⊖   →  p.load.σ4@0.⊖',
                     'p.load.σ4@0.⊖   →  q.store.σ4@0.⊖',
                     'p.load.σ4@0.⊕   →  y.⊕',
                     'p.⊕             →  p.load.⊕         (recall load)',
                     'p.⊖             →  p.load.⊖         (recall load)',
                     'p.⊕             →  q.⊕',
                     'q.⊖             →  p.⊖',
                     'q.⊕             →  q.store.⊖        (recall store)',
                     'q.⊖             →  q.store.⊕        (recall store)',
                     'q.store.⊕       →  p.load.⊕',
                     'q.store.⊖       →  q.⊕              (forget store)',
                     'q.store.⊕       →  q.⊖              (forget store)',
                     'q.store.⊖       →  q.store.⊖',
                     'q.store.⊕       →  q.store.⊕',
                     'q.store.⊖       →  q.store.σ4@0.⊖   (recall σ4@0)',
                     'q.store.⊕       →  q.store.σ4@0.⊕   (recall σ4@0)',
                     'q.store.σ4@0.⊕  →  p.load.σ4@0.⊕',
                     'q.store.σ4@0.⊕  →  q.store.⊕        (forget σ4@0)',
                     'q.store.σ4@0.⊖  →  q.store.⊖        (forget σ4@0)',
                     'q.store.σ4@0.⊕  →  q.store.σ4@0.⊕',
                     'q.store.σ4@0.⊖  →  q.store.σ4@0.⊖',
                     'q.store.σ4@0.⊖  →  x.⊖',
                     'x.⊕             →  q.store.σ4@0.⊕',
                     'y.⊖             →  p.load.σ4@0.⊖']

        saturated_graph = SchemaTestHelper.edges_to_dict(saturated)
        self.graphs_are_equal(graph.graph, saturated_graph)

        solver = Solver(graph, {x, y})
        final_constraints = solver()

        self.assertTrue(SubtypeConstraint(x, y) in final_constraints)

    def test_other_simple_constraints(self):
        '''Another simple test from the paper (the program modeled in Figure 14 on p. 26).
        '''

        constraints = ConstraintSet()
        p = DerivedTypeVariable('p')
        A = DerivedTypeVariable('A')
        B = DerivedTypeVariable('B')
        x = DerivedTypeVariable('x')
        y = DerivedTypeVariable('y')
        x_store = DerivedTypeVariable('x', [StoreLabel.instance()])
        y_load = DerivedTypeVariable('y', [LoadLabel.instance()])
        constraints.add_subtype(y, p)
        constraints.add_subtype(p, x)
        constraints.add_subtype(A, x_store)
        constraints.add_subtype(y_load, B)

        graph = constraints.generate_graph()

        graph.add_forget_recall()

        forget_recall = ['A.⊕        →  x.store.⊕',
                         'B.⊖        →  y.load.⊖',
                         'p.⊕        →  x.⊕',
                         'p.⊖        →  y.⊖',
                         'x.⊖        →  p.⊖',
                         'x.store.⊖  →  A.⊖',
                         'x.store.⊕  →  x.⊖         (forget store)',
                         'x.store.⊖  →  x.⊕         (forget store)',
                         'x.⊕        →  x.store.⊖   (recall store)',
                         'x.⊖        →  x.store.⊕   (recall store)',
                         'y.load.⊕   →  B.⊕',
                         'y.load.⊕   →  y.⊕         (forget load)',
                         'y.load.⊖   →  y.⊖         (forget load)',
                         'y.⊕        →  p.⊕',
                         'y.⊕        →  y.load.⊕    (recall load)',
                         'y.⊖        →  y.load.⊖    (recall load)']

        forget_recall_graph = SchemaTestHelper.edges_to_dict(forget_recall)
        self.graphs_are_equal(graph.graph, forget_recall_graph)

        graph.saturate()

        saturated = ['A.⊕        →  x.store.⊕',
                     'B.⊖        →  y.load.⊖',
                     'p.⊕        →  x.⊕',
                     'p.⊖        →  y.⊖',
                     'x.⊖        →  p.⊖',
                     'x.store.⊖  →  A.⊖',
                     'x.store.⊕  →  x.⊖         (forget store)',
                     'x.store.⊖  →  x.⊕         (forget store)',
                     'x.store.⊕  →  x.store.⊕',
                     'x.store.⊖  →  x.store.⊖',
                     'x.store.⊕  →  y.load.⊕',
                     'x.⊕        →  x.store.⊖   (recall store)',
                     'x.⊖        →  x.store.⊕   (recall store)',
                     'y.load.⊕   →  B.⊕',
                     'y.load.⊖   →  x.store.⊖',
                     'y.load.⊕   →  y.⊕         (forget load)',
                     'y.load.⊖   →  y.⊖         (forget load)',
                     'y.load.⊕   →  y.load.⊕',
                     'y.load.⊖   →  y.load.⊖',
                     'y.⊕        →  p.⊕',
                     'y.⊕        →  y.load.⊕    (recall load)',
                     'y.⊖        →  y.load.⊖    (recall load)']

        saturated_graph = SchemaTestHelper.edges_to_dict(saturated)
        self.graphs_are_equal(graph.graph, saturated_graph)

        solver = Solver(graph, {A, B})
        final_constraints = solver()

        self.assertTrue(SubtypeConstraint(A, B) in final_constraints)


class RecursiveSchemaTest(SchemaTest, unittest.TestCase):
    def test_recursive(self):
        '''A test based on the running example from the paper (Figure 2 on p. 3) and the slides
        (slides 67-83, labeled as slides 13-15).
        '''
        constraints = ConstraintSet()
        F = DerivedTypeVariable('F')
        δ = DerivedTypeVariable('δ')
        φ = DerivedTypeVariable('φ')
        α = DerivedTypeVariable('α')
        α_prime = DerivedTypeVariable("α'")
        close_in = DerivedTypeVariable('close', [InLabel(0)])
        close_out = DerivedTypeVariable('close', [OutLabel.instance()])
        F_in = DerivedTypeVariable('F', [InLabel(0)])
        F_out = DerivedTypeVariable('F', [OutLabel.instance()])
        φ_load_0 = DerivedTypeVariable('φ', [LoadLabel.instance(), DerefLabel(4, 0)])
        φ_load_4 = DerivedTypeVariable('φ', [LoadLabel.instance(), DerefLabel(4, 4)])
        FileDescriptor = DerivedTypeVariable('#FileDescriptor')
        SuccessZ = DerivedTypeVariable('#SuccessZ')
        constraints.add_subtype(F_in, δ)
        constraints.add_subtype(α, φ)
        constraints.add_subtype(δ, φ)
        constraints.add_subtype(φ_load_0, α)
        constraints.add_subtype(φ_load_4, α_prime)
        constraints.add_subtype(α_prime, close_in)
        constraints.add_subtype(close_out, F_out)
        constraints.add_subtype(close_in, FileDescriptor)
        constraints.add_subtype(SuccessZ, close_out)

        graph = constraints.generate_graph()

        graph.add_forget_recall()

        forget_recall = ["close.⊕            →  close.in_0.⊖        (recall in_0)",
                         "close.⊖            →  close.in_0.⊕        (recall in_0)",
                         "close.⊕            →  close.out.⊕         (recall out)",
                         "close.⊖            →  close.out.⊖         (recall out)",
                         "close.in_0.⊕       →  close.⊖             (forget in_0)",
                         "close.in_0.⊖       →  close.⊕             (forget in_0)",
                         "close.in_0.⊕       →  #FileDescriptor.⊕",
                         "close.in_0.⊖       →  α'.⊖",
                         "close.out.⊕        →  close.⊕             (forget out)",
                         "close.out.⊖        →  close.⊖             (forget out)",
                         "close.out.⊕        →  F.out.⊕",
                         "close.out.⊖        →  #SuccessZ.⊖",
                         "F.⊖                →  F.in_0.⊕            (recall in_0)",
                         "F.⊕                →  F.in_0.⊖            (recall in_0)",
                         "F.⊖                →  F.out.⊖             (recall out)",
                         "F.⊕                →  F.out.⊕             (recall out)",
                         "#FileDescriptor.⊖  →  close.in_0.⊖",
                         "F.in_0.⊕           →  F.⊖                 (forget in_0)",
                         "F.in_0.⊖           →  F.⊕                 (forget in_0)",
                         "F.in_0.⊕           →  δ.⊕",
                         "F.out.⊖            →  close.out.⊖",
                         "F.out.⊕            →  F.⊕                 (forget out)",
                         "F.out.⊖            →  F.⊖                 (forget out)",
                         "#SuccessZ.⊕        →  close.out.⊕",
                         "α'.⊕               →  close.in_0.⊕",
                         "α.⊕                →  φ.⊕",
                         "α.⊖                →  φ.load.σ4@0.⊖",
                         "α'.⊖               →  φ.load.σ4@4.⊖",
                         "δ.⊖                →  F.in_0.⊖",
                         "δ.⊕                →  φ.⊕",
                         "φ.load.σ4@0.⊕      →  α.⊕",
                         "φ.load.σ4@0.⊕      →  φ.load.⊕            (forget σ4@0)",
                         "φ.load.σ4@0.⊖      →  φ.load.⊖            (forget σ4@0)",
                         "φ.load.σ4@4.⊕      →  α'.⊕",
                         "φ.load.σ4@4.⊕      →  φ.load.⊕            (forget σ4@4)",
                         "φ.load.σ4@4.⊖      →  φ.load.⊖            (forget σ4@4)",
                         "φ.load.⊖           →  φ.⊖                 (forget load)",
                         "φ.load.⊕           →  φ.⊕                 (forget load)",
                         "φ.load.⊖           →  φ.load.σ4@0.⊖       (recall σ4@0)",
                         "φ.load.⊕           →  φ.load.σ4@0.⊕       (recall σ4@0)",
                         "φ.load.⊖           →  φ.load.σ4@4.⊖       (recall σ4@4)",
                         "φ.load.⊕           →  φ.load.σ4@4.⊕       (recall σ4@4)",
                         "φ.⊖                →  α.⊖",
                         "φ.⊖                →  δ.⊖",
                         "φ.⊕                →  φ.load.⊕            (recall load)",
                         "φ.⊖                →  φ.load.⊖            (recall load)"]
        self.graphs_are_equal(graph.graph, SchemaTestHelper.edges_to_dict(forget_recall))

        graph.saturate()

        saturated = ["close.⊖            →  close.in_0.⊕        (recall in_0)",
                     "close.⊕            →  close.in_0.⊖        (recall in_0)",
                     "close.⊖            →  close.out.⊖         (recall out)",
                     "close.⊕            →  close.out.⊕         (recall out)",
                     "close.in_0.⊕       →  close.⊖             (forget in_0)",
                     "close.in_0.⊖       →  close.⊕             (forget in_0)",
                     "close.in_0.⊕       →  close.in_0.⊕",
                     "close.in_0.⊖       →  close.in_0.⊖",
                     "close.in_0.⊕       →  #FileDescriptor.⊕",
                     "close.in_0.⊖       →  α'.⊖",
                     "close.out.⊕        →  close.⊕             (forget out)",
                     "close.out.⊖        →  close.⊖             (forget out)",
                     "close.out.⊕        →  close.out.⊕",
                     "close.out.⊖        →  close.out.⊖",
                     "close.out.⊕        →  F.out.⊕",
                     "close.out.⊖        →  #SuccessZ.⊖",
                     "F.⊕                →  F.in_0.⊖            (recall in_0)",
                     "F.⊖                →  F.in_0.⊕            (recall in_0)",
                     "F.⊕                →  F.out.⊕             (recall out)",
                     "F.⊖                →  F.out.⊖             (recall out)",
                     "#FileDescriptor.⊖  →  close.in_0.⊖",
                     "F.in_0.⊕           →  F.⊖                 (forget in_0)",
                     "F.in_0.⊖           →  F.⊕                 (forget in_0)",
                     "F.in_0.⊕           →  F.in_0.⊕",
                     "F.in_0.⊖           →  F.in_0.⊖",
                     "F.in_0.⊕           →  δ.⊕",
                     "F.out.⊖            →  close.out.⊖",
                     "F.out.⊕            →  F.⊕                 (forget out)",
                     "F.out.⊖            →  F.⊖                 (forget out)",
                     "F.out.⊕            →  F.out.⊕",
                     "F.out.⊖            →  F.out.⊖",
                     "#SuccessZ.⊕        →  close.out.⊕",
                     "α'.⊕               →  close.in_0.⊕",
                     "α.⊕                →  φ.⊕",
                     "α.⊖                →  φ.load.σ4@0.⊖",
                     "α'.⊖               →  φ.load.σ4@4.⊖",
                     "δ.⊖                →  F.in_0.⊖",
                     "δ.⊕                →  φ.⊕",
                     "φ.load.σ4@0.⊕      →  α.⊕",
                     "φ.load.σ4@0.⊕      →  φ.load.⊕            (forget σ4@0)",
                     "φ.load.σ4@0.⊖      →  φ.load.⊖            (forget σ4@0)",
                     "φ.load.σ4@0.⊕      →  φ.load.σ4@0.⊕",
                     "φ.load.σ4@0.⊖      →  φ.load.σ4@0.⊖",
                     "φ.load.σ4@4.⊕      →  α'.⊕",
                     "φ.load.σ4@4.⊕      →  φ.load.⊕            (forget σ4@4)",
                     "φ.load.σ4@4.⊖      →  φ.load.⊖            (forget σ4@4)",
                     "φ.load.σ4@4.⊕      →  φ.load.σ4@4.⊕",
                     "φ.load.σ4@4.⊖      →  φ.load.σ4@4.⊖",
                     "φ.load.⊕           →  φ.⊕                 (forget load)",
                     "φ.load.⊖           →  φ.⊖                 (forget load)",
                     "φ.load.⊕           →  φ.load.⊕",
                     "φ.load.⊖           →  φ.load.⊖",
                     "φ.load.⊕           →  φ.load.σ4@0.⊕       (recall σ4@0)",
                     "φ.load.⊖           →  φ.load.σ4@0.⊖       (recall σ4@0)",
                     "φ.load.⊕           →  φ.load.σ4@4.⊕       (recall σ4@4)",
                     "φ.load.⊖           →  φ.load.σ4@4.⊖       (recall σ4@4)",
                     "φ.⊖                →  α.⊖",
                     "φ.⊖                →  δ.⊖",
                     "φ.⊕                →  φ.load.⊕            (recall load)",
                     "φ.⊖                →  φ.load.⊖            (recall load)"]
        self.graphs_are_equal(graph.graph, SchemaTestHelper.edges_to_dict(saturated))

        solver = Solver(graph, {F, FileDescriptor, SuccessZ})
        final_constraints = solver()

        self.assertTrue(SubtypeConstraint(SuccessZ, F_out) in final_constraints)
        tv = solver._get_type_var(Vertex(φ, True, True))
        self.assertTrue(SubtypeConstraint(F_in, tv) in final_constraints)
        tv_load = tv.add_suffix(LoadLabel.instance())
        tv_load_0 = tv_load.add_suffix(DerefLabel(4, 0))
        tv_load_4 = tv_load.add_suffix(DerefLabel(4, 4))
        self.assertTrue(SubtypeConstraint(tv_load_0, tv) in final_constraints)
        self.assertTrue(SubtypeConstraint(tv_load_4, FileDescriptor) in final_constraints)

if __name__ == '__main__':
    unittest.main()
