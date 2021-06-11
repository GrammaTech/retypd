from abc import ABC
import re
import sys
import unittest

sys.path.append('src')

# from type_inference import ConstraintSet, DerefLabel, DerivedTypeVariable, LoadLabel, StoreLabel
from schema import ConstraintSet, DerefLabel, DerivedTypeVariable, ExistenceConstraint, InLabel, \
        LoadLabel, OutLabel, StoreLabel, SubtypeConstraint, Vertex

class SchemaTestHelper:
    '''Static helper functions for Schema tests. Since this parsing code is unlikely to be useful in
    the code itself, it is included here.
    '''

    existence_pattern = re.compile('VAR ([^ ]*)')
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
        existence_match = SchemaTestHelper.existence_pattern.match(constraint)
        if existence_match:
            return ExistenceConstraint(SchemaTestHelper.parse_variable(existence_match.group(1)))
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
                label = SchemaTestHelper.parse_label(edge_match.group(5))
                atts[edge_match.group(4)] = label
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
        '''A simple test from the paper. This one has no recursive data structures, so the fixed
        point contains "x ⊑ y", the constraint that this particular analysis hopes to generate.
        '''

        unfixed = ConstraintSet()
        p = DerivedTypeVariable('p')
        q = DerivedTypeVariable('q')
        x = DerivedTypeVariable('x')
        y = DerivedTypeVariable('y')
        q_store = DerivedTypeVariable('q', [StoreLabel.instance(), DerefLabel(4, 0)])
        p_load = DerivedTypeVariable('p', [LoadLabel.instance(), DerefLabel(4, 0)])
        unfixed.add_subtype(p, q)
        unfixed.add_subtype(x, q_store)
        unfixed.add_subtype(p_load, y)
        fixed = unfixed.fix()

        fixed_existence = ['VAR p',
                           'VAR p.load',
                           'VAR p.load.σ4@0',
                           'VAR p.store',
                           'VAR p.store.σ4@0',
                           'VAR q',
                           'VAR q.load',
                           'VAR q.load.σ4@0',
                           'VAR q.store',
                           'VAR q.store.σ4@0',
                           'VAR x',
                           'VAR y']

        fixed_subtype = ['p.load ⊑ p.load',
                         'p.load ⊑ q.load',
                         'p.load.σ4@0 ⊑ p.load.σ4@0',
                         'p.load.σ4@0 ⊑ q.load.σ4@0',
                         'p.load.σ4@0 ⊑ y',
                         'p ⊑ p',
                         'p ⊑ q',
                         'p.store ⊑ p.load',
                         'p.store ⊑ p.store',
                         'p.store ⊑ q.load',
                         'p.store.σ4@0 ⊑ p.load.σ4@0',
                         'p.store.σ4@0 ⊑ p.store.σ4@0',
                         'p.store.σ4@0 ⊑ q.load.σ4@0',
                         'p.store.σ4@0 ⊑ y',
                         'q.load ⊑ q.load',
                         'q.load.σ4@0 ⊑ q.load.σ4@0',
                         'q ⊑ q',
                         'q.store ⊑ p.load',
                         'q.store ⊑ p.store',
                         'q.store ⊑ q.load',
                         'q.store ⊑ q.store',
                         'q.store.σ4@0 ⊑ p.load.σ4@0',
                         'q.store.σ4@0 ⊑ p.store.σ4@0',
                         'q.store.σ4@0 ⊑ q.load.σ4@0',
                         'q.store.σ4@0 ⊑ q.store.σ4@0',
                         'q.store.σ4@0 ⊑ y',
                         'x ⊑ p.load.σ4@0',
                         'x ⊑ p.store.σ4@0',
                         'x ⊑ q.load.σ4@0',
                         'x ⊑ q.store.σ4@0',
                         'x ⊑ x',
                         'x ⊑ y',
                         'y ⊑ y']

        self.assertEqual(fixed.existence,
                         set(map(SchemaTestHelper.parse_constraint, fixed_existence)))
        self.assertEqual(fixed.subtype,
                         set(map(SchemaTestHelper.parse_constraint, fixed_subtype)))

        graph = fixed.generate_graph()

        simple_graph = ['p.load.⊕        →  p.load.⊕',
                        'p.load.⊖        →  p.load.⊖',
                        'p.load.⊖        →  p.store.⊖',
                        'p.load.⊕        →  q.load.⊕',
                        'p.load.⊖        →  q.store.⊖',
                        'p.load.σ4@0.⊕   →  p.load.σ4@0.⊕',
                        'p.load.σ4@0.⊖   →  p.load.σ4@0.⊖',
                        'p.load.σ4@0.⊖   →  p.store.σ4@0.⊖',
                        'p.load.σ4@0.⊕   →  q.load.σ4@0.⊕',
                        'p.load.σ4@0.⊖   →  q.store.σ4@0.⊖',
                        'p.load.σ4@0.⊖   →  x.⊖',
                        'p.load.σ4@0.⊕   →  y.⊕',
                        'p.⊕             →  p.⊕',
                        'p.⊖             →  p.⊖',
                        'p.⊕             →  q.⊕',
                        'p.store.⊕       →  p.load.⊕',
                        'p.store.⊕       →  p.store.⊕',
                        'p.store.⊖       →  p.store.⊖',
                        'p.store.⊕       →  q.load.⊕',
                        'p.store.⊖       →  q.store.⊖',
                        'p.store.σ4@0.⊕  →  p.load.σ4@0.⊕',
                        'p.store.σ4@0.⊕  →  p.store.σ4@0.⊕',
                        'p.store.σ4@0.⊖  →  p.store.σ4@0.⊖',
                        'p.store.σ4@0.⊕  →  q.load.σ4@0.⊕',
                        'p.store.σ4@0.⊖  →  q.store.σ4@0.⊖',
                        'p.store.σ4@0.⊖  →  x.⊖',
                        'p.store.σ4@0.⊕  →  y.⊕',
                        'q.load.⊖        →  p.load.⊖',
                        'q.load.⊖        →  p.store.⊖',
                        'q.load.⊕        →  q.load.⊕',
                        'q.load.⊖        →  q.load.⊖',
                        'q.load.⊖        →  q.store.⊖',
                        'q.load.σ4@0.⊖   →  p.load.σ4@0.⊖',
                        'q.load.σ4@0.⊖   →  p.store.σ4@0.⊖',
                        'q.load.σ4@0.⊕   →  q.load.σ4@0.⊕',
                        'q.load.σ4@0.⊖   →  q.load.σ4@0.⊖',
                        'q.load.σ4@0.⊖   →  q.store.σ4@0.⊖',
                        'q.load.σ4@0.⊖   →  x.⊖',
                        'q.⊖             →  p.⊖',
                        'q.⊕             →  q.⊕',
                        'q.⊖             →  q.⊖',
                        'q.store.⊕       →  p.load.⊕',
                        'q.store.⊕       →  p.store.⊕',
                        'q.store.⊕       →  q.load.⊕',
                        'q.store.⊕       →  q.store.⊕',
                        'q.store.⊖       →  q.store.⊖',
                        'q.store.σ4@0.⊕  →  p.load.σ4@0.⊕',
                        'q.store.σ4@0.⊕  →  p.store.σ4@0.⊕',
                        'q.store.σ4@0.⊕  →  q.load.σ4@0.⊕',
                        'q.store.σ4@0.⊕  →  q.store.σ4@0.⊕',
                        'q.store.σ4@0.⊖  →  q.store.σ4@0.⊖',
                        'q.store.σ4@0.⊖  →  x.⊖',
                        'q.store.σ4@0.⊕  →  y.⊕',
                        'x.⊕             →  p.load.σ4@0.⊕',
                        'x.⊕             →  p.store.σ4@0.⊕',
                        'x.⊕             →  q.load.σ4@0.⊕',
                        'x.⊕             →  q.store.σ4@0.⊕',
                        'x.⊕             →  x.⊕',
                        'x.⊖             →  x.⊖',
                        'x.⊕             →  y.⊕',
                        'y.⊖             →  p.load.σ4@0.⊖',
                        'y.⊖             →  p.store.σ4@0.⊖',
                        'y.⊖             →  q.store.σ4@0.⊖',
                        'y.⊖             →  x.⊖',
                        'y.⊕             →  y.⊕',
                        'y.⊖             →  y.⊖']

        self.graphs_are_equal(graph.graph, SchemaTestHelper.edges_to_dict(simple_graph))

        graph.add_forget_recall()

        forget_recall = ['p.load.⊕        →  p.⊕              (forget load)',
                         'p.load.⊖        →  p.⊖              (forget load)',
                         'p.load.⊕        →  p.load.⊕',
                         'p.load.⊖        →  p.load.⊖',
                         'p.load.⊕        →  p.load.σ4@0.⊕    (recall σ4@0)',
                         'p.load.⊖        →  p.load.σ4@0.⊖    (recall σ4@0)',
                         'p.load.⊖        →  p.store.⊖',
                         'p.load.⊕        →  q.load.⊕',
                         'p.load.⊖        →  q.store.⊖',
                         'p.load.σ4@0.⊕   →  p.load.⊕         (forget σ4@0)',
                         'p.load.σ4@0.⊖   →  p.load.⊖         (forget σ4@0)',
                         'p.load.σ4@0.⊕   →  p.load.σ4@0.⊕',
                         'p.load.σ4@0.⊖   →  p.load.σ4@0.⊖',
                         'p.load.σ4@0.⊖   →  p.store.σ4@0.⊖',
                         'p.load.σ4@0.⊕   →  q.load.σ4@0.⊕',
                         'p.load.σ4@0.⊖   →  q.store.σ4@0.⊖',
                         'p.load.σ4@0.⊖   →  x.⊖',
                         'p.load.σ4@0.⊕   →  y.⊕',
                         'p.⊕             →  p.⊕',
                         'p.⊖             →  p.⊖',
                         'p.⊕             →  p.load.⊕         (recall load)',
                         'p.⊖             →  p.load.⊖         (recall load)',
                         'p.⊕             →  p.store.⊖        (recall store)',
                         'p.⊖             →  p.store.⊕        (recall store)',
                         'p.⊕             →  q.⊕',
                         'p.store.⊕       →  p.⊖              (forget store)',
                         'p.store.⊖       →  p.⊕              (forget store)',
                         'p.store.⊕       →  p.load.⊕',
                         'p.store.⊕       →  p.store.⊕',
                         'p.store.⊖       →  p.store.⊖',
                         'p.store.⊕       →  p.store.σ4@0.⊕   (recall σ4@0)',
                         'p.store.⊖       →  p.store.σ4@0.⊖   (recall σ4@0)',
                         'p.store.⊕       →  q.load.⊕',
                         'p.store.⊖       →  q.store.⊖',
                         'p.store.σ4@0.⊕  →  p.load.σ4@0.⊕',
                         'p.store.σ4@0.⊕  →  p.store.⊕        (forget σ4@0)',
                         'p.store.σ4@0.⊖  →  p.store.⊖        (forget σ4@0)',
                         'p.store.σ4@0.⊕  →  p.store.σ4@0.⊕',
                         'p.store.σ4@0.⊖  →  p.store.σ4@0.⊖',
                         'p.store.σ4@0.⊕  →  q.load.σ4@0.⊕',
                         'p.store.σ4@0.⊖  →  q.store.σ4@0.⊖',
                         'p.store.σ4@0.⊖  →  x.⊖',
                         'p.store.σ4@0.⊕  →  y.⊕',
                         'q.load.⊖        →  p.load.⊖',
                         'q.load.⊖        →  p.store.⊖',
                         'q.load.⊕        →  q.⊕              (forget load)',
                         'q.load.⊖        →  q.⊖              (forget load)',
                         'q.load.⊕        →  q.load.⊕',
                         'q.load.⊖        →  q.load.⊖',
                         'q.load.⊕        →  q.load.σ4@0.⊕    (recall σ4@0)',
                         'q.load.⊖        →  q.load.σ4@0.⊖    (recall σ4@0)',
                         'q.load.⊖        →  q.store.⊖',
                         'q.load.σ4@0.⊖   →  p.load.σ4@0.⊖',
                         'q.load.σ4@0.⊖   →  p.store.σ4@0.⊖',
                         'q.load.σ4@0.⊕   →  q.load.⊕         (forget σ4@0)',
                         'q.load.σ4@0.⊖   →  q.load.⊖         (forget σ4@0)',
                         'q.load.σ4@0.⊕   →  q.load.σ4@0.⊕',
                         'q.load.σ4@0.⊖   →  q.load.σ4@0.⊖',
                         'q.load.σ4@0.⊖   →  q.store.σ4@0.⊖',
                         'q.load.σ4@0.⊖   →  x.⊖',
                         'q.⊖             →  p.⊖',
                         'q.⊕             →  q.⊕',
                         'q.⊖             →  q.⊖',
                         'q.⊕             →  q.load.⊕         (recall load)',
                         'q.⊖             →  q.load.⊖         (recall load)',
                         'q.⊕             →  q.store.⊖        (recall store)',
                         'q.⊖             →  q.store.⊕        (recall store)',
                         'q.store.⊕       →  p.load.⊕',
                         'q.store.⊕       →  p.store.⊕',
                         'q.store.⊕       →  q.⊖              (forget store)',
                         'q.store.⊖       →  q.⊕              (forget store)',
                         'q.store.⊕       →  q.load.⊕',
                         'q.store.⊕       →  q.store.⊕',
                         'q.store.⊖       →  q.store.⊖',
                         'q.store.⊕       →  q.store.σ4@0.⊕   (recall σ4@0)',
                         'q.store.⊖       →  q.store.σ4@0.⊖   (recall σ4@0)',
                         'q.store.σ4@0.⊕  →  p.load.σ4@0.⊕',
                         'q.store.σ4@0.⊕  →  p.store.σ4@0.⊕',
                         'q.store.σ4@0.⊕  →  q.load.σ4@0.⊕',
                         'q.store.σ4@0.⊕  →  q.store.⊕        (forget σ4@0)',
                         'q.store.σ4@0.⊖  →  q.store.⊖        (forget σ4@0)',
                         'q.store.σ4@0.⊕  →  q.store.σ4@0.⊕',
                         'q.store.σ4@0.⊖  →  q.store.σ4@0.⊖',
                         'q.store.σ4@0.⊖  →  x.⊖',
                         'q.store.σ4@0.⊕  →  y.⊕',
                         'x.⊕             →  p.load.σ4@0.⊕',
                         'x.⊕             →  p.store.σ4@0.⊕',
                         'x.⊕             →  q.load.σ4@0.⊕',
                         'x.⊕             →  q.store.σ4@0.⊕',
                         'x.⊕             →  x.⊕',
                         'x.⊖             →  x.⊖',
                         'x.⊕             →  y.⊕',
                         'y.⊖             →  p.load.σ4@0.⊖',
                         'y.⊖             →  p.store.σ4@0.⊖',
                         'y.⊖             →  q.store.σ4@0.⊖',
                         'y.⊖             →  x.⊖',
                         'y.⊕             →  y.⊕',
                         'y.⊖             →  y.⊖']

        forget_recall_graph = SchemaTestHelper.edges_to_dict(forget_recall)
        self.graphs_are_equal(graph.graph, forget_recall_graph)

        graph.saturate()
        self.graphs_are_equal(graph.graph, forget_recall_graph)


class RecursiveSchemaTest(SchemaTest, unittest.TestCase):
    def test_recursive(self):
        constraints = ConstraintSet()
        F = DerivedTypeVariable('F')
        δ = DerivedTypeVariable('δ')
        φ = DerivedTypeVariable('φ')
        α = DerivedTypeVariable('α')
        α_prime = DerivedTypeVariable("α'")
        close = DerivedTypeVariable('close')
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
        just_constraints = ["close.in_0.⊕       →  #FileDescriptor.⊕",
                            "close.in_0.⊖       →  α'.⊖",
                            "close.out.⊕        →  F.out.⊕",
                            "close.out.⊖        →  #SuccessZ.⊖",
                            "#FileDescriptor.⊖  →  close.in_0.⊖",
                            "F.in_0.⊕           →  δ.⊕",
                            "F.out.⊖            →  close.out.⊖",
                            "#SuccessZ.⊕        →  close.out.⊕",
                            "α'.⊕               →  close.in_0.⊕",
                            "α.⊕                →  φ.⊕",
                            "α.⊖                →  φ.load.σ4@0.⊖",
                            "α'.⊖               →  φ.load.σ4@4.⊖",
                            "δ.⊖                →  F.in_0.⊖",
                            "δ.⊕                →  φ.⊕",
                            "φ.load.σ4@0.⊕      →  α.⊕",
                            "φ.load.σ4@4.⊕      →  α'.⊕",
                            "φ.⊖                →  α.⊖",
                            "φ.⊖                →  δ.⊖"]

        self.graphs_are_equal(graph.graph, SchemaTestHelper.edges_to_dict(just_constraints))

        graph.add_forget_recall()
        forget_recall = ["close.⊖            →  close.in_0.⊕        (recall in_0)",
                         "close.⊕            →  close.in_0.⊖        (recall in_0)",
                         "close.⊖            →  close.out.⊖         (recall out)",
                         "close.⊕            →  close.out.⊕         (recall out)",
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
                         "φ.load.⊕           →  φ.⊕                 (forget load)",
                         "φ.load.⊖           →  φ.⊖                 (forget load)",
                         "φ.load.⊕           →  φ.load.σ4@0.⊕       (recall σ4@0)",
                         "φ.load.⊖           →  φ.load.σ4@0.⊖       (recall σ4@0)",
                         "φ.load.⊕           →  φ.load.σ4@4.⊕       (recall σ4@4)",
                         "φ.load.⊖           →  φ.load.σ4@4.⊖       (recall σ4@4)",
                         "φ.⊖                →  α.⊖",
                         "φ.⊖                →  δ.⊖",
                         "φ.⊕                →  φ.load.⊕            (recall load)",
                         "φ.⊖                →  φ.load.⊖            (recall load)"]
        self.graphs_are_equal(graph.graph, SchemaTestHelper.edges_to_dict(forget_recall))

        graph.saturate()

if __name__ == '__main__':
    unittest.main()
