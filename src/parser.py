'''
'''

import re
from .schema import AccessPathLabel, DerefLabel, DerivedTypeVariable, EdgeLabel, \
        InLabel, LoadLabel, OutLabel, StoreLabel, SubtypeConstraint, Vertex
from typing import Dict, Iterable, Tuple


class SchemaParser:
    '''Static helper functions for Schema tests. Since this parsing code is unlikely to be useful in
    the code itself, it is included here.
    '''

    subtype_pattern = re.compile('([^ ]*) ⊑ ([^ ]*)')
    in_pattern = re.compile('in_([0-9]+)')
    deref_pattern = re.compile('σ([0-9]+)@([0-9]+)')
    node_pattern = re.compile(r'([^ ]+)\.([⊕⊖])')
    edge_pattern = re.compile(r'(\S+)\s+→\s+(\S+)(\s+\((forget|recall) ([^ ]*)\))?')

    @staticmethod
    def parse_label(label: str) -> AccessPathLabel:
        if label == 'load':
            return LoadLabel.instance()
        if label == 'store':
            return StoreLabel.instance()
        if label == 'out':
            return OutLabel.instance()
        in_match = SchemaParser.in_pattern.match(label)
        if in_match:
            return InLabel(int(in_match.group(1)))
        deref_match = SchemaParser.deref_pattern.match(label)
        if deref_match:
            return DerefLabel(int(deref_match.group(1)), int(deref_match.group(2)))
        raise ValueError

    @staticmethod
    def parse_variable(var: str) -> DerivedTypeVariable:
        components = var.split('.')
        path = [SchemaParser.parse_label(label) for label in components[1:]]
        return DerivedTypeVariable(components[0], path)

    @staticmethod
    def parse_constraint(constraint: str) -> SubtypeConstraint:
        subtype_match = SchemaParser.subtype_pattern.match(constraint)
        if subtype_match:
            return SubtypeConstraint(SchemaParser.parse_variable(subtype_match.group(1)),
                                     SchemaParser.parse_variable(subtype_match.group(2)))
        raise ValueError

    @staticmethod
    def parse_node(node: str) -> Vertex:
        node_match = SchemaParser.node_pattern.match(node)
        if node_match:
            var = SchemaParser.parse_variable(node_match.group(1))
            return Vertex(var, node_match.group(2) == '⊕')
        raise ValueError

    @staticmethod
    def parse_edge(edge: str) -> Tuple[Vertex, Vertex, Dict[str, EdgeLabel]]:
        edge_match = SchemaParser.edge_pattern.match(edge)
        if edge_match:
            sub = SchemaParser.parse_node(edge_match.group(1))
            sup = SchemaParser.parse_node(edge_match.group(2))
            atts = {}
            if edge_match.group(3):
                capability = SchemaParser.parse_label(edge_match.group(5))
                if edge_match.group(4) == 'forget':
                    kind = EdgeLabel.Kind.FORGET
                elif edge_match.group(4) == 'recall':
                    kind = EdgeLabel.Kind.RECALL
                else:
                    raise ValueError
                atts['label'] = EdgeLabel(capability, kind)
            return (sub, sup, atts)
        raise ValueError

    @staticmethod
    def edges_to_dict(edges: Iterable[str]) -> Dict[Tuple[Vertex, Vertex], Dict[str, EdgeLabel]]:
        graph = {}
        for edge in edges:
            (head, tail, atts) = SchemaParser.parse_edge(edge)
            graph[(head, tail)] = atts
        return graph


