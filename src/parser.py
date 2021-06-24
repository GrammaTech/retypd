import re
from .schema import DerefLabel, DerivedTypeVariable, InLabel, LoadLabel, OutLabel, StoreLabel, \
        ForgetLabel, RecallLabel, SubtypeConstraint, Vertex


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
    def parse_label(label):
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
    def parse_variable(var):
        components = var.split('.')
        return DerivedTypeVariable(components[0], map(SchemaParser.parse_label, components[1:]))

    @staticmethod
    def parse_constraint(constraint):
        subtype_match = SchemaParser.subtype_pattern.match(constraint)
        if subtype_match:
            return SubtypeConstraint(SchemaParser.parse_variable(subtype_match.group(1)),
                                     SchemaParser.parse_variable(subtype_match.group(2)))
        raise ValueError

    @staticmethod
    def parse_node(node):
        node_match = SchemaParser.node_pattern.match(node)
        if node_match:
            var = SchemaParser.parse_variable(node_match.group(1))
            return Vertex(var, node_match.group(2) == '⊕')
        raise ValueError

    @staticmethod
    def parse_edge(edge):
        edge_match = SchemaParser.edge_pattern.match(edge)
        if edge_match:
            sub = SchemaParser.parse_node(edge_match.group(1))
            sup = SchemaParser.parse_node(edge_match.group(2))
            atts = {}
            if edge_match.group(3):
                capability = SchemaParser.parse_label(edge_match.group(5))
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
            (f, t, atts) = SchemaParser.parse_edge(edge)
            graph[(f, t)] = atts
        return graph


