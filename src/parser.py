# Retypd - machine code type inference
# Copyright (C) 2021 GrammaTech, Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# This project is sponsored by the Office of Naval Research, One Liberty
# Center, 875 N. Randolph Street, Arlington, VA 22203 under contract #
# N68335-17-C-0700.  The content of the information does not necessarily
# reflect the position or policy of the Government and no official
# endorsement should be inferred.

'''Parsing helpers, mostly for unit testing.
'''

import re
from .schema import AccessPathLabel, DerefLabel, DerivedTypeVariable, EdgeLabel, \
        InLabel, LoadLabel, OutLabel, StoreLabel, SubtypeConstraint, Variance, Node
from typing import Dict, Tuple


class SchemaParser:
    '''Static helper functions for Schema tests. Since this parsing code is unlikely to be useful in
    the code itself, it is included here.
    '''

    subtype_pattern = re.compile(r'(\S*) (?:⊑|<=) (\S*)')
    in_pattern = re.compile('in_([0-9]+)')
    deref_pattern = re.compile('σ([0-9]+)@([0-9]+)')
    node_pattern = re.compile(r'(\S+)\.([⊕⊖])')
    edge_pattern = re.compile(r'(\S+)\s+(?:→|->)\s+(\S+)(\s+\((forget|recall) (\S*)\))?')
    whitespace_pattern = re.compile(r'\s')

    @staticmethod
    def parse_label(label: str) -> AccessPathLabel:
        '''Parse an AccessPathLabel. Raises ValueError if it is improperly formatted.
        '''
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
        '''Parse a DerivedTypeVariable. Raises ValueError if the string contains whitespace.
        '''
        if SchemaParser.whitespace_pattern.match(var):
            raise ValueError
        components = var.split('.')
        path = [SchemaParser.parse_label(label) for label in components[1:]]
        return DerivedTypeVariable(components[0], path)

    @staticmethod
    def parse_constraint(constraint: str) -> SubtypeConstraint:
        '''Parse a SubtypeConstraint. Raises a ValueError if constraint does not match
        SchemaParser.subtype_pattern.
        '''
        subtype_match = SchemaParser.subtype_pattern.match(constraint)
        if subtype_match:
            return SubtypeConstraint(SchemaParser.parse_variable(subtype_match.group(1)),
                                     SchemaParser.parse_variable(subtype_match.group(2)))
        raise ValueError

    @staticmethod
    def parse_node(node: str) -> Node:
        '''Parse a Node. Raise a ValueError if it does not match SchemaParser.node_pattern.
        '''
        node_match = SchemaParser.node_pattern.match(node)
        if node_match:
            var = SchemaParser.parse_variable(node_match.group(1))
            if node_match.group(2) == '⊕':
                variance = Variance.COVARIANT
            elif node_match.group(2) == '⊖':
                variance = Variance.CONTRAVARIANT
            else:
                raise ValueError
            return Node(var, variance)
        raise ValueError

    @staticmethod
    def parse_edge(edge: str) -> Tuple[Node, Node, Dict[str, EdgeLabel]]:
        '''Parse an edge in the graph, which consists of two nodes and an arrow, with an optional
        edge label.
        '''
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
