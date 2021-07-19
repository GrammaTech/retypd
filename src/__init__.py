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

'''An implementation of retypd based on the paper and slides included in the reference subdirectory.

To invoke, populate a ConstraintSet. Then, instantiate a Solver with the ConstraintSet and a
collection of "interesting" variables, such as functions and globals, specified either as strings or
DerivedTypeVariable objects. Then, invoke the solver.

After computation has finished, the constraints are available in the solver object's constraints
attribute.
'''

from .schema import ConstraintSet, DerefLabel, DerivedTypeVariable, InLabel, \
        LoadLabel, Node, OutLabel, StoreLabel, EdgeLabel, SubtypeConstraint, \
        Variance
from .solver import Solver
from .parser import SchemaParser
