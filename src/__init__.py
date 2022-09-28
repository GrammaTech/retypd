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

"""An implementation of retypd based on the paper and slides included in the reference subdirectory.

To invoke, create a Program, which requires a lattice of atomic types, a set of global variables of
interest, a mapping from functions to constraints generated from them, and a call graph. Then,
instantiate a Solver with the Program. Lastly, invoke the solver. The result of calling the solver
object is the set of constraints generated from the analysis.
"""

from .graph import EdgeLabel, Node
from .dummylattice import DummyLattice, DummyLatticeCTypes
from .schema import (
    ConstraintSet,
    DerefLabel,
    DerivedTypeVariable,
    InLabel,
    LoadLabel,
    OutLabel,
    Program,
    StoreLabel,
    SubtypeConstraint,
    Variance,
    Lattice,
    LatticeCTypes,
)
from .solver import Solver, SolverConfig
from .parser import SchemaParser
from .c_type_generator import CTypeGenerator, CTypeGenerationError
from .clattice import CLattice, CLatticeCTypes
from .c_types import (
    CType,
    VoidType,
    IntType,
    FloatType,
    CharType,
    BoolType,
    ArrayType,
    PointerType,
    FunctionType,
    Field,
    CompoundType,
    StructType,
    UnionType,
)
from .graph_solver import GraphSolverConfig
from .sketches import Sketches
from .loggable import LogLevel
