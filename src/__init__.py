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
