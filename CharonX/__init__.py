from __future__ import division, print_function
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD
from dolfinx.mesh import (create_interval, create_unit_interval, 
                          create_unit_square, create_rectangle, 
                          CellType, create_box, locate_entities_boundary)
from dolfinx.fem import Constant, Function, Expression
from ufl import (cofac, det, as_vector, conditional, lt, And, gt, Or, 
                 action, SpatialCoordinate)
from basix.ufl import element, quadrature_element
import numpy as np
from dolfinx import __version__

from .utils.default_parameters import *
from .utils.tensor_operations import *

from .ConstitutiveLaw.material import *
from .ConstitutiveLaw.thermal_material import *

from .VariationalFormulation.unidimensional import *
from .VariationalFormulation.bidimensional import *
from .VariationalFormulation.tridimensional import *
from .utils.MyExpression import MyConstant
from pandas import read_csv

from .Solve.Solve import *
try:
    from .Solve.RestartSolve import *
except Exception:
    print("Gmsh has not been loaded therefore cannot be used")

# try:
from .Mesh.Gmsh_mesh import *
    # from dolfinx.mesh import GhostMode
# except Exception:
#     print("Gmsh has not been loaded therefore cannot be used")

print("Loading CharonX base on dolfinx version " + __version__)
