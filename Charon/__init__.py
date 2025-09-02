from __future__ import division, print_function
from dolfinx.mesh import (create_interval, create_unit_interval, 
                          create_unit_square, create_rectangle, 
                          CellType, create_box, locate_entities_boundary)
from dolfinx import __version__

from .ConstitutiveLaw.material import Material
from .ConstitutiveLaw.thermal_material import LinearThermal

from .utils.stiffness_builders import build_transverse_isotropic_stiffness, build_orthotropic_stiffness, compute_bulk_modulus

from .Mesh.mesh_manager import MeshManager

from .VariationalFormulation.unidimensional import CartesianUD, CylindricalUD, SphericalUD
from .VariationalFormulation.bidimensional import PlaneStrain, Axisymmetric
from .VariationalFormulation.tridimensional import Tridimensional



from .Solve.Solve import Solve
from .Mesh.gmsh_mesh import *
print("Loading CharonX base on dolfinx version " + __version__)

