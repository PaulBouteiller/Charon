# Copyright 2025 CEA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unified Loading Module
=====================

This module provides a unified framework for managing external loads
across all problem dimensions (1D, 2D, 3D) and coordinate systems.
"""

from dolfinx.fem import Constant
from petsc4py.PETSc import ScalarType
from ufl import FacetNormal, inner, dot
from ..utils.time_dependent_expressions import MyConstant


class Loading:   
    """
    Unified manager for external loads in mechanical problems.
    
    Attributes
    ----------
    kinematic : Kinematic Object handling kinematic transformations
    my_constant_list : list Time-dependent loading expressions
    Wext : ufl.Form External work form
    n : ufl.FacetNormal Outward normal vector
    u_ : ufl.TestFunction Test function
    dim : int Problem dimension
    mapping : dict Component name to index mapping
    """
    
    COMPONENT_MAPPINGS = {
        1: {'F': None},
        2: {'Fx': 0, 'Fy': 1, 'Fr': 0, 'Fz': 1},
        3: {'Fx': 0, 'Fy': 1, 'Fz': 2}
    }
    
    def __init__(self, mesh, u_, dx, kinematic, dim):
        """
        Initialize loading for given dimension.
        
        Parameters
        ----------
        mesh : dolfinx.mesh.Mesh Computational mesh
        u_ : ufl.TestFunction Test function
        dx : ufl.Measure Integration measure
        kinematic : Kinematic Kinematic handler
        dim : int Problem dimension (1, 2, or 3)
        """
        self.kinematic = kinematic
        self.my_constant_list = []
        self.u_ = u_
        self.n = FacetNormal(mesh)
        self.dim = dim
        self.mapping = self.COMPONENT_MAPPINGS[dim]
        
        # Initialize external work form to zero
        u_component = self._get_component(self.u_, 0 if dim > 1 else None)
        self.Wext = kinematic.measure(Constant(mesh, ScalarType(0)) * u_component, dx)
        
    def _get_component(self, u_, component_idx):
        """Get component from test function."""
        return u_[component_idx] if component_idx is not None else u_
        
    def add_loading(self, value, dx, component_name=None):
        """
        Add external loads to the variational form.
        
        Parameters
        ----------
        value : ScalarType, Expression, MyConstant, Load value
        dx : ufl.Measure Integration measure (dx for body forces, ds for surface)
        component_name : str, optional Component name ('Fx', 'Fy', etc.)
        """
        # Determine component index
        if component_name:
            if component_name not in self.mapping:
                raise ValueError(f"Component '{component_name}' not supported for dimension {self.dim}")
            component_idx = self.mapping[component_name]
        else:
            component_idx = 0 if self.dim > 1 else None
            
        u_component = self._get_component(self.u_, component_idx)
        
        if isinstance(value, MyConstant):
            if hasattr(value, "function"):
                load_term = inner(value.Expression.constant * value.function, u_component)
            else: 
                load_term = inner(value.Expression.constant, u_component)
            self.my_constant_list.append(value.Expression)
        # elif isinstance(value, Tabulated_BCs):
        #     return  # Handle separately if needed
        else:
            assert value != 0.
            load_term = inner(value, u_component)
            
        self.Wext += self.kinematic.measure(load_term, dx)
    
    def add_force_by_name(self, component_name, value, dx):
        """
        Add force by component name.
        
        Parameters
        ----------
        component_name : str Component name ('Fx', 'Fy', 'Fz', 'Fr', 'F')
        value : Load value or expression
        dx : ufl.Measure Integration measure
        """
        self.add_loading(value, dx, component_name)
        
    def add_pressure(self, p, ds):
        """
        Add pressure (normal surface force).
        
        Parameters
        ----------
        p : ScalarType or Expression Pressure value
        ds : ufl.Measure Surface integration measure
        """
        pressure_value = p.Expression.constant if isinstance(p, MyConstant) else p
        if isinstance(p, MyConstant):
            self.my_constant_list.append(p.Expression)
        self.Wext += self.kinematic.measure(-pressure_value * dot(self.n, self.u_), ds)