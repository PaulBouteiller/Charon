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
Unidimensional Variational Formulation Module
===========================================

This module defines classes for one-dimensional mechanical problems, including
Cartesian, cylindrical, and spherical formulations.

The module provides specialized boundary condition classes, loading classes,
and problem formulations tailored to 1D mechanical problems with various
coordinate systems. It handles the appropriate implementation of strain-stress
relationships, kinematics, and variational forms specific to 1D domains.

Key components:
- UnidimensionalBoundaryConditions: Boundary conditions for 1D problems
- UnidimensionalLoading: Loading conditions for 1D problems
- Unidimensional: Base class for 1D problems
- CartesianUD: Implementation for Cartesian 1D problems
- CylindricalUD: Implementation for cylindrical 1D problems (radial symmetry)
- SphericalUD: Implementation for spherical 1D problems (radial symmetry)

"""

from .Problem import Problem
from petsc4py.PETSc import ScalarType
from basix.ufl import element    
from .base_boundary_conditions import BoundaryConditions
from .base_loading import Loading

class UnidimensionalBoundaryConditions(BoundaryConditions):
    """
    Boundary conditions for one-dimensional problems.
    
    This class provides methods to impose displacement boundary conditions
    for one-dimensional mechanical problems with various coordinate systems.
    """
    def __init__(self, V, facets, name):
        """
        Initialize 1D boundary conditions.
        
        Parameters
        ----------
        V : dolfinx.fem.FunctionSpace Function space for the displacement field
        facets : dolfinx.mesh.MeshTags Tags identifying different regions of the boundary
        name : str Identifier for the boundary condition type
        """
        super().__init__(V, facets, name, dim=1)
    
    def add_U(self, region, value=ScalarType(0)):
        """
        Impose a Dirichlet boundary condition on the displacement component.
        
        Parameters
        ----------
        region, value : see add_component in Problem.py
        """   
        self.add_component_by_name('U', region, value)

class UnidimensionalLoading(Loading):
    """
    Loading conditions for one-dimensional problems.
    
    This class provides methods to apply external forces in 1D problems.
    
    Parameters
    ----------
    mesh, u_, dx , kinematic : see Loading parameters in Problem.py
    """
    def __init__(self, mesh, u_, dx, kinematic):
        super().__init__(mesh, u_, dx, kinematic, dim=1)
    
    def add_F(self, value, dx):
        """
        Add an external force.
        """
        self.add_force_by_name('F', value, dx)

class Unidimensional(Problem):
    """
    Base class for one-dimensional mechanical problems.
    
    This class provides the foundation for defining and solving 1D
    mechanical problems using the finite element method, with support
    for Cartesian, cylindrical, and spherical coordinate systems.
    """
    def set_finite_element(self):
        """
        Initialize the finite element types for the displacement and stress fields.
        """
        return element("Lagrange", self.mesh_manager.cell_type, degree = self.u_deg) 
        
    def boundary_conditions_class(self):
        """
        Return the boundary conditions class for 1D problems.
        
        Returns
        -------
        class UnidimensionalBoundaryConditions
        """
        return UnidimensionalBoundaryConditions
    
    def loading_class(self):
        """
        Return the loading class for 1D problems.
        
        Returns
        -------
        class UnidimensionalLoading
        """
        return UnidimensionalLoading
        
    def extract_deviatoric(self, deviatoric):
        """
        Extract the deviatoric part of a stress tensor.
        
        Parameters
        ----------
        deviatoric : ufl.tensors.ListTensor Stress tensor
            
        Returns
        -------
        ufl.tensors.ListTensor Vector representation of the deviatoric stress
        """
        return 
    
    def undamaged_stress(self, u, v, T, T0, J):
        """
        Define the current stress in the material.
        
        Returns the stress tensor in reduced form as:
        - a scalar σ for Cartesian problems
        - a 2-vector (σrr, σθθ) for cylindrical problems
        - a 3-vector (σrr, σθθ, σφφ) for spherical problems
        
        Parameters
        ----------
        u, v, T, T0, J : see current_stress in Problem.py
            
        Returns
        -------
        ufl.tensors.ListTensor Current stress in reduced form
        """
        return self.kinematic.tensor_3d_to_compact(self.constitutive.stress_3D(u, v, T, T0, J))
        
    def conjugate_strain(self):
       """
       Return the virtual strain conjugate to the Cauchy stress in compact form.
       
       Returns
       -------
       ufl.algebra.Product or ufl.tensors.ListTensor
           Conjugate strain in compact form
       """
       return self.kinematic.contract_simple(
           self.kinematic.grad_vector_compact(self.u_), 
           self.kinematic.cofactor_compact(self.u)
       )
    
class CartesianUD(Unidimensional):
    """
    Class for one-dimensional Cartesian mechanical problems.
    
    This class implements the specific formulation for 1D problems
    in Cartesian coordinates.
    """
    @property
    def name(self):
        """
        Return the name of the problem type.
        
        Returns
        -------
        str "CartesianUD"
        """
        return "CartesianUD"

class CylindricalUD(Unidimensional):
    """
    Class for one-dimensional cylindrical mechanical problems.
    
    This class implements the specific formulation for 1D problems
    with cylindrical symmetry (radial problems).
    """
    @property
    def name(self):
        """
        Return the name of the problem type.
        
        Returns
        -------
        str "CylindricalUD"
        """
        return "CylindricalUD"

class SphericalUD(Unidimensional):
    """
    Class for one-dimensional spherical mechanical problems.
    
    This class implements the specific formulation for 1D problems
    with spherical symmetry (radial problems).
    """
    @property
    def name(self):
        """
        Return the name of the problem type.
        
        Returns
        -------
        str "SphericalUD"
        """
        return "SphericalUD"