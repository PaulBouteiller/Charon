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
Tridimensional Variational Formulation Module
============================================

This module defines classes for three-dimensional mechanical problems.

The module provides specialized boundary condition classes, loading classes,
and problem formulations tailored to 3D mechanical problems. It handles
the appropriate implementation of strain-stress relationships, kinematics,
and variational forms specific to 3D domains.

Key components:
- TridimensionalBoundaryConditions: Boundary conditions for 3D problems
- TridimensionalLoading: Loading conditions for 3D problems
- Tridimensional: Implementation for 3D problems
"""

from .Problem import Problem
from ufl import grad, dot
from petsc4py.PETSc import ScalarType
from .base_boundary_conditions import BoundaryConditions
from .base_loading import Loading

class TridimensionalBoundaryConditions(BoundaryConditions):
    """
    Boundary conditions for three-dimensional problems.
    
    This class provides methods to impose displacement boundary conditions
    for three-dimensional mechanical problems.
    """
    def __init__(self, V, facets, name):
        """
        Initialize 3D boundary conditions.
            
        Parameters
        ----------
        V : dolfinx.fem.FunctionSpace Function space for the displacement field
        facet_tag : dolfinx.mesh.MeshTags Tags identifying different regions of the boundary
        name : str Identifier for the boundary condition type
        """
        super().__init__(V, facets, name, dim=3)
    
    def add_Ux(self, region, value = ScalarType(0)):
        """
        Add a Dirichlet boundary condition on the x-component of displacement.
        
        Parameters
        ----------
        region, value : see add_component in Problem.py
        """
        self.add_component_by_name('Ux', region, value)
    
    def add_Uy(self, region, value = ScalarType(0)):
        """
        Add a Dirichlet boundary condition on the y-component of displacement.
        
        Parameters
        ----------
        region, value : see add_component in Problem.py
        """
        self.add_component_by_name('Uy', region, value)
    
    def add_Uz(self, region, value=ScalarType(0)):
        """
        Add a Dirichlet boundary condition on the z-component of displacement.
        
        Parameters
        ----------
        region, value : see add_component in Problem.py
        """
        self.add_component_by_name('Uz', region, value)
        
class TridimensionalLoading(Loading):
    """
    Loading conditions for three-dimensional problems.
    
    This class provides methods to apply external forces in 3D problems.
    """
    def __init__(self, mesh, u_, dx, kinematic):
        super().__init__(mesh, u_, dx, kinematic, dim=3)
    
    def add_Fx(self, value, dx):
        self.add_force_by_name('Fx', value, dx)
    
    def add_Fy(self, value, dx):
        self.add_force_by_name('Fy', value, dx)
    
    def add_Fz(self, value, dx):
        self.add_force_by_name('Fz', value, dx)

class Tridimensional(Problem):
    """
    Class for three-dimensional mechanical problems.
    
    This class implements the specific formulation for 3D problems,
    providing methods for computing strains, stresses, and their
    contractions in the three-dimensional framework.
    """
    @property
    def name(self):
        """
        Return the name of the problem type.
        
        Returns
        -------
        str "Tridimensional"
        """
        return "Tridimensional"

    def boundary_conditions_class(self):
        """
        Return the boundary conditions class for 3D problems.
        
        Returns
        -------
        class TridimensionalBoundaryConditions
        """
        return TridimensionalBoundaryConditions
    
    def loading_class(self):
        """
        Return the loading class for 3D problems.
        
        Returns
        -------
        class TridimensionalLoading
        """
        return TridimensionalLoading
    
    def undamaged_stress(self, u, v, T, T0, J):
        """
        Define the current stress in the material.
        
        Computes the 3D stress tensor directly without reduction.
        
        Parameters
        ----------
        u, v, T, T0, J : see current_stress in Problem.py
            
        Returns
        -------
        ufl.tensors.ListTensor Current 3D stress tensor
        """
        return self.constitutive.stress_3D(u, v, T, T0, J)
    
    def conjugate_strain(self):
       """
       Return the strain conjugate to the Cauchy stress.
       
       Computes the appropriate strain measure for 3D problems that is
       work-conjugate to the Cauchy stress.
       
       Returns
       -------
       ufl.tensors.ListTensor Conjugate strain tensor
       """
       return dot(grad(self.u_), self.kinematic.cofactor_compact(self.u))