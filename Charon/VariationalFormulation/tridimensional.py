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

from .Problem import BoundaryConditions, Loading, Problem
from ufl import cofac, inner, div, sym, grad, dot
from petsc4py.PETSc import ScalarType
from basix.ufl import element

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
        BoundaryConditions.__init__(self, V, facets)

    def add_Ux(self, region, value = ScalarType(0)):
        """
        Add a Dirichlet boundary condition on the x-component of displacement.
        
        Parameters
        ----------
        region, value : see add_component in Problem.py
        """
        self.add_component(self.V, 0, self.bcs, region, value)
        self.add_associated_speed_acceleration(self.V, 0, region, value)

    def add_Uy(self, region, value=ScalarType(0)):
        """
        Add a Dirichlet boundary condition on the y-component of displacement.
        
        Parameters
        ----------
        region, value : see add_component in Problem.py
        """
        self.add_component(self.V, 1, self.bcs, region, value)
        self.add_associated_speed_acceleration(self.V, 1, region, value)

    def add_Uz(self, region, value=ScalarType(0)):
        """
        Add a Dirichlet boundary condition on the z-component of displacement.
        
        Parameters
        ----------
        region, value : see add_component in Problem.py
        """
        self.add_component(self.V, 2, self.bcs, region, value)
        self.add_associated_speed_acceleration(self.V, 2, region, value)
        
class TridimensionalLoading(Loading):
    """
    Loading conditions for three-dimensional problems.
    
    This class provides methods to apply external forces in 3D problems.
    """
    def __init__(self, mesh, u_, dx, kinematic):
        """
        Initialize 3D loading conditions.
        
        Parameters
        ----------
        mesh, u_, dx , kinematic : see Loading parameters in Problem.py
        """
        Loading.__init__(self, mesh, u_[0], dx, kinematic)

    def add_Fx(self, value, u_, dx):
        """
        Add an external force in the x-direction.
        
        Parameters
        ----------
        value, dx : see parameters of add_loading in Problem.py
        """
        self.add_loading(value, dx, sub = 0)
        
    def add_Fy(self, value, u_, dx):
        """
        Add an external force in the y-direction.
        
        Parameters
        ----------
        value, dx : see parameters of add_loading in Problem.py
        """
        self.add_loading(value, dx, sub = 1)
        
    def add_Fz(self, value, u_, dx):
        """
        Add an external force in the z-direction.
        
        Parameters
        ----------
        value, dx : see parameters of add_loading in Problem.py
        """
        self.add_loading(value, dx, sub = 2)

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

    def set_finite_element(self):
        """
        Define finite elements for displacement and stress fields.
        
        Sets up vector-valued Lagrange elements for displacement and
        tensor-valued quadrature elements for stress.
        """
        cell = self.mesh.basix_cell()
        self.U_e = element("Lagrange", cell, degree=self.u_deg, shape=(3,))  
        self.Sig_e = self.quad.quad_element(["Tensor", 3, 3])
        self.devia_e = self.Sig_e
        
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
    
    def div(self, v):
        """
        Compute the divergence of a vector field.
        
        Parameters
        ----------
        v : ufl.tensors.ListTensor Vector field
            
        Returns
        -------
        ufl.algebra.Sum Divergence of the vector field
        """
        return div(v)

    def dot(self, a, b):
        """
        Compute the dot product of two tensors.
        
        Parameters
        ----------
        a : ufl.tensors.ListTensor First tensor
        b : ufl.tensors.ListTensor Second tensor
            
        Returns
        -------
        ufl.algebra.Product Result of the dot product
        """
        return dot(a, b)
    
    def dot_grad_scal(self, tensor1, tensor2):
        """
        Compute the dot product of a tensor and a gradient.
        
        Parameters
        ----------
        tensor1 : ufl.tensors.ListTensor First tensor
        tensor2 : ufl.tensors.ListTensor Second tensor
            
        Returns
        -------
        ufl.algebra.Product Result of the dot product
        """
        self.dot(tensor1, tensor2)
        
    def inner(self, a, b):
        """
        Compute the inner product (double contraction) of two tensors.
        
        Parameters
        ----------
        a : ufl.tensors.ListTensor First tensor
        b : ufl.tensors.ListTensor Second tensor
            
        Returns
        -------
        ufl.algebra.Product Result of the inner product
        """
        return inner(a, b)
    
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
        return dot(sym(grad(self.u_)), cofac(self.kinematic.F_3D(self.u)))
    
    def extract_deviatoric(self, deviatoric):
        """
        Extract the deviatoric part of a stress tensor.
        
        For 3D problems, the deviatoric tensor is returned as is.
        
        Parameters
        ----------
        deviatoric : ufl.tensors.ListTensor Deviatoric stress tensor
            
        Returns
        -------
        ufl.tensors.ListTensor Deviatoric stress tensor (unchanged for 3D)
        """
        return deviatoric