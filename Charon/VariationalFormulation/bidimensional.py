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
Bidimensional Variational Formulation Module
===========================================

This module defines classes for two-dimensional mechanical problems, including
plane strain and axisymmetric formulations.

The module provides specialized boundary condition classes, loading classes,
and problem formulations tailored to 2D mechanical problems. It handles
the appropriate implementation of strain-stress relationships, kinematics,
and variational forms specific to 2D domains.

Key components:
- BidimensionalBoundaryConditions: Base class for 2D boundary conditions
- AxiBoundaryConditions: Boundary conditions for axisymmetric problems
- PlaneStrainBoundaryConditions: Boundary conditions for plane strain problems
- BidimensionalLoading: Base class for 2D loading conditions
- PlaneStrainLoading: Loading conditions for plane strain problems
- AxiLoading: Loading conditions for axisymmetric problems
- Bidimensional: Base class for 2D problems
- Plane_strain: Implementation for plane strain problems
- Axisymmetric: Implementation for axisymmetric problems

"""

from .Problem import Problem
from .base_boundary_conditions import BoundaryConditions
from .base_loading import Loading
from ufl import dot, grad
from petsc4py.PETSc import ScalarType

class AxiBoundaryConditions(BoundaryConditions):
    """
    Boundary conditions for axisymmetric problems.
    
    This class provides methods to impose displacement boundary conditions
    specific to axisymmetric problems.
    
    Parameters
    ----------
    V, facet_tag, name : see BidimensionalBoundaryConditions parameters
    """
    def __init__(self, V, V_T, facet_tag, name):
        super().__init__(V, V_T, facet_tag, name, dim=2)
    
    def add_Ur(self, region, value=ScalarType(0)):
        """
        Add a Dirichlet boundary condition on the radial displacement component.
        
        Parameters
        ----------
        region, value : see add_component in Problem.py
        """
        self.add_component_by_name('Ur', region, value)
    
    def add_Uz(self, region, value=ScalarType(0)):
        """
        Add a Dirichlet boundary condition on the axial displacement component.
        
        Parameters
        ----------
        region, value : see add_component in Problem.py
        """   
        self.add_component_by_name('Uz', region, value)

class PlaneStrainBoundaryConditions(BoundaryConditions):
    """
    Boundary conditions for plane strain problems.
    
    This class provides methods to impose displacement boundary conditions
    specific to plane strain problems.
    
    Parameters
    ----------
    V, V_T, facet_tag, name : see BidimensionalBoundaryConditions parameters
    """
    def __init__(self, V, V_T, facet_tag, name):
        super().__init__(V, V_T, facet_tag, name, dim=2)
    
    def add_Ux(self, region, value=ScalarType(0)):
        """
        Add a Dirichlet boundary condition on the x-component of displacement.
        
        Parameters
        ----------
        region, value : see add_component in Problem.py
        """
        self.add_component_by_name('Ux', region, value)
    
    def add_Uy(self, region, value=ScalarType(0)):
        """
        Add a Dirichlet boundary condition on the y-component of displacement.
        
        Parameters
        ----------
        region, value : see add_component in Problem.py
        """
        self.add_component_by_name('Uy', region, value)
        
class PlaneStrainLoading(Loading):
    """
    Loading conditions for plane strain problems.
    
    This class provides methods to apply external forces in plane strain problems.
    
    Parameters
    ----------
    mesh, dx , kinematic : see Loading parameters in base_loading.py
    """
    def __init__(self, mesh, u_, dx, kinematic):
        super().__init__(mesh, u_, dx, kinematic, dim=2)
    
    def add_Fx(self, value, dx):
        """
        Add an external force in the x-direction.
        """
        self.add_force_by_name('Fx', value, dx)
    
    def add_Fy(self, value, dx):
        """
        Add an external force in the y-direction.
        """
        self.add_force_by_name('Fy', value, dx)

class AxiLoading(Loading):
    """
    Loading conditions for axisymmetric problems.
    
    This class provides methods to apply external forces in axisymmetric problems.
    
    Parameters
    ----------
    mesh, dx , kinematic : see Loading parameters in base_loading.py
    """
    def __init__(self, mesh, u_, dx, kinematic):
        super().__init__(mesh, u_, dx, kinematic, dim=2)
    
    def add_Fr(self, value, dx):
        """
        Add an external force in the r-direction.
        """
        self.add_force_by_name('Fr', value, dx)
    
    def add_Fz(self, value, dx):
        """
        Add an external force in the z-direction.
        """
        self.add_force_by_name('Fz', value, dx)

class Bidimensional(Problem):
    """
    Base class for two-dimensional mechanical problems.
    
    This class provides the foundation for defining and solving 2D
    mechanical problems (plane strain or axisymmetric) using the finite
    element method.
    
    The class handles the setup of finite element spaces, strain-stress
    relationships, and variational forms specific to 2D problems.
    """
    def boundary_conditions_class(self):
        """
        Return the appropriate boundary conditions class for 2D problems.
        
        Returns
        -------
        class The boundary conditions class for the specific 2D problem type
        """
        return self.bidim_boundary_conditions_class()

    def loading_class(self):
        """
        Return the appropriate loading class for 2D problems.
        
        Returns
        -------
        class The loading class for the specific 2D problem type
        """
        return self.bidim_loading_class()
    
    def undamaged_stress(self, u, v, T, T0, J):
        """
        Define the current stress in the material.
        
        Computes the stress tensor based on the displacement, velocity,
        temperature fields, and the Jacobian of the transformation.
        
        Parameters
        ----------
        u, v, T, T0, J : see current_stress in Problem.py
            
        Returns
        -------
        ufl.tensors.ListTensor Current stress represented as a vector in reduced form
        """
        sig3D = self.constitutive.stress_3D(u, v, T, T0, J)
        return self.kinematic.tensor_3d_to_compact(sig3D, symmetric=True)
    
class PlaneStrain(Bidimensional):
    """
    Class for plane strain mechanical problems.
    
    This class implements the specific formulation for plane strain problems,
    where the strain in the z-direction is assumed to be zero.
    
    The class provides methods for computing strains, stresses, and
    their contractions in the plane strain framework.
    """
    @property
    def name(self):
        """
        Return the name of the problem type.
        
        Returns
        -------
        str "PlaneStrain"
        """
        return "PlaneStrain"
    
    def bidim_boundary_conditions_class(self):
        """
        Return the boundary conditions class for plane strain problems.
        
        Returns
        -------
        class PlaneStrainBoundaryConditions
        """
        return PlaneStrainBoundaryConditions
    
    def bidim_loading_class(self):
        """
        Return the loading class for plane strain problems.
        
        Returns
        -------
        class PlaneStrainLoading
        """
        return PlaneStrainLoading
    
    def conjugate_strain(self):
       """
       Return the virtual strain conjugate to the Cauchy stress.
       
       Computes the appropriate strain measure that is work-conjugate
       to the Cauchy stress in the plane strain formulation.
       
       Returns
       -------
       ufl.tensors.ListTensor Conjugate strain tensor in compact form
       """
       conj = dot(grad(self.u_), self.kinematic.cofactor_compact(self.u))
       return self.kinematic.tensor_2d_to_compact(conj)

class Axisymmetric(Bidimensional):
    """
    Class for axisymmetric mechanical problems.
    
    This class implements the specific formulation for axisymmetric problems,
    where the domain is symmetric around the y-axis (by convention).
    
    The class provides methods for computing strains, stresses, and
    their contractions in the axisymmetric framework.
    """
    @property
    def name(self):
        """
        Return the name of the problem type.
        
        Returns
        -------
        str Axisymmetric"
        """
        return "Axisymmetric"
    
    def bidim_boundary_conditions_class(self):
        """
        Return the boundary conditions class for axisymmetric problems.
        
        Returns
        -------
        class AxiBoundaryConditions
        """
        return AxiBoundaryConditions
    
    def bidim_loading_class(self):
        """
        Return the loading class for axisymmetric problems.
        
        Returns
        -------
        class AxiLoading
        """
        return AxiLoading

    def conjugate_strain(self):
        """
        Return the strain conjugate to the Cauchy stress.
        
        Computes the appropriate strain measure that is work-conjugate
        to the Cauchy stress in the axisymmetric formulation.
        
        Returns
        -------
        ufl.tensors.ListTensor Conjugate strain tensor in reduced form
        """
        cofF = self.kinematic.cofactor_compact(self.u)
        return self.kinematic.tensor_3d_to_compact(dot(self.kinematic.grad_vector_3d(self.u_), cofF))