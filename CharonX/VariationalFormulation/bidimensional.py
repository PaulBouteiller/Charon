

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
- Axisymetric: Implementation for axisymmetric problems

"""

from .Problem import BoundaryConditions, Loading, Problem
from ufl import cofac, as_vector, dot, grad, as_tensor, Identity
from petsc4py.PETSc import ScalarType
from basix.ufl import element

class BidimensionalBoundaryConditions(BoundaryConditions):
    """
    Base class for boundary conditions in two-dimensional problems.
    
    This class provides methods to impose displacement boundary conditions
    for two-dimensional mechanical problems. It extends the generic
    BoundaryConditions class with specialized methods for 2D domains.
    
    Parameters
    ----------
    V : dolfinx.fem.FunctionSpace Function space for the displacement field
    facet_tag : dolfinx.mesh.MeshTags Tags identifying different regions of the boundary
    name : str Identifier for the boundary condition type
    """
    def __init__(self, V, facet_tag, name):
        self.name = name
        BoundaryConditions.__init__(self, V, facet_tag)
        
    def add_clamped(self, region):
        """
        Add a clamped boundary condition (fixed in all directions).
        
        Imposes zero displacement and zero velocity/acceleration on all
        components of the displacement field on the specified boundary region.
        
        Parameters
        ----------
        region : int Tag identifying the boundary region where the condition is applied
        """
        self.add_component(self.V, 0, self.bcs, region, ScalarType(0))
        self.add_associated_speed_acceleration(self.V, 0, region, ScalarType(0))
        self.add_component(self.V, 1, self.bcs, region, ScalarType(0))
        self.add_associated_speed_acceleration(self.V, 1, region, ScalarType(0))
        
class AxiBoundaryConditions(BidimensionalBoundaryConditions):
    """
    Boundary conditions for axisymmetric problems.
    
    This class provides methods to impose displacement boundary conditions
    specific to axisymmetric problems, including axisymmetry conditions.
    
    Parameters
    ----------
    V, facet_tag, name : see BidimensionalBoundaryConditions parameters
    """
    def add_Ur(self, region, value = ScalarType(0)):
        """
        Add a Dirichlet boundary condition on the radial displacement component.
        
        Parameters
        ----------
        region, value : see add_component in Problem.py
        """
        self.add_component(self.V, 0, self.bcs, region, value)
        self.add_associated_speed_acceleration(self.V, 0, region, value)

    def add_Uz(self, region, value = ScalarType(0)):
        """
        Add a Dirichlet boundary condition on the axial displacement component.
        
        Parameters
        ----------
        region, value : see add_component in Problem.py
        """       
        self.add_component(self.V, 1, self.bcs, region, value)
        self.add_associated_speed_acceleration(self.V, 1, region, value)

    def add_axi(self, region):
        """
        Add an axisymmetry boundary condition.
        
        This condition is necessary when a side of the domain lies on the
        symmetry axis to prevent field divergence.
        
        Parameters
        ----------
        region : see add_component in Problem.py
        """
        self.add_component(self.V, 0, self.bcs_axi, region, ScalarType(1))
        self.add_component(self.V, 0, self.bcs, region, ScalarType(0))
        self.add_associated_speed_acceleration(self.V, 0, region, ScalarType(0))
            
class PlaneStrainBoundaryConditions(BidimensionalBoundaryConditions):
    """
    Boundary conditions for plane strain problems.
    
    This class provides methods to impose displacement boundary conditions
    specific to plane strain problems.
    
    Parameters
    ----------
    V, facet_tag, name : see BidimensionalBoundaryConditions parameters
    """
    def add_Ux(self, region, value=ScalarType(0)):
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

class BidimensionalLoading(Loading):
    """
    Base class for loading conditions in two-dimensional problems.
    
    This class initializes the external work form for 2D mechanical problems.
    
    Parameters
    ----------
    mesh, u_, dx , kinematic : see Loading parameters in Problem.py
    """
    def __init__(self, mesh, u_, dx, kinematic):
        Loading.__init__(self, mesh, u_[0], dx, kinematic)

class PlaneStrainLoading(BidimensionalLoading):
    """
    Loading conditions for plane strain problems.
    
    This class provides methods to apply external forces in plane strain problems.
    
    Parameters
    ----------
    mesh, u_, dx , kinematic : see Loading parameters in Problem.py
    """
    def add_Fx(self, value, u_, dx):
        """
        Add an external force in the x-direction.
        
        Parameters
        ----------
        value, u_, dx : see parameters of add_loading in Problem.py
        """
        self.add_loading(value, u_[0], dx)

    def add_Fy(self, value, u_, dx):
        """
        Add an external force in the y-direction.
        
        Parameters
        ----------
        value, u_, dx : see parameters of add_loading in Problem.py
        """
        self.add_loading(value, u_[1], dx)

class AxiLoading(BidimensionalLoading):
    """
    Loading conditions for axisymmetric problems.
    
    This class provides methods to apply external forces in axisymmetric problems.
    
    Parameters
    ----------
    mesh, u_, dx , kinematic : see Loading parameters in Problem.py
    """
    def add_Fr(self, value, u_, dx):
        """
        Add an external force in the radial direction.
        
        Parameters
        ----------
        value, u_, dx : see parameters of add_loading in Problem.py
        """
        self.add_loading(value, u_[0], dx)

    def add_Fz(self, value, u_, dx):
        """
        Add an external force in the axial direction.
        
        Parameters
        ----------
        value, u_, dx : see parameters of add_loading in Problem.py
        """
        self.add_loading(value, u_[1], dx)

class Bidimensional(Problem):
    """
    Base class for two-dimensional mechanical problems.
    
    This class provides the foundation for defining and solving 2D
    mechanical problems (plane strain or axisymmetric) using the finite
    element method.
    
    The class handles the setup of finite element spaces, strain-stress
    relationships, and variational forms specific to 2D problems.
    """
    def set_finite_element(self):
        """
        Define finite elements for displacement and stress fields.
        
        Sets up the appropriate finite elements for the displacement field
        (vector-valued Lagrange elements) and for the stress field
        (quadrature elements).
        """
        cell = self.mesh.basix_cell()
        self.U_e = element("Lagrange", cell, degree=self.u_deg, shape=(2,))  
        self.Sig_e = self.quad.quad_element(["Vector", self.sig_dim_quadrature()])
        self.devia_e = self.quad.quad_element(["Vector", 4])

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
        return self.kinematic.tridim_to_reduit(sig3D, sym=True)
    
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
        return self.dot(tensor1, tensor2)
        
    def dot(self, tensor1, tensor2):
        """
        Compute the contracted product between two tensors.
        
        Parameters
        ----------
        tensor1 : ufl.tensors.ListTensor First tensor
        tensor2 : ufl.tensors.ListTensor Second tensor
            
        Returns
        -------
        ufl.algebra.Product Result of the dot product
        """
        return dot(tensor1, tensor2)
    
class Plane_strain(Bidimensional):
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
    
    def extract_deviatoric(self, s):
        """
        Extract the deviatoric part of a stress tensor.
        
        Converts a 2D stress tensor to a 4-component vector representation
        of its deviatoric part.
        
        Parameters
        ----------
        s : ufl.tensors.ListTensor Stress tensor
            
        Returns
        -------
        ufl.tensors.ListTensor Vector representation of the deviatoric stress
        """
        return as_vector([s[0, 0], s[1, 1], s[2, 2], s[0, 1]])
    
    def sig_dim_quadrature(self):
        """
        Return the dimension of the stress vector in quadrature space.
        
        Returns
        -------
        int 3 for plane strain (σxx, σyy, σxy)
        """
        return 3

    def conjugate_strain(self):
        """
        Return the virtual strain conjugate to the Cauchy stress.
        
        Computes the appropriate strain measure that is work-conjugate
        to the Cauchy stress in the plane strain formulation.
        
        Returns
        -------
        ufl.tensors.ListTensor Conjugate strain tensor in reduced form
        """
        conj = dot(grad(self.u_), cofac(Identity(2) + grad(self.u)))
        return self.kinematic.bidim_to_reduit(conj)
    
    def inner(self, vector_1, vector_2):
        """
        Compute the double contraction between two tensors represented as vectors.
        
        Parameters
        ----------
        vector_1 : ufl.tensors.ListTensor First vector representing a tensor
        vector_2 : ufl.tensors.ListTensor Second vector representing a symmetrized tensor
            
        Returns
        -------
        ufl.algebra.Sum Result of the double contraction
        """
        shape_1 = vector_1.ufl_shape[0]
        shape_2 = vector_2.ufl_shape[0]
        if shape_1 == 3 and shape_2 == 4:
            return vector_1[0] * vector_2[0] + vector_1[1] * vector_2[1] + \
                    + vector_1[2] * (vector_2[2] + vector_2[3]) 

class Axisymetric(Bidimensional):
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
        str Axisymetric"
        """
        return "Axisymetric"
    
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
    
    def sig_dim_quadrature(self):
        """
        Return the dimension of the stress vector in quadrature space.
        
        Returns
        -------
        int 4 for axisymmetric (σrr, σzz, σθθ, σrz)
        """
        return 4
    
    def extract_deviatoric(self, deviatoric):
        """
        Extract the deviatoric part of a stress tensor.
        
        Converts a stress tensor to its deviatoric part in reduced form.
        
        Parameters
        ----------
        deviatoric : ufl.tensors.ListTensor Stress tensor
            
        Returns
        -------
        ufl.tensors.ListTensor Reduced form of the deviatoric stress
        """
        return self.kinematic.tridim_to_reduit(deviatoric, sym=True)
    
    def inner(self, vector_1, vector_2):
        """
        Compute the double contraction between two tensors represented as vectors.
        
        Parameters
        ----------
        vector_1 : ufl.tensors.ListTensor First vector representing a tensor
        vector_2 : ufl.tensors.ListTensor Second vector representing a symmetrized tensor
            
        Returns
        -------
        ufl.algebra.Sum Result of the double contraction
        """
        shape_1 = vector_1.ufl_shape[0]
        shape_2 = vector_2.ufl_shape[0]
        if shape_1 == 4 and shape_2 == 5:
            return vector_1[0] * vector_2[0] + vector_1[1] * vector_2[1] + \
                    vector_1[2] * vector_2[2] + vector_1[3] * (vector_2[3] + vector_2[4]) 

    def cofac3D(self, x_tens):
        """
        Compute the cofactor transpose (COM A^T = det(A)A^-1) of a tensor.
        
        Parameters
        ----------
        x_tens : ufl.tensors.ListTensor 3x3 tensor with an "X" structure
            
        Returns
        -------
        ufl.tensors.ListTensor Cofactor transpose of the input tensor
        """
        TComA = as_tensor([[x_tens[2, 2] * x_tens[1, 1], 0, -x_tens[0, 2] * x_tens[1, 1]],
                           [0, x_tens[0, 0] * x_tens[2, 2] - x_tens[0, 2] * x_tens[2, 0], 0],
                           [-x_tens[2, 0] * x_tens[1, 1], 0, x_tens[0, 0] * x_tens[1, 1]]])
        return TComA

    def conjugate_strain(self):
        """
        Return the strain conjugate to the Cauchy stress.
        
        Computes the appropriate strain measure that is work-conjugate
        to the Cauchy stress in the axisymmetric formulation.
        
        Returns
        -------
        ufl.tensors.ListTensor Conjugate strain tensor in reduced form
        """
        cofF = self.cofac3D(self.kinematic.F_3D((self.u)))
        return self.kinematic.tridim_to_reduit(dot(self.kinematic.grad_3D(self.u_), cofF))