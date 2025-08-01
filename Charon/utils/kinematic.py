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
Created on Wed Apr  9 10:51:13 2025

@author: bouteillerp
Kinematics Module
===============

This module provides a comprehensive framework for handling kinematic transformations
in finite element simulations of solid mechanics problems. It supports various
coordinate systems and dimensions, automatically adapting computations to the
specific problem type.

Key features:
- Support for 1D, 2D, and 3D problems
- Support for Cartesian, cylindrical, and spherical coordinate systems
- Gradient calculations for scalar and vector fields
- Conversion between different tensor representations
- Deformation gradient and strain calculations
- Push-forward and pull-back operations for tensors

The module enables accurate representation of large deformations, nonlinear
strain measures, and various stress tensors for different material models.
"""

from ufl import (grad, as_tensor, div, Identity, dot, as_vector, det, inv, cofac, inner)
from math import sqrt

class Kinematic:
    """
    Encapsulates kinematic operations for different dimensions and geometries.
    
    This class provides methods for calculating gradients, tensors, 
    transformations, and other kinematic quantities, automatically adapting
    the calculations to the problem dimension and geometry.
    
    Attributes
    ----------
    name : str Name of the mechanical model ('CartesianUD', 'PlaneStrain', etc.)
    r : Function or None Radial coordinate in axisymmetric, cylindrical, and spherical cases
    """
    def __init__(self, name, r):
        """
        Initialize a Kinematic object.

        Parameters
        ----------
        name : str
            Name of the mechanical model, must be one of:
            [CartesianUD, CylindricalUD, SphericalUD, 
             PlaneStrain, Axisymmetric, Tridimensional]
        r : Function or None
            Radial coordinate in axisymmetric, cylindrical, and spherical cases
        """
        self.name = name
        self.r = r
        self._is_1d = self.name in ["CartesianUD", "CylindricalUD", "SphericalUD"]

    # =========================================================================
    # Gradient methods
    # =========================================================================
    
    def grad_scalar_compact(self, scalar_field):
        """
        Return the appropriate representation of a scalar field gradient.

        Parameters
        ----------
        scalar_field : Function Scalar field

        Returns
        -------
        Expression Gradient adapted to dimension and geometry
        """
        if self._is_1d:
            return scalar_field.dx(0)
        else: 
            return grad(scalar_field)
    
    def grad_scalar_3d(self, scalar_field):
       """
       Return the 3D gradient of a scalar field in vector form.
       
       Parameters
       ----------
       scalar_field : Function  Scalar field
           
       Returns
       -------
       Vector  3D gradient adapted to dimension and geometry
       """
       if self._is_1d:
           return as_vector([scalar_field.dx(0), 0, 0])
       elif self.name == "PlaneStrain": 
           return as_vector([scalar_field.dx(0), scalar_field.dx(1), 0])
       elif self.name == "Axisymmetric": 
           return as_vector([scalar_field.dx(0), 0, scalar_field.dx(1)])
       else:  # Tridimensional
           return grad(scalar_field)
    
    def grad_vector_compact(self, vector_field, symmetric=False):
       """
       Return the compact gradient of a vector field.
       
       The representation is adapted to the dimension and geometry.
       
       Parameters
       ----------
       vector_field : Function Vector field
       symmetric : bool, optional  If True, use a symmetric representation. Default: False
           
       Returns
       -------
       Expression  Compact gradient
       """
       if self.name == "CartesianUD":
           return vector_field.dx(0)
       elif self.name == "CylindricalUD":
           return as_vector([vector_field.dx(0), vector_field / self.r])
       elif self.name == "SphericalUD":
           return as_vector([vector_field.dx(0), vector_field / self.r, vector_field / self.r])
       elif self.name == "PlaneStrain":
           grad_u = grad(vector_field)
           if symmetric:
               return as_vector([grad_u[0, 0], grad_u[1, 1], grad_u[0, 1]])
           else:
               return as_vector([grad_u[0, 0], grad_u[1, 1], grad_u[0, 1], grad_u[1, 0]])
       elif self.name == "Axisymmetric":
           grad_u = grad(vector_field)
           if symmetric:
               return as_vector([grad_u[0, 0], vector_field[0] / self.r, grad_u[1, 1], grad_u[0, 1]])
           else:
               return as_vector([grad_u[0, 0], vector_field[0] / self.r, grad_u[1, 1], 
                                 grad_u[0, 1], grad_u[1, 0]])
       else:  # Tridimensional
           return grad(vector_field)
    
    def grad_vector_3d(self, vector_field, symmetric=False):
       """
       Return the three-dimensional gradient of a vector field.
       
       Parameters
       ----------
       vector_field : Function 
           Vector field
       symmetric : bool, optional 
           If True, use a symmetric representation. Default: False
           
       Returns
       -------
       Tensor 
           3D gradient as a tensor
       """
       return self.compact_to_tensor_3d(self.grad_vector_compact(vector_field, symmetric=symmetric), symmetric=symmetric)
    
    def grad_eulerian_compact(self, velocity_field, displacement_field):
       """
       Return the spatial velocity gradient in compact form.
       
       Computes ∂v/∂x where ∂x denotes the derivative with respect to current coordinates.
       
       Parameters
       ----------
       velocity_field : Function Vector field to derive
       displacement_field : Function  Displacement field defining the transformation
           
       Returns
       -------
       Expression  Eulerian gradient in compact form
       """
       inv_F_compact = self.inv_deformation_gradient_compact(displacement_field)
       grad_compact = self.grad_vector_compact(velocity_field)
       
       if self._is_1d:
           if self.name == "CartesianUD":
               return grad_compact * inv_F_compact[0]
           else:  # CylindricalUD or SphericalUD
               return as_vector([grad_compact[i] * inv_F_compact[i] for i in range(len(grad_compact))])
       elif self.name == "PlaneStrain":
           return self.tensor_2d_to_compact(dot(self.compact_to_tensor_2d(grad_compact), 
                                               inv(Identity(2) + grad(displacement_field))))
       elif self.name == "Axisymmetric":
           return self.tensor_3d_to_compact(dot(self.compact_to_tensor_3d(grad_compact), inv_F_compact))
       else:  # Tridimensional
           return dot(grad_compact, inv_F_compact)
       
    # =========================================================================
    # Format conversion methods
    # =========================================================================
    
    def compact_to_tensor_3d(self, compact_tensor, symmetric=False):
       """
       Convert a tensor in compact form to its full three-dimensional form.
       
       Parameters
       ----------
       compact_tensor : Expression 
           Field in compact form
       symmetric : bool, optional 
           If True, use a symmetric representation. Default: False
           
       Returns
       -------
       Tensor 
           Corresponding 3D tensor
       """
       # 1D models
       if self.name == "CartesianUD":
           return as_tensor([[compact_tensor, 0, 0], 
                             [0, 0, 0], [0, 0, 0]])
       elif self.name == "CylindricalUD":
           return as_tensor([[compact_tensor[0], 0, 0], 
                             [0, compact_tensor[1], 0], [0, 0, 0]])
       elif self.name == "SphericalUD":
           return as_tensor([[compact_tensor[0], 0, 0], 
                             [0, compact_tensor[1], 0], 
                             [0, 0, compact_tensor[1]]])
       
       # 2D models
       elif self.name == "PlaneStrain":
           if symmetric:
               return as_tensor([[compact_tensor[0], compact_tensor[2], 0], 
                                [compact_tensor[2], compact_tensor[1], 0], 
                                [0, 0, 0]])
           else:
               return as_tensor([[compact_tensor[0], compact_tensor[2], 0], 
                                [compact_tensor[3], compact_tensor[1], 0], 
                                [0, 0, 0]])
       elif self.name == "Axisymmetric":
           if symmetric:
               return as_tensor([[compact_tensor[0], 0, compact_tensor[3]], 
                                [0, compact_tensor[1], 0], 
                                [compact_tensor[3], 0, compact_tensor[2]]])
           else:
               return as_tensor([[compact_tensor[0], 0, compact_tensor[3]], 
                                [0, compact_tensor[1], 0], 
                                [compact_tensor[4], 0, compact_tensor[2]]])
       
       # 3D model
       else:
           if symmetric:
               return as_tensor([[compact_tensor[0], compact_tensor[3], compact_tensor[4]], 
                                [compact_tensor[3], compact_tensor[1], compact_tensor[5]], 
                                [compact_tensor[4], compact_tensor[5], compact_tensor[2]]])
           else:
               return compact_tensor
    
    def compact_to_tensor_2d(self, compact_tensor):
       """
       Convert a tensor in compact form to its two-dimensional form.
       
       Parameters
       ----------
       compact_tensor : Expression  Tensor in compact form
           
       Returns
       -------
       Tensor  Corresponding 2D tensor
       """
       return as_tensor([[compact_tensor[0], compact_tensor[2]], 
                         [compact_tensor[3], compact_tensor[1]]])
    
    def tensor_2d_to_compact(self, tensor_2d):
       """
       Convert a 2D tensor to its compact form.
       
       Parameters
       ----------
       tensor_2d : Tensor  2D tensor
           
       Returns
       -------
       Vector  Corresponding compact form
       """
       return as_vector([tensor_2d[0, 0], tensor_2d[1, 1], tensor_2d[0, 1], tensor_2d[1, 0]])
    
    def tensor_3d_to_compact(self, tensor_3d, symmetric=False):
       """
       Convert a 3D tensor to its compact form.
       
       Parameters
       ----------
       tensor_3d : Tensor  3D tensor
       symmetric : bool, optional  If True, use a symmetric representation. Default: False
           
       Returns
       -------
       Expression  Corresponding compact form
       """
       # 1D models
       if self.name == "CartesianUD":
           return tensor_3d[0, 0]
       elif self.name == "CylindricalUD":
           return as_vector([tensor_3d[0, 0], tensor_3d[1, 1]])
       elif self.name == "SphericalUD":
           return as_vector([tensor_3d[0, 0], tensor_3d[1, 1], tensor_3d[2, 2]])
       
       # 2D models
       elif self.name == "PlaneStrain":
           if symmetric:
               return as_vector([tensor_3d[0, 0], tensor_3d[1, 1], tensor_3d[0, 1]])
           else:
               return as_vector([tensor_3d[0, 0], tensor_3d[1, 1], tensor_3d[0, 1], tensor_3d[1, 0]])
       elif self.name == "Axisymmetric":
           if symmetric:
               return as_vector([tensor_3d[0, 0], tensor_3d[1, 1], tensor_3d[2, 2], tensor_3d[0, 2]])
           else:
               return as_vector([tensor_3d[0, 0], tensor_3d[1, 1], tensor_3d[2, 2], 
                               tensor_3d[0, 2], tensor_3d[2, 0]])
       
       # 3D model
       elif self.name == "Tridimensional":
           if symmetric:
               return as_vector([tensor_3d[0, 0], tensor_3d[1, 1], tensor_3d[2, 2], 
                               tensor_3d[0, 1], tensor_3d[0, 2], tensor_3d[1, 2]])
           else:
               return tensor_3d
    
    def tensor_3d_to_mandel_compact(self, tensor_3d):
        """
        Convert a 3D tensor to its compact Mandel notation representation.
        
        The off-diagonal terms are weighted by a factor √2, but only the 
        components needed for the specific geometry are kept.
        Parameters
        ----------
        tensor_3d : Tensor  3D symmetric tensor
        Returns
        -------
        Vector  Compact Mandel representation
        """
        sq2 = sqrt(2)
        
        # 1D models
        if self._is_1d:
            return as_vector([tensor_3d[0, 0], tensor_3d[1, 1], tensor_3d[2, 2]])
        
        # 2D models
        elif self.name == "PlaneStrain":
            return as_vector([tensor_3d[0, 0], tensor_3d[1, 1], tensor_3d[2, 2], sq2 * tensor_3d[0, 1]])
        elif self.name == "Axisymmetric":
            return as_vector([tensor_3d[0, 0], tensor_3d[1, 1], tensor_3d[2, 2], sq2 * tensor_3d[0, 2]])
        
        # 3D model
        elif self.name == "Tridimensional":
            return as_vector([tensor_3d[0, 0], tensor_3d[1, 1], tensor_3d[2, 2], 
                            sq2 * tensor_3d[0, 1], sq2 * tensor_3d[0, 2], sq2 * tensor_3d[1, 2]])
        
    def mandel_compact_to_tensor_3d(self, mandel_compact):
       """
       Convert a compact Mandel representation to a 3D tensor.
       
       Parameters
       ----------
       mandel_compact : Vector  Compact Mandel representation
           
       Returns
       -------
       Tensor Corresponding 3D tensor
       """
       sq2 = sqrt(2)
       
       # 1D models
       if self._is_1d:
           return as_tensor([[mandel_compact[0], 0, 0], 
                            [0, mandel_compact[1], 0], 
                            [0, 0, mandel_compact[2]]])
       
       # 2D models
       elif self.name == "PlaneStrain":
           return as_tensor([[mandel_compact[0], mandel_compact[3] / sq2, 0], 
                            [mandel_compact[3] / sq2, mandel_compact[1], 0], 
                            [0, 0, mandel_compact[2]]])
       elif self.name == "Axisymmetric":
           return as_tensor([[mandel_compact[0], 0, mandel_compact[3] / sq2], 
                            [0, mandel_compact[1], 0], 
                            [mandel_compact[3] / sq2, 0, mandel_compact[2]]])
       
       # 3D model
       elif self.name == "Tridimensional":
           return as_tensor([[mandel_compact[0], mandel_compact[3] / sq2, mandel_compact[4] / sq2], 
                            [mandel_compact[3] / sq2, mandel_compact[1], mandel_compact[5] / sq2], 
                            [mandel_compact[4] / sq2, mandel_compact[5] / sq2, mandel_compact[2]]])
    
    # =========================================================================
    # Transformation methods
    # =========================================================================
    def deformation_gradient_compact(self, displacement_field):
        """
        Return the deformation gradient in compact form.
        
        Note: This compact representation differs from standard compact notation
        because it includes additional diagonal terms (e.g., F_zz in plane strain)
        due to the identity tensor contribution.
        
        Parameters
        ----------
        displacement_field : Function  Displacement field
        """
        # 1D models
        F_3d = self.deformation_gradient_3d(displacement_field)
        if self._is_1d:
            return as_vector([F_3d[0, 0], F_3d[1, 1], F_3d[2, 2]])
        elif self.name == "PlaneStrain":
            return as_vector([F_3d[0, 0], F_3d[1, 1], F_3d[0, 1], F_3d[1, 0], F_3d[2, 2]])
        elif self.name == "Axisymmetric":
            return as_vector([F_3d[0, 0], F_3d[1, 1], F_3d[0, 2], F_3d[2, 0], F_3d[2, 2]]) 
        else:
            return Identity(3) + self.grad_vector_3d(displacement_field)
    
    def deformation_gradient_3d(self, displacement_field):
       """
       Return the full 3D deformation gradient.
       
       Parameters
       ----------
       displacement_field : Function  Displacement field
       """
       return Identity(3) + self.grad_vector_3d(displacement_field)
    
    def jacobian(self, displacement_field):
       """
       Return the Jacobian of the transformation (measure of local dilatation).
       
       Parameters
       ----------
       displacement_field : Function  Displacement field
       """
       # 1D models
       if self.name == "CartesianUD":
           return 1 + displacement_field.dx(0)
       elif self.name == "CylindricalUD":
           return (1 + displacement_field.dx(0)) * (1 + displacement_field / self.r)
       elif self.name == "SphericalUD":
           return (1 + displacement_field.dx(0)) * (1 + displacement_field / self.r)**2
       
       # 2D models
       elif self.name == "PlaneStrain":
           return det(Identity(2) + grad(displacement_field))
       elif self.name == "Axisymmetric":
           F_compact = self.deformation_gradient_3d(displacement_field)
           return F_compact[1, 1] * (F_compact[0, 0] * F_compact[2, 2] - F_compact[0, 2] * F_compact[2, 0])
       
       # 3D model
       else:
           return det(Identity(3) + grad(displacement_field))
    
    def inv_deformation_gradient_compact(self, displacement_field):
       """
       Return the inverse of the deformation gradient in compact form.
       
       Parameters
       ----------
       displacement_field : Function Displacement field
       """
       # 1D models
       if self.name == "CartesianUD":
           return as_vector([1 / (1 + displacement_field.dx(0)), 1, 1])
       elif self.name == "CylindricalUD":
           return as_vector([1 / (1 + displacement_field.dx(0)), 1/(1 + displacement_field / self.r), 1])
       elif self.name == "SphericalUD":
           return as_vector([1 / (1 + displacement_field.dx(0)), 1 /(1 + displacement_field / self.r), 1 / (1 + displacement_field / self.r)])
       
       # 2D models
       elif self.name == "PlaneStrain":
           inv_F_2d = inv(Identity(2) + grad(displacement_field))
           return as_tensor([[inv_F_2d[0, 0], inv_F_2d[0, 1], 0], 
                             [inv_F_2d[1, 0], inv_F_2d[1, 1], 0], 
                             [0, 0, 1]])
       elif self.name == "Axisymmetric":
           grad_u = grad(displacement_field)
           prefactor = (1 + grad_u[0, 0]) * (1 + grad_u[1, 1]) - grad_u[0, 1] * (1 + grad_u[1, 0])
           return as_tensor([[(1 + grad_u[1, 1]) / prefactor, 0, -grad_u[0, 1] / prefactor],
                            [0, 1 / (1 + displacement_field[0] / self.r), 0],
                            [-grad_u[1, 0] / prefactor, 0, (1 + grad_u[0, 0]) / prefactor]])
       
       # 3D model
       else:
           return inv(Identity(3) + grad(displacement_field))
    
    def relative_deformation_gradient_3d(self, current_displacement, previous_displacement):
       """
       Return the relative deformation gradient between two configurations.
       
       Parameters
       ----------
       current_displacement : Function Current displacement field
       previous_displacement : Function Previous displacement field
       """
       F_current = self.deformation_gradient_3d(current_displacement)
       inv_F_previous = inv(self.deformation_gradient_3d(previous_displacement))
       return dot(F_current, inv_F_previous)
   
    def cofactor_compact(self, displacement_field):
       """
       Compute the cofactor in compact form adapted to geometry.
       
       Parameters
       ----------
       displacement_field : Function Displacement field
       """
       # 1D models
       if self.name == "CartesianUD":
           return 1
       elif self.name == "CylindricalUD":               
           return as_vector([1 + displacement_field / self.r, 1 + displacement_field.dx(0)])
       elif self.name == "SphericalUD":
           return as_vector([(1 + displacement_field / self.r)**2, 
                             (1 + displacement_field.dx(0)) * (1 + displacement_field / self.r),
                             (1 + displacement_field.dx(0)) * (1 + displacement_field / self.r)])
       
       # 2D models
       elif self.name == "PlaneStrain":
           return cofac(Identity(2) + grad(displacement_field))
       elif self.name == "Axisymmetric":
           F_3d = self.deformation_gradient_3d(displacement_field)
           return as_tensor([[F_3d[2, 2] * F_3d[1, 1], 0, -F_3d[0, 2] * F_3d[1, 1]],
                            [0, F_3d[0, 0] * F_3d[2, 2] - F_3d[0, 2] * F_3d[2, 0], 0],
                            [-F_3d[2, 0] * F_3d[1, 1], 0, F_3d[0, 0] * F_3d[1, 1]]])
       
       # 3D model
       else:  # Tridimensional
           return cofac(self.deformation_gradient_3d(displacement_field))
    
    def reduced_det(self, tensor):
        if self._is_1d:
            return tensor[0, 0] * tensor[1, 1] * tensor[2, 2]
    
    # =========================================================================
    # Strain methods
    # =========================================================================
    
    def left_cauchy_green_compact(self, displacement_field):
        """
        Return the left Cauchy-Green tensor in compact form.
        
        Parameters
        ----------
        displacement_field : Function  Displacement field
            
        Returns
        -------
        Expression Left Cauchy-Green tensor (form adapted to dimension)
        """
        if self._is_1d:
            F_compact = self.deformation_gradient_compact(displacement_field)
            return as_vector([F_compact[0]**2, F_compact[1]**2, F_compact[2]**2])
        else:
            B_3d = self.left_cauchy_green_3d(displacement_field)
            return self.tensor_3d_to_compact(B_3d, symmetric=True)
    
    def left_cauchy_green_3d(self, displacement_field):
       """
       Return the full 3D left Cauchy-Green tensor.
       
       Parameters
       ----------
       displacement_field : Function Displacement field
           
       Returns
       -------
       Tensor  3D left Cauchy-Green tensor
       """
       F_3d = self.deformation_gradient_3d(displacement_field)
       return dot(F_3d, F_3d.T)
    
    # =========================================================================
    # Integration and miscellaneous methods
    # =========================================================================
    
    def measure(self, a, dx):
        """
        Return the integration measure adapted to the geometry.
        
        The measure is:
        - dx in Cartesian 1D, 2D plane or 3D cases
        - r*dr in cylindrical 1D or axisymmetric cases
        - r²*dr in spherical 1D case

        Parameters
        ----------
        a : Expression Field to integrate
        dx : Measure Default integration measure

        Returns
        -------
        Expression
            Adapted integration measure
        """
        if self.name in ["CartesianUD", "PlaneStrain", "Tridimensional"]:
            return a * dx
        elif self.name in ["CylindricalUD", "Axisymmetric"]:
            return a * self.r * dx
        elif self.name == "SphericalUD":
            return a * self.r**2 * dx
    
    def div(self, v):
        """
        Return the divergence of vector field v.

        Parameters
        ----------
        v : Function Vector field
        """
        if self._is_1d:
            return v.dx(0)
        else:
            return div(v)
        
    def contract_scalar_gradients(self, grad_1, grad_2):
       """
       Compute the dot product between two scalar gradients.
       
       Parameters
       ----------
       grad_1 : ufl.tensors.ListTensor or scalar
           First scalar gradient
       grad_2 : ufl.tensors.ListTensor or scalar
           Second scalar gradient
           
       Returns
       -------
       ufl.algebra.Product
           Result of the dot product between gradients
       """
       # 1D models - gradients are scalars
       if self._is_1d:
           return grad_1 * grad_2
       
       # 2D and 3D models - use UFL dot product
       else:  # PlaneStrain, Axisymmetric, Tridimensional
           return dot(grad_1, grad_2)
        
    def contract_simple(self, compact_tensor_1, compact_tensor_2):
       """
       Compute the dot product between two compact tensor representations.
       
       Parameters
       ----------
       compact_tensor_1 : ufl.tensors.ListTensor
           First compact tensor (size varies depending on coordinate system)
       compact_tensor_2 : ufl.tensors.ListTensor
           Second compact tensor
           
       Returns
       -------
       ufl.algebra.Product or ufl.tensors.ListTensor
           Result of the dot product, scalar or vector depending on the problem type
       """
       # 1D models
       if self.name == "CartesianUD":
           return compact_tensor_1 * compact_tensor_2
       elif self.name == "CylindricalUD":
           return as_vector([compact_tensor_1[i] * compact_tensor_2[i] for i in range(2)])
       elif self.name == "SphericalUD":
           return as_vector([compact_tensor_1[i] * compact_tensor_2[i] for i in range(3)])
       
       # 2D and 3D models - use UFL dot product
       else:  # PlaneStrain, Axisymmetric, Tridimensional
           return dot(compact_tensor_1, compact_tensor_2)
       
    def contract_double(self, compact_tensor_1, compact_tensor_2):
        """
        Compute the double contraction between two tensors in compact form.
        
        Performs the operation σ:ε (stress:strain double contraction) adapted
        to each geometry's compact representation.
        
        Parameters
        ----------
        compact_tensor_1 : ufl.tensors.ListTensor
            First compact tensor (typically stress)
        compact_tensor_2 : ufl.tensors.ListTensor  
            Second compact tensor (typically strain)
            
        Returns
        -------
        ufl.algebra.Sum
            Result of the double contraction
        """
        # 1D models
        if self.name == "CartesianUD":
            return compact_tensor_1 * compact_tensor_2
        elif self.name in ["CylindricalUD", "SphericalUD"]:
            return dot(compact_tensor_1, compact_tensor_2)
        
        # 2D models
        elif self.name == "PlaneStrain":
            # Handle different compact formats: [σxx, σyy, σxy] vs [εxx, εyy, εxy, εyx]
            shape_1 = compact_tensor_1.ufl_shape[0]
            shape_2 = compact_tensor_2.ufl_shape[0]
            if shape_1 == 3 and shape_2 == 4:
                return compact_tensor_1[0] * compact_tensor_2[0] + \
                       compact_tensor_1[1] * compact_tensor_2[1] + \
                       compact_tensor_1[2] * (compact_tensor_2[2] + compact_tensor_2[3])
            else:
                return dot(compact_tensor_1, compact_tensor_2)
                
        elif self.name == "Axisymmetric":
            # Handle different compact formats: [σrr, σθθ, σzz, σrz] vs [εrr, εθθ, εzz, εrz, εzr]
            shape_1 = compact_tensor_1.ufl_shape[0]
            shape_2 = compact_tensor_2.ufl_shape[0]
            if shape_1 == 4 and shape_2 == 5:
                return compact_tensor_1[0] * compact_tensor_2[0] + \
                       compact_tensor_1[1] * compact_tensor_2[1] + \
                       compact_tensor_1[2] * compact_tensor_2[2] + \
                       compact_tensor_1[3] * (compact_tensor_2[3] + compact_tensor_2[4])
            else:
                return dot(compact_tensor_1, compact_tensor_2)
        
        # 3D model
        else:  # Tridimensional
            return inner(compact_tensor_1, compact_tensor_2)
    
    def push_forward(self, tensor, u):
        """
        Return the push-forward of a second-order twice covariant tensor.

        Parameters
        ----------
        tensor : Tensor Second-order tensor
        u : Function Displacement field

        Returns
        -------
        Tensor Tensor after push-forward
        """
        F = self.F_3D(u)
        return dot(dot(F, tensor), F.T)
    
    def pull_back(self, tensor, u):
        """
        Return the pull-back of a second-order twice covariant tensor.

        Parameters
        ----------
        tensor : Tensor Second-order tensor
        u : Function Displacement field

        Returns
        -------
        Tensor Tensor after pull-back
        """
        F = self.F_3D(u)
        inv_F = inv(F)
        return dot(dot(inv_F, tensor), inv_F.T)