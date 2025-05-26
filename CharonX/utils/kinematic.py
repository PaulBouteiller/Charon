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

from ufl import (grad, as_tensor, div, tr, Identity, dot, as_vector, det, inv,
                 conditional, ge)
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
        
        # Configurations for different model types
        self._model_config = {"dim1": ["CartesianUD", "CylindricalUD", "SphericalUD"],
                              "dim2": ["PlaneStrain", "Axisymmetric"],
                              "dim3": ["Tridimensional"]}

    def _is_1d(self):
        """Check if the model is one-dimensional"""
        return self.name in self._model_config["dim1"]
        
    # =========================================================================
    # Gradient methods
    # =========================================================================
    
    def grad_scal(self, f):
        """
        Return the appropriate representation of a scalar field gradient.

        Parameters
        ----------
        f : Function Scalar field

        Returns
        -------
        Expression Gradient adapted to dimension and geometry
        """
        if self._is_1d():
            return f.dx(0)
        else: 
            return grad(f)
    
    def v_grad3D(self, f):
        """
        Return the 3D gradient of a scalar field in vector form.

        Parameters
        ----------
        f : Function Scalar field

        Returns
        -------
        Vector 3D gradient adapted to dimension and geometry
        """
        grad_f = self.grad_scal(f)
        
        if self._is_1d():
            return as_vector([grad_f, 0, 0])
        elif self.name == "PlaneStrain": 
            return as_vector([grad_f[0], grad_f[1], 0])
        elif self.name == "Axisymmetric": 
            return as_vector([grad_f[0], 0, grad_f[1]])
        else:  # Tridimensional
            return grad_f
    
    def grad_reduit(self, u, sym = False):
        """
        Return the reduced gradient of a vector field.
        
        The representation is adapted to the dimension and geometry.

        Parameters
        ----------
        u : Function Vector field
        sym : bool, optional If True, use a symmetric representation. Default: False

        Returns
        -------
        Expression Reduced gradient
        """
        if self.name == "CartesianUD":
            return u.dx(0)
        elif self.name == "CylindricalUD":
            return as_vector([u.dx(0), u / self.r])
        elif self.name == "SphericalUD":
            return as_vector([u.dx(0), u / self.r, u / self.r])
        elif self.name == "PlaneStrain":
            grad_u = grad(u)
            return self._get_2d_reduced_grad(grad_u, sym)
        elif self.name == "Axisymmetric":
            grad_u = grad(u)
            return self._get_axi_reduced_grad(grad_u, u, sym)
        else:  # Tridimensional
            return grad(u)
    
    def _get_2d_reduced_grad(self, grad_u, sym):
        """
        Private method to get the reduced 2D gradient.

        Parameters
        ----------
        grad_u : Expression Full gradient tensor
        sym : bool If True, use a symmetric representation

        Returns
        -------
        Vector Reduced gradient
        """
        if sym:
            return as_vector([grad_u[0, 0], grad_u[1, 1], grad_u[0, 1]])
        else:
            return as_vector([grad_u[0, 0], grad_u[1, 1], grad_u[0, 1], grad_u[1, 0]])
    
    def _get_axi_reduced_grad(self, grad_u, u, sym):
        """
        Private method to get the reduced axisymmetric gradient.

        Parameters
        ----------
        grad_u : Expression Full gradient tensor
        u : Function Displacement field
        sym : bool If True, use a symmetric representation

        Returns
        -------
        Vector Reduced gradient
        """
        if sym:
            return as_vector([grad_u[0, 0], u[0] / self.r, grad_u[1, 1], grad_u[0, 1]])
        else:
            return as_vector([grad_u[0, 0], u[0] / self.r, grad_u[1, 1], 
                            grad_u[0, 1], grad_u[1, 0]])
    
    def grad_3D(self, u, sym=False):
        """
        Return the three-dimensional gradient of a vector field.

        Parameters
        ----------
        u : Function Vector field
        sym : bool, optional If True, use a symmetric representation. Default: False

        Returns
        -------
        Tensor 3D gradient as a tensor
        """
        return self.reduit_to_3D(self.grad_reduit(u, sym=sym), sym=sym)
    
    def Eulerian_gradient(self, v, u):
        """
        Return the spatial velocity gradient, which is ∂v/∂x where ∂x denotes
        the derivative with respect to current coordinates.

        Parameters
        ----------
        v : Function Vector field to derive
        u : Function Displacement field defining the transformation

        Returns
        -------
        Expression Eulerian gradient
        """
        invF_reduit = self.invF_reduit(u)
        grad_red = self.grad_reduit(v)
        
        if self._is_1d():
            if self.name == "CartesianUD":
                return grad_red * invF_reduit[0]
            else:  # CylindricalUD or SphericalUD
                return as_vector([grad_red[i] * invF_reduit[i] for i in range(len(grad_red))])
        elif self.name == "PlaneStrain":
            return self.bidim_to_reduit(dot(self.reduit_to_2D(grad_red), inv(Identity(2) + grad(u))))
        elif self.name == "Axisymmetric":
            return self.tridim_to_reduit(dot(self.reduit_to_3D(grad_red), invF_reduit))
        else:  # Tridimensional
            return dot(grad_red, invF_reduit)
    
    # =========================================================================
    # Format conversion methods
    # =========================================================================
    
    def reduit_to_3D(self, red, sym = False):
        """
        Convert a tensor in reduced form to its full three-dimensional form.

        Parameters
        ----------
        red : Expression Field in reduced form
        sym : bool, optional If True, use a symmetric representation. Default: False

        Returns
        -------
        Tensor Corresponding 3D tensor
        """
        # 1D models
        if self.name == "CartesianUD":
            return as_tensor([[red, 0, 0], [0, 0, 0], [0, 0, 0]])
        elif self.name == "CylindricalUD":
            return as_tensor([[red[0], 0, 0], [0, red[1], 0], [0, 0, 0]])
        elif self.name == "SphericalUD":
            return as_tensor([[red[0], 0, 0], [0, red[1], 0], [0, 0, red[1]]])
        
        # 2D models
        elif self.name == "PlaneStrain":
            return self._plane_strain_to_3D(red, sym)
        elif self.name == "Axisymmetric":
            return self._Axisymmetric_to_3D(red, sym)
        
        # 3D model
        else:
            if sym:
                return as_tensor([[red[0], red[3], red[4]], 
                                  [red[3], red[1], red[5]], 
                                  [red[4], red[5], red[2]]])
            else:
                return red
    
    def _plane_strain_to_3D(self, red, sym):
        """
        Private method to convert from PlaneStrain to 3D.

        Parameters
        ----------
        red : Expression Reduced form tensor
        sym : bool If True, use a symmetric representation

        Returns
        -------
        Tensor 3D tensor
        """
        if sym:
            return as_tensor([[red[0], red[2], 0], [red[2], red[1], 0], [0, 0, 0]])
        else:
            return as_tensor([[red[0], red[2], 0], [red[3], red[1], 0], [0, 0, 0]])
    
    def _Axisymmetric_to_3D(self, red, sym):
        """
        Private method to convert from Axisymmetric to 3D.

        Parameters
        ----------
        red : Expression Reduced form tensor
        sym : bool If True, use a symmetric representation

        Returns
        -------
        Tensor 3D tensor
        
        Notes
        -----
        When r is small (r < 1e-3), L'Hôpital's rule is applied to handle
        the potential singularity near the axis. In this case, we approximate
        red[1] ≈ red[0], which is the theoretically correct limit as r→0.
        This avoids numerical instability while maintaining the physical meaning
        of the solution near the symmetry axis.
        """
        condition = ge(self.r, 1e-3)
        
        if sym:
            true_tens = as_tensor([[red[0], 0, red[3]], [0, red[1], 0], [red[3], 0, red[2]]])
            hop_tens = as_tensor([[red[0], 0, red[3]], [0, red[0], 0], [red[3], 0, red[2]]]) 
        else:
            true_tens = as_tensor([[red[0], 0, red[3]], [0, red[1], 0], [red[4], 0, red[2]]])
            hop_tens = as_tensor([[red[0], 0, red[3]], [0, red[0], 0], [red[4], 0, red[2]]]) 
            
        return conditional(condition, true_tens, hop_tens)
    
    def reduit_to_2D(self, red):
        """
        Convert a tensor in reduced form to its two-dimensional form.

        Parameters
        ----------
        red : Expression Tensor in reduced form

        Returns
        -------
        Tensor Corresponding 2D tensor
        """
        return as_tensor([[red[0], red[2]], [red[3], red[1]]])
    
    def bidim_to_reduit(self, tens2D):
        """
        Convert a 2D tensor to its reduced form.

        Parameters
        ----------
        tens2D : Tensor 2D tensor

        Returns
        -------
        Vector Corresponding reduced form
        """
        return as_vector([tens2D[0, 0], tens2D[1, 1], tens2D[0, 1], tens2D[1, 0]])
    
    def tridim_to_reduit(self, tens3D, sym=False):
        """
        Convert a 3D tensor to its reduced form.

        Parameters
        ----------
        tens3D : Tensor 3D tensor
        sym : bool, optional If True, use a symmetric representation. Default: False

        Returns
        -------
        Expression Corresponding reduced form
        """
        # 1D models
        if self.name == "CartesianUD":
            return tens3D[0, 0]
        elif self.name == "CylindricalUD":
            return as_vector([tens3D[0, 0], tens3D[1, 1]])
        elif self.name == "SphericalUD":
            return as_vector([tens3D[0, 0], tens3D[1, 1], tens3D[2, 2]])
        
        # 2D models
        elif self.name == "PlaneStrain":
            if sym:
                return as_vector([tens3D[0, 0], tens3D[1, 1], tens3D[0, 1]])
            else:
                return as_vector([tens3D[0, 0], tens3D[1, 1], tens3D[0, 1], tens3D[1, 0]])
        elif self.name == "Axisymmetric":
            if sym:
                return as_vector([tens3D[0, 0], tens3D[1, 1], tens3D[2, 2], tens3D[0, 2]])
            else:
                return as_vector([tens3D[0, 0], tens3D[1, 1], tens3D[2, 2], 
                                tens3D[0, 2], tens3D[2, 0]])
        
        # 3D model
        elif self.name == "Tridimensional":
            if sym:
                return as_vector([tens3D[0, 0], tens3D[1, 1], tens3D[2, 2], 
                                tens3D[0, 1], tens3D[0, 2], tens3D[1, 2]])
            else:
                return tens3D
    
    def tridim_to_mandel(self, tens3D):
        """
        Convert a 3D tensor to its Mandel notation representation.
        
        The off-diagonal terms are weighted by a factor √2.

        Parameters
        ----------
        tens3D : Tensor 3D symmetric tensor

        Returns
        -------
        Vector Mandel representation
        """
        sq2 = sqrt(2)
        
        # 1D models
        if self._is_1d():
            return as_vector([tens3D[0, 0], tens3D[1, 1], tens3D[2, 2]])
        
        # 2D models
        elif self.name == "PlaneStrain":
            return as_vector([tens3D[0, 0], tens3D[1, 1], tens3D[2, 2], sq2 * tens3D[0, 1]])
        elif self.name == "Axisymmetric":
            return as_vector([tens3D[0, 0], tens3D[1, 1], tens3D[2, 2], sq2 * tens3D[0, 2]])
        
        # 3D model
        elif self.name == "Tridimensional":
            return as_vector([tens3D[0, 0], tens3D[1, 1], tens3D[2, 2], 
                            sq2 * tens3D[0, 1], sq2 * tens3D[0, 2], sq2 * tens3D[1, 2]])
    
    def mandel_to_tridim(self, red):
        """
        Convert a Mandel representation to a 3D tensor.

        Parameters
        ----------
        red : Vector Mandel representation

        Returns
        -------
        Tensor Corresponding 3D tensor
        """
        sq2 = sqrt(2)
        
        # 1D models
        if self._is_1d():
            return as_tensor([[red[0], 0, 0], [0, red[1], 0], [0, 0, red[2]]])
        
        # 2D models
        elif self.name == "PlaneStrain":
            return as_tensor([[red[0], red[3]/sq2, 0], 
                             [red[3]/sq2, red[1], 0], 
                             [0, 0, red[2]]])
        elif self.name == "Axisymmetric":
            return as_tensor([[red[0], 0, red[3]/sq2], 
                             [0, red[1], 0], 
                             [red[3]/sq2, 0, red[2]]])
        
        # 3D model
        elif self.name == "Tridimensional":
            return as_tensor([[red[0], red[3]/sq2, red[4]/sq2], 
                             [red[3]/sq2, red[1], red[5]/sq2], 
                             [red[4]/sq2, red[5]/sq2, red[2]]])
    
    # =========================================================================
    # Transformation methods
    # =========================================================================
    
    def F_reduit(self, u):
        """
        Return the deformation gradient in reduced form.

        Parameters
        ----------
        u : Function Displacement field

        Returns
        -------
        Expression Reduced deformation gradient
        """
        # 1D models
        if self.name == "CartesianUD":
            return as_vector([1 + u.dx(0), 1, 1])
        elif self.name == "CylindricalUD":
            return as_vector([1 + u.dx(0), 1 + u/self.r, 1])
        elif self.name == "SphericalUD":
            return as_vector([1 + u.dx(0), 1 + u/self.r, 1 + u/self.r])
        
        # 2D and 3D models
        else:
            return Identity(3) + self.grad_3D(u)
    
    def F_3D(self, u):
        """
        Return the full 3D deformation gradient.

        Parameters
        ----------
        u : Function Displacement field

        Returns
        -------
        Tensor 3D deformation gradient
        """
        return Identity(3) + self.grad_3D(u)
    
    def J(self, u):
        """
        Return the Jacobian of the transformation (measure of local dilatation).

        Parameters
        ----------
        u : Function Displacement field

        Returns
        -------
        Expression Jacobian of the transformation
        """
        # 1D models
        if self.name == "CartesianUD":
            return 1 + u.dx(0)
        elif self.name == "CylindricalUD":
            return (1 + u.dx(0)) * (1 + u / self.r)
        elif self.name == "SphericalUD":
            return (1 + u.dx(0)) * (1 + u / self.r)**2
        
        # 2D models
        elif self.name == "PlaneStrain":
            return det(Identity(2) + grad(u))
        elif self.name == "Axisymmetric":
            F = self.F_reduit(u)
            return F[1, 1] * (F[0, 0] * F[2, 2] - F[0, 2] * F[2, 0])
        
        # 3D model
        else:
            return det(Identity(3) + grad(u))
    
    def invF_reduit(self, u):
        """
        Return the inverse of the deformation gradient in reduced form.

        Parameters
        ----------
        u : Function Displacement field

        Returns
        -------
        Expression Inverse of the deformation gradient
        """
        # 1D models
        if self.name == "CartesianUD":
            return as_vector([1 / (1 + u.dx(0)), 1, 1])
        elif self.name == "CylindricalUD":
            return as_vector([1 / (1 + u.dx(0)), 1/(1 + u/self.r), 1])
        elif self.name == "SphericalUD":
            return as_vector([1 / (1 + u.dx(0)), 1 /(1 + u/self.r), 1 / (1 + u/self.r)])
        
        # 2D models
        elif self.name == "PlaneStrain":
            inv_F2 = inv(Identity(2) + grad(u))
            return as_tensor([[inv_F2[0,0], inv_F2[0,1], 0], [inv_F2[1,0], inv_F2[1,1], 0], [0, 0, 1]])
        elif self.name == "Axisymmetric":
            return self._get_Axisymmetric_invF(u)
        
        # 3D model
        else:
            return inv(Identity(3) + grad(u))
    
    def _get_Axisymmetric_invF(self, u):
        """
        Private method to calculate the inverse deformation gradient in axisymmetric case.

        Parameters
        ----------
        u : Function Displacement field

        Returns
        -------
        Tensor Inverse deformation gradient
        
        Notes
        -----
        When r is small (r < 1e-3), L'Hôpital's rule is applied to handle
        the potential singularity near the axis. In this case, we approximate
        red[1] ≈ red[0], which is the theoretically correct limit as r→0.
        This avoids numerical instability while maintaining the physical meaning
        of the solution near the symmetry axis.
        """
        grad_u = grad(u)
        prefacteur = (1 + grad_u[0,0]) * (1 + grad_u[1,1]) - grad_u[0,1] * (1 + grad_u[1,0])
        condition = ge(self.r, 1e-3)
        
        true_inv = as_tensor([[(1 + grad_u[1,1]) / prefacteur, 0, -grad_u[0,1] / prefacteur],
                            [0, 1 / (1 + u[0]/self.r), 0],
                            [-grad_u[1,0] / prefacteur, 0, (1 + grad_u[0,0]) / prefacteur]])
        
        hop_inv = as_tensor([[(1 + grad_u[1,1]) / prefacteur, 0, -grad_u[0,1] / prefacteur],
                           [0, 1 / (1 + grad_u[0,0]), 0],
                           [-grad_u[1,0] / prefacteur, 0, (1 + grad_u[0,0]) / prefacteur]])
        
        return conditional(condition, true_inv, hop_inv)
    
    def relative_gradient_3D(self, u, u_old):
        """
        Return the relative deformation gradient between two configurations.

        Parameters
        ----------
        u : Function Current displacement field
        u_old : Function Previous displacement field

        Returns
        -------
        Tensor Relative deformation gradient
        """
        F_new = self.F_3D(u)
        inv_F_old = inv(self.F_3D(u_old))
        return dot(F_new, inv_F_old)
    
    # =========================================================================
    # Strain methods
    # =========================================================================
    
    def B(self, u):
        """
        Return the left Cauchy-Green tensor.

        Parameters
        ----------
        u : Function Displacement field

        Returns
        -------
        Expression Left Cauchy-Green tensor (form adapted to dimension)
        """
        F = self.F_reduit(u)
        if self._is_1d():
            return as_vector([F[0]**2, F[1]**2, F[2]**2])
        else:
            return dot(F, F.T)
    
    def B_3D(self, u):
        """
        Return the full 3D left Cauchy-Green tensor.

        Parameters
        ----------
        u : Function Displacement field

        Returns
        -------
        Tensor 3D left Cauchy-Green tensor
        """
        F = self.F_3D(u)
        return dot(F, F.T)
    
    def C_3D(self, u):
        """
        Return the full 3D right Cauchy-Green tensor.

        Parameters
        ----------
        u : Function Displacement field

        Returns
        -------
        Tensor 3D right Cauchy-Green tensor
        """
        F = self.F_3D(u)
        return dot(F.T, F)
    
    def BI(self, u):
        """
        Return the trace of the left Cauchy-Green tensor.

        Parameters
        ----------
        u : Function Displacement field

        Returns
        -------
        Expression Trace of the left Cauchy-Green tensor
        """
        B = self.B(u)
        if self._is_1d():
            return sum(B[i] for i in range(3))
        else:
            return tr(B)
    
    def BBarI(self, u):
        """
        Return the trace of the isovolumic left Cauchy-Green tensor.

        Parameters
        ----------
        u : Function Displacement field

        Returns
        -------
        Expression Trace of the isovolumic tensor
        """
        return self.BI(u) / self.J(u)**(2./3)
    
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

        Returns
        -------
        Expression Divergence of the field
        """
        if self._is_1d():
            return v.dx(0)
        else:
            return div(v)
    
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