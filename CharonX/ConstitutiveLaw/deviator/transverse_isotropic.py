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
Created on Wed Apr  2 11:36:59 2025

@author: bouteillerp
"""
"""Transversely isotropic hyperelastic deviatoric stress models."""

from ufl import dev, dot, inner, outer, Identity
from .base_deviator import BaseDeviator

class NeoHookTransverseDeviator(BaseDeviator):
    """Transversely isotropic Neo-Hookean hyperelastic deviatoric stress model.
    
    This model extends the Neo-Hookean model to include directional stiffness:
    s = μ/J^(5/3) * dev(B) + additional transverse term
    
    Attributes
    ----------
    mu : float
        Primary shear modulus (Pa)
    mu_T : float
        Transverse shear modulus (Pa)
    """
    
    def required_parameters(self):
        """Return the list of required parameters.
        
        Returns
        -------
        list
            List with "mu" and "mu_T"
        """
        return ["mu", "mu_T"]
    
    def __init__(self, params):
        """Initialize the transversely isotropic Neo-Hookean deviatoric model.
        
        Parameters
        ----------
        params : dict
            Dictionary containing:
            mu : float
                Primary shear modulus (Pa)
            mu_T : float
                Transverse shear modulus (Pa)
        """
        super().__init__(params)
        
        # Store parameters
        self.mu = params["mu"]
        self.mu_T = params["mu_T"]
        
        # Log parameters
        print(f"Shear modulus: {self.mu}")
        print(f"Transverse shear modulus: {self.mu_T}")
    
    def calculate_stress(self, u, v, J, T, T0, kinematic):
        """Calculate the deviatoric stress tensor for transversely isotropic Neo-Hookean model.
        
        Parameters
        ----------
        u : Function
            Displacement field
        v : Function
            Velocity field (unused)
        J : Function
            Jacobian of the deformation
        T : Function
            Current temperature (unused)
        T0 : Function
            Initial temperature (unused)
        kinematic : Kinematic
            Kinematic handler object
            
        Returns
        -------
        Function
            Deviatoric stress tensor
        """
        B = kinematic.B_3D(u)
        nt = kinematic.actual_anisotropic_direction(u)
        Nt = outer(nt, nt)
        I1B = inner(B, Nt)
        BBarI = J**(-2./3) * tr(B)
        symBNt = dot(B, Nt) + dot(Nt, B)
        
        # Isotropic Neo-Hookean contribution
        s_iso = self.mu / J**(5./3) * dev(B)
        
        # Transverse contribution
        s_transverse = 2 * self.mu_T / J**(5./3) * (symBNt - I1B * (Nt + 1./3 * Identity(3)))
        s_transverse *= (I1B - 1) + (BBarI - 3)
        
        return s_iso + s_transverse


class LuTransverseDeviator(BaseDeviator):
    """Lu's transversely isotropic hyperelastic deviatoric stress model.
    
    This model implements Lu's formulation for transversely isotropic materials,
    particularly suitable for biological tissues.
    
    Attributes
    ----------
    k2 : float
        Linear stretch stiffness
    k3 : float
        Transverse shear stiffness
    k4 : float
        In-plane shear stiffness
    c : float
        Exponential parameter
    """
    
    def required_parameters(self):
        """Return the list of required parameters.
        
        Returns
        -------
        list
            List of parameter names
        """
        return ["k2", "k3", "k4", "c"]
    
    def __init__(self, params):
        """Initialize Lu's transversely isotropic deviatoric model.
        
        Parameters
        ----------
        params : dict
            Dictionary containing stiffness and shape parameters
        """
        super().__init__(params)
        
        # Store parameters
        self.k2 = params["k2"]
        self.k3 = params["k3"]
        self.k4 = params["k4"]
        self.c = params["c"]
        
        # Log parameters
        print(f"Linear stretch coefficient: {self.k2}")
        print(f"Transverse shear stiffness: {self.k3}")
        print(f"In-plane shear stiffness: {self.k4}")
    
    def calculate_stress(self, u, v, J, T, T0, kinematic):
        """Calculate the deviatoric stress tensor for Lu's transversely isotropic model.
        
        Parameters
        ----------
        u : Function
            Displacement field
        v : Function
            Velocity field (unused)
        J : Function
            Jacobian of the deformation
        T : Function
            Current temperature (unused)
        T0 : Function
            Initial temperature (unused)
        kinematic : Kinematic
            Kinematic handler object
            
        Returns
        -------
        Function
            Deviatoric stress tensor
        """
        from ufl import exp, sqrt
        
        B = kinematic.B_3D(u)
        C = kinematic.C_3D(u)
        N0 = outer(kinematic.n0, kinematic.n0)
        nt = kinematic.actual_anisotropic_direction(u)
        Nt = outer(nt, nt)
        
        I1C = inner(C, N0)
        I1B = inner(B, Nt)
        lmbda = sqrt(I1C)
        lambdabar = J**(-1/3) * lmbda
        symBNt = dot(B, Nt) + dot(Nt, B)
        
        # Directional stiffness contribution
        s2 = 2 * self.k2 * self.c * lambdabar * (lambdabar - 1) / J * exp(self.c*(lambdabar - 1)**2) * dev(Nt)
        
        # Transverse shear contribution
        s3 = self.k3 / (I1C * J) * (symBNt - 2 * I1B * Nt)
        
        # In-plane shear contribution
        s4 = self.k4 * lmbda / (2 * J**2) * (2 * B - 2 * symBNt - (tr(B) - I1B)*(Identity(3) - Nt))
        
        return s2 + s3 + s4