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
Created on Wed Apr  2 11:35:18 2025

@author: bouteillerp

Isotropic Linear Elastic Deviatoric Stress Model
===============================================

This module implements the isotropic linear elastic deviatoric stress model under 
the small strain assumption. It provides the classical linear elastic stress-strain
relationship for the deviatoric part of the stress tensor.

The model is based on Hooke's law for isotropic materials, where the deviatoric stress
is proportional to the deviatoric strain. This formulation is suitable for small
deformation problems and serves as the foundation for more complex models.

Key features:
- Implementation of small strain linear elasticity
- Support for both direct shear modulus specification and E/ν parameterization
- Calculation of deviatoric stress as s = 2μ·dev(ε)
- Implementation of isochoric Helmholtz energy for phase field coupling

Classes:
--------
IsotropicHPPDeviator : Isotropic linear elastic deviatoric model
    Implements linear elastic deviatoric stress under small strain
    Supports both μ-based and E/ν-based parameterization
    Provides Helmholtz energy calculation for coupling with damage models
"""

from ufl import sym, dev, tr, dot
from .base_deviator import BaseDeviator

class IsotropicHPPDeviator(BaseDeviator):
    """Isotropic linear elastic deviatoric stress model under small strain hypothesis.
    
    This model implements the standard linear elastic deviatoric stress:
    s = 2 * μ * dev(ε)
    
    where ε is the small strain tensor.
    
    Attributes
    ----------
    mu : float Shear modulus (Pa)
    E : float, optional Young's modulus (Pa)
    nu : float, optional Poisson's ratio
    """
    
    def required_parameters(self):
        """Return the list of required parameters.
        
        Returns
        -------
        list List with either "mu" or both "E" and "nu"
        """
        return ["mu"] if "mu" in self.params else ["E", "nu"]
    
    def __init__(self, params):
        """Initialize the isotropic linear elastic deviatoric model.
        
        Parameters
        ----------
        params : dict
            Dictionary containing either:
            mu : float Shear modulus (Pa)
            OR:
            E : float Young's modulus (Pa)
            nu : float Poisson's ratio
        """
        self.params = params  # Store for required_parameters method
        super().__init__(params)
        
        # Initialize parameters
        self.mu = params.get("mu")
        if self.mu is None:
            self.E = params["E"]
            self.nu = params["nu"]
            self.mu = self.E / (2 * (1 + self.nu))
            print(f"Young's modulus: {self.E}")
            print(f"Poisson's ratio: {self.nu}")
            
        print(f"Shear modulus: {self.mu}")
    
    def calculate_stress(self, u, v, J, T, T0, kinematic):
        """Calculate the deviatoric stress tensor for linear elasticity.
        
        s = 2 * μ * dev(sym(∇u))
        
        Parameters
        ----------
        u, v, J, T, T0 : Function See stress_3D method in ConstitutiveLaw.py for details.
        kinematic : Kinematic Kinematic handler object
            
        Returns
        -------
        ufl.core.expr.Expr Deviatoric stress tensor
        """
        # Calculate strain tensor: ε = sym(∇u)
        epsilon = sym(kinematic.grad_3D(u))
        
        # Calculate deviatoric stress: s = 2μ * dev(ε)
        return 2 * self.mu * dev(epsilon)
    
    def isochoric_helmholtz_energy(self, u, kinematic):
        """Calculate the isochoric Helmholtz free energy for isotropic linear elasticity.
        
        Parameters
        ----------
        u, J, kinematic : See Helmholtz_energy method in ConstitutiveLaw.py for details.
            
        Returns
        -------
        Isochoric Helmholtz free energy
        """
        dev_eps = dev(sym(kinematic.grad_3D(u)))
        return self.mu * tr(dot(dev_eps, dev_eps))