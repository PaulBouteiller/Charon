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
Created on Wed Apr  2 11:20:12 2025

@author: bouteillerp
"""
"""Isotropic linear elastic equation of state under small strain hypothesis."""

from math import sqrt
from ufl import sym, tr
from .base_eos import BaseEOS
from ...utils.generic_functions import ppart

class IsotropicHPPEOS(BaseEOS):
    """Isotropic linear elastic equation of state under small strain hypothesis.
    
    This model implements a simple linear relationship between volume change and pressure.
    
    Attributes
    ----------
    E : float Young's modulus (Pa)
    nu : float Poisson's ratio
    kappa : float Bulk modulus (Pa)
    alpha : float Thermal expansion coefficient (1/K)
    """
    
    def required_parameters(self):
        """Return the list of required parameters.
        
        Returns
        -------
        list List of parameter names
        """
        return ["E", "nu", "alpha"]
    
    def __init__(self, params):
        """Initialize the isotropic linear elastic EOS.
        
        Parameters
        ----------
        params : dict
            Parameters including E (Young's modulus), nu (Poisson's ratio),
            and alpha (thermal expansion coefficient)
        """
        super().__init__(params)
        
        # Store parameters
        self.E = params["E"]
        self.nu = params["nu"]
        self.alpha = params["alpha"]
        
        # Calculate bulk modulus
        self.kappa = self.E / (3 * (1 - 2 * self.nu))
        
        # Log parameters
        self._log_parameters()
    
    def _log_parameters(self):
        """Log the material parameters."""
        print(f"Poisson's ratio: {self.nu}")
        print(f"Bulk modulus: {self.kappa}")
        print(f"Young's modulus: {self.E}")
        print(f"Thermal expansion coefficient: {self.alpha}")
    
    def celerity(self, rho_0):
        """Calculate the elastic wave speed.
        
        Parameters
        ----------
        rho_0 : float Initial density
            
        Returns
        -------
        float Elastic wave speed
        """
        return sqrt(self.E / rho_0)
    
    def pressure(self, J, T, T0, material, quadrature):
        """Calculate pressure using the linear elastic model.
        
        P = -kappa * (J - 1 - 3 * alpha * (T - T0))
        
        Parameters
        ----------
        J, T, T0, material : See stress_3D method in ConstitutiveLaw.py for details.
        quadrature : QuadratureHandler Handler for quadrature spaces.
            
        Returns
        -------
        float or Function Pressure
        """
        return -self.kappa * (J - 1 - 3 * self.alpha * (T - T0))
    
    def volumetric_helmholtz_energy(self, u, J, kinematic):
        """Calculate the volumetric Helmholtz free energy for isotropic linear elasticity.
        
        Parameters
        ----------
        u, J, kinematic : See Helmholtz_energy method in ConstitutiveLaw.py for details.
            
        Returns
        -------
        Volumetric Helmholtz free energy
        """
        eps = sym(kinematic.grad_3D(u))
        E1 = tr(eps)
        return self.kappa / 2 * E1 * ppart(E1)