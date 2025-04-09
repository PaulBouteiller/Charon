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
Created on Wed Apr  2 11:23:08 2025

@author: bouteillerp

Jones-Wilkins-Lee (JWL) Equation of State for Explosives
=======================================================

This module implements the Jones-Wilkins-Lee (JWL) equation of state, which is widely
used for modeling the behavior of detonation products from high explosives. The JWL
model accurately represents the pressure-volume-energy relationship of explosion
gases over a wide range of expansions.

The JWL equation of state is particularly important in computational physics for:
- Explosive detonation modeling
- Blast wave simulations
- High-energy impact problems
- Shock physics applications

Classes:
--------
JWLEOS : JWL equation of state implementation
    Models pressure-volume-energy relation for detonation products
    Supports temperature-dependent behavior
    Provides wave speed calculation for detonation simulations
    
References:
-----------
- Lee, E.L., Hornig, H.C., Kury, J.W. (1968). "Adiabatic Expansion of High Explosive
  Detonation Products." University of California, Lawrence Radiation Laboratory, 
  Livermore, Report UCRL-50422.
- R. Menikoff. Jwl equation of state. Technical report, Los Alamos National Lab.(LANL),
  Los Alamos, NM (United States), 2015.
"""

from ufl import exp, sqrt
from .base_eos import BaseEOS

class JWLEOS(BaseEOS):
    """Jones-Wilkins-Lee (JWL) equation of state for detonation products.
    
    This EOS is commonly used for modeling the expansion of explosive detonation products.
    
    Attributes
    ----------
    A, B : float Energy coefficients
    R1, R2 : float Rate coefficients
    w : float Fraction of the energy coefficient
    """
    
    def required_parameters(self):
        """Return the list of required parameters.
        
        Returns
        -------
        list List of parameter names
        """
        return ["A", "R1", "B", "R2", "w"]
    
    def __init__(self, params):
        """Initialize the JWL EOS.
        
        Parameters
        ----------
        params : dict
            Dictionary containing:
            A : float First energy coefficient
            R1 : float First rate coefficient
            B : float Second energy coefficient
            R2 : float Second rate coefficient
            w : float Fraction of the energy coefficient
        """
        super().__init__(params)
        
        # Store parameters
        self.A = params["A"]
        self.R1 = params["R1"]
        self.B = params["B"]
        self.R2 = params["R2"]
        self.w = params["w"]
        
        # Log parameters
        print(f"Coefficient A: {self.A}")
        print(f"Coefficient R1: {self.R1}")
        print(f"Coefficient B: {self.B}")
        print(f"Coefficient R2: {self.R2}")
        print(f"Coefficient w: {self.w}")
    
    def celerity(self, rho_0):
        """Calculate wave velocity in material.
        
        Parameters
        ----------
        rho_0 : float Initial density
            
        Returns
        -------
        float Wave speed
        """
        return sqrt((self.A * self.R1 * exp(-self.R1) + self.B * self.R2 * exp(-self.R2)) / rho_0)
    
    def pressure(self, J, T, T0, material, quadrature):
        """Calculate pressure using the JWL EOS.
        
        P = A * exp(-R1 * J) + B * exp(-R2 * J) + w * rho_0 / J * C_mass * T
        
        Parameters
        ----------
        J, T, T0, material : See stress_3D method in ConstitutiveLaw.py for details.
        quadrature : QuadratureHandler Handler for quadrature spaces.
            
        Returns
        -------
        ufl.algebra.Sum Pressure
        """
        return self.A * exp(-self.R1 * J) + self.B * exp(-self.R2 * J) + self.w * material.rho_0 / J * material.C_mass * T