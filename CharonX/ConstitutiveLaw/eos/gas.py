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
Created on Wed Apr  2 11:24:51 2025

@author: bouteillerp

Ideal Gas Equation of State
==========================

This module implements the ideal gas equation of state and related models for 
compressible gas dynamics. It provides the classical pressure-volume-temperature
relation for ideal gases.

Key features:
- Implementation of the standard ideal gas law: P = (γ-1) * ρ * e
- Wave speed calculation for acoustic phenomena
- Temperature-dependent internal energy relationship

Classes:
--------
GPEOS : Ideal gas equation of state
    Implements the standard gamma-law for ideal gases
    Provides wave speed calculation for acoustic simulations
    Handles temperature-dependent behavior
"""


from math import sqrt
from .base_eos import BaseEOS

class GPEOS(BaseEOS):
    """Ideal gas equation of state.
    
    This implements the classic ideal gas law: P = (γ-1) * ρ * e
    where e is the specific internal energy.
    
    Attributes
    ----------
    gamma : float Polytropic coefficient
    e_max : float Estimated maximum specific internal energy
    """
    
    def required_parameters(self):
        """Return the list of required parameters.
        Returns
        -------
        list List of parameter names
        """
        return ["gamma", "e_max"]
    
    def __init__(self, params):
        """Initialize the ideal gas EOS.
        
        Parameters
        ----------
        params : dict Dictionary containing:
                        gamma : float Polytropic coefficient
                        e_max : float Estimated maximum specific internal energy
        """
        super().__init__(params)
        
        # Store parameters
        self.gamma = params["gamma"]
        self.e_max = params["e_max"]
        
        # Log parameters
        print(f"Polytropic coefficient: {self.gamma}")
        print(f"Estimated maximum specific internal energy: {self.e_max}")
    
    def celerity(self, rho_0):
        """Calculate estimated sound speed in gas.
        Parameters
        ----------
        rho_0 : float Initial density
        Returns
        -------
        float Estimated sound speed
        """
        return sqrt(self.gamma * (self.gamma - 1) * self.e_max)
    
    def pressure(self, J, T, T0, material, quadrature):
        """Calculate pressure using the ideal gas law.
        P = (gamma - 1) * rho_0 / J * C_mass * T
        
        where C_mass *T  is equivalent to the specific internal energy.
        
        Parameters
        ----------
        J, T, T0, material : See stress_3D method in ConstitutiveLaw.py for details.
        quadrature : QuadratureHandler Handler for quadrature spaces.
        Returns
        -------
        Expression Pressure
        """
        return (self.gamma - 1) * material.rho_0 / J * material.C_mass * T