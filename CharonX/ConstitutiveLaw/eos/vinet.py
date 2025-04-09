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
Created on Wed Apr  2 13:23:51 2025

@author: bouteillerp

Vinet Equation of State for Solids
=================================

This module implements the Vinet equation of state, a physically motivated model
for solids under compression. The Vinet EOS is particularly well-suited for
high-pressure simulations and offers improved accuracy compared to polynomial
models for many materials.

The Vinet model is derived from interatomic potentials and provides a more
accurate description of solids at high compression than other EOS models.
It performs especially well for metals and other materials where the repulsive
interactions between atoms become significant under compression.

Classes:
--------
VinetEOS : Vinet equation of state implementation
    Physics-based EOS suitable for solids
    Accounts for temperature effects on material parameters
    Provides accurate pressure calculation at high compressions
    
References:
-----------
- P. Vinet, J. R. Smith, J. Ferrante, and J. H. Rose. Temperature effects on the universal
  equation of state of solids. Physical Review B, 35(4) :1945, 1987.
"""


from ufl import exp, sqrt
from .base_eos import BaseEOS

class VinetEOS(BaseEOS):
    """Vinet equation of state for solids.
    
    The Vinet EOS is well-suited for highly compressed materials
    and is often used in high-pressure physics.
    
    Attributes
    ----------
    iso_T_K0 : float Isothermal bulk modulus
    T_dep_K0 : float Temperature dependence of K0
    iso_T_K1 : float Isothermal pressure derivative of K0
    T_dep_K1 : float Temperature dependence of K1
    """
    
    def required_parameters(self):
        """Return the list of required parameters.
        
        Returns
        -------
        list List of parameter names
        """
        return ["iso_T_K0", "T_dep_K0", "iso_T_K1", "T_dep_K1"]
    
    def __init__(self, params):
        """Initialize the Vinet EOS.
        
        Parameters
        ----------
        params : dict
            Dictionary containing:
            iso_T_K0 : float Isothermal bulk modulus
            T_dep_K0 : float Temperature dependence of K0
            iso_T_K1 : float Isothermal pressure derivative of K0
            T_dep_K1 : float Temperature dependence of K1
        """
        super().__init__(params)
        
        # Store parameters
        self.iso_T_K0 = params["iso_T_K0"]
        self.T_dep_K0 = params["T_dep_K0"]
        self.iso_T_K1 = params["iso_T_K1"]
        self.T_dep_K1 = params["T_dep_K1"]
        
        # Log parameters
        print(f"Isothermal bulk modulus K0: {self.iso_T_K0}")
        print(f"Isothermal coefficient K1: {self.iso_T_K1}")
    
    def celerity(self, rho_0):
        """Calculate wave velocity in material.
        
        Parameters
        ----------
        rho_0 : float Initial density
            
        Returns
        -------
        float Wave speed
        """
        return sqrt(self.iso_T_K0 / rho_0)
    
    def pressure(self, J, T, T0, material, quadrature):
        """Calculate pressure using the Vinet EOS.
        
        P = 3 * K0 * J^(-2/3) * (1 - J^(1/3)) * exp(3/2 * (K1 - 1) * (1 - J^(1/3)))
        
        where K0 and K1 depend on temperature.
        
        Parameters
        ----------
        J, T, T0, material : See stress_3D method in ConstitutiveLaw.py for details.
        quadrature : QuadratureHandler Handler for quadrature spaces.
            
        Returns
        -------
        Expression Pressure
        """
        K0 = self.iso_T_K0 + self.T_dep_K0 * (T - T0)
        K1 = self.iso_T_K1 + self.T_dep_K1 * (T - T0)
        return 3 * K0 * J**(-2./3) * (1 - J**(1./3)) * exp(3./2 * (K1 - 1) * (1 - J**(1./3)))