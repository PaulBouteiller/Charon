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
Created on Wed Apr  2 11:25:33 2025

@author: bouteillerp

Newtonian Fluid Equation of State
================================

This module implements an equation of state for Newtonian fluids, combining 
pressure-volume relations with viscous behavior.

The model includes:
- Pressure-volume relationship based on isothermal compressibility
- Thermal expansion effects
- Sound speed calculation for acoustic phenomena

This equation of state is suitable for:
- Compressible fluid dynamics
- Acoustics and wave propagation in fluids
- Thermal-fluid systems
- Low to moderate strain rate fluid flows

Classes:
--------
NewtonianFluidEOS : Newtonian fluid equation of state
    Implements pressure-volume-temperature relations for compressible fluids
    Includes viscous and thermal effects
    Provides sound speed calculation
"""

from ufl import sqrt, ln
from .base_eos import BaseEOS

class NewtonianFluidEOS(BaseEOS):
    """Newtonian fluid equation of state.
    
    This model combines pressure-volume and viscous behavior.
    
    Attributes
    ----------
    k : float Volumetric viscosity
    alpha : float Thermal conductivity
    chiT : float Isothermal compressibility
    """
    
    def required_parameters(self):
        """Return the list of required parameters.
        
        Returns
        -------
        list List of parameter names
        """
        return ["k", "alpha", "chiT"]
    
    def __init__(self, params):
        """Initialize the Newtonian fluid EOS.
        
        Parameters
        ----------
        params : dict Dictionary containing:
                        k : float Volumetric viscosity
                        alpha : float Thermal conductivity
                        chiT : float Isothermal compressibility
        """
        super().__init__(params)
        
        # Store parameters
        self.k = params["k"]
        self.alpha = params["alpha"]
        self.chiT = params["chiT"]
        
        # Log parameters
        print(f"Volumetric viscosity: {self.k}")
        print(f"Thermal conductivity: {self.alpha}")
        print(f"Isothermal compressibility: {self.chiT}")
    
    def celerity(self, rho_0):
        """Calculate sound speed in fluid.
        
        Parameters
        ----------
        rho_0 : float Initial density
            
        Returns
        -------
        float Sound speed
        """
        return sqrt(1 / (self.chiT * rho_0))
    
    def pressure(self, J, T, T0, material, quadrature):
        """Calculate pressure for a Newtonian fluid.
        
        The total pressure includes both thermodynamic and viscous components.
        
        Parameters
        ----------
        J, T, T0, material : See stress_3D method in ConstitutiveLaw.py for details.
        quadrature : QuadratureHandler Handler for quadrature spaces.
            
        Returns
        -------
        Expression Pressure
        """
        # Thermodynamic pressure
        thermo_p = -1 / self.chiT * ln(J) + self.alpha / self.chiT * (T - T0)
        
        # Viscous pressure from velocity gradient
        # viscous_p = -self.k * kinematic.div(v)
        viscous_p = 0
        
        return thermo_p + viscous_p