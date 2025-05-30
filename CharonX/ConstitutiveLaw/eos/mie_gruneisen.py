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
Created on Wed Apr  2 11:23:58 2025

@author: bouteillerp

Mie-Grüneisen Family of Equations of State
=========================================

This module implements the Mie-Grüneisen family of equations of state, which are
widely used for modeling materials under shock compression. These models combine
a reference curve (often a Hugoniot) with a thermal term proportional to the
Grüneisen parameter.

The Mie-Grüneisen approach is fundamental in shock physics and provides a versatile
framework for describing materials from moderate to high pressures. Different variants
of the basic model offer varying degrees of sophistication and applicability ranges.

The module includes:
- Standard Mie-Grüneisen EOS with polynomial reference curve
- Extended Mie-Grüneisen (xMG) with improved high-pressure behavior
- Puff Mie-Grüneisen (PMG) with specialized formulation for porous materials

Key applications:
- Shock wave propagation
- Impact dynamics
- Explosive loading
- High-pressure material behavior

Classes:
--------
MGEOS : Standard Mie-Grüneisen equation of state
    Basic formulation with polynomial reference curve
    
xMGEOS : Extended Mie-Grüneisen equation of state
    Enhanced formulation for improved high-pressure behavior
    
PMGEOS : Puff Mie-Grüneisen equation of state
    Specialized variant for porous materials

References:
-----------
- Grüneisen, E. (1912). "Theorie des festen Zustandes einatomiger Elemente."
  Annalen der Physik, 344(12), 257-306.
- Meyers, M.A. (1994). "Dynamic Behavior of Materials." John Wiley & Sons.
- Steinberg, D.J. (1996). "Equation of State and Strength Properties of Selected
  Materials." Lawrence Livermore National Laboratory.
"""

from ufl import sqrt
from .base_eos import BaseEOS

class MGEOS(BaseEOS):
    """Standard Mie-Grüneisen equation of state.
    
    This EOS combines a reference curve with a thermal term.
    
    Attributes
    ----------
    C : float Linear coefficient
    D : float Quadratic coefficient
    S : float Cubic coefficient
    gamma0 : float Grüneisen coefficient
    """
    
    def required_parameters(self):
        """Return the list of required parameters.
        
        Returns
        -------
        list List of parameter names
        """
        return ["C", "D", "S", "gamma0"]
    
    def __init__(self, params):
        """Initialize the Mie-Grüneisen EOS.
        
        Parameters
        ----------
        params : dict Dictionary containing:
                    C : float Linear coefficient
                    D : float Quadratic coefficient
                    S : float Cubic coefficient
                    gamma0 : float Grüneisen coefficient
        """
        super().__init__(params)
        
        # Store parameters
        self.C = params["C"]
        self.D = params["D"]
        self.S = params["S"]
        self.gamma0 = params["gamma0"]
        
        # Log parameters
        print(f"Linear coefficient: {self.C}")
        print(f"Quadratic coefficient: {self.D}")
        print(f"Cubic coefficient: {self.S}")
        print(f"Gamma0 coefficient: {self.gamma0}")
    
    def celerity(self, rho_0):
        """Calculate wave velocity in material.
        
        Parameters
        ----------
        rho_0 : float Initial density
            
        Returns
        -------
        float Wave speed
        """
        return sqrt(self.C / rho_0)
    
    def pressure(self, J, T, T0, material, quadrature):
        """Calculate pressure using the Mie-Grüneisen EOS.
        
        P = C * μ + D * μ² + S * μ³ + rho_0 / J * gamma0 * (T - T0)
        
        where μ = 1/J - 1 is the compression strain.
        
        Parameters
        ----------
        J, T, T0, material : See stress_3D method in ConstitutiveLaw.py for details.
        quadrature : QuadratureHandler Handler for quadrature spaces.
            
        Returns
        -------
        float or Function Pressure
        """
        mu = 1/J - 1
        return self.C * mu + self.D * mu**2 + self.S * mu**3 + material.rho_0 / J * self.gamma0 * (T - T0)


class xMGEOS(BaseEOS):
    """Extended Mie-Grüneisen equation of state.
    
    This variant adds more terms to better capture high-pressure behavior.
    
    Attributes
    ----------
    c0 : float Sound speed at zero pressure
    gamma0 : float Grüneisen coefficient
    s1, s2, s3 : float Empirical parameters
    b : float Volumetric parameter
    """
    
    def required_parameters(self):
        """Return the list of required parameters.
        Returns
        -------
        list List of parameter names
        """
        return ["c0", "gamma0", "s1", "s2", "s3", "b"]
    
    def __init__(self, params):
        """Initialize the extended Mie-Grüneisen EOS.
        Parameters
        ----------
        params : dict Dictionary with extended Mie-Grüneisen parameters
        """
        super().__init__(params)
        
        # Store parameters
        self.c0 = params["c0"]
        self.gamma0 = params["gamma0"]
        self.s1 = params["s1"]
        self.s2 = params["s2"]
        self.s3 = params["s3"]
        self.b = params["b"]
        
        # Log parameters
        print(f"Gamma0 coefficient: {self.gamma0}")
        print(f"s1 coefficient: {self.s1}")
        print(f"s2 coefficient: {self.s2}")
        print(f"s3 coefficient: {self.s3}")
        print(f"b coefficient: {self.b}")
        print(f"Estimated elastic wave speed: {self.c0}")
    
    def celerity(self, rho_0):
        """Return the specified sound speed.
        Parameters
        ----------
        rho_0 : float Initial density (unused, kept for interface consistency)
        Returns
        -------
        float Sound speed
        """
        return self.c0
    
    def pressure(self, J, T, T0, material, quadrature):
        """Calculate pressure using the extended Mie-Grüneisen EOS.
        
        This variant has a more complex reference curve and includes
        non-linear effects for high compressions.
        
        Parameters
        ----------
        J, T, T0, material : See stress_3D method in ConstitutiveLaw.py for details.
        quadrature : QuadratureHandler Handler for quadrature spaces.
            
        Returns
        -------
        Expression Pressure
        """
        from ..utils.generic_functions import ppart, npart
        
        mu = 1 / J - 1
        # Complete model
        numerator_pos = material.rho_0 * self.c0**2 * ppart(mu) * (1 + (1 - self.gamma0 / 2) * mu - self.b / 2 * mu**2)
        denominator_pos = 1 - (self.s1 - 1) * mu - self.s2 * mu**2 * J - self.s3 * mu**3 * J**2
        
        # Reduced model
        part_neg = material.rho_0 * self.c0**2 * npart(mu)
        thermal = material.rho_0 / J * self.gamma0 * T
        
        return numerator_pos / denominator_pos + part_neg + thermal


class PMGEOS(BaseEOS):
    """Puff Mie-Grüneisen equation of state.
    
    This variant uses a polynomial form for the reference curve.
    
    Attributes
    ----------
    Pa : float Atmospheric pressure
    Gamma0 : float Grüneisen coefficient
    D, S, H : float Polynomial coefficients
    c0 : float Sound speed
    """
    
    def required_parameters(self):
        """Return the list of required parameters. 
        Returns
        -------
        list List of parameter names
        """
        return ["Pa", "Gamma0", "D", "S", "H", "c0"]
    
    def __init__(self, params):
        """Initialize the polynomial Mie-Grüneisen EOS.
        Parameters
        ----------
        params : dict Dictionary with polynomial Mie-Grüneisen parameters
        """
        super().__init__(params)
        
        # Store parameters
        self.Pa = params["Pa"]
        self.Gamma0 = params["Gamma0"]
        self.D = params["D"]
        self.S = params["S"]
        self.H = params["H"]
        self.c0 = params["c0"]
        
        # Log parameters
        print(f"Atmospheric pressure: {self.Pa}")
        print(f"Gamma0 coefficient: {self.Gamma0}")
        print(f"D coefficient: {self.D}")
        print(f"S coefficient: {self.S}")
        print(f"H coefficient: {self.H}")
        print(f"c0 coefficient: {self.c0}")
    
    def celerity(self, rho_0):
        """Return the specified sound speed.
        Parameters
        ----------
        rho_0 : float Initial density (unused, kept for interface consistency)
        Returns
        -------
        float Sound speed
        """
        return self.c0