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
Created on Wed Apr  2 11:23:34 2025

@author: bouteillerp

MACAW Equation of State for Materials Under Extreme Conditions
============================================================

This module implements the MACAW equation of state, which is designed for 
materials under extreme conditions of pressure and temperature.  The MACAW model 
provides an accurate representation of material behavior from ambient conditions 
to extreme states encountered in shock and high-energy density physics.

The MACAW equation combines:
- A cold compression curve for high-pressure behavior
- Thermal contributions with anharmonic effects
- Temperature and volume-dependent thermodynamic properties

This advanced EOS is particularly suited for:
- Shock-dominated problems
- Materials under extreme pressures
- High temperature phenomena
- Phase transitions

Classes:
--------
MACAWEOS : MACAW equation of state implementation
    Comprehensive model for materials under extreme conditions
    Handles both cold compression and thermal contributions
    Provides accurate wave speed calculation
    Supports temperature-dependent behavior
    
References:
-----------
- E. Lozano and T. D. Aslam. A robust three-parameter reference curve for condensed phase
    materials. Journal of Applied Physics, 131(1), 2022. .
- E. Lozano, M. J. Cawkwell, and T. D. Aslam. An analytic and complete equation of state
for condensed phase materials. Journal of Applied Physics, 134(12), 2023. 
"""

from ufl import exp, sqrt
from .base_eos import BaseEOS

class MACAWEOS(BaseEOS):
    """MACAW equation of state for materials under extreme conditions.
    
    This is a sophisticated model for materials under high pressures and temperatures.
    
    Attributes
    ----------
    A, B, C : float Cold curve parameters
    cvinf, a0, vinf : float Thermal parameters
    n, theta0, Gamma0, Gammainf, m : float Thermal and anharmonic parameters
    rho0 : float Reference density
    """
    
    def required_parameters(self):
        """Return the list of required parameters.
        
        Returns
        -------
        list List of parameter names
        """
        return ["A", "B", "C", "eta", "vinf", "rho0", "theta0", "a0", "m", "n", 
                "Gammainf", "Gamma0", "cvinf"]
    
    def __init__(self, params):
        """Initialize the MACAW EOS.
        
        Parameters
        ----------
        params : dict Dictionary with MACAW parameters
        """
        super().__init__(params)
        
        # Store parameters - cold curve
        self.A = params["A"]
        self.B = params["B"]
        self.C = params["C"]
        
        # Store parameters - thermal
        self.eta = params["eta"]
        self.vinf = params["vinf"]
        self.rho0 = params["rho0"]
        self.theta0 = params["theta0"]
        self.a0 = params["a0"]
        self.m = params["m"]
        self.n = params["n"]
        self.Gammainf = params["Gammainf"]
        self.Gamma0 = params["Gamma0"]
        self.cvinf = params["cvinf"]
        
        # Log parameters
        print(f"Coefficient A: {self.A}")
        print(f"Coefficient B: {self.B}")
        print(f"Coefficient C: {self.C}")
        print(f"Coefficient eta: {self.eta}")
        print(f"Coefficient theta0: {self.theta0}")
        print(f"Coefficient a0: {self.a0}")
        print(f"Coefficient m: {self.m}")
        print(f"Coefficient n: {self.n}")
        print(f"Coefficient Gammainf: {self.Gammainf}")
        print(f"Coefficient Gamma0: {self.Gamma0}")
    
    def celerity(self, rho_0):
        """Calculate wave velocity in material.
        
        Parameters
        ----------
        rho_0 : float Initial density
            
        Returns
        -------
        float Wave speed
        """
        kappa = self.A * (self.B - 1./2 * self.C + (self.B + self.C)**2)
        print(f"Cold compressibility modulus: {kappa}")
        return sqrt(kappa / rho_0)
    
    def pressure(self, J, T, T0, material, quadrature):
        """Calculate pressure using the MACAW EOS.
        
        This method implements the complex MACAW EOS that combines
        cold pressure and thermal pressure components.
        
        Parameters
        ----------
        J, T, T0, material : See stress_3D method in ConstitutiveLaw.py for details.
        quadrature : QuadratureHandler Handler for quadrature spaces.
            
        Returns
        -------
        Expression Pressure
        """
        def thetav(theta0, v, vinf, gamma0, gammainf, m):
            """Helper function for MACAW thermal calculations."""
            eta = v/vinf
            beta = (gammainf - gamma0) / m
            theta = theta0*(eta)**(-gamma0)*((eta)**(-m) + 1)**(beta)
            return theta

        def dthetadv(theta0, v, vinf, gamma0, gammainf, m):
            """Derivative of theta with respect to v."""
            eta=v/vinf
            th=thetav(theta0, v, vinf, gamma0, gammainf, m)
            dtheta=-(th/eta)*(gamma0+(gammainf-gamma0)/(1.+(eta**m)))
            return dtheta/vinf

        def av(a0, vinf, v, n):
            """Another helper function for MACAW."""
            eta=v/vinf
            a=a0/(1.+(eta)**(-n))
            return a

        def dadv(a0, vinf, v, n):
            """Derivative of a with respect to v."""
            eta=v/vinf
            da=(n/eta)*a0*(eta**n)/((1.+eta**n)**2)
            return da/vinf

        def thermal_curve(rho, T, cvinf, a0, vinf, n, theta0, gamma0, gammainf, m):
            """Calculate the thermal pressure component."""
            v=1./rho
            thetavloc=thetav(theta0, v, vinf, gamma0, gammainf, m)
            dthetadvloc=dthetadv(theta0, v, vinf, gamma0, gammainf, m)
            avloc=av(a0, vinf, v, n)
            dadvloc=dadv(a0, vinf, v, n)
    
            q0=dadvloc/3.-dthetadvloc/thetavloc
            q1=5*thetavloc*dadvloc/6.-avloc*dthetadvloc/6.
            q2=(thetavloc**2)*dadvloc/2.-avloc*thetavloc*dthetadvloc/2.
            pref=cvinf/((T+thetavloc)**3)
            thermal=pref*(q0*(T**4)+q1*(T**3)+q2*(T**2))
            return thermal

        def cold_curve(rho0, A, B, C, rho):
            """Calculate the cold pressure component."""
            eta=rho0/rho
            cold_a=A*eta**(-(B+1.))
            cold_b=exp(2.*C*(1.-eta**(3./2.))/3.)
            cold_c=C*eta**(3./2.)+B
            cold_d=A*(B+C)
            cold=cold_a*cold_b*cold_c-cold_d
            return cold

        rholoc = self.rho0/J
        pc = cold_curve(self.rho0, self.A, self.B, self.C, rholoc)
        pth = thermal_curve(rholoc, T, self.cvinf, self.a0, self.vinf, self.n, 
                           self.theta0, self.Gamma0, self.Gammainf, self.m)
        return pc + pth