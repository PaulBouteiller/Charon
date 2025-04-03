#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 11:23:34 2025

@author: bouteillerp
"""
"""MACAW equation of state for materials under extreme conditions."""

from math import exp, sqrt
from .base_eos import BaseEOS

class MACAW_EOS(BaseEOS):
    """MACAW equation of state for materials under extreme conditions.
    
    This is a sophisticated model for materials under high pressures and temperatures.
    
    Attributes
    ----------
    A, B, C : float
        Cold curve parameters
    cvinf, a0, vinf : float
        Thermal parameters
    n, theta0, Gamma0, Gammainf, m : float
        Thermal and anharmonic parameters
    rho0 : float
        Reference density
    """
    
    def required_parameters(self):
        """Return the list of required parameters.
        
        Returns
        -------
        list
            List of parameter names
        """
        return ["A", "B", "C", "eta", "vinf", "rho0", "theta0", "a0", "m", "n", 
                "Gammainf", "Gamma0", "cvinf"]
    
    def __init__(self, params):
        """Initialize the MACAW EOS.
        
        Parameters
        ----------
        params : dict
            Dictionary with MACAW parameters
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
        rho_0 : float
            Initial density
            
        Returns
        -------
        float
            Wave speed
        """
        kappa = self.A * (self.B - 1./2 * self.C + (self.B + self.C)**2)
        print(f"Cold compressibility modulus: {kappa}")
        return sqrt(kappa / rho_0)
    
    def pressure(self, J, T, T0, material):
        """Calculate pressure using the MACAW EOS.
        
        This method implements the complex MACAW EOS that combines
        cold pressure and thermal pressure components.
        
        Parameters
        ----------
        J : float or Function Jacobian of the deformation
        T : float or Function Current temperature
        T0 : float or Function Initial temperature
        material : Material Material properties
            
        Returns
        -------
        float or Function Pressure
        """
        def thetav(theta0, v, vinf, gamma0, gammainf, m):
            """Helper function for MACAW thermal calculations."""
            eta = v/vinf
            beta = (gammainf - gamma0) / m
            theta = theta0*(eta)**(-gamma0)*((eta)**(-m) + 1)**(beta)
            return theta

        def dthetadv(theta0, v, vinf, gamma0, gammainf, m):
            """Derivative of thetav with respect to v."""
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
            """Derivative of av with respect to v."""
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