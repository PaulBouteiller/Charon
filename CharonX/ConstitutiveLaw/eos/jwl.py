#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 11:23:08 2025

@author: bouteillerp
"""
"""Jones-Wilkins-Lee (JWL) equation of state for detonation products."""

from math import exp, sqrt
from .base_eos import BaseEOS

class JWL_EOS(BaseEOS):
    """Jones-Wilkins-Lee (JWL) equation of state for detonation products.
    
    This EOS is commonly used for modeling the expansion of explosive detonation products.
    
    Attributes
    ----------
    A, B : float
        Energy coefficients
    R1, R2 : float
        Rate coefficients
    w : float
        Fraction of the energy coefficient
    """
    
    def required_parameters(self):
        """Return the list of required parameters.
        
        Returns
        -------
        list
            List of parameter names
        """
        return ["A", "R1", "B", "R2", "w"]
    
    def __init__(self, params):
        """Initialize the JWL EOS.
        
        Parameters
        ----------
        params : dict
            Dictionary containing:
            A : float
                First energy coefficient
            R1 : float
                First rate coefficient
            B : float
                Second energy coefficient
            R2 : float
                Second rate coefficient
            w : float
                Fraction of the energy coefficient
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
        rho_0 : float
            Initial density
            
        Returns
        -------
        float
            Wave speed
        """
        return sqrt((self.A * self.R1 * exp(-self.R1) + self.B * self.R2 * exp(-self.R2)) / rho_0)
    
    def pressure(self, J, T, T0, material):
        """Calculate pressure using the JWL EOS.
        
        P = A * exp(-R1 * J) + B * exp(-R2 * J) + w * rho_0 / J * C_mass * T
        
        Parameters
        ----------
        J : float or Function
            Jacobian of the deformation
        T : float or Function
            Current temperature
        material : Material
            Material properties (needed for rho_0 and C_mass)
            
        Returns
        -------
        float or Function
            Pressure
        """
        return self.A * exp(-self.R1 * J) + self.B * exp(-self.R2 * J) + self.w * material.rho_0 / J * material.C_mass * T