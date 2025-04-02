#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 11:20:12 2025

@author: bouteillerp
"""
"""Isotropic linear elastic equation of state under small strain hypothesis."""

from math import sqrt
from .base_eos import BaseEOS

class IsotropicHPP_EOS(BaseEOS):
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
    
    def pressure(self, J, T, T0):
        """Calculate pressure using the linear elastic model.
        
        P = -kappa * (J - 1 - 3 * alpha * (T - T0))
        
        Parameters
        ----------
        J : float or Function Jacobian of the deformation
        T : float or Function Current temperature
        T0 : float or Function Initial temperature
            
        Returns
        -------
        float or Function
            Pressure
        """
        return -self.kappa * (J - 1 - 3 * self.alpha * (T - T0))