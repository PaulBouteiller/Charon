#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 11:24:51 2025

@author: bouteillerp
"""
"""Ideal gas and related equations of state."""

from math import sqrt
from .base_eos import BaseEOS

class GP_EOS(BaseEOS):
    """Ideal gas equation of state.
    
    This implements the classic ideal gas law: P = (γ-1) * ρ * e
    where e is the specific internal energy.
    
    Attributes
    ----------
    gamma : float
        Polytropic coefficient
    e_max : float
        Estimated maximum specific internal energy
    """
    
    def required_parameters(self):
        """Return the list of required parameters.
        
        Returns
        -------
        list
            List of parameter names
        """
        return ["gamma", "e_max"]
    
    def __init__(self, params):
        """Initialize the ideal gas EOS.
        
        Parameters
        ----------
        params : dict
            Dictionary containing:
            gamma : float
                Polytropic coefficient
            e_max : float
                Estimated maximum specific internal energy
        """
        super().__init__(params)
        
        # Store parameters
        self.gamma = params["gamma"]
        self.e_max = params["e_max"]
        
        # Log parameters
        print(f"Polytropic coefficient: {self.gamma}")
        print(f"Estimated maximum temperature: {self.e_max}")
    
    def celerity(self, rho_0):
        """Calculate estimated sound speed in gas.
        
        Parameters
        ----------
        rho_0 : float
            Initial density
            
        Returns
        -------
        float
            Estimated sound speed
        """
        return sqrt(self.gamma * (self.gamma - 1) * self.e_max)
    
    def pressure(self, J, T, material):
        """Calculate pressure using the ideal gas law.
        
        P = (gamma - 1) * rho_0 / J * C_mass * T
        
        where C_mass*T is equivalent to the specific internal energy.
        
        Parameters
        ----------
        J : float or Function
            Jacobian of the deformation
        T : float or Function
            Current temperature
        material : Material
            Material properties
            
        Returns
        -------
        float or Function
            Pressure
        """
        return (self.gamma - 1) * material.rho_0 / J * material.C_mass * T