#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 13:23:51 2025

@author: bouteillerp
"""
"""Vinet equation of state for solids."""


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
    
    def pressure(self, J, T, T0, material):
        """Calculate pressure using the Vinet EOS.
        
        P = 3 * K0 * J^(-2/3) * (1 - J^(1/3)) * exp(3/2 * (K1 - 1) * (1 - J^(1/3)))
        
        where K0 and K1 depend on temperature.
        
        Parameters
        ----------
        J, T, T0, material : See stress_3D method in ConstitutiveLaw.py for details.
            
        Returns
        -------
        Expression Pressure
        """
        K0 = self.iso_T_K0 + self.T_dep_K0 * (T - T0)
        K1 = self.iso_T_K1 + self.T_dep_K1 * (T - T0)
        return 3 * K0 * J**(-2./3) * (1 - J**(1./3)) * exp(3./2 * (K1 - 1) * (1 - J**(1./3)))