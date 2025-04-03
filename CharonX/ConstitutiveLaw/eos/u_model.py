#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 11:21:36 2025

@author: bouteillerp
"""
"""Hyperelastic isotropic equation of state models (U-family)."""

from math import sqrt
from ufl import ln
from .base_eos import BaseEOS

class U_EOS(BaseEOS):
    """Hyperelastic isotropic equation of state with one coefficient.
    
    This class implements various energy potentials (U1 through U8)
    that relate volume change to pressure for hyperelastic materials.
    
    Attributes
    ----------
    kappa : float Bulk modulus (Pa)
    alpha : float Thermal expansion coefficient (1/K)
    """
    
    def required_parameters(self):
        """Return the list of required parameters.
        
        Returns
        -------
        list List of parameter names
        """
        return ["kappa", "alpha"]
    
    def __init__(self, params):
        """Initialize the hyperelastic isotropic EOS.
        
        Parameters
        ----------
        params : dict Dictionary containing:
                        kappa : float Bulk modulus (Pa)
                        alpha : float Thermal expansion coefficient (1/K)
        """
        super().__init__(params)
        
        # Store parameters
        self.kappa = params["kappa"]
        self.alpha = params["alpha"]
        
        # Log parameters
        print(f"Bulk modulus: {self.kappa}")
        print(f"Thermal expansion coefficient: {self.alpha}")
    
    def celerity(self, rho_0):
        """Calculate wave speed estimation for hyperelastic material.
        
        Parameters
        ----------
        rho_0 : float Initial density
            
        Returns
        -------
        float Wave speed
        """
        return sqrt(self.kappa / rho_0)
    
    def pressure(self, J, T, T0, material, model_type="U5"):
        """Calculate pressure using hyperelastic U-model.
        
        Available models:
        - U1: P = -kappa * (J - 1 - 3 * alpha * (T - T0))
        - U5: P = -kappa * (ln(J) - ln(1 + 3 * alpha * (T - T0)))
        - U8: P = -kappa/2 * (ln(J) - 1/J - ln(1 + 3*alpha*(T-T0)) + 1/(1+alpha*(T-T0)))
        
        Parameters
        ----------
        J : float or Function Jacobian of the deformation
        T : float or Function Current temperature
        T0 : float or Function Initial temperature
        model_type : str, optional U-model variant to use (U1, U5, U8, etc.), by default "U5"
            
        Returns
        -------
        float or Function  Pressure
        """
        if model_type == "U1":
            return -self.kappa * (J - 1 - 3 * self.alpha * (T - T0))
        elif model_type == "U5":
            return -self.kappa * (ln(J) - ln(1 + 3 * self.alpha * (T - T0)))
        elif model_type == "U8":
            return -self.kappa/2 * (ln(J) - 1/J - ln(1 + 3*self.alpha*(T-T0)) + 1/(1+self.alpha*(T-T0)))
        else:
            raise ValueError(f"Unsupported U-model: {model_type}")