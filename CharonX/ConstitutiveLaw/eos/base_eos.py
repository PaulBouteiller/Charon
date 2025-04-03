#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 11:16:34 2025

@author: bouteillerp
"""
class BaseEOS:
    """Base class for all equation of state models.
    
    This abstract class defines the common interface that all EOS models
    must implement.
    """
    
    def __init__(self, params):
        """Initialize the equation of state model.
        
        Parameters
        ----------
        params : dict Parameters for the equation of state model
        """
        self._validate_params(params)
        
    def _validate_params(self, params):
        """Validate that all required parameters are provided.
        
        Parameters
        ----------
        params : dict Parameters to validate
            
        Raises
        ------
        ValueError If required parameters are missing
        """
        required_params = self.required_parameters()
        missing_params = [param for param in required_params if param not in params]
        
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")
    
    def required_parameters(self):
        """Return the list of required parameters for this EOS model.
        
        Returns
        -------
        list List of parameter names
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def celerity(self, rho_0):
        """Calculate the wave propagation speed.
        
        Parameters
        ----------
        rho_0 : float Initial density
            
        Returns
        -------
        float Wave speed
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def pressure(self, J, T, T0, material):
        """Calculate the pressure based on the equation of state.
        
        Parameters
        ----------
        J, T, T0, material : See stress_3D method in ConstitutiveLaw.py for details.
            
        Returns
        -------
        Expression Pressure
        """
        raise NotImplementedError("Subclasses must implement this method")