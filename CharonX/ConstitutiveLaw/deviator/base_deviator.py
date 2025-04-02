#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 11:33:41 2025

@author: bouteillerp
"""
"""Base class for all deviatoric stress models."""

class BaseDeviator:
    """Base class for all deviatoric stress models.
    
    This abstract class defines the common interface that all deviatoric
    stress models must implement.
    """
    
    def __init__(self, params):
        """Initialize the deviatoric stress model.
        
        Parameters
        ----------
        params : dict
            Parameters for the deviatoric model
        """
        self._validate_params(params)
        
    def _validate_params(self, params):
        """Validate that all required parameters are provided.
        
        Parameters
        ----------
        params : dict
            Parameters to validate
            
        Raises
        ------
        ValueError
            If required parameters are missing
        """
        required_params = self.required_parameters()
        missing_params = [param for param in required_params if param not in params]
        
        if missing_params:
            raise ValueError(f"Missing required parameters for {self.__class__.__name__}: {missing_params}")
    
    def required_parameters(self):
        """Return the list of required parameters for this deviatoric model.
        
        Returns
        -------
        list
            List of parameter names
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def calculate_stress(self, u, v, J, T, T0):
        """Calculate the deviatoric stress tensor.
        
        Parameters
        ----------
        u : Function
            Displacement field
        v : Function
            Velocity field
        J : Function
            Jacobian of the deformation
        T : Function
            Current temperature
        T0 : Function
            Initial temperature
            
        Returns
        -------
        Function
            Deviatoric stress tensor
        """
        raise NotImplementedError("Subclasses must implement this method")