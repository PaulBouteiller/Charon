#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 11:34:38 2025

@author: bouteillerp
"""
"""Newtonian fluid deviatoric stress model."""

from ufl import sym, dev
from .base_deviator import BaseDeviator

class NewtonianFluidDeviator(BaseDeviator):
    """Deviatoric stress model for Newtonian fluids.
    
    This model implements the standard viscous stress tensor for Newtonian fluids:
    s = 2 * μ * dev(sym(∇v))
    
    Attributes
    ----------
    mu : float
        Dynamic viscosity
    """
    
    def required_parameters(self):
        """Return the list of required parameters.
        
        Returns
        -------
        list
            List of parameter names
        """
        return ["mu"]
    
    def __init__(self, params):
        """Initialize the Newtonian fluid deviatoric model.
        
        Parameters
        ----------
        params : dict
            Dictionary containing:
            mu : float
                Dynamic viscosity
        """
        super().__init__(params)
        
        # Store parameters
        self.mu = params["mu"]
    
    def calculate_stress(self, u, v, J, T, T0, kinematic):
        """Calculate the deviatoric stress tensor for a Newtonian fluid.
        
        s = 2 * μ * dev(sym(∇v))
        
        Parameters
        ----------
        u : Function
            Displacement field (unused for Newtonian fluids)
        v : Function
            Velocity field
        J : Function
            Jacobian of the deformation
        T : Function
            Current temperature (unused)
        T0 : Function
            Initial temperature (unused)
        kinematic : Kinematic
            Kinematic handler object
            
        Returns
        -------
        Function
            Deviatoric stress tensor
        """
        return 2 * self.mu * dev(sym(kinematic.grad_3D(v)))