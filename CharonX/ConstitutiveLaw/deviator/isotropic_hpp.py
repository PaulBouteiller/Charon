#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 11:35:18 2025

@author: bouteillerp
"""
"""Isotropic linear elastic deviatoric stress model (small strain)."""

from ufl import sym, dev
from .base_deviator import BaseDeviator

class IsotropicHPPDeviator(BaseDeviator):
    """Isotropic linear elastic deviatoric stress model under small strain hypothesis.
    
    This model implements the standard linear elastic deviatoric stress:
    s = 2 * μ * dev(ε)
    
    where ε is the small strain tensor.
    
    Attributes
    ----------
    mu : float
        Shear modulus (Pa)
    E : float, optional
        Young's modulus (Pa)
    nu : float, optional
        Poisson's ratio
    """
    
    def required_parameters(self):
        """Return the list of required parameters.
        
        Returns
        -------
        list
            List with either "mu" or both "E" and "nu"
        """
        return ["mu"] if "mu" in self.params else ["E", "nu"]
    
    def __init__(self, params):
        """Initialize the isotropic linear elastic deviatoric model.
        
        Parameters
        ----------
        params : dict
            Dictionary containing either:
            mu : float
                Shear modulus (Pa)
            OR:
            E : float
                Young's modulus (Pa)
            nu : float
                Poisson's ratio
        """
        self.params = params  # Store for required_parameters method
        super().__init__(params)
        
        # Initialize parameters
        self.mu = params.get("mu")
        if self.mu is None:
            self.E = params["E"]
            self.nu = params["nu"]
            self.mu = self.E / (2 * (1 + self.nu))
            print(f"Young's modulus: {self.E}")
            print(f"Poisson's ratio: {self.nu}")
            
        print(f"Shear modulus: {self.mu}")
    
    def calculate_stress(self, u, v, J, T, T0, kinematic):
        """Calculate the deviatoric stress tensor for linear elasticity.
        
        s = 2 * μ * dev(sym(∇u))
        
        Parameters
        ----------
        u : Function
            Displacement field
        v : Function
            Velocity field (unused)
        J : Function
            Jacobian of the deformation (unused)
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
        return 2 * self.mu * dev(sym(kinematic.grad_3D(u)))