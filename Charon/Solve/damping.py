#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Artificial viscosity module for numerical stabilization in dynamic simulations.

Created on Fri Aug  1 15:12:33 2025
@author: bouteillerp
"""
from ..utils.generic_functions import npart
from ..utils.default_parameters import default_damping_parameters
from ufl import dot


class Damping:
    """
    Artificial viscosity implementation for numerical stabilization.
    
    This class provides pseudo-viscosity stabilization techniques commonly used
    in shock-dominated problems and explicit dynamics simulations. The artificial
    viscosity helps control numerical oscillations and improve stability.
    
    Parameters
    ----------
    dictionnaire : dict
        Configuration dictionary containing damping parameters
    u : Function
        Current displacement field
    u_ : Function
        Previous displacement field (test function)
    v : Function
        Velocity field
    J : Function or Expression
        Jacobian of the transformation
    material : Material or list of Material
        Material properties (single material or list for multiphase)
    kinematic : Kinematic
        Kinematic utilities object
    dx : Measure
        Integration measure
    h : Function or Expression
        Characteristic element size
    multiphase : Multiphase or None
        Multiphase object for multi-material problems
    form : Form
        Weak form to which damping will be added
    name : str
        Problem geometry name (affects damping computation)
        
    Attributes
    ----------
    Klin : float
        Linear viscosity coefficient
    Kquad : float
        Quadratic viscosity coefficient
    correction : bool
        Whether to apply Jacobian correction
    damping_form : Form
        Damping contribution to the weak form
    """
    
    def __init__(self, dictionnaire, u, u_, v, J, material, kinematic, dx, h, multiphase, form, name):
        self.name = name
        self.kinematic = kinematic
        self.material = material
        self.J = J
        self.h = h
        self.multiphase = multiphase
        self.dx = dx
        
        self._setup_damping_parameters(dictionnaire)
        self._compute_damping_form(u, u_, v)
    
    def _setup_damping_parameters(self, dictionnaire):
        """
        Initialize artificial viscosity parameters from configuration.
        
        Parameters
        ----------
        dictionnaire : dict
            Configuration dictionary potentially containing 'damping' key
        """
        damping_config = dictionnaire.get("damping", default_damping_parameters())
        self.Klin = damping_config["linear_coeff"]
        self.Kquad = damping_config["quad_coeff"]
        self.correction = damping_config["correction"]
    
    def _pseudo_pressure_single_material(self, velocity, material, jacobian):
        """
        Calculate pseudo-viscous pressure for a single material.
        
        The pseudo-pressure combines linear and quadratic viscosity terms:
        - Linear term: proportional to velocity divergence
        - Quadratic term: proportional to square of velocity divergence
        
        Parameters
        ----------
        velocity : Function Velocity field
        material : Material Material properties containing density and sound speed
        jacobian : Function Jacobian of the transformation
            
        Returns
        -------
        Expression Pseudo-viscous pressure field
        """
        div_v = self.kinematic.div(velocity)
        
        # Linear viscosity term
        lin_Q = self.Klin * material.rho_0 * material.celerity * self.h * npart(div_v)
        
        # Quadratic viscosity term (geometry-dependent)
        if self.name in ["CartesianUD", "CylindricalUD", "SphericalUD"]:
            quad_Q = self.Kquad * material.rho_0 * self.h**2 * npart(div_v) * div_v
        elif self.name in ["PlaneStrain", "Axisymmetric", "Tridimensional"]:
            quad_Q = self.Kquad * material.rho_0 * self.h**2 * dot(npart(div_v), div_v)
        else:
            # Default to dot product formulation
            quad_Q = self.Kquad * material.rho_0 * self.h**2 * dot(npart(div_v), div_v)
        
        # Apply Jacobian correction if enabled
        if self.correction:
            lin_Q /= jacobian
            quad_Q /= jacobian**2
            
        return quad_Q - lin_Q
    
    def _compute_pseudo_pressure(self, velocity):
        """
        Compute pseudo-pressure for single or multi-material case.
        
        Parameters
        ----------
        velocity : Function Velocity field
            
        Returns
        -------
        Expression Total pseudo-pressure field
        """
        if isinstance(self.material, list):
            # Multi-material case
            pseudo_pressures = [self._pseudo_pressure_single_material(velocity, mat, self.J)
                                for mat in self.material]
            return sum(self.multiphase.c[i] * pseudo_pressures[i]
                       for i in range(len(self.material)))
        else:
            # Single material case
            return self._pseudo_pressure_single_material(velocity, self.material, self.J)
    
    def _compute_damping_form(self, u, u_, v):
        """
        Compute the damping contribution to the weak form.
        
        The damping form is integrated over the domain and contributes to
        the overall system of equations for stabilization purposes.
        
        Parameters
        ----------
        u : Function Current displacement field
        u_ : Function Test function for displacement
        v : Function Velocity field
        """
        # Compute pseudo-pressure
        pseudo_pressure = self._compute_pseudo_pressure(v)
        
        # Compute kinematic quantities
        invFTop = self.kinematic.inv_deformation_gradient_3D(u).T
        invFTop_compact = self.kinematic.tensor_3d_to_compact(invFTop)
        grad_u_ = self.kinematic.grad_vector_compact(u_)
        
        # Contract tensors and integrate
        inner_prod = self.kinematic.contract_double(invFTop_compact, grad_u_)
        self.damping_form = pseudo_pressure * inner_prod * self.dx