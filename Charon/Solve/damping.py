#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 15:12:33 2025

@author: bouteillerp
"""
from ..utils.generic_functions import npart
from ..utils.default_parameters import default_damping_parameters
from ufl import dot
class Damping():
    def __init__(self, dictionnaire, u, u_, v, J, material, kinematic, dx, h, multiphase, form, name):
        self.name = name
        self.kin = kinematic
        self.mat = material
        self.u = u
        self.v = v 
        self.J = J
        self.set_damping(dictionnaire)
        self.update_form_with_stabilization(u, u_, v, J, material, kinematic, h, dx, multiphase, form)
    
    def set_damping(self, dictionnaire):
        """Initialize artificial viscosity parameters.
        
        Sets up the parameters for the pseudo-viscosity used for 
        numerical stabilization in shock-dominated problems.
        
        Parameters
        ----------
        damping : dict
            Dictionary containing:
            - "damping" (bool): Whether to enable artificial viscosity
            - "linear_coeff" (float): Linear viscosity coefficient
            - "quad_coeff" (float): Quadratic viscosity coefficient
            - "correction" (bool): Whether to apply Jacobian correction
        """
        damping = dictionnaire.get("damping", default_damping_parameters())
        self.Klin = damping["linear_coeff"]
        self.Kquad = damping["quad_coeff"]
        self.correction = damping["correction"]
        
    def pseudo_pressure(self, velocity, material, jacobian, h, kinematic):
        """Calculate the pseudo-viscous pressure for stabilization.
        
        This pseudo-pressure term is added to improve numerical stability,
        especially in shock-dominated problems.
        
        Parameters
        ----------
        velocity : Function Velocity field.
        material : Material  Material properties.
        jacobian : Function Jacobian of the transformation.
            
        Returns
        -------
        Function Pseudo-viscous pressure field.
        """
        div_v  = kinematic.div(velocity)
        lin_Q = self.Klin * material.rho_0 * material.celerity * h * npart(div_v)
        if self.name in ["CartesianUD", "CylindricalUD", "SphericalUD"]: 
            quad_Q = self.Kquad * material.rho_0 * h**2 * npart(div_v) * div_v 
        elif self.name in ["PlaneStrain", "Axisymmetric", "Tridimensional"]:
            quad_Q = self.Kquad * material.rho_0 * h**2 * dot(npart(div_v), div_v)
        if self.correction :
            lin_Q *= 1/jacobian
            quad_Q *= 1 / jacobian**2
        return quad_Q - lin_Q
    
    def compute_pseudo_pressure(self, v, material, J, h, multiphase, kinematic):
        if isinstance(material, list):
            n_materials = len(material)
            pseudo_pressure_list = []
            for i, material in enumerate(material):
                pseudo_pressure_list.append(self.pseudo_pressure(v, material, J, h, kinematic))
            return sum(multiphase.c[i] * pseudo_pressure_list[i] for i in range(n_materials))
        # Single material case
        return self.pseudo_pressure(v, material, J, h, kinematic)
    
    def update_form_with_stabilization(self, u, u_, v, J, material, kinematic, h, dx, multiphase, form):
        pseudo_pressure = self.compute_pseudo_pressure(v, material, J, h, multiphase, kinematic)
        invFTop = kinematic.inv_deformation_gradient_3D(u).T
        invFTop_compact = kinematic.tensor_3d_to_compact(invFTop)
        grad_u_ = kinematic.grad_vector_compact(u_)
        inner_prod = kinematic.contract_double(invFTop_compact, grad_u_)
        self.damping_form = pseudo_pressure * inner_prod * dx