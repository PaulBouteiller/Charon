#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 11:32:57 2025

@author: bouteillerp
"""
"""Deviatoric stress models for material behaviors."""

from .base_deviator import BaseDeviator
from .none_deviator import NoneDeviator
from .newtonian_fluid import NewtonianFluidDeviator
from .isotropic_hpp import IsotropicHPPDeviator
from .mooney_rivlin import MooneyRivlinDeviator
from .transverse_isotropic import NeoHookTransverseDeviator, LuTransverseDeviator
from .anisotropic import AnisotropicDeviator


from ufl import (tr, sym, dev, Identity, dot, inner, skew)
from dolfinx.fem import Function, Expression

# Classe pont pour la compatibilité
# Deviator = DeviatorCalculator

# Pour la compatibilité avec l'ancien code
None_deviatoric = NoneDeviator
NewtonianFluid_deviatoric = NewtonianFluidDeviator
IsotropicHPP_deviatoric = IsotropicHPPDeviator
MooneyRivlin_deviatoric = MooneyRivlinDeviator
NeoHook_Transverse_deviatoric = NeoHookTransverseDeviator
Lu_Transverse_deviatoric = LuTransverseDeviator
Anisotropic_deviatoric = AnisotropicDeviator

__all__ = [
    'BaseDeviator',
    'Deviator',  # Important!
    'NoneDeviator',
    'NewtonianFluidDeviator',
    'IsotropicHPPDeviator',
    'MooneyRivlinDeviator',
    'NeoHookTransverseDeviator',
    'LuTransverseDeviator',
    'AnisotropicDeviator',
    # Ancien nom pour la compatibilité
    'None_deviatoric',
    'NewtonianFluid_deviatoric',
    'IsotropicHPP_deviatoric',
    'MooneyRivlin_deviatoric',
    'NeoHook_Transverse_deviatoric',
    'Lu_Transverse_deviatoric',
    'Anisotropic_deviatoric'
]

class Deviator:
    """Bridge class for compatibility with existing code.
    
    This class dispatches calculations to the appropriate specialized deviator model.
    """
    
    def __init__(self, kinematic, model, quadrature, is_hypo):
        self.kin = kinematic
        self.model = model
        self.is_hypo = is_hypo
        
        # Pour la compatibilité avec le code existant
        if is_hypo:
            self.set_function_space(model, quadrature)
    
    def set_function_space(self, model, quadrature):
        """Set up function spaces for hypoelastic formulation."""
        if model == "CartesianUD":
            self.V_s = quadrature.quadrature_space(["Scalar"])
        elif model in ["CylindricalUD", "SphericalUD"]:
            self.V_s = quadrature.quadrature_space(["Vector", 2])
        elif model == "PlaneStrain":
            self.V_s = quadrature.quadrature_space(["Vector", 3])
        elif model == "Axisymetric":
            self.V_s = quadrature.quadrature_space(["Vector", 4])
        elif model == "Tridimensionnal":
            self.V_s = quadrature.quadrature_space(["Tensor", 3, 3])
    
    def set_elastic_dev(self, u, v, J, T, T0, material):
        """Dispatch to the appropriate deviator model based on material type."""
        # Le déviateur sera calculé par la classe spécifique du matériau
        return material.devia.calculate_stress(u, v, J, T, T0, self.kin)
    
    def set_deviator(self, u, v, J, mu):
        self.s = Function(self.V_s, name = "Deviator")
        s_3D = self.kin.reduit_to_3D(self.s, sym = True)
        L = self.kin.reduit_to_3D(self.kin.Eulerian_gradient(v, u))
        D = sym(L)
        # dev_D = dev(D)
        
        B = self.kin.B_3D(u)

        s_Jaumann_3D = mu/J**(5./3) * (dot(B, D) + dot(D, B) 
                                      - 2./3 * inner(B,D) * Identity(3)
                                      -5./3 * tr(D) * dev(B))
        # s_Jaumann_3D = 2 * mu * dev_D
        # s_Jaumann_3D = mu * (dot(B, D) + dot(D, B) - 2./3 * inner(B,D) * Identity(3))
        s_Jaumann = self.kin.tridim_to_reduit(s_Jaumann_3D, sym = True)
        if self.model in ["CartesianUD", "CylindricalUD", "SphericalUD"]:
            self.dot_s = Expression(s_Jaumann, self.V_s.element.interpolation_points())
        else:
            Omega = skew(L)
            Jaumann_corr = self.kin.tridim_to_reduit(dot(Omega, s_3D) - dot(s_3D, Omega), sym = True)
            self.dot_s = Expression(s_Jaumann + Jaumann_corr, self.V_s.element.interpolation_points())
        return s_3D