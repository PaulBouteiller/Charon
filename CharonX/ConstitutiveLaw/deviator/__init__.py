#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 11:32:57 2025

@author: bouteillerp
"""
"""Deviatoric stress models for material behaviors.

This module provides various models for calculating deviatoric stresses in materials,
ranging from simple Newtonian fluids to complex anisotropic solids.
"""

from .base_deviator import BaseDeviator
from .none_deviator import NoneDeviator
from .newtonian_fluid import NewtonianFluidDeviator
from .isotropic_hpp import IsotropicHPPDeviator
from .neo_hook import NeoHookDeviator
from .mooney_rivlin import MooneyRivlinDeviator
from .transverse_isotropic import NeoHookTransverseDeviator, LuTransverseDeviator
from .anisotropic import AnisotropicDeviator

# For backward compatibility with original naming
None_deviatoric = NoneDeviator
NewtonianFluid_deviatoric = NewtonianFluidDeviator
IsotropicHPP_deviatoric = IsotropicHPPDeviator
MooneyRivlin_deviatoric = MooneyRivlinDeviator
NeoHook_Transverse_deviatoric = NeoHookTransverseDeviator
Lu_Transverse_deviatoric = LuTransverseDeviator
Anisotropic_deviatoric = AnisotropicDeviator

__all__ = [
    'BaseDeviator',
    'NoneDeviator',
    'NewtonianFluidDeviator',
    'IsotropicHPPDeviator',
    'NeoHookDeviator',
    'MooneyRivlinDeviator',
    'NeoHookTransverseDeviator',
    'LuTransverseDeviator',
    'AnisotropicDeviator',
    # Original names for backward compatibility
    'None_deviatoric',
    'NewtonianFluid_deviatoric',
    'IsotropicHPP_deviatoric',
    'MooneyRivlin_deviatoric',
    'NeoHook_Transverse_deviatoric',
    'Lu_Transverse_deviatoric',
    'Anisotropic_deviatoric'
]