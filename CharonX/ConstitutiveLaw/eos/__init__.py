#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 11:31:54 2025

@author: bouteillerp
"""
"""Equation of State models for material behaviors.

This module provides various equation of state models that relate
volume changes, temperature, and pressure within materials.
"""

from .base_eos import BaseEOS
from .isotropic_hpp import IsotropicHPPEOS
from .u_model import UEOS
from .vinet import VinetEOS
from .jwl import JWLEOS
from .macaw import MACAWEOS
from .mie_gruneisen import MGEOS, xMGEOS, PMGEOS
from .gas import GPEOS
from .newtonian_fluid import NewtonianFluidEOS
from .tabulated import TabulatedEOS, has_tabulated_eos

# For backward compatibility
__all__ = [
    'BaseEOS',
    'EOS',  # Ajoutez cette ligne
    'IsotropicHPPEOS',
    'U_EOS',
    'Vinet_EOS',
    'JWL_EOS',
    'MACAW_EOS',
    'MG_EOS',
    'xMG_EOS',
    'PMG_EOS',
    'GP_EOS',
    'NewtonianFluid_EOS',
    'Tabulated_EOS',
    'has_tabulated_eos'
]

class EOS:
    """Bridge class for compatibility with existing code."""
    
    def __init__(self, kinematic, quadrature):
        pass
        # self.kin = kinematic
        # self.quad = quadrature
    
    def set_eos(self, J, T, T0, mat):
        # Délègue au modèle EOS approprié dans le matériau
        pressure = mat.eos.pressure(J, T, T0, mat)
        return pressure