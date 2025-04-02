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
from .isotropic_hpp import IsotropicHPP_EOS
from .u_model import U_EOS
from .vinet import Vinet_EOS
from .jwl import JWL_EOS
from .macaw import MACAW_EOS
from .mie_gruneisen import MG_EOS, xMG_EOS, PMG_EOS
from .gas import GP_EOS
from .newtonian_fluid import NewtonianFluid_EOS
from .tabulated import Tabulated_EOS, has_tabulated_eos

# For backward compatibility
__all__ = [
    'BaseEOS',
    'IsotropicHPP_EOS',
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