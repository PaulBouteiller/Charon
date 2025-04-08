# Copyright 2025 CEA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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

__all__ = [
    'BaseEOS',
    'EOS',
    'IsotropicHPPEOS',
    'UEOS',
    'VinetEOS',
    'JWLEOS',
    'MACAWEOS',
    'MGEOS',
    'xMGEOS',
    'PMGEOS',
    'GPEOS',
    'NewtonianFluidEOS',
    'TabulatedEOS',
    'has_tabulated_eos'
]

class EOS:
    """Bridge class for compatibility with existing code."""
    
    def __init__(self):
        pass
    
    def set_eos(self, J, T, T0, mat, quadrature):
        # Délègue au modèle EOS approprié dans le matériau
        pressure = mat.eos.pressure(J, T, T0, mat, quadrature)
        return pressure