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

Equation of State (EOS) Module for Material Behavior
===================================================

This module provides a comprehensive framework for equations of state (EOS) in material
modeling. Equations of state relate volumetric deformation, temperature, and pressure.

The module implements various EOS models suitable for different material classes:
- Linear elastic models for small deformations
- Hyperelastic models for rubber-like materials
- Complex models for fluids and gases
- Specialized models for explosives and high-pressure phenomena
- Tabulated EOS for experimental data-driven approaches

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
    """Initialize the EOS bridge class."""
    
    def __init__(self):
        pass
    
    def set_eos(self, J, T, T0, mat, quadrature):
        """Calculate pressure using the appropriate EOS model.
        
        Parameters
        ----------
        J : Function
            Jacobian of the transformation (volumetric deformation)
        T : Function
            Current temperature
        T0 : Function
            Initial temperature
        mat : Material
            Material properties
        quadrature : QuadratureHandler
            Handler for quadrature integration
            
        Returns
        -------
        Expression
            Pressure
        """
        # Delegate to the appropriate EOS model in the material
        pressure = mat.eos.pressure(J, T, T0, mat, quadrature)
        return pressure