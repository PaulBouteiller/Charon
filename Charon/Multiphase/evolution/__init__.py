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
Phase Evolution Module for Material Behavior
===========================================

This module provides a comprehensive framework for phase evolution laws in material
modeling. Phase evolution laws describe how material phases transform over time
based on various driving forces such as temperature, pressure, and chemical kinetics.

The module implements various evolution models suitable for different material classes:
- Arrhenius kinetics for temperature-driven transformations
- KJMA kinetics for nucleation and growth processes
- WGT reactive burn model for explosives
- Desbiens multi-regime explosive model
- Smooth instantaneous transitions for equilibrium-based transformations

The framework follows the same modular design as the EOS module, with a base
abstract class defining the common interface and concrete implementations
for specific evolution laws.

Key Features:
- Modular design with abstract base class
- Parameter validation and error checking
- Auxiliary field management for complex models
- Consistent interface across all evolution laws
- Integration with constitutive framework

Classes:
--------
BaseEvolutionLaw : Abstract base class for evolution laws
ArrheniusEvolution : Temperature-dependent Arrhenius kinetics
KJMAEvolution : Nucleation and growth kinetics
WGTEvolution : WGT reactive burn model
DesbienasEvolution : Multi-regime explosive model
SmoothInstantaneousEvolution : Instantaneous equilibrium transitions
"""

from .base_evolution import BaseEvolutionLaw
from .arrhenius_evolution import ArrheniusEvolution
from .forestfire_evolution import ForestFireEvolution
from .kjma_evolution import KJMAEvolution
from .wgt_evolution import WGTEvolution
from .desbiens_evolution import DesbiensEvolution
from .smooth_instantaneous_evolution import SmoothInstantaneousEvolution

__all__ = [
    'BaseEvolutionLaw',
    'EvolutionLaw',
    'ArrheniusEvolution',
    'ForestFireEvolution',
    'KJMAEvolution', 
    'WGTEvolution',
    'DesbiensEvolution',
    'SmoothInstantaneousEvolution'
]

class EvolutionLaw:
    """Evolution law bridge class for unified interface."""
    
    def __init__(self):
        """Initialize the evolution bridge class."""
        pass
    
    def set_evolution_law(self, concentrations, T, pressure, material, evolution_law):
        """Calculate concentration rates using the appropriate evolution law.
        
        Parameters
        ----------
        concentrations : list of dolfinx.fem.Function Current concentration fields
        T              : dolfinx.fem.Function         Current temperature field
        pressure       : ufl.Expression               Current pressure expression
        material       : Material                     Material properties object
        evolution_law  : BaseEvolutionLaw             Specific evolution law implementation
            
        Returns
        -------
        list of ufl.Expression Concentration rate expressions dc/dt
        """
        # Delegate to the appropriate evolution law
        rates = evolution_law.compute_concentration_rates(
            concentrations, T, pressure, material
        )
        return rates