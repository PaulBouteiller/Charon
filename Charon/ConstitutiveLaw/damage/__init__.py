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
Damage Mechanics Module for Material Failure Modeling
=====================================================

This module implements various damage mechanics models for simulating material failure 
and degradation in mechanical simulations. It provides frameworks for both continuous 
and discrete approaches to damage, including phase field models for brittle fracture 
and porosity-based damage models.

The implemented models capture different aspects of material damage:
- Phase field damage for smooth crack propagation in brittle materials
- Johnson-based porosity models for ductile damage
- Dynamic and inertial extensions for rate-dependent damage phenomena

Each model provides the necessary functionality to:
- Initialize damage fields and parameters
- Calculate damage evolution terms
- Provide degradation functions for material stiffness
- Handle regularization and energy dissipation

Classes:
--------
BaseDamage : Base class for all damage models
    Provides common functionality and parameters
    
JohnsonDamage : Base class for porosity-based damage models
    Implements porosity tracking and evolution
    
PhaseFieldDamage : Implementation of phase field damage model
    Various formulations (AT1, AT2, Wu) for brittle fracture
    Energy-based approach with regularization
    
StaticJohnson : Static porosity-based damage
    Basic Johnson model without inertial terms in pore expansion
    
DynamicJohnson : Rate-dependent porosity evolution
    Includes viscous effects in damage evolution
    
InertialJohnson : Inertial effects in damage
    Accounts for inertial terms in pore expansion

References:
-----------
- Phase field models: Ambrosio-Tortorelli, Wu formulations
- Johnson models: Based on porosity evolution theories for ductile damage
"""

# Import base classes
from .base_damage import BaseDamage

# Import Johnson family damage models
from .johnson_damage import (
    JohnsonDamage,
    StaticJohnson, 
    DynamicJohnson, 
    InertialJohnson
)

# Import phase field damage models
from .phase_field_damage import PhaseFieldDamage

__all__ = [
    'BaseDamage',
    'JohnsonDamage',
    'PhaseFieldDamage',
    'StaticJohnson',
    'DynamicJohnson', 
    'InertialJohnson'
]