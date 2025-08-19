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
Created on Thu Mar 24 09:54:52 2022

@author: bouteillerp
Plasticity Models for Mechanical Simulations
============================================

This module implements various plasticity models for mechanical simulations.
It provides a framework for handling both small strain and finite strain
plasticity, with support for different hardening mechanisms.

Classes:
--------
Plastic : Base class for all plasticity models
HPPPlastic : Small strain (Hypoelastic-Plastic) model
FiniteStrainPlastic : Multiplicative finite strain model
JAXJ2Plasticity : J2 plasticity with algorithmic tangent
JAXGursonPlasticity : Gurson model for porous plasticity
"""

# Import des classes de plasticité modulaires
from .base_plastic import Plastic
from .hpp_plastic import HPPPlastic
from .finite_strain_plastic import FiniteStrainPlastic
from .jax_j2_plastic import JAXJ2Plasticity
from .jax_gurson_plastic import JAXGursonPlasticity
from .jax_gurson_plastic_hpp import GTNSimplePlasticity

# Export des classes pour compatibilité
__all__ = [
    'Plastic',
    'HPPPlastic', 
    'FiniteStrainPlastic',
    'JAXJ2Plasticity',
    'JAXGursonPlasticity',
    'GTNSimplePlasticity'
]