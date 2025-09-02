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
Base Damage Model Framework
==========================

This module defines the abstract base class for all damage models.
It establishes a common interface and validation functionality that all damage
implementations must follow.

Classes:
--------
BaseDamage : Abstract base class for damage models
    Defines the required interface for all damage models
    Provides common functionality and parameters
    Establishes the core method signatures for damage calculations
"""

from abc import ABC, abstractmethod
from ...utils.parameters.default import default_damage_parameters
from petsc4py.PETSc import ScalarType


class BaseDamage(ABC):
    """Base class for all damage models.
    
    This abstract class provides common functionality and parameters used by all damage models,
    including residual stiffness handling and default damage values.
    
    Attributes
    ----------
    mesh : Mesh Computational mesh
    quad : QuadratureHandler Handler for quadrature integration
    dam_parameters : dict Dictionary of default damage parameters
    residual_stiffness : float Residual stiffness factor (prevents complete loss of stiffness)
    default_damage : float Default initial damage value
    """
    
    def __init__(self, mesh, quadrature, dictionnaire, u=None, J=None, pressure=None, material=None, kinematic=None):
        """Initialize the base damage model with default parameters.
        
        Parameters
        ----------
        mesh : Mesh Computational mesh
        quadrature : QuadratureHandler Handler for quadrature integration
        dictionnaire : dict Additional parameters for damage model
        """
        self.mesh = mesh
        self.quad = quadrature
        self.dam_parameters = default_damage_parameters()
        self.residual_stiffness = self.dam_parameters["residual_stiffness"]
        self.default_damage = ScalarType(self.dam_parameters["default_damage"])
        self.set_damage(mesh, dictionnaire)
        self._initialize_driving_force(u, J, pressure, material, kinematic)
        
    @abstractmethod
    def set_damage(self, mesh, dictionnaire):
        """Initialize the damage model parameters.
        
        Parameters
        ----------
        mesh : Mesh Computational mesh
        dictionnaire : dict Additional parameters for damage model
        """
        pass
    
    def set_unbreakable_zone(self, condition):
        """Define zones where damage is prevented (if applicable).
        
        Parameters
        ----------
        condition : Expression Boolean condition defining unbreakable zones
        """
        pass

    @abstractmethod
    def _initialize_driving_force(self, u, J, pressure, material, kinematic):
        """Initialize damage driving force at creation."""
        pass