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
Base Deviatoric Stress Model Framework
======================================

This module defines the abstract base class for all deviatoric stress models.
It establishes a common interface and validation functionality that all deviatoric
stress implementations must follow.

Classes:
--------
BaseDeviator : Abstract base class for deviatoric stress models
    Defines the required interface for all deviatoric models
    Provides validation functionality for parameters
    Establishes the core method signatures for stress calculations
"""

from abc import ABC, abstractmethod

class BaseDeviator(ABC):
    """Base class for all deviatoric stress models.
    
    This abstract class defines the common interface that all deviatoric
    stress models must implement.
    """
    
    def __init__(self, params):
        """Initialize the deviatoric stress model.
        
        Parameters
        ----------
        params : dict Parameters for the deviatoric model
        """
        self._validate_params(params)
        
    def _validate_params(self, params):
        """Validate that all required parameters are provided."""
        required_params = self.required_parameters()
        missing_params = [param for param in required_params if param not in params]
        
        if missing_params:
            raise ValueError(f"Missing required parameters for {self.__class__.__name__}: {missing_params}")
    
    @abstractmethod
    def required_parameters(self):
        """Return the list of required parameters for this deviatoric model.
        
        Returns
        -------
        list List of parameter names
        """
        pass
    
    @abstractmethod
    def calculate_stress(self, u, v, J, T, T0, kinematic):
        """Calculate the deviatoric stress tensor.
        
        Parameters
        ----------
        u, v, J, T, T0 : Function See stress_3D method in ConstitutiveLaw.py for details.
        kinematic : Kinematic Kinematic handler object
        
        Returns
        -------
        ufl.core.expr.Expr Deviatoric stress tensor
        """
        pass