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
Base Evolution Law Framework
===========================

This module defines the abstract base class for all phase evolution models.
It establishes a common interface and validation functionality that all evolution
implementations must follow, similar to the EOS framework.

Classes:
--------
BaseEvolutionLaw : Abstract base class for phase evolution models
    Defines the required interface for all evolution models
    Provides validation functionality for parameters
    Establishes core method signatures for concentration rate calculations
"""

from abc import ABC, abstractmethod

class BaseEvolutionLaw(ABC):
    """Abstract base class for all phase evolution models.
    
    This abstract class defines the common interface that all evolution models
    must implement, ensuring consistency across different formulations.
    
    All evolution models must provide:
    - A list of required parameters
    - Concentration rate calculation capability
    - Auxiliary field setup if needed
    """
    
    def __init__(self, params):
        """Initialize the evolution law with parameter validation.
        
        Parameters
        ----------
        params : dict
            Dictionary containing all parameters required by the specific evolution model.
            The required parameters are defined by the `required_parameters()` method
            of each concrete implementation.
            
        Raises
        ------
        ValueError If any required parameters are missing from the params dictionary.
        """
        self._validate_params(params)
        
    def _validate_params(self, params):
        """Validate that all required parameters are provided.
        
        Parameters
        ----------
        params : dict Parameters to validate against required parameters list.
            
        Raises
        ------
        ValueError If required parameters are missing.
        """
        required_params = self.required_parameters()
        missing_params = [param for param in required_params if param not in params]
        
        if missing_params:
            class_name = self.__class__.__name__
            raise ValueError(
                f"Missing required parameters for {class_name}: {missing_params}. "
                f"Required parameters are: {required_params}"
            )
    
    @abstractmethod
    def required_parameters(self):
        """Return the list of required parameters for this evolution model.
        
        Returns
        -------
        list of str List of parameter names that must be present in the initialization dictionary.
        """
        pass
    
    @abstractmethod
    def setup_auxiliary_fields(self, V_c, **kwargs):
        """Set up auxiliary fields needed for the evolution model.
        
        Some evolution models (like KJMA) require auxiliary fields beyond
        the concentration fields themselves.
        
        Parameters
        ----------
        V_c : dolfinx.fem.FunctionSpace Function space for concentration fields
        **kwargs : dict Additional setup parameters specific to each model
        """
        pass
    
    @abstractmethod
    def compute_concentration_rates(self, concentrations, T, pressure, material, **kwargs):
        """Compute the time derivatives of concentrations.
        
        This is the core method that calculates dc/dt for all phases
        based on the current state and evolution law.
        
        Parameters
        ----------
        concentrations : list of dolfinx.fem.Function Current concentration fields for all phases
        T : dolfinx.fem.Function Current temperature field
        pressure : ufl.Expression Current pressure expression
        material : Material Material object containing properties
        **kwargs : dict Additional parameters specific to each evolution law
            
        Returns
        -------
        list of ufl.Expression List of concentration rate expressions dc/dt for each phase
        """
        pass
    
    @abstractmethod
    def update_auxiliary_fields(self, dt, **kwargs):
        """Update auxiliary fields for the next time step.
        
        Parameters
        ----------
        dt       : float Time step size
        **kwargs : dict Additional update parameters
        """
        pass
    
    def get_auxiliary_fields(self):
        """Return dictionary of auxiliary fields if any.
        
        Returns
        -------
        dict Dictionary of auxiliary fields, empty by default
        """
        return {}