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
Created on Wed Apr  2 11:16:34 2025

@author: bouteillerp

Base Equation of State Framework
===============================

This module defines the abstract base class for all equation of state (EOS) models.
It establishes a common interface and validation functionality that all EOS
implementations must follow.

The framework provides:
- Parameter validation infrastructure
- Required method definitions via abstract methods
- Consistent interface for pressure and wave speed calculations
- Method requirement enforcement for derived classes

By providing this common foundation, the module ensures that all equation of state
models operate consistently within the larger constitutive framework, facilitating
interchangeability and modular design.

Classes:
--------
BaseEOS : Abstract base class for equation of state models
    Defines the required interface for all EOS models
    Provides validation functionality for parameters
    Establishes the core method signatures for pressure and wave speed calculations
"""

class BaseEOS:
    """Base class for all equation of state models.
    
    This abstract class defines the common interface that all EOS models
    must implement.
    """
    
    def __init__(self, params):
        """Initialize the equation of state model.
        
        Parameters
        ----------
        params : dict Parameters for the equation of state model
        """
        self._validate_params(params)
        
    def _validate_params(self, params):
        """Validate that all required parameters are provided.
        
        Parameters
        ----------
        params : dict Parameters to validate
            
        Raises
        ------
        ValueError If required parameters are missing
        """
        required_params = self.required_parameters()
        missing_params = [param for param in required_params if param not in params]
        
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")
    
    def required_parameters(self):
        """Return the list of required parameters for this EOS model.
        
        Returns
        -------
        list List of parameter names
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def celerity(self, rho_0):
        """Calculate the wave propagation speed.
        
        Parameters
        ----------
        rho_0 : float Initial density
            
        Returns
        -------
        float Wave speed
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def pressure(self, J, T, T0, material, quadrature):
        """Calculate the pressure based on the equation of state.
        
        Parameters
        ----------
        J, T, T0, material : See stress_3D method in ConstitutiveLaw.py for details.
        quadrature : QuadratureHandler Handler for quadrature spaces.
            
        Returns
        -------
        Expression Pressure
        """
        raise NotImplementedError("Subclasses must implement this method")