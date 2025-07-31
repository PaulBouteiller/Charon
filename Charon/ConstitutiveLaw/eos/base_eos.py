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

from abc import ABC, abstractmethod


class BaseEOS(ABC):
    """Abstract base class for all equation of state models.
    
    This abstract class defines the common interface that all EOS models
    must implement, ensuring consistency across different formulations.
    
    All EOS models must provide:
    - A list of required parameters
    - Wave speed calculation capability
    - Pressure calculation from thermodynamic state
    
    Parameters are automatically validated during initialization to ensure
    all required values are provided.
    """
    
    def __init__(self, params):
        """Initialize the equation of state model with parameter validation.
        
        Parameters
        ----------
        params : dict
            Dictionary containing all parameters required by the specific EOS model.
            The required parameters are defined by the `required_parameters()` method
            of each concrete implementation.
            
        Raises
        ------
        ValueError
            If any required parameters are missing from the params dictionary.
        """
        self._validate_params(params)
        
    def _validate_params(self, params):
        """Validate that all required parameters are provided.
        
        This method checks that the provided parameters dictionary contains
        all parameters specified by the concrete implementation's
        `required_parameters()` method.
        
        Parameters
        ----------
        params : dict
            Parameters to validate against required parameters list.
            
        Raises
        ------
        ValueError
            If required parameters are missing, with details of which
            parameters are missing.
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
        """Return the list of required parameters for this EOS model.
        
        This method must be implemented by all concrete EOS classes to specify
        which parameters are needed for proper initialization and operation.
        
        Returns
        -------
        list of str
            List of parameter names that must be present in the initialization
            dictionary. Parameter names should be descriptive and consistent
            across similar models.
            
        Examples
        --------
        For a linear elastic EOS:
        >>> return ["E", "nu", "alpha"]
        
        For a JWL explosive EOS:
        >>> return ["A", "R1", "B", "R2", "w"]
        """
        pass
    
    @abstractmethod
    def celerity(self, rho_0):
        """Calculate the characteristic wave propagation speed in the material.
        
        This method computes the speed of elastic waves (typically longitudinal
        or bulk waves) in the material, which is essential for:
        - Determining stable time steps in explicit dynamics
        - Wave propagation studies
        
        Parameters
        ----------
        rho_0 : float
            Initial (reference) mass density of the material. Must be positive.
            
        Returns
        -------
        float
            Characteristic wave speed in the material.
            Should be positive for physical materials.
            
        Notes
        -----
        Common formulations include:
        - Linear elastic: c = sqrt(K/ρ₀) or c = sqrt(E/ρ₀)
        - Nonlinear: Various expressions depending on the EOS model
        
        The returned value is often used as an estimate and may not represent
        the exact acoustic velocity under all conditions.
        """
        pass
    
    @abstractmethod
    def pressure(self, J, T, T0, material, quadrature):
        """Calculate pressure based on the current thermodynamic state.
        
        This is the core method of any EOS, relating the current state
        (volume change, temperature) to the hydrostatic pressure.
        
        Parameters
        ----------
        J : Function or Expression
            Jacobian of the deformation gradient (volumetric deformation ratio).
            J = det(F) = ρ₀/ρ where F is deformation gradient.
            J > 0 for physical deformations; J < 1 indicates compression.
        T : Function or Expression
            Current absolute temperature field (K).
        T0 : Function or Expression  
            Initial/reference absolute temperature field (K).
        material : Material
            Material object containing additional properties that may be
            needed for pressure calculation (density, heat capacity, etc.).
        quadrature : QuadratureHandler
            Handler for quadrature spaces, used when the EOS requires
            function space operations or interpolation.
            
        Returns
        -------
        Function, Expression,
            Pressure field.
        Notes
        -----
        Sign convention: Positive pressure indicates compression (continuum mechanics).
        
        Common EOS forms:
        - Linear: P = -K(J - 1 - α(T - T₀))
        - Hyperelastic: P = -∂Ψ/∂J where Ψ is strain energy
        - Gas: P = (γ-1)ρe where e is specific internal energy
        
        The method should handle both isothermal and non-isothermal cases
        appropriately based on the temperature inputs.
        """
        pass