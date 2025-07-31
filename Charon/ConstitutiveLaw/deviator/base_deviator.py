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

The framework provides:
- Parameter validation infrastructure for material constants
- Required method definitions via abstract methods
- Consistent interface for stress calculations across different material models
- Support for various strain measures (small strain, finite strain, etc.)

Deviatoric stress models handle the shape-changing part of deformation,
complementing volumetric models (EOS) to provide complete stress tensors.

Classes:
--------
BaseDeviator : Abstract base class for deviatoric stress models
    Defines the required interface for all deviatoric models
    Provides validation functionality for parameters
    Establishes the core method signatures for stress calculations
"""

from abc import ABC, abstractmethod


class BaseDeviator(ABC):
    """Abstract base class for all deviatoric stress models.
    
    This abstract class defines the common interface that all deviatoric
    stress models must implement, ensuring consistency across different
    formulations ranging from linear elasticity to complex hyperelastic models.
    
    All deviatoric models must provide:
    - A list of required material parameters
    - Stress calculation capability from kinematic variables
    - Optional methods for energy calculations (for phase field coupling)
    
    The deviatoric stress represents the shape-changing part of the total
    stress tensor and is typically computed as:
    s = σ - (1/3)tr(σ)I
    
    where σ is the total stress tensor and I is the identity tensor.
    
    Parameters are automatically validated during initialization to ensure
    all required material constants are provided.
    """
    
    def __init__(self, params):
        """Initialize the deviatoric stress model with parameter validation.
        
        Parameters
        ----------
        params : dict
            Dictionary containing all material parameters required by the specific
            deviatoric model. The required parameters are defined by the 
            `required_parameters()` method of each concrete implementation.
            
            Common parameter types include:
            - Elastic moduli (E, nu, mu, lambda)
            - Hyperelastic constants (C10, C01, etc.)
            - Anisotropic stiffness coefficients
            - Viscosity parameters for fluid models
            
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
            parameters are missing and which are required.
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
        """Return the list of required parameters for this deviatoric model.
        
        This method must be implemented by all concrete deviatoric classes to
        specify which material parameters are needed for proper initialization
        and stress calculation.
        
        Returns
        -------
        list of str
            List of parameter names that must be present in the initialization
            dictionary. Parameter names should be descriptive and follow
            standard mechanical engineering conventions.
            
        Examples
        --------
        For isotropic linear elasticity:
        >>> return ["mu"]  # or ["E", "nu"]
        
        For Mooney-Rivlin model:
        >>> return ["mu", "mu_quad"]
        
        Notes
        -----
        Some models may accept alternative parameter sets (e.g., E/nu vs mu).
        In such cases, the validation logic should be implemented to handle
        multiple valid combinations.
        """
        pass
    
    @abstractmethod
    def calculate_stress(self, u, v, J, T, T0, kinematic):
        """Calculate the deviatoric stress tensor from kinematic variables.
        
        This is the core method of any deviatoric model, computing the
        deviatoric part of the Cauchy stress tensor based on the current
        deformation state and material properties.
        
        Parameters
        ----------
        u : Function
            Displacement field vector. Used to compute deformation measures
            such as strain tensors or deformation gradients.
        v : Function
            Velocity field vector. Required for rate-dependent models
            (viscous effects, hypoelastic formulations).
        J : Function or Expression
            Jacobian of the deformation gradient (det(F) = ρ₀/ρ).
            Used in finite strain formulations for proper stress measures.
        T : Function or Expression
            Current absolute temperature field (K). Affects material properties
            in temperature-dependent models.
        T0 : Function or Expression
            Initial/reference absolute temperature field (K). Used to compute
            thermal effects and temperature changes.
        kinematic : Kinematic
            Kinematic handler object providing methods for computing:
            - Strain tensors (small and finite strain)
            - Deformation gradients and related measures
            - Tensor operations and transformations
            
        Returns
        -------
        Expression or Function
            3×3 deviatoric stress tensor in the current configuration.
            The return type is typically a UFL Expression that can be
            integrated in variational forms or converted to Functions.
            
            Units: [Pa] (same as pressure)
            
        Notes
        -----
        Common deviatoric stress formulations:
        
        Small strain linear elasticity:
        s = 2μ dev(ε) where ε = sym(∇u)
        
        Neo-Hookean hyperelasticity:  
        s = (μ/J^(5/3)) dev(B) where B is left Cauchy-Green tensor
        
        Hypoelastic formulation:
        ∇s = C : D - Ω·s + s·Ω where D is strain rate, Ω is spin
        
        The stress should be computed in the current (deformed) configuration
        and represent the deviatoric part only. The pressure contribution
        is handled separately by the EOS.
        
        For models with internal variables (plasticity, damage), these should
        be updated consistently within this method or via separate update procedures.
        """
        pass
    
    def isochoric_helmholtz_energy(self, u, kinematic):
        """Calculate the isochoric Helmholtz free energy (optional).
        
        This method provides the strain energy associated with isochoric
        (volume-preserving) deformations. It is used primarily for:
        - Phase field damage models
        - Energy-based formulations
        - Variational approaches
        
        Parameters
        ----------
        u : Function
            Displacement field vector.
        kinematic : Kinematic
            Kinematic handler object for computing deformation measures.
            
        Returns
        -------
        Expression or float
            Isochoric strain energy density per unit reference volume.
            Units: [J/m³] or [Pa]
            
        Notes
        -----
        This method is optional and only needs to be implemented by models
        that support energy-based approaches. The default implementation
        raises NotImplementedError.
        
        For models that do implement this:
        - Linear elasticity: ψ = μ tr(dev(ε)²)
        - Neo-Hookean: ψ = μ(Ī₁ - 3) where Ī₁ is first deviatoric invariant
        
        The energy should be purely isochoric (volume-preserving part).
        Volumetric energy contributions are handled by the EOS.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement isochoric_helmholtz_energy. "
            "This method is only required for energy-based formulations such as "
            "phase field damage models."
        )