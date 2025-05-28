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
Deviatoric Stress Models Module
===============================

This module provides a comprehensive framework for calculating deviatoric 
stresses in material models. It implements various hyperelastic, hypoelastic,
and fluid models for the deviatoric part of the stress tensor.

The module is organized with a base abstract class and specialized implementations
for different material behaviors, including:
- Isotropic elastic and hyperelastic models
- Anisotropic elasticity
- Fluid viscosity models
- Hypoelastic formulations for large deformations

Key features:
- Object-oriented design with a consistent interface
- Support for different strain measures and material symmetries
- Compatible with the overall constitutive framework
- Efficient implementation for numerical simulations

The Deviator class serves as the main interface for the rest of the framework,
delegating calculations to the appropriate specialized implementations based on
material type and simulation settings.

Classes:
--------
BaseDeviator : Abstract base class for all deviatoric models
    Defines the common interface and validation functionality
    
Deviator : Main interface for deviatoric stress calculations
    Acts as a facade to the specialized implementations
    Handles model selection and delegation
    
NoneDeviator : Model for pure hydrostatic materials (no deviatoric stress)
    Used for fluids without viscosity
    
Various specialized models:
    IsotropicHPPDeviator, NeoHookDeviator, MooneyRivlinDeviator, etc.
"""
#Abstract class
from .base_deviator import BaseDeviator
#Common isotropic deviator
from .none_deviator import NoneDeviator
from .isotropic_hpp import IsotropicHPPDeviator
from .neo_hook import NeoHookDeviator
from .mooney_rivlin import MooneyRivlinDeviator
#Anisotropic deviator
from .anisotropic import AnisotropicDeviator
#Hypoelastic deviator
from .hypoelastic import HypoelasticDeviator
__all__ = ['BaseDeviator', 'Deviator', 'NoneDeviator', 'NewtonianFluidDeviator',
           'IsotropicHPPDeviator', 'NeoHookDeviator', 'MooneyRivlinDeviator',
           'AnisotropicDeviator', 'HypoelasticDeviator']

class Deviator:
    """Main interface for deviatoric stress calculations.
    
    This class acts as a facade to the underlying specialized deviatoric models,
    delegating calculations to the appropriate implementation based on the
    material type and simulation settings.
    
    Attributes
    ----------
    kin : Kinematic Kinematic handler for tensor operations
    model : str Model name (e.g., "CartesianUD", "PlaneStrain")
    is_hypo : bool Whether to use hypoelastic formulation
    hypo_deviator : HypoelasticDeviator, optional
            Instance of hypoelastic deviator if is_hypo is True
    """
    
    def __init__(self, kinematic, model, quadrature, material):
        """Initialize the deviator interface.
        
        Parameters
        ----------
        kinematic : Kinematic Kinematic handler for tensor operations
        model : str Model name (e.g., "CartesianUD", "PlaneStrain")
        quadrature : QuadratureHandler Handler for quadrature spaces
        is_hypo : bool Whether to use hypoelastic formulation
        """
        self.kin = kinematic
        self.model = model
        
        def is_in_list(material, attribut, keyword):
            """Check if a keyword appears in a material or list of materials.

            Parameters
            ----------
            material : Material or list Material or list of materials to check
            attribut : str Attribute name to check
            keyword : str Keyword to search for
                
            Returns
            -------
            bool True if the keyword is found, False otherwise
            """
            is_mult = isinstance(material, list)
            return (is_mult and any(getattr(mat, attribut) == keyword for mat in material)) or \
                (not is_mult and getattr(material, attribut) == keyword)

        self.is_hypo = is_in_list(material, "dev_type", "Hypoelastic")
        
        
        # self.is_hypo = material.dev_type == "Hypoelastic"
        self.quadrature = quadrature
        if self.is_hypo:
            material.devia.set_hypoelastic(kinematic, model, quadrature)
    
    def set_elastic_dev(self, u, v, J, T, T0, material):
        """Delegate to the appropriate deviator model based on material type.
        
        Parameters
        ----------
        u, v, J, T, T0 : Function See stress_3D method in ConstitutiveLaw.py for details.
        material : Material Material properties with deviator model
            
        Returns
        -------
        Function Deviatoric stress tensor
        """
        return material.devia.calculate_stress(u, v, J, T, T0, self.kin)
    
    def set_hypoelastic_deviator(self, u, v, J, material):
        """Calculate the deviatoric stress for hypoelastic formulation.
        
        This method handles both the initialization of the hypoelastic deviator
        (if needed) and the calculation of the current stress rate. The actual
        time integration and stress update is handled by HypoElasticSolver.
        
        Parameters
        ----------
        u : Function Displacement field
        v : Function Velocity field
        J : Function Jacobian of the transformation
        mu : float Shear modulus
            
        Returns
        -------
        Function 3D deviatoric stress tensor
        """
        return material.devia.calculate_stress_rate(u, v, J, material)