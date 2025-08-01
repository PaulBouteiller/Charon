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
Constitutive Law Module for Mechanical Simulations
==================================================
This module provides the core framework for managing constitutive relations in mechanical simulations.
It serves as the main interface for handling material behavior, combining effects from various
physical phenomena including elasticity, plasticity, and damage mechanics.

The module implements a comprehensive approach to stress calculations based on the current state
of the system, using a modular design that separates different aspects of material behavior:
- Volumetric response through equations of state (EOS)
- Deviatoric response through various elastic and hyperelastic models
- Plastic deformation through multiple plasticity models
- Material damage through phase field and other damage models

Key features:
- Support for both small and finite strain formulations
- Handling of multiphase materials
- Thermal effects and coupling
- Integration with different numerical stabilization techniques
- Flexible model selection for different material behaviors

Classes:
--------
ConstitutiveLaw : Main class for managing constitutive relations
    Orchestrates the calculation of stresses by combining various physical phenomena
    Handles both single-phase and multiphase materials
    Provides methods for calculating complete stress tensors and energy density

Methods:
--------
stress_3D : Calculate the complete 3D Cauchy stress tensor
    Combines pressure, artificial viscosity, and deviatoric stress components
    Supports multiphase materials through concentration-weighted averages
    
Helmholtz_energy : Calculate the Helmholtz free energy 
    Combines volumetric and isochoric energy contributions
    Used for phase field damage models

"""
from .eos import EOS
from .deviator import Deviator
from .plasticity.plastic import HPPPlastic, FiniteStrainPlastic, JAXJ2Plasticity, JAXGursonPlasticity
from .damage import PhaseFieldDamage, StaticJohnson, DynamicJohnson, InertialJohnson

from ufl import dot, Identity
from ..utils.generic_functions import npart

class ConstitutiveLaw:
    """Manages the constitutive relations for mechanical simulations.

    This class orchestrates the calculation of stresses based on the current state
    of the system, combining effects from various physical phenomena like elasticity,
    plasticity, and damage.
    Attributes
    ----------
    material : Material Material properties object, can be a single material or a list for multiphase
    mesh : Mesh Computational mesh
    h : float or Function Characteristic mesh size
    plastic_model : str or None Type of plasticity model
    damage_model : str or None Type of damage model
    multiphase : Multiphase or None Object managing multiphase properties
    kinematic : Kinematic Kinematic handler for tensor operations
    quadrature : QuadratureHandler Handler for quadrature integration
    is_damping : bool Whether artificial viscosity is enabled
    Klin : float Linear coefficient for artificial viscosity
    Kquad : float Quadratic coefficient for artificial viscosity
    correction : bool Whether to apply Jacobian correction to artificial viscosity
    eos : EOS Equation of state handler
    deviator : Deviator Deviatoric stress handler
    name : str Model name (e.g., "CartesianUD", "PlaneStrain")
    relative_rho_0 : float or Function Relative initial density field
    """
    
    def __init__(self, u, material, plasticity_dictionnary, damage_dictionnary, multiphase, 
                 name, kinematic, quadrature, damping, relative_rho_0, h):
        """Initialize the constitutive law manager.

        Parameters
        ----------
        u : Function Displacement field
        material : Material or list Material properties (single material or list for multiphase)
        multiphase : Multiphase or None Object managing multiphase properties
        name : str Model name (e.g., "CartesianUD", "PlaneStrain")
        kinematic : Kinematic Kinematic handler for tensor operations
        quadrature : QuadratureHandler  Handler for quadrature integration
        damping : dict Dictionary containing artificial viscosity parameters
        relative_rho_0 : float or Function Relative initial density field
        h : float or Function Characteristic mesh size
        """
        self.material = material
        self.mesh = u.function_space.mesh
        self.h = h
        self.plasticity_dictionnary = plasticity_dictionnary
        self.plastic_model = plasticity_dictionnary.get("model")
        self.damage_dictionnary = damage_dictionnary
        self.damage_model = damage_dictionnary.get("model")
        self.multiphase = multiphase
        self.kinematic = kinematic
        self.quadrature = quadrature
        self.set_damping(damping)
        self.eos = EOS()
        self.deviator = Deviator(kinematic, name, quadrature, material)

        self.name = name
        self.relative_rho_0 = relative_rho_0
        if self.damage_model != None:
            damage_mapper = {"PhaseField" : PhaseFieldDamage, 
                              "StaticJohnson" : StaticJohnson,
                              "DynamicJohnson" : DynamicJohnson,
                              "InertialJohnson" : InertialJohnson
                              }
            damage_class = damage_mapper.get(self.damage_model)
            if damage_class is None:
                raise ValueError(f"Unknown damage model: {self.damage_model}")
            self.damage = damage_class(self.mesh, quadrature, damage_dictionnary, 
                          u=u, J=self.kinematic.J(u), pressure=self.p, 
                          material=material, kinematic=kinematic)
            
            
        if self.plasticity_dictionnary != {}:
            plastic_mapper = {"HPP_Plasticity" : HPPPlastic, 
                              "Finite_Plasticity" : FiniteStrainPlastic,
                              "J2_JAX" : JAXJ2Plasticity,
                              "JAX_Gurson" : JAXGursonPlasticity
                              }
            plastic_class = plastic_mapper.get(self.plastic_model)
            if plastic_class is None:
                raise ValueError(f"Unknown plasticity model: {self.plastic_model}") 
            self.plastic = plastic_class(u, material.devia.mu, name, kinematic, quadrature, plasticity_dictionnary)

    def set_damping(self, damping):
        """Initialize artificial viscosity parameters.
        
        Sets up the parameters for the pseudo-viscosity used for 
        numerical stabilization in shock-dominated problems.
        
        Parameters
        ----------
        damping : dict
            Dictionary containing:
            - "damping" (bool): Whether to enable artificial viscosity
            - "linear_coeff" (float): Linear viscosity coefficient
            - "quad_coeff" (float): Quadratic viscosity coefficient
            - "correction" (bool): Whether to apply Jacobian correction
        """
        self.is_damping = damping["damping"]
        self.Klin = damping["linear_coeff"]
        self.Kquad = damping["quad_coeff"]
        self.correction = damping["correction"]

    def pseudo_pressure(self, velocity, material, jacobian, h):
        """Calculate the pseudo-viscous pressure for stabilization.
        
        This pseudo-pressure term is added to improve numerical stability,
        especially in shock-dominated problems.
        
        Parameters
        ----------
        velocity : Function Velocity field.
        material : Material  Material properties.
        jacobian : Function Jacobian of the transformation.
            
        Returns
        -------
        Function Pseudo-viscous pressure field.
        """
        div_v  = self.kinematic.div(velocity)
        lin_Q = self.Klin * material.rho_0 * material.celerity * h * npart(div_v)
        if self.name in ["CartesianUD", "CylindricalUD", "SphericalUD"]: 
            quad_Q = self.Kquad * material.rho_0 * h**2 * npart(div_v) * div_v 
        elif self.name in ["PlaneStrain", "Axisymmetric", "Tridimensional"]:
            quad_Q = self.Kquad * material.rho_0 * h**2 * dot(npart(div_v), div_v)
        if self.correction :
            lin_Q *= 1/jacobian
            quad_Q *= 1 / jacobian**2
        return quad_Q - lin_Q
    
    def stress_3D(self, u, v, T, T0, J):
        """Calculate the complete 3D Cauchy stress tensor.
        
        This method computes the total stress as a combination of pressure and deviatoric 
        components. For multiphase materials, it calculates a weighted average of the 
        stresses in each phase.
        
        Parameters
        ----------
        u : Function Displacement field
        v : Function Velocity field
        T : Function Current temperature field
        T0 : Function Initial temperature field
        J : Function Jacobian of the transformation (determinant of deformation gradient)
        
        Returns
        -------
        Function
            Complete 3D Cauchy stress tensor including elastic, multiphase, and plastic effects,
            but not including damage effects which are applied separately.
        """
        if isinstance(self.material, list):
            return self._calculate_multiphase_stress(u, v, T, T0, J)
        # Single material case
        return self._calculate_single_phase_stress(u, v, T, T0, J)
    
    def _calculate_multiphase_stress(self, u, v, T, T0, J):
        """Calculate stress for multiphase materials.
        
        Computes weighted average of stresses from each material phase.
        
        Parameters
        ----------
        u, v, T, T0, J : Function See stress_3D method for details
            
        Returns
        -------
        Function Weighted average stress tensor
        """
        # Initialize storage for component stresses
        self.pressure_list = []
        self.pseudo_pressure_list = []
        self.deviatoric_list = []
        
        # Calculate stress components for each material phase
        for i, material in enumerate(self.material):
            relative_density = self.relative_rho_0[i] if isinstance(self.relative_rho_0, list) else 1
            pressure, pseudo_pressure, deviatoric = self._calculate_stress_components(
                u, v, T, T0, J, material, relative_density)
            
            self.pressure_list.append(pressure)
            self.pseudo_pressure_list.append(pseudo_pressure)
            self.deviatoric_list.append(deviatoric)
        
        # Calculate weighted averages using concentration fractions
        n_materials = len(self.material)
        self.p = sum(self.multiphase.c[i] * self.pressure_list[i] for i in range(n_materials))
        self.pseudo_p = sum(self.multiphase.c[i] * self.pseudo_pressure_list[i] for i in range(n_materials))
        self.s = sum(self.multiphase.c[i] * self.deviatoric_list[i] for i in range(n_materials))
        
        # Compute total stress
        return -(self.p + self.pseudo_p) * Identity(3) + self.s
    
    def _calculate_single_phase_stress(self, u, v, T, T0, J):
        """Calculate stress for a single phase material.
        
        Parameters
        ----------
        u, v, T, T0, J : Function See stress_3D method for details.
            
        Returns
        -------
        Function Stress tensor for the single material.
        """
        # Calculate stress components
        self.p, self.pseudo_p, self.s = self._calculate_stress_components(
            u, v, T, T0, J, self.material)
        
        # Return total stress
        return -(self.p + self.pseudo_p) * Identity(3) + self.s
        # return -(self.p) * Identity(3) + self.s        
    
    def _calculate_stress_components(self, u, v, T, T0, J, material, relative_density = 1):
        """Calculate individual stress components for a given material.
        
        Breaks down the stress calculation into pressure, pseudo-pressure and 
        deviatoric components.
        
        Parameters
        ----------
        u, v, T, T0, J : Function See stress_3D method for details.
        material : Material Material properties.
        relative_density : float or Function, optional Relative initial density, default is 1.
        Returns
        -------
        tuple (pressure, pseudo_pressure, deviatoric_stress)
        """
        pressure = self.eos.set_eos(J * relative_density, T, T0, material, self.quadrature)
        deviatoric = self._calculate_deviatoric_stress(u, v, J, T, T0, material)
        
        # Calculate pseudo-pressure for stabilization if enabled
        if self.is_damping:
            pseudo_pressure = self.pseudo_pressure(v, material, J, self.h)
        else:
            pseudo_pressure = 0
            
        return pressure, pseudo_pressure, deviatoric
    
    def _calculate_deviatoric_stress(self, u, v, J, T, T0, material):
        """Calculate the deviatoric part of the stress tensor.
        
        Handles different material models including elasticity and plasticity.
        
        Parameters
        ----------
        u, v, J, T, T0 : Function See stress_3D method for details.
        material : Material Material properties.
            
        Returns
        -------
        Function Deviatoric stress tensor.
        """
        
        # Déléguer le calcul au modèle plastique si applicable
        if self.plastic_model:
            return self.plastic.compute_deviatoric_stress(u, v, J, T, T0, material, self.deviator)
        
        # Cas purement élastique
        if material.dev_type == "Hypoelastic":
            return self.deviator.set_hypoelastic_deviator(u, v, J, material)
        else:
            return self.deviator.set_elastic_dev(u, v, J, T, T0, material)        
