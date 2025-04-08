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
Created on Mon Sep 26 18:06:31 2022

@author: bouteillerp
ConstitutiveLaw est le fichier principal gérant la loi de comportement dans CHARON
il appelle les sous fichiers elastic.py, plastic.py et damage.py pour éventuellement
définir un comportement plus complexe.
"""
from .eos import EOS
from .deviator import Deviator
from .plastic import HPPPlastic, FiniteStrainPlastic, JAXJ2Plasticity, JAXGursonPlasticity
from .damage import PhaseField, StaticJohnson, DynamicJohnson, InertialJohnson

from ufl import dot, Identity, dev
from ..utils.generic_functions import npart

class ConstitutiveLaw:
    """Manages the constitutive relations for mechanical simulations.

    This class orchestrates the calculation of stresses based on the current state
    of the system, combining effects from various physical phenomena like elasticity,
    plasticity, and damage.
    """
    
    def __init__(self, u, material, plastic_model, damage_model, multiphase, 
                 name, kinematic, quadrature, damping, is_hypo, relative_rho_0, h):
        """
        Parameters
        ----------
        u : Function, champ de déplacement.
        material : Objet de la classe Material.
        plastic_model : String, modèle de plasticité (HPP_Plasticity, Finite_Plasticity ou None)
        damage_model : String, modèle d'endommagement
        multiphase : Objet de la classe multiphase.
        name : String, nom du modèle mécanique.
        kinematic : Objet de la classe Kinematic.
        quad :  Objet de la classe quadrature.
        damping : Dic : dictionnaire contenant les paramètres d'amortissement
        relative_rho_0 : champ des masses volumiques initiales relatives
        """
        self.material = material
        self.mesh = u.function_space.mesh
        self.h = h
        self.plastic_model = plastic_model
        self.damage_model = damage_model
        self.multiphase = multiphase
        self.kinematic = kinematic
        self.quadrature = quadrature
        self.set_damping(damping)
        self.eos = EOS()
        self.deviator = Deviator(kinematic, name, quadrature, material)

        self.name = name
        self.relative_rho_0 = relative_rho_0
        if self.damage_model != None:
            self.damage = self.damage_class()(self.mesh, quadrature)
        if self.plastic_model != None:
            self.plastic = self.plastic_class(name)(u, material.devia.mu, name, kinematic, quadrature, self.plastic_model)

    def set_damping(self, damping):
        """
        Initialise les paramètres de la pseudo-viscosité
        """
        self.is_damping = damping["damping"]
        self.Klin = damping["linear_coeff"]
        self.Kquad = damping["quad_coeff"]
        self.correction = damping["correction"]

    def pseudo_pressure(self, velocity, material, jacobian):
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
            
        Notes
        -----
        The calculation depends on the model type (Cartesian, cylindrical, etc.)
        and includes linear and quadratic viscosity terms.
        """
        div_v  = self.kinematic.div(velocity)
        lin_Q = self.Klin * material.rho_0 * material.celerity * self.h * npart(div_v)
        if self.name in ["CartesianUD", "CylindricalUD", "SphericalUD"]: 
            quad_Q = self.Kquad * material.rho_0 * self.h**2 * npart(div_v) * div_v 
        elif self.name in ["PlaneStrain", "Axisymetric", "Tridimensionnal"]:
            quad_Q = self.Kquad * material.rho_0 * self.h**2 * dot(npart(div_v), div_v)
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
        u : Function Displacement field.
        v : Function Velocity field.
        T : Function Current temperature field.
        T0 : Function Initial temperature field.
        J : Function Jacobian of the transformation (determinant of deformation gradient).
        
        Returns
        -------
        Function Complete 3D Cauchy stress tensor including elastic, multiphase, and plastic effects,
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
        u, v, T, T0, J : Function See stress_3D method for details.
            
        Returns
        -------
        Function Weighted average stress tensor.
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
    
    def _calculate_stress_components(self, u, v, T, T0, J, material, relative_density=1):
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
            pseudo_pressure = self.pseudo_pressure(v, material, J)
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
        # Select appropriate deviatoric model based on material type
        if material.dev_type == "Hypoelastic":
            deviatoric = self.deviator.set_hypoelastic_deviator(u, v, J, material)
        elif self.plastic_model == "Finite_Plasticity":
            deviatoric = material.devia.mu / J**(5./3) * dev(self.plastic.Be_trial())
        elif self.plastic_model == "J2_JAX":
            deviatoric = material.devia.mu / J * dev(self.plastic.Be_bar_old_3D)
        else:
            deviatoric = self.deviator.set_elastic_dev(u, v, J, T, T0, material)
            
        # Apply plastic correction if needed
        if self.plastic_model == "HPP_Plasticity":
            deviatoric -= self.plastic.plastic_correction(material.devia.mu)
            
        return deviatoric

    def plastic_class(self, name):
        """
        Renvoie le nom de la classe plastique à appeler
        Parameters
        ----------
        name : String, nom du modèle mécanique.
        """
        if self.plastic_model == "HPP_Plasticity":
            return HPPPlastic
        elif self.plastic_model == "Finite_Plasticity":
            return FiniteStrainPlastic
        elif self.plastic_model == "J2_JAX":
            return JAXJ2Plasticity
        elif self.plastic_model == "JAX_Gurson":
            return JAXGursonPlasticity
        else:
            raise ValueError("This model do not exist, did you mean \
                             HPP_Plasticity or Finite_Plasticity ?")
    def damage_class(self):
        """
        Renvoie le nom de la classe endommagement à appeler
        """
        if self.damage_model == "PhaseField":
            return PhaseField
        elif self.damage_model == "Johnson":
            return StaticJohnson   
        elif self.damage_model == "Johnson_dyn":
            return DynamicJohnson
        elif self.damage_model == "Johnson_inertiel":
            return InertialJohnson
        else:
            raise ValueError("Unknown damage model")
         
    def set_plastic_driving(self):
        """
        Calcule la force motrice plastique en appelant la méthode de l'objet plastic.
        Si l'étude est élasto-plastique endommageable, cette force motrice est pondérée
        par la variable d'endommagement.
        """
        if self.plastic_model == "HPP_Plasticity":
            self.plastic.plastic_driving_force(self.s)
            if self.damage_model !=None:
                self.plastic.A *= self.damage.g_d
                
        elif self.plastic_model == "Finite_Plasticity":
            self.plastic.set_expressions()
            
    def set_damage_driving(self, u, J):
        """
        Initialise l'évolution de l'endommagement

        Parameters
        ----------
        u : Function, champ de déplacement.
        J : Expression, jacobien de la transformation.
        """
        if self.damage_model in ["Johnson", "Johnson_dyn", "Johnson_inertiel"]:
            self.damage.set_p_mot(self.p)
        else:
            self.eHelm = self.Helmholtz_energy(u, J, self.material)
            self.damage.set_NL_energy(self.eHelm) 
            
def Helmholtz_energy(self, u, J, mat):
    """Return the Helmholtz free energy.
    
    This method delegates the calculation to the appropriate 
    EOS and deviator models.
    
    Parameters
    ----------
    u : Function Displacement field
    J : Expression Jacobian of the transformation
    mat : Material Material to study
        
    Returns
    -------
    Helmholtz free energy
    """
    # Get volumetric energy from EOS
    try:
        psi_vol = mat.eos.volumetric_helmholtz_energy(u, J, self.kinematic, mat.eos_type)
    except:
        raise ValueError("Phase field analysis has not been implemented for this eos")
    # Get isochoric energy from deviator
    try:
        psi_iso_vol = mat.devia.isochoric_helmholtz_energy(u, self.kinematic)
    except:
        raise ValueError("Phase field analysis has not been implemented for this deviatoric law")
    return psi_vol + psi_iso_vol