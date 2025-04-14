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
Hypoelastic Deviatoric Stress Model
==================================

This module implements a hypoelastic formulation for the deviatoric part of the
stress tensor. Unlike hyperelastic models that derive stress from a strain energy
potential, hypoelastic models directly relate the stress rate to the rate of
deformation.

The hypoelastic approach offers advantages for:
- Path-dependent material behavior
- Large deformation problems with complex loading histories
- Materials where the stress-strain relationship depends on the deformation history
- Simulations where the rate effects are important

Key features:
- Direct stress rate formulation (σ̇ = C:D)
- Objective stress rate implementations (Jaumann)

Classes:
--------
HypoelasticDeviator : Hypoelastic deviatoric stress model
    Implements stress rate calculations based on deformation rate
    Supports different objective stress rates
    Handles integration with time-stepping schemes
    Provides compatibility with the overall constitutive framework
"""

from ufl import (sym, skew, dot, inner, dev, Identity, tr)
from dolfinx.fem import (Function, Expression)
from .base_deviator import BaseDeviator

class HypoelasticDeviator(BaseDeviator):
    """Hypoelastic deviatoric stress model.
    
    This model implements a rate-form constitutive law where the stress rate
    is related to the deformation rate tensor through a constitutive tensor.
    The Jaumann rate is used for objectivity.
    
    Attributes
    ----------
    mu : float Shear modulus (Pa)
    model : str Name of the mechanical model (e.g., "PlaneStrain", "Tridimensionnal")
    """
    
    def required_parameters(self):
        """Return the list of required parameters.
        
        Returns
        -------
        list List with "mu"
        """
        return ["mu"]
    
    def __init__(self, params):
        """Initialize the hypoelastic deviatoric model.
        
        Parameters
        ----------
        params : dict
            Dictionary containing:
            mu : float Shear modulus (Pa)
            kinematic : Kinematic Kinematic handler for tensor operations
            model : str Model name (e.g., "CartesianUD", "PlaneStrain")
            quadrature : QuadratureHandler Handler for quadrature spaces
        """
        super().__init__(params)
        
        # Store parameters
        self.mu = params["mu"]
        # Log parameters
        print(f"Shear modulus: {self.mu}")
        
    def set_hypoelastic(self, kinematic, model, quadrature):
        self.kin = kinematic
        self.model = model
        self._setup_function_space(quadrature)
    
    def _setup_function_space(self, quadrature):
        """Set up function spaces for hypoelastic formulation.
        
        Parameters
        ----------
        quadrature : QuadratureHandler
            Handler for quadrature spaces
        """
        if self.model == "CartesianUD":
            self.V_s = quadrature.quadrature_space(["Scalar"])
        elif self.model in ["CylindricalUD", "SphericalUD"]:
            self.V_s = quadrature.quadrature_space(["Vector", 2])
        elif self.model == "PlaneStrain":
            self.V_s = quadrature.quadrature_space(["Vector", 3])
        elif self.model == "Axisymetric":
            self.V_s = quadrature.quadrature_space(["Vector", 4])
        elif self.model == "Tridimensionnal":
            self.V_s = quadrature.quadrature_space(["Tensor", 3, 3])
        else:
            raise ValueError(f"Unsupported model type for hypoelastic formulation: {self.model}")
            
        # Initialize stress storage
        self.s = Function(self.V_s, name="Deviator")
    
    def calculate_stress(self, u, v, J, T, T0, kinematic):
        """Calculate the deviatoric stress tensor for the hypoelastic model.
        
        This is just a placeholder for compatibility with the BaseDeviator interface.
        For hypoelastic models, use calculate_stress_rate instead.
        
        Parameters
        ----------
        u, v, J, T, T0 : Function See stress_3D method in ConstitutiveLaw.py for details.
        kinematic : Kinematic Kinematic handler object
            
        Returns
        -------
        ufl.core.expr.Expr Current deviatoric stress tensor
        """
        # Return the current stress state (which is updated over time by integration)
        return kinematic.reduit_to_3D(self.s, sym=True)
    
    def calculate_stress_rate(self, u, v, J):
        """Calculate the deviatoric stress rate tensor for hypoelastic formulation.
        
        This method implements the Jaumann rate form for objectivity.
        
        Parameters
        ----------
        u : Function Displacement field
        v : Function Velocity field 
        J : Function Jacobian of the deformation
        
        Returns
        -------
        tuple (stress_rate, stress_3D) The stress rate expression and current 3D stress tensor
        """
        # Get the current 3D stress tensor
        mu = self.mu
        s_3D = self.kin.reduit_to_3D(self.s, sym=True)
        
        # Calculate velocity gradient
        L = self.kin.reduit_to_3D(self.kin.Eulerian_gradient(v, u))
        
        # Strain rate tensor (symmetric part of velocity gradient)
        D = sym(L)
        
        # Left Cauchy-Green tensor
        B = self.kin.B_3D(u)

        # Jaumann rate formulation (objective stress rate)
        s_Jaumann_3D = mu/J**(5./3) * (dot(B, D) + dot(D, B) 
                                      - 2./3 * inner(B,D) * Identity(3)
                                      -5./3 * tr(D) * dev(B))
        s_Jaumann = self.kin.tridim_to_reduit(s_Jaumann_3D, sym=True)
        
        # Calculate the complete stress rate depending on model type
        if self.model in ["CartesianUD", "CylindricalUD", "SphericalUD"]:
            # Simple models - Jaumann rate is sufficient
            dot_s = s_Jaumann
        else:
            # More complex models - add the spin correction term
            Omega = skew(L)
            Jaumann_corr = self.kin.tridim_to_reduit(
                dot(Omega, s_3D) - dot(s_3D, Omega), sym=True
            )
            dot_s = s_Jaumann + Jaumann_corr
        
        self.dot_s = Expression(dot_s, self.V_s.element.interpolation_points())
        return s_3D