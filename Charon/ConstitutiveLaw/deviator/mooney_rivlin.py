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
Created on Wed Apr  2 11:36:30 2025

@author: bouteillerp

Mooney-Rivlin Hyperelastic Deviatoric Stress Model
==================================================

This module implements the Mooney-Rivlin hyperelastic model for the deviatoric part
of the stress tensor. This model extends the Neo-Hookean model with an additional
term that depends on the second invariant of the left Cauchy-Green deformation tensor.
"""

from ufl import dev, dot, tr
from .base_deviator import BaseDeviator

class MooneyRivlinDeviator(BaseDeviator):
    """Mooney-Rivlin hyperelastic deviatoric stress model.
    
    This model extends the Neo-Hookean model with an additional term:
    s = μ/J^(5/3) * dev(B) - μ_quad/J^(7/3) * dev(B·B - tr(B)·B)
    
    Attributes
    ----------
    mu : float Primary shear modulus (Pa)
    mu_quad : float Secondary shear modulus (Pa)
    """
    
    def required_parameters(self):
        """Return the list of required parameters.
        
        Returns
        -------
        list List with "mu" and "mu_quad"
        """
        return ["mu", "mu_quad"]
    
    def __init__(self, params):
        """Initialize the Mooney-Rivlin deviatoric model.
        
        Parameters
        ----------
        params : dict
            Dictionary containing:
            mu : float Primary shear modulus (Pa)
            mu_quad : float Secondary shear modulus (Pa)
        """
        super().__init__(params)
        
        # Store parameters
        self.mu = params["mu"]
        self.mu_quad = params["mu_quad"]
        
        # Log parameters
        print(f"Shear modulus: {self.mu}")
        print(f"Quadratic shear modulus: {self.mu_quad}")
    
    def calculate_stress(self, u, v, J, T, T0, kinematic):
        """Calculate the deviatoric stress tensor for Mooney-Rivlin hyperelasticity.
        
        s = μ/J^(5/3) * dev(B) - μ_quad/J^(7/3) * dev(B·B - tr(B)·B)
        
        Parameters
        ----------
        u, v, J, T, T0 : Function See stress_3D method in ConstitutiveLaw.py for details.
        kinematic : Kinematic Kinematic handler object
            
        Returns
        -------
        ufl.core.expr.Expr Deviatoric stress tensor
        """
        B = kinematic.B_3D(u)
        term1 = self.mu / J**(5./3) * dev(B)
        term2 = self.mu_quad / J**(7./3) * dev(dot(B, B) - tr(B) * B)
        return term1 - term2