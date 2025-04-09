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
Created on Wed Apr  2 11:34:38 2025

@author: bouteillerp

Newtonian Fluid Deviatoric Stress Model
======================================

This module implements the Newtonian fluid model for deviatoric stress. It provides
the standard viscous stress tensor formulation for Newtonian fluids, where the stress
is proportional to the rate of deformation.

Classes:
--------
NewtonianFluidDeviator : Deviatoric stress model for Newtonian fluids
    Implements the viscous stress tensor calculation
    Uses velocity gradient instead of displacement
    Provides correct finite strain behavior for fluids
"""


from ufl import sym, dev
from .base_deviator import BaseDeviator

class NewtonianFluidDeviator(BaseDeviator):
    """Deviatoric stress model for Newtonian fluids.
    
    This model implements the standard viscous stress tensor for Newtonian fluids:
    s = 2 * μ * dev(sym(∇v))
    
    Attributes
    ----------
    mu : float
        Dynamic viscosity
    """
    
    def required_parameters(self):
        """Return the list of required parameters.
        
        Returns
        -------
        list
            List of parameter names
        """
        return ["mu"]
    
    def __init__(self, params):
        """Initialize the Newtonian fluid deviatoric model.
        
        Parameters
        ----------
        params : dict
            Dictionary containing:
            mu : float
                Dynamic viscosity
        """
        super().__init__(params)
        
        # Store parameters
        self.mu = params["mu"]
    
    def calculate_stress(self, u, v, J, T, T0, kinematic):
        """Calculate the deviatoric stress tensor for a Newtonian fluid.
        
        s = 2 * μ * dev(sym(∇v))
        
        Parameters
        ----------
        u, v, J, T, T0 : Function See stress_3D method in ConstitutiveLaw.py for details.
        kinematic : Kinematic Kinematic handler object
            
        Returns
        -------
        Function
            Deviatoric stress tensor
        """
        return 2 * self.mu * dev(sym(kinematic.grad_3D(v)))