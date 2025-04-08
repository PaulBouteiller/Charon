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
Created on Wed Apr  2 11:36:03 2025

@author: bouteillerp
"""
"""Neo-Hookean hyperelastic deviatoric stress model."""

from ufl import dev
from .base_deviator import BaseDeviator

class NeoHookDeviator(BaseDeviator):
    """Neo-Hookean hyperelastic deviatoric stress model.
    
    This model implements the deviatoric stress for a Neo-Hookean solid:
    s = μ/J^(5/3) * dev(B)
    
    where B is the left Cauchy-Green deformation tensor.
    
    Attributes
    ----------
    mu : float Shear modulus (Pa)
    """
    
    def required_parameters(self):
        """Return the list of required parameters.
        
        Returns
        -------
        list List with "mu"
        """
        return ["mu"]
    
    def __init__(self, params):
        """Initialize the Neo-Hookean deviatoric model.
        
        Parameters
        ----------
        params : dict Dictionary containing:
                        mu : float Shear modulus (Pa)
        """
        super().__init__(params)
        self.mu = params["mu"]
        print(f"Shear modulus: {self.mu}")
    
    def calculate_stress(self, u, v, J, T, T0, kinematic):
        """Calculate the deviatoric stress tensor for Neo-Hookean hyperelasticity.
        
        s = μ/J^(5/3) * dev(B)
        
        Parameters
        ----------
        u, v, J, T, T0 : Function See stress_3D method in ConstitutiveLaw.py for details.
        kinematic : Kinematic Kinematic handler object
            
        Returns
        -------
        ufl.core.expr.Expr Deviatoric stress tensor
        """
        B = kinematic.B_3D(u)
        return self.mu / J**(5./3) * dev(B)
    
    def isochoric_helmholtz_energy(self, u, J, kinematic):
        """Calculate the isochoric Helmholtz free energy for Neo-Hookean model.
        
        Parameters
        ----------
        u, J, kinematic : See Helmholtz_energy method in ConstitutiveLaw.py for details.
            
        Returns
        -------
        Isochoric Helmholtz free energy
        """
        return self.mu * (kinematic.BBarI(u) - 3)