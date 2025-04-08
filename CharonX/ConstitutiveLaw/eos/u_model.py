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
Created on Wed Apr  2 11:21:36 2025

@author: bouteillerp
"""
"""Hyperelastic isotropic equation of state models (U-family)."""

from ufl import ln, sqrt
from .base_eos import BaseEOS
from ...utils.generic_functions import ppart

class UEOS(BaseEOS):
    """Hyperelastic isotropic equation of state with one coefficient.
    
    This class implements various energy potentials (U1 through U8)
    that relate volume change to pressure for hyperelastic materials.
    
    Attributes
    ----------
    kappa : float Bulk modulus (Pa)
    alpha : float Thermal expansion coefficient (1/K)
    """
    
    def required_parameters(self):
        """Return the list of required parameters.
        
        Returns
        -------
        list List of parameter names
        """
        return ["kappa", "alpha"]
    
    def __init__(self, params):
        """Initialize the hyperelastic isotropic EOS.
        
        Parameters
        ----------
        params : dict Dictionary containing:
                        kappa : float Bulk modulus (Pa)
                        alpha : float Thermal expansion coefficient (1/K)
        """
        super().__init__(params)
        
        # Store parameters
        self.kappa = params["kappa"]
        self.alpha = params["alpha"]
        
        # Log parameters
        print(f"Bulk modulus: {self.kappa}")
        print(f"Thermal expansion coefficient: {self.alpha}")
    
    def celerity(self, rho_0):
        """Calculate wave speed estimation for hyperelastic material.
        
        Parameters
        ----------
        rho_0 : float Initial density
            
        Returns
        -------
        float Wave speed
        """
        return sqrt(self.kappa / rho_0)
    
    def pressure(self, J, T, T0, material, quadrature, model_type="U5"):
        """Calculate pressure using hyperelastic U-model.
        
        Available models:
        - U1: P = -kappa * (J - 1 - 3 * alpha * (T - T0))
        - U5: P = -kappa * (ln(J) - ln(1 + 3 * alpha * (T - T0)))
        - U8: P = -kappa/2 * (ln(J) - 1/J - ln(1 + 3*alpha*(T-T0)) + 1/(1+alpha*(T-T0)))
        
        Parameters
        ----------
        J, T, T0, material : See stress_3D method in ConstitutiveLaw.py for details.
        quadrature : QuadratureHandler Handler for quadrature spaces.
            
        Returns
        -------
        Expression Pressure
        """
        if model_type == "U1":
            return -self.kappa * (J - 1 - 3 * self.alpha * (T - T0))
        elif model_type == "U5":
            return -self.kappa * (ln(J) - ln(1 + 3 * self.alpha * (T - T0)))
        elif model_type == "U8":
            return -self.kappa/2 * (ln(J) - 1/J - ln(1 + 3*self.alpha*(T-T0)) + 1/(1+self.alpha*(T-T0)))
        else:
            raise ValueError(f"Unsupported U-model: {model_type}")
            
    def volumetric_helmholtz_energy(self, u, J, kinematic, model_type="U5"):
        """Calculate the volumetric Helmholtz free energy for U-models.
        
        Parameters
        ----------
        u, J, kinematic : See Helmholtz_energy method in ConstitutiveLaw.py for details.
            
        Returns
        -------
        Volumetric Helmholtz free energy
        """
        if model_type == "U5":
            return self.kappa * (J * ln(J) - J + 1)
        elif model_type == "U8":
            return self.kappa / 2 * ln(J) * ppart(J-1)
        else:
            raise ValueError(f"Helmholtz energy not implemented for model {model_type}")