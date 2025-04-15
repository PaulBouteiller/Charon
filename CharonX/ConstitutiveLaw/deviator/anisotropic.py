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
Created on Wed Apr  2 11:37:27 2025

@author: bouteillerp
Anisotropic Hyperelastic Deviatoric Stress Model
================================================

This module implements a general anisotropic hyperelastic framework for the deviatoric
part of the stress tensor. It provides a flexible approach to modeling materials with
complex directional properties, supporting various symmetry classes including orthotropic,
transversely isotropic, and fully anisotropic behavior.

The implementation features:
- Support for direct specification of the stiffness tensor
- Built-in material parameter conversion for common symmetry classes
- Transformation capabilities to align material axes
- Optional nonlinear stiffness modulation

Classes:
--------
AnisotropicDeviator : General anisotropic hyperelastic model
    Supports multiple initialization methods
    Implements the stress calculation for anisotropic hyperelasticity
    Provides tensor transformation utilities
"""
from ...utils.tensor_operations import (symetrized_tensor_product, Voigt_to_tridim, 
                                        tridim_to_Voigt, bulk_anisotropy_tensor,
                                        polynomial_expand, polynomial_derivative)
from ufl import as_tensor, as_matrix, dev, inv, inner, dot, Identity
from .base_deviator import BaseDeviator
from scipy.linalg import block_diag
from math import cos, sin
from numpy import array, diag, ndarray, insert
from numpy.linalg import inv as np_inv

class AnisotropicDeviator(BaseDeviator):
    """General anisotropic hyperelastic deviatoric stress model.
    
    This model implements a general anisotropic formulation suitable for
    materials with complex directional properties.
    
    Attributes
    ----------
    C : array Stiffness tensor in Voigt notation
    f_func_coeffs : list or None Coefficients for optional stiffness modulation functions
    """
    
    def required_parameters(self):
        """Return the list of required parameters.
        
        Returns
        -------
        list List of required parameters based on initialization method
        """
        # Case 1: Direct stiffness tensor
        if "C" in self.params:
            return ["C"]
        
        # Case 2: Orthotropic material
        elif "ET" in self.params:
            return ["ET", "EL", "EN", "nuLT", "nuLN", "nuTN", "muLT", "muLN", "muTN"]
        
        # Case 3: Transversely isotropic material
        elif "ET" in self.params and "EL" in self.params and "nuT" in self.params:
            return ["ET", "EL", "nuT", "nuL", "muL"]
        
        # Case 4: Isotropic material
        elif "E" in self.params and "nu" in self.params:
            return ["E", "nu"]
        
        # Default case
        return ["C"]
    
    def __init__(self, params):
        """Initialize the general anisotropic deviatoric model.
        
        The model can be initialized in multiple ways:
        1. With a direct stiffness tensor: params = {"C": stiffness_tensor}
        2. With orthotropic parameters: params = {"ET": ET, "EL": EL, ...}
        3. With transversely isotropic parameters: params = {"ET": ET, "EL": EL, "nuT": nuT, ...}
        4. With isotropic parameters: params = {"E": E, "nu": nu}
        
        Additional parameters:
        - f_func: [optional] Coefficients for stiffness modulation functions
        - g_func: [optional] Coefficients for coupling stress modulation functions 
        doit être un numpy array dont le premier coefficient corrspond au terme linéaire.
        - rotation: [optional] Rotation angle in radians
        
        Parameters
        ----------
        params : dict Parameters for material behavior
        """
        self.params = params  # Store for required_parameters method
        super().__init__(params)
        
        # Store stiffness modulation parameters if provided
        self.f_func_coeffs = params.get("f_func", None)
        if self.f_func_coeffs is not None:
            size = len(self.f_func_coeffs)
            for i in range(size):
                for j in range(size):
                    if isinstance(self.f_func_coeffs[i][j], ndarray):
                        self.f_func_coeffs[i][j] = insert(self.f_func_coeffs[i][j], 0, 1)
        # Store coupling stress modulation parameters if provided
        self.g_func_coeffs = params.get("g_func", None)
        if self.g_func_coeffs is not None:
            self.g_func_coeffs = insert(self.g_func_coeffs, 0, 1)
        
        # Initialize stiffness tensor based on provided parameters
        if "C" in params:
            # Case 1: Direct stiffness tensor provided
            self.C = params["C"]
            self._log_direct_stiffness()
        else:
            # Build stiffness tensor from material parameters
            self._build_stiffness_tensor(params)
        
        # Apply rotation if specified
        if "rotation" in params:
            self._apply_rotation(params["rotation"])
    
    def _log_direct_stiffness(self):
        """Log stiffness tensor components when directly provided."""
        print("Using direct stiffness tensor (C)")
    
    def _build_stiffness_tensor(self, params):
        """Build stiffness tensor from material parameters.
        
        Parameters
        ----------
        params : dict Material parameters
        """
        # Determine material type and build appropriate stiffness tensor
        if "ET" in params and "EL" in params and "EN" in params:
            # Case 2: Orthotropic material
            self._build_orthotropic_stiffness(params)
        elif "ET" in params and "EL" in params and "nuT" in params:
            # Case 3: Transversely isotropic material  
            self._build_transverse_isotropic_stiffness(params)
        else:
            raise ValueError("Invalid parameter set for anisotropic material")
    
    def _build_orthotropic_stiffness(self, params):
        """Build stiffness tensor for orthotropic material.
        
        Parameters
        ----------
        params : dict Orthotropic material parameters
        """
        ET = params["ET"]
        EL = params["EL"]
        EN = params["EN"]
        nuLT = params["nuLT"]
        nuLN = params["nuLN"]
        nuTN = params["nuTN"]
        muLT = params["muLT"]
        muLN = params["muLN"]
        muTN = params["muTN"]
        
        print("Building orthotropic stiffness tensor with parameters:")
        print(f"Young's modulus (longitudinal): {EL}")
        print(f"Young's modulus (transverse): {ET}")
        print(f"Young's modulus (normal): {EN}")
        print(f"Poisson ratio (nu_LT): {nuLT}")
        print(f"Poisson ratio (nu_LN): {nuLN}")
        print(f"Poisson ratio (nu_TN): {nuTN}")
        print(f"Shear modulus (mu_LT): {muLT}")
        print(f"Shear modulus (mu_LN): {muLN}")
        print(f"Shear modulus (mu_TN): {muTN}")
        
        # Create compliance matrix and convert to stiffness
        Splan = array([[1. / EL, -nuLT / EL, -nuLN / EL],
                       [-nuLT / EL, 1. / ET, -nuTN / ET],
                       [-nuLN / EL, -nuTN / ET, 1. / EN]])
        S = block_diag(Splan, diag([1 / muLN, 1 / muLT, 1 / muTN]))
        self.C = np_inv(S)
    
    def _build_transverse_isotropic_stiffness(self, params):
        """Build stiffness tensor for transversely isotropic material.
        
        Parameters
        ----------
        params : dict Transversely isotropic material parameters
        """
        ET = params["ET"]
        EL = params["EL"]
        nuT = params["nuT"]
        nuL = params["nuL"]
        muL = params["muL"]
        
        print("Building transversely isotropic stiffness tensor with parameters:")
        print(f"Young's modulus (longitudinal): {EL}")
        print(f"Young's modulus (transverse): {ET}")
        print(f"Poisson ratio (transverse): {nuT}")
        print(f"Poisson ratio (longitudinal): {nuL}")
        print(f"Shear modulus (longitudinal): {muL}")
        
        # Calculate derived parameters
        muT = ET / (2 * (1 + nuT))
        
        # Reuse orthotropic calculation with appropriate parameters
        self._build_orthotropic_stiffness({
            "ET": ET, "EL": EL, "EN": ET,
            "nuLT": nuL, "nuLN": nuL, "nuTN": nuT,
            "muLT": muL, "muLN": muL, "muTN": muT
        })
    
    def _apply_rotation(self, alpha):
        """Apply rotation to the stiffness tensor.
        
        Parameters
        ----------
        alpha : float Rotation angle in radians
        """
        print(f"Applying rotation of {alpha} radians to stiffness tensor")
        
        c = cos(alpha)
        s = sin(alpha)
        
        # Create rotation matrix for 6x6 tensor in Voigt notation
        R = array([[c**2, s**2, 0, 2*s*c, 0, 0],
                   [s**2, c**2, 0, -2*s*c, 0, 0],
                   [0, 0, 1, 0, 0, 0],
                   [-c*s, c*s, 0, c**2 - s**2, 0, 0],
                   [0, 0, 0, 0, c, s],
                   [0, 0, 0, 0, -s, c]])
        
        # Apply rotation to stiffness tensor
        self.C = R.dot(self.C.dot(R.T))
    
    def calculate_stress(self, u, v, J, T, T0, kinematic):
        """Calculate the deviatoric stress tensor for anisotropic hyperelasticity.
        
        Parameters
        ----------
        u, v, J, T, T0 : Function See stress_3D method in ConstitutiveLaw.py for details.
        kinematic : Kinematic Kinematic handler object
            
        Returns
        -------
        Function Deviatoric stress tensor
        """
        
        def rig_lin_derivative(C, f_func, J):
            size = len(C)
            deriv_C_list = [[C[i][j] * polynomial_derivative(J, 1, f_func[i][j]) 
                             for i in range(size)] for j in range(size)]
            return deriv_C_list
        
        def compute_pibar_contribution(M0, J, u):
            g_func = self.g_func_coeffs
            if g_func is None:
                pibar = 1./3 * (J-1) * M0
            elif isinstance(g_func, ndarray):
                print("Single fit")
                pibar = 1./3 * (J-1) * polynomial_expand(J, 1, g_func)  * M0
            elif isinstance(g_func, list):
                gM0 = [[polynomial_expand(J, 1, g_func[i][j]) * M0[i, j] for i in range(3)] for j in range(3)]
                pibar = 1./3 * (J - 1) * as_tensor(gM0)
            return J**(-5./3) * dev(kinematic.push_forward(pibar, u))
        
        def compute_DEbar_contribution(M0, J, u, GLDBar_V, inv_C):
            g_func = self.g_func_coeffs
            if g_func is None:
                D = 1./3 * symetrized_tensor_product(M0, inv_C)
            elif isinstance(g_func, ndarray):
                D = 1./3 * ((J-1) * polynomial_derivative(J, 1, g_func) + polynomial_expand(J, 1, g_func)) * symetrized_tensor_product(M0, inv_C)
            elif isinstance(g_func, list):
                pibar_derivative = 1./3 * as_tensor([[M0[i, j] * (polynomial_expand(J, 1, g_func[i][j]) 
                                                                  + (J - 1) * polynomial_derivative(J, 1, g_func[i][j]))
                                                      for i in range(3)] for j in range(3)])
                D = symetrized_tensor_product(pibar_derivative(M0, g_func, J), inv_C)
            DE = Voigt_to_tridim(dot(D, GLDBar_V))
            return kinematic.push_forward(DE, u)
        
        def compute_CBarEbar_contribution(RigLin, J, u, GLDBar_V):
            f_func = self.f_func_coeffs
            if f_func is not None:
                size = len(RigLin)
                RigiLinBar = RigLin.tolist()
                for i in range(size):
                    for j in range(size):
                        if f_func[i][j] is not None:
                            RigiLinBar[i][j] *= polynomial_expand(J, 1, f_func[i][j])
                CBarEbar = Voigt_to_tridim(dot(as_matrix(RigiLinBar), GLDBar_V))
            else:
                CBarEbar = Voigt_to_tridim(dot(as_matrix(RigLin), GLDBar_V))
            return J**(-5./3) * dev(kinematic.push_forward(CBarEbar, u)) 
        
        def compute_EEbar_contribution(RigLin, J, u, GLDBar_V, inv_C, GLD_bar):
            f_func = self.f_func_coeffs
            size = len(RigLin)
            DerivRigiLinBar = RigLin.tolist()
            for i in range(size):
                for j in range(size):
                    if f_func[i][j] is not None:
                        DerivRigiLinBar[i][j] *= polynomial_derivative(J, 1, f_func[i][j])
            EE = Voigt_to_tridim(1./2 * inner(inv_C, GLD_bar) * dot(as_matrix(DerivRigiLinBar), GLDBar_V))
            return dev(kinematic.push_forward(EE, u))
            
        
        RigLin = self.C
        M0 = bulk_anisotropy_tensor(RigLin)
        # First term 
        term_1 = compute_pibar_contribution(M0, J, u)
        # Build different strain measure
        C = kinematic.C_3D(u)
        C_bar = J**(-2./3) * C
        inv_C = inv(C)
        GLD_bar = 1./2 * (C_bar - Identity(3))
        GLDBar_V = tridim_to_Voigt(GLD_bar)
        # Second term 
        term_2 = compute_DEbar_contribution(M0, J, u, GLDBar_V, inv_C)
        # Third term 
        term_3 = compute_CBarEbar_contribution(RigLin, J, u, GLDBar_V)  
        
        # Optional fourth term
        if self.f_func_coeffs is not None:
            # DerivRigiLinBar = rig_lin_derivative(RigLin, self.f_func_coeffs, J)
            # EE = Voigt_to_tridim(1./2 * inner(inv_C, GLD_bar) * dot(as_matrix(DerivRigiLinBar), GLDBar_V))
            term_4 = compute_EEbar_contribution(RigLin, J, u, GLDBar_V, inv_C, GLD_bar)
            
            return term_1 + term_2 + term_3 + term_4
            # return term_2 + term_3 + term_4
            # return term_3 + term_4
            # return term_1 + term_3
            # return term_3
            # return term_1
        else:
            return term_1 + term_2 + term_3
            # return term_1
            # return term_3