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
Stiffness Matrix Builders
========================

This module provides utilities to construct stiffness matrices from material parameters
for various symmetry classes (isotropic, transversely isotropic, orthotropic).

Functions
---------
build_isotropic_stiffness : Build stiffness matrix for isotropic material
build_transverse_isotropic_stiffness : Build stiffness matrix for transversely isotropic material  
build_orthotropic_stiffness : Build stiffness matrix for orthotropic material
compute_bulk_modulus : Compute equivalent bulk modulus from stiffness matrix
validate_orthotropic_stability : Check stability conditions for orthotropic materials
"""

import numpy as np
from scipy.linalg import block_diag
from numpy.linalg import inv as np_inv


def build_transverse_isotropic_stiffness(EL, ET, nuL, nuT, muL):
    """Build stiffness matrix for transversely isotropic material.
    
    Parameters
    ----------
    EL : float Young's modulus in longitudinal direction (fiber direction)
    ET : float Young's modulus in transverse direction  
    nuL : float Poisson's ratio longitudinal-transverse (nuLT = nuLN)
    nuT : float Poisson's ratio transverse-transverse (nuTN)
    muL : float Shear modulus in longitudinal planes (muLT = muLN)
        
    Returns
    -------
    ndarray 6x6 stiffness matrix in Voigt notation
        
    Notes
    -----
    Assumes L is the longitudinal (fiber) direction, T and N are transverse directions.
    Convention: L=x, T=y, N=z in Voigt notation [xx, yy, zz, yz, xz, xy]
    """
    print("Building transversely isotropic stiffness tensor:")
    print(f"Young's modulus (longitudinal): {EL}")
    print(f"Young's modulus (transverse): {ET}")
    print(f"Poisson ratio (longitudinal): {nuL}")
    print(f"Poisson ratio (transverse): {nuT}")
    print(f"Shear modulus (longitudinal): {muL}")
 
    # Calculate derived parameters
    muT = ET / (2 * (1 + nuT))  # Shear modulus in transverse plane
    
    # Use orthotropic builder with appropriate parameters
    return build_orthotropic_stiffness(
        EL=EL, ET=ET, EN=ET,
        nuLT=nuL, nuLN=nuL, nuTN=nuT,
        muLT=muL, muLN=muL, muTN=muT
    )


def build_orthotropic_stiffness(EL, ET, EN, nuLT, nuLN, nuTN, muLT, muLN, muTN):
    """Build stiffness matrix for orthotropic material.
    
    Parameters
    ----------
    EL, ET, EN : float Young's moduli in L, T, N directions
    nuLT, nuLN, nuTN : float Poisson's ratios
    muLT, muLN, muTN : float Shear moduli
        
    Returns
    -------
    ndarray 6x6 stiffness matrix in Voigt notation
        
    Notes
    -----
    Convention: L=x, T=y, N=z in Voigt notation [xx, yy, zz, yz, xz, xy]
    """
    print("Building orthotropic stiffness tensor:")
    print(f"Young's modulus (L): {EL}")
    print(f"Young's modulus (T): {ET}")
    print(f"Young's modulus (N): {EN}")
    print(f"Poisson ratio (nu_LT): {nuLT}")
    print(f"Poisson ratio (nu_LN): {nuLN}")
    print(f"Poisson ratio (nu_TN): {nuTN}")
    print(f"Shear modulus (mu_LT): {muLT}")
    print(f"Shear modulus (mu_LN): {muLN}")
    print(f"Shear modulus (mu_TN): {muTN}")
    # Build compliance matrix
    S_diag = np.array([[1/EL, -nuLT/EL, -nuLN/EL],
                       [-nuLT/EL, 1/ET, -nuTN/ET],
                       [-nuLN/EL, -nuTN/ET, 1/EN]])
    
    S_shear = np.diag([1/muTN, 1/muLN, 1/muLT])  # [yz, xz, xy] order
    
    S = block_diag(S_diag, S_shear)
    
    # Convert to stiffness matrix
    C = np_inv(S)
    
    return C


def compute_bulk_modulus(C):
    """Compute equivalent bulk modulus from stiffness matrix.
    
    Parameters
    ----------
    C : ndarray 6x6 stiffness matrix
        
    Returns
    -------
    float Equivalent bulk modulus
        
    Notes
    -----
    For anisotropic materials, this is an approximation based on:
    K_eq = (1/9) * sum_{i,j=1}^{3} C_{ij}
    """
    # Average of first 3x3 block (normal stress components)
    K_eq = np.sum(C[:3, :3]) / 9
    
    print(f"Equivalent bulk modulus: {K_eq:.1f}")
    return K_eq