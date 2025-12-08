# Copyright 2025 CEA
"""
Created on Wed Apr  2 12:00:00 2025

Compressed Intermediate Configuration Hyperelastic Model
======================================================

This module implements the finite-strain anisotropic elasticity model based on a 
pressure-dependent intermediate configuration.

Reference:
    Section "Finite-Strain Elasticity Based on an Intermediate Compressed Configuration"
"""

from ufl import diff, inv, Identity, dot, inner, variable, as_tensor, det
from .base_deviator import BaseDeviator
from ...utils.maths.tensor_rotations import rotation_matrix_direct, rotate_stifness, euler_to_rotation
from numpy import full_like

from dolfinx.fem import functionspace, Function
from dolfinx import default_scalar_type

class CompressedIntermediateDeviator(BaseDeviator):
    """
    Hyperelastic model based on a multiplicative decomposition F = F_tilde * F0(J).
    
    The stress response is derived from:
    1. A geometric evolution of anisotropy F0(J)
    2. An evolving stiffness tensor C_tilde(J)
    
    Attributes
    ----------
    F0_func : callable
        Function taking J and returning the 3x3 tensor F0.
    stiffness_func : callable
        Function taking J and returning the 6x6 stiffness matrix (Voigt) in the intermediate config.
    """
    
    def required_parameters(self):
        return ["F0_func", "stiffness_func"]
    
    def __init__(self, params):
        super().__init__(params)
        self.F0_func = params["F0_func"]
        self.stiffness_func = params["stiffness_func"]
        
    def set_orientation(self, mesh_manager, polycristal_dic):
        mesh = mesh_manager.mesh
        Q = functionspace(mesh, ("DG", 0))
        if "angle" in polycristal_dic and "axis" in polycristal_dic:
            angle_func = Function(Q)
            Q_axis = functionspace(mesh, ("DG", 0, (3, )))    
            axis_func = Function(Q_axis)
            for index, angle, axis in zip(polycristal_dic["tags"], polycristal_dic["angle"], polycristal_dic["axis"]):
                cells = mesh_manager.cell_tags.find(index)
                angle_func.x.array[cells] = full_like(cells, angle, dtype = default_scalar_type)
                axis_func.x.array[3 * cells] = full_like(cells, axis[0], dtype=default_scalar_type)
                axis_func.x.array[3 * cells + 1] = full_like(cells, axis[1], dtype=default_scalar_type)
                axis_func.x.array[3 * cells + 2] = full_like(cells, axis[2], dtype=default_scalar_type)
            self.R = rotation_matrix_direct(angle_func, axis_func)
        elif "euler_angle" in polycristal_dic:
            phi1_func, Phi_func, phi2_func = Function(Q), Function(Q), Function(Q)
            for index, euler_angle in zip(polycristal_dic["tags"], polycristal_dic["euler_angle"]):
                cells = mesh_manager.cell_tags.find(index)
                phi1_func.x.array[cells] = full_like(cells, euler_angle[0], dtype = default_scalar_type)
                Phi_func.x.array[cells] = full_like(cells, euler_angle[1], dtype=default_scalar_type)
                phi2_func.x.array[cells] = full_like(cells, euler_angle[2], dtype=default_scalar_type)
            self.R = euler_to_rotation(phi1_func, Phi_func, phi2_func)     
        
    # def calculate_stress(self, u, v, J, T, T0, kinematic):
    #     """
    #     Calculate the total Cauchy stress tensor including pressure-shear coupling.
        
    #     Implements Eq. (23) and (24):
    #     sigma = -p_eff * I + (1/J) * F_tilde * S_inter * F_tilde.T
    #     """
    #     # 0. UFL Variable for differentiation
    #     J_var = variable(J)
        
    #     # 1. Kinematics
    #     # Macroscopic Right Cauchy-Green C = F.T * F
    #     C = kinematic.right_cauchy_green_3d(u)
    #     F = kinematic.deformation_gradient_3d(u)
    #     I = Identity(3)
        
    #     # 2. Intermediate Configuration Evolution (F0 and derivatives)
    #     F0 = self.F0_func(J_var)
    #     inv_F0 = inv(F0)
    #     dF0_dJ = diff(F0, J_var) # Automatic differentiation of geometric evolution
        
    #     # 3. Intermediate Strain Measure (Eq. 2)
    #     # C_tilde = F0^-T * C * F0^-1
    #     C_tilde = dot(dot(inv_F0.T, C), inv_F0)
    #     E_tilde = 0.5 * (C_tilde - I)
        
    #     # 4. Geometric Coupling Tensor M (Eq. 7 / Section 1.4)
    #     # M = - C_tilde * (dF0/dJ) * F0^-1
    #     M = -dot(dot(C_tilde, dF0_dJ), inv_F0)
        
    #     # 5. Structural Stress in Intermediate Config (S_inter)
    #     # S_inter = C_tilde(J) : E_tilde
    #     if hasattr(self, "R"):
    #         L_tilde_voigt = rotate_stifness(self.stiffness_func(J_var), self.R)
    #     else:
    #         L_tilde_voigt = self.stiffness_func(J_var)
    #     E_tilde_voigt = kinematic.tensor_3d_to_voigt(E_tilde)
        
    #     S_inter_voigt = dot(as_tensor(L_tilde_voigt), E_tilde_voigt)
    #     S_inter = kinematic.voigt_to_tensor_3d(S_inter_voigt)
        
    #     # Term 1: Stiffness evolution coupling
    #     # We need partial derivative of S_inter w.r.t J (keeping strain fixed)
    #     # This is equivalent to (dL_tilde/dJ) : E_tilde
    #     dL_dJ_voigt = diff(L_tilde_voigt, J_var)
    #     dS_dJ_partial_voigt = dot(as_tensor(dL_dJ_voigt), E_tilde_voigt)
    #     dS_dJ_partial = kinematic.voigt_to_tensor_3d(dS_dJ_partial_voigt)
        
    #     term_stiffness = inner(dS_dJ_partial, E_tilde)
        
    #     # Term 2: Geometric coupling
    #     term_geometric = inner(S_inter, M)
        
    #     # 7. Push-forward to Cauchy Stress
    #     # F_tilde = F * F0^-1
    #     F_tilde = dot(F, inv_F0)
        
    #     # Convective part: (1/J) * F_tilde * S_inter * F_tilde.T
    #     sigma_convective = (1.0 / J) * dot(dot(F_tilde, S_inter), F_tilde.T)
        
    #     # Total Stress
    #     return (term_stiffness + term_geometric) * I + sigma_convective
    
    def calculate_stress(self, u, v, J_dummy, T, T0, kinematic):
            """
            Calculate Cauchy stress via automatic differentiation of the Free Energy.
            
            Sigma = (1/J) * (dPsi/dF) * F^T
            """
            # 1. Redefine Kinematics from u to ensure a clean derivative chain w.r.t F
            # We define F as the primary variable for differentiation
            F = variable(kinematic.deformation_gradient_3d(u))
            J = det(F) 
            C = dot(F.T, F)
            I = Identity(3)
            
            # 2. Intermediate Configuration
            F0 = self.F0_func(J)
            inv_F0 = inv(F0)
            
            # 3. Intermediate Strain Measure
            # C_tilde = F0^-T * C * F0^-1
            C_tilde = dot(dot(inv_F0.T, C), inv_F0)
            E_tilde = 0.5 * (C_tilde - I)

            # B. Structural/Anisotropic Energy
            # W_struc = 1/2 * E_tilde : L_tilde(J) : E_tilde
            if hasattr(self, "R"):
                # Rotate stiffness to global frame if orientations are defined
                L_tilde_voigt = rotate_stifness(self.stiffness_func(J), self.R)
            else:
                L_tilde_voigt = self.stiffness_func(J)
                
            E_tilde_voigt = kinematic.tensor_3d_to_voigt(E_tilde)
            
            # Calculation of the scalar energy density
            W_struc = 0.5 * inner(E_tilde_voigt, dot(as_tensor(L_tilde_voigt), E_tilde_voigt))
            
            # 5. Stress Derivation
            # First Piola-Kirchhoff Stress: P = dW/dF
            P = diff(W_struc, F)
            
            # Cauchy Stress: Sigma = (1/J) * P * F^T
            sigma = (1.0 / J) * dot(P, F.T)
            return sigma