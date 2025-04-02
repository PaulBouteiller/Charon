#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 11:37:27 2025

@author: bouteillerp
"""
"""General anisotropic hyperelastic deviatoric stress model."""

from ufl import as_tensor, as_matrix, dev, inv, symetrized_tensor_product, inner, dot, Identity
from .base_deviator import BaseDeviator

class AnisotropicDeviator(BaseDeviator):
    """General anisotropic hyperelastic deviatoric stress model.
    
    This model implements a general anisotropic formulation suitable for
    materials with complex directional properties.
    
    Attributes
    ----------
    C : array
        Stiffness tensor in Voigt notation
    f_func_coeffs : list or None
        Coefficients for optional stiffness modulation functions
    """
    
    def required_parameters(self):
        """Return the list of required parameters.
        
        Returns
        -------
        list
            List with "C" (required) and "f_func" (optional)
        """
        return ["C"]
    
    def __init__(self, params):
        """Initialize the general anisotropic deviatoric model.
        
        Parameters
        ----------
        params : dict
            Dictionary containing:
            C : array
                Stiffness tensor in Voigt notation
            f_func : list, optional
                Coefficients for stiffness modulation functions
        """
        super().__init__(params)
        
        # Store parameters
        self.C = params["C"]
        self.f_func_coeffs = params.get("f_func", None)
    
    def calculate_stress(self, u, v, J, T, T0, kinematic):
        """Calculate the deviatoric stress tensor for anisotropic hyperelasticity.
        
        Parameters
        ----------
        u : Function
            Displacement field
        v : Function
            Velocity field (unused)
        J : Function
            Jacobian of the deformation
        T : Function
            Current temperature (unused)
        T0 : Function
            Initial temperature (unused)
        kinematic : Kinematic
            Kinematic handler object
            
        Returns
        -------
        Function
            Deviatoric stress tensor
        """
        RigLin = self.C
        
        # Initial term
        M_0 = as_tensor([[RigLin[0,0] + RigLin[0,1] + RigLin[0,2], 0, 0],
                          [0, RigLin[1,0] + RigLin[1,1] + RigLin[1,2], 0],
                          [0, 0, RigLin[2,0] + RigLin[2,1] + RigLin[2,2]]])
        pi_0 = 1./3 * (J - 1) * M_0
        term_1 = J**(-5./3) * dev(kinematic.push_forward(pi_0, u))

        # Second term
        C = kinematic.C_3D(u)
        C_bar = J**(-2./3) * C
        inv_C = inv(C)
        
        GLD_bar = 1./2 * (C_bar - Identity(3))
        GLDBar_V = kinematic.tridim_to_Voigt(GLD_bar)
        D = 1./3 * symetrized_tensor_product(M_0, inv_C)
        DE = kinematic.Voigt_to_tridim(dot(D, GLDBar_V))
        term_2 = kinematic.push_forward(DE, u)
        
        # Optional modulation of stiffness with J
        def polynomial_expand(x, point, coeffs):
            return coeffs[0] + sum(coeff * (x - point)**(i+1) for i, coeff in enumerate(coeffs[1:]))
        
        def polynomial_derivative(x, point, coeffs):
            return coeffs[1] * (x - point) + sum(coeff * (i+2) * (x - point)**(i+1) for i, coeff in enumerate(coeffs[2:]))
        
        def rig_lin_correction(C, Rig_func_coeffs, J, derivative_degree):
            size = len(C)
            C_list = [[C[i][j] for i in range(size)] for j in range(size)]
            for i in range(size):
                for j in range(size):
                    if derivative_degree == 0:
                        C_list[i][j] *= polynomial_expand(J, 1, Rig_func_coeffs[i][j])
                    elif derivative_degree == 1:
                        C_list[i][j] *= polynomial_derivative(J, 1, Rig_func_coeffs[i][j])
            return C_list

        # Third term (and optional fourth term)
        if self.f_func_coeffs is not None:
            RigiLinBar = rig_lin_correction(RigLin, self.f_func_coeffs, J, 0)
            CE = kinematic.Voigt_to_tridim(dot(as_matrix(RigiLinBar), GLDBar_V))
            term_3 = J**(-5./3) * dev(kinematic.push_forward(CE, u))
            
            DerivRigiLinBar = rig_lin_correction(RigLin, self.f_func_coeffs, J, 1)
            EE = kinematic.Voigt_to_tridim(1./2 * inner(inv_C, GLD_bar) * dot(as_matrix(DerivRigiLinBar), GLDBar_V))
            term_4 = dev(kinematic.push_forward(EE, u))
            
            return term_1 + term_2 + term_3 + term_4
        else:
            CE = kinematic.Voigt_to_tridim(dot(as_matrix(RigLin), GLDBar_V))
            term_3 = J**(-5./3) * dev(kinematic.push_forward(CE, u))
            return term_1 + term_2 + term_3