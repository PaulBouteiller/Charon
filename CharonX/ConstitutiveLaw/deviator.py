"""
Created on Thu Jun 23 10:47:33 2022

@author: bouteillerp
"""
from ufl import (tr, sym, dev, Identity, dot, exp, outer, inner, sqrt, skew, 
                 inv, as_matrix, as_tensor)
from dolfinx.fem import Function, Expression
from ..utils.tensor_operations import symetrized_tensor_product

class Deviator:
    def __init__(self, kinematic, model, quadrature, is_hypo):
        self.kin = kinematic
        self.model = model
        if is_hypo:
            self.set_function_space(model, quadrature)
        
    def set_elastic_dev(self, u, v, J, T, T0, mat):
        """
        Renvoie la contrainte déviatorique

        Parameters
        ----------
        u : Function, champ de déplacement
        v : Function, champ des vitesses.
        J : Jacobien de la transformation
        mat : Objet de la classe material.
        T : champ de température actuelle
        T0 : champ de température initiale

        Returns
        -------
        s : deviateur de la contrainte de Cauchy
        """
        if mat.dev_type == None:
            s = 0 * Identity(3)
        elif mat.dev_type == "IsotropicHPP":
            s = self.HPP_deviatoric(u, J, T, T0, mat.devia.mu)
        elif mat.dev_type == "NewtonianFluid":
            s = self.fluid_deviatoric(v, J, T, T0, mat.devia.mu)
        elif mat.dev_type == "NeoHook":
            s = self.Neo_Hook_deviatoric(u, J, T, T0, mat.devia.mu)
        elif mat.dev_type == "MooneyRivlin":
            s = self.Mooney_Rivlin_deviatoric(u, J, T, T0, mat.devia)
        elif mat.dev_type == "NeoHook_Transverse":
            s = self.NeoHook_Transverse_deviatoric(u, J, T, T0, mat.devia)
        elif mat.dev_type == "Lu_Transverse":
            s = self.Lu_Transverse_deviatoric(u, J, T, T0, mat.devia)
        elif mat.dev_type == "Anisotropic":
            s = self.anisotropic_dev(u, J, T, T0, mat.devia)
        return s
    
    def HPP_deviatoric(self, u, J, T, T0, mu):
        """
        Renvoie la contrainte déviatorique du modèle élastique linéaire de Hooke
        """
        return 2 * mu * dev(sym(self.kin.grad_3D(u)))
    
    def fluid_deviatoric(self, v, J, T, T0, mu):
        """
        Renvoie la contrainte déviatorique du modèle de fluide Newtonien
        """
        return 2 * mu * dev(sym(self.kin.grad_3D(v)))
    
    def Neo_Hook_deviatoric(self, u, J, T, T0, mu):
        """
        Renvoie la contrainte déviatorique du modèle hyper-élastique Néo-Hookéen
        """
        return mu / J**(5./3) * dev(self.kin.B_3D(u))
    
    def NeoHook_Transverse_deviatoric(self, u, J, T, T0, mat_dev):
        """
        Renvoie la contrainte déviatorique du modèle hyper-élastique isotrope transverse Néo-Hookéen
        """
        B = self.kin.B_3D(u)
        nt = self.kin.actual_anisotropic_direction(u)
        Nt = outer(nt, nt)
        I1B = inner(B, Nt)
        BBarI = J**(-2./3) * tr(B)
        symBNt = dot(B, Nt) + dot(Nt, B)
        s_transverse = 2 * mat_dev.mu_T / J**(5./3) * (symBNt - I1B * (Nt + 1./3 * Identity(3)))
        #A supprimer car incohérent ?
        s_transverse *= (I1B - 1) + (BBarI - 3)
        return self.Neo_Hook_deviatoric(u, J, T, T0, mat_dev.mu) + s_transverse
    
    def Mooney_Rivlin_deviatoric(self, u, J, T, T0, mat_dev):
        """
        Renvoie la contrainte déviatorique du modèle hyper-élastique Mooney-Rivlin
        """
        B = self.kin.B_3D(u)
        return mat_dev.mu / J**(5./3) * dev(B) - mat_dev.mu_quad / J**(7./3) * dev(dot(B,B) - tr(B) * B)
    
    def Lu_Transverse_deviatoric(self, u, J, T, T0, mat_dev):
        """
        Renvoie le déviateur du modèle isotrope transverse de Lu
        """
        B = self.kin.B_3D(u)
        C = self.kin.C_3D(u)
        N0 = outer(self.kin.n0, self.kin.n0)
        nt = self.kin.actual_anisotropic_direction(u)
        Nt = outer(nt, nt)
        I1C = inner(C, N0)
        I1B = inner(B, Nt)
        lmbda = sqrt(I1C)
        lambdabar = J**(-1/3) * lmbda
        symBNt = dot(B, Nt) + dot(Nt, B)
        s2 = 2 * mat_dev.k2 * mat_dev.c * lambdabar * (lambdabar - 1) / J * exp(mat_dev.c*(lambdabar - 1)**2) * dev(Nt)
        s3 = mat_dev.k3 / (I1C * J) * (symBNt - 2 * I1B * Nt)
        s4 = mat_dev.k4 * lmbda / (2 * J**2) * (2 * B - 2 * symBNt - (tr(B) - I1B)*(Identity(3) - Nt))
        return s2 + s3 + s4
    
    def set_function_space(self, model, quadrature):
        if model == "CartesianUD":
            self.V_s = quadrature.quadrature_space(["Scalar"])
        elif model == ["CylindricalUD", "SphericalUD"]:
            self.V_s = quadrature.quadrature_space(["Vector", 2])
        elif model == "PlaneStrain":
            self.V_s = quadrature.quadrature_space(["Vector", 3])
        elif model == "Axisymetric":
            self.V_s = quadrature.quadrature_space(["Vector", 4])
        elif model =="Tridimensionnal":
            self.V_s = quadrature.quadrature_space(["Tensor", 3, 3])
            
    def set_deviator(self, u, v, J, mu):
        self.s = Function(self.V_s, name = "Deviator")
        s_3D = self.kin.reduit_to_3D(self.s, sym = True)
        L = self.kin.reduit_to_3D(self.kin.Eulerian_gradient(v, u))
        D = sym(L)
        # dev_D = dev(D)
        
        B = self.kin.B_3D(u)

        s_Jaumann_3D = mu/J**(5./3) * (dot(B, D) + dot(D, B) 
                                      - 2./3 * inner(B,D) * Identity(3)
                                      -5./3 * tr(D) * dev(B))
        # s_Jaumann_3D = 2 * mu * dev_D
        # s_Jaumann_3D = mu * (dot(B, D) + dot(D, B) - 2./3 * inner(B,D) * Identity(3))
        s_Jaumann = self.kin.tridim_to_reduit(s_Jaumann_3D, sym = True)
        if self.model in ["CartesianUD", "CylindricalUD", "SphericalUD"]:
            self.dot_s = Expression(s_Jaumann, self.V_s.element.interpolation_points())
        else:
            Omega = skew(L)
            Jaumann_corr = self.kin.tridim_to_reduit(dot(Omega, s_3D) - dot(s_3D, Omega), sym = True)
            self.dot_s = Expression(s_Jaumann + Jaumann_corr, self.V_s.element.interpolation_points())
        return s_3D
    

    def anisotropic_dev(self, u, J, T, T0, mat_devia):     
        
        RigLin = mat_devia.C
        M_0 = as_tensor([[RigLin[0,0] + RigLin[0,1] + RigLin[0,2], 0, 0],
                          [0, RigLin[1,0] + RigLin[1,1] + RigLin[1,2], 0],
                          [0, 0, RigLin[2,0] + RigLin[2,1] + RigLin[2,2]]])
        pi_0 = 1./3 * (J - 1) * M_0
        term_1 = J**(-5./3) * dev(self.kin.push_forward(pi_0, u))

        C = self.kin.C_3D(u)
        C_bar = J**(-2./3) * C
        inv_C = inv(C)
        
        GLD_bar = 1./2 * (C_bar - Identity(3))
        GLDBar_V = self.kin.tridim_to_Voigt(GLD_bar)
        D = 1./3 * symetrized_tensor_product(M_0, inv_C)
        DE = self.kin.Voigt_to_tridim(dot(D, GLDBar_V))
        term_2 = self.kin.push_forward(DE, u)


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

        Rig_func_coeffs = mat_devia.f_func_coeffs
        if Rig_func_coeffs !=None:
            RigiLinBar = rig_lin_correction(RigLin, Rig_func_coeffs, J, 0)
        else:
            RigiLinBar = RigLin
        CE = self.kin.Voigt_to_tridim(dot(as_matrix(RigiLinBar), GLDBar_V))
        term_3 = J**(-5./3) * dev(self.kin.push_forward(CE, u))
        if Rig_func_coeffs !=None:
            DerivRigiLinBar = rig_lin_correction(RigLin, Rig_func_coeffs, J, 1)
            EE = self.kin.Voigt_to_tridim(1./2 * inner(inv_C, GLD_bar) * dot(as_matrix(DerivRigiLinBar), GLDBar_V))
            term_4 = dev(self.kin.push_forward(EE, u))
            # return term_1 + term_2 + term_3
            return term_1 + term_2 + term_3 + term_4
        else:
            return term_1 + term_2 + term_3