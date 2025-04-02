"""
Created on Wed Nov 23 16:33:35 2022

@author: bouteillerp
"""
from ..utils.generic_functions import ppart

class Thermal:
    """
    La classe Thermal sert à définir le comportement thermique du solide
    """
    
    def __init__(self, mat, multiphase, kinematic, T0, T, P):
        self.material = mat
        self.multiphase = multiphase
        self.T0 = T0
        self.T = T
        self.P = P
        
    def set_tangent_thermal_capacity(self):
        """
        Renvoie la capacité thermique volumique tangente totale, c'est à dire la variation 
        d'énergie interne volumique consécutive à une variation de température
        pour un des sous-constituants

        Parameters
        ----------
        mat : Object de la classe material.
        """
        if isinstance(self.material, list):
            n_mat = len(self.material)
            if all([self.material[0].C_mass == self.material[i].C_mass for i in range(n_mat)]):
                self.C_tan = self.partial_C_vol_tan(self.material[0])
            else:
                self.C_tan = sum(self.multiphase.c[i] * self.partial_C_vol_tan(self.material[i]) for i in range(n_mat))
        else:
            self.C_tan = self.partial_C_vol_tan(self.material)
    
    def partial_C_vol_tan(self, mat):
        """
        Renvoie la capacité thermique volumique tangente associée au matériau mat.
        Parameters
        ----------
        mat : Object de la classe material.
        """
        CTan = mat.rho_0 * mat.C_mass
        return CTan   
    
    def thermal_constitutive_law(self, therm_mat, grad_dT):
        """
        Défini la loi de comportement thermique totale, moyenne pondérée des lois
        de comportement thermique des sous-matériaux.

        Parameters
        ----------
        therm_mat : List d'objet de la classe Thermal material ou simple objet.
        grad_dT : Gradient du champ de température test.
        """
        if isinstance(therm_mat, list):
            return sum(self.multiphase.g[i] * self.partial_thermal_constitutive_law(therm_mat[i]) for i in range(len(therm_mat)))
        else:
            return self.partial_thermal_constitutive_law(therm_mat, grad_dT)
    
    def partial_thermal_constitutive_law(self, therm_mat, grad_dT):
        """
        Défini la loi de comportement thermique partielle

        Parameters
        ----------
        therm_mat : Objet.de la classe Thermal material
        grad_dT : Gradient du champ de température test.
        """
        if therm_mat.type == "LinearIsotropic":
            return self.LinearFourrier(therm_mat.lmbda, grad_dT)
        elif therm_mat.type == "NonLinearIsotropic":
            return self.NonLinearFourrier(therm_mat.lmbda, therm_mat.a1, therm_mat.a2, grad_dT)
        
    def TATB_massique_capacity_correction(self, a, b, T, C_mass):
        assert min(T.x.array)>10
        C_mass *= 1 / pow(1 + pow(T, a), b)
        
    def LinearFourrier(self, lmbda, grad_dT):
        """
        Loi de comportement thermique linéaire isotrope

        Parameters
        ----------
        lmbda : Float, coefficient de diffusion thermique
        grad_dT : Gradient du champ de température test.
        """
        return lmbda * grad_dT
    
    def NonLinearFourrier(self, lmbda, a1, a2, grad_dT):
        """
        Loi de comportement thermique linéaire isotrope

        Parameters
        ----------
        lmbda : Float, coefficient de diffusion thermique
        a1    : Float, coefficient dépendance en T
        a2    : Float, coefficient dépendance en P
        grad_dT : Gradient du champ de température test.
        """
        return (lmbda + a1/self.T + a2 * ppart(self.P)) * grad_dT