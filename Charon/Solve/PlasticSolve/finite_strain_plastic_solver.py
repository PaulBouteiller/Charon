# Copyright 2025 CEA
"""
Solver pour la plasticité en déformations finies

@author: bouteillerp
"""

from ...utils.petsc_operations import petsc_assign
from ufl import sqrt, tr, dev, inner
from dolfinx.fem import Expression
from ...utils.generic_functions import ppart

class FiniteStrainPlasticSolver:
    """Solveur pour la plasticité en déformations finies"""
    
    def __init__(self, problem, plastic, u):
        """
        Initialise le solveur en déformations finies
        
        Parameters
        ----------
        plastic : FiniteStrainPlastic
            Instance de la classe FiniteStrainPlastic
        u : Function
            Champ de déplacement
        """
        self.plastic = plastic
        self.u = u
        Be_trial = self.plastic.Be_trial(u, self.plastic.u_old)
        self.barI_e_expr = Expression(1./3 * tr(Be_trial), self.plastic.V_Ie.element.interpolation_points())       
        self.set_dev_Be_expression(dev(Be_trial))

    def set_dev_Be_expression(self, dev_Be_trial):
        """Define the expression for the updated deviatoric elastic left Cauchy-Green tensor.

        Implements the return mapping algorithm for J2 plasticity in finite strain.
        
        Parameters
        ----------
        dev_Be_trial : Expression Trial deviatoric elastic left Cauchy-Green tensor
        """
        norme_dev_Be_trial = inner(dev_Be_trial, dev_Be_trial)**(1./2) 
        mu_bar = self.plastic.mu * self.plastic.barI_e
        F_charge = self.plastic.mu * norme_dev_Be_trial - sqrt(2/3) * self.plastic.sig_yield 
        Delta_gamma = F_charge / (2 * mu_bar)
        if self.plastic.hardening == "Isotropic":
            Delta_gamma *= 1 / (1 + self.plastic.H / (3 * mu_bar))
        eps = 1e-6
        dev_Be_expr_3D = (1 - (2 * self.plastic.barI_e * ppart(Delta_gamma)) / (norme_dev_Be_trial + eps)) * dev_Be_trial
        dev_Be_expr = self.plastic.kin.tridim_to_mandel(dev_Be_expr_3D)
        self.dev_Be_expr = Expression(dev_Be_expr, self.plastic.V_dev_BE.element.interpolation_points())
        self.plastic.dev_Be.interpolate(self.dev_Be_expr)
        
    def solve(self):
        """
        Résout le problème de plasticité en déformations finies
        
        Met à jour les variables plastiques selon le modèle multiplicatif :
        - Partie volumétrique du tenseur élastique de Cauchy-Green gauche
        - Partie déviatorique du tenseur élastique de Cauchy-Green gauche
        - Champ de déplacement ancien pour le pas suivant
        """
        self.plastic.barI_e.interpolate(self.barI_e_expr)
        self.plastic.dev_Be.interpolate(self.dev_Be_expr)#Si l'évolution est plastique on vient actualisé dev_Be ce qui va actualisé la contrainte pour le prochaine pas
        petsc_assign(self.plastic.u_old, self.u)