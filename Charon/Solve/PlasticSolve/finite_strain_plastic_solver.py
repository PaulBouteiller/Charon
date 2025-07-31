"""
Finite Strain Plasticity Solver
===============================

Implements return mapping algorithm for finite strain J2 plasticity
using multiplicative decomposition of the deformation gradient.

@author: bouteillerp
"""

from ...utils.petsc_operations import petsc_assign
from ufl import sqrt, tr, dev, inner
from dolfinx.fem import Expression
from ...utils.generic_functions import ppart

class FiniteStrainPlasticSolver:
    """Solver for finite strain plasticity with multiplicative decomposition.
    
    Implements return mapping algorithm for J2 plasticity in finite strain
    using the multiplicative decomposition F = Fe * Fp.
    
    Attributes
    ----------
    plastic : FiniteStrainPlastic Finite strain plasticity model instance
    u : Function Current displacement field
    barI_e_expr : Expression Expression for volumetric part of elastic left Cauchy-Green tensor
    dev_Be_expr : Expression Expression for deviatoric part of elastic left Cauchy-Green tensor
    """
    
    def __init__(self, problem, plastic, u):
        """Initialize the finite strain plasticity solver.
        
        Parameters
        ----------
        problem : Problem Problem instance (unused but kept for interface consistency)
        plastic : FiniteStrainPlastic Finite strain plasticity model instance
        u : Function Current displacement field
        """
        self.plastic = plastic
        self.u = u
        Be_trial = self.plastic.Be_trial(u, self.plastic.u_old)
        self.barI_e_expr = Expression(1./3 * tr(Be_trial), self.plastic.V_Ie.element.interpolation_points())       
        self.set_dev_Be_expression(dev(Be_trial))

    def set_dev_Be_expression(self, dev_Be_trial):
        """Compute updated deviatoric elastic left Cauchy-Green tensor.

        Implements return mapping algorithm for J2 plasticity in finite strain
        with radial return mapping on the deviatoric part.
        
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
        """Solve finite strain plasticity problem.
        
        Updates plastic variables according to multiplicative model:
        - Volumetric part of elastic left Cauchy-Green tensor
        - Deviatoric part of elastic left Cauchy-Green tensor  
        - Previous displacement field for next time step
        """
        self.plastic.barI_e.interpolate(self.barI_e_expr)
        self.plastic.dev_Be.interpolate(self.dev_Be_expr)#Si l'évolution est plastique, on vient actualiser dev_Be ce qui va en retour mettre à jour la contrainte pour le prochain pas
        petsc_assign(self.plastic.u_old, self.u)