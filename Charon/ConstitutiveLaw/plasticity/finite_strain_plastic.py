# Copyright 2025 CEA
"""
Modèle de plasticité en déformations finies

@author: bouteillerp
"""

from .base_plastic import Plastic
from ...utils.generic_functions import ppart
from dolfinx.fem import functionspace, Function, Expression
from ufl import dot, sqrt, tr, dev, inner


class FiniteStrainPlastic(Plastic):         
    """Finite strain plasticity model with multiplicative decomposition.

    This class implements finite strain plasticity based on the multiplicative
    decomposition of the deformation gradient and evolution of the elastic
    left Cauchy-Green tensor.
    
    Attributes
    ----------
    V_dev_BE : FunctionSpace Function space for deviatoric elastic left Cauchy-Green tensor
    dev_Be : Function Deviatoric elastic left Cauchy-Green tensor
    dev_Be_3D : Expression 3D representation of deviatoric elastic left Cauchy-Green tensor
    u_old : Function Previous displacement field
    F_rel : Expression Relative deformation gradient
    V_Ie : FunctionSpace Function space for volumetric elastic left Cauchy-Green tensor
    barI_e : Function Volumetric elastic left Cauchy-Green tensor
    barI_e_expr : Expression Expression for updated volumetric elastic left Cauchy-Green tensor
    dev_Be_expr : Expression Expression for updated deviatoric elastic left Cauchy-Green tensor
    """
    def _set_function(self, element, quadrature):
        """Initialize functions for finite strain plasticity.
        
        Parameters
        ----------
        quadrature : QuadratureHandler Handler for quadrature integration
        """
        self.V_dev_BE = functionspace(self.mesh, element)
        self.dev_Be = Function(self.V_dev_BE)
        self.dev_Be_3D  = self.kin.mandel_to_tridim(self.dev_Be)
        self.u_old = Function(self.V, name = "old_displacement")
        self.F_rel = self.kin.relative_gradient_3D(self.u, self.u_old)
        self.V_Ie = quadrature.quadrature_space(["Scalar"])
        self.barI_e = Function(self.V_Ie, name = "Bar_I_elastique")
        self.barI_e.x.petsc_vec.set(1.)
        
    def set_expressions(self):
        """Define symbolic expressions for predictors.
        
        Creates expressions for the trial elastic left Cauchy-Green tensor
        and its deviatoric part, which will be used in the return mapping algorithm.
        """
        Be_trial = self.Be_trial()
        self.barI_e_expr = Expression(1./3 * tr(Be_trial), self.V_Ie.element.interpolation_points())       
        self.set_dev_Be_expression(dev(Be_trial))
        
    def Be_trial(self):
        """Define the elastic left Cauchy-Green tensor predictor.
        
        Returns
        -------
        Expression Trial elastic left Cauchy-Green tensor
        """
        Be_trial_part_1 = self.barI_e * dot(self.F_rel, self.F_rel.T)
        Be_trial_part_2 = dot(dot(self.F_rel, self.dev_Be_3D), self.F_rel.T)
        return Be_trial_part_1 + Be_trial_part_2
    
    def set_dev_Be_expression(self, dev_Be_trial):
        """Define the expression for the updated deviatoric elastic left Cauchy-Green tensor.

        Implements the return mapping algorithm for J2 plasticity in finite strain.
        
        Parameters
        ----------
        dev_Be_trial : Expression Trial deviatoric elastic left Cauchy-Green tensor
        """
        norme_dev_Be_trial = inner(dev_Be_trial, dev_Be_trial)**(1./2) 
        mu_bar = self.mu * self.barI_e
        F_charge = self.mu * norme_dev_Be_trial - sqrt(2/3) * self.sig_yield 
        Delta_gamma = F_charge / (2 * mu_bar)
        if self.hardening == "Isotropic":
            Delta_gamma *= 1 / (1 + self.H / (3 * mu_bar))
        eps = 1e-6
        dev_Be_expr_3D = (1 - (2 * self.barI_e * ppart(Delta_gamma)) / (norme_dev_Be_trial + eps)) * dev_Be_trial
        dev_Be_expr = self.kin.tridim_to_mandel(dev_Be_expr_3D)
        self.dev_Be_expr = Expression(dev_Be_expr, self.V_dev_BE.element.interpolation_points())