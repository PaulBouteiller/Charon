# Copyright 2025 CEA
"""
Modèle de plasticité en déformations finies

@author: bouteillerp
"""

from .base_plastic import Plastic

from dolfinx.fem import functionspace, Function
from ufl import dot, dev



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
        self.dev_Be_3D = self.kin.mandel_to_tridim(self.dev_Be)
        self.V_Ie = quadrature.quadrature_space(["Scalar"])
        self.barI_e = Function(self.V_Ie, name = "Bar_I_elastique")
        self.barI_e.x.petsc_vec.set(1.)

    def Be_trial(self, u, u_old):
        """Define the elastic left Cauchy-Green tensor predictor.
        
        Returns
        -------
        Expression Trial elastic left Cauchy-Green tensor
        """
        F_rel = self.kin.relative_gradient_3D(u, u_old)
        Be_trial_part_1 = self.barI_e * dot(F_rel, F_rel.T)
        Be_trial_part_2 = dot(dot(F_rel, self.dev_Be_3D), F_rel.T)
        return Be_trial_part_1 + Be_trial_part_2
        
    def compute_deviatoric_stress(self, u, v, J, T, T0, material, deviator):
        """Finite strain: direct from Be_trial"""
        return material.devia.mu / J**(5./3) * dev(self.Be_trial(u, self.u_old))