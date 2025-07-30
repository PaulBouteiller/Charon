# Copyright 2025 CEA
"""
Modèle de plasticité J2 avec JAX

@author: bouteillerp
"""

from .base_plastic import Plastic
from dolfinx.fem import functionspace, Function
from ufl import dot, dev


class JAXJ2Plasticity(Plastic):
    """J2 plasticity model with improved numerical performance.
    
    This class implements J2 plasticity with algorithmic tangent,
    providing improved convergence properties in nonlinear simulations.
    
    Attributes
    ----------
    V_Be : FunctionSpace Function space for elastic left Cauchy-Green tensor
    Be_Bar_trial_func : Function Trial elastic left Cauchy-Green tensor
    Be_Bar_old : Function Previous elastic left Cauchy-Green tensor
    len_plas : int Length of plastic variable array
    Be_bar_old_3D : Expression 3D representation of previous elastic left Cauchy-Green tensor
    u_old : Function Previous displacement field
    Be_Bar_trial : Expression Expression for trial elastic left Cauchy-Green tensor
    V_p : FunctionSpace Function space for cumulated plasticity
    p : Function Cumulated plasticity
    """
    def _set_function(self, element, quadrature):
        """Initialize functions for J2 plasticity.
        
        Parameters
        ----------
        quadrature : QuadratureHandler Handler for quadrature integration
        """
        self.V_Be_bar = functionspace(self.mesh, element)
        self.Be_bar = Function(self.V_Be_bar)
        self.Be_bar_3D = self.kin.mandel_to_tridim(self.Be_bar)
        self.len_plas = len(self.Be_bar)
        self.Be_bar.x.array[::self.len_plas] = 1
        self.Be_bar.x.array[1::self.len_plas] = 1
        self.Be_bar.x.array[2::self.len_plas] = 1
        self.V_p = quadrature.quadrature_space(["Scalar"])
        self.p = Function(self.V_p, name = "Cumulated_plasticity")
        
    def Be_bar_trial(self, u, u_old):
        """Define the elastic left Cauchy-Green tensor predictor.
        
        Returns
        -------
        Expression Trial elastic left Cauchy-Green tensor
        """
        F_rel = self.kin.relative_gradient_3D(u, u_old)
        J_rel = self.kin.reduced_det(F_rel)
        F_rel_bar = J_rel**(-1./3) * F_rel
        return dot(dot(F_rel_bar, self.Be_bar_3D), F_rel_bar.T)
        
    def compute_deviatoric_stress(self, u, v, J, T, T0, material, deviator):
        """Finite strain: direct from Be_trial"""
        return material.devia.mu * dev(self.Be_bar_trial(u, self.u_old))