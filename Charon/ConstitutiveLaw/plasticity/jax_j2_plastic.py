# Copyright 2025 CEA
"""
Modèle de plasticité J2 avec JAX

@author: bouteillerp
"""

from .base_plastic import Plastic
from dolfinx.fem import functionspace, Function, Expression
from ufl import dot, det


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
        self.V_Be = functionspace(self.mesh, element)
        self.Be_Bar_trial_func = Function(self.V_Be)
        self.Be_Bar_old = Function(self.V_Be)
        len_plas = len(self.Be_Bar_old.x.array)
        self.len_plas = len_plas
        self.Be_Bar_old.x.array[::len_plas] = 1
        self.Be_Bar_old.x.array[1::len_plas] = 1
        self.Be_Bar_old.x.array[2::len_plas] = 1
        
        self.Be_bar_old_3D = self.kin.mandel_to_tridim(self.Be_Bar_old)
        
        self.u_old = Function(self.V, name = "old_displacement")
        F_rel = self.kin.relative_gradient_3D(self.u, self.u_old)
        
        expr = det(F_rel)**(-2./3) * dot(dot(F_rel.T, self.Be_bar_old_3D), F_rel.T)
        self.Be_Bar_trial = Expression(self.kin.tridim_to_mandel(expr), self.V_Be.element.interpolation_points())
        self.V_p = quadrature.quadrature_space(["Scalar"])
        self.p = Function(self.V_p, name = "Cumulated_plasticity")