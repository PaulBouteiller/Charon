# Copyright 2025 CEA
"""
Modèle de plasticité HPP (petites déformations)

@author: bouteillerp
"""

from .base_plastic import Plastic
from ...utils.generic_functions import ppart
from dolfinx.fem import functionspace, Function, Expression
from ufl import dot, sqrt


class HPPPlastic(Plastic):
    """Small strain plasticity model.
    
    This class implements J2 plasticity with small strain assumption,
    supporting both isotropic and kinematic hardening.
    
    Attributes
    ----------
    Vepsp : FunctionSpace Function space for plastic strain
    eps_p : Function Plastic strain tensor
    eps_P_3D : Expression 3D representation of plastic strain tensor
    delta_eps_p : Function Increment of plastic strain
    """
    def _set_function(self, element, quadrature):
        """Initialize functions for small strain plasticity.

        Parameters
        ----------
        quadrature : QuadratureHandler
            Handler for quadrature integration
        """
        self.Vepsp = functionspace(self.mesh, element)
        self.eps_p = Function(self.Vepsp, name = "Plastic_strain")
        self.eps_P_3D = self.kin.mandel_to_tridim(self.eps_p)
        self.Delta_eps_p = Function(self.Vepsp)
        if self.hardening == "Isotropic":
            self.Vp = quadrature.quadrature_space(["Scalar"])
            self.p = Function(self.Vp, name = "Cumulative_plastic_strain")
            self.Delta_p = Function(self.Vp, name = "Plastic_strain_increment")

    def plastic_correction(self, mu):
        """Calculate plastic stress correction.
        
        Computes the plastic contribution to the stress tensor,
        which is purely deviatoric.
        
        Parameters
        ----------
        mu : float Shear modulus
            
        Returns
        -------
        Expression Plastic stress tensor
        """
        return 2 * mu * self.eps_P_3D
    
    def plastic_driving_force(self, s_3D):
        """Calculate plastic driving force and plastic strain increment.
        
        Implements return mapping algorithm for J2 plasticity with
        different hardening options.
        
        Parameters
        ----------
        s_3D : Expression 3D deviatoric stress tensor
        """
        eps = 1e-10
        if self.hardening == "LinearKinematic":
            self.A = self.kin.tridim_to_mandel(s_3D - self.H * self.eps_P_3D)
            norm_A = sqrt(dot(self.A, self.A)) + eps
            Delta_eps_p = ppart(1 - (2/3.)**(1./2) * self.sig_yield / norm_A) / \
                                (2 * self.mu + self.H) *self.A

        elif self.hardening == "Isotropic":
            s_mandel = self.kin.tridim_to_mandel(s_3D)
            sig_VM = sqrt(3.0 / 2.0 * dot(s_mandel, s_mandel)) + eps
            f_elas = sig_VM - self.sig_yield - self.H * self.p
            Delta_p = ppart(f_elas) / (3. * self.mu + self.H)
            Delta_eps_p = 3. * Delta_p / (2. * sig_VM) * s_mandel
            self.Delta_p_expression = Expression(Delta_p, self.Vp.element.interpolation_points())
        self.Delta_eps_p_expression = Expression(Delta_eps_p, self.Vepsp.element.interpolation_points())