# Copyright 2025 CEA
"""
Modèle de plasticité HPP (petites déformations)

@author: bouteillerp
"""

from .base_plastic import Plastic

from dolfinx.fem import functionspace, Function



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
        if self.hardening == "Isotropic":
            self.Vp = quadrature.quadrature_space(["Scalar"])
            self.p = Function(self.Vp, name = "Cumulative_plastic_strain")

            
    def compute_deviatoric_stress(self, u, v, J, T, T0, material, deviator):
        """HPP: elastic + plastic correction"""
        if material.dev_type == "Hypoelastic":
            deviatoric = deviator.set_hypoelastic_deviator(u, v, J, material)
        else:
            deviatoric = deviator.set_elastic_dev(u, v, J, T, T0, material)
        return deviatoric - 2 * material.devia.mu * self.eps_P_3D