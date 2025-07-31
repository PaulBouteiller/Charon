# Copyright 2025 CEA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Modèle de plasticité HPP (petites déformations)

@author: bouteillerp
"""

from .base_plastic import Plastic

from dolfinx.fem import functionspace, Function



class HPPPlastic(Plastic):
    """Small strain J2 plasticity model.
    
    Implements J2 plasticity with small strain assumption,
    supporting isotropic and kinematic hardening.
    
    Attributes
    ----------
    Vepsp : FunctionSpace Function space for plastic strain
    eps_p : Function Plastic strain tensor
    eps_P_3D : Expression 3D representation of plastic strain tensor
    Vp : FunctionSpace, optional
        Function space for cumulative plastic strain (isotropic hardening)
    p : Function, optional
        Cumulative plastic strain (isotropic hardening)
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