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
Modèle de plasticité J2 avec JAX

@author: bouteillerp
"""

from .base_plastic import Plastic
from dolfinx.fem import functionspace, Function
from ufl import dot, dev


class JAXJ2Plasticity(Plastic):
    """J2 plasticity model with improved numerical performance.
    
    Implements J2 plasticity with algorithmic tangent for
    improved convergence in nonlinear simulations.
    
    Attributes
    ----------
    V_Be_bar : FunctionSpace Function space for elastic left Cauchy-Green tensor
    Be_bar : Function Elastic left Cauchy-Green tensor
    Be_bar_3D : Expression 3D representation of elastic left Cauchy-Green tensor
    len_plas : int Length of plastic variable array
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
        self.Be_bar_3D = self.kin.mandel_compact_to_tensor_3d(self.Be_bar)
        self.len_plas = len(self.Be_bar)
        self.Be_bar.x.array[::self.len_plas] = 1
        self.Be_bar.x.array[1::self.len_plas] = 1
        self.Be_bar.x.array[2::self.len_plas] = 1
        self.V_p = quadrature.quadrature_space(["Scalar"])
        self.p = Function(self.V_p, name = "Cumulated_plasticity")
        
    def Be_bar_trial(self, u, u_old):
        """Compute elastic left Cauchy-Green tensor predictor.
        
        Parameters
        ----------
        u : Function Current displacement field
        u_old : Function Previous displacement field
            
        Returns
        -------
        Expression
            Trial elastic left Cauchy-Green tensor
        """
        F_rel = self.kin.relative_deformation_gradient_3d(u, u_old)
        J_rel = self.kin.reduced_det(F_rel)
        F_rel_bar = J_rel**(-1./3) * F_rel
        return dot(dot(F_rel_bar, self.Be_bar_3D), F_rel_bar.T)
        
    def compute_deviatoric_stress(self, u, v, J, T, T0, material, deviator):
        """Finite strain: direct from Be_trial"""
        return material.devia.mu * dev(self.Be_bar_trial(u, self.u_old))