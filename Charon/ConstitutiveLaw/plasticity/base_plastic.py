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
Classe de base pour tous les modèles de plasticité

@author: bouteillerp
"""

from dolfinx.fem import Function

class Plastic:
    """Base class for all plasticity models.

    This class provides common functionality for plasticity models,
    including space initialization, parameter handling, and utility methods.
    
    Attributes
    ----------
    u : Function Displacement field
    u_old : Function Previous displacement field
    kin : Kinematic Kinematic handler for tensor operations
    mu : float Shear modulus
    mesh : Mesh Computational mesh
    mesh_dim : int Topological dimension of mesh
    plastic_model : str Type of plasticity model
    hardening : str Type of hardening ("Isotropic", "LinearKinematic", "NonLinear")
    sig_yield : float Yield stress
    H : float, optional Hardening modulus (for Isotropic/LinearKinematic)
    yield_stress : callable, optional Hardening function (for NonLinear)
    """
    def __init__(self, u, mu, name, kinematic, quadrature, plasticity_dictionnary):
        """Initialize the plasticity model.
        
        Parameters
        ----------
        u : Function Displacement field
        mu : float Shear modulus
        name : str Model name
        kinematic : Kinematic Kinematic handler for tensor operations
        quadrature : QuadratureHandler Handler for quadrature integration
        plasticity_dictionnary : str Type of plasticity model
        """
        self.u = u
        self.V = self.u.function_space
        self.u_old = Function(self.V, name = "old_displacement")
        self.kin = kinematic
        self.mu = mu
        self.mesh = self.u.function_space.mesh
        self.mesh_dim =self.mesh.topology.dim
        self._set_plastic(plasticity_dictionnary)
        element = self._plastic_element(quadrature, self.mesh.topology.dim)
        self._set_function(element, quadrature)
        
    def _plastic_element(self, quadrature, mesh_dim):
        """Create appropriate element for plastic variables.

        Parameters
        ----------
        quadrature : QuadratureHandler Handler for quadrature integration
            
        Returns
        -------
        Element Appropriate element for plastic variables
        """
        if mesh_dim == 1:
            return quadrature.quad_element(["Vector", 3])
        elif mesh_dim == 2:
            return quadrature.quad_element(["Vector", 4])
        elif mesh_dim == 3:
            return quadrature.quad_element(["Vector", 6])
        
    def _set_plastic(self, plasticity_dictionnary):
        """Set plastic parameters from dictionary.
        
        Parameters
        ----------
        plasticity_dictionnary : dict Dictionary containing plasticity parameters
        """
        self.plastic_model = plasticity_dictionnary["model"]
        self.hardening = plasticity_dictionnary.get("Hardening", "Isotropic")
        self.sig_yield = plasticity_dictionnary["sigY"]
        if self.hardening in ["Isotropic", "LinearKinematic"]:
            self.H = plasticity_dictionnary["Hardening_modulus"]
        elif self.hardening in ["NonLinear"]:
            self.yield_stress = plasticity_dictionnary["Hardening_func"]
            
    def _set_function(self, element, quadrature):
        """To be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _set_function")
        
    def compute_deviatoric_stress(self, u, v, J, T, T0, material, deviator):
        """To be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement compute_deviatoric_stress")