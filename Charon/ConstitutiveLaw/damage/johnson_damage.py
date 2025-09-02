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
Johnson Porosity-Based Damage Models
====================================

This module implements the Johnson family of damage models based on porosity evolution,
tracking void growth in ductile materials. It includes static, dynamic, and inertial
variants of the Johnson model.

Classes:
--------
JohnsonDamage : Base class for porosity-based Johnson damage models
StaticJohnson : Static porosity-based damage model
DynamicJohnson : Rate-dependent porosity evolution model
InertialJohnson : Inertial effects in damage evolution
"""

from ufl import conditional, sqrt, Constant
from dolfinx.fem import Function, Expression
from petsc4py.PETSc import ScalarType

from .base_damage import BaseDamage
from ...utils.parameters.default import default_porosity_parameters


class JohnsonDamage(BaseDamage):
    """Base class for porosity-based Johnson damage models.
    
    Implements core functionality for damage models based on
    porosity evolution, tracking void growth in ductile materials.
    
    Attributes
    ----------
    initial_porosity : float Initial porosity value
    V_d : FunctionSpace Function space for porosity field
    inf_d : Function Minimum porosity field
    max_d : Function Maximum porosity field (upper bound)
    d : Function Current porosity field
    g_d : Expression Degradation function (1 - d)
    interp_points : array Interpolation points for the function space
    p_mot : Expression Driving force expression
    p_func : Function Pressure function
    """
    
    def __init__(self, mesh, quadrature, dictionnaire, u=None, J=None, pressure=None, material=None, kinematic=None):


        """Initialize the Johnson porosity-based damage model.
        
        Parameters
        ----------
        mesh : Mesh Computational mesh
        quadrature : QuadratureHandler Handler for quadrature integration
        dictionnaire : dict Parameters for the Johnson model
        """
        super().__init__(mesh, quadrature, dictionnaire, u, J, pressure, material, kinematic)
        self.initial_porosity = default_porosity_parameters()["initial_porosity"]
        # Initialize function spaces and functions
        self.V_d = quadrature.quadrature_space(["Scalar"])
        self.inf_d = Function(self.V_d, name="Minimum porosity")
        self.max_d = Function(self.V_d, name="Maximum porosity") 
        self.d = Function(self.V_d, name="Porosity")
        self.g_d = (1 - self.d)
        self.interp_points = self.V_d.element.interpolation_points()
        
    def _initialize_johnson(self, mesh, dictionnaire):
        """Initialize the Johnson model fields.

        Parameters
        ----------
        mesh : Mesh Computational mesh
        dictionnaire : dict Additional parameters including optional initial porosity (f0)
        """
        f0 = dictionnaire.get("f0", Expression(Constant(mesh, self.initial_porosity), self.interp_points))
        self.inf_d.interpolate(f0)
        self.max_d.x.petsc_vec.set(1 - self.residual_stiffness)
        self.d.interpolate(f0)

    def _initialize_driving_force(self, u, J, pressure, material, kinematic):
        """Initialize driving force for Johnson models."""
        self.p_mot = Expression(pressure, self.interp_points)
        self.p_func = Function(self.V_d)
        
    def set_unbreakable_zone(self, condition):
        """Define zones where damage is prevented.

        Parameters
        ----------
        condition : Expression Boolean condition defining unbreakable zones
        """
        ufl_condition = conditional(condition, self.initial_porosity, 1 - self.residual_stiffness)
        expr = Expression(ufl_condition, self.interp_points)
        self.max_d.interpolate(expr)


class StaticJohnson(JohnsonDamage):
    """Static Johnson porosity-based damage model.

    This class implements a quasi-static version of the Johnson damage model,
    with optional regularization for improved numerical stability.
    
    Attributes
    ----------
    regularization : bool Whether to use regularization
    eta : float Viscosity parameter
    sigma_0 : float Yield stress parameter
    lc : float, optional Regularization length (if regularization is enabled)
    """
    
    def set_damage(self, mesh, dictionnaire):
        """Initialize the static Johnson damage model.
        
        Parameters
        ----------
        mesh : Mesh Computational mesh
        dictionnaire : dict
            Additional parameters:
            - regularization (bool): Whether to use regularization
            - eta (float): Viscosity parameter
            - sigma_0 (float): Yield stress parameter
            - l0 (float): Regularization length (if regularization is enabled)
        """
        self.regularization = dictionnaire.get("regularization", False)
        self.eta = dictionnaire["eta"]
        self.sigma_0 = dictionnaire["sigma_0"]
        self._initialize_johnson(mesh, dictionnaire)
        
        if self.regularization:
            from dolfinx.fem import functionspace
            self.V_d_regul = functionspace(mesh, ('CG', 2))
            self.lc = dictionnaire["l0"]


class DynamicJohnson(JohnsonDamage):
    """Dynamic Johnson porosity-based damage model.

    This class extends the Johnson model with rate effects, accounting for
    viscous behavior in damage evolution.
    
    Attributes
    ----------
    eta : float Viscosity parameter
    sigma_0 : float Yield stress parameter
    initial_pore_distance : float Initial distance between pores
    tau : float Characteristic viscous time
    v_0 : float Characteristic velocity
    l_dyn : float Dynamic characteristic length
    V_a : FunctionSpace Function space for pore length
    a_tilde : Function Normalized pore length
    dot_a_tilde : Function Rate of change of normalized pore length
    d_expr : Expression Expression relating pore length to damage
    """
    
    def set_damage(self, mesh, dictionnaire):
        """Initialize the dynamic Johnson damage model.
        
        Parameters
        ----------
        mesh : Mesh Computational mesh
        dictionnaire : dict
            Additional parameters:
            - eta (float): Viscosity parameter
            - sigma_0 (float): Yield stress parameter
            - b (float): Initial pore distance
            - material (Material): Material properties
        """
        self.eta = dictionnaire["eta"]
        self.sigma_0 = dictionnaire["sigma_0"]
        self.initial_pore_distance = dictionnaire["b"]
        self.tau = self.eta / self.sigma_0
        self.v_0 = sqrt(self.sigma_0 / dictionnaire["material"].rho_0)
        self.l_dyn = self.tau * self.v_0
        
        print("Le temps caractéristique visqueux vaut", self.tau)
        print("La longueur inertielle vaut", self.l_dyn)
        
        self._initialize_johnson(mesh, dictionnaire)
        self._set_dynamic_johnson_functions(mesh)
            
    def _set_dynamic_johnson_functions(self, mesh):
        """Initialize functions for dynamic Johnson model.
        
        Sets up the necessary functions to track pore evolution in the
        dynamic model.
        
        Parameters
        ----------
        mesh : Mesh Computational mesh
        """
        self.V_a = self.quad.quadrature_space(["Scalar"])
        a0 = self.initial_pore_distance * self.initial_porosity**(1./3)
        a_tilde_0 = a0 / self.l_dyn
        
        self.a_tilde = Function(self.V_a, name="Pore_length")
        self.dot_a_tilde = Function(self.V_a, name="Pore_length_velocity")
        self.a_tilde.x.petsc_vec.set(a_tilde_0)
        
        d_expr = self.a_tilde**3 / (self.a_tilde**3 - a_tilde_0**3 + (self.initial_pore_distance / self.l_dyn)**3)
        self.d_expr = Expression(d_expr, self.interp_points)


class InertialJohnson(JohnsonDamage):
    """Inertial Johnson porosity-based damage model.

    This class extends the Johnson model with inertial effects,
    accounting for dynamical behavior in pore evolution.
    
    Attributes
    ----------
    sigma_0 : float Yield stress parameter
    initial_pore_distance : float Initial distance between pores
    V_a : FunctionSpace Function space for pore length
    a : Function Pore length
    dot_a : Function Rate of change of pore length
    d_expr : Expression Expression relating pore length to damage
    """
    
    def set_damage(self, mesh, dictionnaire):
        """Initialize the inertial Johnson damage model.

        Parameters
        ----------
        mesh : Mesh Computational mesh
        dictionnaire : dict
            Additional parameters:
            - material (Material): Material properties
            - sigma_0 (float): Yield stress parameter
            - b (float): Initial pore distance
        """
        self.sigma_0 = dictionnaire["sigma_0"]
        self.initial_pore_distance = dictionnaire["b"]
        self._initialize_johnson(mesh, dictionnaire)
        self._set_inertial_johnson_functions(mesh)
    
    def _set_inertial_johnson_functions(self, mesh):
        """Initialize functions for the inertial Johnson damage model.
        
        Parameters
        ----------
        mesh : Mesh Computational mesh
        """
        self.V_a = self.quad.quadrature_space(["Scalar"])
        a0 = self.initial_pore_distance * self.initial_porosity**(1./3)
        
        self.a = Function(self.V_a, name="Pore_length")
        self.dot_a = Function(self.V_a, name="Pore_length_velocity")
        self.a.x.petsc_vec.set(a0)
        self.dot_a.interpolate(Expression(Constant(mesh, ScalarType(0)), self.V_a.element.interpolation_points()))
        
        d_expr = self.a**3 / (self.a**3 - a0**3 + (self.initial_pore_distance)**3)
        self.d_expr = Expression(d_expr, self.interp_points)