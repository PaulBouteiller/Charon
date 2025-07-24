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
Damage Mechanics Module for Material Failure Modeling
=====================================================

This module implements various damage mechanics models for simulating material failure 
and degradation in mechanical simulations. It provides frameworks for both continuous 
and discrete approaches to damage, including phase field models for brittle fracture 
and porosity-based damage models.

The implemented models capture different aspects of material damage:
- Phase field damage for smooth crack propagation in brittle materials
- Johnson-based porosity models for ductile damage
- Dynamic and inertial extensions for rate-dependent damage phenomena

Each model provides the necessary functionality to:
- Initialize damage fields and parameters
- Calculate damage evolution terms
- Provide degradation functions for material stiffness
- Handle regularization and energy dissipation

Key Classes:
------------
Damage : Base class for all damage models
    Provides common functionality and parameters
    
Johnson : Base class for porosity-based damage models
    Implements porosity tracking and evolution
    
PhaseField : Implementation of phase field damage model
    Various formulations (AT1, AT2, Wu) for brittle fracture
    Energy-based approach with regularization
    
StaticJohnson : Static porosity-based damage
    Basic Johnson model without inertial terms in pore expansion
    
DynamicJohnson : Rate-dependent porosity evolution
    Includes viscous effects in damage evolution
    
InertialJohnson : Inertial effects in damage
    Accounts for inertial terms in pore expansion

References:
-----------
- Phase field models: Ambrosio-Tortorelli, Wu formulations
- Johnson models: Based on porosity evolution theories for ductile damage
"""
from ufl import (TestFunction, TrialFunction, dot, grad, sqrt, conditional)
from dolfinx.fem import (Function, functionspace, Constant, Expression)
from petsc4py.PETSc import ScalarType
from numpy import pi

from ..utils.default_parameters import default_damage_parameters, default_porosity_parameters

class Damage:
    """Base class for all damage models.
    
    This class provides common functionality and parameters used by all damage models,
    including residual stiffness handling and default damage values.
    
    Attributes
    ----------
    dam_parameters : dict Dictionary of default damage parameters
    residual_stiffness : float Residual stiffness factor (prevents complete loss of stiffness)
    default_damage : float Default initial damage value
    """
    def __init__(self):
        """Initialize the base damage model with default parameters."""
        self.dam_parameters = default_damage_parameters()
        self.residual_stifness = self.dam_parameters["residual_stiffness"]
        self.default_damage = ScalarType(self.dam_parameters["default_damage"])
        
class Johnson(Damage):
    """Base class for porosity-based Johnson damage models.
    
    This class implements the core functionality for damage models based on
    porosity evolution, tracking void growth in ductile materials.
    
    Attributes
    ----------
    quad : QuadratureHandler Handler for quadrature integration
    initial_porosity : float Initial porosity value
    V_d : FunctionSpace Function space for porosity field
    inf_d : Function Minimum porosity field
    max_d : Function Maximum porosity field (upper bound)
    d : Function Current porosity field
    g_d : Expression Degradation function (1 - d)
    """
    def __init__(self, mesh, quadrature, dictionnaire):
        """Initialize the Johnson porosity-based damage model.
        
        Parameters
        ----------
        mesh : Mesh Computational mesh
        quadrature : QuadratureHandler Handler for quadrature integration
        """
        Damage.__init__(self)
        self.quad = quadrature
        self.initial_porosity = default_porosity_parameters()["initial_porosity"]
        self.V_d = quadrature.quadrature_space(["Scalar"])
        self.inf_d = Function(self.V_d, name = "Minimum porosity")
        self.max_d = Function(self.V_d, name = "Maximum porosity") 
        self.d = Function(self.V_d, name = "Porosity")
        self.g_d = (1 - self.d)
        self.interp_points = self.V_d.element.interpolation_points()
        self.set_damage(mesh, dictionnaire)
        
    def _initialize_Johnson(self, mesh, dictionnaire):
        """Initialize the Johnson model fields.

        Parameters
        ----------
        mesh : Mesh Computational mesh
        dictionnaire : dict Additional parameters including optional initial porosity (f0)
        """
        f0 = dictionnaire.get("f0", Expression(Constant(mesh, self.initial_porosity), self.interp_points))
        self.inf_d.interpolate(f0)
        self.max_d.x.petsc_vec.set(1 - self.residual_stifness)
        self.d.interpolate(f0)
        
    def set_p_mot(self, undammaged_pressure):
        """Set the driving pressure for damage evolution.

        Parameters
        ----------
        undammaged_pressure : Expression Pressure field without damage effects
        """
        self.p_mot = Expression(undammaged_pressure, self.interp_points)
        self.p_func = Function(self.V_d)
        
    def set_unbreakable_zone(self, condition):
        """Define zones where damage is prevented.

        Parameters
        ----------
        condition : Expression Boolean condition defining unbreakable zones
        """
        ufl_condition = conditional(condition, self.initial_porosity, 1 - self.residual_stifness)
        expr = Expression(ufl_condition, self.interp_points)
        self.max_d.interpolate(expr)

class StaticJohnson(Johnson):
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
        mesh : Mesh
            Computational mesh
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
        self._initialize_Johnson(mesh, dictionnaire)
        if self.regularization:
            self.V_d_regul = functionspace(mesh, ('CG', 2))
            self.lc = dictionnaire["l0"]  

class DynamicJohnson(Johnson):
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
        self._initialize_Johnson(mesh, dictionnaire)
        self.set_dyn_johnson_function(mesh)
            
    def set_dyn_johnson_function(self, mesh):
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
        self.a_tilde = Function(self.V_a, name = "Pore_length")
        self.dot_a_tilde = Function(self.V_a, name = "Pore_length_velocity")
        self.a_tilde.x.petsc_vec.set(a_tilde_0)
        d_expr = self.a_tilde**3  / (self.a_tilde**3 - a_tilde_0**3 + (self.initial_pore_distance / self.l_dyn)**3)
        self.d_expr = Expression(d_expr, self.interp_points)

class InertialJohnson(Johnson):
    """Inertial Johnson porosity-based damage model.

    This class extends the Johnson model with inertial effects,
    accounting for dynamical behavior in pore evolution.
    
    Attributes
    ----------
    rho_0 : float Initial density
    sigma_0 : float Yield stress parameter
    initial_pore_distance : float Initial distance between pores
    V_a : FunctionSpace Function space for pore length
    a : Function Pore length
    dot_a : Function Rate of change of pore length
    d_expr : Expression Expression relating pore length to damage
    """
    def set_damage(self, mesh, dictionnaire):
        """
        Initialise les paramètres requis pour le modèle d'endommagement
        inséré en mot clé de damage_model.

        Parameters
        ----------
        mesh : Mesh, maillage du domaine.
        dictionnaire : Paramètres nécessaire au modèle d'endommagement choisi.
        """
        # self.rho_0 = dictionnaire["material"].rho_0
        self.sigma_0 = dictionnaire["sigma_0"]
        self.initial_pore_distance = dictionnaire["b"]
        self._initialize_Johnson(mesh, dictionnaire)
        self.set_iner_johnson_function(mesh)
    
    def set_iner_johnson_function(self, mesh):
        """Initialize the inertial Johnson damage model.
        
        Parameters
        ----------
        mesh : Mesh
            Computational mesh
        dictionnaire : dict
            Additional parameters:
            - material (Material): Material properties
            - sigma_0 (float): Yield stress parameter
            - b (float): Initial pore distance
        """
        self.V_a = self.quad.quadrature_space(["Scalar"])
        a0 = self.initial_pore_distance * self.initial_porosity**(1./3)
        self.a = Function(self.V_a, name = "Pore_length")
        self.dot_a = Function(self.V_a, name = "Pore_length_velocity")
        self.a.x.petsc_vec.set(a0)
        self.dot_a.interpolate(Expression(Constant(mesh, ScalarType(0)), self.V_a.element.interpolation_points()))
        d_expr = self.a**3  / (self.a**3 - a0**3 + (self.initial_pore_distance)**3)
        self.d_expr = Expression(d_expr, self.interp_points)

class PhaseField(Damage):
    """Phase field damage model for brittle fracture.

    This class implements phase field approaches to brittle fracture,
    including AT1, AT2, and Wu formulations. These models represent cracks
    as diffuse damage bands rather than discrete discontinuities.
    
    Attributes
    ----------
    V_d : FunctionSpace Function space for damage field
    max_d : Function Maximum damage field (upper bound)
    d : Function Current damage field
    inf_d : Function Minimum damage field
    d_prev : Function Previous damage field (for prediction)
    d_ : TestFunction Test function for damage
    dd : TrialFunction Trial function for damage
    Gc : float Critical energy release rate
    l0 : float Regularization length
    E : float, optional Young's modulus (for Wu model)
    PF_model : str Phase field model type (AT1, AT2, Wu)
    g_d : Expression Degradation function
    energy : Expression Elastic energy with damage
    fracture_energy : Expression Energy dissipated by fracture
    """
    def __init__(self, mesh):
        """Initialize the phase field damage model.

        Parameters
        ----------
        mesh : Mesh
            Computational mesh
        """
        Damage.__init__(self)
        self.V_d = functionspace(mesh, ('CG', self.dam_parameters["degree"]))
        interp_points = self.V_d.element.interpolation_points()
        self.max_d = Function(self.V_d, name = "maximum damage")
        self.max_d.interpolate(Expression(Constant(mesh, 1 - self.residual_stifness), interp_points))
        self.d = Function(self.V_d, name="Damage")
        self.d.interpolate(Expression(Constant(mesh,self.default_damage), interp_points))
        self.inf_d = Function(self.V_d, name="Inf Damage")
        self.inf_d.interpolate(Expression(Constant(mesh,self.default_damage), interp_points))
        self.d_prev = Function(self.V_d, name="Damage_predictor")
        self.d_prev.interpolate(Expression(Constant(mesh,self.default_damage), interp_points))
        self.d_ = TestFunction(self.V_d)
        self.dd = TrialFunction(self.V_d)

    def set_damage(self, mesh, dictionnaire, PF_model = "AT2"):
        """Initialize the phase field damage model parameters.

        Parameters
        ----------
        mesh : Mesh
            Computational mesh
        PF_model : str, optional
            Phase field model type: "AT1", "AT2", or "wu", default is "AT2"
        dictionnaire : dict
            Additional parameters:
            - Gc (float): Critical energy release rate
            - l0 (float): Regularization length
            - E (float, optional): Young's modulus (for Wu model)
            - sigma_c (float, optional): Critical stress (for Wu model)
            - wu_softening (str, optional): Softening type for Wu model
        """
        self.Gc = dictionnaire["Gc"]
        self.l0 = dictionnaire["l0"]
        self.E = dictionnaire.get("E", None)
        self.PF_model = PF_model
        if self.PF_model == "wu":
            self.sigma_c = dictionnaire["sigma_c"]
            self.wu_softening_type = dictionnaire["wu_softening"]
        self.set_dissipated_function_array_damage()
        
    def set_dissipated_function_array_damage(self):
        """Initialize the stiffness degradation functions.
        
        Sets up the degradation function that weights material stiffness
        based on the damage field.
        """
        if self.PF_model in ["AT1","AT2"]:
            self.g_d = (1 - self.d)**2
            if self.E != None:
                sig_c_AT1 = (3 * self.E * self.Gc / (8 * self.l0))**(1./2)
                print("La contrainte critique du model AT1 vaut ici", sig_c_AT1)
        elif self.PF_model == "wu":
            self.g_d = self.wu_degradation_function(self.d, self.sigma_c, self.Gc,self.l0, self.E)
            
    def wu_degradation_function(self, d, sigma_c, Gc, l_0, E):
        """Compute Wu's degradation function for cohesive phase field.
        
        Parameters
        ----------
        d : Function Damage field
        sigma_c : float or Expression Critical stress
        Gc : float or Expression Critical energy release rate
        l_0 : float or Expression Regularization length
        E : float or Expression Young's modulus
            
        Returns
        -------
        Expression Wu degradation function
            
        Raises
        ------
        ValueError If regularization length is too large for the given parameters
        """
        a_1 = 4 * E * Gc / (pi * l_0 * sigma_c**2)
        if type(sigma_c) == float:
            if a_1<=3./2.:
                raise ValueError("A smaller regularization length l_0 has to be chosen")
        p, a_2 = self.wu_softening()
        return (1 - d)**p / ((1 - d)**p + a_1 * d + a_1 * a_2 * d**2)
    
    def wu_softening(self):
        """Get coefficients for Wu's softening laws.
        
        Returns
        -------
        tuple (p, a_2) coefficients for the chosen softening type
        """
        if self.wu_softening_type == "exp":
            p = 2.5
            a_2 = 2**(5./3) - 3
        elif self.wu_softening_type == "lin":
            p = 2
            a_2 = -0.5
        elif self.wu_softening_type == "bilin":
            p = 4
            a_2 = 2**(7./3) - 4.5
        return p, a_2

    def set_NL_energy(self, psi):
        """Set elastic and fracture energies for phase field models.
        
        Parameters
        ----------
        psi : Expression Undamaged elastic energy
        """
        self.energy = self.g_d * psi 
        self.fracture_energy = self.set_phase_field_fracture_energy(self.Gc, self.l0, self.d)
                
    def set_phase_field_fracture_energy(self, Gc, l0, d):
        """Set the fracture energy density for phase field.
        
        Parameters
        ----------
        Gc : float or Expression Critical energy release rate
        l0 : float or Expression Regularization length
        d : Function Damage field
            
        Returns
        -------
        Expression Fracture energy density
        """
        if self.PF_model=="AT1":
            cw = 8. / 3
            w_d = d
        elif self.PF_model=="AT2":
            cw = 2.
            w_d = d**2
        elif self.PF_model=="wu":
            cw = pi
            xi = 2
            w_d = xi * d + (1 - xi) * d**2
        # Le gradient est bien cohérent quelque soit le modèle.
        return Gc / cw * (w_d / l0 + l0 * dot(grad(d), grad(d)))