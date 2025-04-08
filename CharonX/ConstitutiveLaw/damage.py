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
Created on Thu Mar 17 16:17:46 2022

@author: bouteillerp
"""
from ufl import (TestFunction, TrialFunction, dot, grad, sqrt, conditional)
from dolfinx.fem import (Function, functionspace, Constant, Expression)
from petsc4py.PETSc import ScalarType
from numpy import pi

from ..utils.default_parameters import default_damage_parameters, default_porosity_parameters

class Damage:
    def __init__(self):
        self.dam_parameters = default_damage_parameters()
        self.residual_stifness = self.dam_parameters["residual_stiffness"]
        self.default_damage = ScalarType(self.dam_parameters["default_damage"])
        
class Johnson(Damage):
    def __init__(self, mesh, quadrature):
        Damage.__init__(self)
        self.quad = quadrature
        self.initial_porosity = default_porosity_parameters()["initial_porosity"]
        self.V_d = quadrature.quadrature_space(["Scalar"])
        self.inf_d = Function(self.V_d, name = "Minimum porosity")
        self.max_d = Function(self.V_d, name = "Maximum porosity") 
        self.d = Function(self.V_d, name = "Porosity")
        self.g_d = (1 - self.d)
        self.Vd_interp_points = self.V_d.element.interpolation_points()
        
    def initialize_Johnson(self, mesh, **kwargs):
        interp_points = self.V_d.element.interpolation_points()
        f0 = kwargs.get("f0", Expression(Constant(mesh, self.initial_porosity), interp_points))
        self.inf_d.interpolate(f0)
        self.max_d.interpolate(Expression(Constant(mesh, 1 - self.residual_stifness), interp_points))
        self.d.interpolate(f0)
        
    def set_p_mot(self, undammaged_pressure):
        self.p_mot = Expression(undammaged_pressure, self.Vd_interp_points)
        self.p_func = Function(self.V_d)
        
    def set_unbreakable_zone(self, condition):
        ufl_condition = conditional(condition, self.initial_porosity, 1 - self.residual_stifness)
        expr = Expression(ufl_condition, self.V_d.element.interpolation_points())
        self.max_d.interpolate(expr)

class StaticJohnson(Johnson):
    def set_damage(self, mesh, **kwargs):
        """
        Initialise les paramètres requis pour le modèle d'endommagement
        inséré en mot clé de damage_model.

        Parameters
        ----------
        mesh : Mesh, maillage du domaine.
        **kwargs : Paramètres nécessaire au modèle d'endommagement choisi.
        """
        self.regularization = kwargs.get("regularization", False)
        self.eta = kwargs["eta"]
        self.sigma_0 = kwargs["sigma_0"]
        self.initialize_Johnson(mesh, **kwargs)
        if self.regularization:
            self.V_d_regul = functionspace(mesh, ('CG', 2))
            self.lc = kwargs["l0"]  

class DynamicJohnson(Johnson):
    def set_damage(self, mesh, **kwargs):
        """
        Initialise les paramètres requis pour le modèle d'endommagement
        inséré en mot clé de damage_model.

        Parameters
        ----------
        mesh : Mesh, maillage du domaine.
        **kwargs : Paramètres nécessaire au modèle d'endommagement choisi.
        """
        self.eta = kwargs["eta"]
        self.sigma_0 = kwargs["sigma_0"]
        self.initial_pore_distance = kwargs["b"]
        self.tau = self.eta / self.sigma_0
        self.v_0 = sqrt(self.sigma_0 / kwargs["material"].rho_0)
        self.l_dyn = self.tau * self.v_0
        print("Le temps caractéristique visqueux vaut", self.tau)
        print("La longueur inertielle vaut", self.l_dyn)
        self.initialize_Johnson(mesh, **kwargs)
        self.set_dyn_johnson_function(mesh)
            
    def set_dyn_johnson_function(self, mesh):
        """
        Initialise les fonctions nécessaires au modèle de Johnson dynamique
        la variable d désigne ici la porosité du matériau, et a désigne le rayon du pore
        Parameters
        ----------
        mesh : Mesh, maillage du domaine.
        """
        self.V_a = self.quad.quadrature_space(["Scalar"])
        a0 = self.initial_pore_distance * self.initial_porosity**(1./3)
        a_tilde_0 = a0 / self.l_dyn
        self.a_tilde = Function(self.V_a, name = "Pore_length")
        self.dot_a_tilde = Function(self.V_a, name = "Pore_length_velocity")
        self.a_tilde.x.petsc_vec.set(a_tilde_0)
        d_expr = self.a_tilde**3  / (self.a_tilde**3 - a_tilde_0**3 + (self.initial_pore_distance / self.l_dyn)**3)
        self.d_expr = Expression(d_expr, self.V_d.element.interpolation_points())

class InertialJohnson(Johnson):
    def set_damage(self, mesh, **kwargs):
        """
        Initialise les paramètres requis pour le modèle d'endommagement
        inséré en mot clé de damage_model.

        Parameters
        ----------
        mesh : Mesh, maillage du domaine.
        **kwargs : Paramètres nécessaire au modèle d'endommagement choisi.
        """
        self.rho_0 = kwargs["material"].rho_0
        self.sigma_0 = kwargs["sigma_0"]
        self.initial_pore_distance = kwargs["b"]
        self.initialize_Johnson(mesh, **kwargs)
        self.set_iner_johnson_function(mesh)
    
    def set_iner_johnson_function(self, mesh):
        """
        Initialise les fonctions nécessaires au modèle de Johnson dynamique
        la variable d désigne ici la porosité du matériau, et a désigne le rayon du pore
        Parameters
        ----------
        mesh : Mesh, maillage du domaine.
        """
        self.V_a = self.quad.quadrature_space(["Scalar"])
        a0 = self.initial_pore_distance * self.initial_porosity**(1./3)
        self.a = Function(self.V_a, name = "Pore_length")
        self.dot_a = Function(self.V_a, name = "Pore_length_velocity")
        self.a.x.petsc_vec.set(a0)
        self.dot_a.interpolate(Expression(Constant(mesh, ScalarType(0)), self.V_a.element.interpolation_points()))
        d_expr = self.a**3  / (self.a**3 - a0**3 + (self.initial_pore_distance)**3)
        self.d_expr = Expression(d_expr, self.V_d.element.interpolation_points())

class PhaseField(Damage):
    def __init__(self, mesh):
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

    def set_damage(self, mesh, PF_model = "AT2", **kwargs):
        self.Gc = kwargs["Gc"]
        self.l0 = kwargs["l0"]
        self.E = kwargs.get("E", None)
        self.PF_model = PF_model
        if self.PF_model == "wu":
            self.sigma_c = kwargs["sigma_c"]
            self.wu_softening_type = kwargs["wu_softening"]
        self.set_dissipated_function_array_damage()
        
    def set_dissipated_function_array_damage(self):
        """
        Initialise les fonctions de dégradation pondérant la rigidité du matériau.
        """
        if self.PF_model in ["AT1","AT2"]:
            self.g_d = (1 - self.d)**2
            if self.E != None:
                sig_c_AT1 = (3 * self.E * self.Gc / (8 * self.l0))**(1./2)
                print("La contrainte critique du model AT1 vaut ici", sig_c_AT1)
        elif self.PF_model == "wu":
            self.g_d = self.wu_degradation_function(self.d, self.sigma_c, self.Gc,self.l0, self.E)
            
    def wu_degradation_function(self, d, sigma_c, Gc, l_0, E):
        """
        Renvoie les fonctions de Wu dégradant la rigidité dans le cadre 
        du modèle phase field cohésif de Wu.
        Parameters
        ----------
        d : Function, champ d'endommagement.
        sigma_c : Float ou Expression, contrainte critique à la rupture.
        Gc : Float ou Expression, taux de restitution d'énergie critique.
        l_0 : Float ou Expression, longueur de régularisation.
        E : Float ou Expression, module de Young.
        """
        a_1 = 4 * E * Gc / (pi * l_0 * sigma_c**2)
        if type(sigma_c) == float:
            if a_1<=3./2.:
                raise ValueError("A smaller regularization length l_0 has to be chosen")
        p, a_2 = self.wu_softening()
        return (1 - d)**p / ((1 - d)**p + a_1 * d + a_1 * a_2 * d**2)
    
    def wu_softening(self):
        """
        Retourne les coefficients pour les lois d'adoucissement dans le modèle de Wu-phase field
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
        """
        Initialise les énergies élastiques et de fissuration pour les modèles
        de rupture par champ de phase.

        Parameters
        ----------
        psi : Expression, énergie élastique non endommagée.
        """
        self.energy = self.g_d * psi 
        self.fracture_energy = self.set_phase_field_fracture_energy(self.Gc, self.l0, self.d)
                
    def set_phase_field_fracture_energy(self, Gc, l0, d):
        """
        Initialise la densité d'énergie de fissuration dans le cadre phase field.
        Parameters
        ----------
        Gc : Float ou Expression, taux de restitution d'énergie critique.
        l_0 : Float ou Expression, longueur de régularisation.
        d : Function, champ d'endommagement.
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