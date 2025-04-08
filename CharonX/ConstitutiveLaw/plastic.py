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
Created on Thu Mar 24 09:54:52 2022

@author: bouteillerp
"""
from ..utils.generic_functions import ppart

from ufl import (dot, sqrt, tr, dev, inner, det)
from dolfinx.fem import functionspace, Function, Expression
from math import pi, exp

class Plastic():
    def __init__(self, u, mu, name, kinematic, quadrature, plastic_model):
        """
        Parameters
        ----------
        u : Function, champ de déplacement.
        mu : Float, coefficient de cisaillement.
        name : String, nom du modèle mécanique.
        kinematic : Objet de la classe kinematic.
        plastic_model : String, HPP_Plasticity, Finite_Plasticity ou None
        """
        self.u = u
        self.V = self.u.function_space
        self.kin = kinematic
        self.mu = mu
        self.mesh = self.u.function_space.mesh
        self.mesh_dim = self.mesh.topology.dim
        self.name = name
        self.plastic_model = plastic_model
        self.quadrature = quadrature
        self.set_function(quadrature)
        
    def plastic_element(self, quadrature):
        if self.mesh_dim == 1:
            return quadrature.quad_element(["Vector", 3])
        elif self.mesh_dim == 2:
            return quadrature.quad_element(["Vector", 4])
        elif self.mesh_dim == 3:
            return quadrature.quad_element(["Vector", 6])
        
    def set_plastic(self, sigY, hardening = "CinLin", **kwargs):
        """
        Initialise les paramètres et les espaces fonctionnelles pour une étude
        élasto-plastique

        Parameters
        ----------
        hardening : String, optional, type d'écrouissage. The default is "CinLin".
        **kwargs : TYPE
            DESCRIPTION.
        """
        self.hardening = hardening
        self.sig_yield = sigY
        if self.hardening == "Iso":
            self.H = kwargs.get("H") 
        elif self.hardening == "CinLin":
            self.H = kwargs.get("H") 
        if self.plastic_model == "J2_JAX":
            self.yield_stress = kwargs.get("Hardening_func")
            assert hasattr(self, "yield_stress"), "yield_stress doit être défini pour le modèle J2_JAX"
        
class FiniteStrainPlastic(Plastic):         
    
    def set_function(self, quadrature):
        """
        Initialise les fonctions supplémentaires nécessaire à la définition
        du modèle élasto plastique en transformations finies.
        """
        element = self.plastic_element(quadrature)
        self.V_dev_BE = functionspace(self.mesh, element)
        self.dev_Be = Function(self.V_dev_BE)
        self.dev_Be_3D  = self.kin.mandel_to_tridim(self.dev_Be)
        self.u_old = Function(self.V, name = "old_displacement")
        self.F_rel = self.kin.relative_gradient_3D(self.u, self.u_old)
        self.V_Ie = quadrature.quadrature_space(["Scalar"])
        self.barI_e = Function(self.V_Ie, name = "Bar_I_elastique")
        self.barI_e.x.petsc_vec.set(1.)
        
    def set_expressions(self):
        """
        Définition des expressions symboliques des prédicteurs 
        """
        Be_trial = self.Be_trial()
        self.barI_e_expr = Expression(1./3 * tr(Be_trial), self.V_Ie.element.interpolation_points())       
        self.set_dev_Be_expression(dev(Be_trial))
        
    def Be_trial(self):
        """
        Définition du prédicteur du tenseur de Cauchy-Green gauche élastique

        Returns
        -------
        Prédicteur tridimensionnel du tenseur de Cauchy-Green gauche élastique
        """
        Be_trial_part_1 = self.barI_e * dot(self.F_rel, self.F_rel.T)
        Be_trial_part_2 = dot(dot(self.F_rel, self.dev_Be_3D), self.F_rel.T)
        return Be_trial_part_1 + Be_trial_part_2
    
    def set_dev_Be_expression(self, dev_Be_trial):
        norme_dev_Be_trial = inner(dev_Be_trial, dev_Be_trial)**(1./2) 
        mu_bar = self.mu * self.barI_e
        F_charge = self.mu * norme_dev_Be_trial - sqrt(2/3) * self.sig_yield 
        Delta_gamma = F_charge / (2 * mu_bar)
        if self.hardening == "Iso":
            Delta_gamma *= 1 / (1 + self.H / (3 * mu_bar))
        eps = 1e-6
        dev_Be_expr_3D = (1 - (2 * self.barI_e * ppart(Delta_gamma)) / (norme_dev_Be_trial + eps)) * dev_Be_trial
        dev_Be_expr = self.kin.tridim_to_mandel(dev_Be_expr_3D)
        self.dev_Be_expr = Expression(dev_Be_expr, self.V_dev_BE.element.interpolation_points())
    
class JAXJ2Plasticity(Plastic):

    def set_function(self, quadrature):
        """
        Initialise les fonctions supplémentaires nécessaire à la définition
        du modèle élasto plastique en transformations finies.
        """
        element = self.plastic_element(quadrature)
        self.V_Be = functionspace(self.mesh, element)
        self.Be_Bar_trial_func = Function(self.V_Be)
        self.Be_Bar_old = Function(self.V_Be)
        self.len_plas = len(self.Be_Bar_old)
        self.Be_Bar_old.x.array[::self.len_plas] = 1
        self.Be_Bar_old.x.array[1::self.len_plas] = 1
        self.Be_Bar_old.x.array[2::self.len_plas] = 1
        
        self.Be_bar_old_3D = self.kin.mandel_to_tridim(self.Be_Bar_old)
        
        self.u_old = Function(self.V, name = "old_displacement")
        F_rel = self.kin.relative_gradient_3D(self.u, self.u_old)
        
        expr = det(F_rel)**(-2./3) * dot(dot(F_rel.T, self.Be_bar_old_3D), F_rel.T)
        self.Be_Bar_trial = Expression(self.kin.tridim_to_mandel(expr), self.V_Be.element.interpolation_points())
        self.V_p = quadrature.quadrature_space(["Scalar"])
        self.p = Function(self.V_p, name = "Cumulated_plasticity")

class JAXGursonPlasticity(Plastic):
    def set_function(self, quadrature):
        """
        Initialise les fonctions supplémentaires nécessaire à la définition
        du modèle élasto plastique en transformations finies.
        """
        element = self.plastic_element(quadrature)
        self.V_Be = functionspace(self.mesh, element)
        self.Be_Bar_trial_func = Function(self.V_Be)
        self.Be_Bar_old = Function(self.V_Be)

        # Initialisation de Be_Bar_old avec l'identité
        self.len_plas = len(self.Be_Bar_old)
        self.Be_Bar_old.x.array[::self.len_plas] = 1
        self.Be_Bar_old.x.array[1::self.len_plas] = 1
        self.Be_Bar_old.x.array[2::self.len_plas] = 1
        self.Be_bar_old_3D = self.kin.mandel_to_tridim(self.Be_Bar_old)
        
        # Champ de déplacement ancien
        self.u_old = Function(self.V, name="old_displacement")
        F_rel = self.kin.relative_gradient_3D(self.u, self.u_old)
        
        # Prédicteur élastique
        expr = det(F_rel)**(-2./3) * dot(dot(F_rel.T, self.Be_bar_old_3D), F_rel.T)
        self.Be_Bar_trial = Expression(self.kin.tridim_to_mandel(expr), self.V_Be.element.interpolation_points())
        
        # Déformation plastique cumulée
        self.V_p = quadrature.quadrature_space(["Scalar"])
        self.p = Function(self.V_p, name="Cumulated_plasticity")
        
        # Porosité
        self.V_f = quadrature.quadrature_space(["Scalar"])
        self.f = Function(self.V_f, name="Porosity")
        
        # Paramètres du modèle de Gurson-Tvergaard-Needleman (GTN)
        self.q1 = 1.5  # Paramètre de Tvergaard
        self.q2 = 1.0  # Paramètre de Tvergaard
        self.q3 = self.q1**2  # Généralement q3 = q1^2
        
        # Paramètres d'évolution de la porosité
        self.f0 = 0.001  # Porosité initiale
        self.fc = 0.15   # Porosité critique
        self.ff = 0.25   # Porosité à rupture
        
        # Initialisation de la porosité
        self.f.x.array[:] = self.f0
        
    def compute_f_star(self, f):
        """
        Calcule la porosité effective selon le modèle GTN
        """
        if f <= self.fc:
            return f
        else:
            fu = 1/self.q1  # Porosité ultime
            return self.fc + (fu - self.fc)*(f - self.fc)/(self.ff - self.fc)
            
    def update_porosity(self, be_bar, dp, f_old, p_old):
        """
        Mise à jour de la porosité selon le modèle GTN
        """
        # Croissance des vides
        tr_D = dp * tr(self.normal(be_bar))  # Trace du taux de déformation plastique
        f_growth = (1 - f_old) * tr_D
        
        # Nucléation contrôlée par la déformation
        eN = 0.3  # Déformation moyenne de nucléation
        sN = 0.1  # Écart-type
        fN = 0.04  # Fraction volumique de vides nucléés
        
        f_nucleation = fN/(sN*sqrt(2*pi)) * \
                      exp(-0.5*((p_old - eN)/sN)**2) * dp
        
        return f_old + f_growth + f_nucleation

class HPPPlastic(Plastic):
    def set_function(self, quadrature):
        element = self.plastic_element(quadrature)
        self.Vepsp = functionspace(self.mesh, element)
        self.eps_p = Function(self.Vepsp, name = "Plastic_strain")
        self.eps_P_3D = self.kin.mandel_to_tridim(self.eps_p)
        self.delta_eps_p = Function(self.Vepsp)

    def plastic_correction(self, mu):
        """
        Correction de la contrainte par une contribution plastique purement
        déviatorique.

        Parameters
        ----------
        mat : Float, coefficient de cisaillement du matériau.
        Returns
        -------
        Tensor, contrainte plastique tridimensionnelle.
        """
        return 2 * mu * self.eps_P_3D
    
    def plastic_driving_force(self, s_3D):
        """
        Initialise la force motrice plastique A et la variation de la déformation
        plastique Delta_eps_p en fonction de cette déformation plastique
        Parameters
        ----------
        s_3D : Deviateur des contraintes 3D.
        """
        eps = 1e-10
        if self.hardening == "CinLin":
            self.A = self.kin.tridim_to_mandel(s_3D - self.H * self.eps_P_3D)
            norm_A = sqrt(dot(self.A, self.A)) + eps
            Delta_eps_p = ppart(1 - (2/3.)**(1./2) * self.sig_yield / norm_A) / \
                                (2 * self.mu + self.H) *self.A

        elif self.hardening == "Iso":
            Vp = self.quadrature.quadrature_space(["Scalar"])
            self.p = Function(Vp, name = "Cumulative_plastic_strain")
            self.dp = Function(Vp, name = "Plastic_strain_increment")
            s_mandel = self.kin.tridim_to_mandel(s_3D)
            sig_VM = sqrt(3.0 / 2.0 * dot(s_mandel, s_mandel)) + eps
            f_elas = sig_VM - self.sig_yield - self.H * self.p
            dp = ppart(f_elas) / (3 * self.mu + self.H)
            Delta_eps_p = 3 * dp / sig_VM * s_mandel
            dp_test = ppart(sig_VM - self.sig_yield)/ (3 * self.mu + self.H)
            self.dp_expression = Expression(dp_test, Vp.element.interpolation_points())
        self.Delta_eps_p_expression = Expression(Delta_eps_p, self.Vepsp.element.interpolation_points())
        