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
Plasticity Models for Mechanical Simulations
============================================

This module implements various plasticity models for mechanical simulations.
It provides a framework for handling both small strain and finite strain
plasticity, with support for different hardening mechanisms.

The module includes implementations for:
- Small strain (HPP) plasticity with isotropic/kinematic hardening
- Finite strain plasticity with multiplicative decomposition
- J2 plasticity implementation for both small and finite strain
- Gurson-Tvergaard-Needleman model for porous plasticity

Classes:
--------
Plastic : Base class for all plasticity models
    Provides common functionality and parameter handling
    
HPPPlastic : Small strain (Hypoelastic-Plastic) model
    Classical small strain J2 plasticity
    Supports both isotropic and kinematic hardening
    
FiniteStrainPlastic : Multiplicative finite strain model
    Evolutionary equations for elastic left Cauchy-Green tensor
    
JAXJ2Plasticity : J2 plasticity with algorithmic tangent
    Improved numerical performance through consistent tangent
    
JAXGursonPlasticity : Gurson model for porous plasticity
    Models void growth and coalescence
    Accounts for pressure-dependent yield
"""
from ..utils.generic_functions import ppart

from ufl import (dot, sqrt, tr, dev, inner, det, exp)
from dolfinx.fem import functionspace, Function, Expression
from math import pi

class Plastic():
    """Base class for all plasticity models.

    This class provides common functionality for plasticity models,
    including space initialization, parameter handling, and utility methods.
    
    Attributes
    ----------
    u : Function Displacement field
    V : FunctionSpace Function space for displacement
    kin : Kinematic Kinematic handler for tensor operations
    mu : float Shear modulus
    mesh : Mesh Computational mesh
    mesh_dim : int Topological dimension of mesh
    name : str Model name
    plastic_model : str Type of plasticity model
    quadrature : QuadratureHandler Handler for quadrature integration
    """
    def __init__(self, u, mu, name, kinematic, quadrature, plastic_model):
        """Initialize the plasticity model.
        
        Parameters
        ----------
        u : Function Displacement field
        mu : float Shear modulus
        name : str Model name
        kinematic : Kinematic Kinematic handler for tensor operations
        quadrature : QuadratureHandler Handler for quadrature integration
        plastic_model : str Type of plasticity model
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
        self.element = self._plastic_element(quadrature)
        self._set_function(quadrature)
        
    def _plastic_element(self, quadrature):
        """Create appropriate element for plastic variables.

        Parameters
        ----------
        quadrature : QuadratureHandler Handler for quadrature integration
            
        Returns
        -------
        Element Appropriate element for plastic variables
        """
        if self.mesh_dim == 1:
            return quadrature.quad_element(["Vector", 3])
        elif self.mesh_dim == 2:
            return quadrature.quad_element(["Vector", 4])
        elif self.mesh_dim == 3:
            return quadrature.quad_element(["Vector", 6])
        
    def set_plastic(self, sigY, hardening = "CinLin", **kwargs):
        """Initialize plasticity parameters and function spaces.
        
        Parameters
        ----------
        sigY : float Yield stress
        hardening : str, optional Type of hardening ("CinLin", "Iso"), default is "CinLin"
        **kwargs : dict
            Additional parameters:
            - H (float): Hardening modulus
            - Hardening_func (Function): Yield stress function (for J2_JAX)
        """
        self.hardening = hardening
        self.sig_yield = sigY
        if self.hardening in ["Isotropic", "Kinematic"]:
            self.H = kwargs.get("H")
        if self.plastic_model == "J2_JAX":
            self.yield_stress = kwargs.get("Hardening_func")
            assert hasattr(self, "yield_stress"), "yield_stress doit être défini pour le modèle J2_JAX"
        
class FiniteStrainPlastic(Plastic):         
    """Finite strain plasticity model with multiplicative decomposition.

    This class implements finite strain plasticity based on the multiplicative
    decomposition of the deformation gradient and evolution of the elastic
    left Cauchy-Green tensor.
    
    Attributes
    ----------
    V_dev_BE : FunctionSpace Function space for deviatoric elastic left Cauchy-Green tensor
    dev_Be : Function Deviatoric elastic left Cauchy-Green tensor
    dev_Be_3D : Expression 3D representation of deviatoric elastic left Cauchy-Green tensor
    u_old : Function Previous displacement field
    F_rel : Expression Relative deformation gradient
    V_Ie : FunctionSpace Function space for volumetric elastic left Cauchy-Green tensor
    barI_e : Function Volumetric elastic left Cauchy-Green tensor
    barI_e_expr : Expression Expression for updated volumetric elastic left Cauchy-Green tensor
    dev_Be_expr : Expression Expression for updated deviatoric elastic left Cauchy-Green tensor
    """
    def _set_function(self, quadrature):
        """Initialize functions for finite strain plasticity.
        
        Parameters
        ----------
        quadrature : QuadratureHandler Handler for quadrature integration
        """
        self.V_dev_BE = functionspace(self.mesh, self.element)
        self.dev_Be = Function(self.V_dev_BE)
        self.dev_Be_3D  = self.kin.mandel_to_tridim(self.dev_Be)
        self.u_old = Function(self.V, name = "old_displacement")
        self.F_rel = self.kin.relative_gradient_3D(self.u, self.u_old)
        self.V_Ie = quadrature.quadrature_space(["Scalar"])
        self.barI_e = Function(self.V_Ie, name = "Bar_I_elastique")
        self.barI_e.x.petsc_vec.set(1.)
        
    def set_expressions(self):
        """Define symbolic expressions for predictors.
        
        Creates expressions for the trial elastic left Cauchy-Green tensor
        and its deviatoric part, which will be used in the return mapping algorithm.
        """
        Be_trial = self.Be_trial()
        self.barI_e_expr = Expression(1./3 * tr(Be_trial), self.V_Ie.element.interpolation_points())       
        self.set_dev_Be_expression(dev(Be_trial))
        
    def Be_trial(self):
        """Define the elastic left Cauchy-Green tensor predictor.
        
        Returns
        -------
        Expression Trial elastic left Cauchy-Green tensor
        """
        Be_trial_part_1 = self.barI_e * dot(self.F_rel, self.F_rel.T)
        Be_trial_part_2 = dot(dot(self.F_rel, self.dev_Be_3D), self.F_rel.T)
        return Be_trial_part_1 + Be_trial_part_2
    
    def set_dev_Be_expression(self, dev_Be_trial):
        """Define the expression for the updated deviatoric elastic left Cauchy-Green tensor.

        Implements the return mapping algorithm for J2 plasticity in finite strain.
        
        Parameters
        ----------
        dev_Be_trial : Expression Trial deviatoric elastic left Cauchy-Green tensor
        """
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
    def _set_function(self, quadrature):
        """Initialize functions for J2 plasticity.
        
        Parameters
        ----------
        quadrature : QuadratureHandler Handler for quadrature integration
        """
        self.V_Be = functionspace(self.mesh, self.element)
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
    """Gurson-Tvergaard-Needleman model for porous plasticity.
    
    This class implements the GTN model for porous ductile materials,
    accounting for void growth, nucleation, and coalescence.
    
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
    V_f : FunctionSpace Function space for porosity
    f : Function Porosity (void volume fraction)
    q1, q2, q3 : float Tvergaard parameters
    f0, fc, ff : float Initial, critical, and failure porosity
    """
    def _set_function(self, quadrature):
        """Initialize functions for Gurson plasticity.
        
        Parameters
        ----------
        quadrature : QuadratureHandler Handler for quadrature integration
        """
        self.V_Be = functionspace(self.mesh, self.element)
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
        """Calculate effective porosity according to GTN model.
        
        Parameters
        ----------
        f : float or Function Current porosity
            
        Returns
        -------
        float or Function Effective porosity accounting for void coalescence
        """
        if f <= self.fc:
            return f
        else:
            fu = 1/self.q1  # Porosité ultime
            return self.fc + (fu - self.fc)*(f - self.fc)/(self.ff - self.fc)
            
    def update_porosity(self, be_bar, dp, f_old, p_old):
        """Update porosity according to GTN model.
        
        Accounts for void growth and nucleation of new voids.
        
        Parameters
        ----------
        be_bar : Expression Elastic left Cauchy-Green tensor
        dp : float or Function Increment of plastic strain
        f_old : float or Function Previous porosity
        p_old : float or Function Previous cumulated plastic strain
            
        Returns
        -------
        float or Function Updated porosity
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
    def _set_function(self, quadrature):
        """Initialize functions for small strain plasticity.

        Parameters
        ----------
        quadrature : QuadratureHandler
            Handler for quadrature integration
        """
        self.Vepsp = functionspace(self.mesh, self.element)
        self.eps_p = Function(self.Vepsp, name = "Plastic_strain")
        self.eps_P_3D = self.kin.mandel_to_tridim(self.eps_p)
        self.delta_eps_p = Function(self.Vepsp)

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
        if self.hardening == "Kinematic":
            self.A = self.kin.tridim_to_mandel(s_3D - self.H * self.eps_P_3D)
            norm_A = sqrt(dot(self.A, self.A)) + eps
            Delta_eps_p = ppart(1 - (2/3.)**(1./2) * self.sig_yield / norm_A) / \
                                (2 * self.mu + self.H) *self.A

        elif self.hardening == "Isotropic":
            Vp = self.quadrature.quadrature_space(["Scalar"])
            self.p = Function(Vp, name = "Cumulative_plastic_strain")
            self.Delta_p = Function(Vp, name = "Plastic_strain_increment")
            s_mandel = self.kin.tridim_to_mandel(s_3D)
            sig_VM = sqrt(3.0 / 2.0 * dot(s_mandel, s_mandel)) + eps
            f_elas = sig_VM - self.sig_yield - self.H * self.p
            Delta_p = ppart(f_elas) / (3. * self.mu + self.H)
            Delta_eps_p = 3. * Delta_p / (2. * sig_VM) * s_mandel
            self.Delta_p_expression = Expression(Delta_p, Vp.element.interpolation_points())
        self.Delta_eps_p_expression = Expression(Delta_eps_p, self.Vepsp.element.interpolation_points())
        