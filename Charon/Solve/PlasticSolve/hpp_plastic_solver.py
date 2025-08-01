# Copyright 2025 CEA
"""
Small Strain Plasticity Solver (HPP)
====================================

Implements return mapping algorithm for small strain J2 plasticity
with isotropic and kinematic hardening options.

@author: bouteillerp
"""

from ...utils.petsc_operations import petsc_add
from ...utils.generic_functions import ppart
from ufl import dot, sqrt
from dolfinx.fem import Expression, Function


class HPPPlasticSolver:
    """Solver for small strain J2 plasticity (HPP).
    
    Implements return mapping algorithm for J2 plasticity under
    small strain assumption with different hardening models.
    
    Attributes
    ----------
    plastic : HPPPlastic Small strain plasticity model instance
    hardening : str Type of hardening ("Isotropic", "LinearKinematic")
    p : Function, optional Cumulative plastic strain (isotropic hardening)
    kin : Kinematic Kinematic handler
    is_damage : str or None Damage model type if present
    Delta_eps_p : Function Plastic strain increment
    Delta_p : Function, optional Cumulative plastic strain increment (isotropic hardening)
    Delta_eps_p_expression : Expression Expression for plastic strain increment
    Delta_p_expression : Expression, optional Expression for cumulative plastic strain increment
    """
    
    def __init__(self, problem, plastic, u):
        """Initialize the small strain plasticity solver.
        
        Parameters
        ----------
        problem : Problem Problem instance containing constitutive law
        plastic : HPPPlastic Small strain plasticity model instance
        u : Function Current displacement field (unused)
        """
        self.plastic = plastic
        self.hardening = plastic.hardening
        self.p = plastic.p
        self.kin = plastic.kin

        self.is_damage = problem.constitutive.damage_model
        self.plastic_driving_force(problem.constitutive.s)
        self.Delta_eps_p = Function(plastic.Vepsp)
        self.Delta_p = Function(plastic.Vp, name = "Plastic_strain_increment")
        
    def plastic_driving_force(self, s_3D):
        """Calculate plastic driving force and strain increment.
        
        Implements return mapping algorithm for J2 plasticity with
        different hardening options and optional damage coupling.
        
        Parameters
        ----------
        s_3D : Expression 3D deviatoric stress tensor
        """
        eps = 1e-10
        if self.hardening == "LinearKinematic":
            self.A = self.kin.tensor_3d_to_mandel_compact(s_3D - self.plastic.H * self.plastic.eps_P_3D)
            if self.is_damage != None:
                self.A *= self.pb.damage.g_d
            norm_A = sqrt(dot(self.A, self.A)) + eps
            Delta_eps_p = ppart(1 - (2/3.)**(1./2) * self.sig_yield / norm_A) / \
                                (2 * self.mu + self.H) *self.A

        elif self.hardening == "Isotropic":
            if self.is_damage != None:
                s_3D *= self.pb.damage.g_d
            s_mandel = self.kin.tensor_3d_to_mandel_compact(s_3D)
            sig_VM = sqrt(3.0 / 2.0 * dot(s_mandel, s_mandel)) + eps
            f_elas = sig_VM - self.plastic.sig_yield - self.plastic.H * self.plastic.p
            Delta_p = ppart(f_elas) / (3. * self.plastic.mu + self.plastic.H)
            Delta_eps_p = 3. * Delta_p / (2. * sig_VM) * s_mandel
            self.Delta_p_expression = Expression(Delta_p, self.plastic.Vp.element.interpolation_points())
        self.Delta_eps_p_expression = Expression(Delta_eps_p, self.plastic.Vepsp.element.interpolation_points())
    
    def solve(self):
        """Solve small strain plasticity problem.
        
        Updates plastic variables according to HPP model:
        - Cumulative plastic strain (if isotropic hardening)
        - Plastic strain increment
        """
        if self.hardening == "Isotropic":
            # Mise à jour de la déformation plastique cumulée
            self.Delta_p.interpolate(self.Delta_p_expression)
            petsc_add(self.p.x.petsc_vec, self.Delta_p.x.petsc_vec)
        
        # Mise à jour de l'incrément de déformation plastique
        self.Delta_eps_p.interpolate(self.Delta_eps_p_expression)
        petsc_add(self.plastic.eps_p.x.petsc_vec, self.Delta_eps_p.x.petsc_vec)