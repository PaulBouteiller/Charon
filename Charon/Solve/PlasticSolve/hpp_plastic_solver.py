# Copyright 2025 CEA
"""
Solver pour la plasticité HPP (petites déformations)

@author: bouteillerp
"""

from ...utils.petsc_operations import petsc_add
from ...utils.generic_functions import ppart
from ufl import dot, sqrt
from dolfinx.fem import Expression, Function


class HPPPlasticSolver:
    """Solveur pour la plasticité HPP (Hypoelastic-Plastic)"""
    
    def __init__(self, problem, plastic, u):
        """
        Initialise le solveur HPP
        
        Parameters
        ----------
        plastic : HPPPlastic
            Instance de la classe HPPPlastic
        u : Function
            Champ de déplacement
        """
        self.plastic = plastic
        self.eps_p = plastic.eps_p
        self.hardening = plastic.hardening
        self.p = plastic.p
        self.kin = plastic.kin
        self.plastic_driving_force(problem.constitutive.s)

        self.Delta_eps_p = Function(plastic.Vepsp)
        self.Delta_p = Function(plastic.Vp, name = "Plastic_strain_increment")
        
    def plastic_driving_force(self, s_3D):
        """Calculate plastic driving force and plastic strain increment.
        
        Implements return mapping algorithm for J2 plasticity with
        different hardening options.
        
        Parameters
        ----------
        s_3D : Expression 3D deviatoric stress tensor
        """
        eps = 1e-10
        if self.hardening == "LinearKinematic":
            self.A = self.kin.tridim_to_mandel(s_3D - self.plastic.H * self.plastic.eps_P_3D)
            if self.pb.damage_model != None:
                self.A *= self.pb.damage.g_d
            norm_A = sqrt(dot(self.A, self.A)) + eps
            Delta_eps_p = ppart(1 - (2/3.)**(1./2) * self.sig_yield / norm_A) / \
                                (2 * self.mu + self.H) *self.A

        elif self.hardening == "Isotropic":
            s_mandel = self.kin.tridim_to_mandel(s_3D)
            sig_VM = sqrt(3.0 / 2.0 * dot(s_mandel, s_mandel)) + eps
            f_elas = sig_VM - self.plastic.sig_yield - self.plastic.H * self.plastic.p
            Delta_p = ppart(f_elas) / (3. * self.plastic.mu + self.plastic.H)
            Delta_eps_p = 3. * Delta_p / (2. * sig_VM) * s_mandel
            self.Delta_p_expression = Expression(Delta_p, self.plastic.Vp.element.interpolation_points())
        self.Delta_eps_p_expression = Expression(Delta_eps_p, self.plastic.Vepsp.element.interpolation_points())
    
    def solve(self):
        """
        Résout le problème de plasticité HPP
        
        Met à jour les variables plastiques selon le modèle HPP :
        - Déformation plastique cumulée (si durcissement isotrope)  
        - Incrément de déformation plastique
        """
        if self.hardening == "Isotropic":
            # Mise à jour de la déformation plastique cumulée
            self.Delta_p.interpolate(self.Delta_p_expression)
            petsc_add(self.p.x.petsc_vec, self.Delta_p.x.petsc_vec)
        
        # Mise à jour de l'incrément de déformation plastique
        self.Delta_eps_p.interpolate(self.Delta_eps_p_expression)
        petsc_add(self.eps_p.x.petsc_vec, self.Delta_eps_p.x.petsc_vec)