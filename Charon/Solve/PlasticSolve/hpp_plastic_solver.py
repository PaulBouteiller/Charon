# Copyright 2025 CEA
"""
Solver pour la plasticité HPP (petites déformations)

@author: bouteillerp
"""

from ...utils.petsc_operations import petsc_add


class HPPPlasticSolver:
    """Solveur pour la plasticité HPP (Hypoelastic-Plastic)"""
    
    def __init__(self, plastic, u):
        """
        Initialise le solveur HPP
        
        Parameters
        ----------
        plastic : HPPPlastic
            Instance de la classe HPPPlastic
        u : Function
            Champ de déplacement
        """
        self.Delta_p_expression = plastic.Delta_p_expression
        self.Delta_p = plastic.Delta_p
        self.Delta_eps_p_expression = plastic.Delta_eps_p_expression
        self.Delta_eps_p = plastic.Delta_eps_p
        self.eps_p = plastic.eps_p
        self.hardening = plastic.hardening
        self.p = plastic.p
    
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