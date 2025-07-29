# Copyright 2025 CEA
"""
Solver pour la plasticité en déformations finies

@author: bouteillerp
"""

from ...utils.petsc_operations import petsc_assign


class FiniteStrainPlasticSolver:
    """Solveur pour la plasticité en déformations finies"""
    
    def __init__(self, plastic, u):
        """
        Initialise le solveur en déformations finies
        
        Parameters
        ----------
        plastic : FiniteStrainPlastic
            Instance de la classe FiniteStrainPlastic
        u : Function
            Champ de déplacement
        """
        self.plastic = plastic
        self.u = u
        
    def solve(self):
        """
        Résout le problème de plasticité en déformations finies
        
        Met à jour les variables plastiques selon le modèle multiplicatif :
        - Partie volumétrique du tenseur élastique de Cauchy-Green gauche
        - Partie déviatorique du tenseur élastique de Cauchy-Green gauche
        - Champ de déplacement ancien pour le pas suivant
        """
        # Mise à jour de la partie volumétrique
        self.plastic.barI_e.interpolate(self.plastic.barI_e_expr)
        
        # Mise à jour de la partie déviatorique  
        self.plastic.dev_Be.interpolate(self.plastic.dev_Be_expr)
        
        # Sauvegarde du déplacement pour le pas suivant
        petsc_assign(self.plastic.u_old, self.u)