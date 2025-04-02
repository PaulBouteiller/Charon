"""
Created on Wed Sep  4 18:20:47 2024

@author: bouteillerp
"""
from dolfinx.fem import Function
from ..utils.generic_functions import dt_update

class HypoElasticSolve:
    def __init__(self, hypo_elast, dt):
        """
        Initialise le solveur pour la plasticité.

        Parameters
        ----------
        plastic : Objet de la classe plastic.
        u : Function, champ de déplacement.
        """
        self.hypo_elast = hypo_elast
        self.dt = dt
        self.dot_s_func = Function(self.hypo_elast.V_s)
        
    def solve(self):
        """
        Projection et actualisation de la déformation plastique
        """
        self.dot_s_func.interpolate(self.hypo_elast.dot_s)
        dt_update(self.hypo_elast.s, self.dot_s_func, self.dt)