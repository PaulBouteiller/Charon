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
Created on Wed Apr 12 13:57:33 2023

@author: bouteillerp
"""
from ..utils.petsc_operations import (dt_update, set_correction, petsc_assign)
from dolfinx.fem import Function, Expression

class MultiphaseSolver:
    """Simplified solver for multiphase evolution systems."""
    
    def __init__(self, multiphase_object, dt):
        """Initialize the multiphase solver."""
        self.mult = multiphase_object
        self.dt = dt
        self.nb_evol = self.mult.nb_phase - self.mult.inertes.count(True)
        V_c = self.mult.V_c
        # Créer seulement les Function/Expression nécessaires selon le masque
        self.dot_c = []
        self.dot_c_expr = []
        self.interpolation_indices = []  # Indices des phases qui nécessitent interpolation  
        for i in range(self.nb_evol):
            mask_value = self.mult.interpolation_mask[i]
            if mask_value is not None:  # Pas inerte
                if mask_value or (mask_value is False and self.mult.reactifs[i]):
                    # True: calcul complet OU False mais réactif (on interpole une fois)
                    self.dot_c.append(Function(V_c))
                    self.dot_c_expr.append(Expression(self.mult.dot_c[i], V_c.element.interpolation_points()))
                    self.interpolation_indices.append(i)
        self.interpolation_length = len(self.dot_c)
    
    def solve(self):
        """Actualisation optimisée des champs de concentrations."""
        # Interpolation seulement des expressions nécessaires
        for idx in range(self.interpolation_length):
            self.dot_c[idx].interpolate(self.dot_c_expr[idx])
        
        # Mise à jour temporelle
        dot_c_idx = 0
        for i in range(self.nb_evol):
            mask_value = self.mult.interpolation_mask[i]
            if mask_value is True:  # Calcul complet
                dt_update(self.mult.c[i], self.dot_c[dot_c_idx], self.dt)
                dot_c_idx += 1
            else:  # Couple réactif-produit (mask_value is False)
                if self.mult.reactifs[i]:
                    # Mise à jour réactif et produit avec le même dot_c
                    dt_update(self.mult.c[i], self.dot_c[dot_c_idx], self.dt)
                    dt_update(self.mult.c[i+1], self.dot_c[dot_c_idx], -self.dt)
                    dot_c_idx += 1
                # Si produit, déjà traité avec le réactif
        
        # Correction des bornes
        for i in range(self.nb_evol):
            set_correction(self.mult.c[i], self.mult.inf_c, self.mult.max_c)
        
    def update_auxiliary_fields(self, dt, **kwargs):
        """Update auxiliary fields for all evolution laws.
        
        Parameters
        ----------
        dt : float Time step size
        **kwargs : dict Update parameters
        """
        if not self.has_evolution:
            return
        
        for evolution_law in self.evolution_laws:
            if evolution_law is not None:
                evolution_law.update_auxiliary_fields(dt, **kwargs)
      
    def update_c_old(self):
        """
        Mise à jour des concentrations
        """
        for i in range(self.nb_evolution):
            petsc_assign(self.mult.c_old[i], self.mult.c[i])