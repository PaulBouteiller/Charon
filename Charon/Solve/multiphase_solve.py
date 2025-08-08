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
        self.nb_evol = len(self.mult.dot_c)
        V_c = self.mult.V_c
        self.dot_c = [Function(V_c) for _ in range(self.nb_evol)]
        self.dot_c_expr = [Expression(self.mult.dot_c[i], V_c.element.interpolation_points()) for i in range(self.nb_evol)]
                
    def solve(self):
        """
        Actualisation des champs de concentrations et d'eventuels champs auxiliaires
        """
        # Interpolation des expressions
        for i in range(self.nb_evol):
            self.dot_c[i].interpolate(self.dot_c_expr[i])
        
        # Mise à jour temporelle
        for i in range(self.nb_evol):
            dt_update(self.mult.c[i], self.dot_c[i], self.dt)
        
        # Correction des bornes
        for i in range(self.nb_evol):
            set_correction(self.mult.c[i], self.mult.inf_c, self.mult.max_c)
        
    def two_phase_evolution(self):
        """
        Mise à jour des concentrations dans un modèle à deux phases.
        """
        self.dot_c.interpolate(self.dot_c_expression)
        dt_update(self.c_list[0], self.dot_c, -self.dt)
        dt_update(self.c_list[1], self.dot_c, self.dt)
        
    def instantaneous_evolution(self):
        """
        Mise à jour des concentrations dans un modèle à deux phases.
        """
        self.mult.c[0].interpolate(self.mult.c_expr)
        self.mult.c[1].x.array[:] = 1 - self.mult.c[0].x.array

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
            petsc_assign(self.mult.c[i], self.mult.c[i])