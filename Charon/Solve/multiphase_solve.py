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
from ..utils.petsc_operations import (dt_update, set_correction, 
                                 petsc_assign )
from ufl import exp
from dolfinx.fem import Function, Expression
        
from dolfinx.fem import Function, Expression
from ..utils.petsc_operations import dt_update, set_correction


class MultiphaseSolver:
    """Simplified solver for multiphase evolution systems."""
    
    def __init__(self, multiphase_object, dt, material):
        """Initialize the multiphase solver."""
        self.mult = multiphase_object
        self.dt = dt
        self.material = material
        
        if self.mult.has_evolution:
            V_c = self.mult.V_c
            self.dot_c_list = [Function(V_c, name=f"dot_c_{i}") for i in range(self.mult.nb_phase)]
                
    def solve(self):
        """
        Actualisation des champs de concentrations et d'eventuels champs auxiliaires
        """
        # Interpolation des expressions
        for i in range(self.nb_evol):
            self.dot_c_list[i].interpolate(self.dot_c_expression_list[i])
        
        # Mise à jour temporelle
        for i in range(self.nb_evol):
            dt_update(self.c_list[i], self.dot_c_list[i], self.dt)
        
        # Correction des bornes
        for i in range(self.nb_evol):
            set_correction(self.c_list[i], self.mult.inf_c, self.mult.max_c)
        
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
        self.c_list[0].interpolate(self.mult.c_expr)
        self.c_list[1].x.array[:] = 1 - self.c_list[0].x.array

        
    def update_c_old(self):
        """
        Mise à jour des concentrations
        """
        for i in range(self.nb_evolution):
            petsc_assign(self.c_old_list[i], self.c_list[i])