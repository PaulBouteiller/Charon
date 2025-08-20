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
Hypoelastic Constitutive Solver
===============================

Solver for hypoelastic constitutive models with objective stress rates.

This module handles the integration of hypoelastic stress-strain relationships
where the stress rate depends on the current stress state and deformation rate.

Classes
-------
HypoElasticSolve  Solver for hypoelastic stress integration

"""
from dolfinx.fem import Function
from ..utils.petsc_operations import dt_update

class HypoElasticSolve:
    def __init__(self, hypo_elast, dt):
        """Solver for hypoelastic stress evolution.
        
        Integrates hypoelastic constitutive equations using explicit
        time integration of the stress rate.
        
        Parameters
        ----------
        hypo_elast : HypoElastic Hypoelastic material model object
        dt : float Time step size
        """
        self.hypo_elast = hypo_elast
        self.dt = dt
        self.dot_s_func = Function(self.hypo_elast.V_s)
        
    def solve(self):
        """Project and update stress state.
        """
        self.dot_s_func.interpolate(self.hypo_elast.dot_s)
        dt_update(self.hypo_elast.s, self.dot_s_func, self.dt)