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
Created on Mon Sep 26 17:51:49 2022

@author: bouteillerp
"""
from dolfinx.fem.petsc import assemble_vector, set_bc
from dolfinx.fem import Function, form
from ufl import action
from petsc4py.PETSc import ScatterMode, InsertMode
from ..utils.default_parameters import default_dynamic_parameters
from ..utils.generic_functions import petsc_div, dt_update

class ExplicitDisplacementSolver:
    def __init__(self, u, v, dt, m_form, form, bcs):
        """
        Initilise le solveur explicit de calcul du déplacement

        Parameters
        ----------
        u : Function, champ de déplacement
        v : Function, champ de vitesse
        dt : float, pas de temps
        m_form : Form, forme biilinéaire de masse
        form : Form, résidu du problème mécanique: a(u,v) - L(v)
        bcs : DirichletBC, conditions aux limites en déplacement
        """
        self.u = u
        self.v = v
        self.a = Function(self.u.function_space, name = "Accelerations")
        self.dt = dt
        self.bcs = bcs
        self.scheme = default_dynamic_parameters()["scheme"]
        if self.scheme == "Yoshida":
            self.d1 = 1 / (2 - 2**(3./2))
            self.d2 = -2**(3/2) / (2 - 2**(3./2))
            self.c1 = self.d1/2
            self.c2 = (self.d2 + self.d1)/2

        self.set_explicit_function(form, m_form)
        
    def set_explicit_function(self, residual_form, m_form):
        """
        Définition du vecteur de masse, issu de la condensation de la matrice de masse
        
        Parameters
        ----------
        residual_form: Form, résidu = forme linéaire a(u,v)-l(v)
        m_form : Form, forme biilinéaire de masse.
        """
        u1 = Function(self.u.function_space)
        u1.x.petsc_vec.set(1.)
        self.diag_M = assemble_vector(form(action(m_form, u1)))    

        set_bc(self.diag_M, self.bcs.bcs_axi)
        self.diag_M.ghostUpdate(addv = InsertMode.ADD, mode = ScatterMode.REVERSE)
        self.diag_M.ghostUpdate(addv = InsertMode.INSERT, mode = ScatterMode.FORWARD)

        self.local_res = form(-residual_form)
        if self.scheme == "LeapFrog":
            self.compute_acceleration_speed(self.dt/2)

    def compute_acceleration_speed(self, dt):
        """
        Calcul le résidu puis l'utilise afin de calculer la vitesse et l'accélération
        """
        res = assemble_vector(self.local_res)
        res.ghostUpdate(addv = InsertMode.ADD, mode = ScatterMode.REVERSE)
        res.ghostUpdate(addv = InsertMode.INSERT, mode = ScatterMode.FORWARD)
        petsc_div(res, self.diag_M, self.a.x.petsc_vec)
        set_bc(self.a.x.petsc_vec, self.bcs.a_bcs)
        dt_update(self.v, self.a, dt)
        set_bc(self.v.x.petsc_vec, self.bcs.v_bcs)
        self.v.x.petsc_vec.ghostUpdate(addv = InsertMode.INSERT, mode = ScatterMode.FORWARD)
        res.destroy()
            
    def u_solve(self):
        """
        Calcl du déplacement sur une iteration explicite:
        -actualisation des déplacements
        -imposition des CLs en déplacements
        -calcul de l'acceleration et de la vitesse
        """      
        if self.scheme == "LeapFrog":
            # dt_update(self.u, self.v, self.dt)
            # set_bc(self.u.x.petsc_vec, self.bcs.bcs)
            # self.u.x.petsc_vec.ghostUpdate(addv = InsertMode.INSERT, mode = ScatterMode.FORWARD)
            self.compute_acceleration_speed(self.dt)
            #Debug
            dt_update(self.u, self.v, self.dt)
            set_bc(self.u.x.petsc_vec, self.bcs.bcs)
            self.u.x.petsc_vec.ghostUpdate(addv = InsertMode.INSERT, mode = ScatterMode.FORWARD)
        elif self.scheme == "Yoshida":
            dt_update(self.u, self.v, self.c1 * self.dt)
            set_bc(self.u.x.petsc_vec, self.bcs.bcs)
            self.compute_acceleration_speed(self.d1 * self.dt)
            dt_update(self.u, self.v, self.c2 * self.dt)
            set_bc(self.u.x.petsc_vec, self.bcs.bcs)
            self.compute_acceleration_speed(self.d2 * self.dt)
            dt_update(self.u, self.v, self.c2 * self.dt)
            set_bc(self.u.x.petsc_vec, self.bcs.bcs)
            self.compute_acceleration_speed(self.d1 * self.dt)
            dt_update(self.u, self.v, self.c1 * self.dt)
            set_bc(self.u.x.petsc_vec, self.bcs.bcs)