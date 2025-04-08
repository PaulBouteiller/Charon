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
from ..utils.solver_utils import petsc_div, dt_update

from .TimeIntegrator import SymplecticIntegrator

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
        self.order =  default_dynamic_parameters()["order"]
        self._set_explicit_function(form, m_form)
        
    def _update_ghost_values(self, vector):
        """Update ghost values for a PETSc vector.
        
        Parameters
        ----------
        vector : PETSc.Vec Vector to update
        """
        vector.ghostUpdate(addv=InsertMode.ADD, mode=ScatterMode.REVERSE)
        vector.ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)
        
    def _set_explicit_function(self, residual_form, m_form):
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
        self._update_ghost_values(self.diag_M)
        self.local_res = form(-residual_form)
        # if self.scheme == "LeapFrog":
        #     self._update_acceleration_velocity(self.dt/2)
            
        # Initialisation optionnelle de l'intégrateur symplectique
        self.use_symplectic_integrator = True  # Peut être activé si nécessaire
        if self.use_symplectic_integrator:
            def calculate_acceleration(dt=None, update_velocity=False):
                if update_velocity:
                    self._update_acceleration_velocity(dt if dt is not None else self.dt)
                return self.a
            
            self.symplectic_integrator = SymplecticIntegrator(calculate_acceleration)
            
        
    def _update_acceleration_velocity(self, dt):
        """Update acceleration and velocity.
        
        Parameters
        ----------
        dt : float Time step
        """
        # Assemble residual
        res = assemble_vector(self.local_res)
        self._update_ghost_values(res)
        
        # Compute acceleration: a = M^(-1) * res
        petsc_div(res, self.diag_M, self.a.x.petsc_vec)
        set_bc(self.a.x.petsc_vec, self.bcs.a_bcs)
        
        # Update velocity: v += dt * a
        dt_update(self.v, self.a, dt)
        set_bc(self.v.x.petsc_vec, self.bcs.v_bcs)
        self.v.x.petsc_vec.ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)
        
        res.destroy()

    def _integration_step(self, dt_u_factor, dt_a_factor=None, apply_acceleration=True):
        """Effectue une étape d'intégration générique.
        
        Parameters
        ----------
        dt_u_factor : float Facteur de multiplication pour le pas de temps dans la mise à jour du déplacement
        dt_a_factor : float, optional Facteur de multiplication pour le pas de temps dans la mise à jour de l'accélération
        apply_acceleration : bool, optional Si True, met à jour l'accélération et la vitesse, par défaut True
        """
        # Mise à jour du déplacement
        dt_update(self.u, self.v, dt_u_factor * self.dt)
        set_bc(self.u.x.petsc_vec, self.bcs.bcs)
        self.u.x.petsc_vec.ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)
        
        # Mise à jour de l'accélération et de la vitesse si nécessaire
        if apply_acceleration and dt_a_factor is not None:
            self._update_acceleration_velocity(dt_a_factor * self.dt)
    
    def _execute_scheme(self, steps):
        """Exécute un schéma d'intégration générique.
        
        Parameters
        ----------
        steps : list of tuple Liste de tuples (dt_u_factor, dt_a_factor, apply_acceleration)
        """
        for dt_u_factor, dt_a_factor, apply_acceleration in steps:
            self._integration_step(dt_u_factor, dt_a_factor, apply_acceleration)
    
    # def u_solve(self):
    #     """Résout le problème de déplacement pour un pas de temps en utilisant le schéma sélectionné."""
    #     if self.scheme == "LeapFrog":
    #         # Schéma LeapFrog: une seule étape
    #         self._update_acceleration_velocity(self.dt)
    #         self._integration_step(1.0, None, False)
    #     elif self.scheme == "Yoshida":
    #         # Schéma Yoshida: quatre étapes
    #         steps = [(self.c1, self.d1, True),
    #                  (self.c2, self.d2, True),
    #                  (self.c2, self.d1, True),
    #                  (self.c1, None, False)]
    #         self._execute_scheme(steps)
    
    def u_solve(self):
        """Résout le problème de déplacement pour un pas de temps en utilisant le schéma sélectionné."""
        if hasattr(self, 'symplectic_integrator') and self.use_symplectic_integrator:
            # Utilisation de l'intégrateur symplectique si activé
            self.symplectic_integrator.solve(
                order=self.order,
                primary_field=self.u,
                secondary_field=self.v,
                tertiary_field=self.a,
                dt=self.dt,
                bcs=self.bcs
            )
        else:
            # Implémentation originale inchangée
            if self.scheme == "LeapFrog":
                # Schéma LeapFrog: une seule étape
                self._update_acceleration_velocity(self.dt)
                self._integration_step(1.0, None, False)
            elif self.scheme == "Yoshida":
                # Schéma Yoshida: quatre étapes
                steps = [(self.c1, self.d1, True),
                         (self.c2, self.d2, True),
                         (self.c2, self.d1, True),
                         (self.c1, None, False)]
                self._execute_scheme(steps)