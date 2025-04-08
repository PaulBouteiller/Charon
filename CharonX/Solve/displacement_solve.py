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
from numpy import sqrt
from petsc4py.PETSc import ScatterMode, InsertMode
from ..utils.default_parameters import default_dynamic_parameters
from ..utils.solver_utils import petsc_div, dt_update

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
        def calculate_acceleration(dt=None):
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

    def u_solve(self):
        """Résout le problème de déplacement pour un pas de temps en utilisant le schéma sélectionné."""
        self.symplectic_integrator.solve(order=self.order, primary_field=self.u,
                                         secondary_field=self.v, dt=self.dt, bcs=self.bcs)
        
class SymplecticIntegrator:
    """Intégrateur symplectique pour les équations différentielles du second ordre."""
    
    def __init__(self, acceleration_calculator):
        self.derivative_calculator = acceleration_calculator
        self.methods = self._create_symplectic_methods()
    
    def _create_symplectic_methods(self):
        """Crée une bibliothèque de méthodes symplectiques."""
        methods = {}
        # a_i/b_i - coefficients pour la position/vitesse
        methods["Optimal_Order1"] = {"a_coeffs": [1], "b_coeffs": [1]}
        
        # Méthode LeapFrog/Verlet
        a12 = 1 / sqrt(2)
        a22 = 1 - a12
        methods["Optimal_Order2"] = {"a_coeffs": [a12, a22], "b_coeffs": [a12, a22]}
        
        # Méthode McLachlan d'ordre 3 optimale
        a13 = 0.919661523017399857
        a23 = 1./ (4 * a13) - a13/2
        a33 = 1 - a13 - a23
        methods["Optimal_Order3"] = {"a_coeffs": [a13, a23, a33],
                                     "b_coeffs": [a33, a23, a13]} #Symétrie: b_i = a_{4-i}
        
        # ---- Méthode d'ordre 4 optimale (pour T quadratique) ----
        a14 = 0.5153528374311228364
        a24 = -0.085782019412973646
        a34 = 0.4415830236164665242
        a44 = 0.1288461583653841854
        
        b14 = 0.1343961992774310892
        b24 = -0.2248198030794208058
        b34 = 0.7563200005156682911
        b44 = 0.3340036032863214255
        
        methods["Optimal_Order4"] = {"a_coeffs": [a14, a24, a34, a44],
                                     "b_coeffs": [b14, b24, b34, b44]}
        
        # ---- Méthode d'ordre 5 optimale (pour T quadratique) ----
        # Valeurs transcrites directement du tableau 2
        a15 = 0.339839625839110
        a25 = -0.088601336903027329
        a35 = 0.5858564768259621188
        a45 = -0.603039356536491888
        a55 = 0.3235807965546976394
        a65 = 0.4423637942197494587
        
        b15 = 0.1193900292875672758
        b25 = 0.6989273703824752308
        b35 = -0.1713123582716007754
        b45 = 0.401269502251353448
        b55 = 0.0107050818482359840
        b65 = -0.0589796254980311632
        
        methods["Optimal_Order5"] = {"a_coeffs": [a15, a25, a35, a45, a55, a65],
                                     "b_coeffs": [b15, b25, b35, b45, b55, b65]}
        
        return methods
    
    def solve(self, order, primary_field, secondary_field, dt, bcs):
        """Résout une étape de temps avec la méthode symplectique spécifiée."""
        u, v = primary_field, secondary_field
        method = self.methods["Optimal_Order"+str(order)]
        
        # Format avec listes de coefficients a_i et b_i
        a_coeffs = method["a_coeffs"]
        b_coeffs = method["b_coeffs"]
        for i in range(len(a_coeffs)):
            # Mise à jour de la vitesse
            if b_coeffs[i] != 0:
                self.derivative_calculator(dt=b_coeffs[i]*dt)
            # Mise à jour de la position
            if a_coeffs[i] != 0:
                dt_update(u, v, a_coeffs[i]*dt)
                set_bc(u.x.petsc_vec, bcs.bcs)
        return