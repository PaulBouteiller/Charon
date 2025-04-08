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
Intégrateurs symplectiques pour les équations différentielles du second ordre.
Comprend des méthodes basées sur les tableaux de Butcher partitionnés symplectiques (SPRK)
et des méthodes de composition.
"""

import numpy as np
from dolfinx.fem import Function
from petsc4py.PETSc import InsertMode, ScatterMode

class SymplecticIntegrator:
    """Intégrateur symplectique pour les équations différentielles du second ordre."""
    
    def __init__(self, acceleration_calculator):
        """Initialise l'intégrateur symplectique.
        
        Parameters
        ----------
        acceleration_calculator : callable
            Fonction qui calcule l'accélération à partir de la position et de la vitesse
            Signature: acceleration_calculator(dt=0.0, update_velocity=False) -> None
        """
        self.acceleration_calculator = acceleration_calculator
        self.methods = self._create_symplectic_methods()
    
    def _create_symplectic_methods(self):
        """Crée une bibliothèque de méthodes symplectiques."""
        methods = {}
        
        # Méthode LeapFrog/Verlet
        methods["LeapFrog"] = {
            "type": "leapfrog",
            "name": "LeapFrog/Verlet",
            "order": 2,
            "stages": 2,
            # Pas de coefficients spécifiques, méthode gérée directement
        }
        
        # Méthode de Yoshida (version basic)
        methods["Yoshida"] = {
            "type": "composition",
            "name": "Yoshida 4th order",
            "order": 4,
            "stages": 4,
            "coefficients": {
                "d1": 1 / (2 - 2**(3./2)),
                "d2": -2**(3/2) / (2 - 2**(3./2)),
                "c1": (1 / (2 - 2**(3./2))) / 2,
                "c2": (-2**(3/2) / (2 - 2**(3./2)) + 1 / (2 - 2**(3./2))) / 2,
            }
        }
        
        # Ajout des méthodes SPRK de Sundials
        # ARKODE_SPRK_EULER_1_1
        methods["SPRK_EULER_1_1"] = {
            "type": "composition",
            "name": "Symplectic Euler",
            "order": 1,
            "stages": 1,
            "coefficients": [1.0]
        }
        
        # ARKODE_SPRK_LEAPFROG_2_2
        methods["SPRK_LEAPFROG_2_2"] = {
            "type": "leapfrog",
            "name": "Leapfrog/Verlet (SPRK)",
            "order": 2,
            "stages": 2,
            # Utilisera l'implémentation standard de LeapFrog
        }
        
        # ARKODE_SPRK_MCLACHLAN_4_4
        w1 = 0.17318
        w2 = 0.5 - w1
        w3 = 1.0 - 2*(w1 + w2)
        methods["SPRK_MCLACHLAN_4_4"] = {
            "type": "composition",
            "name": "McLachlan 4th order",
            "order": 4,
            "stages": 5,
            "coefficients": [w1, w2, w3, w2, w1]
        }
        
        # ARKODE_SPRK_YOSHIDA_6_8
        methods["SPRK_YOSHIDA_6_8"] = {
            "type": "composition",
            "name": "Yoshida 6th order",
            "order": 6,
            "stages": 8,
            "coefficients": [
                0.39225680523878,
                0.5100434119184585,
                -0.47105338540976,
                0.06875316825252003,
                0.06875316825252003,
                -0.47105338540976,
                0.5100434119184585,
                0.39225680523878
            ]
        }
        
        # ARKODE_SPRK_MCLACHLAN_3_3 (McLachlan 3rd order)
        methods["SPRK_MCLACHLAN_3_3"] = {
            "type": "composition",
            "name": "McLachlan 3rd order",
            "order": 3,
            "stages": 3,
            "coefficients": [
                0.9171523357008925,
                -0.8343046714017851,
                0.9171523357008925
            ]
        }
        
        # ARKODE_SPRK_CANDY_ROZMUS_4_4
        methods["SPRK_CANDY_ROZMUS_4_4"] = {
            "type": "composition",
            "name": "Candy-Rozmus 4th order",
            "order": 4,
            "stages": 4,
            "coefficients": [
                0.5, 
                1.0,
                -0.5,
                0.0
            ]
        }
        
        
        return methods
    
    def solve_leapfrog(self, u, v, a, dt, bcs):
        """Exécute une étape de temps avec la méthode LeapFrog/Verlet.
        
        Parameters
        ----------
        u : Function
            Fonction de déplacement
        v : Function
            Fonction de vitesse
        a : Function
            Fonction d'accélération
        dt : float
            Pas de temps
        bcs : BoundaryConditions
            Conditions aux limites
        """
        # Calcul de l'accélération et mise à jour de la vitesse
        self.acceleration_calculator(dt=dt, update_velocity=True)
        
        # Mise à jour du déplacement
        u.x.array[:] += dt * v.x.array[:]
        u.x.petsc_vec.ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)
        bcs.apply(u, v)
    
    def solve_composition(self, u, v, a, dt, bcs, coefficients):
        """Exécute une étape de temps avec une méthode de composition symplectique.
        
        Parameters
        ----------
        u : Function
            Fonction de déplacement
        v : Function
            Fonction de vitesse
        a : Function
            Fonction d'accélération
        dt : float
            Pas de temps
        bcs : BoundaryConditions
            Conditions aux limites
        coefficients : list or dict
            Coefficients de la méthode de composition
        """
        if isinstance(coefficients, dict):
            # Cas spécial pour Yoshida 4 original
            if all(k in coefficients for k in ["c1", "c2", "d1", "d2"]):
                # Étape 1
                u.x.array[:] += coefficients["c1"] * dt * v.x.array[:]
                u.x.petsc_vec.ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)
                bcs.apply(u, v)
                self.acceleration_calculator(dt=coefficients["d1"]*dt, update_velocity=True)
                
                # Étape 2
                u.x.array[:] += coefficients["c2"] * dt * v.x.array[:]
                u.x.petsc_vec.ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)
                bcs.apply(u, v)
                self.acceleration_calculator(dt=coefficients["d2"]*dt, update_velocity=True)
                
                # Étape 3
                u.x.array[:] += coefficients["c2"] * dt * v.x.array[:]
                u.x.petsc_vec.ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)
                bcs.apply(u, v)
                self.acceleration_calculator(dt=coefficients["d1"]*dt, update_velocity=True)
                
                # Étape 4
                u.x.array[:] += coefficients["c1"] * dt * v.x.array[:]
                u.x.petsc_vec.ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)
                bcs.apply(u, v)
                
        elif isinstance(coefficients, list):
            # Méthodes de composition générale avec coefficients
            for coef in coefficients:
                # Mise à jour de la position
                u.x.array[:] += coef * dt * v.x.array[:]
                u.x.petsc_vec.ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)
                bcs.apply(u, v)
                
                # Calcul de l'accélération et mise à jour de la vitesse
                self.acceleration_calculator(dt=coef*dt, update_velocity=True)
    
    def solve(self, scheme_name, u, v, a, dt, bcs):
        """Résout une étape de temps avec la méthode symplectique spécifiée.
        
        Parameters
        ----------
        scheme_name : str
            Nom du schéma à utiliser
        u : Function
            Fonction de déplacement
        v : Function
            Fonction de vitesse
        a : Function
            Fonction d'accélération
        dt : float
            Pas de temps
        bcs : BoundaryConditions
            Conditions aux limites
            
        Returns
        -------
        bool
            True si succès, False sinon
        """
        if scheme_name not in self.methods:
            return False
            
        method = self.methods[scheme_name]
        
        if method["type"] == "leapfrog":
            self.solve_leapfrog(u, v, a, dt, bcs)
        elif method["type"] == "composition":
            self.solve_composition(u, v, a, dt, bcs, method["coefficients"])
        else:
            return False
            
        return True