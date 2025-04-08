#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 11:20:12 2025

@author: bouteillerp
"""
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
Intégrateurs basés sur les tableaux de Butcher pour les équations différentielles.
"""

import numpy as np
from petsc4py.PETSc import InsertMode, ScatterMode

class ButcherTableau:
    """Représentation d'un tableau de Butcher pour les méthodes Runge-Kutta."""
    
    def __init__(self, a, b, c, name="Custom", description="", order=0):
        """Initialise un tableau de Butcher.
        
        Parameters
        ----------
        a : ndarray
            Matrice des coefficients a_ij (triangulaire inférieure)
        b : ndarray
            Vecteur des poids b_i
        c : ndarray
            Vecteur des points c_i
        name : str, optional
            Nom de la méthode, par défaut "Custom"
        description : str, optional
            Description de la méthode, par défaut ""
        order : int, optional
            Ordre de la méthode, par défaut 0
        """
        self.a = np.array(a, dtype=float)
        self.b = np.array(b, dtype=float)
        self.c = np.array(c, dtype=float)
        self.name = name
        self.description = description
        self.order = order
        self.stages = len(b)
        
        # Vérification de cohérence
        if self.a.shape[0] != self.stages or self.a.shape[1] != self.stages:
            raise ValueError(f"La matrice a doit être de taille {self.stages}×{self.stages}")
        if len(self.c) != self.stages:
            raise ValueError(f"Le vecteur c doit être de longueur {self.stages}")
            
    def __repr__(self):
        return f"ButcherTableau(name='{self.name}', stages={self.stages}, order={self.order})"
    
    def is_explicit(self):
        """Vérifie si la méthode est explicite (a_ij = 0 pour j ≥ i)."""
        for i in range(self.stages):
            for j in range(i, self.stages):
                if abs(self.a[i, j]) > 1e-14:
                    return False
        return True

class ButcherIntegrator:
    """Intégrateur temporel basé sur les tableaux de Butcher."""
    
    def __init__(self, acceleration_calculator):
        """Initialise l'intégrateur.
        
        Parameters
        ----------
        acceleration_calculator : callable
            Fonction qui calcule l'accélération à partir de la position et de la vitesse
            Signature: acceleration_calculator(update_velocity=False) -> None
        """
        self.acceleration_calculator = acceleration_calculator
        self.tableaux = self._create_butcher_tableaux()
    
    def _create_butcher_tableaux(self):
        """Crée une bibliothèque de tableaux de Butcher prédéfinis."""
        tableaux = {}
        
        # Méthode d'Euler explicite (RK1)
        tableaux["RK1"] = ButcherTableau(
            a=[[0]],
            b=[1],
            c=[0],
            name="Forward Euler",
            description="Méthode d'Euler explicite d'ordre 1",
            order=1
        )
        
        # Méthode de Heun (RK2)
        tableaux["RK2"] = ButcherTableau(
            a=[[0, 0], 
               [1, 0]],
            b=[1/2, 1/2],
            c=[0, 1],
            name="Heun",
            description="Méthode de Heun d'ordre 2",
            order=2
        )
        
        # Méthode RK4 classique
        tableaux["RK4"] = ButcherTableau(
            a=[[0, 0, 0, 0],
               [1/2, 0, 0, 0],
               [0, 1/2, 0, 0],
               [0, 0, 1, 0]],
            b=[1/6, 1/3, 1/3, 1/6],
            c=[0, 1/2, 1/2, 1],
            name="Classical RK4",
            description="Méthode de Runge-Kutta classique d'ordre 4",
            order=4
        )
        
        # Méthode Dormand-Prince (DOPRI5)
        tableaux["DOPRI5"] = ButcherTableau(
            a=[[0, 0, 0, 0, 0, 0, 0],
               [1/5, 0, 0, 0, 0, 0, 0],
               [3/40, 9/40, 0, 0, 0, 0, 0],
               [44/45, -56/15, 32/9, 0, 0, 0, 0],
               [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
               [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
               [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]],
            b=[35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0],
            c=[0, 1/5, 3/10, 4/5, 8/9, 1, 1],
            name="Dormand-Prince",
            description="Méthode Dormand-Prince d'ordre 5 (utilisée dans ode45)",
            order=5
        )
        
        return tableaux
    
    def solve(self, scheme_name, u, v, a, dt, bcs):
        """Résout une étape de temps avec la méthode RK spécifiée.
        
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
        if scheme_name not in self.tableaux:
            return False
            
        tableau = self.tableaux[scheme_name]
        
        # Vérification que le tableau est explicite
        if not tableau.is_explicit():
            raise ValueError(f"Le tableau {tableau.name} n'est pas explicite")
        
        # Sauvegarde de l'état initial
        u0 = u.copy()
        v0 = v.copy()
        
        # Stockage des k (accélérations) pour chaque étape
        k_stages = []
        
        # Calcul des étapes intermédiaires
        for i in range(tableau.stages):
            # Restauration de l'état initial
            u.x.array[:] = u0.x.array[:]
            v.x.array[:] = v0.x.array[:]
            
            # Mise à jour basée sur les étapes précédentes
            for j in range(i):
                v.x.array[:] += tableau.a[i,j] * dt * k_stages[j].x.array[:]
            
            # Mise à jour du déplacement pour cette étape
            u.x.array[:] += tableau.c[i] * dt * v.x.array[:]
            
            # Application des conditions aux limites
            u.x.petsc_vec.ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)
            v.x.petsc_vec.ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)
            bcs.apply(u, v)
            
            # Calcul de l'accélération à cette étape
            self.acceleration_calculator(update_velocity=False)
            k_stages.append(a.copy())
        
        # Application de la solution finale
        u.x.array[:] = u0.x.array[:]
        v.x.array[:] = v0.x.array[:]
        
        for i in range(tableau.stages):
            v.x.array[:] += tableau.b[i] * dt * k_stages[i].x.array[:]
            
        u.x.array[:] += dt * v.x.array[:]
        
        # Application des conditions aux limites finales
        u.x.petsc_vec.ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)
        v.x.petsc_vec.ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)
        bcs.apply(u, v)
        
        return True