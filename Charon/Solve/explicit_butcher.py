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
Unified Temporal Integrators Based on Butcher Tableaux
======================================================

Module for solving ordinary differential equations using Butcher tableau methods.

This module provides a unified framework for explicit Runge-Kutta methods using
Butcher tableaux representation, supporting orders 1-4 with various schemes.

Classes
-------
ButcherTableau
    Representation of a Butcher tableau for Runge-Kutta methods
ButcherIntegrator
    Temporal integrator based on Butcher tableaux for ODEs

Functions
---------
second_order_rk1, second_order_rk2, second_order_rk4
    Legacy functions for second-order ODEs (maintained for compatibility)

Notes
-----
The module supports explicit methods only. Implicit methods require
different solution strategies not implemented here.
"""

import numpy as np
from dolfinx.fem import Function
from ..utils.petsc_operations import dt_update, petsc_assign
from dolfinx.fem.petsc import set_bc

class ButcherTableau:
    """
    Représentation d'un tableau de Butcher pour les méthodes Runge-Kutta.
    
    Attributes
    ----------
    a : numpy.ndarray
        Matrice A du tableau de Butcher (coefficients des étapes intermédiaires)
    b : numpy.ndarray
        Vecteur b du tableau de Butcher (coefficients des poids)
    c : numpy.ndarray
        Vecteur c du tableau de Butcher (points d'évaluation)
    name : str
        Nom de la méthode
    description : str
        Description de la méthode
    order : int
        Ordre de précision de la méthode
    stages : int
        Nombre d'étapes de la méthode
    """
    
    def __init__(self, a, b, c, name="Custom", description="", order=0):
        """
        Initialise un tableau de Butcher.
        
        Parameters
        ----------
        a, b, c, name, description, order : see ButcherTable class description
        """
        self.a = np.array(a, dtype=float)
        self.b = np.array(b, dtype=float)
        self.c = np.array(c, dtype=float)
        self.name = name
        self.description = description
        self.order = order
        self.stages = len(b)
        
        # Vérification de la cohérence des dimensions
        if self.a.ndim == 1:
            # Convertir en matrice pour les tableaux de forme simplifiée
            self.a = np.array([self.a])
        
        if self.a.shape[0] != self.stages or (self.a.shape[1] != self.stages and self.a.ndim > 1):
            raise ValueError(f"La matrice a doit être de taille {self.stages}×{self.stages}")
        if len(self.c) != self.stages:
            raise ValueError(f"Le vecteur c doit être de longueur {self.stages}")
            
    def __repr__(self):
        return f"ButcherTableau(name='{self.name}', stages={self.stages}, order={self.order})"
    
    def is_explicit(self):
        """
        Vérifie si la méthode est explicite (a_ij = 0 pour j ≥ i).
        
        Returns
        -------
        bool True si la méthode est explicite, False sinon
        """
        if self.a.ndim == 1:
            return True  # Un vecteur a est toujours explicite
            
        for i in range(self.stages):
            for j in range(i, self.stages):
                if abs(self.a[i, j]) > 1e-14:
                    return False
        return True


class ButcherIntegrator:
    """
    Intégrateur temporel basé sur les tableaux de Butcher pour les équations différentielles ordinaires.
    """
    
    def __init__(self, derivative_calculator):
        """
        Initialise l'intégrateur temporel.
        
        Parameters
        ----------
        derivative_calculator : callable
            Fonction qui calcule la dérivée du champ, peut être une expression
            ou une fonction qui retourne une expression
        """
        self.derivative_calculator = derivative_calculator
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
        tableaux["ARKODE_RALSTON_3_1_2"] = ButcherTableau(
            a=[[0, 0, 0], 
               [2./3, 0, 0],
               [1./4, 3./4, 0]],
            b=[1./4, 3./4, 0],
            c=[0, 2./3, 1],
            name="Heun",
            description="Méthode de Heun d'ordre 2",
            order=2
        )
        # Méthode de Bogacki-Shampine (RK3)
        tableaux["BS3"] = ButcherTableau(
            a=[[0, 0, 0, 0],
               [1/2, 0, 0, 0],
               [0, 3/4, 0, 0],
               [2/9, 1/3, 4/9, 0]],
            b=[2/9, 1/3, 4/9, 0],
            c=[0, 1/2, 3/4, 1],
            name="Bogacki-Shampine",
            description="Méthode de Bogacki-Shampine d'ordre 3",
            order=3
        )
        
        # Méthode SOFRONIOU_SPALETTA
        tableaux["SOFRONIOU_SPALETTA"] = ButcherTableau(
            a=[
                [0, 0, 0, 0, 0],
                [2/5, 0, 0, 0, 0],
                [-3/20, 3/4, 0, 0, 0],
                [19/44, -15/44, 10/11, 0, 0],
                [11/72, 25/72, 25/72, 11/72, 0]],
            b=[11/72, 25/72, 25/72, 11/72, 0],
            c=[0, 2/5, 3/5, 1, 1],
            name="Sofroniou-Spaletta",
            description="Méthode de Sofroniou-Spaletta d'ordre 5(3)4",
            order=4
        )
        
        return tableaux
    
    def solve(self, order, primary_field, secondary_field, tertiary_field, 
              dt=1.0, bcs=None):
        """
        Résout une étape de temps avec la méthode RK spécifiée.
        
        Parameters
        ----------
        scheme_name : str
            Nom du schéma à utiliser (ex: "RK1", "RK2", "RK4")
        primary_field : Function
            Champ principal à mettre à jour (ex: température)
        secondary_field : Expression, optional
            Expression de la dérivée du champ
        tertiary_field : Function, optional
            Fonction pour stocker la dérivée calculée
        dt : float
            Pas de temps
        bcs : object, optional
            Conditions aux limites
            
        Returns
        -------
        bool
            True si succès, False sinon
        """
        scheme_map = {1: "RK1", 2: "ARKODE_RALSTON_3_1_2", 3 : "BS3", 4: "SOFRONIOU_SPALETTA"}
        scheme_name = scheme_map[order]
        if scheme_name not in self.tableaux:
            raise ValueError(f"Schéma '{scheme_name}' non disponible. Options: {list(self.tableaux.keys())}")
            
        tableau = self.tableaux[scheme_name]
        
        # Vérification que le tableau est explicite
        if not tableau.is_explicit():
            raise ValueError(f"Le tableau {tableau.name} n'est pas explicite, " 
                            "seules les méthodes explicites sont supportées")
        
        # Cas particulier: RK1 (Euler explicite) - optimisé
        if scheme_name == "RK1":
            tertiary_field.interpolate(secondary_field)
            dt_update(primary_field, tertiary_field, dt)
            self._apply_boundary_conditions(primary_field, bcs)
            return True
        
        # Sauvegarde de l'état initial
        y0 = primary_field.copy()
        
        # Allocation des fonctions temporaires pour les étapes
        V = primary_field.function_space
        k_stages = [Function(V) for _ in range(tableau.stages)]
        
        # Calcul des étapes intermédiaires
        for i in range(tableau.stages):
            # Configuration de l'état pour cette étape
            primary_field.x.array[:] = y0.x.array[:]
            
            for j in range(i):
                if tableau.a.ndim == 1:
                    a_ij = tableau.a[j] if j < len(tableau.a) else 0
                else:
                    a_ij = tableau.a[i, j]
                    
                if abs(a_ij) > 1e-14:
                    dt_update(primary_field, k_stages[j], dt * a_ij)
            
            # Application des conditions aux limites
            self._apply_boundary_conditions(primary_field, bcs)
            
            # Calcul de la dérivée à cette étape
            k_stages[i].interpolate(secondary_field)
        
        # Application de la solution finale
        primary_field.x.array[:] = y0.x.array[:]
        for i in range(tableau.stages):
            dt_update(primary_field, k_stages[i], dt * dt * tableau.b[i])
        
        # Application des conditions aux limites finales
        self._apply_boundary_conditions(primary_field, bcs)
        
        return True
    
    def _apply_boundary_conditions(self, field, bcs=None):
        """
        Applique les conditions aux limites à un champ.
        
        Parameters
        ----------
        field : Function Champ auquel appliquer les conditions aux limites
        bcs : object, optional Conditions aux limites
        """
        if bcs is None or field is None:
            return
        else:
            set_bc(field.x.petsc_vec, bcs)

def second_order_rk1(f, dot_f, ddot_f_function, ddot_f_expression, dt):
    """
    Schema d'Euler explicite pour EDO d'ordre 2, maintenu pour compatibilité.
    """
    ddot_f_function.interpolate(ddot_f_expression)
    dt_update(dot_f, ddot_f_function, dt)
    dt_update(f, dot_f, dt)  

def second_order_rk2(f, dot_f, ddot_f_function, ddot_f_expression, dt):
    """
    Schema RK2 pour EDO d'ordre 2, maintenu pour compatibilité.
    """
    ddot_f_function.interpolate(ddot_f_expression)
    dt_update(f, dot_f, dt/2)
    dt_update(dot_f, ddot_f_function, dt/2)
    
    ddot_f_function.interpolate(ddot_f_expression)
    dt_update(f, dot_f, dt/2)
    dt_update(f, ddot_f_function, dt**2/4)
    dt_update(dot_f, ddot_f_function, dt/2)

def second_order_rk4(f, dot_f, ddot_f_function, ddot_f_expression, dt):
    """
    Schema RK4 pour EDO d'ordre 2, maintenu pour compatibilité.
    """
    prev_f = f.copy()
    prev_dot_f = dot_f.copy()
    V_f = f.function_space
    
    ddot_f_function_1 = Function(V_f)
    ddot_f_function_2 = Function(V_f)
    ddot_f_function_3 = Function(V_f)
    ddot_f_function_4 = Function(V_f)
    
    ddot_f_function_1.interpolate(ddot_f_expression)
    dt_update(f, dot_f, dt/2)
    dt_update(dot_f, ddot_f_function_1, dt/2)
    
    ddot_f_function_2.interpolate(ddot_f_expression)
    dt_update(f, ddot_f_function_2, dt**2/4)
    petsc_assign(dot_f, dt_update(prev_dot_f, ddot_f_function_2, dt/2, new_vec=True))
    
    ddot_f_function_3.interpolate(ddot_f_expression)
    petsc_assign(f, dt_update(prev_f, prev_dot_f, dt, new_vec=True))
    dt_update(f, ddot_f_function_2, dt**2 / 2)
    petsc_assign(dot_f, dt_update(prev_dot_f, ddot_f_function_3, dt, new_vec=True))
    
    ddot_f_function_4.interpolate(ddot_f_expression)
    dt_update(prev_f, prev_dot_f, dt)
    petsc_assign(f, prev_f)
    dt_update(f, ddot_f_function_1, dt**2 / 6)
    dt_update(f, ddot_f_function_2, dt**2 / 6)
    dt_update(f, ddot_f_function_3, dt**2 / 6)    
    petsc_assign(dot_f, prev_dot_f)
    dt_update(dot_f, ddot_f_function_1, dt / 6)
    dt_update(dot_f, ddot_f_function_2, dt / 3)
    dt_update(dot_f, ddot_f_function_3, dt / 3)
    dt_update(dot_f, ddot_f_function_4, dt / 6)