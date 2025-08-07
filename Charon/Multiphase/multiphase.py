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
Multiphase Material Module
========================

This module provides tools for modeling materials with multiple phases, 
including phase concentration tracking and evolution over time.

The module implements various phase evolution models, such as the
Kolmogorov-Johnson-Mehl-Avrami (KJMA) kinetic model for phase transformations,
and supports explosive materials with energy release during phase changes.

Key features:
- Tracking of multiple material phases with concentration fields
- Phase evolution models including KJMA kinetics and smooth transitions
- Support for energetic materials with chemical energy release
- Integration with constitutive laws for mechanical behavior
- Temperature and density-dependent phase transformations
"""

from dolfinx.fem import Function, Expression
from ufl import conditional
from ..utils.interpolation import interpolate_multiple

class Multiphase:
    """
    Class for handling materials with multiple phases.
    
    This class manages the evolution and concentration fields of different
    material phases, supporting various phase transformation models and
    explosive materials with energy release.
    
    Attributes
    ----------
    multiphase_evolution : list of bool Flags indicating whether each phase evolves over time
    explosive : bool Flag indicating if the material is explosive
    nb_phase : int Number of phases being modeled
    c : list of dolfinx.fem.Function Concentration fields for each phase
    """
    def __init__(self, nb_phase, V_quad, multiphase_dictionnaire):
        """
        Initialize a multiphase object.
        
        The number of phases is determined by the size of the materials list.
        By default, phase concentrations remain fixed and materials are
        considered non-reactive.
        
        Parameters
        ----------
        nb_phase : int Number of phases to model
        quadrature : Quadrature Quadrature scheme for function spaces
        """
        self.nb_phase = nb_phase
        self._set_multiphase_function(V_quad)
        self._set_multiphase(multiphase_dictionnaire)
        self.multiphase_evolution = "phase_transition" in multiphase_dictionnaire
        if self.multiphase_evolution:
            self.phase_dictionnary = multiphase_dictionnaire["phase_transition"]
            self._set_chemical_energy_release(multiphase_dictionnaire)
            self.evolution_law = multiphase_dictionnaire["evolution_law"]
            self.reactifs, self.intermediaires, self.produits_finaux, self.inertes = self._species_classifier(self.phase_dictionnary)
            print("\nVérification - chaînes de réaction :")
            for i in range(len(self.phase_dictionnary)):
                if self.reactifs[i]:
                    print(f"Espèce {i+1}: RÉACTIF (se transforme en {i+2})")
                elif self.intermediaires[i]:
                    print(f"Espèce {i+1}: INTERMÉDIAIRE (produit par {i}, se transforme en {i+2})")
                elif self.produits_finaux[i]:
                    print(f"Espèce {i+1}: PRODUIT FINAL (produit par {i}, ne peut plus réagir)")
                elif self.inertes[i]:
                    print(f"Espèce {i+1}: INERTE (jamais impliqué dans les réactions)")
        
    def _set_multiphase_function(self, V_quad):
        """
        Initialize the function spaces and fields for phase concentrations.
        
        Creates the function space and concentration fields for each phase,
        along with minimum and maximum concentration bounds.
        
        Parameters
        ----------
        quadrature : Quadrature Quadrature scheme for function spaces
        """
        self.V_c = V_quad
        self.inf_c = Function(self.V_c)
        self.max_c = Function(self.V_c)
        self.max_c.x.petsc_vec.set(1.)
        self.inf_c.x.petsc_vec.set(0.)
        self.c = [Function(self.V_c, name="Current_concentration") for i in range(self.nb_phase)]
    
    def _set_multiphase(self, multiphase_dictionnaire):
        """
        Define the concentrations of different components.
        
        Parameters
        ----------
        expression_list : list Initial concentration expressions for each phase
        """
        conditions = multiphase_dictionnaire["conditions"]
        ufl_conditions = [conditional(condition, 1, 0) for condition in conditions]
        interp = self.V_c.element.interpolation_points()
        expression_list = [Expression(condition, interp) for condition in ufl_conditions]
        interpolate_multiple(self.c, expression_list)
        
    def _set_chemical_energy_release(self, dic):
        self.c_old = [c.copy() for c in self.c]
        self.Delta_e_vol_chim = 0
        range_list = range(self.nb_phase)
        for i, boolean, e_vol in zip(range_list, dic["phase_transition"], dic["volumic_energy_release"]):
            if boolean:
                self.Delta_e_vol_chim += (self.c[i] - self.c_old[i]) * e_vol
            
    def _species_classifier(self, tableau_bool):
        """
        Classifie les espèces chimiques selon leurs rôles dans les réactions.
        
        Règles:
        - True : peut évoluer vers l'espèce suivante
        - False : ne peut pas évoluer (produit final ou inerte)
        - Les inertes sont les False consécutifs à la fin
        - Pas de False au début
        
        Args:
            tableau_bool: Liste de booléens représentant les espèces
        
        Returns:
            tuple: (reactifs, intermediaires, produits_finaux, inertes)
        """
        n = len(tableau_bool)
        
        # Initialisation des listes
        reactifs = [False] * n
        intermediaires = [False] * n
        produits_finaux = [False] * n
        inertes = [False] * n
        
        # Identifier le dernier True
        derniere_position_true = -1
        for i in range(n-1, -1, -1):
            if tableau_bool[i]:  # Si on trouve un True
                derniere_position_true = i
                break
        
        # Les inertes sont les False à la fin, mais pas celui qui suit directement le dernier True
        if derniere_position_true != -1:
            # Le False juste après le dernier True est un produit final, pas un inerte
            for i in range(derniere_position_true + 2, n):  # +2 pour ignorer le produit final
                inertes[i] = True
        else:
            # Si aucun True, tous les False sont inertes
            for i in range(n):
                if not tableau_bool[i]:
                    inertes[i] = True
        
        # Maintenant classifier les espèces actives (non-inertes)
        for i in range(n):
            if inertes[i]:
                continue  # Skip les inertes
                
            if tableau_bool[i]:  # True - peut évoluer
                # Vérifier si elle peut être produite par l'espèce précédente
                peut_etre_produite = (i > 0) and tableau_bool[i-1]
                
                if peut_etre_produite:
                    intermediaires[i] = True  # Peut être produite ET disparaître
                else:
                    reactifs[i] = True        # Réactif (début de chaîne), peut seulement disparaître
                    
            else:  # False - ne peut pas évoluer
                # Si ce n'est pas un inerte, c'est forcément un produit final
                produits_finaux[i] = True
        
        return reactifs, intermediaires, produits_finaux, inertes