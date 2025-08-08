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
====================================

This module provides tools for modeling materials with multiple phases, 
including phase concentration tracking and evolution over time using 
modular evolution laws.

The refactored design separates concerns:
- Multiphase class: manages concentration fields and species classification
- Evolution laws: handle the kinetic calculations (in separate evolution module)
- Solver: coordinates time integration (simplified MultiphaseSolver)

Key features:
- Modular evolution law system similar to EOS
- Automatic species classification (reactants, intermediates, products, inerts)
- Support for chemical energy release during transformations
- Integration with constitutive laws for mechanical behavior
- Clean separation between field management and kinetics

Classes:
--------
Multiphase : Main class for multiphase material management
    Manages concentration fields and species classification
    Handles chemical energy release calculations
    Provides interface to evolution laws
"""

from dolfinx.fem import Function, Expression
from ufl import conditional
from ..utils.interpolation import interpolate_multiple
from .evolution import ArrheniusEvolution, KJMAEvolution, WGTEvolution, DesbienasEvolution, SmoothInstantaneousEvolution


class Multiphase:
    """
    Class for handling materials with multiple phases using modular evolution laws.
    
    This refactored class focuses on managing concentration fields and species
    classification, while delegating kinetic calculations to separate evolution
    law classes.
    
    Attributes
    ----------
    nb_phase : int
        Number of phases being modeled
    c : list of dolfinx.fem.Function
        Concentration fields for each phase
    c_old : list of dolfinx.fem.Function
        Previous time step concentration fields
    evolution_laws : list of BaseEvolutionLaw
        Evolution law objects for each phase
    has_evolution : bool
        Flag indicating if any phases evolve
    species_types : dict
        Classification of species (reactants, intermediates, products, inerts)
    """
    
    def __init__(self, nb_phase, V_quad, multiphase_dictionary):
        """
        Initialize a multiphase object with modular evolution laws.
        
        Parameters
        ----------
        nb_phase : int
            Number of phases to model
        V_quad : dolfinx.fem.FunctionSpace
            Function space for concentration fields
        multiphase_dictionary : dict
            Configuration dictionary containing:
            - 'conditions': list of UFL expressions for initial concentrations
            - 'evolution_laws': list of evolution law configurations (optional)
            - 'chemical_energy_release': configuration for energy release (optional)
        """
        self.nb_phase = nb_phase
        self.V_c = V_quad
        
        # Initialize concentration fields
        self._setup_concentration_fields()
        self._set_initial_concentrations(multiphase_dictionary["conditions"])
        
        # Initialize evolution system if specified
        self.has_evolution = "evolution_laws" in multiphase_dictionary
        if self.has_evolution:
            self._setup_evolution_system(multiphase_dictionary["evolution_laws"])
            self._setup_species_classification()
            self._setup_chemical_energy_release(multiphase_dictionary.get("chemical_energy_release", {}))
        else:
            self.evolution_laws = [None] * nb_phase
            self.species_types = {
                'reactifs': [False] * nb_phase,
                'intermediaires': [False] * nb_phase,
                'produits_finaux': [False] * nb_phase,
                'inertes': [True] * nb_phase  # All inert if no evolution
            }
    
    def _setup_concentration_fields(self):
        """Initialize concentration function spaces and bounds."""
        self.inf_c = Function(self.V_c)
        self.max_c = Function(self.V_c)
        self.max_c.x.petsc_vec.set(1.0)
        self.inf_c.x.petsc_vec.set(0.0)
        self.c = [Function(self.V_c, name=f"c_{i}") for i in range(self.nb_phase)]
        self.c_old = [Function(self.V_c, name=f"c_old_{i}") for i in range(self.nb_phase)]
    
    def _set_initial_concentrations(self, conditions):
        """Set initial concentration distributions.
        
        Parameters
        ----------
        conditions : list List of UFL expressions for initial concentrations
        """
        if len(conditions) != self.nb_phase:
            raise ValueError(f"Number of conditions ({len(conditions)}) must match number of phases ({self.nb_phase})")
        
        # Convert conditions to UFL expressions and create Expression objects
        ufl_conditions = [conditional(condition, 1, 0) for condition in conditions]
        interp = self.V_c.element.interpolation_points()
        expression_list = [Expression(condition, interp) for condition in ufl_conditions]
        
        # Interpolate onto concentration fields and initialize old concentrations
        interpolate_multiple(self.c, expression_list)
        for i in range(self.nb_phase):
            self.c_old[i].x.array[:] = self.c[i].x.array[:]
    
    def _setup_evolution_system(self, evolution_config):
        """Setup evolution laws for each phase.
        
        Parameters
        ----------
        evolution_config : list
            List of evolution law configurations, one per phase
        """
        self.evolution_laws = []
        
        for i, config in enumerate(evolution_config):
            if config is None:
                # No evolution for this phase
                self.evolution_laws.append(None)
            else:
                # Create evolution law based on type
                evolution_law = self._create_evolution_law(config)
                self.evolution_laws.append(evolution_law)
        
        print(f"Initialized {len([law for law in self.evolution_laws if law is not None])} evolution laws")
    
    def _create_evolution_law(self, config):
        """Factory method to create evolution law objects.
        
        Parameters
        ----------
        config : dict
            Evolution law configuration containing 'type' and parameters
            
        Returns
        -------
        BaseEvolutionLaw
            Configured evolution law object
        """
        evolution_type = config["type"]
        params = config.get("params", {})
        
        if evolution_type == "Arrhenius":
            return ArrheniusEvolution(params)
        elif evolution_type == "KJMA":
            return KJMAEvolution(params)
        elif evolution_type == "WGT":
            return WGTEvolution(params)
        elif evolution_type == "Desbiens":
            return DesbienasEvolution(params)
        elif evolution_type == "SmoothInstantaneous":
            return SmoothInstantaneousEvolution(params)
        else:
            raise ValueError(f"Unknown evolution type: {evolution_type}")
    
    def _setup_species_classification(self):
        """Classify species based on their evolution laws.
        
        This method determines which species are reactants, intermediates,
        products, or inerts based on the chain of evolution laws.
        """
        # Create boolean array indicating which phases can evolve
        phase_transitions = [law is not None for law in self.evolution_laws]
        
        # Use existing classification logic
        self.species_types = self._classify_species(phase_transitions)
        
        # Store individual lists for backward compatibility
        self.reactifs = self.species_types['reactifs']
        self.intermediaires = self.species_types['intermediaires'] 
        self.produits_finaux = self.species_types['produits_finaux']
        self.inertes = self.species_types['inertes']
        
        # Log classification
        self._log_species_classification()
    
    def _classify_species(self, phase_transitions):
        """Classify species according to their roles in reactions.
        
        Parameters
        ----------
        phase_transitions : list of bool
            Boolean array indicating which phases can evolve
            
        Returns
        -------
        dict
            Dictionary with species classification lists
        """
        n = len(phase_transitions)
        
        # Initialize classification
        reactifs = [False] * n
        intermediaires = [False] * n
        produits_finaux = [False] * n
        inertes = [False] * n
        
        # Find last True position
        last_true_pos = -1
        for i in range(n-1, -1, -1):
            if phase_transitions[i]:
                last_true_pos = i
                break
        
        # Classify inert species (False at the end, after the last True)
        if last_true_pos != -1:
            for i in range(last_true_pos + 2, n):  # +2 to skip the final product
                inertes[i] = True
        else:
            # No evolution anywhere - all inert
            for i in range(n):
                inertes[i] = True
        
        # Classify active species (non-inert)
        for i in range(n):
            if inertes[i]:
                continue
            
            if phase_transitions[i]:  # Can evolve
                can_be_produced = (i > 0) and phase_transitions[i-1]
                
                if can_be_produced:
                    intermediaires[i] = True  # Can be produced AND disappear
                else:
                    reactifs[i] = True        # Reactant (start of chain)
            else:  # Cannot evolve
                produits_finaux[i] = True     # Final product
        
        return {
            'reactifs': reactifs,
            'intermediaires': intermediaires,
            'produits_finaux': produits_finaux,
            'inertes': inertes
        }
    
    def _log_species_classification(self):
        """Log the species classification for debugging."""
        print("\nSpecies classification:")
        for i in range(self.nb_phase):
            if self.reactifs[i]:
                print(f"Species {i+1}: REACTANT (transforms to {i+2})")
            elif self.intermediaires[i]:
                print(f"Species {i+1}: INTERMEDIATE (produced by {i}, transforms to {i+2})")
            elif self.produits_finaux[i]:
                print(f"Species {i+1}: FINAL PRODUCT (produced by {i})")
            elif self.inertes[i]:
                print(f"Species {i+1}: INERT (never involved in reactions)")
    
    def _setup_chemical_energy_release(self, energy_config):
        """Setup chemical energy release tracking.
        
        Parameters
        ----------
        energy_config : dict
            Configuration for chemical energy release
        """
        self.has_chemical_energy = bool(energy_config)
        
        if self.has_chemical_energy:
            self.volumic_energy_release = energy_config.get("volumic_energy_release", [0] * self.nb_phase)
            self.Delta_e_vol_chim = 0  # Will be computed during evolution
            print(f"Chemical energy release enabled: {self.volumic_energy_release}")
        else:
            self.volumic_energy_release = [0] * self.nb_phase
            self.Delta_e_vol_chim = 0
    
    def setup_evolution_auxiliary_fields(self, **kwargs):
        """Setup auxiliary fields for all evolution laws.
        
        Parameters
        ----------
        **kwargs : dict
            Setup parameters (T, rho, pressure, etc.)
        """
        if not self.has_evolution:
            return
        
        for i, evolution_law in enumerate(self.evolution_laws):
            if evolution_law is not None:
                evolution_law.setup_auxiliary_fields(self.V_c, **kwargs)
                print(f"Auxiliary fields setup for phase {i} evolution law")
    
    def compute_concentration_rates(self, T, pressure, material, **kwargs):
        if not self.has_evolution:
            return [0] * self.nb_phase
        
        # Get intrinsic rates
        phase_rates = []
        for i, evolution_law in enumerate(self.evolution_laws):
            if evolution_law is not None:
                rate = evolution_law.compute_single_phase_rate(self.c[i], T, pressure, material, **kwargs)
                phase_rates.append(rate)
            else:
                phase_rates.append(0)
        
        # Apply classification weights
        total_rates = [0] * self.nb_phase
        for i in range(self.nb_phase):
            if self.reactifs[i]:# Réactif : peut seulement diminuer (-1 * expression)
                total_rates[i] = -phase_rates[i]
            elif self.intermediaires[i]:# Intermédiaire : peut être produit (+1) ET disparaître (-1)
                production = phase_rates[i-1] if i > 0 else 0
                consumption = phase_rates[i]
                total_rates[i] = production - consumption
            elif self.produits_finaux[i]:# Produit final : peut seulement augmenter (+1 * expression précédente)
                total_rates[i] = phase_rates[i-1] if i > 0 else 0
        
        return total_rates
    
    def update_auxiliary_fields(self, dt, **kwargs):
        """Update auxiliary fields for all evolution laws.
        
        Parameters
        ----------
        dt : float
            Time step size
        **kwargs : dict
            Update parameters
        """
        if not self.has_evolution:
            return
        
        for evolution_law in self.evolution_laws:
            if evolution_law is not None:
                evolution_law.update_auxiliary_fields(dt, **kwargs)
    def get_auxiliary_fields(self):
        """Get auxiliary fields from all evolution laws.
        
        Returns
        -------
        dict
            Dictionary of auxiliary fields from all evolution laws
        """
        all_fields = {}
        
        if self.has_evolution:
            for i, evolution_law in enumerate(self.evolution_laws):
                if evolution_law is not None:
                    fields = evolution_law.get_auxiliary_fields()
                    for name, field in fields.items():
                        all_fields[f"phase_{i}_{name}"] = field
        
        return all_fields