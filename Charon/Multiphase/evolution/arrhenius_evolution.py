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
Arrhenius Kinetic Evolution Law
==============================

This module implements the classical Arrhenius temperature-dependent kinetics
for phase transformations. The Arrhenius equation is widely used to describe
the temperature dependence of reaction rates in chemical processes.

The model follows the standard form:
    k = A * exp(-E_a / (R * T))

where:
- A is the pre-exponential factor
- E_a is the activation energy
- R is the gas constant
- T is the absolute temperature

This evolution law is particularly suitable for:
- Temperature-driven phase transformations
- Chemical decomposition processes
- Solid-state reactions
- Crystallization phenomena

Classes:
--------
ArrheniusEvolution : Arrhenius kinetic evolution law
    Implements temperature-dependent reaction kinetics
    Handles sequential phase transformation chains
    Provides standard chemical kinetics framework
"""

from ufl import exp as ufl_exp
from .base_evolution import BaseEvolutionLaw


class ArrheniusEvolution(BaseEvolutionLaw):
    """Arrhenius kinetic evolution law.
    
    Standard temperature-dependent kinetics following the Arrhenius equation.
    Suitable for thermally activated phase transformations.
    
    Attributes
    ----------
    kin_pref : float Pre-exponential factor (kinetic prefactor) [s^-1]
    e_activation : float Activation energy [J/mol]
    R : float Universal gas constant [J/(mol·K)]
    """
    
    def required_parameters(self):
        """Return the list of required parameters.
        
        Returns
        -------
        list of str
            List of parameter names required for Arrhenius kinetics
        """
        return ["kin_pref", "e_activation"]
    
    def __init__(self, params):
        """Initialize the Arrhenius evolution law.
        
        Parameters
        ----------
        params : dict Dictionary containing:
                                    kin_pref : float Pre-exponential factor [s^-1]
                                    e_activation : float Activation energy [J/mol]
        """
        super().__init__(params)
        
        self.kin_pref = params["kin_pref"]
        self.e_activation = params["e_activation"]
        self.R = 8.3144  # Universal gas constant [J/(mol·K)]
        # Log parameters
        print(f"Arrhenius kinetic prefactor: {self.kin_pref} s^-1")
        print(f"Activation energy: {self.e_activation} J/mol")
        print(f"Activation temperature: {self.e_activation/self.R:.1f} K")
        
    def setup_auxiliary_fields(self, V_c, **kwargs):
        """Setup auxiliary fields for Arrhenius kinetics.
        
        No auxiliary fields are needed for the standard Arrhenius model.
        
        Parameters
        ----------
        V_c : dolfinx.fem.FunctionSpace Function space for concentration fields (unused)
        **kwargs : dict Additional setup parameters (unused)
        """
        pass  # No auxiliary fields needed for Arrhenius
    
    def compute_concentration_rates(self, concentrations, T, pressure, material, 
                                   phase_transitions, species_types, **kwargs):
        """Compute concentration rates using Arrhenius kinetics.
        
        The method handles different types of species (reactants, intermediates,
        products) in a reaction chain based on the species classification.
        
        Parameters
        ----------
        concentrations : list of dolfinx.fem.Function
            Current concentration fields for all phases
        T : dolfinx.fem.Function or ufl.Expression
            Current temperature field
        pressure : ufl.Expression
            Current pressure expression (unused for Arrhenius)
        material : Material
            Material object containing properties (unused for basic Arrhenius)
        phase_transitions : list of bool
            Boolean list indicating which phases can evolve
        species_types : dict
            Dictionary containing species classification:
            - 'reactifs': list of bool for reactant species
            - 'intermediaires': list of bool for intermediate species
            - 'produits_finaux': list of bool for final product species
            - 'inertes': list of bool for inert species
            
        Returns
        -------
        list of ufl.Expression
            List of concentration rate expressions dc/dt for each phase
        """
        nb_phase = len(concentrations)
        rates = [0] * nb_phase
        
        # Arrhenius rate expression: k = A * exp(-E_a / (R * T))
        arrhenius_rate = self.kin_pref * ufl_exp(-self.e_activation / (self.R * T))
        
        # Extract species type lists
        reactifs = species_types.get('reactifs', [False] * nb_phase)
        intermediaires = species_types.get('intermediaires', [False] * nb_phase)
        produits_finaux = species_types.get('produits_finaux', [False] * nb_phase)
        inertes = species_types.get('inertes', [False] * nb_phase)
        
        for i in range(nb_phase):
            # Skip inert species
            if inertes[i]:
                rates[i] = 0
                continue
                
            # Skip non-reactive species
            if not phase_transitions[i] and not (i > 0 and phase_transitions[i-1]):
                rates[i] = 0
                continue
            
            if reactifs[i]:
                # Reactant species: only consumption (-dc/dt)
                rates[i] = -concentrations[i] * arrhenius_rate
                
            elif intermediaires[i]:
                # Intermediate species: production from previous - consumption to next
                production = concentrations[i-1] * arrhenius_rate if i > 0 else 0
                consumption = concentrations[i] * arrhenius_rate if phase_transitions[i] else 0
                rates[i] = production - consumption
                
            elif produits_finaux[i]:
                # Final product species: only production (+dc/dt)
                rates[i] = concentrations[i-1] * arrhenius_rate if i > 0 else 0
        
        return rates
    
    def update_auxiliary_fields(self, dt, **kwargs):
        """Update auxiliary fields for the next time step.
        
        No auxiliary fields need updating for the Arrhenius model.
        
        Parameters
        ----------
        dt : float
            Time step size (unused)
        **kwargs : dict
            Additional update parameters (unused)
        """
        pass  # No auxiliary fields to update
    
    def get_auxiliary_fields(self):
        """Return dictionary of auxiliary fields.
        
        Returns
        -------
        dict
            Empty dictionary as no auxiliary fields are used
        """
        return {}