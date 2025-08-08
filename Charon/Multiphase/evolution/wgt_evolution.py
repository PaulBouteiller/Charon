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
WGT Reactive Burn Model Evolution Law
====================================

This module implements the WGT (Ward-Ganguly-Thurston) mesoscale-informed reactive
burn model for explosive materials. The WGT model describes the reaction progress
in high explosives through a combination of diffusion-growth and burn regimes,
with smooth transitions between different reaction phases.

The model is particularly suited for:
- High explosive detonation modeling
- Reactive flow simulations  
- Blast wave calculations
- Deflagration-to-detonation transition studies

Key features:
- Temperature-dependent reaction rates
- Smooth shape functions for regime transitions
- Physically-based reaction kinetics
- Suitable for mesoscale explosive modeling

Classes:
--------
WGTEvolution : WGT reactive burn evolution law
    Implements temperature-dependent explosive reaction kinetics
    Handles diffusion-growth and burn regimes
    Provides smooth transitions between reaction phases

References:
-----------
- Ward, M.J., Ganguly, T., Thurston, G.B. "WGT: A mesoscale-informed reactive burn model."
  Journal of Applied Physics (specific reference to be added)
"""

from math import tanh, sqrt
from ufl import tanh as ufl_tanh, sqrt as ufl_sqrt
from .base_evolution import BaseEvolutionLaw


class WGTEvolution(BaseEvolutionLaw):
    """WGT reactive burn model evolution law.
    
    Implementation of the mesoscale-informed reactive burn model for
    explosive materials with temperature-dependent kinetics.
    
    Attributes
    ----------
    SG1 : float
        Shape function parameter controlling transition sharpness
    SG2 : float  
        Shape function parameter controlling transition position
    TALL : float
        Ignition temperature [K]
    TADIM : float
        Dimensioning temperature [K]
    KDG : float
        Diffusion-growth rate constant [s^-1]
    NDG : float
        Diffusion-growth temperature exponent
    KB : float
        Burn rate constant [s^-1]
    NB : float
        Burn temperature exponent
    """
    
    def required_parameters(self):
        """Return the list of required parameters.
        
        Returns
        -------
        list of str
            List of parameter names required for WGT model
        """
        return ["SG1", "SG2", "TALL", "TADIM", "KDG", "NDG", "KB", "NB"]
    
    def __init__(self, params):
        """Initialize the WGT evolution law.
        
        Parameters
        ----------
        params : dict
            Dictionary containing WGT model parameters:
            SG1 : float, optional
                Shape function sharpness parameter (default: 25)
            SG2 : float, optional  
                Shape function position parameter (default: 0.92)
            TALL : float, optional
                Ignition temperature in K (default: 400)
            TADIM : float, optional
                Dimensioning temperature in K (default: 1035)
            KDG : float, optional
                Diffusion-growth rate constant in s^-1 (default: 1.6e7)
            NDG : float, optional
                Diffusion-growth exponent (default: 2)
            KB : float, optional
                Burn rate constant in s^-1 (default: 2.8e5)  
            NB : float, optional
                Burn exponent (default: 1)
        """
        super().__init__(params)
        
        # Store parameters with default values
        self.SG1 = params.get("SG1", 25.0)
        self.SG2 = params.get("SG2", 0.92)
        self.TALL = params.get("TALL", 400.0)
        self.TADIM = params.get("TADIM", 1035.0)
        self.KDG = params.get("KDG", 1.6e7)
        self.NDG = params.get("NDG", 2.0)
        self.KB = params.get("KB", 2.8e5)
        self.NB = params.get("NB", 1.0)
        
        # Log parameters
        print("WGT Reactive Burn Model Parameters:")
        print(f"Shape function SG1: {self.SG1}")
        print(f"Shape function SG2: {self.SG2}")
        print(f"Ignition temperature TALL: {self.TALL} K")
        print(f"Dimensioning temperature TADIM: {self.TADIM} K")
        print(f"Diffusion-growth rate KDG: {self.KDG} s^-1")
        print(f"Diffusion-growth exponent NDG: {self.NDG}")
        print(f"Burn rate KB: {self.KB} s^-1")
        print(f"Burn exponent NB: {self.NB}")
    
    def setup_auxiliary_fields(self, V_c, **kwargs):
        """Setup auxiliary fields for WGT model.
        
        The WGT model does not require auxiliary fields beyond
        the standard concentration fields.
        
        Parameters
        ----------
        V_c : dolfinx.fem.FunctionSpace
            Function space for concentration fields (unused)
        **kwargs : dict
            Additional setup parameters (unused)
        """
        pass  # No auxiliary fields needed for WGT
    
    def shape_function(self, burn_fraction):
        """Calculate the shape function for regime transition.
        
        The shape function sf2 controls the smooth transition between
        diffusion-growth and burn regimes based on the burn fraction.
        
        Parameters
        ----------
        burn_fraction : ufl.Expression or dolfinx.fem.Function
            Current burn fraction (0 ≤ λ ≤ 1)
            
        Returns
        -------
        ufl.Expression
            Shape function value (0 to 1)
        """
        return 0.5 * (1 - ufl_tanh(self.SG1 * (burn_fraction - self.SG2)))
    
    def temperature_ratio(self, T):
        """Calculate normalized temperature ratio.
        
        Parameters
        ----------
        T : ufl.Expression or dolfinx.fem.Function
            Current temperature field
            
        Returns
        -------
        ufl.Expression  
            Normalized temperature ratio (T - T_all) / T_adim
        """
        return (T - self.TALL) / self.TADIM
    
    def diffusion_growth_rate(self, T, burn_fraction):
        """Calculate diffusion-growth reaction rate.
        
        This regime dominates at lower burn fractions and represents
        the initial phases of the explosive reaction.
        
        Parameters
        ----------
        T : ufl.Expression or dolfinx.fem.Function
            Temperature field
        burn_fraction : ufl.Expression or dolfinx.fem.Function
            Current burn fraction
            
        Returns
        -------
        ufl.Expression
            Diffusion-growth rate
        """
        from ...utils.generic_functions import ppart
        
        r_t = self.temperature_ratio(T)
        return self.KDG * ppart(r_t)**self.NDG * (1 - burn_fraction)
    
    def burn_rate(self, T, burn_fraction):
        """Calculate burn reaction rate.
        
        This regime dominates at higher burn fractions and represents
        the final consumption of the explosive material.
        
        Parameters
        ----------
        T : ufl.Expression or dolfinx.fem.Function
            Temperature field  
        burn_fraction : ufl.Expression or dolfinx.fem.Function
            Current burn fraction
            
        Returns
        -------
        ufl.Expression
            Burn rate
        """
        from ...utils.generic_functions import ppart
        
        r_t = self.temperature_ratio(T)
        return self.KB * ppart(r_t)**self.NB * ufl_sqrt(1 - burn_fraction)
    
    def compute_concentration_rates(self, concentrations, T, pressure, material,
                                   phase_transitions, species_types, **kwargs):
        """Compute WGT concentration rates.
        
        The WGT model typically describes binary explosive reactions
        (unreacted -> reacted) with smooth regime transitions.
        
        Parameters
        ----------
        concentrations : list of dolfinx.fem.Function
            Current concentration fields [c_unreacted, c_reacted, ...]
        T : dolfinx.fem.Function
            Current temperature field
        pressure : ufl.Expression
            Current pressure expression (unused for WGT)
        material : Material
            Material object (unused for basic WGT)
        phase_transitions : list of bool
            Phase transition flags
        species_types : dict
            Species classification (unused for WGT)
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        list of ufl.Expression
            Concentration rate expressions dc/dt for each phase
        """
        from ...utils.generic_functions import ppart
        
        nb_phase = len(concentrations)
        rates = [0] * nb_phase
        
        if nb_phase < 2:
            return rates
        
        # Get burn fraction (typically second concentration field)
        c_burn = concentrations[1]  # Burned/reacted fraction
        
        # Shape function for regime transition
        sf2 = self.shape_function(c_burn)
        
        # Temperature-dependent rates
        rate_dg = self.diffusion_growth_rate(T, c_burn)
        rate_b = self.burn_rate(T, c_burn)
        
        # Combined rate with smooth transition
        total_rate = ppart(sf2 * rate_dg + (1 - sf2) * rate_b)
        
        # Apply to concentration fields
        if phase_transitions[0]:  # Unreacted material can react
            rates[0] = -total_rate  # Reactant decreases
            
        if len(concentrations) > 1:  # Reacted material exists
            rates[1] = total_rate   # Product increases
            
        return rates
    
    def update_auxiliary_fields(self, dt, **kwargs):
        """Update auxiliary fields for the next time step.
        
        No auxiliary fields need updating for the WGT model.
        
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
    
    def get_reaction_progress(self, concentrations):
        """Get the overall reaction progress.
        
        Parameters
        ----------
        concentrations : list of dolfinx.fem.Function
            Current concentration fields
            
        Returns
        -------
        dolfinx.fem.Function or float
            Reaction progress (0 = no reaction, 1 = complete reaction)
        """
        if len(concentrations) < 2:
            return 0.0
        return concentrations[1]  # Burned fraction
    
    def get_regime_indicator(self, concentrations):
        """Get indicator of which regime dominates.
        
        Parameters
        ----------
        concentrations : list of dolfinx.fem.Function
            Current concentration fields
            
        Returns
        -------
        ufl.Expression
            Regime indicator (1 = diffusion-growth, 0 = burn)
        """
        if len(concentrations) < 2:
            return 0.0
        return self.shape_function(concentrations[1])
    
    def estimate_characteristic_time(self, T_reference, burn_fraction=0.1):
        """Estimate characteristic reaction time.
        
        Parameters
        ----------
        T_reference : float
            Reference temperature [K]
        burn_fraction : float, optional
            Reference burn fraction for estimation (default: 0.1)
            
        Returns
        -------
        float
            Characteristic time [s]
        """
        if T_reference <= self.TALL:
            return float('inf')  # No reaction below ignition temperature
        
        r_t = (T_reference - self.TALL) / self.TADIM
        
        # Use diffusion-growth rate as characteristic (dominant at low burn fractions)
        rate_dg = self.KDG * (r_t**self.NDG) * (1 - burn_fraction)
        
        return 1.0 / rate_dg if rate_dg > 0 else float('inf')