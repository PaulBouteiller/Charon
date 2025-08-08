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
Desbiens Temperature-Based Reactive Burn Model Evolution Law
===========================================================

This module implements the Desbiens temperature-based reactive burn model for
high explosives, as described in "Modeling of the Jack Rabbit Series of Experiments
with a Temperature Based Reactive Burn Model" (2017).

The Desbiens model features a sophisticated 4-regime approach:
1. Initiation regime - Initial reaction driven by hot spots
2. Ignition-Growth regime - Nucleation and growth mechanisms
3. Diffusion-Growth regime - Thermal diffusion controlled growth
4. Burn regime - Final consumption phase

Key features:
- Temperature-dependent switching between regimes
- Shape functions for smooth transitions
- Shock temperature vs local temperature differentiation
- Physically-based rate expressions for each regime

Classes:
--------
Desbiens evolution : Desbiens temperature-based reactive evolution law
    Implements 4-regime explosive reaction kinetics
    Handles shock and local temperature effects
    Provides comprehensive explosive modeling capability

References:
-----------
- Desbiens, N. (2017). "Modeling of the Jack Rabbit Series of Experiments 
  with a Temperature Based Reactive Burn Model"
"""
from ufl import tanh as ufl_tanh, ln as ufl_ln, conditional
from .base_evolution import BaseEvolutionLaw


class DesbiensEvolution(BaseEvolutionLaw):
    """Desbiens temperature-based reactive burn model evolution law.
    
    Implementation of the comprehensive 4-regime explosive reaction model
    with temperature-dependent regime switching and shape functions.
    
    Attributes
    ----------
    Tadim : float Dimensioning temperature [K]
    Tall  : float Ignition threshold temperature [K]  
    Tc    : float Critical temperature for regime switching [K]
    T0    : float Reference temperature [K]
    kI, kIG, kDG, kB : float Rate constants for each regime [μs^-1]
    nI, nIG, nDG, nB : float Temperature exponents for each regime
    W1 : float Switching function parameter
    SI1, SI2 : float Initiation shape function parameters
    SG1, SG2 : float Growth shape function parameters
    """
    
    def required_parameters(self):
        """Return the list of required parameters.
        
        Returns
        -------
        list of str
            List of parameter names required for Desbiens model
        """
        return [
            "Tadim", "Tall", "Tc", "T0",
            "kI", "nI", "kIG", "nIG", "kDG", "nDG", "kB", "nB",
            "W1", "SI1", "SI2", "SG1", "SG2"
        ]
    
    def __init__(self, params):
        """Initialize the Desbiens evolution law.
        
        Parameters
        ----------
        params : dict
            Dictionary containing Desbiens model parameters.
            Can include default LX-17 parameters if not all specified.
        """
        # Default LX-17 parameters from Desbiens (2017) Table 1
        default_params = {
            'Tadim': 1035.0,     # K - Dimensioning temperature
            'Tall': 510.0,       # K - Ignition temperature
            'Tc': 1090.0,        # K - Critical temperature
            'T0': 293.0,         # K - Reference temperature
            
            'kI': 0.1e-6,        # μs^-1 - Initiation rate constant
            'nI': 1.0,           # Initiation exponent
            'kIG': 6.8e-6,       # μs^-1 - Ignition-growth rate constant
            'nIG': 1.5,          # Ignition-growth exponent
            'kDG': 120.0e-6,     # μs^-1 - Diffusion-growth rate constant
            'nDG': 0.5,          # Diffusion-growth exponent
            'kB': 0.7e-6,        # μs^-1 - Burn rate constant
            'nB': 1.0,           # Burn exponent
            
            'W1': 8.0,           # Switching function parameter
            'SI1': 200.0,        # Initiation shape parameter
            'SI2': 0.025,        # Initiation shape parameter
            'SG1': 40.0,         # Growth shape parameter
            'SG2': 0.835,        # Growth shape parameter
        }
        
        # Merge with provided parameters
        full_params = {**default_params, **params}
        super().__init__(full_params)
        
        # Store all parameters as attributes
        for key, value in full_params.items():
            setattr(self, key, value)
        
        # Log key parameters
        print("Desbiens Reactive Burn Model Parameters:")
        print(f"Dimensioning temperature: {self.Tadim} K")
        print(f"Ignition temperature: {self.Tall} K")
        print(f"Critical temperature: {self.Tc} K")
        print(f"Rate constants: kI={self.kI}, kIG={self.kIG}, kDG={self.kDG}, kB={self.kB} μs^-1")
    
    def setup_auxiliary_fields(self, V_c, **kwargs):
        """Setup auxiliary fields for Desbiens model.
        
        The model may track shock temperature separately from local temperature.
        
        Parameters
        ----------
        V_c : dolfinx.fem.FunctionSpace
            Function space for fields
        **kwargs : dict
            May contain 'T_shock' for shock temperature tracking
        """
        # Store shock temperature if provided
        self.T_shock = kwargs.get('T_shock', None)
    
    def switching_function_W(self, T_shock):
        """Switching function W(T_shock) for regime transition.
        
        Controls transition between ignition-growth and diffusion-growth regimes.
        
        Parameters
        ----------
        T_shock : ufl.Expression or dolfinx.fem.Function
            Shock temperature field
            
        Returns
        -------
        ufl.Expression
            Switching function value (0 to 1)
        """
        arg = self.W1 * (T_shock / self.Tc - 1.0)
        return 0.5 * (1.0 - ufl_tanh(arg))
    
    def shape_function_SI(self, lambda_burn):
        """Shape function SI(λ) for initiation regime.
        
        Controls the contribution of the initiation regime based on burn fraction.
        
        Parameters
        ----------
        lambda_burn : ufl.Expression or dolfinx.fem.Function Burn fraction (0 ≤ λ ≤ 1)
            
        Returns
        -------
        ufl.Expression Shape function value
        """
        return conditional(lambda_burn < self.SI2,
                          self.SI1 * lambda_burn,
                          self.SI1 * self.SI2 * (2.0 - lambda_burn / self.SI2))
    
    def shape_function_SG(self, lambda_burn):
        """Shape function SG(λ) for growth regimes.
        
        Controls the contribution of growth regimes vs burn regime.
        
        Parameters
        ----------
        lambda_burn : ufl.Expression or dolfinx.fem.Function Burn fraction (0 ≤ λ ≤ 1)
            
        Returns
        -------
        ufl.Expression Shape function value (0 to 1)
        """
        arg = self.SG1 * (lambda_burn - self.SG2)
        return 0.5 * (1.0 - ufl_tanh(arg))
    
    def rate_initiation(self, T_shock, lambda_burn):
        """Initiation rate rI based on shock temperature.
        
        Parameters
        ----------
        T_shock : ufl.Expression Shock temperature
        lambda_burn : ufl.Expression Current burn fraction
            
        Returns
        -------
        ufl.Expression Initiation rate
        """
        temp_ratio = conditional(T_shock > self.Tall,
                                (T_shock - self.Tall) / self.Tall,
                                0.0)
        
        return self.kI * (temp_ratio ** self.nI) * ((1.0 - lambda_burn) ** (2.0/3.0))
    
    def rate_ignition_growth(self, T_shock, lambda_burn):
        """Ignition-growth rate rIG based on shock temperature.
        
        Parameters
        ----------
        T_shock : ufl.Expression Shock temperature  
        lambda_burn : ufl.Expression Current burn fraction
            
        Returns
        -------
        ufl.Expression Ignition-growth rate
        """
        temp_ratio = conditional(T_shock > self.Tall,
                                (T_shock - self.Tall) / self.Tall,
                                0.0)
        
        # Protection against log(0) - use conditional
        log_term = conditional(lambda_burn < 0.9999,
                              -ufl_ln(1.0 - lambda_burn),
                              0.0)
        
        return self.kIG * (temp_ratio ** self.nIG) * (1.0 - lambda_burn) * (log_term ** (2.0/3.0))
    
    def rate_diffusion_growth(self, T, lambda_burn):
        """Diffusion-growth rate rDG based on local temperature.
        
        Parameters
        ----------
        T : ufl.Expression Local temperature
        lambda_burn : ufl.Expression Current burn fraction
            
        Returns
        -------
        ufl.Expression Diffusion-growth rate
        """
        temp_ratio = conditional(T > self.Tall,
                                (T - self.Tall) / self.Tadim,
                                0.0)
        
        return (self.kDG * (temp_ratio ** self.nDG) * 
                (lambda_burn ** (2.0/3.0)) * ((1.0 - lambda_burn) ** (2.0/3.0)))
    
    def rate_burn(self, T, lambda_burn):
        """Burn rate rB based on local temperature.
        
        Parameters
        ----------
        T : ufl.Expression Local temperature
        lambda_burn : ufl.Expression  Current burn fraction
            
        Returns
        -------
        ufl.Expression Burn rate
        """
        temp_ratio = conditional(T > self.Tall,
                                (T - self.Tall) / self.Tadim,
                                0.0)
        
        return self.kB * (temp_ratio ** self.nB) * ((1.0 - lambda_burn) ** 0.5)
    
    def compute_single_phase_rate(self, concentration, T, pressure, material, T_shock=None, **kwargs):
        """Compute Desbiens rate for single phase."""
        from ...utils.generic_functions import ppart
        
        if T_shock is None:
            T_shock = T
        
        # Individual rates
        rI = self.rate_initiation(T_shock, concentration)
        rIG = self.rate_ignition_growth(T_shock, concentration)
        rDG = self.rate_diffusion_growth(T, concentration)
        rB = self.rate_burn(T, concentration)
        
        # Shape and switching functions
        SI = self.shape_function_SI(concentration)
        SG = self.shape_function_SG(concentration)
        W = self.switching_function_W(T_shock)
        
        # Combined rate
        total_rate = (rI * SI + (rIG * W + rDG * (1.0 - W)) * SG + rB * (1.0 - SG))
        return ppart(total_rate)
    
    def update_auxiliary_fields(self, dt, **kwargs):
        """Update auxiliary fields.
        
        No auxiliary fields to update for basic Desbiens model.
        
        Parameters
        ----------
        dt : float Time step size
        **kwargs : dict Additional parameters
        """
        pass
    
    def get_auxiliary_fields(self):
        """Return auxiliary fields.
        
        Returns
        -------
        dict Dictionary with shock temperature if available
        """
        fields = {}
        if self.T_shock is not None:
            fields['T_shock'] = self.T_shock
        return fields
    
    def get_regime_contributions(self, concentrations, T, T_shock=None):
        """Get individual regime contributions for analysis.
        
        Parameters
        ----------
        concentrations : list of dolfinx.fem.Function Current concentrations
        T : dolfinx.fem.Function  Local temperature
        T_shock : dolfinx.fem.Function, optional Shock temperature
            
        Returns
        -------
        dict
            Dictionary with individual regime contributions
        """
        if len(concentrations) < 2:
            return {}
        
        if T_shock is None:
            T_shock = T
        
        lambda_burn = concentrations[1]
        
        # Individual rates
        rI = self.rate_initiation(T_shock, lambda_burn)
        rIG = self.rate_ignition_growth(T_shock, lambda_burn)
        rDG = self.rate_diffusion_growth(T, lambda_burn)
        rB = self.rate_burn(T, lambda_burn)
        
        # Functions
        SI = self.shape_function_SI(lambda_burn)
        SG = self.shape_function_SG(lambda_burn)
        W = self.switching_function_W(T_shock)
        
        # Effective contributions
        contrib_I = rI * SI
        contrib_IG = rIG * W * SG
        contrib_DG = rDG * (1.0 - W) * SG
        contrib_B = rB * (1.0 - SG)
        
        return {
            'initiation': contrib_I,
            'ignition_growth': contrib_IG,
            'diffusion_growth': contrib_DG,
            'burn': contrib_B,
            'total': contrib_I + contrib_IG + contrib_DG + contrib_B,
            'switching_W': W,
            'shape_SI': SI,
            'shape_SG': SG
        }