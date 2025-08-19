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
Smooth Instantaneous Evolution Law
=================================

This module implements a smooth instantaneous phase transformation model based
on thermodynamic or mechanical triggers (e.g., density, pressure, temperature).
The model provides smooth transitions between phases using logistic functions.

This approach is particularly useful for:
- Phase transformations that occur rapidly compared to the simulation timescale
- Equilibrium-based phase transitions
- Density-driven transformations (e.g., shock-induced phase changes)
- Temperature-driven phase transitions with smooth transitions

Key features:
- Smooth Heaviside-like transitions
- Configurable transition width and center point
- Instantaneous equilibration assumption
- No auxiliary fields required

Classes:
--------
SmoothInstantaneousEvolution : Smooth instantaneous phase transition law
    Implements density/temperature-based instantaneous phase transitions
    Uses logistic functions for smooth transitions
    Suitable for equilibrium-based transformations
"""

from dolfinx.fem import Expression
from .base_evolution import BaseEvolutionLaw


class SmoothInstantaneousEvolution(BaseEvolutionLaw):
    """Smooth instantaneous evolution law.
    
    This model assumes that phase transformations occur instantaneously
    when a trigger condition (density, temperature, pressure) crosses
    a threshold, with smooth transitions provided by logistic functions.
    
    Attributes
    ----------
    trigger_variable : str   Variable that triggers the transformation ('rho', 'T', 'P')
    trigger_value    : float Critical value for transformation
    width            : float Width of the smooth transition region
    """
    
    def required_parameters(self):
        """Return the list of required parameters.
        
        Returns
        -------
        list of str List of parameter names required for smooth instantaneous evolution
        """
        return ["trigger_variable", "trigger_value", "width"]
    
    def __init__(self, params):
        """Initialize the smooth instantaneous evolution law.
        
        Parameters
        ----------
        params : dict
            Dictionary containing:
            trigger_variable : str
                Variable name that triggers transformation ('rho', 'T', 'P')
            trigger_value : float
                Critical value at which transformation occurs
            width : float
                Width parameter controlling smoothness of transition
        """
        super().__init__(params)
        
        self.trigger_variable = params["trigger_variable"]
        self.trigger_value = params["trigger_value"]
        self.width = params["width"]
        
        # Validate trigger variable
        valid_triggers = ['rho', 'T', 'P', 'density', 'temperature', 'pressure']
        if self.trigger_variable not in valid_triggers:
            raise ValueError(f"Invalid trigger variable: {self.trigger_variable}. "
                           f"Must be one of {valid_triggers}")
        
        # Log parameters
        print("Smooth instantaneous evolution parameters:")
        print(f"Trigger variable: {self.trigger_variable}")
        print(f"Trigger value: {self.trigger_value}")
        print(f"Transition width: {self.width}")
    
    def setup_auxiliary_fields(self, V_c, **kwargs):
        """Setup auxiliary fields and expressions.
        
        Creates the smooth transition expression based on the trigger variable.
        
        Parameters
        ----------
        V_c : dolfinx.fem.FunctionSpace Function space for concentration fields
        **kwargs : dict Must contain the trigger variable data (e.g., 'rho', 'T', 'P')
        """
        self.V_c = V_c
        
        # Get the trigger variable from kwargs
        trigger_data = None
        if self.trigger_variable in ['rho', 'density']:
            trigger_data = kwargs.get('rho') or kwargs.get('density')
        elif self.trigger_variable in ['T', 'temperature']:
            trigger_data = kwargs.get('T') or kwargs.get('temperature')  
        elif self.trigger_variable in ['P', 'pressure']:
            trigger_data = kwargs.get('P') or kwargs.get('pressure')
        
        if trigger_data is None:
            raise ValueError(f"Trigger variable '{self.trigger_variable}' not found in setup kwargs")
        
        # Create smooth transition expression
        from ...utils.generic_functions import smooth_shifted_heaviside
        
        self.transition_expr = smooth_shifted_heaviside(
            trigger_data, 
            self.trigger_value, 
            self.width
        )
        
        # Create Expression object for interpolation
        interp_points = V_c.element.interpolation_points()
        self.c_expr = Expression(self.transition_expr, interp_points)
        
        print(f"Smooth transition expression created for {self.trigger_variable}")
    
    def compute_concentration_rates(self, concentrations, T, pressure, material,
                                   phase_transitions, species_types, **kwargs):
        """Compute concentration rates for smooth instantaneous evolution.
        
        For instantaneous evolution, the rates are typically zero since
        the concentrations adjust instantly to equilibrium values.
        However, this method can be used to compute the equilibrium
        concentrations directly.
        
        Parameters
        ----------
        concentrations : list of dolfinx.fem.Function
            Current concentration fields (will be overwritten)
        T : dolfinx.fem.Function Temperature field
        pressure : ufl.Expression Pressure expression
        material : Material Material object
        phase_transitions : list of bool Phase transition flags
        species_types : dict Species classification
        **kwargs : dict Additional parameters
            
        Returns
        -------
        list of ufl.Expression
            Concentration rate expressions (typically zero for instantaneous)
        """
        nb_phase = len(concentrations)
        rates = [0] * nb_phase
        
        # For instantaneous evolution, rates are zero but concentrations
        # are set directly to equilibrium values
        if hasattr(self, 'c_expr') and nb_phase >= 2:
            # Update concentrations directly to equilibrium values
            concentrations[1].interpolate(self.c_expr)  # New phase
            
            # Conservation: c[0] = 1 - c[1] (for binary system)
            concentrations[0].x.array[:] = 1.0 - concentrations[1].x.array[:]
        
        return rates
    
    def update_auxiliary_fields(self, dt, **kwargs):
        """Update auxiliary fields.
        
        For smooth instantaneous evolution, concentrations are updated
        directly rather than through time integration.
        
        Parameters
        ----------
        dt : float Time step size (unused for instantaneous)
        **kwargs : dict May contain updated trigger variable data
        """
        # Update trigger variable if provided
        trigger_data = None
        if self.trigger_variable in ['rho', 'density']:
            trigger_data = kwargs.get('rho') or kwargs.get('density')
        elif self.trigger_variable in ['T', 'temperature']:
            trigger_data = kwargs.get('T') or kwargs.get('temperature')
        elif self.trigger_variable in ['P', 'pressure']:
            trigger_data = kwargs.get('P') or kwargs.get('pressure')
        
        if trigger_data is not None and hasattr(self, 'V_c'):
            # Update the transition expression
            from ...utils.generic_functions import smooth_shifted_heaviside
            
            self.transition_expr = smooth_shifted_heaviside(
                trigger_data,
                self.trigger_value,
                self.width
            )
            
            # Update Expression object
            interp_points = self.V_c.element.interpolation_points()
            self.c_expr = Expression(self.transition_expr, interp_points)
    
    def get_auxiliary_fields(self):
        """Return auxiliary fields.
        
        Returns
        -------
        dict Dictionary containing transition expression
        """
        fields = {}
        if hasattr(self, 'transition_expr'):
            fields['transition_expr'] = self.transition_expr
        if hasattr(self, 'c_expr'):
            fields['c_expr'] = self.c_expr
        return fields
    
    def get_equilibrium_concentrations(self, trigger_value_current):
        """Get equilibrium concentrations for given trigger value.
        
        Parameters
        ----------
        trigger_value_current : float Current value of the trigger variable
            
        Returns
        -------
        tuple (c_phase1, c_phase2) equilibrium concentrations
        """
        from math import tanh
        
        # Smooth Heaviside function
        arg = (trigger_value_current - self.trigger_value) / self.width
        c_phase2 = 0.5 * (1.0 + tanh(arg))  # New phase concentration
        c_phase1 = 1.0 - c_phase2           # Original phase concentration
        
        return c_phase1, c_phase2
    
    def get_transformation_progress(self, trigger_value_current):
        """Get transformation progress (0 to 1).
        
        Parameters
        ----------
        trigger_value_current : float Current value of the trigger variable
            
        Returns
        -------
        float Transformation progress (0 = no transformation, 1 = complete)
        """
        _, progress = self.get_equilibrium_concentrations(trigger_value_current)
        return progress
    
    def estimate_transition_range(self):
        """Estimate the range over which transition occurs.
        
        Returns
        -------
        tuple (start_value, end_value) for 1% to 99% transformation
        """
        from math import atanh
        
        # Values where transformation is 1% and 99% complete
        # tanh(x) = 0.98 => x = atanh(0.98) ≈ 2.65
        # tanh(x) = -0.98 => x = atanh(-0.98) ≈ -2.65
        
        start_value = self.trigger_value - 2.65 * self.width  # 1% transformation
        end_value = self.trigger_value + 2.65 * self.width    # 99% transformation
        
        return start_value, end_value
    
    def is_transformation_complete(self, trigger_value_current, tolerance=0.01):
        """Check if transformation is essentially complete.
        
        Parameters
        ----------
        trigger_value_current : float Current trigger variable value
        tolerance : float, optional Tolerance for considering transformation complete (default: 0.01)
            
        Returns
        -------
        bool True if transformation is essentially complete
        """
        progress = self.get_transformation_progress(trigger_value_current)
        return progress > (1.0 - tolerance)
    
    def update_concentrations_direct(self, concentrations, **kwargs):
        """Update concentrations directly based on current trigger value.
        
        This method bypasses the rate-based approach and sets concentrations
        directly to their equilibrium values.
        
        Parameters
        ----------
        concentrations : list of dolfinx.fem.Function
            Concentration fields to update
        **kwargs : dict
            Must contain current trigger variable data
        """
        if not hasattr(self, 'c_expr'):
            raise RuntimeError("Auxiliary fields not initialized. Call setup_auxiliary_fields first.")
        
        if len(concentrations) < 2:
            return
        
        # Update the expression with current trigger data
        self.update_auxiliary_fields(0.0, **kwargs)  # dt=0 for instantaneous
        
        # Set concentrations directly
        concentrations[1].interpolate(self.c_expr)  # New phase
        
        # Conservation for binary system
        concentrations[0].x.array[:] = 1.0 - concentrations[1].x.array[:]