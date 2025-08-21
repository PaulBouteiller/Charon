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
Kolmogorov-Johnson-Mehl-Avrami (KJMA) Evolution Law
==================================================

This module implements the KJMA kinetic model for phase transformations based on
nucleation and growth processes. The KJMA model is particularly well-suited for
describing crystallization phenomena, solidification processes, and other
transformations involving nucleation and subsequent growth.

The KJMA model considers:
- Temperature and density-dependent nucleation rates
- Interface velocity between phases
- Geometric factors for growth morphology
- Induction time effects

Key features:
- Auxiliary field management (U, G, J fields)
- Temperature-dependent melting point
- Density-dependent kinetic parameters
- Higher-order time integration for auxiliary fields

Classes:
--------
KJMAEvolution : KJMA kinetic evolution law
    Implements nucleation and growth kinetics
    Manages auxiliary fields for geometric calculations
    Supports temperature and density-dependent parameters

References:
-----------
- Kolmogorov, A.N. (1937). "A statistical theory for the recrystallization of metals."
- Johnson, W.A., Mehl, R.F. (1939). "Reaction kinetics in processes of nucleation and growth."
- Avrami, M. (1939). "Kinetics of phase change I: General theory."


L'implémentation pratique est fondé sur: Coupling solidification kinetics with phase-behavior computations in hydrodynamic simulations of high-pressure, dynamic-compression processes'
"""

from math import pi, exp
from ufl import exp as ufl_exp
from dolfinx.fem import Function, Expression
from .base_evolution import BaseEvolutionLaw


class KJMAEvolution(BaseEvolutionLaw):
    """Kolmogorov-Johnson-Mehl-Avrami evolution law.
    
    Nucleation and growth kinetics for phase transformations with
    auxiliary field management for geometric calculations.
    
    Attributes
    ----------
    melt_param  : list of float Parameters [a, b] for melting temperature: T_melt = a * rho^b
    gamma_param : float         Interface velocity parameter
    alpha_param : list of float Parameters [a0, a1, a2] for nucleation rate: alpha = exp(a0 + a1*rho + a2*T)
    tau_param   : list of float Parameters [t0, t1, t2] for induction time: tau = exp(t0 + t1*rho + t2*T)
    """
    
    def required_parameters(self):
        """Return the list of required parameters.
        
        Returns
        -------
        list of str List of parameter names required for KJMA kinetics
        """
        return ["melt_param", "gamma_param", "alpha_param", "tau_param"]
    
    def __init__(self, params):
        """Initialize the KJMA evolution law.
        
        Parameters
        ----------
        params : dict Dictionary containing:
                        melt_param  : list of float Melting temperature parameters [a, b]
                        gamma_param : float         Interface velocity parameter
                        alpha_param : list of float Nucleation rate parameters [a0, a1, a2]
                        tau_param   : list of float Induction time parameters [t0, t1, t2]
        """
        super().__init__(params)
        
        self.melt_param = params["melt_param"]
        self.gamma_param = params["gamma_param"]
        self.alpha_param = params["alpha_param"]
        self.tau_param = params["tau_param"]
        
        # Log parameters
        print(f"KJMA melting temperature parameters: {self.melt_param}")
        print(f"Interface velocity parameter: {self.gamma_param}")
        print(f"Nucleation rate parameters: {self.alpha_param}")
        print(f"Induction time parameters: {self.tau_param}")
        
        # Initialize auxiliary fields (will be set in setup_auxiliary_fields)
        self.U = None
        self.G = None
        self.J = None
        
    def setup_auxiliary_fields(self, V_c, rho, T, **kwargs):
        """Setup KJMA auxiliary fields and temperature-dependent parameters.
        
        Parameters
        ----------
        V_c : dolfinx.fem.FunctionSpace Function space for concentration and auxiliary fields
        rho : dolfinx.fem.Function or ufl.Expression Current density field
        T : dolfinx.fem.Function or ufl.Expression Current temperature field
        **kwargs : dict Additional setup parameters
        """
        # Temperature-dependent melting point
        T_fusion = self.melt_param[0] * rho ** self.melt_param[1]
        
        # Interface velocity (negative for solidification)
        self.gamma = -self.gamma_param * (T - T_fusion)
        
        # Nucleation rate (exponential form)
        self.alpha = ufl_exp(self.alpha_param[0] + 
                            self.alpha_param[1] * rho + 
                            self.alpha_param[2] * T)
        
        # Induction time (exponential form)
        self.tau = ufl_exp(self.tau_param[0] + 
                          self.tau_param[1] * rho + 
                          self.tau_param[2] * T)
        
        # Initialize auxiliary fields
        self.U = Function(V_c, name="KJMA_U")
        self.G = Function(V_c, name="KJMA_G") 
        self.J = Function(V_c, name="KJMA_J")
        
        # Setup derivative expressions
        self._setup_derivative_expressions(V_c)
        
        print("KJMA auxiliary fields initialized successfully")
    
    def _setup_derivative_expressions(self, V_c):
        """Setup derivative expressions for KJMA auxiliary fields.
        
        The KJMA model requires tracking higher-order derivatives of
        auxiliary fields U, G, and J for accurate integration.
        
        Parameters
        ----------
        V_c : dolfinx.fem.FunctionSpace Function space for expressions
        """
        interp = V_c.element.interpolation_points()
        
        # U field derivatives (up to third order)
        self.dot_U = Function(V_c, name="dot_U")
        self.ddot_U = Function(V_c, name="ddot_U")
        self.dddot_U = Function(V_c, name="dddot_U")
        
        self.dot_U_expr = Expression(2 * self.gamma * self.G, interp)
        self.ddot_U_expr = Expression(2 * self.gamma**2 * self.J, interp)
        self.dddot_U_expr = Expression(self.gamma**2 * self.alpha, interp)
        
        # G field derivatives (up to second order)
        self.dot_G = Function(V_c, name="dot_G")
        self.ddot_G = Function(V_c, name="ddot_G")
        
        self.dot_G_expr = Expression(self.gamma * self.J, interp)
        self.ddot_G_expr = Expression(self.gamma * self.alpha, interp)
        
        # J field derivative (first order)
        self.dot_J = Function(V_c, name="dot_J")
        self.dot_J_expr = Expression(self.alpha, interp)
    
    def _compute_transient_alpha(self, t, n_lim=100):
        """Compute transient correction for nucleation rate.
        
        This implements the series expansion for transient nucleation effects.
        
        Parameters
        ----------
        t : float Current time
        n_lim : int, optional Maximum number of terms in series expansion
            
        Returns
        -------
        float Transient correction factor
        """
        S = 2.0
        for i in range(1, n_lim):
            S += 2 * ((-1)**i * exp(-i**2 * t / self.tau))
        return S
    
    def compute_concentration_rates(self, concentrations, T, pressure, material,
                                   phase_transitions, species_types, t=None, **kwargs):
        """Compute KJMA concentration rates.
        
        The KJMA model typically describes binary transformations (e.g., 
        liquid-solid) using auxiliary fields for geometric calculations.
        
        Parameters
        ----------
        concentrations : list of dolfinx.fem.Function Current concentration fields
        T              : dolfinx.fem.Function Current temperature field
        pressure       : ufl.Expression Current pressure expression (unused for KJMA)
        material       : Material Material object
        phase_transitions : list of bool Phase transition flags
        species_types  : dict Species classification
        t : float, optional Current time for transient effects
        **kwargs : dict Additional parameters
            
        Returns
        -------
        list of ufl.Expression Concentration rate expressions
        """
        nb_phase = len(concentrations)
        rates = [0] * nb_phase
        
        if nb_phase < 2:
            return rates
        
        # Apply transient correction if time is provided
        alpha_effective = self.alpha
        if t is not None:
            transient_factor = self._compute_transient_alpha(t)
            alpha_effective = self.alpha * transient_factor
        
        # KJMA transformation rate (typically for phase 0 -> phase 1)
        V_c = concentrations[0].function_space
        interp = V_c.element.interpolation_points()
        
        # Main KJMA rate expression: 4π * (1-c₁) * γ * U
        # This represents the volumetric transformation rate
        dot_c_expr = Expression(4 * pi * (1 - concentrations[1]) * self.gamma * self.U, interp)
        
        # Apply to appropriate phases
        if phase_transitions[0]:  # Phase 0 can transform
            rates[0] = -dot_c_expr  # Phase 0 decreases
            
        if len(concentrations) > 1:  # Phase 1 exists
            rates[1] = dot_c_expr   # Phase 1 increases
        
        return rates
    
    def update_auxiliary_fields(self, dt, **kwargs):
        """Update KJMA auxiliary fields using higher-order integration.
        
        The KJMA model requires careful integration of auxiliary fields
        to maintain accuracy of the geometric calculations.
        
        Parameters
        ----------
        dt : float Time step size
        **kwargs : dict Additional update parameters
        """
        if self.U is None or self.G is None or self.J is None:
            raise RuntimeError("Auxiliary fields not initialized. Call setup_auxiliary_fields first.")
        
        # Import update utilities
        try:
            from ...utils.petsc_operations import higher_order_dt_update, dt_update
        except ImportError:
            from ..utils.petsc_operations import higher_order_dt_update, dt_update
        
        # Interpolate derivative expressions
        self.dot_U.interpolate(self.dot_U_expr)
        self.ddot_U.interpolate(self.ddot_U_expr)
        self.dddot_U.interpolate(self.dddot_U_expr)
        self.dot_G.interpolate(self.dot_G_expr)
        self.ddot_G.interpolate(self.ddot_G_expr)
        self.dot_J.interpolate(self.dot_J_expr)
        
        # Update auxiliary fields with appropriate order
        higher_order_dt_update(self.U, [self.dot_U, self.ddot_U, self.dddot_U], dt)
        higher_order_dt_update(self.G, [self.dot_G, self.ddot_G], dt)
        dt_update(self.J, self.dot_J, dt)
    
    def get_auxiliary_fields(self):
        """Return dictionary of auxiliary fields.
        
        Returns
        -------
        dict Dictionary containing auxiliary fields and parameters
        """
        return {
            'U': self.U,
            'G': self.G,
            'J': self.J,
            'gamma': self.gamma,
            'alpha': self.alpha,
            'tau': self.tau,
            'derivatives': {
                'dot_U': self.dot_U,
                'ddot_U': self.ddot_U,
                'dddot_U': self.dddot_U,
                'dot_G': self.dot_G,
                'ddot_G': self.ddot_G,
                'dot_J': self.dot_J
            }
        }