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
from ufl import exp
from ..utils.interpolation import interpolate_multiple
from ..utils.generic_functions import smooth_shifted_heaviside

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
    def __init__(self, nb_phase, quadrature):
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
        self.multiphase_evolution = [False] * nb_phase
        self.explosive = False
        self.nb_phase = nb_phase
        self.set_multiphase_function(quadrature)
        
    def set_multiphase_function(self, quadrature):
        """
        Initialize the function spaces and fields for phase concentrations.
        
        Creates the function space and concentration fields for each phase,
        along with minimum and maximum concentration bounds.
        
        Parameters
        ----------
        quadrature : Quadrature Quadrature scheme for function spaces
        """
        self.V_c = quadrature.quadrature_space(["Scalar"])
        self.inf_c = Function(self.V_c)
        self.max_c = Function(self.V_c)
        self.max_c.x.petsc_vec.set(1.)
        self.inf_c.x.petsc_vec.set(0.)
        self.c = [Function(self.V_c, name="Current_concentration") for i in range(self.nb_phase)]
    
    def set_multiphase(self, expression_list):
        """
        Define the concentrations of different components.
        
        Parameters
        ----------
        expression_list : list Initial concentration expressions for each phase
        """
        interpolate_multiple(self.c, expression_list)
            
    def set_two_phase_explosive(self, E_vol):
        """
        Define the energy variation for the heat equation due to a
        change in the concentration of phase 1.
        
        Parameters
        ----------
        E_vol : float Volumetric energy released by the explosive
        """
        self.c_old = [c.copy() for c in self.c]   
        self.Delta_e_vol_chim = (self.c[1] - self.c_old[1]) * E_vol
        
    def set_evolution_parameters(self, params):
        """
        Unified method for configuring phase evolution.
        
        Parameters
        ----------
        params : dict Dictionary of evolution parameters
            
        Raises
        ------
        ValueError If an unknown evolution type is specified
        """
        if params.get("type") == "KJMA":
            self._set_KJMA_kinetic(
                params["rho"], 
                params["T"], 
                params["melt_param"], 
                params["gamma_param"], 
                params["alpha_param"],
                params["tau_param"]
            )
        elif params.get("type") == "smooth_instantaneous":
            self._set_smooth_instantaneous_evolution(
                params["rho"],
                params["rholim"],
                params["width"]
            )
        else:
            raise ValueError(f"Unknown evolution type: {params.get('type')}")
    
    def _set_KJMA_kinetic(self, rho, T, melt_param, gamma_param, alpha_param, tau_param):
        """
        Initialize functions needed for the KJMA kinetic model.
        
        This model describes phase transformations based on nucleation
        and growth processes, often used for crystallization phenomena.
        
        Parameters
        ----------
        rho : float or Function Current density field
        T : Function Current temperature field
        melt_param : list List containing two floats needed for the melting temperature definition
        gamma_param : float Speed of the liquid-solid interface as a function of temperature
        alpha_param : list List containing three parameters for the alpha field (nucleation rate)
        tau_param : list List containing three parameters for the tau field (induction time)
        """
        T_fusion = melt_param[0] * rho ** melt_param[1]
        self.gamma = - gamma_param * (T - T_fusion)
        self.alpha = exp(alpha_param[0] + alpha_param[1] * rho + alpha_param[2] * T)
        self.tau = exp(tau_param[0] + tau_param[1] * rho + tau_param[2] * T)
        self.U = Function(self.V_c)
        self.G = Function(self.V_c)
        self.J = Function(self.V_c)
        
    def _set_smooth_instantaneous_evolution(self, rho, rholim, width):
        """
        Create a smooth interpolation function between 0 and 1 around rholim.
        
        This function creates a smooth transition for phase concentration
        based on density, using a logistic function.
        
        Parameters
        ----------
        rho : float or numpy.array Density field
        rholim : float Central point of the transition
        width : float Width over which the function changes from 0.01 to 0.99
        """
        c_expr = smooth_shifted_heaviside(rho, rholim, width)
        self.c_expr = Expression(c_expr, self.V_c.element.interpolation_points())