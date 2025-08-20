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
Energy and Temperature Solvers  
==============================

Solvers for energy balance and heat transfer in thermomechanical problems.

This module provides solvers for temperature evolution under different
thermal conditions:
- Explicit adiabatic energy integration
- Implicit diffusion for heat conduction problems

Classes
-------
ExplicitEnergySolver Explicit adiabatic temperature evolution
DiffusionSolver  Implicit thermal diffusion solver

Notes
-----
The explicit solver uses Butcher tableau methods for temporal integration
while the diffusion solver employs implicit schemes for stability with
large conductivity values.
"""
from ..utils.default_parameters import default_energy_solver_order
from .explicit_butcher import ButcherIntegrator
from .hybrid_solver import create_linear_solver

from dolfinx.fem import Function, Expression
        
class ExplicitEnergySolver:
    def __init__(self, dt, T, C_tan, PVol):
        """Explicit solver for adiabatic energy evolution.
        
        Solves the energy balance equation under adiabatic conditions using
        explicit time integration with adaptive order Runge-Kutta methods.
        
        Parameters
        ----------
        dt : float Time step size
        T : Function  Current temperature field
        C_tan : float or Function Tangent volumetric heat capacity
        PVol : Function Volumetric power source term
        """
        self.dt = dt
        self.T = T
        self.dot_T = PVol / C_tan
        self.dot_T_expr = Expression(self.dot_T, self.T.function_space.element.interpolation_points())
        self.dot_T_func = Function(self.T.function_space)
        self.integrator = ButcherIntegrator(lambda: self.dot_T_expr)
        
    def energy_solve(self):
        """Explicit update of temperature field."""
        order = default_energy_solver_order()
        self.integrator.solve(order, self.T, self.dot_T_expr, self.dot_T_func, self.dt)
        
class DiffusionSolver:
    def __init__(self, dt, T, T_, dT, PVol, C_tan, flux_form, T_bcs, kinematic, dx):
        """Implicit solver for thermal diffusion problems.
        
        Solves the heat equation with conduction using backward Euler
        time integration for unconditional stability.
        
        Parameters
        ----------
        dt : float Time step size
        T : Function Current temperature field
        T_ : TestFunction Temperature test function
        dT : TrialFunction   Temperature trial function
        PVol : Function Volumetric heat source
        C_tan : float or Function Tangent volumetric heat capacity
        flux_form : Form Heat flux bilinear form
        T_bcs : DirichletBC Temperature boundary conditions
        kinematic : Kinematic Kinematic utilities object
        dx : Measure Integration measure
        """
        self.set_transient_thermal_form(dt, T, T_, dT, PVol, C_tan, flux_form, kinematic, dx)
        self.set_T_solver(T, T_bcs)
        
    def set_transient_thermal_form(self, dt, T, T_, dT, PVol, C_tan, flux_form, kin, dx):
        """Define bilinear and linear forms for diffusion problem solution.

        Parameters
        ----------
        dt : float Time step size
        T : Function Current temperature
        T_ : TestFunction Temperature test function
        dT : TrialFunction Temperature trial function
        PVol : Expression Volumetric power source
        C_tan : Expression Tangent heat capacity
        flux_form : Form Heat flux bilinear form
        kin : Kinematic Kinematic utilities
        dx : Measure Integration measure
        """
        self.bilinear_therm_form = kin.measure(C_tan * dT * T_ / dt, dx) + flux_form
        self.linear_therm_form = kin.measure((C_tan * T / dt + PVol) * T_,  dx)

    def set_T_solver(self, T, T_bcs):
        """Initialize solver for diffusion variational problem.
        
        Parameters
        ----------
        T : Function Temperature field
        T_bcs : list Temperature boundary conditions
        """
        self.problem_T = create_linear_solver(self.bilinear_therm_form, self.linear_therm_form, T, 
                                              bcs=T_bcs, solver_type="hybrid")
        
    def energy_solve(self):
        self.problem_T.solve()