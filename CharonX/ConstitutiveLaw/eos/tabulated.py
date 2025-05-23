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
Created on Wed Apr  2 11:26:06 2025

@author: bouteillerp

Tabulated Equation of State Module
=================================

This module implements a tabulated equation of state approach, which uses interpolation
from pre-computed tables of pressure values rather than analytical formulas. This
approach is particularly useful for materials with complex behavior that doesn't
fit standard equations of state models, or when using experimentally measured data.

The implementation uses JAX for efficient interpolation and supports:
- 2D interpolation in temperature-volume space
- Multiple input options (DataFrame or raw arrays)
- Smooth interpolation between tabulated points
- Integration with the overall constitutive framework

Classes:
--------
TabulatedEOS : Tabulated equation of state
    Implements pressure calculation using 2D interpolation
    Supports various input formats
    Uses JAX for efficient computation
    
Helper Functions:
----------------
has_tabulated_eos : Boolean flag indicating JAX availability
optional_import : Utility for importing optional dependencies
Dataframe_to_array : Convert pandas DataFrame to arrays for tabulation
"""

import importlib.util
from dolfinx.fem import Function, Expression

def optional_import(module_name, as_name=None):
    """Attempt to import an optional module.
    
    Parameters
    ----------
    module_name : str Name of the module to import
    as_name : str, optional Name to import the module as
        
    Returns
    -------
    module or None The imported module or None if import failed
    """
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except:
        print(f"Warning: Optional module {module_name} not found. Some functionality may be limited.")
        return None

# Import optional modules
jax_numpy = optional_import("jax.numpy")
jax = optional_import("jax")

# Check if tabulated EOS is available
has_tabulated_eos = jax_numpy is not None and jax is not None
if not has_tabulated_eos:
    print("JAX not available: Tabulated equation of state functionality is disabled.")

from .base_eos import BaseEOS

def Dataframe_to_array(df):
    """Convert a pandas DataFrame to arrays for tabulated EOS.
    
    Parameters
    ----------
    df : pandas.DataFrame DataFrame with temperature rows, J columns, and pressure values
        
    Returns
    -------
    tuple (T_list, J_list, P_list) as numpy arrays
    """
    if not has_tabulated_eos:
        raise RuntimeError("JAX is required for tabulated EOS functionality")
        
    T_list = jax_numpy.array(df.index.values)
    J_list = jax_numpy.array(df.columns.astype(float).values)
    P_list = jax_numpy.array(df.values)
    return T_list, J_list, P_list

class TabulatedEOS(BaseEOS):
    """Tabulated equation of state using interpolation.
    
    This model uses pre-computed tables of pressure values for different
    temperature and deformation states.
    
    Note: Requires JAX for efficient interpolation.
    """
    
    def required_parameters(self):
        """Return the list of required parameters.
        
        Returns
        -------
        list List of parameter names
        """
        if not has_tabulated_eos:
            raise RuntimeError("Tabulated EOS requires JAX to be installed")
        
        return ["c0"] + (["Dataframe"] if "Dataframe" in self.params else ["T", "J", "P"])
    
    def __init__(self, params):
        """Initialize the tabulated EOS.
        
        Parameters
        ----------
        params : dict Parameters including either Dataframe (pandas DataFrame with tabulated data)
                        or T, J, P arrays for manual specification
        """
        if not has_tabulated_eos:
            raise RuntimeError("Tabulated EOS requires JAX to be installed")
        
        self.params = params  # Store for required_parameters method
        super().__init__(params)
        
        # Store wave speed
        self.c0 = params.get("c0")
        
        # Initialize tabulated data
        if "Dataframe" in params:
            self.T_list, self.J_list, self.P_list = Dataframe_to_array(params.get("Dataframe"))
        else:
            self.T_list = params.get("T")
            self.J_list = params.get("J")
            self.P_list = params.get("P")
        
        # Create interpolator
        self._setup_interpolator()
    
    def _setup_interpolator(self):
        """Set up the JAX interpolator for the tabulated data."""
        def find_index(x, xs):
            """Find the index in xs where x would be inserted.

            Parameters
            ----------
            x : float or array Value to search for
            xs : array Sorted array to search in
                
            Returns
            -------
            int or array Index where x would be inserted, clipped to valid range
            """
            return jax_numpy.clip(jax_numpy.searchsorted(xs, x, side='right') - 1, 0, len(xs) - 2)

        def interpolate_2d(x, y, x_grid, y_grid, values):
            """Perform bilinear interpolation.

            Parameters
            ----------
            x, y : float or array Coordinates to interpolate at
            x_grid, y_grid : array Grid coordinates
            values : 2D array Values at grid points
                
            Returns
            -------
            float or array Interpolated value
            """
            i = find_index(x, x_grid)
            j = find_index(y, y_grid)
            
            x1, x2 = x_grid[i], x_grid[i+1]
            y1, y2 = y_grid[j], y_grid[j+1]
            
            fx = (x - x1) / (x2 - x1)
            fy = (y - y1) / (y2 - y1)
            
            v11, v12 = values[i, j], values[i, j+1]
            v21, v22 = values[i+1, j], values[i+1, j+1]
            
            return (1-fx)*(1-fy)*v11 + fx*(1-fy)*v21 + (1-fx)*fy*v12 + fx*fy*v22

        def interpolate_jax(T, J):
            """Interpolate pressure at given T and J.

            Parameters
            ----------
            T : float or array Temperature
            J : float or array Jacobian (volumetric deformation)
                
            Returns
            -------
            float or array Interpolated pressure
            """
            return interpolate_2d(T, J, self.T_list, self.J_list, self.P_list)
        
        self._tabulated_interpolator = jax.jit(jax.vmap(interpolate_jax, in_axes=(0, 0)))
    
    def celerity(self, rho_0):
        """Return the specified wave speed.
        
        Parameters
        ----------
        rho_0 : float Initial density (unused, kept for interface consistency)
            
        Returns
        -------
        float Wave speed
        """
        return self.c0
        
    def pressure(self, J, T, T0, material, quadrature):
        """Calculate pressure using interpolation from tabulated data.
        
        J, T, T0, material : See stress_3D method in ConstitutiveLaw.py for details.
        quadrature : QuadratureHandler Handler for quadrature spaces.
            
        Returns
        -------
        ndarray Interpolated pressure values
        """
        V = quadrature.quadrature_space(["Scalar"])
        self.J_func = Function(V)
        self.J_expr = Expression(J, V.element.interpolation_points())
        self.J_func.interpolate(self.J_expr)
        self.T = T
        p = Function(V)
        pressures = self._tabulated_interpolator(jax_numpy.array(T.x.array), jax_numpy.array(self.J_func .x.array))
        p.x.array[:] = pressures
        return p
    
    def update_pressure(self):
        """Update pressure calculation with current values.

        Returns
        -------
        ndarray Updated pressure values
        """
        self.J_func.interpolate(self.J_expr)
        p = self._tabulated_interpolator(jax_numpy.array(self.T.x.array), jax_numpy.array(self.J_func.x.array))
        return p  