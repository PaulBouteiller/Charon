#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 11:26:06 2025

@author: bouteillerp
"""
"""Tabulated equation of state using interpolation."""

import importlib.util

def optional_import(module_name, as_name=None):
    """Attempt to import an optional module.
    
    Parameters
    ----------
    module_name : str
        Name of the module to import
    as_name : str, optional
        Name to import the module as
        
    Returns
    -------
    module or None
        The imported module or None if import failed
    """
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except ImportError:
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
    tuple
        (T_list, J_list, P_list) as numpy arrays
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
        list
            List of parameter names
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
            return jax_numpy.clip(jax_numpy.searchsorted(xs, x, side='right') - 1, 0, len(xs) - 2)

        def interpolate_2d(x, y, x_grid, y_grid, values):
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
            return interpolate_2d(T, J, self.T_list, self.J_list, self.P_list)
        
        self.tabulated_interpolator = jax.jit(jax.vmap(interpolate_jax, in_axes=(0, 0)))
    
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
        
    def pressure(self, J, T, T0, material):
        """Calculate pressure using interpolation from tabulated data.
        
        Parameters
        ----------
        T : ndarray Temperature values
        J : ndarray Jacobian values
            
        Returns
        -------
        ndarray Interpolated pressure values
        """
        return self.tabulated_interpolator(jax_numpy.array(T.x.array), jax_numpy.array(J.x.array))