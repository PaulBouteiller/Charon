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
Generic Utility Functions Module
==============================

This module provides general-purpose utility functions that are used throughout
the CharonX framework. 
"""

from ufl import exp

def ppart(x):
    """
    Return the positive part of x: max(x, 0).
    
    Parameters
    ----------
    x : float, Expression, or Function Value to extract the positive part from
        
    Returns
    -------
    float, Expression, or Function Positive part of x, calculated as (x + |x|)/2
    """
    return (x + abs(x)) / 2

def npart(x):
    """
    Return the negative part of x: min(x, 0).
    
    Parameters
    ----------
    x : float, Expression, or Function
        Value to extract the negative part from
        
    Returns
    -------
    float, Expression, or Function Negative part of x, calculated as (x - |x|)/2
    """
    return (x - abs(x)) / 2

def Heav(x, eps=1e-3):
    """
    Return the Heaviside step function of x.
    
    Returns 1 if x >= 0, and 0 otherwise.
    
    Parameters
    ----------
    x : float, Expression, or Function
        Value to evaluate the Heaviside function at
    eps : float, optional
        Small value to avoid division by zero, by default 1e-3
        
    Returns
    -------
    float, Expression, or Function Heaviside function evaluated at x
    """
    return ppart(x) / (abs(x) + eps)

def smooth_shifted_heaviside(x, x_lim, width):
    """
    Create a smooth interpolation function between 0 and 1 around x_lim.
    
    This function creates a smooth transition for x, using a logistic function.
    
    Parameters
    ----------
    x : float, Expression, or Function
        Value to evaluate the smooth shifted Heaviside function at
    x_lim : float Central point of the transition
    width : float Width over which the function changes from 0.01 to 0.99
    """
    k = 9.19 / width  # Relation derived from 2*ln(99)/k = width
    return 1 / (1 + exp(-k * (x - x_lim)))

def over_relaxed_predictor(d, d_old, omega):
    """
    Apply over-relaxation to a predictor.
    
    Uses the PETSc axpy function: VecAXPY(Vec y, PetscScalar a, Vec x),
    which computes y = y + a * x
    
    Parameters
    ----------
    d : dolfinx.fem.Function
        Function to over-relax
    d_old : dolfinx.fem.Function
        Previous value of the function
    omega : float
        Over-relaxation parameter
        
    Returns
    -------
    dolfinx.fem.Function
        Over-relaxed function
    """
    d.x.petsc_vec.axpy(omega, d.x.petsc_vec - d_old.x.petsc_vec)