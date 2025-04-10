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
Created on Wed Apr  9 09:41:08 2025

@author: bouteillerp
Interpolation Module
==================

This module provides utilities for interpolating values into finite element
functions. It includes functions for interpolating scalar values, expressions,
or other functions into target functions, as well as creating new functions
from expressions.

Key components:
- Interpolation of various value types into functions
- Creation of functions from expressions
- Batch interpolation for multiple functions
"""
from dolfinx.fem import Function, Expression
from ..utils.petsc_operations import petsc_assign

def interpolate_to_function(target_function, value):
    """
    Interpolate a value into a target function.
    
    Parameters
    ----------
    target_function : dolfinx.fem.Function Target function where the result will be stored
    value : float, Expression, or Function Value to interpolate
        
    Raises
    ------
    ValueError If the value type is not supported
    """
    if isinstance(value, float):
        target_function.x.array[:] = value
    elif isinstance(value, Expression):
        target_function.interpolate(value)
    elif isinstance(value, Function):
        petsc_assign(target_function, value)
    else:
        raise ValueError(f"Unsupported value type: {type(value)}")

def create_function_from_expression(V, expr):
    """
    Create a function from an expression.
    
    Parameters
    ----------
    V : dolfinx.fem.FunctionSpace Function space where the function will be created
    expr : dolfinx.fem.Expression Expression to interpolate
        
    Returns
    -------
    dolfinx.fem.Function
        New function containing the interpolated expression
    """
    func = Function(V)
    func.interpolate(expr)
    return func

def interpolate_multiple(target_functions, values):
    """
    Interpolate multiple values into multiple target functions.
    
    Parameters
    ----------
    target_functions : list of dolfinx.fem.Function
        Target functions where results will be stored
    values : list
        Values to interpolate (same length as target_functions)
        
    Raises
    ------
    ValueError If the lists don't have the same length
    """
    if len(target_functions) != len(values):
        raise ValueError("The function and value lists must have the same length")
    
    for func, value in zip(target_functions, values):
        interpolate_to_function(func, value)