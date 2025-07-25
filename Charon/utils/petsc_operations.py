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
PETSc Operations Module
=====================

This module provides utility functions for performing operations on PETSc
vectors and DOLFINx functions. These operations are important for efficient
manipulation of numerical data in the finite element framework.

Key components:
- Functions for setting bounds and corrections on vectors
- Vector operations (division, addition, assignment)
- Time integration utilities for explicit time stepping
"""

from petsc4py.PETSc import Vec

def set_correction(current, inf, maxi):
    """
    Constrain values to be between lower and upper bounds.
    
    Takes the maximum between the current function and the lower bound,
    then takes the minimum with the upper bound.

    Parameters
    ----------
    current : dolfinx.fem.Function Current state to be constrained
    inf : dolfinx.fem.Function Lower bound
    maxi : dolfinx.fem.Function Upper bound
    """
    set_min(current, inf)
    set_max(current, maxi)
    
def set_min(current, inf):
    """
    Set a lower bound on function values.
    
    Takes the maximum between the current function and the lower bound.

    Parameters
    ----------
    current : dolfinx.fem.Function Current state to be constrained
    inf : dolfinx.fem.Function Lower bound
    """
    current.x.petsc_vec.pointwiseMax(inf.x.petsc_vec, current.x.petsc_vec)
    
def set_max(current, maxi):
    """
    Set an upper bound on function values.
    
    Takes the minimum between the current function and the upper bound.

    Parameters
    ----------
    current : dolfinx.fem.Function Current state to be constrained
    maxi : dolfinx.fem.Function Upper bound
    """
    current.x.petsc_vec.pointwiseMin(maxi.x.petsc_vec, current.x.petsc_vec)

def petsc_div(numerateur, denominateur, output):
    """ 
    Element-wise division of two vectors using PETSc.
    
    The output vector is filled with the result x/y.

    Parameters
    ----------
    numerateur : petsc4py.PETSc.Vec Numerator vector
    denominateur : petsc4py.PETSc.Vec Denominator vector
    output : petsc4py.PETSc.Vec Output vector to store the result
    """
    output.pointwiseDivide(numerateur, denominateur)

def petsc_add(target_vec, source_vec, new_vec=False):
    """ 
    Element-wise addition of two vectors using PETSc.
    
    Parameters
    ----------
    target_vec : petsc4py.PETSc.Vec Target vector
    source_vec : petsc4py.PETSc.Vec Source vector to add
    new_vec : bool, optional Whether to create a new vector for the result, by default False
        
    Returns
    -------
    petsc4py.PETSc.Vec
        Result of the addition (new vector or modified target_vec)
    """
    if new_vec:
        xp = target_vec.copy()
        xp.axpy(1, source_vec)
        return xp
    else:
        return target_vec.axpy(1, source_vec)

def petsc_assign(target, source):
    """ 
    Element-wise assignment between two Functions using PETSc.
    
    Parameters
    ----------
    target : dolfinx.fem.Function Target function
    source : dolfinx.fem.Function Source function
    """
    Vec.copy(source.x.petsc_vec, target.x.petsc_vec)
    
def dt_update(x, dot_x, dt, new_vec=False):
    """
    Explicit update of x using its time derivative (Euler-explicit scheme).

    Parameters
    ----------
    x : dolfinx.fem.Function Function to update
    dot_x : dolfinx.fem.Function Time derivative of x
    dt : float Time step
    new_vec : bool, optional Whether to create a new vector for the result, by default False
        
    Returns
    -------
    dolfinx.fem.Function or None Updated function if new_vec is True, None otherwise
    """
    if new_vec:
        u = x.copy()
        u.x.petsc_vec.axpy(dt, dot_x.x.petsc_vec)
        return u
    else:
        x.x.petsc_vec.axpy(dt, dot_x.x.petsc_vec)
        return
    
def higher_order_dt_update(x, derivative_list, dt):
    """
    Explicit update of x using higher-order time derivatives.
    
    Implements a Taylor series expansion for more accurate time integration.

    Parameters
    ----------
    x : dolfinx.fem.Function Function to update
    derivative_list : list of dolfinx.fem.Function List containing successive time derivatives of x
    dt : float Time step
    """
    for k in range(len(derivative_list)):
        x.x.petsc_vec.axpy(dt**(k+1)/(k+1), derivative_list[k].x.petsc_vec)
    return