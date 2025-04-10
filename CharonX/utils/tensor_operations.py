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
Created on Tue Oct 15 15:32:00 2024

@author: bouteillerp
"""
from ufl import as_matrix, cos, sin
from ufl import dot as ufl_dot
from numpy import array, ndarray
from numpy import dot as np_dot


def rotation_matrix_direct(theta, axis):
    """
    Calculate the rotation matrix for rotating around an arbitrary axis by angle theta
    using direct computation of matrix elements.
    
    Parameters:
    -----------
    theta : float, Rotation angle in radians
    axis : array-like, 3D vector [x, y, z] representing the rotation axis. The 
                        array must be normalized
        
    Returns:
    --------
    List: 3x3 rotation matrix
    """
    x, y, z = axis[0], axis[1], axis[2] 

    c = cos(theta)
    s = sin(theta)
    C = 1 - c
    
    # Compute the matrix elements directly
    R = [[x*x*C + c,    x*y*C - z*s,  x*z*C + y*s],
        [y*x*C + z*s,  y*y*C + c,    y*z*C - x*s],
        [z*x*C - y*s,  z*y*C + x*s,  z*z*C + c]]
    
    return R

def stifness_rotation_matrix(Q):
    """
    Computes the 6x6 transformation matrix for rotating fourth-order stiffness tensors in Voigt notation.
    
    Parameters
    ----------
    Q : array-like or ufl.Matrix
        3x3 rotation matrix, can be either numpy array or UFL matrix
    
    Returns
    -------
    array-like or ufl.Matrix
        6x6 transformation matrix in Voigt notation. Returns numpy array if input is numpy array,
        otherwise returns UFL matrix.
    
    Notes
    -----
    The function constructs a transformation matrix that maps between different orientations
    of a stiffness tensor, following the Voigt notation convention:
    [11, 22, 33, 12, 23, 13] for both rows and columns.
    """
    if type(Q) == ndarray:
        poly_cristal = False
        Ql = Q.tolist()
    else:
        poly_cristal = True
    Q_sigma = [[1 for _ in range(6)] for _ in range(6)]
    for i in range(3):
        for j in range(3):
            Q_sigma[i][j] = Q[i][j]**2
    for i in range(3):
        Q_sigma[i][3] = 2 * Q[i][1]* Q[i][2]
        Q_sigma[i][4] = 2 * Q[i][2]* Q[i][0]
        Q_sigma[i][5] = 2 * Q[i][0]* Q[i][1]
    for j in range(3):
        Q_sigma[3][j] = Q[1][j]* Q[2][j]
        Q_sigma[4][j] = Q[2][j]* Q[0][j]
        Q_sigma[5][j] = Q[0][j]* Q[1][j]
    Q_sigma[3][3] = Q[1][1] * Q[2][2] + Q[1][2] * Q[2][1]
    Q_sigma[3][4] = Q[1][2] * Q[2][0] + Q[1][0] * Q[2][2]
    Q_sigma[3][5] = Q[1][0] * Q[2][1] + Q[1][1] * Q[2][0]
    
    Q_sigma[4][3] = Q[2][1] * Q[0][2] + Q[2][2] * Q[0][1]
    Q_sigma[4][4] = Q[2][2] * Q[0][0] + Q[2][0] * Q[0][2]
    Q_sigma[4][5] = Q[2][0] * Q[0][1] + Q[2][1] * Q[0][0]
    
    Q_sigma[5][3] = Q[0][1] * Q[1][2] + Q[0][2] * Q[1][1]
    Q_sigma[5][4] = Q[0][2] * Q[1][0] + Q[0][0] * Q[1][2]
    Q_sigma[5][5] = Q[0][0] * Q[1][1] + Q[0][1] * Q[0][1]
    if not poly_cristal:
        return array(Q_sigma)
    elif poly_cristal:
        return as_matrix(Q_sigma)
    

def rotate_stifness(stiff, Q):
    """
    Rotates a stiffness matrix using a given rotation matrix.
    
    Parameters
    ----------
    stiff : array-like or ufl.Matrix 6x6 stiffness matrix in Voigt notation
    Q : array-like or ufl.Matrix 3x3 rotation matrix
    
    Returns
    -------
    array-like or ufl.Matrix
        Rotated stiffness matrix: Q_sigma * stiff * Q_sigma^T
        Returns same type as input (numpy array or UFL matrix)
    """
    Q_sigma = stifness_rotation_matrix(Q)
    if type(Q_sigma) == ndarray:
        return np_dot(np_dot(Q_sigma, stiff), Q_sigma.T)
    else:
        Q_sig = as_matrix(Q_sigma)
        return ufl_dot(ufl_dot(Q_sig, as_matrix(stiff)), Q_sig.T)
    
def sym_mat_to_vec(mat):
    """
    Converts a symmetric 3x3 matrix to a vector in Voigt notation.
    
    Parameters
    ----------
    mat : array-like 3x3 symmetric matrix
    
    Returns
    -------
    list
        6-component vector in Voigt notation order: [11, 22, 33, 12, 13, 23]
    """
    return [mat[0,0], mat[1,1], mat[2,2], mat[0,1], mat[0,2], mat[1,2]]
    
def symetrized_tensor_product(S1, S2):
    """
    Computes the symmetrized tensor product of two symmetric tensors in Voigt notation.
    
    Parameters
    ----------
    S1 : array-like First 3x3 symmetric tensor
    S2 : array-like Second 3x3 symmetric tensor
    
    Returns
    -------
    ufl.Matrix
        6x6 matrix representing the symmetrized tensor product (S1⊗S2 + S2⊗S1)/2
        in Voigt notation
    """
    list1 = sym_mat_to_vec(S1)
    list2 = sym_mat_to_vec(S2)
    mat_tot = [[list1[i] * list2[j] for j in range(6)] for i in range(6)]
    ufl_mat = as_matrix(mat_tot)
    return ufl_mat + ufl_mat.T