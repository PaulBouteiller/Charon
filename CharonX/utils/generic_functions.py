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
Created on Thu Jul 21 09:52:08 2022

@author: bouteillerp
"""
# from numpy import int32, arange
from mpi4py import MPI
# import basix

try:
    import __builtin__
except ImportError:
    import builtins as __builtin__
    
def print(*args, **kwargs):
    """ 
    Surcharge la version print de Python pour n'afficher qu'une seule
    fois la chaine de caractère demandé si l'on se trouve en MPI
    """
    if MPI.COMM_WORLD.Get_rank() == 0:
        __builtin__.print(*args, **kwargs)

def ppart(x):
    """
    Renvoie la partie positive de x

    Parameters
    ----------
    x : Float, Expression ou Fonction, grandeur dont on cherche la partie positive.
    """
    return (x + abs(x)) / 2

def npart(x):
    """
    Renvoie la partie positive de x

    Parameters
    ----------
    x : Float, Expression ou Fonction, grandeur dont on cherche la partie positive.
    """
    return (x - abs(x)) / 2

def Heav(x, eps = 1e-3):
    """
    Renvoie un Heaviside, si x\geq 0 alors Heav renvoie 1, 0 sinon

    Parameters
    ----------
    x : Float, Expression ou Fonction, grandeur dont on cherche le Heavyside.
    eps : TYPE, optional Paramètre numérique évitant les divisions par 0. The default is 1e-3.
    """
    return ppart(x) / (abs(x) + eps)

def over_relaxed_predictor(d, d_old, omega):
    """
    Sur-relaxation d'un prédicteur, utilisation de la fonction PETSc axpy:
    VecAXPY(Vec y,PetscScalar a,Vec x); return y = y + a ∗ x
    Parameters
    ----------
    d : Function, fonction à sur-relaxer
    d_old : Function, ancienne valeur de la fonction
    omega : Float, paramètre de sur-relaxation.

    Returns
    -------
    d : Function, fonction sur-relaxée
    """
    d.x.petsc_vec.axpy(omega, d.x.petsc_vec - d_old.x.petsc_vec)
    
# def slice_array(vecteur, quotient, reste):
#   """Récupère tous les indices pairs d'un numpy array.

#   Args:
#     array: Le tableau numpy dont on veut récupérer les indices pairs.

#   Returns:
#     Un tableau contenant tous les indices pairs du tableau original.
#   """
#   return vecteur[arange(reste, len(vecteur), quotient)]

# def set_quadrature(mesh, deg_quad):
#     topo = mesh.topology
#     basix_celltype = getattr(basix.CellType, topo.cell_types[0].name)
#     quadrature_points, weights = basix.make_quadrature(basix_celltype, deg_quad)
#     return quadrature_points

# def set_cells(mesh, deg_quad):
#     topo = mesh.topology
#     map_c = topo.index_map(mesh.topology.dim)
#     num_cells = map_c.size_local + map_c.num_ghosts
#     return arange(0, num_cells, dtype = int32)


# def interpolate_quadrature(function, expr, mesh, cells):
#     """
#     Interpolate Expression into Function of Quadrature type

#     Parameters
#     ----------
#     function : Function, function living in a quadrature space.
#     expr : UFLExpression, expression UFL
#     mesh : Mesh, maillage.
#     cells : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     None.

#     """
#     expr_eval = expr.eval(mesh, cells)
#     function.x.array[:] = expr_eval.flatten()[:]