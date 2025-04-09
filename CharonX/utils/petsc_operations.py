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
from mpi4py import MPI
from petsc4py.PETSc import Vec
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

def set_correction(current, inf, maxi):
    """
    Maximum entre la fonction current et la fonction prev puis minimum avec 
    la fonction maximum.

    Parameters
    ----------
    current : Function, état courant.
    prev : Function, borne inférieure.
    maxi : Function, borne supérieure.
    """
    current.x.petsc_vec.pointwiseMax(inf.x.petsc_vec, current.x.petsc_vec)
    current.x.petsc_vec.pointwiseMin(maxi.x.petsc_vec, current.x.petsc_vec)
    
def set_min(current, inf):
    """
    Maximum entre la fonction current et la fonction prev.

    Parameters
    ----------
    current : Function, état courant.
    prev : Function, borne inférieure.
    """
    current.x.petsc_vec.pointwiseMax(inf.x.petsc_vec, current.x.petsc_vec)
    
def set_max(current, maxi):
    """
    Minimum entre la fonction current et la fonction prev.

    Parameters
    ----------
    current : Function, état courant.
    maxi : Function, borne supérieure.
    """
    current.x.petsc_vec.pointwiseMin(maxi.x.petsc_vec, current.x.petsc_vec)

def petsc_div(numerateur, denominateur, output):
    """ 
    Division élément par élément de deux vecteurs via PETSc. Le vecteur
    output est rempli avec le résultat x/y

    numerateur : PETScVector
    denominateur : PETScVector
    output : PETScVector
    """
    output.pointwiseDivide(numerateur, denominateur)

def petsc_add(target_vec, source_vec, new_vec = False):
    """ 
    Addition élément par élément de deux vecteurs via PETSc
    x : PETScVector
    y : PETScVector

    Return a PETSc Vec
    """
    if new_vec:
        xp = target_vec.copy()
        xp.axpy(1, source_vec)
        return xp
    else:
        return target_vec.axpy(1, source_vec)

def petsc_assign(target, source):
    """ 
    Pointwise assignation between two Functions using PETSc
    x : Function
    y : Function
    """
    Vec.copy(source.x.petsc_vec, target.x.petsc_vec)
    
def dt_update(x, dot_x, dt, new_vec = False):
    """
    Mise jour explicite de x en utilisant sa dérivée = schéma de Euler-explicite

    Parameters
    ----------
    x : Function, fonction à mettre à jour.
    dot_x : Function, dérivée temporelle de x.
    dt : Float, pas de temps temporel.
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
    Mise jour explicite de x en utilisant ses dérivées d'ordres supérieurs.

    Parameters
    ----------
    x : Function, fonction à mettre à jour.
    derivative_list : List, liste contenant les dérivées temporelles successives de x.
    dt : Float, pas de temps temporel.
    """
    for k in range(len(derivative_list)):
        x.x.petsc_vec.axpy(dt**(k+1)/(k+1), derivative_list[k].x.petsc_vec)
    return