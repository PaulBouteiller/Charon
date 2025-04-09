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
from numpy import asarray, int32, zeros, arange
from mpi4py import MPI
import basix

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

def gather_function(u):
    """
    Rassemble les inconnus dans un unique vecteur sur le processus 0
    Parameters
    ----------
    u : Function.
    Returns
    -------
    global_array : np.array contenant la concaténation des inconnues portées
                    par les différents processus
    """
    dofmap = u.function_space.dofmap
    imap = dofmap.index_map
    local_range = asarray(imap.local_range, dtype = int32) * dofmap.index_map_bs
    size_global = imap.size_global * dofmap.index_map_bs
    return gather_vector(u.x.petsc_vec.array, local_range, size_global)
    
def gather_vector(local_vector, local_range, size):
    """
    Rassemble les inconnus dans un unique vecteur sur le processus 0
    Parameters
    ----------
    u : Function.
    Returns
    -------
    global_array : np.array contenant la concaténation des inconnues portées
                    par les différents processus
    """
    comm = MPI.COMM_WORLD
    ranges = comm.gather(local_range, root=0)
    data = comm.gather(local_vector, root=0)
    global_array = zeros(size)
    if comm.rank == 0:
        for r, d in zip(ranges, data):
            global_array[r[0]:r[1]] = d
        return global_array
    
def gather_coordinate(V):
    """
    Gathers mesh coordinates from all MPI processes into a single global array on rank 0.
    
    Parameters
    ----------
    V : FunctionSpace
        The function space whose mesh coordinates are to be gathered.
        Must be a vector-valued space for geometric coordinates.
    
    Returns
    -------
    numpy.ndarray or None
        On rank 0: Returns a (N, 3) array containing all mesh coordinates,
        where N is the global number of degrees of freedom.
        On other ranks: Returns None.
    
    Notes
    -----
    - Operates in parallel using MPI communication
    - Only rank 0 receives and returns the complete coordinate array
    - Assumes 3D coordinates (x, y, z)
    - Coordinates are ordered according to the global DOF numbering
    """
    
    comm = MPI.COMM_WORLD
    dofmap = V.dofmap
    imap = dofmap.index_map
    local_range = asarray(imap.local_range, dtype = int32) * dofmap.index_map_bs
    size_global = imap.size_global * dofmap.index_map_bs
    x = V.tabulate_dof_coordinates()[:imap.size_local]
    x_glob = comm.gather(x, root = 0)
    ranges = comm.gather(local_range, root=0)
    global_array = zeros((size_global,3))
    if comm.rank == 0:
        for r, d in zip(ranges, x_glob):
            global_array[r[0]:r[1], :] = d
        return global_array