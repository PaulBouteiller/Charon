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

def petsc_add(x, y, new_vec = False):
    """ 
    Addition élément par élément de deux vecteurs via PETSc
    x : PETScVector
    y : PETScVector

    Return a PETSc Vec
    """
    if new_vec:
        xp = x.copy()
        xp.axpy(1, y)
        return xp
    else:
        return x.axpy(1, y)

def petsc_assign(x, y):
    """ 
    Pointwise assignation between two Functions using PETSc
    x : Function
    y : Function
    """
    x.x.petsc_vec.array[:] = y.x.petsc_vec.array

def dt_update(x, dot_x, dt, new_vec = False, method = "PETSc"):
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
        # if method = "PETSc":
        u.x.petsc_vec.axpy(dt, dot_x.x.petsc_vec)
        return u
    else:
        x.x.petsc_vec.axpy(dt, dot_x.x.petsc_vec)
        return
    
def dolfinx_dt_update(x, dot_x, dt, new_vec = False, method = "PETSc"):
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
        u.x.array(dt, dot_x.x.petsc_vec)
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
    
def slice_array(vecteur, quotient, reste):
  """Récupère tous les indices pairs d'un numpy array.

  Args:
    array: Le tableau numpy dont on veut récupérer les indices pairs.

  Returns:
    Un tableau contenant tous les indices pairs du tableau original.
  """
  return vecteur[arange(reste, len(vecteur), quotient)]

def set_quadrature(mesh, deg_quad):
    topo = mesh.topology
    basix_celltype = getattr(basix.CellType, topo.cell_types[0].name)
    quadrature_points, weights = basix.make_quadrature(basix_celltype, deg_quad)
    return quadrature_points

def set_cells(mesh, deg_quad):
    topo = mesh.topology
    map_c = topo.index_map(mesh.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    return arange(0, num_cells, dtype = int32)


def interpolate_quadrature(function, expr, mesh, cells):
    """
    Interpolate Expression into Function of Quadrature type

    Parameters
    ----------
    function : Function, function living in a quadrature space.
    expr : UFLExpression, expression UFL
    mesh : Mesh, maillage.
    cells : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    expr_eval = expr.eval(mesh, cells)
    function.x.array[:] = expr_eval.flatten()[:]
    
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