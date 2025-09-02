from Charon import Material, create_box, Tridimensional, Solve, MeshManager, build_transverse_isotropic_stiffness, compute_bulk_modulus
from mpi4py.MPI import COMM_WORLD
import numpy as np
from math import pi
from numpy import array
from dolfinx.mesh import compute_midpoints, meshtags

###### Modèle mécanique ######
rho0 = 1
C_mass = 1

# Paramètres du comportement déviatorique anisotrope
# T300/Époxy (composite standard)
EL = 181000    # MPa - direction fibres
ET = 10300     # MPa - direction transverse  
nuL = 0    # Poisson longitudinal-transverse
nuT = 0    # Poisson transverse-transverse
muL = 7170     # MPa - cisaillement longitudinal
C =  build_transverse_isotropic_stiffness(EL, ET, nuL, nuT, muL)

dev_type = "Anisotropic"
deviator_params = {"C" : C}
iso_T_K0 = compute_bulk_modulus(C)
T_dep_K0 = 0
iso_T_K1 = 1
T_dep_K1 = 0
eos_type = "Vinet"
dico_eos = {"iso_T_K0": iso_T_K0, "T_dep_K0" : T_dep_K0, "iso_T_K1": iso_T_K1, "T_dep_K1" : T_dep_K1}

Fibre = Material(rho0, C_mass, eos_type, dev_type, dico_eos, deviator_params)


#%%Maillage
Longueur, Largeur, hauteur = 2., 1., 1.
nb_el_z = 4
Nx, Ny, Nz = 2*nb_el_z, nb_el_z, nb_el_z
mesh = create_box(COMM_WORLD, [array([0, 0, 0]),  array([Longueur, Largeur, hauteur])],
                              [Nx, Ny, Nz])

# Obtenir centres des cellules
tdim = mesh.topology.dim
cells = np.arange(mesh.topology.index_map(tdim).size_local, dtype=np.int32)
midpoints = compute_midpoints(mesh, tdim, cells)

tags = np.where(midpoints[:, 0] < 1, 1, 2).astype(np.int32)#La où le x du centre de gravité est plus petit que 1 on affecte 1 sinon 2
cell_tags = meshtags(mesh, tdim, cells, tags)
dictionnaire_mesh = {"tags": [1, 2, 3, 4],
                     "coordinate": ["x", "y", "z", "x"], 
                     "positions": [0, 0, 0, Longueur],
                     "cell_tags" : cell_tags,
                     }
mesh_manager = MeshManager(mesh, dictionnaire_mesh)

###### Paramètre du problème ######
eps = 0.01
Umax = eps
chargement = {"type" : "rampe", "pente" : Umax}
dictionnaire = {"material" : Fibre,
                "mesh_manager" : mesh_manager,
                "boundary_conditions": 
                    [{"component": "Ux", "tag": 1},
                     {"component": "Uy", "tag": 2},
                     {"component": "Uz", "tag": 3},
                     {"component": "Ux", "tag": 4, "value": chargement}
                    ],
                "analysis" : "static",
                "isotherm" : True,
                "polycristal" : {"tags" : [1, 2], "angle" : [0, pi/2], "axis" : [[0, 0, 0], [0, 0, 1]]}
                }

pb = Tridimensional(dictionnaire)

dico_solve = {"Prefix" : "Traction_3D_polycristal", "output" : {"U" : True}}
solve_instance = Solve(pb, dico_solve, compteur=1, npas=10)
solve_instance.solve()
