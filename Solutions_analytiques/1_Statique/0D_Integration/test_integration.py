
"""
Test des schémas d'intégration numériques pour différentes dimensions.

Ce module vérifie le bon fonctionnement des points d'intégration (quadrature) pour les
éléments finis en 1D et 2D. Il teste notamment les dimensions des vecteurs de quadrature
pour s'assurer qu'ils sont correctement initialisés selon la dimension du problème.

Tests effectués:
    - Vérification de la taille du vecteur quadrature en 1D (doit être égal à 1 par élément)
    - Vérification de la taille du vecteur quadrature en 2D (doit être égal à 4 avec schéma réduit)

Auteur: bouteillerp
"""
from Charon import CartesianUD, Material, create_1D_mesh, CellType, Axisymmetric, create_rectangle, MeshManager
from mpi4py import MPI
from dolfinx.fem import Function
import pytest
import sys
sys.path.append("../../")
from Generic_isotropic_material import Acier
###### Modèle géométrique ######
rho = 2
tol = 1e-16
DummyMat = Material(rho, 1, "IsotropicHPP", "IsotropicHPP", {"E" : 1, "nu" : 0, "alpha" : 1}, {"E":1, "nu" : 0})

mesh = create_1D_mesh(0, 1, 1)
dictionnaire_mesh = {"tags": [1, 2], "coordinate": ["x", "x"], "positions": [0, 1]}
mesh_manager = MeshManager(mesh, dictionnaire_mesh)


dictionnaire_1D = {"mesh_manager" : mesh_manager,
                "boundary_setup": 
                    {"tags": [1, 2],
                     "coordinate": ["x", "x"], 
                     "positions": [0, 1]
                     },
                "analysis" : "static",
                "isotherm" : True
                }
pb_1D = CartesianUD(Acier, dictionnaire_1D)
quad_func = Function(pb_1D.V_quad_UD)
print("Longueur du vecteur quadrature", len(quad_func.x.array))
assert len(quad_func.x.array) == 1

mesh_2D = create_rectangle(MPI.COMM_WORLD, [(0, 0), (1, 1)], [1, 1], CellType.quadrilateral)
dictionnaire_mesh = {"tags": [1, 2], "coordinate": ["r", "r"], "positions": [0, 1],
                     "fem_parameters" : {"u_degree" : 2, "schema" : "reduit"}}
mesh_manager2D = MeshManager(mesh_2D, dictionnaire_mesh)

dictionnaire_2D = dictionnaire_1D
dictionnaire_2D["mesh_manager"] = mesh_manager2D
    
pb_2D = Axisymmetric(DummyMat, dictionnaire_2D)
u = Function(pb_2D.V)
print("Nombre de noeuds", len(u.x.array)//2)
quad_func = Function(pb_2D.V_quad_UD)
print("Longueur du vecteur quadrature", len(quad_func.x.array))
assert len(quad_func.x.array) == 4