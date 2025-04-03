
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
from CharonX import *
import pytest
from sympy import Symbol, integrate
###### Modèle géométrique ######
rho = 2
tol = 1e-16
DummyMat = Material(rho, 1, "IsotropicHPP", "IsotropicHPP", {"E" : 1, "nu" : 0, "alpha" : 1}, {"E":1, "nu" : 0})

class Mesh_1D(CartesianUD):
    def __init__(self, material):
        CartesianUD.__init__(self, material)
          
    def define_mesh(self):
        return create_interval(MPI.COMM_WORLD, 1, [np.array(0), np.array(1)])

    def set_boundary(self):
        self.mesh_manager.mark_boundary([1, 2], ["x", "x"], [0, 1])
    
    def fem_parameters(self):
        self.u_deg =1
        self.schema = "default"

pb_1D = Mesh_1D(DummyMat)
quad_func = Function(pb_1D.V_quad_UD)
print("Longueur du vecteur quadrature", len(quad_func.x.array))
assert len(quad_func.x.array) == 1

class Mesh_2D_Plan(Plane_strain):
    def __init__(self, material):
        Plane_strain.__init__(self, material)
          
    def define_mesh(self):
        return create_rectangle(MPI.COMM_WORLD, [(0, 0), (1, 1)], [1, 1], CellType.quadrilateral) 
    
    def set_boundary(self):
        self.mesh_manager.mark_boundary([1, 2], ["x", "x"], [0, 1])

    def fem_parameters(self):
        self.u_deg = 2
        self.schema = "reduit"
  
pb_2D = Mesh_2D_Plan(DummyMat)
u = Function(pb_2D.V)
print("Nombre de noeuds", len(u.x.array)//2)
quad_func = Function(pb_2D.V_quad_UD)
print("Longueur du vecteur quadrature", len(quad_func.x.array))
assert len(quad_func.x.array) == 4
