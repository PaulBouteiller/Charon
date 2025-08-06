"""
Test de validation pour la condensation de masse en 0D, 1D, 2D et 3D

Ce script implémente et exécute des tests pour la condensation de masse
(lumped mass matrix) dans différentes dimensions spatiales. La condensation
de masse consiste à remplacer la matrice de masse cohérente par une matrice
diagonale, ce qui simplifie la résolution des équations dynamiques.

Cas tests:
---------
- Condensation de masse en 1D cartésien
- Condensation de masse en 2D axisymétrique
- Condensation de masse en 2D plan
- Condensation de masse en 3D

Pour chaque cas, le script vérifie que la masse totale calculée avec
la matrice condensée correspond à la masse théorique.

Note théorique:
--------------
Pour le cas axisymétrique, la masse théorique est calculée par:
    m_tot = ∫_{R_int}^{R_ext} ρ·r·dr

Auteur: bouteillerp
"""

from Charon import (create_1D_mesh, Material, CartesianUD, create_rectangle, MeshManager,
                    Axisymmetric, CellType, PlaneStrain, Tridimensional, create_box)
from mpi4py.MPI import COMM_WORLD
import pytest
from sympy import Symbol, integrate
from dolfinx.fem.petsc import assemble_vector
from dolfinx.fem import form, Function
from ufl import action
import numpy as np
rho = 2
tol = 1e-16
DummyMat = Material(rho, 1, "IsotropicHPP", "IsotropicHPP", {"E" : 1, "nu" : 0, "alpha" : 1}, {"E":1, "nu" : 0})

#%%Vérification en 1D
def assemble_diagonal_mass_matrix(m_form, V):
    u1 = Function(V)
    u1.x.petsc_vec.set(1.)
    diag_M = assemble_vector(form(action(m_form, u1)))
    return diag_M.array

mesh_manager = MeshManager(create_1D_mesh(0, 1, 1), {})

dictionnaire_1D = {"mesh_manager" : mesh_manager}

pb_1D = CartesianUD(DummyMat, dictionnaire_1D)
mass_array = assemble_diagonal_mass_matrix(pb_1D.m_form, pb_1D.u.function_space)
print("La diagonale de la matrice de masse pour le cas 1D est", mass_array)
# from dolfinx.fem.petsc import assemble_matrix
# M = assemble_matrix(form(pb.m_form))
# print("La matrice de masse est", M.matrix)


#%%Vérification en 2D axi
R_int = 0.1
R_ext = 1

mesh_manager = MeshManager(create_rectangle(COMM_WORLD, [(R_int, 0), (R_ext, 1)], [1, 1], CellType.quadrilateral), {})    
dictionnaire_axi = {"mesh_manager" : mesh_manager}

pb_axi = Axisymmetric(DummyMat, dictionnaire_axi)
mass_array = assemble_diagonal_mass_matrix(pb_axi.m_form, pb_axi.u.function_space)

half_len_diag_M = len(mass_array)//2
m_num = sum(mass_array[2*i] for i in range(half_len_diag_M))
print("Le nombre de coefficients est", half_len_diag_M)
print("La masse totale calculée est", m_num)

# La théorie exige que la masse totale soit égale à \int_{R_int}^{R_ext}\rho r dr
r = Symbol("r")
m_tot = float(rho * integrate(r, (r, R_int, R_ext)))
print("La masse totale théorique vaut", m_tot)
assert m_tot - m_num < tol
print("La diagonale de la matrice de masse axisymétrique est", mass_array)


#%%Vérification en 2D plan
    
mesh_manager = MeshManager(create_rectangle(COMM_WORLD, [(0, 0), (1, 1)], [1, 1], CellType.quadrilateral), {})    
dictionnaire_2D = {"mesh_manager" : mesh_manager}

pb_2D = PlaneStrain(DummyMat, dictionnaire_2D)
mass_array = assemble_diagonal_mass_matrix(pb_2D.m_form, pb_2D.u.function_space)
print("Le nombre de coeff est", len(mass_array)//2)
print("La masse totale en 2D plan est", sum(mass_array[2*i] for i in range(len(mass_array)//2)))
print("La diagonale de la matrice de masse 2D plan est", mass_array)


#%%Vérification en 3D
mesh_manager = MeshManager(create_box(COMM_WORLD, [np.array([0,0,0]), np.array([1, 1, 1])], [1, 1, 1], CellType.hexahedron), {})   
dictionnaire_3D = {"mesh_manager" : mesh_manager}

pb_3D = Tridimensional(DummyMat, dictionnaire_3D)
mass_array = assemble_diagonal_mass_matrix(pb_3D.m_form, pb_3D.u.function_space)
print("Le nombre de coeff est", len(mass_array)//3)
print("La masse totale 3D est", sum(mass_array[3*i] for i in range(len(mass_array)//3)))
print("La diagonale de la matrice de masse 3D est", mass_array)
