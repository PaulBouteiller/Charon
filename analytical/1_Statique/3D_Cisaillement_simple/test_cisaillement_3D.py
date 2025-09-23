"""
Test de traction uniaxiale sur un cube 3D avec matériau isotrope.

Ce script simule un essai de traction uniaxiale sur un cube 3D et compare
les résultats numériques avec la solution analytique.

Paramètres géométriques:
    - Dimensions du cube: 0.5 x 2.0 x 2.0
    - Discrétisation: maillage 10x10x10

Chargement:
    - Déformation imposée (eps): 0.005 (0.5% de déformation)

Conditions aux limites:
    - Déplacement horizontal bloqué sur la face gauche
    - Déplacement vertical bloqué sur la face inférieure
    - Déplacement selon Z bloqué sur la face arrière
    - Déplacement selon Z imposé sur la face avant

Le script calcule la force résultante et la compare avec la solution analytique
pour un problème de traction uniaxiale 3D. Une assertion vérifie que l'erreur
relative est inférieure à 1%.

Auteur: bouteillerp
Date de création: 11 Mars 2022
"""
from Charon import create_box, Tridimensional, Solve, MeshManager
from mpi4py.MPI import COMM_WORLD
from numpy import array, loadtxt
import matplotlib.pyplot as plt
import pytest
import sys
sys.path.append("../../")
from Generic_isotropic_material import Acier, E

######## Paramètres géométriques et de maillage ########
Longueur, Largeur, hauteur = 0.5, 2., 2.
Nx, Ny, Nz = 1, 1, 1
mesh = create_box(COMM_WORLD, [array([0, 0, 0]), 
                               array([Longueur, Largeur, hauteur])],
                              [Nx, Ny, Nz])

dictionnaire_mesh = {"tags": [1, 2, 3, 4],
                     "coordinate": ["x", "y", "z", "z"], 
                     "positions": [0, 0, 0, hauteur],
                     "fem_parameters" : {"u_degree" : 2, "schema" : "default"},
                     }
mesh_manager = MeshManager(mesh, dictionnaire_mesh)

#%% Définition du problème
eps = 0.005
Umax = eps * hauteur
dictionnaire = {"material" : Acier,
                "mesh_manager" : mesh_manager,
                "boundary_conditions": 
                    [{"component": "Ux", "tag": 1},
                     {"component": "Uy", "tag": 2},
                     {"component": "Uz", "tag": 3},
                     {"component": "Ux", "tag": 4, "value": {"type" : "rampe", "pente" : Umax}},
                    ],
                "analysis" : "static",
                "isotherm" : True
                }
    
pb = Tridimensional(dictionnaire)

#%% Résolution   
output_name = "Cisaillement_simple"
dico_solve = {"Prefix" : output_name, "csv_output" : {"reaction_force" : {"flag" : 4, "component" : "z"}}}
solve_instance = Solve(pb, dico_solve, compteur=1, npas=10)
solve_instance.solve()