"""
Test de comportement mécanique 0D pour différents modèles constitutifs.

Ce script teste le comportement mécanique d'un matériau en imposant une déformation
volumétrique (traction/compression) et en comparant les résultats numériques aux
solutions analytiques.

La simulation utilise:
    - Modèle: CartesianUD (1D cartésien)
    - Matériau: défini par les paramètres eos_type et devia_type
    - Chargement: déformation imposée varepsilon
    - Analyse: statique, isotherme
    
Le test vérifie la cohérence entre les contraintes calculées numériquement et analytiquement.

Auteur: bouteillerp
Date de création: 24 Juillet 2023
"""
from Charon import create_1D_mesh, Solve, CartesianUD, MeshManager

import pytest
from material_definition import set_material
from numerical_analytical_comparison import comparison

###### Materiau ######
eos_type = "Tabulated"
devia_type = "NeoHook"
Mat = set_material(eos_type, devia_type)

###### Chargement ######
varepsilon = -0.3
T0 = 1e3

mesh = create_1D_mesh(0, 1, 1)
dictionnaire_mesh = {"tags": [1, 2], "coordinate": ["x", "x"], "positions": [0, 1]}
mesh_manager = MeshManager(mesh, dictionnaire_mesh)

dictionnaire = {"material" : Mat,
                "mesh_manager" : mesh_manager,
                "boundary_conditions": 
                    [{"component": "U", "tag": 1},
                     {"component": "U", "tag": 2, "value": {"type" : "rampe", "pente" : varepsilon}}
                    ],
                "analysis" : "static",
                "isotherm" : True
                }
    
pb = CartesianUD(dictionnaire)

pb.T.x.petsc_vec.set(T0)
pb.T0.x.petsc_vec.set(T0)

dictionnaire_solve = {
    "Prefix" : "Test_0D_" + eos_type,
    "csv_output" : {"p" : True, "U" : ["Boundary", 2], "s" :  True}
    }

def final_output(problem):
    comparison(Mat, varepsilon, T0)
    return
solve_instance = Solve(pb, dictionnaire_solve, compteur=1, npas=20)
solve_instance.final_output = final_output
solve_instance.solve()