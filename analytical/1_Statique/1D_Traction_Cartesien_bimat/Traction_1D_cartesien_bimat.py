"""
Test de traction sur une barre composite 1D (deux matériaux).

Ce script simule un essai de traction uniaxiale sur une barre 1D composée de
deux matériaux différents (acier et aluminium) et compare la solution numérique
avec la solution analytique.

Paramètres géométriques:
    - Longueur totale de la barre: 1
    - Discrétisation: 20 éléments
    - Deux moitiés égales de matériaux différents

Chargement:
    - Déplacement imposé (Umax): 1e-2 (1% de déformation)

Matériaux:
    - Acier: Module d'Young E
    - Aluminium: Module d'Young E/ratio (ratio = 3)
    - MÃªme coefficient de Poisson pour les deux matériaux

Solution analytique basée sur la continuité des contraintes à l'interface
et la répartition des déformations proportionnellement à l'inverse du module d'Young.

Auteur: bouteillerp
Date de création: 24 Juillet 2023
"""
from Charon import create_1D_mesh, CartesianUD, Solve, MeshManager
from pandas import read_csv
from ufl import SpatialCoordinate
import matplotlib.pyplot as plt
import pytest
from numpy import linspace

import sys
sys.path.append("../../")
from Generic_isotropic_material import Acier, Alu, ratio

Mat = [Acier, Alu]

###### Paramètre géométrique ######
Longueur = 1

###### Chargement ######
Umax=1e-2

Nx = 20
mesh = create_1D_mesh(0, Longueur, Nx)
dictionnaire_mesh = {"tags": [1, 2], "coordinate": ["x", "x"], "positions": [0, Longueur]}
mesh_manager = MeshManager(mesh, dictionnaire_mesh)

x = SpatialCoordinate(mesh)
demi_long = Longueur / 2      

dictionnaire = {"mesh_manager" : mesh_manager,
                "boundary_conditions": 
                    [{"component": "U", "tag": 1},
                     {"component": "U", "tag": 2, "value": {"type" : "rampe", "amplitude" : Umax}}
                    ],
                "multiphase" : {"conditions" : [x[0]<demi_long, x[0]>=demi_long]},
                "analysis" : "static",
                "isotherm" : True
                }

pb = CartesianUD(Mat, dictionnaire)


dictionnaire_solve = {
    "Prefix" : "Traction_1D",
    "csv_output" : {"U" : True}
    }

solve_instance = Solve(pb, dictionnaire_solve, compteur=1, npas=100)
solve_instance.solve()

#%%Validation et tracé du résultat
u_csv = read_csv("Traction_1D-results/U.csv")
resultat = [u_csv[colonne].to_numpy() for colonne in u_csv.columns]
x_result = resultat[0]
half_n_node = len(x_result)//2
eps_tot = Umax
eps_acier = 2 * eps_tot / (1 + ratio)
eps_alu = ratio * eps_acier 
dep_acier = [eps_acier * x for x in linspace(0, 0.5, half_n_node+1)]
dep_alu = [eps_acier * 0.5 + eps_alu * x for x in linspace(0, 0.5, half_n_node+1)]
dep_acier.pop()
dep_tot = dep_acier + dep_alu
solution_numerique = resultat[-1]
if __name__ == "__main__": 
    plt.scatter(x_result, solution_numerique, marker = "x", color = "red")
    plt.plot(x_result, dep_tot, linestyle = "--", color = "blue")            
    plt.xlim(0, 1)
    plt.ylim(0, 1.1 * max(dep_tot))
    plt.xlabel(r"Position (mm)", size = 18)
    plt.ylabel(r"Déplacement (mm)", size = 18)