"""
Test de traction 2D avec conditions aux limites de Neumann.

Ce script simule un essai de traction uniaxiale sur une plaque rectangulaire
en conditions de déformation plane avec des forces imposées sur les bords.

Paramètres géométriques:
    - Longueur: 1
    - Largeur: 0.5
    - Discrétisation: maillage 20×20 (quadrilatères)

Chargement:
    - Force surfacique imposée (f_surf): 1e3
    - Nombre de pas: 20

Conditions aux limites:
    - Déplacement horizontal bloqué sur le bord gauche
    - Déplacement vertical bloqué sur le bord inférieur
    - Force horizontale imposée sur le bord droit

Le script calcule la contrainte résultante et la compare avec la valeur
imposée (f_surf) pour vérifier la cohérence de l'implémentation des
conditions aux limites de Neumann.

Auteur: bouteillerp
Date de création: 11 Mars 2022
"""
from Charon import create_rectangle, PlaneStrain, CellType, Solve, MeshManager
from mpi4py.MPI import COMM_WORLD
import matplotlib.pyplot as plt
import pytest
import numpy as np
import sys
sys.path.append("../")
from Generic_isotropic_material import Acier

###### Paramètre géométrique ######
Largeur = 0.5
Longueur = 1

###### Maillage ######
mesh = create_rectangle(COMM_WORLD, [(0, 0), (Longueur, Largeur)], [20, 20], CellType.quadrilateral)

dictionnaire_mesh = {"tags": [1, 2, 3],
                     "coordinate": ["x", "y", "x"], 
                     "positions": [0, 0, Longueur]
                     }
mesh_manager = MeshManager(mesh, dictionnaire_mesh)

###### Paramètre du problème ######
f_surf = 1e3
dictionnaire = {"mesh_manager" : mesh_manager,
                "boundary_conditions": 
                    [{"component": "Ux", "tag": 1},
                     {"component": "Uy", "tag": 2}
                    ],
                "loading_conditions": 
                    [{"type": "surfacique", "component" : "Fx", "tag": 3, "value" : f_surf}
                    ],
                "isotherm" : True,
                "analysis" : "static"
                }
    
pb = PlaneStrain(Acier, dictionnaire)

dico_solve = {"Prefix" : "Traction_2D", "csv_output" : {"reaction_force" : {"flag" : 1, "component" : "x"}}}
solve_instance = Solve(pb, dico_solve, compteur=1, npas=20)
solve_instance.solve()

temps = np.loadtxt("Traction_2D-results/export_times.csv",  delimiter=',', skiprows=1)
reaction_force = np.loadtxt("Traction_2D-results/reaction_force.csv",  delimiter=',', skiprows=1)
sig_list = -reaction_force/Largeur
force_elast_list = np.linspace(0, f_surf, len(temps))

if __name__ == "__main__": 
    plt.scatter(temps, sig_list, marker = "x", color = "blue", label="CHARON")
    plt.plot(temps, force_elast_list, linestyle = "--", color = "red", label = "Analytical")
    plt.xlim(0, 1.1 * temps[-1])
    plt.ylim(0, 1.1 * sig_list[-1])
    plt.xlabel(r"Temps virtuel", size = 18)
    plt.ylabel(r"Contrainte (MPa)", size = 18)
    plt.legend()
    plt.show()
            