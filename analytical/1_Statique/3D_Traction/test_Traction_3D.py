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
Nx, Ny, Nz = 10, 10, 10
mesh = create_box(COMM_WORLD, [array([0, 0, 0]), 
                               array([Longueur, Largeur, hauteur])],
                              [Nx, Ny, Nz])

dictionnaire_mesh = {"tags": [1, 2, 3, 4],
                     "coordinate": ["x", "y", "z", "z"], 
                     "positions": [0, 0, 0, hauteur]
                     }
mesh_manager = MeshManager(mesh, dictionnaire_mesh)

#%% Définition du problème
eps = 0.005
Umax = eps * hauteur
dictionnaire = {"mesh_manager" : mesh_manager,
                "boundary_conditions": 
                    [{"component": "Ux", "tag": 1},
                     {"component": "Uy", "tag": 2},
                     {"component": "Uz", "tag": 3},
                     {"component": "Uz", "tag": 4, "value": {"type" : "rampe", "amplitude" : Umax}},
                    ],
                "analysis" : "static",
                "isotherm" : True
                }
    
pb = Tridimensional(Acier, dictionnaire)

#%% Résolution   
dico_solve = {"Prefix" : "Traction_3D", "csv_output" : {"reaction_force" : {"flag" : 4, "component" : "z"}}}
solve_instance = Solve(pb, dico_solve, compteur=1, npas=10)
solve_instance.solve()

#%% Validation et tracé du résultat
def force_elast(eps):
    return E * eps * Largeur * Longueur

temps = loadtxt("Traction_3D-results/export_times.csv",  delimiter=',', skiprows=1)
numerical_results = loadtxt("Traction_3D-results/reaction_force.csv",  delimiter=',', skiprows=1)
eps_list = [eps * t for t in temps]    

solution_analytique = array([force_elast(eps) for eps in eps_list])
eps_list_percent = [100 * eps for eps in eps_list]

# On calcule la différence entre les deux courbes
len_vec = len(solution_analytique)
diff_tot = solution_analytique - numerical_results
# Puis on réalise une sorte d'intégration discrète
integrale_discrete = sum(abs(diff_tot[j]) for j in range(len_vec))/sum(abs(solution_analytique[j]) for j in range(len_vec))
print("La difference est de", integrale_discrete)
assert integrale_discrete < 0.01, "Static 1D traction fail"
if __name__ == "__main__": 
    plt.scatter(eps_list_percent, numerical_results, marker = "x", color = "blue", label="CHARON")
    plt.plot(eps_list_percent, solution_analytique, linestyle = "--", color = "red", label = "Analytique")
    plt.xlim(0, 1.1 * eps_list_percent[-1])
    plt.ylim(0, 1.1 * numerical_results[-1])
    plt.xlabel(r"Déformation(%)", size = 18)
    plt.ylabel(r"Force (N)", size = 18)
    plt.legend()
    plt.show()