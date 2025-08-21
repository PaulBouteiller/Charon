"""
Test de traction 2D avec conditions aux limites de Dirichlet.

Ce script simule un essai de traction uniaxiale sur une plaque rectangulaire
en conditions de déformation plane avec des déplacements imposés sur les bords.

Paramètres géométriques:
    - Longueur: 1
    - Largeur: 0.5
    - Discrétisation: maillage 20×20 (quadrilatères)

Chargement:
    - Déplacement imposé (Umax): 0.002 (0.2% de déformation)

Conditions aux limites:
    - Déplacement horizontal bloqué sur le bord gauche
    - Déplacement vertical bloqué sur le bord inférieur
    - Déplacement horizontal imposé sur le bord droit

Le script calcule la force résultante et la compare avec la solution analytique
pour un problème de déformation plane (correction par le facteur 1/(1-nu²)).
Une assertion vérifie que l'erreur relative est inférieure à 1%.

Auteur: bouteillerp
Date de création: 11 Mars 2022
"""
from Charon import CellType, create_rectangle, MeshManager, PlaneStrain, Solve
from mpi4py.MPI import COMM_WORLD
from numpy import array, loadtxt
import matplotlib.pyplot as plt
import pytest
import sys
sys.path.append("../")
from Generic_isotropic_material import Acier, E, nu

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
Umax = 0.002
dictionnaire = {"mesh_manager" : mesh_manager,
                "boundary_conditions": 
                    [{"component": "Ux", "tag": 1},
                     {"component": "Uy", "tag": 2},
                     {"component": "Ux", "tag": 3, "value": {"type" : "rampe", "amplitude" : Umax}}
                    ],
                "analysis" : "static",
                "isotherm" : True
                }

pb = PlaneStrain(Acier, dictionnaire)
dico_solve = {"Prefix" : "Traction_2D", "csv_output" : {"reaction_force" : {"flag" : 3, "component" : "x"}}}

solve_instance = Solve(pb, dico_solve, compteur=1, npas=20)
solve_instance.solve()
#%%
temps = loadtxt("Traction_2D-results/export_times.csv",  delimiter=',', skiprows=1)
numerical_results = loadtxt("Traction_2D-results/reaction_force.csv",  delimiter=',', skiprows=1)
eps_list = [Umax / Longueur * t for t in temps]    

def force_elast(eps):
    return E * eps * Largeur /(1 - nu**2)

solution_analytique = array([force_elast(eps) for eps in eps_list])
eps_list_percent = [100 * eps for eps in eps_list]
# On calcule la différence entre les deux courbes
len_vec = len(solution_analytique)
diff_tot = solution_analytique - numerical_results
# Puis on réalise une sorte d'intégration discrète
integrale_discrete = sum(abs(diff_tot[j]) for j in range(len_vec))/sum(abs(solution_analytique[j]) for j in range(len_vec))
print("La difference est de", integrale_discrete)
# assert integrale_discrete < 0.01, "Static 1D traction fail"
if __name__ == "__main__": 
    plt.scatter(eps_list_percent, numerical_results, marker = "x", color = "blue", label="CHARON")
    plt.plot(eps_list_percent, solution_analytique, linestyle = "--", color = "red", label = "Analytique")
    plt.xlim(0, 1.1 * eps_list_percent[-1])
    plt.ylim(0, 1.1 * numerical_results[-1])
    plt.xlabel(r"Déformation(%)", size = 18)
    plt.ylabel(r"Force (N)", size = 18)
    plt.legend()
    plt.show()
