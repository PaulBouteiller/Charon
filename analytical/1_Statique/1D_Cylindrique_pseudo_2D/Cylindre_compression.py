"""
Test de compression d'un cylindre creux en axisymétrique (pseudo-2D).

Ce script simule la compression d'un cylindre creux soumis à des pressions interne et externe
en utilisant un modèle axisymétrique, puis compare la solution numérique avec la solution
analytique de Lamé.

Paramètres géométriques:
    - Rayon interne (Rint): 9
    - Rayon externe (Rext): 11
    - Hauteur du cylindre: 1
    - Discrétisation: maillage 20Ã—10 (quadrilatères)

Chargement:
    - Pression interne (Pint): -5
    - Pression externe (Pext): -10

Conditions aux limites:
    - Déplacement vertical bloqué sur les faces supérieure et inférieure
    - Pressions sur les faces internes et externes

Une vérification est effectuée pour comparer le champ de déplacement radial calculé
numériquement avec la solution analytique.

Auteur: bouteillerp
Date de création: 11 Mars 2022
"""
from Charon import Axisymmetric, Solve, create_rectangle, MeshManager
from mpi4py.MPI import COMM_WORLD
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
import pytest
import sys
sys.path.append("../../")
from Generic_isotropic_material import Acier, E, nu, mu

###### Paramètre géométrique ######
L = 2
Nx = 100
Rint = 9
Rext = Rint + L
hauteur = 1

###### Chargement ######
Pext = -10
Pint = -5

mesh = create_rectangle(COMM_WORLD, [(Rint, 0), (Rext, hauteur)], [20, 10])
dictionnaire_mesh = {"tags": [1, 2, 3, 4],
                     "coordinate": ["z", "r", "r", "z"], 
                     "positions": [0, Rint, Rext, hauteur]
                     }
mesh_manager = MeshManager(mesh, dictionnaire_mesh)

###### Paramètre du problème ######
dictionnaire = {"material": Acier,
                "mesh_manager" : mesh_manager,
                "loading_conditions": 
                    [{"type": "surfacique", "component" : "Fr", "tag": 2, "value" : -Pint},
                     {"type": "surfacique", "component" : "Fr", "tag": 3, "value" : Pext}
                    ],
                "boundary_conditions": 
                    [{"component": "Uz", "tag": 1},
                     {"component": "Uz", "tag": 4}
                    ],
                "analysis" : "static",
                "isotherm" : True
                }
   
pb = Axisymmetric(dictionnaire)

###### Paramètre de la résolution ######
dictionnaire_solve = {
    "Prefix" : "Cylindre_axi",
    "csv_output" : {"U" : ["Boundary", 1]}
    }

solve_instance = Solve(pb, dictionnaire_solve, compteur=1, npas=100)
solve_instance.solve()

#%%Validation et tracé du résultat
u_csv = read_csv("Cylindre_axi-results/U.csv")
resultat = [u_csv[colonne].to_numpy() for colonne in u_csv.columns]
r_result = resultat[0]
solution_numerique = -resultat[-2]

len_vec = len(r_result)
A = (Pint * Rint**2 - Pext * Rext**2) / (Rext**2 - Rint**2)
B = (Pint - Pext) / (Rext**2 - Rint**2) * Rint**2 * Rext**2
C = 2 * nu * A
a = (1-nu)/E * A - nu /E * C
b = B / (2 * mu)
   
def ur(r):
    return  a * r + b / r
solution_analytique = np.array([ur(x) for x in r_result])
# On calcule la différence entre les deux courbes
diff_tot = solution_analytique - solution_numerique
# Puis on réalise une sorte d'intégration discrète
integrale_discrete = sum(abs(diff_tot[j]) for j in range(len_vec))/sum(abs(solution_analytique[j]) for j in range(len_vec))
print("La difference est de", integrale_discrete)
assert integrale_discrete < 1e-3, "Cylindrical static compression fail"
if __name__ == "__main__": 
    plt.plot(r_result, solution_analytique, linestyle = "--", color = "red")
    plt.scatter(r_result, solution_numerique, marker = "x", color = "blue")
    
    plt.xlim(Rint, Rext)
    plt.xlabel(r"$r$", size = 18)
    plt.ylabel(r"Déplacement radial", size = 18)