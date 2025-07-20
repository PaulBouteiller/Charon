"""
Test de compression d'un cylindre creux en 2D axisymétrique.

Ce script simule la compression d'un cylindre creux soumis Ã  des pressions interne
et externe en utilisant un modèle axisymétrique, puis compare la solution numérique
avec la solution analytique de Lamé.

Paramètres géométriques:
    - Rayon interne (Rint): 9
    - Rayon externe (Rext): 11
    - Hauteur du cylindre: 1
    - Discrétisation: maillage 10x—5 (triangles)

Chargement:
    - Pression interne (Pint): -5
    - Pression externe (Pext): -10

Conditions aux limites:
    - Déplacement vertical bloqué sur la face inférieure

Une comparaison est effectuée entre le champ de déplacement radial calculé
numériquement et la solution analytique via le module depouillement.py.
L'erreur relative entre les deux solutions est calculée pour vérifier la précision
du modèle numérique.

Auteur: bouteillerp
Date de création: 11 Mars 2022
"""
from CharonX import Axisymmetric, Solve, create_rectangle
from mpi4py import MPI
import numpy as np
from pandas import read_csv
import pytest
import sys
sys.path.append("../")
from Generic_isotropic_material import Acier, mu, E, nu
import matplotlib.pyplot as plt

###### Paramètre géométrique ######
L = 2
Nx = 100
Rint = 9
Rext = Rint + L
hauteur = 1

###### Chargement ######
Pext = -10
Pint = -5

#Ne fonctionne qu'avec des triangles voir pourquoi
mesh = create_rectangle(MPI.COMM_WORLD, [(Rint, 0), (Rext, hauteur)], [10, 5])
dictionnaire = {"mesh" : mesh,
                "boundary_setup": 
                    {"tags": [1, 2, 3],
                     "coordinate": ["r", "r", "z"], 
                     "positions": [Rint, Rext, 0]
                     },
                "boundary_conditions": 
                    [{"component": "Uz", "tag": 3}
                    ],
                "loading_conditions": 
                    [{"type": "surfacique", "component" : "Fr", "tag": 1, "value" : -Pint},
                     {"type": "surfacique", "component" : "Fr", "tag": 2, "value" : Pext}
                    ],
                "isotherm" : True,
                "analysis" : "static"
                }
    
pb = Axisymmetric(Acier, dictionnaire)


dictionnaire_solve = {
    "Prefix" : "Cylindre_axi",
    "csv_output" : {"U" : ["Boundary", 3]}
    }
solve_instance = Solve(pb, dictionnaire_solve, compteur=1, npas=10)
solve_instance.solve()

u_csv = read_csv("Cylindre_axi-results/U.csv")
resultat = [u_csv[colonne].to_numpy() for colonne in u_csv.columns]
r_unsorted = resultat[0]
sort_indices = np.argsort(r_unsorted)
r_result = r_unsorted[sort_indices]
solution_numerique = -resultat[-2][sort_indices]

len_vec = len(r_result)
A = (Pint * Rint**2 - Pext * Rext**2) / (Rext**2 - Rint**2)
B = (Pint - Pext) / (Rext**2 - Rint**2) * Rint**2 * Rext**2
C = 0
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
# assert integrale_discrete < 1e-3, "Cylindrical static compression fail"
if __name__ == "__main__": 
    plt.plot(r_result, solution_analytique, linestyle = "--", color = "red")
    plt.scatter(r_result, solution_numerique, marker = "x", color = "blue")
    
    # plt.xlim(Rint, Rext)
    plt.xlabel(r"$r$", size = 18)
    plt.ylabel(r"Déplacement radial", size = 18)