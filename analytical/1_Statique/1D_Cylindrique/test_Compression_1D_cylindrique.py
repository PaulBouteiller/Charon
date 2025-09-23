"""
Test de compression d'un cylindre en coordonnées cylindriques 1D.

Ce script simule la compression d'un cylindre creux soumis à des pressions interne et externe,
puis compare la solution numérique au champ de déplacement radial analytique.

Paramètres géométriques:
    - Rayon interne (Rint): 9
    - Rayon externe (Rext): Rint + e
    - Epaisseur (e): 2
    - Discrétisation (Nx): 20 éléments radiaux

Chargement:
    - Pression interne (Pint): -5
    - Pression externe (Pext): -10

La solution analytique utilise les équations de Lamé pour un cylindre à paroi épaisse.
Une assertion vérifie que l'erreur relative entre les solutions est inférieure à 0.1%.
"""
from Charon import CylindricalUD, Solve, create_1D_mesh, MeshManager
import pytest
import sys
sys.path.append("../../")
from Generic_isotropic_material import Acier, E, nu, mu

import matplotlib.pyplot as plt
from numpy import linspace, array
from pandas import read_csv

###### Paramètre géométrique ######
e = 2
Nx = 20
Rint = 9
Rext = Rint + e

###### Chargement ######
Pext = -10
Pint = -5

###### Maillage ######
mesh = create_1D_mesh(Rint, Rext, Nx)
dictionnaire_mesh = {"tags": [1, 2], "coordinate": ["x", "x"], "positions": [Rint, Rext]}
mesh_manager = MeshManager(mesh, dictionnaire_mesh)

###### Paramètre du problème ######
dictionnaire = {"material": Acier,
                "mesh_manager" : mesh_manager,
                "loading_conditions": 
                    [{"type": "surfacique", "component" : "F", "tag": 1, "value" : -Pint},
                     {"type": "surfacique", "component" : "F", "tag": 2, "value" : Pext}
                    ],
                "analysis" : "static",
                "isotherm" : True
                }

pb = CylindricalUD(dictionnaire)

###### Paramètre de la résolution ######
output_name ="Compression_cylindrique_1D"
dictionnaire_solve = {"Prefix" : output_name,"csv_output" : {"U" : True}}

solve_instance = Solve(pb, dictionnaire_solve, compteur=1, npas=10)
solve_instance.solve()

#%%Validation et tracé du résultat
u_csv = read_csv(output_name+"-results/U.csv")
resultat = [u_csv[colonne].to_numpy() for colonne in u_csv.columns]
solution_numerique = -resultat[-1]

len_vec = len(solution_numerique)
A = (Pint * Rint**2 - Pext * Rext**2) / (Rext**2 - Rint**2)
B = (Pint - Pext) / (Rext**2 - Rint**2) * Rint**2 * Rext**2
C = 2 * nu * A
a = (1 - nu) / E * A - nu /E * C
b = B / (2 * mu)
   
def ur(r):
    return  a * r + b / r
pas_espace = linspace(Rint, Rext, len_vec)
solution_analytique = array([ur(x) for x in pas_espace])
# On calcule la différence entre les deux courbes
vecteur_difference = solution_analytique - solution_numerique
# Puis on réalise une sorte d'intégration discrète
integrale_discrete = sum(abs(vecteur_difference[j]) for j in range(len_vec)) / sum(abs(solution_analytique[j]) for j in range(len_vec))
print("La difference est de", integrale_discrete)
assert integrale_discrete < 1e-3, "Cylindrical static compression fail"
if __name__ == "__main__": 
    plt.plot(pas_espace, solution_analytique, linestyle = "--", color = "red", label = "Analytique")
    plt.scatter(pas_espace, solution_numerique, marker = "x", color = "blue", label = "CHARON")
    
    plt.xlim(Rint, Rext)
    plt.xlabel(r"$r$ (mm)", size = 18)
    plt.ylabel(r"Déplacement radial (mm)", size = 18)
    plt.legend()
    # plt.savefig("../../../Notice/fig/Cylindric_compression.pdf", bbox_inches = 'tight')
    plt.show()
