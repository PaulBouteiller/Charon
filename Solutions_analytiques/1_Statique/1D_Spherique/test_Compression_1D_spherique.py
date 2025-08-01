"""
Test de compression d'une sphère creuse en coordonnées sphériques 1D.

Ce script simule la compression d'une sphère creuse soumise à une pression externe,
puis compare la solution numérique au champ de déplacement radial analytique.

Paramètres géométriques:
    - Rayon interne (Rint): 9
    - Rayon externe (Rext): Rint + e
    - Épaisseur (e): 2
    - Discrétisation (Nx): 10 éléments radiaux

Chargement:
    - Pression appliquée (Pext): 10 (en externe)

La solution analytique est basée sur les équations d'élasticité linéaire en coordonnées sphériques.
Une assertion vérifie que l'erreur relative entre les solutions est inférieure à 0.1%.

Auteur: bouteillerp
Date de création: 11 Mars 2022
"""
from Charon import create_1D_mesh, SphericalUD, Solve, MeshManager
import numpy as np
import matplotlib.pyplot as plt
import pytest
import sys
sys.path.append("../../")
from Generic_isotropic_material import E, Acier, nu

###### Paramètre géométrique ######
e = 2
Nx = 10
Rint = 9
Rext = Rint + e
Pext = 10

mesh = create_1D_mesh(Rint, Rext, Nx)
dictionnaire_mesh = {"tags": [1, 2], "coordinate": ["x", "x"], "positions": [Rint, Rext]}
mesh_manager = MeshManager(mesh, dictionnaire_mesh)

###### Paramètre du problème ######
dictionnaire = {"mesh_manager" : mesh_manager,
                "loading_conditions": 
                    [{"type": "surfacique", "component" : "F", "tag": 2, "value" : -Pext}],
                "analysis" : "static",
                "isotherm" : True
                }
    
pb = SphericalUD(Acier, dictionnaire)
solve_instance = Solve(pb, {}, compteur=1, npas=10)
solve_instance.solve()

#%%Validation et tracé du résultat
solution_numerique = pb.u.x.array
len_vec = len(solution_numerique)
def ur(r, p_ext):
    return  - Rext**3 / (Rext**3 - Rint**3) * ((1 - 2*nu) * r + (1 + nu) * Rint**3 / (2 * r**2)) * p_ext / E
pas_espace = np.linspace(Rint, Rext, len_vec)
solution_analytique = np.array([ur(x, Pext) for x in pas_espace])
# On calcule la différence entre les deux courbes
diff_tot = solution_analytique - solution_numerique
# Puis on réalise une sorte d'intégration discrète
integrale_discrete = sum(abs(diff_tot[j]) for j in range(len_vec))/sum(abs(solution_analytique[j]) for j in range(len_vec))
print("La difference est de", integrale_discrete)
assert integrale_discrete < 0.001, "Spheric static compression fail"
if __name__ == "__main__": 
    plt.plot(pas_espace, solution_analytique, linestyle = "--", color = "red", label = "Analytique")
    plt.scatter(pas_espace, solution_numerique, marker = "x", color = "blue", label = "CHARON")
    
    plt.xlim(Rint, Rext)
    # plt.ylim(- 1.05 * magnitude, 0)

    plt.xlabel(r"$r$ (mm)", size = 18)
    plt.ylabel(r"Déplacement radial (mm)", size = 18)
    plt.legend()
    plt.show()