"""
Test de validation pour l'élasticité 1D en coordonnées sphériques

Ce script implémente et exécute un test de validation pour les équations
d'élasticité linéaire en 1D dans un système de coordonnées sphériques.
Il compare la solution numérique obtenue avec CharonX à la solution analytique.

Cas test:
---------
- Sphère creuse avec rayon intérieur R_int = 5 mm et rayon extérieur R_ext = 10 mm
- Application d'une pression échelon sur la surface extérieure
- Propagation d'onde radiale vers l'intérieur avec atténuation géométrique
- Comparaison des contraintes radiales numériques et analytiques

Théorie:
--------
L'équation d'onde en coordonnées sphériques inclut des termes supplémentaires
liés à la courbure géométrique, ce qui conduit à une atténuation de l'onde en 1/r²
lors de sa propagation vers le centre.

La solution analytique est implémentée dans le module Solution_analytique_spherique.py.

Auteur: bouteillerp
"""

from CharonX import Solve, SphericalUD, MyConstant, create_interval, MeshManager
from pandas import read_csv
from mpi4py.MPI import COMM_WORLD
import matplotlib.pyplot as plt
import pytest
import numpy as np
from Solution_analytique_spherique import main_analytique
import sys
sys.path.append("../../")
from Generic_isotropic_material import Acier, lmbda, mu, rho
    
###### Paramètre géométrique ######
e = 5
R_int = 5
R_ext = R_int + e

###### Temps simulation ######
Tfin = 3e-4
print("le temps de fin de simulation est", Tfin )
pas_de_temps = Tfin / 10000
largeur_creneau = e
magnitude = 1e4
T_unload = Tfin

sortie = 2500
pas_de_temps_sortie = sortie * pas_de_temps
n_sortie = int(Tfin/pas_de_temps_sortie)
   
Nx = 2000
mesh = create_interval(COMM_WORLD, Nx, [np.array(R_int), np.array(R_ext)])
dictionnaire_mesh = {"tags": [1, 2], "coordinate": ["r", "r"], "positions": [R_int, R_ext]}
mesh_manager = MeshManager(mesh, dictionnaire_mesh)

chargement = MyConstant(mesh, T_unload, magnitude, Type = "Creneau")
dictionnaire = {"mesh_manager" : mesh_manager,
                "loading_conditions": 
                    [{"type": "surfacique", "component" : "F", "tag": 2, "value" : chargement}
                    ],
                "isotherm" : True
                }

pb = SphericalUD(Acier, dictionnaire)

dictionnaire_solve = {
    "Prefix" : "Onde_spherique",
    "csv_output" : {"Sig" : True}
    }

solve_instance = Solve(pb, dictionnaire_solve, compteur=sortie, TFin=Tfin, scheme = "fixed", dt = pas_de_temps)
solve_instance.solve()
               

df = read_csv("Onde_spherique-results/Sig.csv")
plt.plot(df['r'], df.iloc[:, -3], 
        linestyle="--", label=f'CHARON t={Tfin:.2e}ms')


    
# Exécution des deux solutions
main_analytique(R_int, R_ext, lmbda, mu, rho, magnitude, Tfin)
plt.legend()
plt.xlabel('r (mm)')
plt.ylabel(r'$\sigma_{rr}$ (MPa)')
plt.xlim(5, 10)
plt.show()