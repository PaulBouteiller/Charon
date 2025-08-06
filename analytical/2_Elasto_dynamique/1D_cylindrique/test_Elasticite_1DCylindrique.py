"""
Test de validation pour l'élasticité 1D en coordonnées cylindriques

Ce script implémente et exécute un test de validation pour les équations
d'élasticité linéaire en 1D dans un système de coordonnées cylindriques.
Il compare la solution numérique obtenue avec CharonX à la solution analytique.

Cas test:
---------
- Cylindre creux avec rayon intérieur R_int = 5 mm et rayon extérieur R_ext = 10 mm
- Application d'une pression échelon sur la surface extérieure
- Propagation d'onde radiale vers l'intérieur avec atténuation géométrique
- Comparaison des contraintes radiales numériques et analytiques

Théorie:
--------
L'équation d'onde en coordonnées cylindriques inclut des termes supplémentaires
liés à la courbure géométrique, ce qui conduit à une atténuation de l'onde en 1/√r
lors de sa propagation vers le centre.

La solution analytique est implémentée dans le module Solution_analytique_cylindrique.py.

Auteur: bouteillerp
"""

from Charon import Solve, CylindricalUD, MyConstant, create_interval, MeshManager
from pandas import read_csv
import numpy as np
from mpi4py.MPI import COMM_WORLD
import matplotlib.pyplot as plt
import pytest
from Solution_analytique_cylindrique import main_analytique

import sys
sys.path.append("../../")
from Generic_isotropic_material import Acier, lmbda, mu, rho

###### Paramètre géométrique ######
e = 5
R_int = 10
R_ext = R_int + e

###### Temps simulation ######
Tfin = 6e-4
pas_de_temps = Tfin/12000
magnitude = -1e2

sortie = 1000
pas_de_temps_sortie = sortie * pas_de_temps
n_sortie = int(Tfin/pas_de_temps_sortie)

Nx = 2000
mesh = create_interval(COMM_WORLD, Nx, [np.array(R_int), np.array(R_ext)])
dictionnaire_mesh = {"tags": [1, 2], "coordinate": ["r", "r"], "positions": [R_int, R_ext]}
mesh_manager = MeshManager(mesh, dictionnaire_mesh)

chargement = MyConstant(mesh, Tfin, magnitude, Type = "Creneau")
dictionnaire = {"mesh_manager" : mesh_manager,
                "loading_conditions": 
                    [{"type": "surfacique", "component" : "F", "tag": 2, "value" : chargement}
                    ],
                "isotherm" : True
                }

pb = CylindricalUD(Acier, dictionnaire)

dictionnaire_solve = {
    "Prefix" : "Onde_cylindrique",
    "csv_output" : {"Sig" : True}
    }

solve_instance = Solve(pb, dictionnaire_solve, compteur=sortie, TFin=Tfin, scheme = "fixed", dt = pas_de_temps)
solve_instance.solve()

df = read_csv("Onde_cylindrique-results/Sig.csv")
plt.plot(df['r'], df.iloc[:, -2], 
        linestyle="--", label=f'CHARON t={Tfin:.2e}ms')

main_analytique(R_int, R_ext, lmbda, mu, rho, magnitude, Tfin, num_points= 4000)
plt.legend()
plt.xlabel('r (mm)')
plt.ylabel(r'$\sigma_{rr}$ (MPa)')
plt.xlim(R_int, R_ext)
# plt.savefig(f"../../../Notice/fig/Compression_cylindrique_dynamique.pdf", bbox_inches = 'tight')
plt.show()