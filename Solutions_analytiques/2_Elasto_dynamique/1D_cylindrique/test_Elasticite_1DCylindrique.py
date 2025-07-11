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

from CharonX import *
import matplotlib.pyplot as plt
import pytest
import time
from Solution_analytique_cylindrique import main_analytique

###### Modèle géométrique ######
model = CylindricalUD
###### Modèle matériau ######
E = 210e3
nu = 0.3 
lmbda = E * nu / (1 - 2 * nu) / (1 + nu)
mu = E / 2. / (1 + nu)
rho = 7.8e-3
# rho = 1e-3
C=500
alpha=12e-6
rigi = lmbda + 2 * mu
wave_speed = (rigi/rho)**(1./2)
dico_eos = {"E":E, "nu" : nu, "alpha" : 12e-6}
dico_devia = {"E":E, "nu" : nu}
eos_type = "IsotropicHPP"

# iso_T_K0 = 175e3
# T_dep_K0 = 0
# iso_T_K1 = 0
# T_dep_K1 = 0
# dico_eos = {"iso_T_K0": iso_T_K0, "T_dep_K0" : T_dep_K0, "iso_T_K1": iso_T_K1, "T_dep_K1" : T_dep_K1}
# eos_type = "Vinet"

Acier = Material(rho, C, eos_type, "IsotropicHPP", dico_eos, dico_devia)

###### Paramètre géométrique ######
e = 5
R_int = 10
R_ext = R_int + e


###### Temps simulation ######
Tfin = 6e-4
pas_de_temps = Tfin/12000
magnitude = -1e2
T_unload = Tfin

sortie = 1000
pas_de_temps_sortie = sortie * pas_de_temps
n_sortie = int(Tfin/pas_de_temps_sortie)

t_etude =6e-4
n = int(Tfin / t_etude)


Nx = 2000
mesh = create_interval(MPI.COMM_WORLD, Nx, [np.array(R_int), np.array(R_ext)])

chargement = MyConstant(mesh, T_unload, magnitude, Type = "Creneau")
dictionnaire = {"mesh" : mesh,
                "boundary_setup": 
                    {"tags": [1, 2],
                     "coordinate": ["r", "r"], 
                     "positions": [R_int, R_ext]
                     },
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

main_analytique(R_int, R_ext, lmbda, mu, rho, magnitude, t_etude, num_points= 4000)
plt.legend()
plt.xlabel('r (mm)')
plt.ylabel(r'$\sigma_{rr}$ (MPa)')
plt.xlim(R_int, R_ext)
# plt.savefig(f"../../../Notice/fig/Compression_cylindrique_dynamique.pdf", bbox_inches = 'tight')
plt.show()