"""
Test de validation pour l'élasticité 1D en coordonnées cartésiennes

Ce script implémente et exécute un test de validation pour les équations
d'élasticité linéaire en 1D dans un système de coordonnées cartésiennes.
Il compare la solution numérique obtenue avec CharonX à la solution analytique.

Cas test:
---------
- Barre élastique homogène avec conditions aux limites imposées
- Propagation d'onde longitudinale (compression/traction)
- Comparaison des déplacements, vitesses et contraintes

Théorie:
--------
L'équation d'onde 1D en élasticité linéaire s'écrit:
    ρ·∂²u/∂t² = ∂/∂x(E·∂u/∂x)

Pour un matériau homogène (E constant), l'équation se simplifie en:
    ∂²u/∂t² = c²·∂²u/∂x²

où c = √(E/ρ) est la vitesse des ondes élastiques.

La solution analytique pour une onde progressive est de la forme:
    u(x,t) = f(x - c·t) + g(x + c·t)

où f et g représentent respectivement les ondes se propageant vers la droite et vers la gauche.
"""

from CharonX import *
import matplotlib.pyplot as plt
import pytest
import time
import sys
from materiau import set_material
Acier, wave_speed, isotherm, T0  = set_material()

import os
import sys
# Obtenir le chemin absolu du dossier parent
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from Analytical_wave_propagation import *

###### Paramètre géométrique ######
L = 50
bord_gauche = 0
bord_droit = bord_gauche + L

###### Temps simulation ######
Tfin = 3./4 * L / wave_speed
print("le temps de fin de simulation est", Tfin )
pas_de_temps = Tfin/8000
largeur_creneau = L/4
magnitude = 1e3

sortie = 4000
Nx = 1000

mesh = create_interval(MPI.COMM_WORLD, Nx, [np.array(bord_gauche), np.array(bord_droit)])

T_unload = largeur_creneau/wave_speed
chargement = MyConstant(mesh, T_unload, magnitude, Type = "Creneau")
dictionnaire = {"mesh" : mesh,
                "boundary_setup": 
                    {"tags": [1, 2],
                     "coordinate": ["x", "x"], 
                     "positions": [bord_gauche, bord_droit]
                     },
                "boundary_conditions": 
                    [{"component": "U", "tag": 2}
                    ],
                "loading_conditions": 
                    [{"type": "surfacique", "component" : "F", "tag": 1, "value" : chargement}
                    ],
                "isotherm" : True
                }

pb = CartesianUD(Acier, dictionnaire)

dictionnaire_solve = {
    "Prefix" : "Test_elasticite",
    "csv_output" : {"Sig" : True}
    }

solve_instance = Solve(pb, dictionnaire_solve, compteur=sortie, TFin=Tfin, scheme = "fixed", dt = pas_de_temps)
tps1 = time.perf_counter()
solve_instance.solve()
tps2 = time.perf_counter()
print("temps d'execution", tps2 - tps1)

df = read_csv("Test_elasticite-results/Sig.csv")
import re
temps = np.array([float(re.search(r't=([0-9.]+)', col).group(1)) 
                  for col in df.columns if "t=" in col])
resultat = [df[colonne].to_numpy() for colonne in df.columns]
pas_espace = np.linspace(bord_gauche, bord_droit, len(resultat[-1]))
t_output = temps[1:]
for i, t in enumerate(t_output):
    plt.plot(resultat[0], resultat[i + 2], linestyle = "--")
    analytics = cartesian1D_progressive_wave(-magnitude, -largeur_creneau, 0, wave_speed, pas_espace, t)
    plt.plot(pas_espace, analytics)
plt.xlim(0, L)
plt.ylim(-1.1 * magnitude, 100)
plt.xlabel(r"Position (mm)", size = 18)
plt.ylabel(r"Contrainte (MPa)", size = 18)
plt.legend()