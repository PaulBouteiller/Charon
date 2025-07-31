"""
Test de validation pour l'élasticité en déformation plane quasi-1D

Ce script implémente et exécute un test de validation pour les équations
d'élasticité linéaire en déformation plane dans une configuration géométrique
qui permet une comparaison avec une solution 1D.

Cas test:
---------
- Rectangle très allongé (rapport longueur/largeur = 50/0.05)
- Conditions aux limites de déplacement vertical nul sur les bords supérieur et inférieur
- Application d'un chargement en créneau sur le bord gauche
- Propagation d'onde longitudinale (compression/traction)
- Comparaison avec la solution analytique 1D cartésienne

Théorie:
--------
En déformation plane avec un domaine très allongé et des conditions aux limites
appropriées, le comportement de l'onde est essentiellement 1D dans la direction
longitudinale, permettant ainsi une comparaison avec la solution analytique 1D.

Auteur: bouteillerp
"""

from Charon import Solve, MyConstant, create_rectangle, PlaneStrain, CellType, MeshManager
from pandas import read_csv
import numpy as np
from mpi4py.MPI import COMM_WORLD
import matplotlib.pyplot as plt
import pytest
import os
import sys
# Obtenir le chemin absolu du dossier parent
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from Analytical_wave_propagation import cartesian1D_progressive_wave
sys.path.append("../../")
from Generic_isotropic_material import Acier, lmbda, mu, rho


rigi = lmbda + 2 * mu
wave_speed = (rigi/rho)**(1./2)

###### Paramètre géométrique ######
Nx = 1000
Longueur = 50
Largeur = Longueur / Nx
###### Temps simulation ######
Tfin = 7e-3
print("le temps de fin de simulation est", Tfin )
pas_de_temps = Tfin/8000
largeur_creneau = Longueur/4
magnitude = 1e3

sortie = 4000
pas_de_temps_sortie = sortie * pas_de_temps
n_sortie = int(Tfin/pas_de_temps_sortie)

mesh = create_rectangle(COMM_WORLD, [(0, 0), (Longueur, Largeur)], [Nx, 1], CellType.quadrilateral)

dictionnaire_mesh = {"tags": [1, 2, 3],
                     "coordinate": ["x", "y", "y"], 
                     "positions": [0, 0, Largeur]
                     }
mesh_manager = MeshManager(mesh, dictionnaire_mesh)
T_unload = largeur_creneau/wave_speed
chargement = MyConstant(mesh, T_unload, magnitude, Type = "Creneau")

dictionnaire = {"mesh_manager" : mesh_manager,
                "boundary_conditions": 
                    [{"component": "Uy", "tag": 2}, {"component": "Uy", "tag": 3}],
                "loading_conditions": 
                    [{"type": "surfacique", "component" : "Fx", "tag": 1, "value" : chargement}],
                "isotherm" : True,
                "damping" : {"damping" : True, 
                             "linear_coeff" : 0.1,
                             "quad_coeff" : 0.01,
                             "correction" : True
                             }
                }

pb = PlaneStrain(Acier, dictionnaire)

dictionnaire_solve = {"Prefix" : "Test_elasticite", "csv_output" : {"Sig" : True}}
solve_instance = Solve(pb, dictionnaire_solve, compteur=sortie, TFin=Tfin, scheme = "fixed", dt = pas_de_temps)
solve_instance.solve()

df = read_csv("Test_elasticite-results/Sig.csv")
import re
temps = np.array([float(re.search(r't=([0-9.]+)', col).group(1)) 
                  for col in df.columns if "t=" in col])
resultat = [df[colonne].to_numpy() for colonne in df.columns]
sigma_xx = [resultat[3 * i + 2] for i in range((len(resultat)-2)//3)]
pas_espace = np.linspace(0, Longueur, len(sigma_xx[0]))
t_output = temps[3::3]
for j, t in enumerate(t_output):
    plt.plot(resultat[0], sigma_xx[j+1], linestyle = "--")
    analytics = cartesian1D_progressive_wave(-magnitude, -largeur_creneau, 0, wave_speed, pas_espace, t)
    plt.plot(pas_espace, analytics)
plt.xlim(0, Longueur)
plt.ylim(-1.1 * magnitude, 100)
plt.xlabel(r"Position (mm)", size = 18)
plt.ylabel(r"Contrainte (MPa)", size = 18)
plt.legend()

