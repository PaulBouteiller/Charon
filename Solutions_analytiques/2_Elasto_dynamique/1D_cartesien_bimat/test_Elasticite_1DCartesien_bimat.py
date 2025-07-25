"""
Test de validation pour l'élasticité 1D dans un bi-matériau en coordonnées cartésiennes

Ce script implémente et exécute un test de validation pour la propagation d'ondes 
élastiques à travers l'interface entre deux matériaux différents en coordonnées cartésiennes 1D.
Il compare la solution numérique obtenue avec CharonX à la solution analytique.

Cas test:
---------
- Barre élastique avec deux matériaux (acier et aluminium)
- Interface située au milieu de la barre (x = 25 mm)
- Chargement en créneau sur l'extrémité gauche
- Propagation, réflexion et transmission d'onde à l'interface
- Comparaison des contraintes numériques et analytiques à différents instants

Théorie:
--------
Lors de la rencontre d'une onde avec une interface entre deux matériaux d'impédances
acoustiques Z₁ et Z₂ différentes, une partie de l'onde est réfléchie et une partie est
transmise selon les coefficients:
    R = (Z₂ - Z₁)/(Z₁ + Z₂)    (coefficient de réflexion)
    T = 2·Z₂/(Z₁ + Z₂)         (coefficient de transmission)

La solution analytique complète est implémentée dans le module Solution_analytique.py.

Auteur: bouteillerp
"""

from CharonX import CartesianUD, create_interval, MyConstant, SpatialCoordinate, Solve, Material
from ufl import conditional
from dolfinx.fem import Expression
from pandas import read_csv
import numpy as np
from mpi4py.MPI import COMM_WORLD
import matplotlib.pyplot as plt
import pytest
import time
from Solution_analytique import compute_sigma_tot
import sys
sys.path.append("../../")
from Generic_isotropic_material import Acier, Alu

# ###### Modèle mécanique ######
E_acier = 210e3
nu_acier = 0.3
# mu_acier = E_acier / 2. / (1 + nu_acier)
rho_acier = 7.8e-3
dico_eos = {"E" : E_acier, "nu" : nu_acier, "alpha" : 1}
dico_devia = {"E":E_acier, "nu" : nu_acier}
eos_type = "IsotropicHPP"
devia_type = "IsotropicHPP"
Acier = Material(rho_acier, 1, eos_type, devia_type, dico_eos, dico_devia)

###### Modèle mécanique ######
E_alu = 70e3
nu_alu = 0.34
# mu_alu = E_alu / 2. / (1 + nu_alu)
rho_alu = 2.7e-3
dico_eos_alu = {"E" : E_alu, "nu" : nu_alu, "alpha" : 1}
dico_devia_alu = {"E" : E_alu, "nu" : nu_alu}
eos_type_alu = "IsotropicHPP"
devia_type_alu = "IsotropicHPP"
Alu = Material(rho_alu, 1, eos_type_alu, devia_type_alu, dico_eos_alu, dico_devia_alu)

# Mat = [Acier, Acier]
Mat = Acier

###### Paramètre géométrique ######
L = 50
demi_longueur = L/2
bord_gauche = 0
bord_droit = bord_gauche + L

###### Temps simulation ######
wave_speed = Acier.celerity


Tfin = 1./8 * L / wave_speed
# Tfin = 3./4 * L / wave_speed
print("le temps de fin de simulation est", Tfin )
pas_de_temps = Tfin/8000
largeur_creneau = L/4
T_unload = largeur_creneau/wave_speed
magnitude = 1e3

sortie = 4000
pas_de_temps_sortie = sortie * pas_de_temps
n_sortie = int(Tfin/pas_de_temps_sortie)

Nx = 2000
mesh = create_interval(COMM_WORLD, Nx, [np.array(bord_gauche), np.array(bord_droit)])
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
    
# pb = CartesianUD(Mat, dictionnaire)
# x = SpatialCoordinate(pb.mesh)
# mult = pb.multiphase
# interp = mult.V_c.element.interpolation_points()
# ufl_condition_1 = conditional(x[0]<demi_longueur, 1, 0)
# c1_expr = Expression(ufl_condition_1, interp)
# ufl_condition_2 = conditional(x[0]>=demi_longueur, 1, 0)
# c2_expr = Expression(ufl_condition_2, interp)
# mult.set_multiphase([c1_expr, c2_expr])

pb = CartesianUD([Acier, Alu], dictionnaire)
# x = SpatialCoordinate(pb.mesh)
# mult = pb.multiphase
# interp = mult.V_c.element.interpolation_points()
# ufl_condition_1 = conditional(x[0]<99, 1, 0)
# c1_expr = Expression(ufl_condition_1, interp)
# ufl_condition_2 = conditional(x[0]>=99, 1, 0)
# c2_expr = Expression(ufl_condition_2, interp)
# mult.set_multiphase([c1_expr, c2_expr])

        
dictionnaire_solve = {
    "Prefix" : "Test_elasticite",
    "csv_output" : {"Sig" : True}
    }

solve_instance = Solve(pb, dictionnaire_solve, compteur=sortie, TFin=Tfin, scheme = "fixed", dt = pas_de_temps)
solve_instance.solve()

df = read_csv("Test_elasticite-results/Sig.csv")
import re
temps = np.array([float(re.search(r't=([0-9.]+)', col).group(1)) 
                  for col in df.columns if "t=" in col])
resultat = [df[colonne].to_numpy() for colonne in df.columns]
pas_espace = np.linspace(bord_gauche, bord_droit, len(resultat[-1]))
t_output = temps[1:]
x_vals = np.linspace(0, L, 1000)
for i, t in enumerate(t_output):
    plt.plot(resultat[0], resultat[i + 2], linestyle = "--")
    # sigma_vals = compute_sigma_tot(t, T_unload, L, demi_longueur, magnitude, rho_acier, rho_alu, E_acier, E_alu, nu_acier, nu_alu)
    # plt.plot(x_vals, sigma_vals, color='r')
    
plt.xlim(0, L)
plt.ylim(-1.1 * magnitude, 1.1 * magnitude)
plt.xlabel(r"Position (mm)", size = 18)
plt.ylabel(r"Contrainte (MPa)", size = 18)
plt.legend()