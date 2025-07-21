"""
Test de validation pour l'élasticité dynamique en déformation plane 2D

Ce script implémente et exécute un test de validation pour les équations
d'élasticité linéaire en déformation plane 2D.

Cas test:
---------
- Rectangle avec longueur 50 mm et largeur 1 mm
- Condition aux limites de déplacement vertical nul sur le bord inférieur
- Application d'un chargement en créneau sur le bord gauche
- Propagation d'onde dans le domaine 2D

Ce test sert principalement à vérifier le comportement du code en 2D
et à évaluer les performances de calcul.

Auteur: bouteillerp
"""

from CharonX import Solve, MyConstant, create_rectangle, Plane_strain, CellType
from mpi4py.MPI import COMM_WORLD
import pytest

import sys
sys.path.append("../../")
from Generic_isotropic_material import Acier, lmbda, mu, rho
rigi = lmbda + 2 * mu
wave_speed = (rigi/rho)**(1./2)

###### Paramètre géométrique ######
Largeur = 1
Longueur = 50

###### Temps simulation ######
Tfin = 7e-3
print("le temps de fin de simulation est", Tfin )
pas_de_temps = Tfin/8000
largeur_creneau = Longueur/4
magnitude = 1e4

sortie = 1000
pas_de_temps_sortie = sortie * pas_de_temps
n_sortie = int(Tfin/pas_de_temps_sortie)

Nx = 200
mesh = create_rectangle(COMM_WORLD, [(0, 0), (Longueur, Largeur)], [Nx, 1], CellType.quadrilateral)
T_unload = largeur_creneau/wave_speed
chargement = MyConstant(mesh, T_unload, magnitude, Type = "Creneau")
dictionnaire = {"mesh" : mesh,
                "boundary_setup": 
                    {"tags": [1, 2],
                     "coordinate": ["x", "y"], 
                     "positions": [0, 0]
                     },
                "boundary_conditions": 
                    [{"component": "Uy", "tag": 2},
                    ],
                "loading_conditions": 
                    [{"type": "surfacique", "component" : "Fx", "tag": 1, "value" : chargement}
                    ],
                "isotherm" : True,
                }

pb = Plane_strain(Acier, dictionnaire)

dictionnaire_solve = {
    "Prefix" : "Test_elasticite",
    "csv_output" : {"Sig" : True},
    "output" : {"Sig" : True}
    }

solve_instance = Solve(pb, dictionnaire_solve, compteur=sortie, TFin=Tfin, scheme = "fixed", dt = pas_de_temps)
solve_instance.solve()