"""
Test de validation pour l'élasticité dynamique en 3D

Ce script implémente et exécute un test de validation pour les équations
d'élasticité linéaire en 3D.

Cas test:
---------
- Parallélépipède rectangle avec dimensions L=50, b=4, h=4 mm
- Application d'un chargement en créneau sur la face x=0
- Propagation d'onde dans le domaine 3D

Ce test sert principalement à vérifier le comportement du code en 3D
et à évaluer les performances de calcul pour des problèmes tridimensionnels.

Auteur: bouteillerp
"""

from CharonX import Solve, MyConstant, create_box, Tridimensional, CellType
from mpi4py.MPI import COMM_WORLD
import pytest
import numpy as np
import sys
sys.path.append("../../")
from Generic_isotropic_material import Acier, lmbda, mu, rho
rigi = lmbda + 2 * mu
wave_speed = (rigi/rho)**(1./2)


######## Paramètres géométriques et de maillage ########
L, b, h = 50, 4, 4
Nx, Ny, Nz = 100, 8, 8

###### Temps simulation ######
Tfin = 2e-2
print("le temps de fin de simulation est", Tfin )
pas_de_temps = Tfin/8000
largeur_creneau = L/4
magnitude = 1e2

sortie = 200
pas_de_temps_sortie = sortie * pas_de_temps
n_sortie = int(Tfin/pas_de_temps_sortie)

Nx = 200
mesh = create_box(COMM_WORLD, [np.array([0,0,0]), np.array([L, b, h])],
          [Nx, Ny, Nz], cell_type = CellType.hexahedron)
T_unload = largeur_creneau/wave_speed
chargement = MyConstant(mesh, T_unload, magnitude, Type = "Creneau")
dictionnaire = {"mesh" : mesh,
                "boundary_setup": 
                    {"tags": [1, 2, 3],
                     "coordinate": ["x", "y", "z"], 
                     "positions": [0, 0, 0]
                     },
                "boundary_conditions": 
                    [{"component": "Uy", "tag": 2}, {"component": "Uz", "tag": 3}],
                "loading_conditions": 
                    [{"type": "surfacique", "component" : "Fx", "tag": 1, "value" : chargement}
                    ],
                "isotherm" : True,
                }
   
pb = Tridimensional(Acier, dictionnaire)

dictionnaire_solve = {
    "Prefix" : "Test_elasticite",
    "csv_output" : {"Sig" : True},
    "output" : {"Sig" : True}
    }

solve_instance = Solve(pb, dictionnaire_solve, compteur=sortie, TFin=Tfin, scheme = "fixed", dt = pas_de_temps)
solve_instance.solve()