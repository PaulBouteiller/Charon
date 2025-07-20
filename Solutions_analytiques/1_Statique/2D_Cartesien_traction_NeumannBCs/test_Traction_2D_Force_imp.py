"""
Test de traction 2D avec conditions aux limites de Neumann.

Ce script simule un essai de traction uniaxiale sur une plaque rectangulaire
en conditions de déformation plane avec des forces imposées sur les bords.

Paramètres géométriques:
    - Longueur: 1
    - Largeur: 0.5
    - Discrétisation: maillage 20×20 (quadrilatères)

Chargement:
    - Force surfacique imposée (f_surf): 1e3
    - Nombre de pas: 20

Conditions aux limites:
    - Déplacement horizontal bloqué sur le bord gauche
    - Déplacement vertical bloqué sur le bord inférieur
    - Force horizontale imposée sur le bord droit

Le script calcule la contrainte résultante et la compare avec la valeur
imposée (f_surf) pour vérifier la cohérence de l'implémentation des
conditions aux limites de Neumann.

Auteur: bouteillerp
Date de création: 11 Mars 2022
"""
from CharonX import create_rectangle, Plane_strain, CellType, Solve
from mpi4py import MPI
import matplotlib.pyplot as plt
import pytest
import numpy as np
import sys
sys.path.append("../")
from Generic_isotropic_material import Acier

###### Paramètre géométrique ######
Largeur = 0.5
Longueur = 1

###### Chargement ######
f_surf = 1e3
Npas = 20

mesh = create_rectangle(MPI.COMM_WORLD, [(0, 0), (Longueur, Largeur)], [20, 20], CellType.quadrilateral)
dictionnaire = {"mesh" : mesh,
                "boundary_setup": 
                    {"tags": [1, 2, 3],
                     "coordinate": ["x", "y", "x"], 
                     "positions": [0, 0, Longueur]
                     },
                "boundary_conditions": 
                    [{"component": "Ux", "tag": 1},
                     {"component": "Uy", "tag": 2}
                    ],
                "loading_conditions": 
                    [{"type": "surfacique", "component" : "Fx", "tag": 3, "value" : f_surf}
                    ],
                "isotherm" : True,
                "analysis" : "static"
                }
    
pb = Plane_strain(Acier, dictionnaire)
pb.sig_list = []
pb.Force = pb.set_F(1, "x")

def query_output(problem, t):
    problem.sig_list.append(-problem.get_F(problem.Force)/Largeur)

dictionnaire_solve = {}
solve_instance = Solve(pb, dictionnaire_solve, compteur=1, npas=20)
solve_instance.query_output = query_output #Attache une fonction d'export appelée à chaque pas de temps
solve_instance.solve()

virtual_t = np.linspace(0, 1, Npas)
force_elast_list = np.linspace(0, f_surf, len(virtual_t))

if __name__ == "__main__": 
    plt.scatter(virtual_t, pb.sig_list, marker = "x", color = "blue", label="CHARON")
    plt.plot(virtual_t, force_elast_list, linestyle = "--", color = "red", label = "Analytical")
    plt.xlim(0, 1.1 * virtual_t[-1])
    plt.ylim(0, 1.1 * pb.sig_list[-1])
    plt.xlabel(r"Temps virtuel", size = 18)
    plt.ylabel(r"Contrainte (MPa)", size = 18)
    plt.legend()
    plt.show()
            