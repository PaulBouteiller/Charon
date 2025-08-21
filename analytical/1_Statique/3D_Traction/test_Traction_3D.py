"""
Test de traction uniaxiale sur un cube 3D avec mat�riau isotrope.

Ce script simule un essai de traction uniaxiale sur un cube 3D et compare
les r�sultats num�riques avec la solution analytique.

Param�tres g�om�triques:
    - Dimensions du cube: 0.5 x 2.0 x 2.0
    - Discr�tisation: maillage 10x10x10

Chargement:
    - D�formation impos�e (eps): 0.005 (0.5% de d�formation)

Conditions aux limites:
    - D�placement horizontal bloqu� sur la face gauche
    - D�placement vertical bloqu� sur la face inf�rieure
    - D�placement selon Z bloqu� sur la face arri�re
    - D�placement selon Z impos� sur la face avant

Le script calcule la force r�sultante et la compare avec la solution analytique
pour un probl�me de traction uniaxiale 3D. Une assertion v�rifie que l'erreur
relative est inf�rieure � 1%.

Auteur: bouteillerp
Date de cr�ation: 11 Mars 2022
"""
from Charon import create_box, Tridimensional, Solve, MeshManager
from mpi4py.MPI import COMM_WORLD
import numpy as np
import matplotlib.pyplot as plt
import pytest
import sys
sys.path.append("../../")
from Generic_isotropic_material import Acier, E

######## Param�tres g�om�triques et de maillage ########
Longueur, Largeur, hauteur = 0.5, 2., 2.
Nx, Ny, Nz = 10, 10, 10

eps = 0.005
Umax = eps * hauteur
mesh = create_box(COMM_WORLD, [np.array([0, 0, 0]), 
                                   np.array([Longueur, Largeur, hauteur])],
                                  [Nx, Ny, Nz])

dictionnaire_mesh = {"tags": [1, 2, 3, 4],
                     "coordinate": ["x", "y", "z", "z"], 
                     "positions": [0, 0, 0, hauteur]
                     }
mesh_manager = MeshManager(mesh, dictionnaire_mesh)

###### Param�tre du probl�me ######
dictionnaire = {"mesh_manager" : mesh_manager,
                "boundary_conditions": 
                    [{"component": "Ux", "tag": 1},
                     {"component": "Uy", "tag": 2},
                     {"component": "Uz", "tag": 3},
                     {"component": "Uz", "tag": 4, "value": {"type" : "rampe", "amplitude" : Umax}},
                    ],
                "analysis" : "static",
                "isotherm" : True
                }
    
pb = Tridimensional(Acier, dictionnaire)
pb.eps_list = [0]
pb.F_list = [0]
pb.Force = pb.set_F(4, "z")

def query_output(problem, t):
    problem.eps_list.append(eps * t)
    problem.F_list.append(problem.get_F(problem.Force))
    
dictionnaire_solve = {}

solve_instance = Solve(pb, dictionnaire_solve, compteur=1, npas=10)
solve_instance.query_output = query_output #Attache une fonction d'export appel�e � chaque pas de temps
solve_instance.solve()

def force_elast(eps):
    return E * eps * Largeur * Longueur

solution_analytique = np.array([force_elast(eps) for eps in pb.eps_list])
eps_list_percent = [100 * eps for eps in pb.eps_list]
numerical_results = np.array(pb.F_list)
# On calcule la diff�rence entre les deux courbes
len_vec = len(solution_analytique)
diff_tot = solution_analytique - numerical_results
# Puis on r�alise une sorte d'int�gration discr�te
integrale_discrete = sum(abs(diff_tot[j]) for j in range(len_vec))/sum(abs(solution_analytique[j]) for j in range(len_vec))
print("La difference est de", integrale_discrete)
assert integrale_discrete < 0.01, "Static 1D traction fail"
if __name__ == "__main__": 
    plt.scatter(eps_list_percent, pb.F_list, marker = "x", color = "blue", label="CHARON")
    plt.plot(eps_list_percent, solution_analytique, linestyle = "--", color = "red", label = "Analytique")
    plt.xlim(0, 1.1 * eps_list_percent[-1])
    plt.ylim(0, 1.1 * pb.F_list[-1])
    plt.xlabel(r"D�formation(%)", size = 18)
    plt.ylabel(r"Force (N)", size = 18)
    plt.legend()
    plt.show()