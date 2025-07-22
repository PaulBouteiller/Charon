"""
Test de traction uniaxiale sur un cube 3D (pseudo-1D).

Ce script simule un essai de traction uniaxiale sur un cube 3D avec des conditions
aux limites imposant un état de déformation homogène équivalent à un problème 1D.

Paramètres géométriques:
    - Dimensions du cube: 0.5 × 2.0 × 2.0
    - Discrétisation: maillage 10×10×10

Chargement:
    - Déformation imposée (eps): 0.01 (1% de déformation)

Conditions aux limites:
    - Blocage des déplacements normaux sur toutes les faces, sauf:
    - Déplacement imposé dans la direction Z sur la face supérieure

Une comparaison est effectuée entre la force calculée numériquement et la solution analytique
pour un problème de traction uniaxiale 1D, en tenant compte de l'influence des conditions
aux limites 3D.

Auteur: bouteillerp
Date de création: 11 Mars 2022
"""
from CharonX import create_box, MyConstant, Tridimensional, Solve, MeshManager
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import pytest
import sys
sys.path.append("../")
from Traction_1D_cartesien_solution_analytique import sigma_xx
from Generic_isotropic_material import Acier, kappa, mu, eos_type, devia_type

######## Paramètres géométriques et de maillage ########
Longueur, Largeur, hauteur = 0.5, 2., 2.
Nx, Ny, Nz = 10, 10, 10

eps = 0.01
Umax = eps * hauteur
mesh = create_box(MPI.COMM_WORLD, [np.array([0, 0, 0]), 
                                   np.array([Longueur, Largeur, hauteur])],
                                  [Nx, Ny, Nz])
dictionnaire_mesh = {"tags": [1, 2, 3, 4, 5, 6],
                     "coordinate": ["x", "x", "y", "y", "z", "z"], 
                     "positions": [0, Longueur, 0, Largeur, 0, hauteur]
                     }
mesh_manager = MeshManager(mesh, dictionnaire_mesh)

chargement = MyConstant(mesh, Umax, Type = "Rampe")

###### Paramètre du problème ######
dictionnaire = {"mesh_manager" : mesh_manager,
                "boundary_conditions": 
                    [{"component": "Ux", "tag": 1},
                     {"component": "Ux", "tag": 2},
                     {"component": "Uy", "tag": 3},
                     {"component": "Uy", "tag": 4},
                     {"component": "Uz", "tag": 5},
                     {"component": "Uz", "tag": 6, "value": chargement},
                    ],
                "analysis" : "static",
                "isotherm" : True
                }
    
pb = Tridimensional(Acier, dictionnaire)
pb.eps_list = [0]
pb.F_list = [0]
pb.Force = pb.set_F(6, "z")

def query_output(problem, t):
    problem.eps_list.append(eps * t)
    problem.F_list.append(2 * problem.get_F(problem.Force))
    
dictionnaire_solve = {}

solve_instance = Solve(pb, dictionnaire_solve, compteur=1, npas=10)
solve_instance.query_output = query_output #Attache une fonction d'export appelée à chaque pas de temps
solve_instance.solve()

solution_analytique = np.array([2 * sigma_xx(epsilon, kappa, mu, eos_type, devia_type) for epsilon in pb.eps_list])
eps_list_percent = [100 * eps for eps in pb.eps_list]
numerical_results = np.array(pb.F_list)
# On calcule la différence entre les deux courbes
len_vec = len(solution_analytique)
diff_tot = solution_analytique - numerical_results
# Puis on réalise une sorte d'intégration discrète
integrale_discrete = sum(abs(diff_tot[j]) for j in range(len_vec))/sum(abs(solution_analytique[j]) for j in range(len_vec))
print("La difference est de", integrale_discrete)
assert integrale_discrete < 0.001, "Static 1D traction fail"
if __name__ == "__main__": 
    plt.scatter(eps_list_percent, pb.F_list, marker = "x", color = "blue", label="CHARON")
    plt.plot(eps_list_percent, solution_analytique, linestyle = "--", color = "red", label = "Analytique")
    plt.xlim(0, 1.1 * eps_list_percent[-1])
    plt.ylim(0, 1.1 * pb.F_list[-1])
    plt.xlabel(r"Déformation(%)", size = 18)
    plt.ylabel(r"Force (N)", size = 18)
    plt.legend()
    plt.show()
