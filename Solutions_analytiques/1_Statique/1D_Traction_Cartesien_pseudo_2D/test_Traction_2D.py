"""
Test de traction uniaxiale sur un rectangle en déformation plane (pseudo-1D).

Ce script simule un essai de traction uniaxiale sur une plaque rectangulaire
en conditions de déformation plane, avec des conditions aux limites imposant
un état de déformation homogène équivalent ˆ un problème 1D.

Paramètres géométriques:
    - Longueur: 1
    - Largeur: 0.5
    - Discrétisation: maillage 20x20 (quadrilatères)

Chargement:
    - Déplacement imposé (Umax): 0.002 (0.2% de déformation)

Conditions aux limites:
    - Blocage latéral sur les côtés gauche et droite
    - Blocage vertical en bas et en haut
    - Déplacement horizontal imposé sur le côté droit

Une comparaison est effectuée entre la force calculée numériquement et la solution analytique
dérivée de la théorie 1D, corrigée pour tenir compte de l'état de déformation plane.

Auteur: bouteillerp
Date de création: 11 Mars 2022
"""
from numpy import array
import matplotlib.pyplot as plt
import pytest
import sys

from CharonX import create_rectangle, MPI, MyConstant, Plane_strain, Solve, CellType
sys.path.append("../")
from Traction_1D_cartesien_solution_analytique import sigma_xx
sys.path.append("../../")
from Generic_isotropic_material import Acier, kappa, mu, eos_type, devia_type

###### Paramètre géométrique ######
Largeur = 0.5
Longueur = 1

###### Chargement ######
Umax = 0.002

mesh = create_rectangle(MPI.COMM_WORLD, [(0, 0), (Longueur, Largeur)], [20, 20], CellType.quadrilateral)
chargement = MyConstant(mesh, Umax, Type = "Rampe")

###### Paramètre du problème ######
dictionnaire = {"mesh" : mesh,
                "boundary_setup": 
                    {"tags": [1, 2, 3, 4],
                     "coordinate": ["x", "y", "x", "y"], 
                     "positions": [0, 0, Longueur, Largeur]
                     },
                "boundary_conditions": 
                    [{"component": "Ux", "tag": 1},
                     {"component": "Uy", "tag": 2},
                     {"component": "Ux", "tag": 3, "value": chargement},
                     {"component": "Uy", "tag": 4}
                    ],
                "analysis" : "static",
                "isotherm" : True
                }

pb = Plane_strain(Acier, dictionnaire)
pb.eps_list = [0]
pb.F_list = [0]
pb.Force = pb.set_F(3, "x")

def query_output(problem, t):
    problem.eps_list.append(Umax / Longueur * t)
    problem.F_list.append(2 * problem.get_F(problem.Force))
    
dictionnaire_solve = {}

solve_instance = Solve(pb, dictionnaire_solve, compteur=1, npas=10)
solve_instance.query_output = query_output #Attache une fonction d'export appelée à chaque pas de temps
solve_instance.solve()

solution_analytique = array([sigma_xx(epsilon, kappa, mu, eos_type, devia_type) for epsilon in pb.eps_list])
eps_list_percent = [100 * eps for eps in pb.eps_list]
numerical_results = array(pb.F_list)
# On calcule la différence entre les deux courbes
len_vec = len(solution_analytique)
diff_tot = solution_analytique - numerical_results
# Puis on réalise une sorte d'intégration discrète
integrale_discrete = sum(abs(diff_tot[j]) for j in range(len_vec))/sum(abs(solution_analytique[j]) for j in range(len_vec))
print("La difference est de", integrale_discrete)
assert integrale_discrete < 0.001, "Static 1D traction fail"
if __name__ == "__main__": 
    plt.scatter(eps_list_percent, numerical_results, marker = "x", color = "blue", label="CHARON")
    plt.plot(eps_list_percent, solution_analytique, linestyle = "--", color = "red", label = "Analytique")
    plt.xlim(0, 1.05 * eps_list_percent[-1])
    plt.ylim(0, 1.05 * numerical_results[-1])
    plt.xlabel(r"Déformation(%)", size = 18)
    plt.ylabel(r"Force (N)", size = 18)
    plt.legend()
    plt.show()