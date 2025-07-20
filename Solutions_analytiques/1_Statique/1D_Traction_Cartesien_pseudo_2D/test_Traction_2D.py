"""
Test de traction uniaxiale sur un rectangle en d�formation plane (pseudo-1D).

Ce script simule un essai de traction uniaxiale sur une plaque rectangulaire
en conditions de d�formation plane, avec des conditions aux limites imposant
un �tat de d�formation homog�ne �quivalent � un probl�me 1D.

Param�tres g�om�triques:
    - Longueur: 1
    - Largeur: 0.5
    - Discr�tisation: maillage 20x20 (quadrilat�res)

Chargement:
    - D�placement impos� (Umax): 0.002 (0.2% de d�formation)

Conditions aux limites:
    - Blocage lat�ral sur les c�t�s gauche et droite
    - Blocage vertical en bas et en haut
    - D�placement horizontal impos� sur le c�t� droit

Une comparaison est effectu�e entre la force calcul�e num�riquement et la solution analytique
d�riv�e de la th�orie 1D, corrig�e pour tenir compte de l'�tat de d�formation plane.

Auteur: bouteillerp
Date de cr�ation: 11 Mars 2022
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

###### Param�tre g�om�trique ######
Largeur = 0.5
Longueur = 1

###### Chargement ######
Umax = 0.002

mesh = create_rectangle(MPI.COMM_WORLD, [(0, 0), (Longueur, Largeur)], [20, 20], CellType.quadrilateral)
chargement = MyConstant(mesh, Umax, Type = "Rampe")

###### Param�tre du probl�me ######
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
solve_instance.query_output = query_output #Attache une fonction d'export appel�e � chaque pas de temps
solve_instance.solve()

solution_analytique = array([sigma_xx(epsilon, kappa, mu, eos_type, devia_type) for epsilon in pb.eps_list])
eps_list_percent = [100 * eps for eps in pb.eps_list]
numerical_results = array(pb.F_list)
# On calcule la diff�rence entre les deux courbes
len_vec = len(solution_analytique)
diff_tot = solution_analytique - numerical_results
# Puis on r�alise une sorte d'int�gration discr�te
integrale_discrete = sum(abs(diff_tot[j]) for j in range(len_vec))/sum(abs(solution_analytique[j]) for j in range(len_vec))
print("La difference est de", integrale_discrete)
assert integrale_discrete < 0.001, "Static 1D traction fail"
if __name__ == "__main__": 
    plt.scatter(eps_list_percent, numerical_results, marker = "x", color = "blue", label="CHARON")
    plt.plot(eps_list_percent, solution_analytique, linestyle = "--", color = "red", label = "Analytique")
    plt.xlim(0, 1.05 * eps_list_percent[-1])
    plt.ylim(0, 1.05 * numerical_results[-1])
    plt.xlabel(r"D�formation(%)", size = 18)
    plt.ylabel(r"Force (N)", size = 18)
    plt.legend()
    plt.show()