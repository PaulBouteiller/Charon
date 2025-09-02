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
from numpy import array, loadtxt
import matplotlib.pyplot as plt
import pytest
import sys
from mpi4py import MPI

from Charon import create_rectangle, PlaneStrain, Solve, CellType, MeshManager
sys.path.append("../")
from Traction_1D_cartesien_solution_analytique import sigma_xx
sys.path.append("../../")
from Generic_isotropic_material import Acier, kappa, mu, eos_type, devia_type

###### Param�tre g�om�trique ######
Largeur = 0.5
Longueur = 1

###### Maillage ######
mesh = create_rectangle(MPI.COMM_WORLD, [(0, 0), (Longueur, Largeur)], [20, 20], CellType.quadrilateral)
dictionnaire_mesh = {"tags": [1, 2, 3, 4],
                     "coordinate": ["x", "y", "x", "y"], 
                     "positions": [0, 0, Longueur, Largeur]
                     }
mesh_manager = MeshManager(mesh, dictionnaire_mesh)

###### Param�tre du probl�me ######
Umax = 0.002
dictionnaire = {"material" : Acier,
                "mesh_manager" : mesh_manager,
                "boundary_conditions": 
                    [{"component": "Ux", "tag": 1},
                     {"component": "Uy", "tag": 2},
                     {"component": "Ux", "tag": 3, "value": {"type" : "rampe", "pente" : Umax}},
                     {"component": "Uy", "tag": 4}
                    ],
                "analysis" : "static",
                "isotherm" : True
                }

pb = PlaneStrain(dictionnaire)
dico_solve = {"Prefix" : "Traction_2D", "csv_output" : {"reaction_force" : {"flag" : 3, "component" : "x"}}}

solve_instance = Solve(pb, dico_solve, compteur=1, npas=10)
solve_instance.solve()

#%%Validation et trac� du r�sultat
temps = loadtxt("Traction_2D-results/export_times.csv",  delimiter=',', skiprows=1)
half_reaction = loadtxt("Traction_2D-results/reaction_force.csv",  delimiter=',', skiprows=1)
numerical_results = array(2*half_reaction)
eps_list = [Umax / Longueur * t for t in temps]    

solution_analytique = array([sigma_xx(epsilon, kappa, mu, eos_type, devia_type) for epsilon in eps_list])
eps_list_percent = [100 * eps for eps in eps_list]

# On calcule la diff�rence entre les deux courbes
len_vec = len(solution_analytique)
diff_tot = solution_analytique - numerical_results
# Puis on r�alise une sorte d'int�gration discr�te
integrale_discrete = sum(abs(diff_tot[j]) for j in range(len_vec))/sum(abs(solution_analytique[j]) for j in range(len_vec))
print("La difference est de", integrale_discrete)
assert integrale_discrete < 0.001, "Static 1D traction fail"
if __name__ == "__main__": 
    plt.scatter(eps_list_percent, numerical_results, marker = "x", color = "blue", label="Charon")
    plt.plot(eps_list_percent, solution_analytique, linestyle = "--", color = "red", label = "Analytique")
    plt.xlim(0, 1.05 * eps_list_percent[-1])
    plt.ylim(0, 1.05 * numerical_results[-1])
    plt.xlabel(r"D�formation(%)", size = 18)
    plt.ylabel(r"Force (N)", size = 18)
    plt.legend()
    plt.show()