"""
Test de traction simple en coordonnées cartésiennes 1D.

Ce script simule un essai de traction uniaxiale sur une barre 1D et compare
les résultats numériques avec la solution analytique.

Paramètres géométriques:
    - Longueur de la barre: 1
    - Discrétisation: 2 éléments

Chargement:
    - Déplacement imposé (Umax): 1e-3 (0.1% de déformation)

La solution analytique utilise les expressions développées dans le module
Traction_1D_cartesien_solution_analytique pour différents modèles constitutifs.
Une comparaison est effectuée entre la force calculée numériquement et analytiquement.

Auteur: bouteillerp
Date de création: 24 Juillet 2023
"""
from CharonX import create_1D_mesh, CartesianUD, Solve, MyConstant, MeshManager
import matplotlib.pyplot as plt
import numpy as np
import pytest
import sys
sys.path.append("../")
from Traction_1D_cartesien_solution_analytique import sigma_xx
sys.path.append("../../")
from Generic_isotropic_material import Acier, eos_type, devia_type, mu, kappa

###### Paramètre géométrique ######
Longueur = 1
Nx = 2
###### Chargement ######
Umax=1e-3   
mesh = create_1D_mesh(0, Longueur, Nx)


dictionnaire_mesh = {"tags": [1, 2], "coordinate": ["x", "x"], "positions": [0, Longueur]}
mesh_manager = MeshManager(mesh, dictionnaire_mesh)

chargement = MyConstant(mesh, Umax, Type = "Rampe")
dictionnaire = {"mesh_manager" : mesh_manager,
                "boundary_setup": 
                    {"tags": [1, 2],
                     "coordinate": ["x", "x"], 
                     "positions": [0, Longueur]
                     },
                "boundary_conditions": 
                    [{"component": "U", "tag": 1},
                     {"component": "U", "tag": 2, "value": chargement}
                    ],
                "analysis" : "static",
                "isotherm" : True
                }

pb = CartesianUD(Acier, dictionnaire)
pb.eps_list = [0]
pb.F_list = [0]
pb.Force = pb.set_F(2, "x")

def query_output(problem, t):
    problem.eps_list.append(Umax / Longueur * t)
    problem.F_list.append(problem.get_F(problem.Force))
    
dictionnaire_solve = {
    "Prefix" : "Traction_1D",
    "output" : {"U" : True}
    }

solve_instance = Solve(pb, dictionnaire_solve, compteur=1, npas=10)
solve_instance.query_output = query_output #Attache une fonction d'export appelée à chaque pas de temps
solve_instance.solve()

#%%Validation et tracé du résultat      
solution_analytique = np.array([sigma_xx(epsilon, kappa, mu, eos_type, devia_type) for epsilon in pb.eps_list])
eps_list_percent = [100 * eps for eps in pb.eps_list]
numerical_results = np.array(pb.F_list)

# On calcule la différence entre les deux courbes
len_vec = len(solution_analytique)
diff_tot = solution_analytique - numerical_results
# Puis on réalise une sorte d'intégration discrète
integrale_discrete = sum(abs(diff_tot[j]) for j in range(len_vec))/sum(abs(solution_analytique[j]) for j in range(len_vec))
print("La difference est de", integrale_discrete)
# assert integrale_discrete < 0.001, "Static 1D traction fail"
if __name__ == "__main__": 
    plt.plot(eps_list_percent, solution_analytique, linestyle = "--", color = "red", label = "Analytique")
    plt.scatter(eps_list_percent, numerical_results, marker = "x", color = "blue", label="CHARON")
    plt.xlim(0, 1.05 * eps_list_percent[-1])
    plt.ylim(0, 1.05 * numerical_results[-1])
    plt.xlabel(r"Déformation(%)", size = 18)
    plt.ylabel(r"Force (N)", size = 18)
    plt.legend()
    plt.show()