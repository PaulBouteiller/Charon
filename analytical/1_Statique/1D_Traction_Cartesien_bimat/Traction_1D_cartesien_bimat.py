"""
Test de traction sur une barre composite 1D (deux mat�riaux).

Ce script simule un essai de traction uniaxiale sur une barre 1D compos�e de
deux mat�riaux diff�rents (acier et aluminium) et compare la solution num�rique
avec la solution analytique.

Param�tres g�om�triques:
    - Longueur totale de la barre: 1
    - Discr�tisation: 20 �l�ments
    - Deux moiti�s �gales de mat�riaux diff�rents

Chargement:
    - D�placement impos� (Umax): 1e-2 (1% de d�formation)

Mat�riaux:
    - Acier: Module d'Young E
    - Aluminium: Module d'Young E/ratio (ratio = 3)
    - Même coefficient de Poisson pour les deux mat�riaux

Solution analytique bas�e sur la continuit� des contraintes � l'interface
et la r�partition des d�formations proportionnellement � l'inverse du module d'Young.

Auteur: bouteillerp
Date de cr�ation: 24 Juillet 2023
"""
from Charon import create_1D_mesh, CartesianUD, Solve, MeshManager
from pandas import read_csv
from ufl import SpatialCoordinate
import matplotlib.pyplot as plt
import pytest
from numpy import linspace

import sys
sys.path.append("../../")
from Generic_isotropic_material import Acier, Alu, ratio

Mat = [Acier, Alu]

###### Param�tre g�om�trique ######
Longueur = 1

###### Chargement ######
Umax=1e-2

Nx = 20
mesh = create_1D_mesh(0, Longueur, Nx)
dictionnaire_mesh = {"tags": [1, 2], "coordinate": ["x", "x"], "positions": [0, Longueur]}
mesh_manager = MeshManager(mesh, dictionnaire_mesh)

x = SpatialCoordinate(mesh)
demi_long = Longueur / 2      

dictionnaire = {"mesh_manager" : mesh_manager,
                "boundary_conditions": 
                    [{"component": "U", "tag": 1},
                     {"component": "U", "tag": 2, "value": {"type" : "rampe", "amplitude" : Umax}}
                    ],
                "multiphase" : {"conditions" : [x[0]<demi_long, x[0]>=demi_long]},
                "analysis" : "static",
                "isotherm" : True
                }

pb = CartesianUD(Mat, dictionnaire)


dictionnaire_solve = {
    "Prefix" : "Traction_1D",
    "csv_output" : {"U" : True}
    }

solve_instance = Solve(pb, dictionnaire_solve, compteur=1, npas=100)
solve_instance.solve()

#%%Validation et trac� du r�sultat
u_csv = read_csv("Traction_1D-results/U.csv")
resultat = [u_csv[colonne].to_numpy() for colonne in u_csv.columns]
x_result = resultat[0]
half_n_node = len(x_result)//2
eps_tot = Umax
eps_acier = 2 * eps_tot / (1 + ratio)
eps_alu = ratio * eps_acier 
dep_acier = [eps_acier * x for x in linspace(0, 0.5, half_n_node+1)]
dep_alu = [eps_acier * 0.5 + eps_alu * x for x in linspace(0, 0.5, half_n_node+1)]
dep_acier.pop()
dep_tot = dep_acier + dep_alu
solution_numerique = resultat[-1]
if __name__ == "__main__": 
    plt.scatter(x_result, solution_numerique, marker = "x", color = "red")
    plt.plot(x_result, dep_tot, linestyle = "--", color = "blue")            
    plt.xlim(0, 1)
    plt.ylim(0, 1.1 * max(dep_tot))
    plt.xlabel(r"Position (mm)", size = 18)
    plt.ylabel(r"D�placement (mm)", size = 18)