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

Solution analytique bas�e sur la continuit� des contraintes �  l'interface
et la r�partition des d�formations proportionnellement �  l'inverse du module d'Young.

Auteur: bouteillerp
Date de cr�ation: 24 Juillet 2023
"""
from CharonX import *
import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy import linspace

###### Mod�le m�canique ######
E = 210e3
nu = 0.3
mu = E / 2. / (1 + nu)
dico_eos = {"E" : E, "nu" : nu, "alpha" : 1}
dico_devia = {"E":E, "nu" : nu}
eos_type = "IsotropicHPP"
devia_type = "IsotropicHPP"
Acier = Material(1, 1, eos_type, devia_type, dico_eos, dico_devia)

###### Mod�le m�canique ######
ratio = 3
E_alu = E / ratio
nu_alu = nu
mu_alu = E_alu / 2. / (1 + nu_alu)
dico_eos_alu = {"E" : E_alu, "nu" : nu_alu, "alpha" : 1}
dico_devia_alu = {"E" : E_alu, "nu" : nu_alu}
eos_type_alu = "IsotropicHPP"
devia_type_alu = "IsotropicHPP"
Alu = Material(1, 1, eos_type_alu, devia_type_alu, dico_eos_alu, dico_devia_alu)

Mat = [Acier, Alu]

###### Param�tre g�om�trique ######
Longueur = 1

###### Chargement ######
Umax=1e-2

mesh = create_interval(MPI.COMM_WORLD, 20, [np.array(0), np.array(Longueur)])

chargement = MyConstant(mesh, Umax, Type = "Rampe")

dictionnaire = {"mesh" : mesh,
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

pb = CartesianUD(Mat, dictionnaire)
x = SpatialCoordinate(pb.mesh)
mult = pb.multiphase
interp = mult.V_c.element.interpolation_points()
demi_long = Longueur / 2
ufl_condition_1 = conditional(x[0]<demi_long, 1, 0)
c1_expr = Expression(ufl_condition_1, interp)
ufl_condition_2 = conditional(x[0]>=demi_long, 1, 0)
c2_expr = Expression(ufl_condition_2, interp)
mult.set_multiphase([c1_expr, c2_expr])
        

        
dictionnaire_solve = {
    "Prefix" : "Traction_1D",
    "csv_output" : {"U" : True}
    }

solve_instance = Solve(pb, dictionnaire_solve, compteur=1, npas=10)
solve_instance.solve()


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
    plt.xlabel(r"Position (mm)", size = 18)
    plt.ylabel(r"D�placement (mm)", size = 18)